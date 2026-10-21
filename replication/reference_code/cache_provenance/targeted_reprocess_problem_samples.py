#!/usr/bin/env python3
"""
Enhanced Data Quality Investigation - Phase 3: Targeted Reprocessing

Reprocess only the problematic outlet-month combinations using enhanced selectors
to validate and apply the critical fixes identified in Phase 2.

Input: docs/debugging/analysis/problem_samples/*.parquet
Output: Updated database with improved extractions for problematic samples
"""

import polars as pl
from pathlib import Path
import duckdb
import time
from datetime import datetime
from selectolax.lexbor import LexborHTMLParser
from tqdm import tqdm
import re
import logging
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Setup CJK font support
ensure_cjk_font_setup()

# Configuration
PROBLEM_SAMPLES_DIR = Path("docs/debugging/analysis/problem_samples")
DB_PATH = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
LOG_DIR = Path("logs/warc_extract/stage2_targeted")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Enhanced selectors from Phase 2 analysis
ENHANCED_SELECTORS = {
    "peoplenews.tw": { # People News (民報) - CRITICAL FIX for 0.01% extraction rate
        "title": ['meta[property="og:title"]', "h1", "title"],
        "content": ["#main"],  # Phase 2 analysis: 100% coverage, avg 1784 chars
        "date": ['meta[property="article:published_time"]', 'meta[name="pubdate"]', 'time[datetime]']
    },
    "cna.com.tw": { # Central News Agency (中央社) - MAJOR FIX for 93.14% extraction rate  
        "title": ['meta[property="og:title"]', "h1", "title"],
        "content": ['[class*="content"]', '[class*="news"]', 'div[class*="text"]'],  # Phase 2: top 3 selectors
        "date": ['meta[property="article:published_time"]', 'meta[name="pubdate"]', 'time[datetime]']
    },
    "news.ltn.com.tw": { # Liberty Times (news subdomain) - ENHANCED based on Phase 2 analysis
        "title": ['meta[property="og:title"]', "h1.article-title", "h1"],
        "content": [".content", '[class*="content"]', "div.text", "div.articlebody", "div.news_content"],
        "date": ['meta[property="article:published_time"]', 'span.time', 'meta[name="pubdate"]']
    },
    "default": { # Fallback selectors
        "title": ["title", "h1", 'meta[property="og:title"]', 'meta[name="twitter:title"]'],
        "content": ["article", "main", ".article-content", ".story-content", ".post-content", ".entry-content"],
        "date": [
            'meta[property="article:published_time"]', 'meta[name="pubdate"]', 'meta[name="Date"]',
            'meta[name="publishdate"]', 'time[datetime]'
        ]
    }
}

def setup_logging():
    """Setup logging for targeted reprocessing."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"targeted_reprocess_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Targeted reprocessing started: {timestamp}")
    logging.info(f"Log file: {log_file}")
    return log_file

def clean_text(text: str | None) -> str | None:
    """Clean whitespace from text."""
    if text is None:
        return None
    try:
        cleaned = re.sub(r'\s+', ' ', str(text))
        return cleaned.strip()
    except Exception:
        return None

def extract_content_with_enhanced_selectors(html_bytes: bytes, url: str, domain: str) -> dict[str, str | None]:
    """Extract content using enhanced selectors from Phase 2 analysis."""
    extracted = {
        "title": None,
        "content": None,
        "extracted_publish_date": None
    }
    
    if not html_bytes:
        return extracted
    
    # Get enhanced selectors for this domain
    selectors = ENHANCED_SELECTORS.get(domain, ENHANCED_SELECTORS["default"])
    
    try:
        tree = LexborHTMLParser(html_bytes)
        if not tree.root:
            return extracted
        
        root = tree.root
        
        # Extract Title
        for selector in selectors["title"]:
            title_node = root.css_first(selector)
            if title_node:
                raw_title = title_node.text(strip=True)
                cleaned_title = clean_text(raw_title)
                if cleaned_title:
                    extracted["title"] = cleaned_title
                    break
        
        # Extract Content
        for selector in selectors["content"]:
            content_node = root.css_first(selector)
            if content_node:
                raw_content = content_node.text(separator=' ', strip=True)
                if raw_content and len(raw_content) > 50:  # Minimum content length
                    extracted["content"] = clean_text(raw_content[:20000])  # Limit length
                    break
        
        # Extract Date
        for selector in selectors["date"]:
            date_node = root.css_first(selector)
            if date_node:
                date_str = None
                if date_node.tag == 'meta':
                    date_str = date_node.attributes.get('content')
                elif date_node.tag == 'time':
                    date_str = date_node.attributes.get('datetime')
                else:
                    date_str = date_node.text(strip=True)
                
                if date_str:
                    cleaned_date = date_str.strip()
                    if cleaned_date:
                        extracted["extracted_publish_date"] = cleaned_date
                        break
    
    except Exception as e:
        logging.warning(f"Error parsing HTML for {url}: {e}")
    
    return extracted

def get_problem_sample_files() -> list[Path]:
    """Get all problem sample parquet files."""
    sample_files = list(PROBLEM_SAMPLES_DIR.glob("*.parquet"))
    sample_files.sort()
    
    logging.info(f"Found {len(sample_files)} problem sample files to reprocess")
    for f in sample_files:
        logging.info(f"  • {f.name}")
    
    return sample_files

def connect_to_database():
    """Connect to the enhanced Stage 2 database."""
    if not Path(DB_PATH).exists():
        logging.error(f"Database not found: {DB_PATH}")
        return None
    
    logging.info(f"Connecting to database: {DB_PATH}")
    return duckdb.connect(DB_PATH)

def reprocess_problem_samples():
    """Main function to reprocess problematic samples with enhanced selectors."""
    logging.info("🚀 Starting targeted reprocessing of problematic outlet-month combinations")
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        # Get all problem sample files
        sample_files = get_problem_sample_files()
        
        if not sample_files:
            logging.warning("No problem sample files found")
            return
        
        total_processed = 0
        total_improved = 0
        total_errors = 0
        
        # Process each file
        for file_path in tqdm(sample_files, desc="Processing problem sample files"):
            logging.info(f"\n📄 Processing: {file_path.name}")
            
            try:
                # Load the problem sample data
                df = pl.read_parquet(file_path)
                
                batch_updates = []
                file_processed = 0
                file_improved = 0
                
                # Process each article in the file
                for row in tqdm(df.iter_rows(named=True), 
                              desc=f"Processing {file_path.stem}", 
                              total=len(df), 
                              leave=False):
                    
                    warc_id = row.get('warc_id')
                    html_bytes = row.get('html_content_bytes')
                    url = row.get('url', '')
                    domain = row.get('domain', '')
                    
                    if not warc_id or not html_bytes:
                        continue
                    
                    # Extract content with enhanced selectors
                    extracted = extract_content_with_enhanced_selectors(html_bytes, url, domain)
                    
                    # Check if we got meaningful content
                    if extracted["title"] or (extracted["content"] and len(extracted["content"]) > 100):
                        batch_updates.append({
                            'warc_id': warc_id,
                            'title': extracted["title"],
                            'content': extracted["content"],
                            'extracted_publish_date': extracted["extracted_publish_date"]
                        })
                        file_improved += 1
                    
                    file_processed += 1
                
                # Update database with improved extractions
                if batch_updates:
                    logging.info(f"   📝 Updating {len(batch_updates)} articles in database")
                    
                    for update in batch_updates:
                        try:
                            conn.execute("""
                                UPDATE articles 
                                SET title = ?, 
                                    content = ?, 
                                    extracted_publish_date = ?,
                                    stage2_processed_at = current_timestamp
                                WHERE warc_id = ?
                            """, [
                                update['title'],
                                update['content'], 
                                update['extracted_publish_date'],
                                update['warc_id']
                            ])
                        except Exception as e:
                            logging.warning(f"Failed to update article {update['warc_id']}: {e}")
                            total_errors += 1
                    
                    conn.commit()
                
                total_processed += file_processed
                total_improved += file_improved
                
                logging.info(f"   ✅ File results: {file_processed} processed, {file_improved} improved")
                
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}")
                total_errors += 1
                continue
        
        # Final summary
        logging.info("\n" + "="*80)
        logging.info("📊 TARGETED REPROCESSING COMPLETE")
        logging.info("="*80)
        logging.info(f"Total Articles Processed: {total_processed:,}")
        logging.info(f"Total Articles Improved: {total_improved:,}")
        logging.info(f"Improvement Rate: {(total_improved/total_processed*100):.2f}%" if total_processed > 0 else "0%")
        logging.info(f"Total Errors: {total_errors:,}")
        
        # Validate improvements by checking a few key outlets
        validate_improvements(conn)
        
    except Exception as e:
        logging.error(f"Critical error during reprocessing: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed")

def validate_improvements(conn):
    """Validate the improvements by checking extraction rates for key outlets."""
    logging.info("\n🔍 Validating improvements...")
    
    key_outlets = [
        "People News (民報)",
        "Central News Agency (中央社)", 
        "Liberty Times (自由時報)"
    ]
    
    for outlet in key_outlets:
        try:
            # Get total articles for this outlet
            total_result = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE outlet = ?", [outlet]
            ).fetchone()
            total_articles = total_result[0] if total_result else 0
            
            # Get articles with content
            content_result = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE outlet = ? AND content IS NOT NULL AND content != ''", 
                [outlet]
            ).fetchone()
            articles_with_content = content_result[0] if content_result else 0
            
            if total_articles > 0:
                extraction_rate = (articles_with_content / total_articles) * 100
                logging.info(f"   📊 {outlet}: {extraction_rate:.2f}% extraction rate ({articles_with_content:,}/{total_articles:,})")
            else:
                logging.info(f"   📊 {outlet}: No articles found")
                
        except Exception as e:
            logging.warning(f"Error validating {outlet}: {e}")

def main():
    """Main execution function."""
    print("🚨 Enhanced Data Quality Investigation - Phase 3: Targeted Reprocessing")
    print("=" * 85)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Mission: Reprocess problematic outlet-month combinations with enhanced selectors")
    print("=" * 85)
    
    # Setup logging
    log_file = setup_logging()
    
    try:
        # Start targeted reprocessing
        start_time = time.time()
        reprocess_problem_samples()
        end_time = time.time()
        
        duration = end_time - start_time
        logging.info(f"\n⏱️ Total Processing Time: {duration:.2f} seconds")
        logging.info(f"📁 Log file saved: {log_file}")
        
        print("\n🎉 Targeted reprocessing completed successfully!")
        print(f"📋 Check log file for detailed results: {log_file}")
        
    except Exception as e:
        logging.error(f"❌ Targeted reprocessing failed: {e}")
        print(f"❌ Error occurred. Check log file: {log_file}")
        raise

if __name__ == "__main__":
    main() 