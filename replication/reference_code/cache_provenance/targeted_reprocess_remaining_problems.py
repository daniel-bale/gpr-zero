#!/usr/bin/env python3
"""
Phase 5: Targeted Reprocessing for Remaining Problems

This script reprocesses the remaining 7 problematic outlet-month combinations
using the enhanced selectors and the approach from targeted_reprocess_problem_samples.py.

Uses existing problem samples from: docs/debugging/analysis/problem_samples/

Target Issues:
- Now News (今日新聞): 2020-05, 2020-06, 2020-07, 2020-08 (0% extraction)
- Taiwan News: 2021-06 (91% extraction)
- United Daily (聯合報): 2018-06, 2018-07 (6-9% extraction)
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
LOG_DIR = Path("logs/warc_extract/stage2_targeted_remaining")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remaining problematic files to process
REMAINING_PROBLEM_FILES = [
    "2020-05_Now_News_今日新聞_2003articles.parquet",
    "2020-06_Now_News_今日新聞_5702articles.parquet", 
    "2020-07_Now_News_今日新聞_6694articles.parquet",
    "2020-08_Now_News_今日新聞_6749articles.parquet",
    "2021-06_Taiwan_News_8036articles.parquet",
    "2018-06_United_Daily_聯合報_9692articles.parquet",
    "2018-07_United_Daily_聯合報_4659articles.parquet"
]

# Enhanced selectors for remaining problematic domains
ENHANCED_SELECTORS = {
    "nownews.com": { # Now News (今日新聞) - PHASE 5 FIX for 0% extraction in 2020
        "title": ['meta[property="og:title"]', "h1", "title"],
        "content": [
            ".entry-content",     # Common WordPress/CMS content selector
            ".post-content",      # Alternative post content selector
            ".article-content",   # Generic article content selector
            "#content",           # Simple content ID selector
            ".content",           # Basic content class selector
            "article",            # HTML5 article element
            "main"                # HTML5 main element
        ],
        "date": ['meta[property="article:published_time"]', 'meta[name="pubdate"]', 'time[datetime]']
    },
    "taiwannews.com.tw": { # Taiwan News - PHASE 5 FIX for 91% extraction in 2021-06
        "title": ['meta[property="og:title"]', "h1", "title"],
        "content": [
            ".post-content",      # Common post content selector
            ".entry-content",     # WordPress entry content selector
            ".article-body",      # Article body selector
            ".content-area",      # Content area selector
            ".article-content",   # Generic article content
            "article",            # HTML5 article element
            "main"                # HTML5 main element
        ],
        "date": ['meta[property="article:published_time"]', 'meta[name="pubdate"]', 'time[datetime]']
    },
    "udn.com": { # United Daily News - ENHANCED for Phase 5 (2018 extraction issues)
        "title": ["#story_art_title", 'meta[property="og:title"]', "h1"],
        "content": [
            "#story_body_content",   # Modern UDN selector
            ".story__content",       # Alternative story content
            ".article-content",      # Generic article content
            "#story_body",           # Historical story body ID
            ".story-body",           # Historical story body class
            ".content-article",      # Article content class
            "article",               # HTML5 article element
            ".post-content",         # Post content fallback
            "main"                   # HTML5 main element
        ],
        "date": ['meta[name="date"]', '.story_bady_info_author span:first-child', '.shareBar__info--author span:first-child']
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
    log_file = LOG_DIR / f"remaining_problems_reprocess_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Phase 5: Targeted reprocessing for remaining problems started: {timestamp}")
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
    """Extract content using enhanced selectors for remaining problematic domains."""
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
        
        # Extract Content with domain-specific handling
        for selector in selectors["content"]:
            content_node = root.css_first(selector)
            if content_node:
                # Domain-specific cleaning for UDN
                if domain == "udn.com":
                    # Remove unwanted elements before extracting text
                    for unwanted in content_node.css('script, style, .article-footer, .article-tags, .social-share, .related-news'):
                        unwanted.decompose()
                
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

def connect_to_database():
    """Connect to the enhanced Stage 2 database."""
    if not Path(DB_PATH).exists():
        logging.error(f"Database not found: {DB_PATH}")
        return None
    
    logging.info(f"Connecting to database: {DB_PATH}")
    return duckdb.connect(DB_PATH)

def reprocess_remaining_problems():
    """Main function to reprocess remaining problematic samples with enhanced selectors."""
    logging.info("🚀 Starting Phase 5: Targeted reprocessing for remaining problems")
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        total_processed = 0
        total_improved = 0
        total_errors = 0
        
        # Process each remaining problem file
        for filename in REMAINING_PROBLEM_FILES:
            file_path = PROBLEM_SAMPLES_DIR / filename
            
            if not file_path.exists():
                logging.warning(f"Problem sample file not found: {filename}")
                continue
                
            logging.info(f"\n📄 Processing: {filename}")
            
            try:
                # Load the problem sample data
                df = pl.read_parquet(file_path)
                
                batch_updates = []
                file_processed = 0
                file_improved = 0
                
                # Process each article in the file
                for row in tqdm(df.iter_rows(named=True), 
                              desc=f"Processing {filename}", 
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
                
                improvement_rate = (file_improved / file_processed * 100) if file_processed > 0 else 0
                logging.info(f"   ✅ File results: {file_processed} processed, {file_improved} improved ({improvement_rate:.1f}%)")
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                total_errors += 1
                continue
        
        # Final summary
        logging.info("\n" + "="*80)
        logging.info("📊 PHASE 5 TARGETED REPROCESSING COMPLETE")
        logging.info("="*80)
        logging.info(f"Total Articles Processed: {total_processed:,}")
        logging.info(f"Total Articles Improved: {total_improved:,}")
        if total_processed > 0:
            improvement_rate = (total_improved / total_processed) * 100
            logging.info(f"Improvement Rate: {improvement_rate:.2f}%")
        logging.info(f"Total Errors: {total_errors:,}")
        
        # Validate improvements by checking the specific problematic periods
        validate_improvements(conn)
        
    except Exception as e:
        logging.error(f"Critical error during reprocessing: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed")

def validate_improvements(conn):
    """Validate the improvements by checking extraction rates for the specific problematic periods."""
    logging.info("\n🔍 Validating improvements for remaining problems...")
    
    validation_periods = [
        ("Now News (今日新聞)", "2020-05"),
        ("Now News (今日新聞)", "2020-06"), 
        ("Now News (今日新聞)", "2020-07"),
        ("Now News (今日新聞)", "2020-08"),
        ("Taiwan News", "2021-06"),
        ("United Daily (聯合報)", "2018-06"),
        ("United Daily (聯合報)", "2018-07")
    ]
    
    for outlet, year_month in validation_periods:
        try:
            # Query current extraction rate for this period
            result = conn.execute("""
                SELECT 
                    COUNT(*) as total_articles,
                    COUNT(CASE WHEN content IS NOT NULL AND LENGTH(content) > 100 THEN 1 END) as content_extracted,
                    ROUND(COUNT(CASE WHEN content IS NOT NULL AND LENGTH(content) > 100 THEN 1 END) * 100.0 / COUNT(*), 2) as extraction_rate
                FROM articles 
                WHERE outlet = ? 
                AND strftime('%Y-%m', warc_date) = ?
            """, [outlet, year_month]).fetchone()
            
            if result:
                total, extracted, rate = result
                logging.info(f"   📊 {outlet} ({year_month}): {rate}% extraction rate ({extracted}/{total})")
            else:
                logging.warning(f"   ⚠️ No data found for {outlet} - {year_month}")
                
        except Exception as e:
            logging.error(f"Error validating {outlet} - {year_month}: {e}")

def main():
    """Main execution function."""
    print("🚨 Phase 5: Targeted Reprocessing for Remaining Problems")
    print("=" * 75)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Mission: Fix remaining 7 problematic outlet-month combinations")
    print(f"📁 Using problem samples from: {PROBLEM_SAMPLES_DIR}")
    print("=" * 75)
    
    # Setup logging
    log_file = setup_logging()
    
    try:
        # Start targeted reprocessing
        start_time = time.time()
        reprocess_remaining_problems()
        end_time = time.time()
        
        duration = end_time - start_time
        logging.info(f"\n⏱️ Total Processing Time: {duration:.2f} seconds")
        logging.info(f"📁 Log file saved: {log_file}")
        
        print("\n🎉 Phase 5 targeted reprocessing completed!")
        print(f"📋 Check log file for detailed results: {log_file}")
        
    except Exception as e:
        logging.error(f"❌ Phase 5 reprocessing failed: {e}")
        print(f"❌ Error occurred. Check log file: {log_file}")
        raise

if __name__ == "__main__":
    main() 