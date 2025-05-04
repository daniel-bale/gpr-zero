#!/usr/bin/env python3
import os
import re
import json
import argparse
import time
import uuid
import duckdb
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from fastwarc.warc import ArchiveIterator, WarcRecordType, WarcRecord
from fastwarc.stream_io import GZipStream, FileStream, LZ4Stream
from urllib.parse import urlparse
import logging
import multiprocessing
from tqdm import tqdm
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("taiwan_news_analysis_fastwarc.log"),
        logging.StreamHandler()
    ]
)

# Define Taiwanese news domains to search for
TAIWANESE_DOMAINS = [
    # Major News Outlets
    {"name": "Economic Daily (經濟日報)", "domains": ["money.udn.com"]},
    {"name": "PTS News (公視新聞)", "domains": ["news.pts.org.tw", "pts.org.tw"]},
    {"name": "TVBS News (TVBS新聞台)", "domains": ["news.tvbs.com.tw", "tvbs.com.tw"]},
    {"name": "ETtoday (ETtoday新聞雲)", "domains": ["www.ettoday.net", "ettoday.net", "fashion.ettoday.net"]},
    {"name": "United Daily (聯合報)", "domains": ["udn.com"]},
    {"name": "EBC News (東森新聞)", "domains": ["news.ebc.net.tw", "ebc.net.tw"]},
    {"name": "TTV News (台視新聞)", "domains": ["www.ttv.com.tw", "ttv.com.tw"]},
    {"name": "CTV News (中視新聞)", "domains": ["new.ctv.com.tw", "ctv.com.tw"]},
    {"name": "CTS News (華視新聞)", "domains": ["news.cts.com.tw", "cts.com.tw"]},
    {"name": "Liberty Times (自由時報)", "domains": ["www.ltn.com.tw", "ltn.com.tw", "news.ltn.com.tw"]},
    {"name": "Storm Media (風傳媒)", "domains": ["www.storm.mg", "storm.mg"]},
    {"name": "Formosa TV News (民視新聞)", "domains": ["www.ftvnews.com.tw", "ftvnews.com.tw"]},
    {"name": "SET News (三立新聞)", "domains": ["www.setn.com", "setn.com"]},
    # Additional Taiwan News Sources
    {"name": "China Times (中國時報)", "domains": ["www.chinatimes.com", "chinatimes.com"]},
    {"name": "Apple Daily (蘋果日報)", "domains": ["tw.appledaily.com", "appledaily.com.tw"]},
    {"name": "Central News Agency (中央社)", "domains": ["www.cna.com.tw", "cna.com.tw"]},
    {"name": "Yahoo News Taiwan", "domains": ["tw.news.yahoo.com"]},
    {"name": "Now News (今日新聞)", "domains": ["www.nownews.com", "nownews.com"]}
]

# Create lookup dictionaries for faster processing
domain_to_outlet = {}
for outlet in TAIWANESE_DOMAINS:
    for domain in outlet["domains"]:
        domain_to_outlet[domain] = outlet["name"]

all_tw_domains = set()
for outlet in TAIWANESE_DOMAINS:
    all_tw_domains.update(outlet["domains"])

def extract_domain(url: str) -> tuple[str | None, str | None]:
    """Extract domain from URL, returning both raw and www/no-www variants."""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        if not domain:
            return None, None
        if domain.startswith('www.'):
            domain_no_www = domain[4:]
            return domain, domain_no_www
        else:
            www_domain = f"www.{domain}"
            return domain, www_domain
    except Exception as e:
        logging.warning(f"Could not parse domain from URL '{url}': {e}")
        return None, None

def clean_text(text: str | None) -> str:
    """Clean extracted text by normalizing whitespace."""
    if not text:
        return ""
    # Replace excessive whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_article_content(html_content: bytes, domain: str) -> dict[str, str]:
    """Extract article content from HTML based on domain-specific rules."""
    extracted_data = {"title": "", "content": "", "publish_date": ""}
    try:
        # Use lxml for potentially faster parsing if available, fallback to html.parser
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except ImportError:
            soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'iframe', 'noscript', 'header', 'footer', 'nav', 'aside', 'form']):
            element.decompose()

        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            extracted_data["title"] = clean_text(title_tag.get_text())

        # Extract publish date from meta tags (common patterns)
        publish_date = ""
        meta_selectors = [
            'meta[property="article:published_time"]',
            'meta[property="og:published_time"]',
            'meta[name="pubdate"]',
            'meta[name="publishdate"]',
            'meta[name="date"]',
            'meta[itemprop="datePublished"]'
        ]
        for selector in meta_selectors:
            meta_tag = soup.select_one(selector)
            if meta_tag and meta_tag.get('content'):
                publish_date = meta_tag['content'].strip()
                # Basic validation/cleanup - can be improved
                if len(publish_date) > 5: # Avoid short/invalid values
                     extracted_data["publish_date"] = publish_date
                     break # Take the first one found

        # Domain-specific content extraction rules (prioritized)
        content = ""
        if "ettoday.net" in domain:
            selectors = ['div.story', 'div#news-content', 'article']
        elif "udn.com" in domain:
            selectors = ['div.article-content__paragraph', 'div#article_body', 'div.article-content', 'article']
        elif "ltn.com.tw" in domain:
            selectors = ['div.text', 'div.articlebody', 'article']
        elif "tvbs.com.tw" in domain:
            selectors = ['div.article_content', 'article']
        elif "storm.mg" in domain:
             selectors = ['div.article-content', 'div#CMS_wrapper']
        elif "chinatimes.com" in domain:
             selectors = ['div.article-body', 'article.article-box']
        else:
            # Default selectors if no specific rules match
            selectors = [
                'article', '.article', '.content', '.article-content', '#article-content',
                '.post-content', '.news-content', '[itemprop="articleBody"]',
                '.news-article', '.story-content', 'main'
            ]

        # Try selectors to find main content area
        article_div = None
        for selector in selectors:
            article_div = soup.select_one(selector)
            if article_div:
                break # Found a potential main content area

        # Extract text from the found container or fallback
        if article_div:
            # Try extracting only paragraphs first for cleaner text
            paragraphs = article_div.find_all('p', recursive=False) # Find direct paragraphs
            if paragraphs:
                content = '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            else:
                # Fallback to getting all text if no direct paragraphs found
                 content = article_div.get_text(separator='\n', strip=True)
        else:
            # Broad fallback: Take all paragraph text from the body if no container found
            paragraphs = soup.body.find_all('p') if soup.body else []
            content = '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))


        extracted_data["content"] = clean_text(content) # Clean final content

    except Exception as e:
        logging.error(f"Error extracting content for domain {domain}: {e}", exc_info=False) # exc_info=False to avoid huge tracebacks

    return extracted_data


def setup_database(db_path: str | Path, truncate: bool = False) -> duckdb.DuckDBPyConnection:
    """Setup DuckDB database for storing articles."""
    db_path = Path(db_path)
    db_dir = db_path.parent
    db_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    if truncate and db_path.exists():
        try:
            db_path.unlink()
            logging.info(f"Truncated existing database: {db_path}")
        except Exception as e:
            logging.error(f"Failed to remove database file {db_path}: {e}")

    # Connect to DuckDB
    conn = duckdb.connect(str(db_path)) # Connect using string path

    # Create articles table with ON CONFLICT clause
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id VARCHAR PRIMARY KEY,
            title VARCHAR,
            url VARCHAR UNIQUE, -- Add UNIQUE constraint to URL
            datetime TIMESTAMP,
            publish_date VARCHAR, -- Store extracted publish date string
            domain VARCHAR,
            outlet VARCHAR,
            content VARCHAR,
            source_file VARCHAR
        );
    """)

    # Create indexes for faster queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_datetime ON articles(datetime);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_outlet ON articles(outlet);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);") # Index on URL

    return conn

def process_warc_file(args_tuple: tuple[Path, int | None]) -> dict:
    """Process a single WARC file using fastwarc and return found articles."""
    warc_file_path, limit = args_tuple

    if not warc_file_path.exists():
        return {
            "warc_file": str(warc_file_path),
            "error": "File not found",
            "records_processed": 0,
            "articles_found": 0,
            "processing_time": 0,
            "articles": []
        }

    file_basename = warc_file_path.name

    results = {
        "warc_file": str(warc_file_path),
        "records_processed": 0,
        "articles_found": 0,
        "articles": [],
        "stats": {
            "by_outlet": Counter(),
            "by_date": Counter() # Based on WARC record date
        }
    }

    start_time = time.time()
    fs = None # Initialize FileStream variable

    try:
        # Use FileStream and GZipStream for potentially better performance
        fs = FileStream(str(warc_file_path), 'rb')
        # Assume GZip compression based on common pattern, could add logic for LZ4 later if needed
        stream = GZipStream(fs)

        # Use ArchiveIterator from fastwarc, filtering for response records
        # parse_http=True automatically parses HTTP headers
        archive_iterator = ArchiveIterator(stream, record_types=WarcRecordType.response, parse_http=True)

        for record_count, record in enumerate(archive_iterator):
            results["records_processed"] += 1

            if limit and record_count >= limit:
                logging.debug(f"Reached record limit ({limit}) for {file_basename}")
                break

            # fastwarc automatically filters by record_type, so no need to check record.rec_type
            # Access headers using record.headers dict-like object
            url = record.headers.get('WARC-Target-URI')
            if not url:
                continue

            domain, alt_domain = extract_domain(url)
            if not domain:
                continue

            # Check if this is a Taiwan news domain
            outlet_name = domain_to_outlet.get(domain) or domain_to_outlet.get(alt_domain)

            if outlet_name:
                # Get basic metadata
                # Use record.headers.get for raw string, or record.record_date for parsed datetime
                timestamp_str = record.headers.get('WARC-Date')
                # If parse_http=True, use record.http_content_type, otherwise check headers
                content_type = record.http_content_type if record.http_content_type else record.headers.get('Content-Type', '').lower()

                # Update statistics
                results["articles_found"] += 1
                results["stats"]["by_outlet"][outlet_name] += 1
                if timestamp_str:
                    try:
                        # Attempt to parse the date for stats, ignore if invalid
                        date_str = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                        results["stats"]["by_date"][date_str] += 1
                    except ValueError:
                        logging.debug(f"Could not parse WARC-Date '{timestamp_str}' for stats in {file_basename}")


                # Extract content if it's HTML
                if 'html' in content_type:
                    try:
                        # Read content using record.reader
                        html_content = record.reader.read()
                        if not html_content:
                             logging.debug(f"Empty HTML content for {url} in {file_basename}")
                             continue

                        content_data = extract_article_content(html_content, domain)

                        # Only add if content was likely found (avoid empty entries)
                        if content_data["title"] or content_data["content"]:
                            # Generate a UUID based on the URL for potential deduplication
                            article_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))

                            # Add article to results list for batch insertion later
                            article = {
                                "id": article_id,
                                "title": content_data["title"],
                                "url": url,
                                # Store original WARC timestamp string or use parsed record.record_date
                                "datetime": timestamp_str,
                                "publish_date": content_data["publish_date"],
                                "domain": domain,
                                "outlet": outlet_name,
                                "content": content_data["content"],
                                "source_file": file_basename
                            }
                            results["articles"].append(article)
                        else:
                             logging.debug(f"No significant title or content extracted from {url} in {file_basename}")

                    except Exception as e:
                        logging.error(f"Error processing HTML content from {url} in {file_basename}: {e}", exc_info=False)
            # else: # No need for else, already filtered by domain check
            #     # Consume the stream if not processing the record to move to the next one
            #     # record.reader.consume() # fastwarc handles skipping automatically based on filters

    except Exception as e:
        elapsed_time = time.time() - start_time
        results["error"] = str(e)
        results["processing_time"] = elapsed_time
        logging.error(f"Failed processing {warc_file_path}: {e}", exc_info=True)
        return results
    finally:
        # Ensure the FileStream is closed
        if fs:
            fs.close()


    # Calculate final stats for this file
    elapsed_time = time.time() - start_time
    processed_speed = results["records_processed"] / elapsed_time if elapsed_time > 0 else 0

    results["processing_time"] = elapsed_time
    results["processing_speed"] = processed_speed

    # Log summary for this file upon completion
    logging.debug(f"Finished {file_basename}: {results['records_processed']} records, "
                 f"{len(results['articles'])} articles extracted in {elapsed_time:.2f}s.")

    return results


# Global args for access in report generation (consider passing explicitly if preferred)
args = None

def analyze_warc_files_parallel(warc_files: list[Path], db_path: str | Path, limit_per_file: int | None = None, truncate_db: bool = False, num_processes: int | None = None):
    """Analyze a list of WARC files in parallel using fastwarc and store results in DuckDB."""
    global args # Use global args for reporting details

    if not warc_files:
        logging.warning("No WARC files found or provided to analyze.")
        return

    logging.info(f"Starting analysis of {len(warc_files)} WARC files using fastwarc.")

    # Setup database
    db_conn = setup_database(db_path, truncate=truncate_db)

    all_results = {
        "files_analyzed": len(warc_files),
        "total_records_processed": 0,
        "total_articles_extracted": 0,
        "total_articles_inserted": 0,
        "total_cpu_processing_time": 0,
        "by_outlet": Counter(),
        "by_date": Counter(),
        "individual_results": [],
        "errors": []
    }

    start_time = time.time()

    # Determine number of processes
    effective_num_processes = num_processes
    if effective_num_processes is None:
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1
        effective_num_processes = min(available_cpus, len(warc_files))
    effective_num_processes = max(1, effective_num_processes)

    logging.info(f"Using {effective_num_processes} processes for parallel analysis.")

    # Prepare arguments for parallel processing
    process_args = [(warc_file, limit_per_file) for warc_file in warc_files]

    articles_to_insert = []
    processed_files_count = 0

    # Process files in parallel
    with multiprocessing.Pool(processes=effective_num_processes) as pool:
        try:
             # Use tqdm for progress bar
            progress_bar = tqdm(pool.imap_unordered(process_warc_file, process_args),
                                total=len(warc_files), desc="Processing WARC files (fastwarc)")

            for result in progress_bar:
                processed_files_count += 1
                file_path = result.get("warc_file", "Unknown file")

                if "error" in result and result["error"]:
                    logging.error(f"Error processing {file_path}: {result['error']}")
                    all_results["errors"].append({"file": file_path, "error": result["error"]})
                    all_results["total_records_processed"] += result.get("records_processed", 0)
                    all_results["total_cpu_processing_time"] += result.get("processing_time", 0)
                    continue

                # Accumulate results
                all_results["total_records_processed"] += result["records_processed"]
                all_results["total_articles_extracted"] += len(result.get("articles", []))
                all_results["total_cpu_processing_time"] += result["processing_time"]

                if "stats" in result:
                    all_results["by_outlet"].update(result["stats"].get("by_outlet", {}))
                    all_results["by_date"].update(result["stats"].get("by_date", {}))

                all_results["individual_results"].append({
                    "file": file_path,
                    "records_processed": result["records_processed"],
                    "articles_extracted": len(result.get("articles", [])),
                    "processing_time": result["processing_time"],
                    "processing_speed": result.get("processing_speed", 0)
                })

                articles_to_insert.extend(result.get("articles", []))

                # Batch Insert into Database
                INSERT_BATCH_SIZE = 5000
                FILES_PER_COMMIT = min(5, len(warc_files))
                if len(articles_to_insert) >= INSERT_BATCH_SIZE or processed_files_count % FILES_PER_COMMIT == 0:
                    if articles_to_insert:
                        logging.info(f"Inserting batch of {len(articles_to_insert)} articles into database...")
                        try:
                            db_conn.executemany("""
                                INSERT INTO articles
                                (id, title, url, datetime, publish_date, domain, outlet, content, source_file)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(url) DO NOTHING;
                            """, [
                                (
                                    a["id"], a["title"], a["url"], a["datetime"], a["publish_date"],
                                    a["domain"], a["outlet"], a["content"], a["source_file"]
                                ) for a in articles_to_insert
                            ])
                            db_conn.commit()
                            all_results["total_articles_inserted"] += len(articles_to_insert) # Assume success for now
                            logging.info(f"Successfully inserted/updated batch. Total potentially inserted: {all_results['total_articles_inserted']}.")
                            articles_to_insert = []
                        except Exception as e:
                            logging.error(f"Database batch insert error: {e}", exc_info=True)
                            articles_to_insert = []

        except KeyboardInterrupt:
            logging.warning("Analysis interrupted by user. Processing partial results...")
            pool.terminate()
            pool.join()
        finally:
             # Final Insert
             if articles_to_insert:
                 logging.info(f"Inserting final batch of {len(articles_to_insert)} articles...")
                 try:
                     db_conn.executemany("""
                         INSERT INTO articles
                         (id, title, url, datetime, publish_date, domain, outlet, content, source_file)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                         ON CONFLICT(url) DO NOTHING;
                     """, [
                         (
                             a["id"], a["title"], a["url"], a["datetime"], a["publish_date"],
                             a["domain"], a["outlet"], a["content"], a["source_file"]
                         ) for a in articles_to_insert
                     ])
                     db_conn.commit()
                     all_results["total_articles_inserted"] += len(articles_to_insert)
                     logging.info(f"Successfully inserted final batch. Total potentially inserted: {all_results['total_articles_inserted']}.")
                 except Exception as e:
                     logging.error(f"Database final batch insert error: {e}", exc_info=True)
                     db_conn.rollback()


    # Calculate overall stats
    total_elapsed = time.time() - start_time

    # Verify final count in database
    try:
        final_article_count_db = db_conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        all_results["final_article_count_db"] = final_article_count_db
        if truncate_db:
             all_results["total_articles_inserted"] = final_article_count_db
    except Exception as e:
        logging.error(f"Could not get final article count from DB: {e}")
        all_results["final_article_count_db"] = "Error"


    # Create/refresh views
    try:
        create_views(db_conn)
    except Exception as e:
         logging.error(f"Failed to create views: {e}")

    # Performance Calculations
    avg_speed_wall = all_results["total_records_processed"] / total_elapsed if total_elapsed > 0 else 0
    avg_speed_cpu = all_results["total_records_processed"] / all_results["total_cpu_processing_time"] if all_results["total_cpu_processing_time"] > 0 else 0
    avg_time_per_file = total_elapsed / processed_files_count if processed_files_count > 0 else 0
    parallel_speedup = all_results["total_cpu_processing_time"] / total_elapsed if total_elapsed > 0 else 0


    # Report Generation
    report_file = Path(f"taiwan_news_analysis_fastwarc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    logging.info(f"Generating analysis report: {report_file}")
    try:
        with report_file.open('w', encoding='utf-8') as f:
            f.write("# Taiwanese News Media Analysis Report (using fastwarc)\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if args:
                f.write(f"**Input Paths Processed:** `{', '.join(args.input)}`\n")
                f.write(f"**File Pattern:** `{args.pattern}`\n")
                f.write(f"**Recursive Search:** Enabled for directories\n")
                if args.limit:
                    f.write(f"**Record Limit Per File:** {args.limit:,}\n")
                if args.max_files:
                     f.write(f"**Overall File Limit:** {args.max_files:,} (applied before processing)\n")
                f.write(f"**Database Truncated:** {'Yes' if args.truncate else 'No'}\n")
            f.write("\n---\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Files Analyzed:** {processed_files_count:,} / {len(warc_files):,} (processed/found)\n")
            f.write(f"- **Total Records Processed:** {all_results['total_records_processed']:,}\n")
            f.write(f"- **Total Articles Extracted:** {all_results['total_articles_extracted']:,}\n")
            f.write(f"- **Final Articles in Database:** {all_results.get('final_article_count_db', 'N/A'):,}\n")
            if all_results["errors"]:
                 f.write(f"- **Errors Encountered:** {len(all_results['errors']):,} (see logs for details)\n")
            f.write("\n---\n\n")


            f.write("## Performance Metrics\n\n")
            f.write(f"- **Parallel Processing:** {effective_num_processes} processes\n")
            f.write(f"- **Total Wall-Clock Time:** {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)\n")
            f.write(f"- **Total CPU Processing Time (Sum):** {all_results['total_cpu_processing_time']:.2f} seconds ({all_results['total_cpu_processing_time']/60:.2f} minutes)\n")
            f.write(f"- **Parallel Speedup Factor:** {parallel_speedup:.2f}x (CPU Time / Wall Time)\n")
            f.write(f"- **Avg Processing Speed (Wall Clock):** {avg_speed_wall:,.2f} records/second\n")
            f.write(f"- **Avg Processing Speed (CPU):** {avg_speed_cpu:,.2f} records/second\n")
            f.write(f"- **Avg Wall-Clock Time per File:** {avg_time_per_file:.2f} seconds\n\n")

            if processed_files_count > 0 and avg_time_per_file > 0:
                 ESTIMATED_YEARLY_FILES = 618 * 12
                 estimated_full_time_secs = avg_time_per_file * ESTIMATED_YEARLY_FILES
                 estimated_full_hours = estimated_full_time_secs / 3600
                 f.write(f"**Estimated Time for {ESTIMATED_YEARLY_FILES:,} files (approx. 1 year):** "
                         f"{estimated_full_hours:.2f} hours (based on observed avg. time/file)\n\n")
            else:
                 f.write("**Could not estimate time for a full year.**\n\n")

            f.write("---\n\n")

            f.write("## Results by News Outlet\n\n")
            f.write("| News Outlet | Articles Extracted |\n")
            f.write("|-------------|--------------------|\n")
            sorted_outlets = all_results["by_outlet"].most_common()
            if not sorted_outlets:
                 f.write("| *No articles found* | 0 |\n")
            for outlet, count in sorted_outlets:
                f.write(f"| {outlet} | {count:,} |\n")
            f.write("\n---\n\n")


            f.write("## Results by Date (WARC Record Date)\n\n")
            f.write("| Date       | Articles Extracted |\n")
            f.write("|------------|--------------------|\n")
            sorted_dates = sorted(all_results["by_date"].items())
            if not sorted_dates:
                 f.write("| *No dates found* | 0 |\n")
            display_limit = 40
            if len(sorted_dates) > display_limit:
                dates_to_show = sorted_dates[:display_limit//2] + sorted_dates[-display_limit//2:]
                dates_omitted = len(sorted_dates) - display_limit
            else:
                dates_to_show = sorted_dates
                dates_omitted = 0

            for date, count in dates_to_show:
                f.write(f"| {date} | {count:,} |\n")
            if dates_omitted > 0:
                f.write(f"| ... *({dates_omitted} other dates omitted)* ... | ... |\n")

            f.write("\n---\n\n")

            f.write("## Individual File Performance (Sample)\n\n")
            f.write("| File | Records | Articles Extracted | Time (s) | Speed (rec/s) |\n")
            f.write("|------|---------|--------------------|----------|---------------|\n")
            sample_results = sorted(all_results["individual_results"], key=lambda x: x['processing_time'], reverse=True)
            display_count = 40
            if not sample_results:
                 f.write("| *No files processed successfully* | - | - | - | - |\n")
            else:
                if len(sample_results) > display_count:
                    sample_to_show = sample_results[:display_count]
                    files_omitted = len(sample_results) - display_count
                else:
                    sample_to_show = sample_results
                    files_omitted = 0

                for result in sample_to_show:
                    file_name = Path(result["file"]).name
                    f.write(f"| {file_name} | {result['records_processed']:,} | {result['articles_extracted']:,} | {result['processing_time']:.2f} | {result['processing_speed']:,.2f} |\n")

                if files_omitted > 0:
                     f.write(f"| ... *({files_omitted} other files omitted)* ... | ... | ... | ... | ... |\n")

            if all_results["errors"]:
                 f.write("\n---\n\n")
                 f.write("## Errors Summary\n\n")
                 f.write(f"Encountered {len(all_results['errors'])} errors during processing. Check `taiwan_news_analysis_fastwarc.log` for details.\n")
                 f.write("| File | Error Message |\n")
                 f.write("|------|---------------|\n")
                 for err in all_results["errors"][:20]:
                     f.write(f"| {Path(err['file']).name} | `{err['error'][:100]}`... |\n")
                 if len(all_results["errors"]) > 20:
                     f.write("| ... | *(additional errors logged)* |\n")


    except Exception as e:
         logging.error(f"Failed to write analysis report: {e}", exc_info=True)


    # Final Logging
    logging.info("-" * 50)
    logging.info("Analysis Summary (fastwarc):")
    logging.info(f"  Files Analyzed: {processed_files_count:,} / {len(warc_files):,}")
    logging.info(f"  Total Records Processed: {all_results['total_records_processed']:}")
    logging.info(f"  Total Articles Extracted: {all_results['total_articles_extracted']:}")
    logging.info(f"  Final Articles in DB: {all_results.get('final_article_count_db', 'N/A'):,}")
    logging.info(f"  Total Wall Clock Time: {total_elapsed:.2f} s ({total_elapsed/60:.2f} min)")
    logging.info(f"  Avg Wall Clock Time / File: {avg_time_per_file:.2f} s")
    logging.info(f"  Avg Speed (Wall Clock): {avg_speed_wall:,.2f} rec/s")
    if all_results["errors"]:
         logging.warning(f"  Errors Encountered: {len(all_results['errors'])}")
    logging.info(f"Analysis report generated at: {report_file}")
    logging.info("-" * 50)

    # Close database connection
    db_conn.close()

    return all_results

def create_views(conn: duckdb.DuckDBPyConnection):
    """Create or refresh useful views in the DuckDB database."""
    logging.info("Creating/refreshing database views...")

    # Articles by date (using parsed WARC datetime)
    conn.execute("""
        CREATE OR REPLACE VIEW articles_by_date AS
        SELECT
            CAST(datetime AS DATE) as article_date,
            COUNT(*) as article_count
        FROM articles
        WHERE article_date IS NOT NULL
        GROUP BY article_date
        ORDER BY article_date;
    """)

    # Articles by extracted publish date (might be less complete)
    conn.execute("""
        CREATE OR REPLACE VIEW articles_by_publish_date AS
        SELECT
            CAST(strptime(publish_date, '%Y-%m-%dT%H:%M:%SZ') AS DATE) as published_date_parsed,
            COUNT(*) as article_count
        FROM articles
        WHERE published_date_parsed IS NOT NULL
        GROUP BY published_date_parsed
        ORDER BY published_date_parsed;
    """)

    # Articles by domain
    conn.execute("""
        CREATE OR REPLACE VIEW articles_by_domain AS
        SELECT
            domain,
            COUNT(*) as article_count
        FROM articles
        GROUP BY domain
        ORDER BY article_count DESC;
    """)

    # Articles by outlet
    conn.execute("""
        CREATE OR REPLACE VIEW articles_by_outlet AS
        SELECT
            outlet,
            COUNT(*) as article_count
        FROM articles
        GROUP BY outlet
        ORDER BY article_count DESC;
    """)

    # View for articles missing content
    conn.execute("""
         CREATE OR REPLACE VIEW articles_missing_content AS
         SELECT id, url, title, outlet, source_file
         FROM articles
         WHERE content IS NULL OR TRIM(content) = '';
    """)

    logging.info("Database views created successfully.")

def main():
    global args

    parser = argparse.ArgumentParser(
        description="Analyze WARC files for Taiwanese news content using fastwarc and parallel processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input",
                        help="Input WARC file(s) or director(y/ies) containing WARC files.",
                        required=True,
                        nargs='+')
    parser.add_argument("--output",
                        help="Output DuckDB database path.",
                        default="./data/processed/taiwan_news_articles_fastwarc.duckdb")
    parser.add_argument("--max-files",
                        help="Maximum number of WARC files to process overall (processed in found order).",
                        type=int,
                        default=None)
    parser.add_argument("--limit",
                        help="Limit number of records to process per WARC file (for testing).",
                        type=int,
                        default=None)
    parser.add_argument("--pattern",
                        help="Glob pattern for matching WARC files within input directories (recursive).",
                        default="*.warc.gz")
    parser.add_argument("--truncate",
                        help="Truncate (delete and recreate) the output database before starting.",
                        action="store_true")
    parser.add_argument("--processes",
                        help="Number of parallel processes to use (default: auto-detect available CPUs).",
                        type=int)

    args = parser.parse_args()

    # Input File Discovery
    warc_files_to_process = []
    input_paths_were_directories = False

    logging.info(f"Scanning input paths: {args.input}")
    logging.info(f"Using file pattern: {args.pattern}")

    for input_path_str in args.input:
        input_path = Path(input_path_str).resolve()
        if not input_path.exists():
            logging.warning(f"Input path not found, skipping: {input_path_str}")
            continue

        if input_path.is_file():
            if input_path.match(args.pattern):
                 warc_files_to_process.append(input_path)
                 logging.debug(f"Added specific file: {input_path}")
            else:
                 logging.warning(f"Skipping explicitly provided file {input_path} as it doesn't match pattern '{args.pattern}'")

        elif input_path.is_dir():
            input_paths_were_directories = True
            logging.info(f"Recursively scanning directory '{input_path}' for pattern '{args.pattern}'...")
            found_files = list(input_path.rglob(args.pattern))
            warc_files_to_process.extend(found_files)
            logging.info(f"Found {len(found_files):,} files in '{input_path}' and its subdirectories.")
        else:
            logging.warning(f"Input path is not a file or directory, skipping: {input_path_str}")

    # File List Preparation
    unique_warc_files = sorted(list(set(warc_files_to_process)))
    logging.info(f"Found {len(unique_warc_files):,} unique WARC files matching the pattern across all inputs.")

    if not unique_warc_files:
        logging.error("No WARC files found to process based on the input paths and pattern. Exiting.")
        return

    if args.max_files is not None and len(unique_warc_files) > args.max_files:
        logging.info(f"Limiting analysis to the first {args.max_files:,} files found (out of {len(unique_warc_files):,}).")
        unique_warc_files = unique_warc_files[:args.max_files]

    # Execute Analysis
    try:
        analyze_warc_files_parallel(
            warc_files=unique_warc_files,
            db_path=args.output,
            limit_per_file=args.limit,
            truncate_db=args.truncate,
            num_processes=args.processes
        )
    except Exception as e:
        logging.critical(f"An unhandled error occurred during parallel analysis: {e}", exc_info=True)
    finally:
        logging.info("Analysis script finished.")


if __name__ == "__main__":
    # Example Usage from command line (updated for fastwarc version):
    # uv run analyze_taiwan_news_fastwarc.py --input /path/to/warc/dir1 /path/to/warc/file.warc.gz --output ./output_fastwarc.duckdb --truncate --limit 1000
    # uv run analyze_taiwan_news_fastwarc.py --input /data/crawl-data/CC-NEWS/2022 --output ./data/processed/tw_news_2022_fastwarc.duckdb --truncate
    main() 