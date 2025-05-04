#!/usr/bin/env python3
import os
import re
import argparse
import time
import duckdb
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from fastwarc.warc import ArchiveIterator, WarcRecordType, WarcRecord
from fastwarc.stream_io import GZipStream, FileStream, FastWARCError # Assuming GZip, add LZ4 if needed
from urllib.parse import urlparse
from tqdm import tqdm

# --- Constants ---
INTERMEDIATE_DB_TABLE = "raw_html"
# Example batch size, can be tuned
INSERT_BATCH_SIZE = 1000
# Example commit frequency (in terms of files processed), can be tuned
FILES_PER_COMMIT = 5
# Default log directory
DEFAULT_LOG_DIR = Path("./logs/warc_extract/stage1")
# Define intermediate DB schema (using TEXT for warc_id for flexibility, BLOB for content)
INTERMEDIATE_SCHEMA = f"""
    warc_id TEXT PRIMARY KEY,
    warc_date TIMESTAMP,
    url TEXT,
    domain TEXT,
    outlet TEXT,
    html_content_bytes BLOB
"""

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Stage 1: Scan WARC files, filter by domain, extract metadata and raw HTML bytes, store in intermediate DuckDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input",
                        help="Input WARC file(s) or director(y/ies) containing WARC files.",
                        required=True,
                        nargs='+')
    parser.add_argument("--intermediate-db",
                        help="Path for the intermediate DuckDB database file.",
                        default="./data/intermediate/intermediate_articles.duckdb")
    parser.add_argument("--max-files",
                        help="Maximum number of WARC files to process overall.",
                        type=int,
                        default=None)
    parser.add_argument("--limit-records",
                        help="Limit number of records to process per WARC file (for testing).",
                        type=int,
                        default=None)
    parser.add_argument("--pattern",
                        help="Glob pattern for matching WARC files within input directories.",
                        default="*.warc.gz")
    parser.add_argument("--truncate",
                        help="Truncate (delete and recreate) the intermediate database before starting.",
                        action="store_true")
    parser.add_argument("--processes",
                        help="Number of parallel processes (default: auto-detect).",
                        type=int)
    parser.add_argument("--log-dir",
                        help="Directory for log files.",
                        default=str(DEFAULT_LOG_DIR))
    # TODO: Add argument for config file path (for domains) later
    # parser.add_argument("--config", help="Path to config file (YAML/JSON) for domains.", default="config.yaml")

    return parser.parse_args()

# --- Logging Setup ---
def setup_logging(log_dir: str | Path):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_name = f"stage1_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = log_dir / log_file_name

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler() # Also print logs to console
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file_path}")
    return log_file_path

# --- Domain Loading & Utility Functions ---
# TODO: Load this from a config file (YAML/JSON) via args.config
# For now, using a simplified version of the list from the original script
TAIWANESE_DOMAINS_CONFIG = [
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
    {"name": "China Times (中國時報)", "domains": ["www.chinatimes.com", "chinatimes.com"]},
    {"name": "Apple Daily (蘋果日報)", "domains": ["tw.appledaily.com", "appledaily.com.tw"]},
    {"name": "Central News Agency (中央社)", "domains": ["www.cna.com.tw", "cna.com.tw"]},
    {"name": "Yahoo News Taiwan", "domains": ["tw.news.yahoo.com"]},
    {"name": "Now News (今日新聞)", "domains": ["www.nownews.com", "nownews.com"]}
]

def load_domain_config(config_data: list[dict]) -> tuple[dict[str, str], set[str]]:
    """Process the domain configuration list into lookup structures."""
    domain_to_outlet: dict[str, str] = {}
    all_target_domains: set[str] = set()
    for outlet in config_data:
        outlet_name = outlet["name"]
        for domain in outlet["domains"]:
            domain_to_outlet[domain] = outlet_name
            all_target_domains.add(domain)
    logging.info(f"Loaded {len(all_target_domains)} target domains for {len(config_data)} outlets.")
    return domain_to_outlet, all_target_domains

# (This function could be moved to a shared warc_utils.py later)
def extract_domain(url: str | None) -> tuple[str | None, str | None]:
    """Extract domain from URL, returning both raw and www/no-www variants."""
    if not url:
        return None, None
    try:
        parsed_url = urlparse(url)
        # Handle URLs without // (e.g., "dns:example.com") gracefully
        if '//' not in url and not parsed_url.scheme and parsed_url.path:
            domain = parsed_url.path.split('/')[0].lower()
        else:
            domain = parsed_url.netloc.lower()

        if not domain:
            # Attempt to extract from path if netloc is empty (e.g., file:///path/domain.com)
            if parsed_url.path and '.' in parsed_url.path.split('/')[-1]:
                 domain = parsed_url.path.split('/')[-1].lower()
            else:
                 return None, None

        # Simple check if it looks like a domain name (contains a dot)
        # More robust validation could be added if needed
        if '.' not in domain:
             logging.debug(f"Extracted domain '{domain}' from URL '{url}' lacks a dot, ignoring.")
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

# --- Database Setup ---
def setup_intermediate_database(db_path_str: str | Path, truncate: bool = False) -> duckdb.DuckDBPyConnection:
    """Setup the intermediate DuckDB database for storing raw HTML bytes."""
    db_path = Path(db_path_str)
    db_dir = db_path.parent
    db_dir.mkdir(parents=True, exist_ok=True)

    operation = "Connecting to" if db_path.exists() else "Creating"
    logging.info(f"{operation} intermediate database: {db_path}")

    if truncate and db_path.exists():
        try:
            db_path.unlink()
            logging.info(f"Truncated existing intermediate database: {db_path}")
        except OSError as e:
            logging.error(f"Failed to remove intermediate database file {db_path}: {e}")
            # Decide if this is fatal or if we should try to continue
            raise # Re-raise for now, assuming it's fatal

    try:
        # Connect to DuckDB
        conn = duckdb.connect(str(db_path))

        # Create raw_html table if it doesn't exist
        # Use INTERMEDIATE_SCHEMA constant defined earlier
        conn.execute(f"CREATE TABLE IF NOT EXISTS {INTERMEDIATE_DB_TABLE} ({INTERMEDIATE_SCHEMA});")

        # Optional: Add indexes if needed for potential lookups, though Stage 2 reads full table
        # conn.execute(f"CREATE INDEX IF NOT EXISTS idx_intermediate_domain ON {INTERMEDIATE_DB_TABLE}(domain);"))

        logging.info(f"Intermediate database table '{INTERMEDIATE_DB_TABLE}' setup complete.")
        return conn
    except duckdb.Error as e:
        logging.critical(f"Failed to connect to or setup intermediate database {db_path}: {e}")
        raise # Propagate the error

# --- WARC Processing Function ---
def process_warc_file_stage1(args_tuple: tuple) -> dict:
    """Processes a single WARC file for Stage 1.

    Args:
        args_tuple: A tuple containing (warc_file_path, limit_records, domain_to_outlet, all_target_domains).

    Returns:
        A dictionary summarizing the results for this file.
    """
    warc_file_path, limit_records, domain_to_outlet, all_target_domains = args_tuple
    file_basename = warc_file_path.name
    results = {
        "warc_file": str(warc_file_path),
        "records_processed": 0,
        "html_records_found": 0, # Total HTML records encountered (any domain)
        "target_domain_hits": 0, # Records matching target domains
        "target_html_extracted": 0, # Target domain records that were HTML and bytes extracted
        "html_bytes_extracted": 0, # Sum of bytes for extracted HTML
        "domain_counts": Counter(),
        "articles_to_insert": [],
        "error": None,
        "processing_time": 0
    }
    start_time = time.time()

    try:
        # Use FileStream and GZipStream with context manager
        with FileStream(str(warc_file_path), 'rb') as fs, GZipStream(fs) as stream: # Add LZ4Stream logic if needed based on filename

            # Iterate through response records without initial HTTP parsing
            # Note: Only WarcRecordType.response should contain HTTP data we need
            archive_iterator = ArchiveIterator(stream, record_types=WarcRecordType.response, parse_http=False)

            for record_count, record in enumerate(archive_iterator):
                results["records_processed"] += 1

                if limit_records and record_count >= limit_records:
                    logging.debug(f"Reached record limit ({limit_records}) for {file_basename}")
                    break

                # --- Basic Header Extraction ---
                url = record.headers.get('WARC-Target-URI')
                warc_id = record.headers.get('WARC-Record-ID')
                date_str = record.headers.get('WARC-Date')

                if not url or not warc_id: # URL and WARC-ID are essential
                    logging.debug(f"Skipping record {record_count} in {file_basename} due to missing URL or WARC-ID.")
                    continue

                warc_dt = None
                if date_str:
                    try:
                        # Convert to datetime object, assuming UTC (Z suffix)
                        warc_dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except ValueError:
                        logging.warning(f"Could not parse WARC-Date '{date_str}' in {file_basename}, record {warc_id}. Storing as None.")

                # --- Domain Extraction and Counting ---
                domain, alt_domain = extract_domain(url)
                if domain:
                    results["domain_counts"][domain] += 1
                    # Optionally count alt_domain as well if different?
                    # if alt_domain and alt_domain != domain:
                    #     results["domain_counts"][alt_domain] += 1
                else:
                    logging.debug(f"Could not extract domain from URL: {url} in {file_basename}")
                    continue # Skip if no domain found

                # --- Target Domain Check ---
                outlet_name = None
                is_target = False
                if domain in all_target_domains:
                    outlet_name = domain_to_outlet.get(domain)
                    is_target = True
                elif alt_domain and alt_domain in all_target_domains:
                    outlet_name = domain_to_outlet.get(alt_domain)
                    is_target = True
                    domain = alt_domain # Use the matched domain for consistency

                if is_target:
                    results["target_domain_hits"] += 1
                    # --- Process Target Record (Parse HTTP, Check Type, Extract Bytes) ---
                    try:
                        record.parse_http(auto_decode='content') # Handle headers & decompression

                        content_type = record.http_content_type
                        if content_type and 'html' in content_type.lower():
                            results["html_records_found"] += 1 # Increment here as it's confirmed HTML

                            html_bytes = record.reader.read()

                            if not html_bytes:
                                logging.debug(f"Empty HTML content body for {url} in {file_basename}, record {warc_id}.")
                                continue

                            results["target_html_extracted"] += 1
                            results["html_bytes_extracted"] += len(html_bytes)

                            # Prepare data for batch insertion
                            article_data = (
                                warc_id,
                                warc_dt, # Store datetime object or None
                                url,
                                domain, # Store the primary matched domain
                                outlet_name,
                                html_bytes # Store raw bytes
                            )
                            results["articles_to_insert"].append(article_data)

                        # else: Record was target domain but not HTML content

                    except FastWARCError as e:
                        logging.error(f"FastWARC error processing record {warc_id} ({url}) in {file_basename}: {e}", exc_info=False)
                        # Decide if this error is per-record or file-level
                        # For now, log and continue to next record
                    except IOError as e:
                        logging.error(f"IO error reading content for record {warc_id} ({url}) in {file_basename}: {e}", exc_info=False)
                        # Likely indicates a stream/content issue, maybe stop processing file?
                        results["error"] = f"IOError reading record content: {e}"
                        break # Stop processing this file on IO error
                    except Exception as e:
                        logging.error(f"Unexpected error processing record {warc_id} ({url}) in {file_basename}: {e}", exc_info=True)
                        # Log unexpected errors and continue if possible

    except FileNotFoundError:
        logging.error(f"WARC file not found: {warc_file_path}")
        results["error"] = "File not found"
    except FastWARCError as e:
        logging.error(f"FastWARC error processing file {file_basename}: {e}")
        results["error"] = f"FastWARCError: {e}"
    except IOError as e:
        logging.error(f"IO error opening or reading file {file_basename}: {e}")
        results["error"] = f"IOError: {e}"
    except Exception as e:
        logging.error(f"Unhandled exception processing file {file_basename}: {e}", exc_info=True)
        results["error"] = f"Unhandled exception: {e}"
    finally:
        results["processing_time"] = time.time() - start_time
        log_level = logging.INFO if results["error"] else logging.DEBUG
        status_message = f"error: {results['error']}" if results["error"] else "successfully"
        
        logging.log(log_level, 
                    f"Finished processing {file_basename} {status_message} "
                    f"({results['records_processed']} records processed, "
                    f"{results['target_html_extracted']} target HTML extracted in {results['processing_time']:.2f}s)"
                   )

    return results

# --- Report Generation ---
def generate_stage1_report(results: dict, args: argparse.Namespace, log_file_path: Path):
    """Generates a Markdown report summarizing the Stage 1 scan results."""
    report_file_name = f"stage1_scan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    # Place report in the same directory as the log file
    report_file_path = log_file_path.parent / report_file_name
    logging.info(f"Generating Stage 1 analysis report: {report_file_path}")

    # Sort domain counts for reporting
    sorted_domain_counts = results.get("overall_domain_counts", Counter()).most_common()

    try:
        with report_file_path.open('w', encoding='utf-8') as f:
            f.write("# Stage 1: WARC Scan & Raw HTML Extraction Report\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Log File:** `{log_file_path.name}`\n")
            f.write(f"**Input Paths Processed:** `{', '.join(args.input)}`\n")
            f.write(f"**File Pattern:** `{args.pattern}`\n")
            if args.limit_records:
                f.write(f"**Record Limit Per File:** {args.limit_records:,}\n")
            if args.max_files:
                 f.write(f"**Overall File Limit:** {args.max_files:,}\n")
            f.write(f"**Intermediate DB:** `{args.intermediate_db}`\n")
            f.write(f"**Intermediate DB Truncated:** {'Yes' if args.truncate else 'No'}\n")
            f.write("\n---\n\n")

            f.write("## Summary\n\n")
            processed_files = results.get('files_processed', 0)
            total_files_found = len(args.input) # This isn't quite right if inputs are dirs, need total from discovery
            # TODO: Pass total files found into this function if needed for accurate reporting
            f.write(f"- **Files Processed:** {processed_files:,}\n")
            f.write(f"- **Total Records Scanned:** {results.get('total_records_processed', 0):,}\n")
            f.write(f"- **Total HTML Records Found (any domain):** {results.get('total_html_records_found', 0):,}\n")
            f.write(f"- **Target Domain Hits (any content type):** {results.get('total_target_domain_hits', 0):,}\n")
            f.write(f"- **Target HTML Records Extracted (Bytes):** {results.get('total_target_html_extracted', 0):,}\n")
            f.write(f"- **Articles Inserted into Intermediate DB:** {results.get('total_articles_inserted', 0):,}\n")
            total_bytes_mb = results.get('total_html_bytes_extracted', 0) / (1024 * 1024)
            f.write(f"- **Total HTML Bytes Extracted (Target):** {results.get('total_html_bytes_extracted', 0):,} bytes ({total_bytes_mb:.2f} MB)\n")
            if results.get("errors"):
                 f.write(f"- **Errors Encountered in Files:** {len(results['errors']):,} (see log/report details)\n")
            f.write("\n---\n\n")

            f.write("## Performance Metrics\n\n")
            wall_time = results.get('total_wall_time', 0)
            cpu_time = results.get('total_cpu_processing_time', 0)
            avg_speed_wall = results['total_records_processed'] / wall_time if wall_time > 0 else 0
            avg_speed_cpu = results['total_records_processed'] / cpu_time if cpu_time > 0 else 0
            avg_time_per_file = wall_time / processed_files if processed_files > 0 else 0
            parallel_speedup = cpu_time / wall_time if wall_time > 0 else 0

            f.write(f"- **Parallel Processing:** {args.processes or 'Auto'} processes used\n") # Reflect actual used if possible
            f.write(f"- **Total Wall-Clock Time:** {wall_time:.2f} seconds ({wall_time/60:.2f} minutes)\n")
            f.write(f"- **Total CPU Processing Time (Sum):** {cpu_time:.2f} seconds ({cpu_time/60:.2f} minutes)\n")
            f.write(f"- **Parallel Speedup Factor:** {parallel_speedup:.2f}x (CPU Time / Wall Time)\n")
            f.write(f"- **Avg Processing Speed (Wall Clock):** {avg_speed_wall:,.2f} records/second\n")
            f.write(f"- **Avg Processing Speed (CPU):** {avg_speed_cpu:,.2f} records/second\n")
            f.write(f"- **Avg Wall-Clock Time per File:** {avg_time_per_file:.2f} seconds\n\n")
            f.write("---\n\n")

            f.write("## Domain Statistics (All Encountered Domains)\n\n")
            f.write("| Domain | Records Found |\n")
            f.write("|--------|---------------|\n")
            if not sorted_domain_counts:
                 f.write("| *No domains found* | 0 |\n")
            display_limit = 100 # Show top N domains
            for domain, count in sorted_domain_counts[:display_limit]:
                f.write(f"| {domain} | {count:,} |\n")
            if len(sorted_domain_counts) > display_limit:
                f.write(f"| ... *({len(sorted_domain_counts) - display_limit} other domains omitted)* ... | ... |\n")
            f.write("\n---\n\n")

            if results.get("errors"):
                 f.write("## Errors Summary\n\n")
                 f.write(f"Encountered errors during processing for {len(results['errors'])} file(s). Check log `{log_file_path.name}` for full details.\n\n")
                 f.write("| File | Error Message Snippet |\n")
                 f.write("|------|-----------------------|\n")
                 for err in results["errors"][:50]: # Show sample of errors
                     file_name = Path(err['file']).name
                     error_msg_snippet = str(err['error'])[:100].replace("\n", " ")
                     f.write(f"| {file_name} | `{error_msg_snippet}`... |\n")
                 if len(results["errors"]) > 50:
                     f.write("| ... | *(additional errors logged)* |\n")

    except Exception as e:
         logging.error(f"Failed to write Stage 1 analysis report: {e}", exc_info=True)

# --- Main Parallel Execution Logic ---
def scan_warc_files_parallel(warc_files: list[Path], intermediate_db_path: str | Path, limit_records_per_file: int | None, truncate_db: bool, num_processes: int | None, domain_config: list[dict], args_for_report: argparse.Namespace, log_file_path_for_report: Path):
    """Scan WARC files in parallel and store results in the intermediate DB."""
    if not warc_files:
        logging.warning("No WARC files found or provided for Stage 1 scanning.")
        return

    logging.info(f"Starting Stage 1 scan of {len(warc_files)} WARC files.")

    # Load domain lookups
    domain_to_outlet, all_target_domains = load_domain_config(domain_config)

    # Setup intermediate database connection (main process handles schema)
    # Close connection, main process will handle inserts from results queue.
    try:
        db_conn_setup = setup_intermediate_database(intermediate_db_path, truncate=truncate_db)
        db_conn_setup.close()
    except Exception as e:
         logging.critical(f"Failed to setup intermediate database. Exiting Stage 1. Error: {e}")
         return # Stop if DB setup fails

    # Aggregate results and stats
    all_results = {
        "files_processed": 0,
        "total_records_processed": 0,
        "total_html_records_found": 0,
        "total_target_domain_hits": 0,
        "total_target_html_extracted": 0,
        "total_articles_inserted": 0,
        "total_html_bytes_extracted": 0,
        "overall_domain_counts": Counter(),
        "errors": [],
        "total_cpu_processing_time": 0,
        # Add more summary stats as needed
    }

    start_time_wall = time.time()

    # Determine number of processes
    effective_num_processes = num_processes
    if effective_num_processes is None:
        try:
            # Use sched_getaffinity on Linux/Unix for more accurate available CPU count
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1 # Fallback for other OS
        effective_num_processes = min(available_cpus, len(warc_files))
        # Consider adding a max cap if available_cpus is very high to avoid overwhelming IO
        # effective_num_processes = min(effective_num_processes, 32) # Example cap

    effective_num_processes = max(1, effective_num_processes)

    logging.info(f"Using {effective_num_processes} processes for parallel scanning.")

    # Prepare arguments for parallel processing
    # Pass necessary domain lookups to each process
    process_args = [
        (warc_file, limit_records_per_file, domain_to_outlet, all_target_domains)
        for warc_file in warc_files
    ]

    articles_to_insert_batch = []
    processed_files_count = 0
    db_conn_main = None # Initialize

    try:
        # Re-open DB connection in the main process for batch inserts
        # This avoids complex connection sharing/pickling issues with multiprocessing
        db_conn_main = duckdb.connect(str(intermediate_db_path))
        # Set PRAGMAs for potentially faster inserts if needed (use with caution)
        # db_conn_main.execute("PRAGMA threads=4;") # Example: Use multiple threads for DuckDB ops
        # db_conn_main.execute("PRAGMA memory_limit='1GB';") # Example: Set memory limit

        with multiprocessing.Pool(processes=effective_num_processes) as pool:
            progress_bar = tqdm(pool.imap_unordered(process_warc_file_stage1, process_args),
                                total=len(warc_files), desc="Stage 1 Scanning WARC files")

            for result in progress_bar:
                processed_files_count += 1
                file_path = result.get("warc_file", "Unknown file")

                # Handle errors from the process
                if result.get("error"):
                    logging.error(f"Error processing {file_path}: {result['error']}")
                    all_results["errors"].append({"file": file_path, "error": result["error"]})
                    # Still accumulate basic stats even if there was an error mid-file
                    all_results["total_records_processed"] += result.get("records_processed", 0)
                    all_results["total_cpu_processing_time"] += result.get("processing_time", 0)
                    all_results["overall_domain_counts"].update(result.get("domain_counts", Counter()))
                    all_results["total_html_records_found"] += result.get("html_records_found", 0)
                    all_results["total_target_domain_hits"] += result.get("target_domain_hits", 0)
                    all_results["total_target_html_extracted"] += result.get("target_html_extracted", 0)
                    all_results["total_html_bytes_extracted"] += result.get("html_bytes_extracted", 0)
                    continue

                # Accumulate results from successful file processing
                all_results["total_records_processed"] += result["records_processed"]
                all_results["total_html_records_found"] += result["html_records_found"]
                all_results["total_target_domain_hits"] += result["target_domain_hits"]
                all_results["total_target_html_extracted"] += result["target_html_extracted"]
                all_results["total_html_bytes_extracted"] += result["html_bytes_extracted"]
                all_results["overall_domain_counts"].update(result["domain_counts"])
                all_results["total_cpu_processing_time"] += result["processing_time"]

                # Add articles from this file's results to the main batch
                articles_to_insert_batch.extend(result.get("articles_to_insert", []))

                # Batch Insert into Intermediate Database (from main process)
                if len(articles_to_insert_batch) >= INSERT_BATCH_SIZE or processed_files_count % FILES_PER_COMMIT == 0:
                    if articles_to_insert_batch:
                        logging.info(f"Inserting batch of {len(articles_to_insert_batch)} raw articles into intermediate DB...")
                        try:
                            # Use INSERT OR IGNORE for basic PK conflict handling during the run
                            # Or ON CONFLICT DO NOTHING if INSERT OR IGNORE not supported/desired
                            # Note: DuckDB often prefers ON CONFLICT
                            start_insert_time = time.time()
                            db_conn_main.executemany(f"INSERT INTO {INTERMEDIATE_DB_TABLE} (warc_id, warc_date, url, domain, outlet, html_content_bytes) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(warc_id) DO NOTHING",
                                                       articles_to_insert_batch)
                            inserted_count = db_conn_main.last_changes()
                            db_conn_main.commit() # Commit after successful executemany
                            insert_duration = time.time() - start_insert_time
                            all_results["total_articles_inserted"] += inserted_count
                            logging.info(f"Successfully inserted batch ({inserted_count} new). Total inserted: {all_results['total_articles_inserted']:,}. Time: {insert_duration:.2f}s")
                            articles_to_insert_batch = [] # Clear batch
                        except duckdb.Error as e:
                            logging.error(f"Intermediate DB batch insert error: {e}", exc_info=True)
                            try:
                                db_conn_main.rollback()
                            except Exception as rb_e:
                                logging.error(f"Rollback failed after insert error: {rb_e}")
                            # Decide whether to stop or continue. For robustness, log and clear batch.
                            articles_to_insert_batch = [] # Clear batch to avoid re-attempting bad data
                        except Exception as e:
                             logging.error(f"Unexpected error during batch insert: {e}", exc_info=True)
                             try:
                                 db_conn_main.rollback()
                             except Exception as rb_e:
                                 logging.error(f"Rollback failed after unexpected insert error: {rb_e}")
                             articles_to_insert_batch = [] # Clear batch

    except KeyboardInterrupt:
        logging.warning("Stage 1 scan interrupted by user. Processing partial results...")
        if 'pool' in locals(): # Check if pool was created
             pool.terminate()
             pool.join()
    except Exception as e:
         logging.critical(f"Error during multiprocessing setup or coordination: {e}", exc_info=True)
         # Ensure pool is handled if it exists
         if 'pool' in locals() and pool._state == multiprocessing.pool.RUN: # Check state before terminate
             pool.terminate()
             pool.join()
    finally:
        # Final Insert for any remaining items in the batch
        if articles_to_insert_batch:
            logging.info(f"Inserting final batch of {len(articles_to_insert_batch)} raw articles...")
            try:
                start_insert_time = time.time()
                db_conn_main.executemany(f"INSERT INTO {INTERMEDIATE_DB_TABLE} (warc_id, warc_date, url, domain, outlet, html_content_bytes) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(warc_id) DO NOTHING",
                                           articles_to_insert_batch)
                inserted_count = db_conn_main.last_changes()
                db_conn_main.commit()
                insert_duration = time.time() - start_insert_time
                all_results["total_articles_inserted"] += inserted_count
                logging.info(f"Successfully inserted final batch ({inserted_count} new). Total inserted: {all_results['total_articles_inserted']:,}. Time: {insert_duration:.2f}s")
            except duckdb.Error as e:
                logging.error(f"Intermediate DB final batch insert error: {e}", exc_info=True)
                try:
                    db_conn_main.rollback()
                except Exception as rb_e:
                    logging.error(f"Rollback failed after final insert error: {rb_e}")
            except Exception as e:
                 logging.error(f"Unexpected error during final batch insert: {e}", exc_info=True)
                 try:
                     db_conn_main.rollback()
                 except Exception as rb_e:
                     logging.error(f"Rollback failed after unexpected final insert error: {rb_e}")

        # Close the main process DB connection
        if db_conn_main:
            try:
                db_conn_main.close()
                logging.info("Main process intermediate database connection closed.")
            except Exception as e:
                 logging.error(f"Error closing main DB connection: {e}")

    # Calculate overall stats
    total_elapsed_wall = time.time() - start_time_wall
    all_results["total_wall_time"] = total_elapsed_wall
    all_results["files_processed"] = processed_files_count

    logging.info("Finished Stage 1 scan.")
    # Generate the report using the collected results
    generate_stage1_report(all_results, args_for_report, log_file_path_for_report)

    return all_results

# --- Main Function ---
def main():
    args = parse_arguments()
    log_file_path = setup_logging(args.log_dir)
    logging.info(f"Stage 1 Arguments: {vars(args)}")

    # --- Input File Discovery ---
    warc_files_to_process = []
    logging.info(f"Scanning input paths: {args.input}")
    logging.info(f"Using file pattern: {args.pattern}")

    for input_path_str in args.input:
        input_path = Path(input_path_str).resolve()
        if not input_path.exists():
            logging.warning(f"Input path not found, skipping: {input_path_str}")
            continue

        if input_path.is_file():
            # Use fnmatch.fnmatch or similar if pattern is complex, Path.match is simpler
            if input_path.match(args.pattern):
                 warc_files_to_process.append(input_path)
                 logging.debug(f"Added specific file: {input_path}")
            else:
                 logging.warning(f"Skipping explicitly provided file {input_path} as it doesn't match pattern '{args.pattern}'")

        elif input_path.is_dir():
            logging.info(f"Recursively scanning directory '{input_path}' for pattern '{args.pattern}'...")
            # Using rglob for recursive search
            found_files = list(input_path.rglob(args.pattern))
            warc_files_to_process.extend(found_files)
            logging.info(f"Found {len(found_files):,} files matching pattern in '{input_path}' and subdirectories.")
        else:
            logging.warning(f"Input path is not a file or directory, skipping: {input_path_str}")

    # Deduplicate and sort the list of files
    unique_warc_files = sorted(list(set(warc_files_to_process)))
    logging.info(f"Found {len(unique_warc_files):,} unique WARC files matching the pattern across all inputs.")

    if not unique_warc_files:
        logging.error("No WARC files found to process based on the input paths and pattern. Exiting.")
        return

    # Apply overall file limit if specified
    if args.max_files is not None and len(unique_warc_files) > args.max_files:
        logging.info(f"Limiting analysis to the first {args.max_files:,} files found (out of {len(unique_warc_files):,}).")
        unique_warc_files = unique_warc_files[:args.max_files]

    # --- Execute Stage 1 Scan ---
    try:
        scan_results = scan_warc_files_parallel(
            warc_files=unique_warc_files,
            intermediate_db_path=args.intermediate_db,
            limit_records_per_file=args.limit_records,
            truncate_db=args.truncate,
            num_processes=args.processes,
            domain_config=TAIWANESE_DOMAINS_CONFIG,
            args_for_report=args,             # Pass args for reporting
            log_file_path_for_report=log_file_path # Pass log path for report location
        )

        # --- Final Summary Logging & Reporting ---
        # Check if scan_results is None (could happen if DB setup failed)
        if scan_results is None:
             logging.error("Stage 1 scan did not complete successfully.")
             return

        logging.info("--- Stage 1 Summary ---")
        logging.info(f"  Files Processed: {scan_results.get('files_processed', 0):,}")
        logging.info(f"  Total Records Scanned: {scan_results.get('total_records_processed', 0):,}")
        logging.info(f"  Total HTML Records Found (any domain): {scan_results.get('total_html_records_found', 0):,}")
        logging.info(f"  Target Domain HTML Records Found: {scan_results.get('total_target_domain_hits', 0):,}")
        logging.info(f"  Target HTML Records Extracted (Bytes): {scan_results.get('total_target_html_extracted', 0):,}")
        logging.info(f"  Articles Inserted into Intermediate DB: {scan_results.get('total_articles_inserted', 0):,}")
        logging.info(f"  Total HTML Bytes Extracted (Target): {scan_results.get('total_html_bytes_extracted', 0):,}")
        logging.info(f"  Total Wall Clock Time: {scan_results.get('total_wall_time', 0):.2f} s")
        logging.info(f"  Total CPU Time (sum): {scan_results.get('total_cpu_processing_time', 0):.2f} s")
        if scan_results.get('errors'):
            logging.warning(f"  Errors encountered in {len(scan_results['errors'])} file(s). Check log '{log_file_path}' for details.")

        # TODO: Implement detailed report generation to a file (e.g., markdown)
        # Include overall domain counts: scan_results.get('overall_domain_counts')
        # Include performance metrics, errors summary, etc.

    except Exception as e:
        logging.critical(f"An unhandled error occurred during the Stage 1 scan: {e}", exc_info=True)
    finally:
        logging.info("Stage 1 script finished.")


if __name__ == "__main__":
    # Example Usage:
    # uv run scripts/stage1_scan_warc.py --input /path/to/warc/dir --intermediate-db ./data/intermediate/stage1_output.duckdb --truncate
    # uv run scripts/stage1_scan_warc.py --input file1.warc.gz file2.warc.gz --intermediate-db ./inter.db --limit-records 1000 --processes 4
    main() 