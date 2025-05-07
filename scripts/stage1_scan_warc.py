#!/usr/bin/env python3
import os
import re
import argparse
import time
import logging
import multiprocessing
import uuid
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import polars as pl
from fastwarc.warc import ArchiveIterator, WarcRecordType, WarcRecord
from fastwarc.stream_io import GZipStream, FileStream, FastWARCError # Assuming GZip, add LZ4 if needed
from urllib.parse import urlparse
from tqdm import tqdm

# --- Constants ---
# Parquet output settings
DEFAULT_OUTPUT_DIR = Path("./data/intermediate/stage1_parquet")
# Removed INSERT_BATCH_SIZE, FILES_PER_COMMIT
DEFAULT_LOG_DIR = Path("./logs/warc_extract/stage1")
# Define expected columns for the output Parquet files (aligns with previous schema)
OUTPUT_COLUMNS = [
    'warc_id', 'warc_date', 'url', 'domain', 'outlet', 'is_english', 'html_content_bytes'
]

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Stage 1: Scan WARC files, filter by domain, extract metadata and raw HTML bytes, store in intermediate Parquet files (using zstd compression).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input",
                        help="Input WARC file(s) or director(y/ies) containing WARC files.",
                        required=True,
                        nargs='+')
    parser.add_argument("--output-dir",
                        help="Directory for intermediate Parquet output files.",
                        default=str(DEFAULT_OUTPUT_DIR))
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
def setup_logging(log_dir: Path | str, timestamp_str: str) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Use the provided timestamp string
    log_file_name = f"stage1_scan_{timestamp_str}.log"
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
TAIWANESE_DOMAINS_CONFIG = [
    {"name": "Economic Daily (經濟日報)", "domains": ["money.udn.com"], "is_english": False},
    {"name": "PTS News (公視新聞)", "domains": ["news.pts.org.tw", "pts.org.tw"], "is_english": False},
    {"name": "TVBS News (TVBS新聞台)", "domains": ["news.tvbs.com.tw", "tvbs.com.tw"], "is_english": False},
    {"name": "ETtoday (ETtoday新聞雲)", "domains": ["ettoday.net", "fashion.ettoday.net"], "is_english": False}, # Excluded www.ettoday.net; fashion.ettoday.net?
    {"name": "United Daily (聯合報)", "domains": ["udn.com"], "is_english": False},
    {"name": "EBC News (東森新聞)", "domains": ["news.ebc.net.tw", "ebc.net.tw"], "is_english": False},
    {"name": "TTV News (台視新聞)", "domains": ["ttv.com.tw"], "is_english": False},
    {"name": "CTV News (中視新聞)", "domains": ["new.ctv.com.tw", "ctv.com.tw"], "is_english": False},
    {"name": "CTS News (華視新聞)", "domains": ["news.cts.com.tw", "cts.com.tw"], "is_english": False},
    {"name": "Liberty Times (自由時報)", "domains": ["ltn.com.tw", "news.ltn.com.tw"], "is_english": False},
    {"name": "Storm Media (風傳媒)", "domains": ["storm.mg"], "is_english": False},
    {"name": "Formosa TV News (民視新聞)", "domains": ["ftvnews.com.tw"], "is_english": False},
    {"name": "SET News (三立新聞)", "domains": ["setn.com"], "is_english": False},
    {"name": "China Times (中國時報)", "domains": ["chinatimes.com"], "is_english": False},
    # {"name": "Apple Daily (蘋果日報)", "domains": ["tw.appledaily.com", "appledaily.com.tw"], "is_english": False}, # Defunct
    {"name": "Central News Agency (中央社)", "domains": ["cna.com.tw"], "is_english": False},
    # {"name": "Yahoo News Taiwan", "domains": ["tw.news.yahoo.com"], "is_english": False}, # Aggregator, handle separately?
    {"name": "Now News (今日新聞)", "domains": ["nownews.com"], "is_english": False},
    # Added English sources
    {"name": "Taipei Times", "domains": ["taipeitimes.com"], "is_english": True},
    {"name": "Taiwan News", "domains": ["taiwannews.com.tw"], "is_english": True},
    # Added based on whitelist check
    {"name": "The News Lens (關鍵評論網)", "domains": ["thenewslens.com"], "is_english": False} # Assuming Chinese source primarily
]

def load_domain_config(config_data: list[dict]) -> tuple[dict[str, str], set[str], dict[str, bool]]:
    """Process the domain configuration list into lookup structures."""
    domain_to_outlet: dict[str, str] = {}
    all_target_domains: set[str] = set()
    domain_to_language: dict[str, bool] = {}
    for outlet in config_data:
        outlet_name = outlet["name"]
        is_english = outlet.get("is_english", False) # Default to False if missing
        for domain in outlet["domains"]:
            domain_to_outlet[domain] = outlet_name
            all_target_domains.add(domain)
            domain_to_language[domain] = is_english
    logging.info(f"Loaded {len(all_target_domains)} target domains for {len(config_data)} outlets.")
    return domain_to_outlet, all_target_domains, domain_to_language

# (This function could be moved to a shared warc_utils.py later)
def extract_domain(url: str | None) -> str | None:
    """Extract the base domain from a URL, stripping www. and port."""
    if not url:
        return None
    try:
        parsed_url = urlparse(url)
        # Handle URLs without // (e.g., "dns:example.com") gracefully
        if '//' not in url and not parsed_url.scheme and parsed_url.path:
            domain_part = parsed_url.path.split('/')[0].lower()
        else:
            domain_part = parsed_url.netloc.lower()

        # Strip port number if present
        domain = domain_part.split(':')[0]

        if not domain:
            # Attempt to extract from path if netloc is empty (e.g., file:///path/domain.com)
            if parsed_url.path and '.' in parsed_url.path.split('/')[-1]:
                 domain = parsed_url.path.split('/')[-1].lower().split(':')[0]
            else:
                 return None

        # Always strip www.
        if domain.startswith('www.'):
            domain = domain[4:]

        # Allow 'localhost' explicitly
        if domain == 'localhost':
            return domain
        # Simple check if it looks like a domain name (contains a dot)
        elif '.' not in domain:
             logging.debug(f"Extracted part '{domain}' from URL '{url}' lacks a dot (and is not localhost), ignoring.")
             return None
        
        return domain

    except Exception as e:
        logging.warning(f"Could not parse domain from URL '{url}': {e}")
        return None

# --- WARC Processing Function ---
def process_warc_file_stage1(args_tuple: tuple) -> dict:
    """Processes a single WARC file for Stage 1, writing output to Parquet using zstd.

    Args:
        args_tuple: A tuple containing (warc_file_path, limit_records, domain_to_outlet,
                    all_target_domains, domain_to_language, output_subdir).
                    Note: output_subdir is the pre-calculated year/month path.

    Returns:
        A dictionary summarizing the results for this file (counts, errors).
    """
    # Unpack arguments, including pre-calculated output_subdir
    warc_file_path, limit_records, domain_to_outlet, all_target_domains, domain_to_language, output_subdir_path = args_tuple
    file_basename = warc_file_path.name
    results = {
        "warc_file": str(warc_file_path),
        "records_processed": 0,
        "html_records_found": 0, # Total HTML records encountered (any domain)
        "target_domain_hits": 0, # Records matching target domains
        "target_html_extracted": 0, # Target domain records that were HTML and bytes extracted
        "html_bytes_extracted": 0, # Sum of bytes for extracted HTML
        "domain_counts": Counter(),
        "parquet_file_path": None, # Store the path of the generated parquet file
        "error": None,
        "processing_time": 0
    }
    start_time = time.time()

    extracted_articles_data = []

    try:
        with FileStream(str(warc_file_path), 'rb') as fs, GZipStream(fs) as stream:
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
                domain = extract_domain(url)
                if domain:
                    results["domain_counts"][domain] += 1
                else:
                    logging.debug(f"Could not extract domain from URL: {url} in {file_basename}")
                    continue # Skip if no domain found

                # --- Target Domain Check ---
                outlet_name = None
                is_target = False
                if domain in all_target_domains:
                    outlet_name = domain_to_outlet.get(domain)
                    is_target = True

                if is_target:
                    results["target_domain_hits"] += 1
                    # Get the language flag for the matched domain
                    is_english_flag = domain_to_language.get(domain, False) # Default should align with load_domain_config

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

                            # Ensure we are inside the block where html_content_bytes is valid
                            if html_bytes: # Check if html_content_bytes exists and is not None/empty
                                extracted_articles_data.append({
                                    "warc_id": warc_id,
                                    "warc_date": warc_dt,
                                    "url": url,
                                    "domain": domain,
                                    "outlet": outlet_name,
                                    "is_english": is_english_flag,
                                    "html_content_bytes": html_bytes,
                                })

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

        # --- Write Extracted Data to Parquet (if any) ---
        if extracted_articles_data and not results["error"]:
            try:
                # Create Polars DataFrame
                # Specify schema to handle potential empty lists and ensure correct types
                # Convert datetime objects to timezone-naive UTC if they are timezone-aware
                # DuckDB stores TIMESTAMP WITH TIME ZONE, Parquet often expects naive UTC or specific formats
                # Polars handles aware datetime objects well, but conversion might be needed depending on Stage 2
                output_df = pl.DataFrame(extracted_articles_data) # Schema inferred here

                # MODIFIED: Parquet filename generation
                # Remove .warc.gz, .warc.lz4, then add .parquet
                base_for_parquet = file_basename
                if base_for_parquet.endswith(".warc.gz"):
                    base_for_parquet = base_for_parquet[:-len(".warc.gz")]
                elif base_for_parquet.endswith(".warc.lz4"):
                    base_for_parquet = base_for_parquet[:-len(".warc.lz4")]
                # Add other extensions if necessary
                
                output_filename = f"{base_for_parquet}.parquet"
                
                output_path = output_subdir_path / output_filename
                results["parquet_file_path"] = str(output_path)

                logging.debug(f"Writing {len(output_df)} extracted records from {file_basename} to {output_path} using zstd compression")
                # Write to Parquet using the specified compression
                # Ensure the output directory (including year/month) exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_df.write_parquet(output_path, compression='zstd')
                logging.debug(f"Successfully wrote Parquet file: {output_path}")

            except ImportError:
                # PyArrow is needed for zstd
                logging.error(f"Polars or PyArrow (for zstd) not installed. Cannot write Parquet. pip install polars[pyarrow]")
                results["error"] = f"Missing polars[pyarrow] for zstd Parquet output"
            except Exception as e:
                logging.error(f"Failed to write Parquet file {output_filename} for {file_basename}: {e}", exc_info=True)
                results["error"] = f"Parquet writing error: {e}"
                # Clear the parquet file path if writing failed
                results["parquet_file_path"] = None

        log_level = logging.INFO if results["error"] else logging.DEBUG
        status_message = f"error: {results['error']}" if results["error"] else "successfully"

        logging.log(log_level,
                    f"Finished processing {file_basename} {status_message} "
                    f"({results['records_processed']} records processed, "
                    f"{results['target_html_extracted']} target HTML extracted in {results['processing_time']:.2f}s)"
                   )

    # Return only summary stats
    # Remove sensitive/large data like the list of articles before returning
    # results.pop("articles_to_insert", None) # Already replaced with Parquet writing
    return results

# --- Report Generation ---
def generate_stage1_report(results: dict, args: argparse.Namespace, log_file_path: Path, timestamp_str: str):
    """Generates a Markdown report and saves domain counts to JSON."""
    report_dir = log_file_path.parent # Use log directory for reports
    # Use the provided timestamp string for report name
    report_file_name_md = f"stage1_scan_report_{timestamp_str}.md"
    report_file_path_md = report_dir / report_file_name_md
    # Use the same timestamp for the JSON domain counts file
    domain_counts_file_name_json = f"stage1_domain_counts_{timestamp_str}.json"
    domain_counts_file_path_json = report_dir / domain_counts_file_name_json

    logging.info(f"Generating Stage 1 analysis report (Markdown): {report_file_path_md}")
    logging.info(f"Saving full domain counts (JSON): {domain_counts_file_path_json}")

    overall_domain_counts = results.get("overall_domain_counts", Counter())
    sorted_domain_counts = overall_domain_counts.most_common()

    # --- Save Domain Counts to JSON --- 
    try:
        import json
        with domain_counts_file_path_json.open('w', encoding='utf-8') as f_json:
            # Convert Counter to a standard dict for JSON serialization
            json.dump(dict(overall_domain_counts), f_json, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save domain counts to JSON: {e}", exc_info=True)

    # --- Generate Markdown Report --- 
    try:
        with report_file_path_md.open('w', encoding='utf-8') as f:
            f.write("# Stage 1: WARC Scan & Raw HTML Extraction Report\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Log File:** `{log_file_path.name}`\n")
            f.write(f"**Domain Counts File:** `{domain_counts_file_path_json.name}`\n")
            f.write(f"**Input Paths Processed:** `{', '.join(args.input)}`\n")
            f.write(f"**File Pattern:** `{args.pattern}`\n")
            if args.limit_records:
                f.write(f"**Record Limit Per File:** {args.limit_records:,}\n")
            if args.max_files:
                 f.write(f"**Overall File Limit:** {args.max_files:,}\n")
            f.write(f"**Output Directory:** `{args.output_dir}`\n")
            f.write("\n---\n\n")

            f.write("## Summary\n\n")
            processed_files = results.get('files_processed', 0)
            total_files_found = len(args.input) # This isn't quite right if inputs are dirs, need total from discovery
            # TODO: Pass total files found into this function if needed for accurate reporting
            f.write(f"- **Files Processed:** {processed_files:,}\n")
            f.write(f"- **Total Records Scanned:** {results.get('total_records_processed', 0):,}\n")
            f.write(f"- **Total HTML Records Found (any domain):** {results.get('total_html_records_found', 0):,}\n")
            f.write(f"- **Target Domain Hits (any content type):** {results.get('total_target_domain_hits', 0):,}\n")
            f.write(f"- **Target HTML Records Extracted (to Parquet):** {results.get('total_target_html_extracted', 0):,}\n")
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
         logging.error(f"Failed to write Stage 1 analysis report (Markdown): {e}", exc_info=True)

# --- Main Parallel Execution Logic ---
def scan_warc_files_parallel(warc_files: list[Path], output_dir: Path | str, limit_records_per_file: int | None, num_processes: int | None, domain_config: list[dict], args_for_report: argparse.Namespace, log_file_path_for_report: Path, timestamp_str_for_report: str):
    """Scan WARC files in parallel and store results in Parquet files in the output directory.
       Output files are placed in year/month subdirectories derived from the WARC file path.
    """
    if not warc_files:
        logging.warning("No WARC files found or provided for Stage 1 scanning.")
        return

    logging.info(f"Starting Stage 1 scan of {len(warc_files)} WARC files. Outputting to: {output_dir}")
    output_dir_path = Path(output_dir)
    # Ensure output directory exists (worker process also checks, but good to have here too)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load domain lookups
    domain_to_outlet, all_target_domains, domain_to_language = load_domain_config(domain_config)

    # Aggregate results and stats
    all_results = {
        "files_processed": 0,
        "total_records_processed": 0,
        "total_html_records_found": 0,
        "total_target_domain_hits": 0,
        "total_target_html_extracted": 0,
        "total_html_bytes_extracted": 0,
        "overall_domain_counts": Counter(),
        "parquet_files_created": [], # List paths of created files
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
            # Fallback for other OS or if sched_getaffinity fails
            available_cpus = multiprocessing.cpu_count() or 1
        effective_num_processes = min(available_cpus, len(warc_files))
        # Consider adding a max cap if available_cpus is very high to avoid overwhelming IO
        # effective_num_processes = min(effective_num_processes, 32) # Example cap

    effective_num_processes = max(1, effective_num_processes)

    logging.info(f"Using {effective_num_processes} processes for parallel scanning.")

    # Pre-calculate output subdirectories and prepare arguments
    process_args = []
    path_pattern_ym = re.compile(r'.*/(\d{4})/(\d{2})/.*') # Pattern like /YYYY/MM/
    path_pattern_cc = re.compile(r'.*CC-MAIN-(\d{4})-(\d{2}).*') # Pattern like CC-MAIN-YYYY-MM
    path_pattern_cy = re.compile(r'.*-(\d{8})-.*') # Pattern like common_crawl/crawl-data/CC-MAIN-2023-06/segments/1674746793803.93/warc/CC-MAIN-20230126152633-20230126182633-00000.warc.gz extract 20230126
    
    for warc_file in warc_files:
        year, month = "unknown_date", "unknown_month"
        warc_path_str = str(warc_file)
        
        # Try different patterns to extract year/month
        match_ym = path_pattern_ym.match(warc_path_str)
        match_cc = path_pattern_cc.match(warc_path_str)
        match_cy = path_pattern_cy.match(warc_path_str)
        
        if match_ym:
            year, month = match_ym.groups()
            logging.debug(f"Extracted date {year}/{month} from path (YYYY/MM pattern): {warc_path_str}")
        elif match_cc:
            year, month = match_cc.groups()
            logging.debug(f"Extracted date {year}-{month} from path (CC-MAIN pattern): {warc_path_str}")
        elif match_cy:
            date_str = match_cy.group(1) # YYYYMMDD
            try:
                 year = date_str[:4]
                 month = date_str[4:6]
                 logging.debug(f"Extracted date {year}/{month} from path (YYYYMMDD pattern): {warc_path_str}")
            except IndexError:
                  logging.warning(f"Could not parse year/month from extracted date string '{date_str}' in path: {warc_path_str}")
        else:
            logging.warning(f"Could not extract year/month from WARC file path using known patterns: {warc_path_str}. Using fallback directory.")
            
        # Construct the specific output subdirectory for this file
        output_subdir_for_file = output_dir_path / year / month
        
        process_args.append(
            (warc_file, 
             limit_records_per_file, 
             domain_to_outlet, 
             all_target_domains, 
             domain_to_language, 
             output_subdir_for_file # Pass the specific subdir path
            )
        )

    processed_files_count = 0

    try:
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
                    # Still accumulate basic stats even if there was an error
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
                if result.get("parquet_file_path"):
                    all_results["parquet_files_created"].append(result["parquet_file_path"])

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
        # Calculate overall stats
        total_elapsed_wall = time.time() - start_time_wall
        all_results["total_wall_time"] = total_elapsed_wall
        all_results["files_processed"] = processed_files_count

        logging.info("Finished Stage 1 scan.")
        # Generate the report using the collected results and the timestamp
        generate_stage1_report(all_results, args_for_report, log_file_path_for_report, timestamp_str_for_report)

    return all_results

# --- Main Function ---
def main():
    args = parse_arguments()
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = setup_logging(Path(args.log_dir), run_timestamp) # Ensure Path object
    logging.info(f"Stage 1 Arguments: {vars(args)}")

    # Ensure output directory exists before starting parallel processes
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_dir}")

    # --- Whitelist Domains for Check ---
    # From user input - domains to check if they were found but are not currently targeted
    whitelist_domains = {
        "ltn.com.tw",           # 自由時報
        "taipeitimes.com",      # Taipei Times
        "udn.com",              # 聯合報
        "money.udn.com",        # 經濟日報
        "chinatimes.com",       # 中國時報
        "ctee.com.tw",          # 工商時報
        "sunmedia.tw",          # 商傳媒
        "cna.com.tw",           # 中央通訊社
        "peopo.org",            # PeoPo公民新聞
        "ettoday.net",          # ETtoday新聞雲
        "nownews.com",          # NOWnews今日新聞
        "tw.appledaily.com",    # 蘋果新聞網 (台灣版)
        "enn.tw",               # 台灣電報
        "17news.net",           # 民生好報
        "linews.tw",            # 新民生報
        "seenews.tw",           # 台新傳媒（台灣新新聞報）
        "storm.mg",             # 風傳媒
        "newtalk.tw",           # 新頭殼（Newtalk新聞）
        "thenewslens.com",      # 關鍵評論網（The News Lens）
        "twreporter.org",       # 報導者（The Reporter）
        "my-formosa.com.tw",    # 美麗島電子報
        "peoplenews.tw",        # 民報（Taiwan People News）
        "upmedia.mg",           # 上報（Up Media）
        "newsmarket.com.tw",    # 上下游新聞市集（News Market）
        "taiwannews.com.tw",    # Added based on previous steps
        # Add any other domains you want to specifically check here
    }
    logging.info(f"Will check if {len(whitelist_domains)} specific domains were found but not targeted.")

    # Load the target domain config to get the set of *currently targeted* domains
    _, current_target_domains, _ = load_domain_config(TAIWANESE_DOMAINS_CONFIG)
    logging.info(f"Loaded {len(current_target_domains)} domains currently targeted for extraction.")

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
    scan_results = None # Initialize
    try:
        scan_results = scan_warc_files_parallel(
            warc_files=unique_warc_files,
            output_dir=output_dir, # Pass Path object
            limit_records_per_file=args.limit_records,
            num_processes=args.processes,
            domain_config=TAIWANESE_DOMAINS_CONFIG, # This is used *during* the scan
            args_for_report=args, # Pass the full args namespace
            log_file_path_for_report=log_file_path,
            timestamp_str_for_report=run_timestamp
        )

        # --- Final Summary Logging ---
        if scan_results is None:
             logging.error("Stage 1 scan did not complete successfully.")
             return
        # Log summary based on collected results
        logging.info("--- Stage 1 Scan Summary ---")
        logging.info(f"Files Processed: {scan_results.get('files_processed', 0):,}")
        logging.info(f"Total Records Scanned: {scan_results.get('total_records_processed', 0):,}")
        logging.info(f"Target HTML Extracted (to Parquet): {scan_results.get('total_target_html_extracted', 0):,}")
        parquet_files_count = len(scan_results.get('parquet_files_created', []))
        logging.info(f"Parquet Files Created: {parquet_files_count:,} in {output_dir}")
        total_bytes_mb = scan_results.get('total_html_bytes_extracted', 0) / (1024 * 1024)
        logging.info(f"Total HTML Bytes Extracted: {total_bytes_mb:.2f} MB")
        logging.info(f"Total Wall Clock Time: {scan_results.get('total_wall_time', 0):.2f}s")
        errors_count = len(scan_results.get('errors', []))
        if errors_count > 0:
            logging.warning(f"Errors encountered in {errors_count:,} file(s). See log for details.")

        # --- Specific Whitelist Domain Check ---
        # Check for domains that are on the whitelist, were found, but are NOT currently targeted.
        logging.info("Performing specific whitelist domain check...")
        domains_found_counts = scan_results.get('overall_domain_counts', Counter())
        all_found_domains = set(domains_found_counts.keys())

        # Find domains that are BOTH in the whitelist AND were found in the scan
        found_whitelist_domains = whitelist_domains.intersection(all_found_domains)

        # From those, find the ones NOT currently being targeted for extraction
        missed_whitelist_targets = found_whitelist_domains - current_target_domains

        if not missed_whitelist_targets:
            logging.info("All found whitelist domains are already included in the target configuration.")
        else:
            logging.warning(f"Found {len(missed_whitelist_targets)} domains from the specific whitelist that were encountered but are NOT currently in the target list. Consider adding them:")
            # Sort for consistent output
            sorted_missed = sorted(list(missed_whitelist_targets))
            for domain in sorted_missed:
                logging.warning(f"  - Whitelist Miss: {domain} (Found Count: {domains_found_counts.get(domain, 0):,})")

    except Exception as e:
        logging.critical(f"An unhandled error occurred during the Stage 1 scan: {e}", exc_info=True)
    finally:
        logging.info("Stage 1 script finished.")


if __name__ == "__main__":
    # Example Usage:
    # uv run scripts/stage1_scan_warc.py --input /path/to/warc/dir --output-dir ./data/intermediate/stage1_parquet
    # uv run scripts/stage1_scan_warc.py --input file1.warc.gz file2.warc.gz --output-dir ./output_pq --limit-records 1000 --processes 4
    main() 