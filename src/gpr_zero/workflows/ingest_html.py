import argparse
import logging
import re  # Make sure re is imported
import time
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl
from selectolax.lexbor import LexborHTMLParser
from tqdm import tqdm

# Placeholder for shared utils - assumes it will be created
# from warc_utils import clean_text, setup_database # etc.

# --- Constants ---
FINAL_TABLE_NAME = "articles"
# INTERMEDIATE_TABLE_NAME = "raw_html" # No longer needed
INSERT_BATCH_SIZE = 1000  # Tune as needed

# Final DB Schema
# Ensure column names match Stage 1 where applicable and add new ones
FINAL_SCHEMA = """
    warc_id TEXT PRIMARY KEY,
    warc_date TIMESTAMP,
    url TEXT,
    domain TEXT,
    outlet TEXT,
    is_english BOOLEAN,
    title TEXT,
    content TEXT,
    extracted_publish_date TEXT, -- Store as text for flexibility
    source_warc_file TEXT, -- To be derived from Parquet filename
    stage2_processed_at TIMESTAMP DEFAULT current_timestamp
"""

# --- Selectors (Default Only) ---
# TODO: Move to a config file (e.g., config.yaml)
# Using only default selectors for initial testing.
# Domain-specific overrides can be added later based on results.
# UPDATED: Added enhanced selectors based on Phase 2 critical analysis (2025-06-01)
# UPDATED: Added Phase 5 remaining problem fixes (2025-06-01)
DOMAIN_SELECTORS = {
    "default": {  # Keep defaults as a fallback, slightly expanded
        "title": [
            "title",
            "h1",
            'meta[property="og:title"]',
            'meta[name="twitter:title"]',
        ],
        "content": [
            "article",
            "main",
            ".article-content",
            ".story-content",
            ".post-content",
            ".entry-content",
        ],
        "date": [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            'meta[name="Date"]',
            'meta[name="publishdate"]',
            'meta[name="sailthru.date"]',
            'meta[property="rnews:datePublished"]',
            'meta[itemprop="datePublished"]',
            "time[datetime][pubdate]",
            "time[datetime]",
        ],
    },
    "peoplenews.tw": {  # People News (民報) - CRITICAL FIX for 0.01% extraction rate
        "title": ['meta[property="og:title"]', "h1", "title"],
        "content": ["#main"],  # Phase 2 analysis: 100% coverage, avg 1784 chars
        "date": [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            "time[datetime]",
        ],
    },
    "cna.com.tw": {  # Central News Agency (中央社) - MAJOR FIX for 93.14% extraction rate
        "title": ['meta[property="og:title"]', "h1", "title"],
        "content": [
            '[class*="content"]',
            '[class*="news"]',
            'div[class*="text"]',
        ],  # Phase 2: top 3 selectors
        "date": [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            "time[datetime]",
        ],
    },
    "nownews.com": {  # Now News (今日新聞) - PHASE 5 FIX for 0% extraction in 2020
        "title": ['meta[property="og:title"]', "h1", "title"],
        "content": [
            ".entry-content",  # Common WordPress/CMS content selector
            ".post-content",  # Alternative post content selector
            ".article-content",  # Generic article content selector
            "#content",  # Simple content ID selector
            ".content",  # Basic content class selector
            "article",  # HTML5 article element
            "main",  # HTML5 main element
        ],
        "date": [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            "time[datetime]",
        ],
    },
    "taiwannews.com.tw": {  # Taiwan News - PHASE 5 FIX for 91% extraction in 2021-06
        "title": ['meta[property="og:title"]', "h1", "title"],
        "content": [
            ".post-content",  # Common post content selector
            ".entry-content",  # WordPress entry content selector
            ".article-body",  # Article body selector
            ".content-area",  # Content area selector
            ".article-content",  # Generic article content
            "article",  # HTML5 article element
            "main",  # HTML5 main element
        ],
        "date": [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            "time[datetime]",
        ],
    },
    "ltn.com.tw": {  # Liberty Times
        "title": ['meta[property="og:title"]', "h1.article-title", "h1"],
        "content": [
            "div.text",
            "div.articlebody",
            "div.news_content",
        ],  # div.text is often good for ltn
        "date": [
            'meta[property="article:published_time"]',
            "span.time",
            'meta[name="pubdate"]',
        ],
    },
    "news.ltn.com.tw": {  # Liberty Times (news subdomain) - ENHANCED based on Phase 2 analysis
        "title": ['meta[property="og:title"]', "h1.article-title", "h1"],
        "content": [
            ".content",
            '[class*="content"]',
            "div.text",
            "div.articlebody",
            "div.news_content",
        ],  # Added Phase 2 recommendations
        "date": [
            'meta[property="article:published_time"]',
            "span.time",
            'meta[name="pubdate"]',
        ],
    },
    "udn.com": {  # United Daily News - ENHANCED for Phase 5 (2018 extraction issues)
        "title": ["#story_art_title", 'meta[property="og:title"]', "h1"],
        # Enhanced with additional selectors for historical periods (2018)
        "content": [
            "#story_body_content",  # Modern UDN selector
            ".story__content",  # Alternative story content
            ".article-content",  # Generic article content
            "#story_body",  # Historical story body ID
            ".story-body",  # Historical story body class
            ".content-article",  # Article content class
            "article",  # HTML5 article element
            ".post-content",  # Post content fallback
            "main",  # HTML5 main element
        ],
        "date": [
            'meta[name="date"]',
            ".story_bady_info_author span:first-child",
            ".shareBar__info--author span:first-child",
        ],
    },
    "chinatimes.com": {  # China Times - Updated for 2016+ compatibility
        "title": [".article-title", 'meta[property="og:title"]', "h1", "title"],
        "content": [
            # Modern selectors (for newer articles)
            ".article-body",
            "article .post-content",
            "div.article-content-wrapper",
            # 2016 era selectors - content appears as paragraphs after article_info
            "article",
            "main",
            # Fallback: all paragraphs (will be filtered for substantial content)
            "p",
        ],
        "date": [
            'meta[property="article:published_time"]',
            "time.meta-data",
            'meta[name="pubdate"]',
            "time[datetime]",
        ],
    },
    "taipeitimes.com": {  # Taipei Times - Fixed based on HTML structure investigation
        "title": ['meta[property="og:title"]', "title", "h1"],
        "content": [
            # Primary content container identified in investigation
            "div.text",
            ".text",
            # Fallback selectors
            "article",
            "main",
            "p",
        ],
        "date": [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            "time[datetime]",
        ],
    },
}


# --- Argument Parsing ---
def parse_arguments(argv=None):
    from gpr_zero.project import get_project

    project = get_project()
    parser = argparse.ArgumentParser(
        prog="gpr ingest html",
        description="Stage 2: Parse raw HTML from intermediate Parquet files, extract content, load into final DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing intermediate Parquet files (output of Stage 1).",
        default=str(project.data_path("intermediate", "stage1_parquet")),
    )
    parser.add_argument(
        "--truncate",
        help="Truncate (delete and recreate) the final articles table before starting.",
        action="store_true",
    )
    parser.add_argument(
        "--processes",
        help="Number of parallel processes for parsing (default: 1, single-threaded).",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--years",
        help="Specific years to process (e.g., --years 2016 2017 2023). If not specified, processes all years.",
        nargs="+",
        type=int,
    )
    parser.set_defaults(
        final_db=str(project.database),
        log_dir=str(project.root / "logs" / "warc_extract" / "stage2"),
    )

    return parser.parse_args(argv)


# --- Logging Setup ---
def setup_logging(log_dir: str | Path, timestamp_str: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_name = f"stage2_parse_load_{timestamp_str}.log"
    log_file_path = log_dir / log_file_name

    log_format = "%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Logging initialized. Log file: {log_file_path}")
    return log_file_path


# --- Database Setup (adapted for Final DB) ---
def setup_final_database(
    db_path_str: str | Path, truncate: bool = False
) -> duckdb.DuckDBPyConnection:
    """Setup the final DuckDB database."""
    db_path = Path(db_path_str)
    db_dir = db_path.parent
    db_dir.mkdir(parents=True, exist_ok=True)

    operation = "Connecting to" if db_path.exists() else "Creating"
    logging.info(f"{operation} final database: {db_path}")

    try:
        conn = duckdb.connect(str(db_path))
        if truncate:
            logging.warning(
                f"Truncating (dropping) existing table '{FINAL_TABLE_NAME}' in final DB."
            )
            conn.execute(f"DROP TABLE IF EXISTS {FINAL_TABLE_NAME};")

        # Create final table if it doesn't exist
        conn.execute(f"CREATE TABLE IF NOT EXISTS {FINAL_TABLE_NAME} ({FINAL_SCHEMA});")
        logging.info(f"Final database table '{FINAL_TABLE_NAME}' setup complete.")
        return conn
    except duckdb.Error as e:
        logging.critical(f"Failed to connect to or setup final database {db_path}: {e}")
        raise


# --- Text Cleaning Utility ---
# Moved here from below extract_article_content_selectolax
def clean_text(text: str | None) -> str | None:
    """Clean whitespace from text."""
    if text is None:
        return None
    # Replace multiple whitespace chars (including newlines, tabs) with a single space
    # Use regex for robustness
    try:
        cleaned = re.sub(r"\s+", " ", str(text))  # Ensure input is string
        return cleaned.strip()
    except TypeError:
        logging.warning(f"clean_text received non-string input: {type(text)}")
        return None  # Or return original if non-string handling is needed


# --- HTML Content Extraction ---
def extract_article_content_selectolax(
    html_content_bytes: bytes | None, url: str, domain: str
) -> dict[str, str | None]:
    """Extracts title, content, and date using Selectolax based on domain-specific rules."""
    extracted = {"title": None, "content": None, "extracted_publish_date": None}
    # Return early if html_content_bytes is None
    if html_content_bytes is None:
        logging.debug(f"Received None for html_content_bytes for {url}, cannot parse.")
        return extracted

    # Choose selectors: specific domain or default
    selectors_config = DOMAIN_SELECTORS.get(domain, DOMAIN_SELECTORS["default"])
    title_selectors = selectors_config.get(
        "title", DOMAIN_SELECTORS["default"]["title"]
    )
    content_selectors = selectors_config.get(
        "content", DOMAIN_SELECTORS["default"]["content"]
    )
    date_selectors = selectors_config.get("date", DOMAIN_SELECTORS["default"]["date"])

    try:
        # Detect encoding automatically and parse using the constructor
        tree = LexborHTMLParser(
            html_content_bytes
        )  # Correct: Pass bytes directly to constructor

        if tree.root is None:
            logging.debug(f"Selectolax could not find root node for {url}")
            # Return empty dict if root node is missing
            return extracted

        root = tree.root

        # --- Extract Title ---
        extracted_title = None
        for selector in title_selectors:
            title_node = root.css_first(selector)
            if title_node:
                raw_title = title_node.text(strip=True)
                cleaned_title = clean_text(raw_title)
                if cleaned_title:
                    extracted_title = cleaned_title
                    break  # Stop after finding the first non-empty title
        extracted["title"] = extracted_title

        # --- Extract Content ---
        extracted_content = None
        for selector in content_selectors:
            content_node = root.css_first(selector)
            if content_node:
                # --- Add cleaning for specific domains (e.g., UDN) ---
                # Make a copy or select again if decompose modifies original node significantly
                temp_content_node = root.css_first(selector)
                if temp_content_node:
                    if domain == "udn.com":
                        # Remove script and style tags before extracting text for UDN
                        for unwanted_tag in temp_content_node.css(
                            "script, style, .article-footer, .article-tags, .social-share, .related-news"
                        ):
                            unwanted_tag.decompose()  # Removes the tag and its content
                        logging.debug(
                            f"Removed script/style/footer elements for UDN URL: {url}"
                        )

                    # Extract text after potential cleaning
                    raw_content = " ".join(
                        temp_content_node.text(separator=" ", strip=True).split()
                    )
                else:
                    raw_content = None  # Should not happen if content_node was found, but for safety

                if raw_content:
                    # Apply a length limit to avoid excessively large content fields
                    extracted_content = raw_content[:20000]  # Limit length
                    break  # Stop after finding the first non-empty content

            # Special handling for China Times when using 'p' selector
            elif selector == "p" and domain == "chinatimes.com":
                # For China Times 2016, collect all substantial paragraphs
                all_paragraphs = root.css(selector)
                if all_paragraphs:
                    paragraph_texts = []
                    for p_node in all_paragraphs:
                        p_text = p_node.text(strip=True)
                        # Filter out short paragraphs (likely navigation/ads)
                        # and paragraphs that are likely not article content
                        if (
                            p_text
                            and len(p_text) > 30
                            and not any(
                                skip_word in p_text.lower()
                                for skip_word in [
                                    "廣告",
                                    "推薦",
                                    "按讚",
                                    "分享",
                                    "留言",
                                    "facebook",
                                    "twitter",
                                    "weibo",
                                    "line",
                                    "點閱",
                                    "更多",
                                    "javascript",
                                    "comscore",
                                ]
                            )
                        ):
                            paragraph_texts.append(p_text)

                    if paragraph_texts:
                        # Combine paragraphs with proper spacing
                        combined_content = " ".join(paragraph_texts)
                        if len(combined_content) > 100:  # Ensure substantial content
                            extracted_content = combined_content[
                                :20000
                            ]  # Apply length limit
                            break

        extracted["content"] = extracted_content

        # --- Extract Publish Date ---
        extracted_publish_date = None
        for selector in date_selectors:
            date_node = root.css_first(selector)
            if date_node:
                date_str = None
                if date_node.tag == "meta":
                    date_str = date_node.attributes.get("content")
                elif date_node.tag == "time":
                    date_str = date_node.attributes.get("datetime")
                else:
                    date_str = date_node.text(strip=True)

                if date_str:
                    # Basic cleaning, more robust parsing might be needed later
                    cleaned_date = date_str.strip()
                    if cleaned_date:
                        extracted_publish_date = cleaned_date
                        break  # Stop after finding the first date
        extracted["extracted_publish_date"] = extracted_publish_date

    except Exception as e:
        logging.warning(f"Error parsing HTML for {domain} ({url}): {e}", exc_info=False)

    # Basic check for empty results before returning
    if not extracted["title"] and not extracted["content"]:
        logging.debug(f"Could not extract title or content for {url}. Skipping.")
        # Return the dictionary even if empty, main loop checks content

    return extracted


# --- Main Processing Logic ---
def process_intermediate_data(
    input_dir: str | Path,
    final_db_conn: duckdb.DuckDBPyConnection,
    num_processes: int,
    final_db_path: str | Path,  # Added for logging
    years: list[int] | None = None,  # Added for year filtering
):
    """Reads intermediate Parquet files, parses HTML, inserts into final DB."""
    input_dir_path = Path(input_dir)
    logging.info(
        f"Starting Stage 2 processing from intermediate Parquet directory: {input_dir_path}"
    )

    if years:
        logging.info(f"Year filtering enabled: {years}")

    if not input_dir_path.is_dir():
        logging.error(
            f"Input directory not found or is not a directory: {input_dir_path}. Cannot proceed."
        )
        return

    start_time = time.time()
    total_processed = 0
    total_inserted = 0
    total_errors = 0

    try:
        # --- Removed Intermediate DB Connection ---

        # Use Polars scan_parquet for lazy loading with year filtering
        logging.info(f"Scanning Parquet files in {input_dir_path}...")
        try:
            # Build parquet file pattern based on years
            if years:
                # Create patterns for specific years: 2016/**/*.parquet, 2017/**/*.parquet, etc.
                year_patterns = [
                    f"{input_dir_path}/{year}/**/*.parquet" for year in years
                ]
                logging.info(f"Scanning parquet files for years: {years}")
                logging.info(f"Patterns: {year_patterns}")
                # MODIFIED: Scan multiple patterns for specific years
                lazy_df = pl.scan_parquet(year_patterns, include_file_paths="file_path")
            else:
                # Original behavior: scan all parquet files
                lazy_df = pl.scan_parquet(
                    f"{input_dir_path}/**/*.parquet", include_file_paths="file_path"
                )

            total_rows_estimate = lazy_df.select(pl.len()).collect().item()
            logging.info(
                f"Estimated {total_rows_estimate:,} total rows across Parquet files."
            )
        except Exception as e:
            logging.error(
                f"Error scanning Parquet files in {input_dir_path}: {e}", exc_info=True
            )
            logging.error(
                "Ensure the directory contains valid Parquet files and polars[pyarrow] is installed."
            )
            return

        if total_rows_estimate == 0:
            logging.warning(
                "Found Parquet files, but they appear to contain no data. No data to process for Stage 2."
            )
            return

        final_data_batch = []

        # --- Single-threaded processing (for now) ---
        # TODO: Implement multiprocessing for parsing if needed
        if num_processes > 1:
            logging.warning(
                f"Multiprocessing (processes={num_processes}) not yet implemented for Stage 2 parsing. Running single-threaded."
            )

        logging.info("Starting HTML parsing and content extraction...")
        # Collect the LazyFrame into an eager DataFrame to iterate
        # For very large datasets, consider processing in chunks: lazy_df.collect(streaming=True)
        df_eager = lazy_df.collect()

        # Iterate through the eager DataFrame
        progress_bar = tqdm(
            df_eager.iter_rows(named=True),
            total=total_rows_estimate,
            desc="Stage 2 Parsing HTML",
        )

        for row in progress_bar:
            total_processed += 1
            warc_id = row.get("warc_id")
            html_content_bytes = row.get("html_content_bytes")
            url = row.get("url")
            domain = row.get("domain")
            parquet_file_path_str = row.get(
                "file_path"
            )  # MODIFIED: Get the Parquet file path

            # MODIFIED: Derive source_warc_file from parquet_file_path_str
            derived_source_warc_file = None
            if parquet_file_path_str:
                parquet_filename = Path(
                    parquet_file_path_str
                ).name  # e.g., some_warc_basename.parquet
                # Assuming original was .warc.gz. This might need to be more robust if other extensions like .warc.lz4 were used.
                derived_source_warc_file = (
                    parquet_filename.removesuffix(".parquet") + ".warc.gz"
                )
            else:
                logging.warning(
                    f"Missing 'file_path' for row {total_processed}, cannot derive source_warc_file."
                )

            # Basic check for essential fields
            if not all([warc_id, url, domain]):
                logging.warning(
                    f"Skipping row {total_processed} due to missing warc_id, url, or domain."
                )
                total_errors += 1
                continue

            # Handle potential null bytes from Parquet
            if html_content_bytes is None:
                logging.debug(
                    f"Skipping record {warc_id} ({url}) due to missing HTML content in Parquet."
                )
                # Decide if this counts as an error or just skipped
                # total_errors += 1
                continue

            # Extract content using the function (now accepts None)
            extracted_content_dict = extract_article_content_selectolax(
                html_content_bytes, url, domain
            )

            # Add to batch if content was extracted
            if extracted_content_dict.get("title") or extracted_content_dict.get(
                "content"
            ):
                # Safely access other fields from the row dictionary
                final_data_batch.append(
                    (
                        warc_id,
                        row.get("warc_date"),
                        url,
                        domain,
                        row.get("outlet"),
                        row.get("is_english"),
                        extracted_content_dict["title"],
                        extracted_content_dict["content"],
                        extracted_content_dict["extracted_publish_date"],
                        derived_source_warc_file,  # MODIFIED: Use derived value
                    )
                )
            else:
                total_errors += 1  # Count as error if extraction failed

            # Insert batch into final DB
            if len(final_data_batch) >= INSERT_BATCH_SIZE:
                try:
                    logging.info(
                        f"Inserting batch of {len(final_data_batch)} processed articles into final DB..."
                    )
                    # MODIFIED: Column list and placeholders updated (10 columns)
                    final_db_conn.executemany(
                        f"INSERT INTO {FINAL_TABLE_NAME} (warc_id, warc_date, url, domain, outlet, is_english, title, content, extracted_publish_date, source_warc_file) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        final_data_batch,
                    )
                    final_db_conn.commit()
                    total_inserted += len(final_data_batch)
                    final_data_batch = []  # Clear batch
                except duckdb.Error as e:
                    logging.error(f"Final DB batch insert error: {e}", exc_info=True)
                    final_db_conn.rollback()
                    total_errors += len(final_data_batch)  # Count batch as errors
                    final_data_batch = []  # Clear batch
                except Exception as e:
                    logging.error(
                        f"Unexpected error during final batch insert: {e}",
                        exc_info=True,
                    )
                    final_db_conn.rollback()
                    total_errors += len(final_data_batch)
                    final_data_batch = []

        # Insert any remaining items
        if final_data_batch:
            try:
                logging.info(
                    f"Inserting final batch of {len(final_data_batch)} processed articles..."
                )
                # MODIFIED: Column list and placeholders updated (10 columns)
                final_db_conn.executemany(
                    f"INSERT INTO {FINAL_TABLE_NAME} (warc_id, warc_date, url, domain, outlet, is_english, title, content, extracted_publish_date, source_warc_file) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    final_data_batch,
                )
                final_db_conn.commit()
                total_inserted += len(final_data_batch)
            except duckdb.Error as e:
                logging.error(f"Final DB final batch insert error: {e}", exc_info=True)
                final_db_conn.rollback()
                total_errors += len(final_data_batch)
            except Exception as e:
                logging.error(
                    f"Unexpected error during final batch insert: {e}", exc_info=True
                )
                final_db_conn.rollback()
                total_errors += len(final_data_batch)

    except Exception as e:
        logging.critical(
            f"Unhandled error during Stage 2 processing: {e}", exc_info=True
        )
    finally:
        # No intermediate connection to close
        pass

    end_time = time.time()
    duration = end_time - start_time
    logging.info("--- Stage 2 Summary ---")
    logging.info(f"  Total Records Scanned from Parquet Files: {total_processed:,}")
    logging.info(f"  Successfully Parsed & Inserted into Final DB: {total_inserted:,}")
    logging.info(f"  Final Database Path: {final_db_path}")
    logging.info(
        f"  Records with Missing Data/Extraction/Insert Errors: {total_errors:,}"
    )
    logging.info(f"  Total Processing Time: {duration:.2f} s")
    # TODO: Add reporting


# --- Main Function ---
def main(argv=None):
    args = parse_arguments(argv)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(args.log_dir, run_timestamp)
    logging.info(f"Stage 2 Arguments: {vars(args)}")

    final_db_conn = None
    try:
        # Ensure input directory exists before proceeding
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            logging.critical(
                f"Input directory does not exist or is not a directory: {input_dir}. Exiting."
            )
            return

        final_db_conn = setup_final_database(args.final_db, truncate=args.truncate)
        # Pass input_dir instead of intermediate_db
        process_intermediate_data(
            input_dir, final_db_conn, args.processes, Path(args.final_db), args.years
        )
    except Exception as e:
        logging.critical(
            f"An unhandled error occurred during the Stage 2 process: {e}",
            exc_info=True,
        )
    finally:
        if final_db_conn:
            try:
                final_db_conn.close()
                logging.info("Final database connection closed.")
            except Exception as e:
                logging.error(f"Error closing final DB connection: {e}")
        logging.info("Stage 2 script finished.")
