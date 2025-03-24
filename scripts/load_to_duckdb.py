#!/usr/bin/env python3
"""
Load news articles from NDJSON files to DuckDB.
This script orchestrates the entire process:
1. Parse TXT news articles to NDJSON
2. Parse RTF news articles to NDJSON
3. Load NDJSON files to DuckDB
"""

import os
import sys
import glob
import subprocess
import logging
import tempfile
import argparse
import ibis
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/news/output/gpr_analysis.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def clean_processed_files():
    """Clean previously processed NDJSON files."""
    processed_dir = "data/news/processed"
    ndjson_files = glob.glob(os.path.join(processed_dir, "*.ndjson"))

    if ndjson_files:
        logger.info(
            f"Cleaning up {len(ndjson_files)} NDJSON files from {processed_dir}"
        )
        for file in ndjson_files:
            try:
                os.remove(file)
                logger.debug(f"Removed file: {file}")
            except Exception as e:
                logger.error(f"Failed to remove file {file}: {str(e)}")
    else:
        logger.info(f"No NDJSON files to clean in {processed_dir}")


def parse_txt_files():
    """Parse TXT news articles to NDJSON."""
    logger.info("Running TXT parser")
    try:
        result = subprocess.run(
            ["uv", "run", "scripts/parse_news_to_ndjson.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"TXT parser output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running TXT parser: {e}")
        logger.error(f"TXT parser stderr: {e.stderr}")
        return False


def parse_rtf_files():
    """Parse RTF news articles to NDJSON."""
    logger.info("Running RTF parser")
    try:
        result = subprocess.run(
            ["uv", "run", "scripts/parse_rtf_to_ndjson.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"RTF parser output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running RTF parser: {e}")
        logger.error(f"RTF parser stderr: {e.stderr}")
        return False


def count_processed_files():
    """Count the number of processed NDJSON files."""
    processed_dir = "data/news/processed"
    ndjson_files = glob.glob(os.path.join(processed_dir, "*.ndjson"))
    return len(ndjson_files)


def create_duckdb(truncate=False):
    """Create or connect to DuckDB database and load NDJSON files."""
    db_file = "data/news/news.duckdb"
    processed_dir = "data/news/processed"

    # Check if database exists and if truncate is requested
    if os.path.exists(db_file) and truncate:
        try:
            os.remove(db_file)
            logger.info(f"Truncated existing database: {db_file}")
        except Exception as e:
            logger.error(f"Failed to remove database file {db_file}: {str(e)}")

    # Connect to database using DuckDB directly for more control
    import duckdb

    # Connect to database
    logger.info(f"Connecting to DuckDB: {db_file}")
    conn = duckdb.connect(db_file)

    # Create articles table if it doesn't exist
    logger.info("Creating articles table if it doesn't exist")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id VARCHAR PRIMARY KEY,
            newspaper VARCHAR,
            date DATE,
            headline VARCHAR,
            content VARCHAR,
            category VARCHAR,
            page VARCHAR,
            author VARCHAR,
            source_file VARCHAR
        );
    """)

    # Get NDJSON files
    ndjson_files = glob.glob(os.path.join(processed_dir, "*.ndjson"))
    if not ndjson_files:
        logger.warning(f"No NDJSON files found in {processed_dir}")
        # Close connection and connect with ibis for the views
        conn.close()
        ibis_conn = ibis.connect(f"duckdb://{db_file}")
        create_views(ibis_conn)
        return 0

    logger.info(f"Found {len(ndjson_files)} NDJSON files to load")

    # Load each NDJSON file
    total_processed = 0
    duplicates = 0

    for ndjson_file in ndjson_files:
        logger.info(f"Loading {ndjson_file}")

        # Create a temporary table for this file
        temp_table = f"temp_{os.path.basename(ndjson_file).replace('.', '_')}"
        logger.info(f"Creating temporary table: {temp_table}")

        try:
            # Clean any existing temp table
            conn.execute(f"DROP TABLE IF EXISTS {temp_table}")

            # Create the temp table from the NDJSON file
            conn.execute(f"""
                CREATE TABLE {temp_table} AS
                SELECT * FROM read_ndjson_auto('{ndjson_file}');
            """)

            # Get count of records in the temp table
            result = conn.execute(f"SELECT COUNT(*) FROM {temp_table}")
            temp_count = result.fetchone()[0]
            logger.info(f"Loaded {temp_count} articles from {ndjson_file}")

            # Check for duplicates
            result = conn.execute(f"""
                SELECT COUNT(*) FROM {temp_table} t 
                WHERE EXISTS (SELECT 1 FROM articles a WHERE a.id = t.id)
            """)
            dup_count = result.fetchone()[0]
            duplicates += dup_count

            if dup_count > 0:
                logger.warning(f"Found {dup_count} duplicate articles in {ndjson_file}")

            # Insert non-duplicate records
            conn.execute(f"""
                INSERT INTO articles 
                SELECT * FROM {temp_table} t 
                WHERE NOT EXISTS (SELECT 1 FROM articles a WHERE a.id = t.id)
            """)

            # Drop the temporary table
            conn.execute(f"DROP TABLE IF EXISTS {temp_table}")

            total_processed += temp_count

        except Exception as e:
            logger.error(f"Error processing {ndjson_file}: {e}")
            # Continue with next file

    # Get total articles count
    result = conn.execute("SELECT COUNT(*) FROM articles")
    total_articles = result.fetchone()[0]
    logger.info(f"Total articles in database: {total_articles}")
    logger.info(f"Total processed: {total_processed}, Duplicates skipped: {duplicates}")

    # Close DuckDB connection
    conn.close()

    # Connect with ibis for the views
    ibis_conn = ibis.connect(f"duckdb://{db_file}")
    create_views(ibis_conn)

    return total_articles


def create_views(conn):
    """Create or refresh views for analysis."""
    logger.info("Creating/refreshing views")

    # Articles by date
    conn.raw_sql("""
        CREATE OR REPLACE VIEW articles_by_date AS
        SELECT date, COUNT(*) as article_count
        FROM articles
        GROUP BY date
        ORDER BY date;
    """)

    # Articles by newspaper
    conn.raw_sql("""
        CREATE OR REPLACE VIEW articles_by_newspaper AS
        SELECT newspaper, COUNT(*) as article_count
        FROM articles
        GROUP BY newspaper
        ORDER BY article_count DESC;
    """)

    # Articles by category
    conn.raw_sql("""
        CREATE OR REPLACE VIEW articles_by_category AS
        SELECT category, COUNT(*) as article_count
        FROM articles
        GROUP BY category
        ORDER BY article_count DESC;
    """)

    logger.info("Views created successfully")


def main():
    """Main function to orchestrate the process."""
    parser = argparse.ArgumentParser(
        description="Process news articles and load to DuckDB"
    )
    parser.add_argument(
        "--clean-files",
        action="store_true",
        help="Clean processed NDJSON files before starting",
    )
    parser.add_argument(
        "--truncate", action="store_true", help="Truncate existing database"
    )
    parser.add_argument(
        "--skip-txt", action="store_true", help="Skip processing TXT files"
    )
    parser.add_argument(
        "--skip-rtf", action="store_true", help="Skip processing RTF files"
    )

    args = parser.parse_args()

    # Step 1: Clean processed files if requested
    if args.clean_files:
        clean_processed_files()

    # Step 2: Parse TXT files unless skipped
    if not args.skip_txt:
        parse_txt_files()
    else:
        logger.info("Skipping TXT parsing as requested")

    # Step 3: Parse RTF files unless skipped
    if not args.skip_rtf:
        parse_rtf_files()
    else:
        logger.info("Skipping RTF parsing as requested")

    # Count processed files
    processed_count = count_processed_files()
    logger.info(f"Total processed NDJSON files: {processed_count}")

    # Step 4: Create DuckDB and load data
    total_articles = create_duckdb(truncate=args.truncate)

    logger.info("Process completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
