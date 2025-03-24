# Data Parsing and Loading Process for GPR Analysis

## Overview
This document summarizes the data processing pipeline for news articles used in the GPR analysis.

## Components
1. **TXT Parser** (parse_news_to_ndjson.py)
   - Processes TXT format news articles from multiple newspapers
   - Extracts metadata including ID, newspaper, date, category, headline, and content
   - Outputs NDJSON files with standardized format

2. **RTF Parser** (parse_rtf_to_ndjson.py)
   - Processes RTF format news articles (particularly for Liberty Times)
   - Handles different RTF formatting patterns
   - Uses striprtf library with fallback conversion methods
   - Outputs NDJSON files with same format as TXT parser

3. **Data Loader** (load_to_duckdb.py)
   - Orchestrates the entire process
   - Cleans old processed files when needed
   - Runs both parsers in sequence
   - Loads all NDJSON files into DuckDB
   - Creates analysis views

## Process Flow
1. Clean previous processed files (optional)
2. Parse TXT news articles to NDJSON
3. Parse RTF news articles to NDJSON
4. Create/connect to DuckDB database
5. Load each NDJSON file with deduplication
6. Create analysis views

## Database Schema
- **Main Table**: articles
  - Primary key: id (article ID)
  - Fields: newspaper, date, headline, content, category, page, author, source_file

- **Views**:
  - articles_by_date: Article counts by date
  - articles_by_newspaper: Article counts by newspaper
  - articles_by_category: Article counts by category

## Current Status
- Successfully processes TXT files from multiple newspapers (聯合報, 中國時報, 工商時報)
- RTF files for Liberty Times are parsed but have table name issues due to hyphens in filenames
- Total of 5,886 articles loaded with 967 duplicates identified and skipped

## Usage
```bash
# Full process with cleaning and database recreation
uv run scripts/load_to_duckdb.py --clean-files --truncate

# Skip specific parts
uv run scripts/load_to_duckdb.py --skip-txt  # Skip TXT processing
uv run scripts/load_to_duckdb.py --skip-rtf  # Skip RTF processing
```

## Prerequisites
Before running the pipeline, ensure the following dependencies are installed:
- striprtf: For parsing RTF files (`uv add striprtf`)
- duckdb: For database operations (installed as part of base environment)
- polars: For data manipulation (installed as part of base environment)
- ibis: For database interface (installed as part of base environment) 