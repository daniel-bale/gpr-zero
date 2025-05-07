# End-to-End (E2E) Testing Plan for WARC Processing Pipeline

## Goal

To implement an automated end-to-end health check test for the WARC processing pipeline (Stage 1 and Stage 2). This test will ensure the pipeline can run successfully on a small, variable subset of data and that its output adheres to basic quality and schema standards.

## Step 1: Prerequisite - Verify Current Pipeline Changes (Manual)

*(This step assumes successful manual verification of the latest changes to `scripts/stage1_scan_warc.py` and `scripts/stage2_parse_html_load_db.py`)*

-   **Stage 1 (`stage1_scan_warc.py`) Verification:**
    -   Confirm Parquet output filenames are simplified (e.g., `ORIGINAL_WARC_BASENAME.parquet`), with UUID removed.
    -   Confirm Parquet files *do not* contain `source_warc_file` or `stage1_processed_at` as data columns.
-   **Stage 2 (`stage2_parse_html_load_db.py`) Verification:**
    -   Confirm the final DuckDB `articles` table schema does *not* include `stage1_processed_at`.
    -   Confirm the `source_warc_file` column in the `articles` table is correctly populated by deriving it from the input Parquet filename (e.g., as `ORIGINAL_WARC_BASENAME.warc.gz`).
    -   Note: If `polars.scan_parquet(..., include_file_paths=True)` adds the column as `_file_path` instead of `file_path`, ensure `row.get('_file_path')` is used in Stage 2.

## Step 2: Refactor `compare_dbs.py` for Testability

**Objective:** Modify the existing `compare_dbs.py` script so its core database interaction and comparison logic can be imported and used as utility functions within `pytest` test scripts. The current command-line interface of `compare_dbs.py` should be preserved.

**Key Functions to Implement/Expose:**

1.  `connect_db(db_path_str: str) -> duckdb.DuckDBPyConnection | None:`
    *   Handles connecting to a DuckDB database.
    *   Returns a connection object or `None` on failure.
2.  `get_table_as_polars(conn: duckdb.DuckDBPyConnection, table_name: str, columns: list[str] | None = None, limit: int | None = None, filter_conditions: str | None = None) -> pl.DataFrame:`
    *   Fetches data from a specified table into a Polars DataFrame.
    *   Allows selecting specific columns, limiting rows, and applying filter conditions.
3.  `compare_dataframes_for_diff(df1: pl.DataFrame, df2: pl.DataFrame, join_cols: list[str], compare_cols: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:`
    *   (Optional, if needed for more detailed E2E diffing later, not critical for health check)
    *   Compares two DataFrames based on join columns.
    *   Identifies rows present only in `df1`, only in `df2`, and rows with differing values in `compare_cols`.
    *   Returns three DataFrames: `diff_df1_only`, `diff_df2_only`, `diff_rows_updated_values`.

**Implementation Notes:**
-   Move the core logic for database connection, querying, and data retrieval into these functions.
-   The `if __name__ == "__main__":` block in `compare_dbs.py` will call these functions to retain its current CLI behavior.
-   Ensure functions are well-documented with type hints.

## Step 3: Develop E2E Health Check Test (`tests/e2e/test_pipeline_health.py`)

**Objective:** Create a `pytest` test that runs the full Stage 1 and Stage 2 pipeline on a randomly selected small subset of input WARC files and performs basic sanity checks on the output.

**Strategy:** Random Small Subset for Health Check.

**Test Script Components (`tests/e2e/test_pipeline_health.py`):**

1.  **Test Setup (`pytest` fixtures if applicable):**
    *   Define paths to data directories (e.g., where Feb 2022 WARC files are stored).
    *   Define temporary output directories for Stage 1 Parquet files and the Stage 2 test database.
2.  **Input Selection:**
    *   List all available WARC files from the source directory (e.g., Feb 2022 data).
    *   Randomly select a small number of files (e.g., 2-3) for the current test run. Ensure this selection is logged for reproducibility of a specific test failure.
3.  **Pipeline Execution:**
    *   Programmatically execute `scripts/stage1_scan_warc.py` using `subprocess.run` (or by importing its `main` function if refactored for that).
        *   Pass the randomly selected WARC files as input.
        *   Direct output to the temporary Parquet directory.
    *   Programmatically execute `scripts/stage2_parse_html_load_db.py` using `subprocess.run` (or by importing its `main` function).
        *   Use the temporary Parquet directory as input.
        *   Output to a temporary DuckDB file for this test run.
4.  **Assertions (using `pytest` assert and utility functions from refactored `compare_dbs.py`):**
    *   **Pipeline Success:**
        *   Assert that both Stage 1 and Stage 2 scripts complete with a success exit code (e.g., 0).
    *   **Output Database Created:**
        *   Assert that the temporary output DuckDB file was created.
    *   **Data Generation (connect using `connect_db` from utilities):**
        *   Fetch data from the `articles` table using `get_table_as_polars`.
        *   Assert that the table is not empty (assuming the input WARCs are expected to produce some target articles). The number of articles can be logged but not strictly asserted unless input is fixed.
    *   **Key Fields Populated & Plausible:**
        *   For all rows in the output DataFrame (or a sample):
            *   Assert `warc_id`, `url`, `domain`, `title` are not NULL and not empty strings.
            *   Assert `content` is not NULL (it can be an empty string if a page truly had no extractable content, but should exist).
            *   Assert `warc_date` is a valid datetime/timestamp type (Polars will type it).
            *   Assert `is_english` is boolean.
            *   Assert `source_warc_file` is not NULL, not empty, and its value correctly corresponds to one of the input WARC files (e.g., ends with `.warc.gz` or `.warc.lz4` and the base name matches one of the inputs).
            *   Assert `stage2_processed_at` is a valid timestamp.
    *   **Schema Conformance (Optional but good):**
        *   Retrieve the schema of the `articles` table from the output DB (e.g., `PRAGMA table_info('articles');`).
        *   Assert that column names and types match the expected schema (defined in `FINAL_SCHEMA` in Stage 2, minus `stage1_processed_at`).
5.  **Cleanup (`pytest` teardown or `finally` blocks):**
    *   Ensure the temporary Stage 1 Parquet directory and its contents are deleted.
    *   Ensure the temporary Stage 2 output DuckDB file is deleted, regardless of test success or failure.

**Implementation Notes:**
-   Use `pytest` conventions.
-   Consider making the number of random files configurable.
-   Log the chosen random files for each run to help debug if a particular combination causes issues.
-   This test focuses on ensuring the pipeline *runs* and produces *structurally sound* output, not on bit-for-bit content identity with a fixed baseline. 