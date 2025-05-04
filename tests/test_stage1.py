#!/usr/bin/env python3
import pytest
import duckdb
from pathlib import Path
import logging

# Assuming conftest.py handles path adjustments
from scripts.stage1_scan_warc import (
    process_warc_file_stage1,
    scan_warc_files_parallel,
    setup_intermediate_database,
    load_domain_config,
    TAIWANESE_DOMAINS_CONFIG, # Using the main config for now
    INTERMEDIATE_DB_TABLE
)

# --- Fixtures ---

@pytest.fixture(scope="module")
def domain_config_data():
    """Provides the domain configuration list."""
    # Using the config defined in the script for simplicity
    # Could define a smaller, test-specific config here if needed
    return TAIWANESE_DOMAINS_CONFIG

@pytest.fixture(scope="module")
def loaded_domains(domain_config_data):
    """Provides the loaded domain lookup structures."""
    return load_domain_config(domain_config_data)

@pytest.fixture
def intermediate_db_path(tmp_path: Path) -> Path:
    """Provides a path for a temporary intermediate database."""
    return tmp_path / "test_intermediate.duckdb"

@pytest.fixture(scope="session") # Use session scope if path is constant
def sample_warc_dir() -> Path:
    """Provides the path to the directory containing sample WARC files.

    IMPORTANT: The test execution environment MUST have access to this path
    and it should contain sample .warc.gz files for testing.
    The path provided by the user is used here.
    """
    path = Path("/home5/chuenho/data/crawl-data/CC-NEWS/2022/02/")
    # Optional: Add a check here if running locally where you can access it
    # if not path.is_dir():
    #     pytest.skip(f"Sample WARC directory not found or accessible: {path}")
    return path

@pytest.fixture
def sample_warc_files(sample_warc_dir: Path) -> list[Path]:
    """Lists the first ~32 sample WARC files from the directory."""
    if not sample_warc_dir.is_dir():
        pytest.skip(f"Sample WARC directory not accessible: {sample_warc_dir}")
    
    files = sorted(list(sample_warc_dir.glob("*.warc.gz"))) # Assuming .warc.gz
    if not files:
        pytest.skip(f"No *.warc.gz files found in sample directory: {sample_warc_dir}")
    
    return files[:32] # Take up to 32 files as requested


# --- Integration Tests ---

def test_process_warc_file_stage1(sample_warc_files: list[Path], loaded_domains: tuple):
    """Tests processing a single sample WARC file."""
    if not sample_warc_files:
        pytest.skip("No sample WARC files available for single file test.")
        
    sample_file = sample_warc_files[0] # Test with the first file found
    limit_records = 100 # Process a small number of records for speed
    domain_to_outlet, all_target_domains = loaded_domains
    
    args_tuple = (sample_file, limit_records, domain_to_outlet, all_target_domains)
    
    result = process_warc_file_stage1(args_tuple)
    
    assert result["error"] is None, f"Processing {sample_file.name} failed: {result['error']}"
    assert result["warc_file"] == str(sample_file)
    assert result["records_processed"] > 0
    # Add more specific assertions based on expected content of sample_file[0]
    # e.g., assert result["target_html_extracted"] > 0
    # e.g., assert len(result["articles_to_insert"]) == result["target_html_extracted"]
    # e.g., check structure of items in result["articles_to_insert"]
    logging.info(f"Single file test result for {sample_file.name}: Processed {result['records_processed']} records, found {result['target_html_extracted']} target HTML.")
    # Further assertions would require knowing the expected content of the sample file

def test_scan_warc_files_parallel(sample_warc_files: list[Path], intermediate_db_path: Path, domain_config_data: list[dict], caplog):
    """Tests the parallel scanning function with sample files."""
    if len(sample_warc_files) < 2: # Need at least 2 files to test parallelism reasonably
         pytest.skip("Not enough sample WARC files available for parallel test (>1 required).")

    limit_records_per_file = 50 # Keep it small for testing
    truncate_db = True
    num_processes = 2 # Use a small fixed number for testing
    
    # Mock args and log path for report generation
    class MockArgs:
        def __init__(self):
            self.input = [str(f) for f in sample_warc_files]
            self.pattern = "*.warc.gz"
            self.limit_records = limit_records_per_file
            self.max_files = None
            self.intermediate_db = str(intermediate_db_path)
            self.truncate = truncate_db
            self.processes = num_processes
            # Add other args if needed by report function
    mock_args = MockArgs()
    mock_log_path = Path(intermediate_db_path.parent / "mock_test_log.log")

    caplog.set_level(logging.INFO)

    results = scan_warc_files_parallel(
        warc_files=sample_warc_files,
        intermediate_db_path=intermediate_db_path,
        limit_records_per_file=limit_records_per_file,
        truncate_db=truncate_db,
        num_processes=num_processes,
        domain_config=domain_config_data,
        args_for_report=mock_args, 
        log_file_path_for_report=mock_log_path
    )

    assert results is not None, "scan_warc_files_parallel returned None, indicating setup failure."
    assert results["files_processed"] == len(sample_warc_files)
    assert not results.get("errors"), f"Errors occurred during parallel scan: {results.get('errors')}"
    
    # Check if the intermediate database was created and populated
    assert intermediate_db_path.exists(), "Intermediate database file was not created."
    conn = None
    try:
        conn = duckdb.connect(str(intermediate_db_path), read_only=True)
        count_query = conn.execute(f"SELECT COUNT(*) FROM {INTERMEDIATE_DB_TABLE}").fetchone()
        assert count_query is not None
        db_rows = count_query[0]
        logging.info(f"Parallel scan inserted {db_rows} rows into intermediate DB.")
        # We expect *some* rows to be inserted, but exact number depends on sample data
        assert db_rows == results["total_articles_inserted"]
        assert db_rows >= 0 # Should be at least 0, likely more if samples have target HTML
        
        # Optional: Query a sample row to check schema/content
        if db_rows > 0:
            sample_row = conn.execute(f"SELECT warc_id, warc_date, url, domain, outlet, length(html_content_bytes) FROM {INTERMEDIATE_DB_TABLE} LIMIT 1").fetchone()
            assert sample_row is not None
            assert isinstance(sample_row[0], str) # warc_id
            # warc_date might be None if parsing failed
            assert isinstance(sample_row[2], str) # url
            assert isinstance(sample_row[3], str) # domain
            # outlet might be None
            assert isinstance(sample_row[5], int) # length(html_content_bytes)
            assert sample_row[5] > 0 # Bytes should have length > 0

    finally:
        if conn:
            conn.close()
    
    # Check if report file was mentioned in logs (doesn't check content)
    report_log_found = any("Generating Stage 1 analysis report" in rec.message for rec in caplog.records)
    assert report_log_found, "Report generation was not logged." 