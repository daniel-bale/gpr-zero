"""Stage 1 integration checks using tiny WARC archives generated locally."""

import argparse
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from gpr_zero.workflows.ingest_warc import (
    OUTPUT_COLUMNS,
    load_domain_config,
    process_warc_file_stage1,
    scan_warc_files_parallel,
)

DOMAIN_CONFIG = [
    {"name": "Fixture Mandarin", "domains": ["example.tw"]},
    {"name": "Fixture English", "domains": ["english.example.tw"], "is_english": True},
]
HTML = "<html><body>臺灣新聞</body></html>".encode()


def write_warc(path: Path, records: list[dict]) -> Path:
    """Serialize real response records, without using production extraction code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as archive:
        for index, record in enumerate(records):
            body = record.get("body", HTML)
            http = (
                "HTTP/1.1 200 OK\r\n"
                f"Content-Type: {record.get('content_type', 'text/html; charset=utf-8')}\r\n"
                f"Content-Length: {len(body)}\r\n\r\n"
            ).encode() + body
            headers = [
                "WARC/1.0",
                f"WARC-Type: {record.get('record_type', 'response')}",
                f"WARC-Record-ID: <urn:uuid:00000000-0000-0000-0000-{index:012d}>",
                f"WARC-Date: {record.get('date', '2020-01-02T03:04:05Z')}",
                "Content-Type: application/http; msgtype=response",
                f"Content-Length: {len(http)}",
            ]
            if record.get("url") is not None:
                headers.append(f"WARC-Target-URI: {record['url']}")
            archive.write(
                "\r\n".join(headers).encode() + b"\r\n\r\n" + http + b"\r\n\r\n"
            )
    return path


def process(path, output, limit=None):
    return process_warc_file_stage1(
        (path, limit, *load_domain_config(DOMAIN_CONFIG), output)
    )


def test_extracts_html_and_metadata_to_parquet_and_filters_other_records(tmp_path):
    english_html = b"<html><body>English news</body></html>"
    path = write_warc(
        tmp_path / "fixture.warc.gz",
        [
            {"url": "https://www.example.tw/news/1"},
            {
                "url": "https://english.example.tw/news/2",
                "body": english_html,
                "content_type": "TEXT/HTML",
            },
            {"url": "https://outside.example/news/3"},
            {
                "url": "https://example.tw/data",
                "content_type": "application/json",
                "body": b"{}",
            },
            {"url": "https://example.tw/empty", "body": b""},
            {"url": "https://example.tw/invalid-date", "date": "not-a-date"},
            {"url": None},
            {"url": "https://example.tw/request", "record_type": "request"},
        ],
    )
    output = tmp_path / "parquet" / "2020" / "01"
    result = process(path, output)

    assert result["error"] is None
    assert result["warc_file"] == str(path)
    assert result["records_processed"] == 7
    assert result["target_domain_hits"] == 5
    assert result["html_records_found"] == 4
    assert result["target_html_extracted"] == 3
    assert result["html_bytes_extracted"] == 2 * len(HTML) + len(english_html)
    assert result["domain_counts"] == {
        "example.tw": 4,
        "english.example.tw": 1,
        "outside.example": 1,
    }
    assert result["parquet_file_path"] == str(output / "fixture.warc.parquet")

    actual = pl.read_parquet(result["parquet_file_path"])
    assert actual.columns == OUTPUT_COLUMNS
    assert actual["domain"].to_list() == [
        "example.tw",
        "english.example.tw",
        "example.tw",
    ]
    assert actual["outlet"].to_list() == [
        "Fixture Mandarin",
        "Fixture English",
        "Fixture Mandarin",
    ]
    assert actual["is_english"].to_list() == [False, True, False]
    assert actual["html_content_bytes"].to_list() == [HTML, english_html, HTML]
    assert actual["warc_date"].to_list() == [
        datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    ] * 2 + [None]
    assert actual["warc_id"].to_list() == [
        f"<urn:uuid:00000000-0000-0000-0000-{index:012d}>" for index in [0, 1, 5]
    ]


def test_no_matching_records_does_not_create_empty_parquet(tmp_path):
    path = write_warc(
        tmp_path / "unrelated.warc.gz", [{"url": "https://outside.example/article"}]
    )
    output = tmp_path / "parquet"
    result = process(path, output)

    assert result["error"] is None
    assert result["records_processed"] == 1
    assert result["domain_counts"] == {"outside.example": 1}
    assert result["target_html_extracted"] == 0
    assert result["parquet_file_path"] is None
    assert not output.exists()


def test_record_limit_caps_extracted_articles(tmp_path):
    path = write_warc(
        tmp_path / "limited.warc.gz",
        [{"url": f"https://example.tw/news/{index}"} for index in range(3)],
    )
    result = process(path, tmp_path / "parquet", limit=1)

    assert result["error"] is None
    assert result["target_html_extracted"] == 1
    actual = pl.read_parquet(result["parquet_file_path"])
    assert actual["url"].to_list() == ["https://example.tw/news/0"]


def test_missing_archive_reports_failure_without_output(tmp_path):
    result = process(tmp_path / "missing.warc.gz", tmp_path / "parquet")

    assert result["error"]
    assert result["records_processed"] == 0
    assert result["target_html_extracted"] == 0
    assert result["parquet_file_path"] is None


def test_parquet_write_failure_is_reported_without_success_path(tmp_path):
    path = write_warc(
        tmp_path / "fixture.warc.gz", [{"url": "https://example.tw/news/1"}]
    )
    output = tmp_path / "not_a_directory"
    output.write_text("existing file")
    result = process(path, output)

    assert result["target_html_extracted"] == 1
    assert result["error"].startswith("Parquet writing error:")
    assert result["parquet_file_path"] is None
    assert output.read_text() == "existing file"


def test_parallel_scan_writes_partitioned_parquet_and_reports_partial_failure(tmp_path):
    paths = [
        write_warc(
            tmp_path / "warc" / "2020" / "01" / "first.warc.gz",
            [{"url": "https://example.tw/news/1"}],
        ),
        write_warc(
            tmp_path / "warc" / "2021" / "02" / "second.warc.gz",
            [{"url": "https://english.example.tw/news/2"}],
        ),
        tmp_path / "warc" / "2021" / "02" / "missing.warc.gz",
    ]
    output = tmp_path / "parquet"
    logs = tmp_path / "logs"
    logs.mkdir()
    timestamp = "fixture"
    args = argparse.Namespace(
        input=list(map(str, paths)),
        pattern="*.warc.gz",
        limit_records=None,
        max_files=None,
        output_dir=str(output),
        processes=2,
    )

    result = scan_warc_files_parallel(
        warc_files=paths,
        output_dir=output,
        limit_records_per_file=None,
        num_processes=2,
        domain_config=DOMAIN_CONFIG,
        args_for_report=args,
        log_file_path_for_report=logs / "scan.log",
        timestamp_str_for_report=timestamp,
    )

    assert result["files_processed"] == 3
    assert result["total_records_processed"] == 2
    assert result["total_target_html_extracted"] == 2
    assert result["total_html_bytes_extracted"] == 2 * len(HTML)
    assert len(result["errors"]) == 1
    assert result["errors"][0]["file"] == str(paths[2])
    expected_paths = {
        output / "2020" / "01" / "first.warc.parquet",
        output / "2021" / "02" / "second.warc.parquet",
    }
    assert set(map(Path, result["parquet_files_created"])) == expected_paths
    actual = pl.concat([pl.read_parquet(path) for path in sorted(expected_paths)])
    assert actual.height == 2
    assert set(actual["url"]) == {
        "https://example.tw/news/1",
        "https://english.example.tw/news/2",
    }
    assert actual["html_content_bytes"].to_list() == [HTML, HTML]
    counts_path = logs / f"stage1_domain_counts_{timestamp}.json"
    assert json.loads(counts_path.read_text()) == {
        "example.tw": 1,
        "english.example.tw": 1,
    }
    report = (logs / f"stage1_scan_report_{timestamp}.md").read_text()
    assert "**Files Processed:** 3" in report
    assert "**Target HTML Records Extracted (to Parquet):** 2" in report
    assert "missing.warc.gz" in report
