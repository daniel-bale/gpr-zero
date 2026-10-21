"""Ingestion uses the active project and preserves the two-stage data contract."""

from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from gpr_zero.project import Project, use_project
from gpr_zero.workflows import ingest_html, ingest_warc


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return Project(
        root, root / "corpus", root / "database" / "articles.duckdb", root / "llm.yaml"
    )


def test_ingestion_arguments_use_one_project_for_destinations(
    project, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    with use_project(project):
        warc = ingest_warc.parse_arguments(
            ["--input", "crawl", "--processes", "2", "--limit-records", "5"]
        )
        html = ingest_html.parse_arguments(["--years", "2020", "2021"])
    intermediate = project.data_path("intermediate", "stage1_parquet")
    assert Path(warc.output_dir) == intermediate
    assert Path(html.input_dir) == intermediate
    assert Path(html.final_db) == project.database
    assert Path(warc.log_dir) == project.root / "logs" / "warc_extract" / "stage1"
    assert Path(html.log_dir) == project.root / "logs" / "warc_extract" / "stage2"
    assert warc.processes == 2
    assert warc.limit_records == 5
    assert html.years == [2020, 2021]
    assert not intermediate.exists()
    assert not project.database.exists()


@pytest.mark.parametrize(
    "module,arguments",
    [
        (ingest_warc, ["--input", "crawl", "--output-dir", "elsewhere"]),
        (ingest_warc, ["--input", "crawl", "--log-dir", "elsewhere"]),
        (ingest_html, ["--final-db", "elsewhere.duckdb"]),
    ],
)
def test_ingestion_rejects_competing_destination_flags(project, module, arguments):
    with use_project(project), pytest.raises(SystemExit) as error:
        module.parse_arguments(arguments)
    assert error.value.code == 2


def test_html_ingestion_loads_selected_year_into_project_database(project):
    for year in [2019, 2020]:
        output = project.data_path("intermediate", "stage1_parquet", str(year), "01")
        output.mkdir(parents=True)
        pl.DataFrame(
            {
                "warc_id": [f"record-{year}"],
                "warc_date": [datetime(year, 1, 2, 3, 4, 5)],
                "url": [f"https://example.tw/{year}"],
                "domain": ["example.tw"],
                "outlet": ["Fixture"],
                "is_english": [False],
                "html_content_bytes": [
                    '<html><head><title>臺灣新聞</title><meta property="article:published_time" content="2020-01-02"></head><body><article>區域 安全 新聞</article></body></html>'.encode()
                ],
            }
        ).write_parquet(output / f"fixture-{year}.warc.parquet")

    with use_project(project):
        ingest_html.main(["--years", "2020"])
    with duckdb.connect(str(project.database), read_only=True) as connection:
        rows = connection.execute(
            "SELECT warc_id, title, content, extracted_publish_date, source_warc_file FROM articles"
        ).fetchall()
    assert rows == [
        (
            "record-2020",
            "臺灣新聞",
            "區域 安全 新聞",
            "2020-01-02",
            "fixture-2020.warc.warc.gz",
        )
    ]

    # The explicit truncate option keeps its original replace-table behavior.
    with use_project(project):
        ingest_html.main(["--years", "2019", "--truncate"])
    with duckdb.connect(str(project.database), read_only=True) as connection:
        assert connection.execute("SELECT warc_id FROM articles").fetchall() == [
            ("record-2019",)
        ]
