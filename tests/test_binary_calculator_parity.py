"""Differential checks against the pinned thesis binary-matrix calculator.

The oracle must remain available in Git history. These checks intentionally fail
when it cannot be loaded, rather than comparing the implementation to itself.
"""

# The executed oracle is code from the pinned local thesis commit.
# ruff: noqa: S102

import json
import subprocess
import types
from pathlib import Path

import ibis
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gpr_zero.gpr_index.calculator import GPRCalculator

THESIS_COMMIT = "290b7c4c8eb900a712384deea0b68e2e47009624"
THESIS_PATH = "src/gpr_zero/gpr_index/calculator_binary_matrix.py"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def thesis_calculator_class():
    result = subprocess.run(
        ["git", "show", f"{THESIS_COMMIT}:{THESIS_PATH}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        pytest.fail(
            f"Required thesis oracle {THESIS_COMMIT}:{THESIS_PATH} is unavailable: "
            f"{result.stderr.strip()}",
            pytrace=False,
        )
    module = types.ModuleType("thesis_binary_calculator_oracle")
    exec(
        compile(result.stdout, f"{THESIS_COMMIT}:{THESIS_PATH}", "exec"),
        module.__dict__,
    )
    return module.GPRCalculatorBinaryMatrix


@pytest.fixture
def calculators(tmp_path, thesis_calculator_class):
    dictionaries = {
        "threats": {
            "word_sets": {
                "threat": ["戰爭", "WAR", "a+b"],
                "target": ["臺灣", "台灣"],
                "unused": [],
            },
            "categories": {
                "joint": "threat AND target",
                "plus": "threat + target",
                "proximity": "threat N/5 AND target",
                "never": "unused AND target",
                "blank": "",
                "total_reserved": "threat",
            },
        },
        "empty_word": {
            "word_sets": {"presence": [""], "specific": ["制裁"]},
            "categories": {"present": "presence", "sanctions": "specific"},
        },
    }
    paths = []
    for name, dictionary in dictionaries.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(dictionary, ensure_ascii=False), encoding="utf-8")
        paths.append(str(path))

    articles = pl.DataFrame(
        {
            "article_id": list(range(9)),
            "content": [
                "戰爭影響臺灣，戰爭消息",
                "WAR 台灣",
                "war 台灣",
                None,
                "一般新聞",
                "a+b 臺灣",
                "制裁",
                "戰爭",
                "",
            ],
            "outlet": ["A", "A", "A", "B", "B", "A", "B", "A", "B"],
            "source_warc_file": ["fixture.warc"] * 9,
            "warc_date": [
                "2019-12-29",
                "2019-12-30",
                "2020-01-01",
                "2020-01-05",
                "2020-01-06",
                "2020-01-31",
                "2020-02-01",
                "2020-12-31",
                "2021-01-01",
            ],
        }
    )
    connections = [ibis.duckdb.connect(), ibis.duckdb.connect()]
    current = GPRCalculator(paths)
    thesis = thesis_calculator_class(paths)
    tables = [conn.create_table("articles", articles) for conn in connections]
    try:
        yield current, thesis, tables
    finally:
        for conn in connections:
            conn.disconnect()


def prepare_pair(calculators, preparation):
    current, thesis, tables = calculators
    for calculator, table in zip((current, thesis), tables):
        calculator.prepare_binary_matrix(table, materialize=preparation != "fresh")
        if preparation == "cached":
            # A new instance must reconstruct the same vocabulary for the cache.
            replacement = type(calculator)(calculator.dictionary_paths)
            replacement.prepare_binary_matrix(table, materialize=True)
            if calculator is current:
                current = replacement
            else:
                thesis = replacement
    return current, thesis


@pytest.mark.parametrize("preparation", ["fresh", "materialized", "cached"])
@pytest.mark.parametrize(
    ("time_bucket", "group_by_cols"),
    [
        (None, ["outlet"]),
        ("W", []),
        ("week", ["outlet"]),
        ("M", ["outlet", "warc_date"]),
        ("month", []),
        ("Y", []),
        ("year", ["outlet"]),
    ],
)
@pytest.mark.parametrize("output_format", ["wide", "long"])
@pytest.mark.parametrize("dictionary_name", ["threats", "empty_word"])
def test_results_equal_thesis(
    calculators, preparation, time_bucket, group_by_cols, output_format, dictionary_name
):
    current, thesis = prepare_pair(calculators, preparation)
    arguments = {
        "dictionary_name": dictionary_name,
        "group_by_cols": group_by_cols,
        "time_bucket": time_bucket,
        "output_format": output_format,
    }
    expected = thesis.query(**arguments)
    actual = current.query(**arguments)
    # Verify exact values, column order, dtypes and thesis row ordering. No epsilon
    # tolerance may hide a changed share denominator or category matching rule.
    assert_frame_equal(actual, expected, check_exact=True)
    assert_frame_equal(
        current.labeled_articles_table.order_by("article_id").to_polars(),
        thesis.labeled_articles_table.order_by("article_id").to_polars(),
        check_exact=True,
    )


@pytest.mark.parametrize("output_format", ["wide", "long"])
def test_empty_article_results_equal_thesis(calculators, output_format):
    current, thesis, tables = calculators
    for calculator, table in zip((current, thesis), tables):
        calculator.prepare_binary_matrix(table.filter(table.article_id < 0))
    kwargs = {
        "dictionary_name": "threats",
        "group_by_cols": ["outlet"],
        "time_bucket": "W",
        "output_format": output_format,
    }
    assert_frame_equal(
        current.query(**kwargs), thesis.query(**kwargs), check_exact=True
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"dictionary_name": "missing", "group_by_cols": ["outlet"]},
        {"dictionary_name": "threats", "group_by_cols": []},
        {
            "dictionary_name": "threats",
            "group_by_cols": ["outlet"],
            "time_bucket": "W",
            "time_col": "missing",
        },
        {
            "dictionary_name": "threats",
            "group_by_cols": ["outlet"],
            "output_format": "unknown",
        },
    ],
)
def test_query_errors_equal_thesis(calculators, arguments):
    current, thesis = prepare_pair(calculators, "fresh")
    with pytest.raises(ValueError) as expected:
        thesis.query(**arguments)
    with pytest.raises(type(expected.value)) as actual:
        current.query(**arguments)
    assert str(actual.value) == str(expected.value)
