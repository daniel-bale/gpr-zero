"""Differential checks against the pinned thesis article calculator."""

import json
import subprocess
import types
from pathlib import Path

import ibis
import pytest
from polars.testing import assert_frame_equal

from gpr_zero.gpr_index.article_calculator import GPRCalculator


THESIS_REVISION = "290b7c4c8eb900a712384deea0b68e2e47009624"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def thesis_calculator_class():
    """Use the historical implementation as the oracle, without copying it."""
    source = subprocess.run(
        [
            "git",
            "show",
            f"{THESIS_REVISION}:src/gpr_zero/gpr_index/calculator.py",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType("thesis_article_calculator")
    exec(compile(source, "thesis:calculator.py", "exec"), module.__dict__)
    return module.GPRCalculator


@pytest.fixture
def calculator_pair(tmp_path, thesis_calculator_class):
    dictionary = {
        "word_sets": {
            "danger": ["Risk", "a+b", "戰爭"],
            "action": ["Mobilize", "部署"],
            "empty": ["", ""],
            "no_words": [],
        },
        "categories": {
            "risk": "danger",
            "joint": "danger AND action",
            "proximity": "danger N/2 + action",
            "invalid": "missing",
            "blank": " ",
            "empty": "empty",
            "no_words": "no_words",
        },
    }
    path = tmp_path / "article-dictionary.json"
    path.write_text(json.dumps(dictionary), encoding="utf-8")
    return GPRCalculator(str(path)), thesis_calculator_class(str(path))


@pytest.fixture
def articles():
    """Small adversarial corpus covering regex, RTF, nulls, and time grouping."""
    rows = [
        (0, "risk", "web", "", "2020-01-05", "A"),
        (1, "RISK", "web", "", "2020-01-06", "A"),
        (2, "", "web", "risk", "2020-01-06", "A"),
        (3, None, "web", "risk", "2020-01-07", "A"),
        (4, "a+b", "web", "", "2020-01-07", "B"),
        (5, "ab", "web", "", "2020-01-08", "B"),
        (6, None, "LT-export.rtf", "risk", "2020-01-08", "B"),
        (7, "Risk Mobilize", "LT-export.rtf", "joint", "2020-01-09", "B"),
        (8, "Risk", "LT-export.rtf", "unknown", "2020-01-10", "A"),
        (9, "Risk", "lt-export.rtf", "joint", "2020-01-11", "A"),
        (10, "戰爭部署", "web", "", "2020-02-02", "A"),
        (11, "Risk", None, "", "2020-02-03", "B"),
        (12, "Risk", "LT-export.rtf", "invalid", "2020-02-03", "B"),
        (13, "Risk", "LT-export.rtf", None, "2020-02-03", "B"),
        (14, "", "LT-export.rtf", "empty", "2021-01-04", "A"),
    ]
    con = ibis.duckdb.connect()
    schema = {
        "article_id": "int64",
        "content": "string",
        "source_file": "string",
        "category": "string",
        "date": "string",
        "outlet": "string",
    }
    table = con.create_table("articles", ibis.memtable(rows, schema=schema))
    yield table
    con.disconnect()


def test_prepared_labels_preserve_case_rtf_and_null_semantics(calculator_pair, articles):
    current, thesis = calculator_pair
    actual = current.prepare(articles).order_by("article_id").to_polars()
    expected = thesis.prepare(articles).order_by("article_id").to_polars()
    assert_frame_equal(actual, expected, check_exact=True)

    # Independent expectations ensure the corpus exercises the intended differences
    # from binary substring matching and honors RTF assignments over article text.
    matches = {
        "risk": {0, 1, 4, 6, 9, 10, 11},
        "joint": {7, 10},
        "proximity": {10},
        "empty": {14},
        "invalid": set(),
        "blank": set(),
        "no_words": set(),
    }
    for category, article_ids in matches.items():
        assert actual[f"is_{category}"].to_list() == [
            article_id in article_ids for article_id in range(actual.height)
        ]
    overall_ids = set().union(*matches.values())
    assert actual["matches_any_category"].to_list() == [
        article_id in overall_ids for article_id in range(actual.height)
    ]
    assert actual["has_empty"].to_list() == [False] * actual.height
    assert actual["has_no_words"].to_list() == [False] * actual.height


@pytest.mark.parametrize("calculate_share", [False, True], ids=["counts", "shares"])
@pytest.mark.parametrize(
    ("group_by_cols", "time_bucket"),
    [
        ([], None),
        (["outlet"], None),
        (["date", "outlet"], None),
        (["date", "outlet"], "W"),
        (["outlet"], "week"),
        (["date"], "M"),
        (["date"], "month"),
        (["date", "outlet"], "Y"),
    ],
)
def test_aggregations_match_thesis_exactly(
    calculator_pair, articles, group_by_cols, time_bucket, calculate_share
):
    current, thesis = calculator_pair
    current.prepare(articles)
    thesis.prepare(articles)
    query_args = {
        "group_by_cols": group_by_cols,
        "time_bucket": time_bucket,
        "calculate_share": calculate_share,
    }
    assert_frame_equal(
        current.query(**query_args), thesis.query(**query_args), check_exact=True
    )


def test_overlapping_category_counts_and_epsilon_shares(calculator_pair, articles):
    current, _ = calculator_pair
    current.prepare(articles)
    counts = current.query(group_by_cols=[]).row(0, named=True)
    assert counts["overall_index"] == 9
    assert counts["risk_count"] == 7
    assert counts["joint_count"] == 2
    assert counts["proximity_count"] == 1
    assert counts["empty_count"] == 1

    shares = current.query(group_by_cols=[], calculate_share=True).row(0, named=True)
    assert shares["total_articles_in_group"] == 15
    assert shares["overall_share"] == 9 / (15 + 1e-9)
    assert shares["risk_share"] == 7 / (15 + 1e-9)


def test_materialized_labels_can_be_reused_by_thesis(calculator_pair, articles):
    current, thesis = calculator_pair
    current.prepare(articles, materialize=True)
    # The historical dictionary-derived table name is part of the cache contract.
    assert current._get_labeled_table_name() == thesis._get_labeled_table_name()
    # A cache hit must ignore different incoming article rows, just as thesis did.
    thesis.prepare(articles.filter(articles.article_id == 5), materialize=True)
    assert_frame_equal(
        current.query(group_by_cols=["outlet"], calculate_share=True),
        thesis.query(group_by_cols=["outlet"], calculate_share=True),
        check_exact=True,
    )


def test_query_before_prepare_preserves_error(calculator_pair):
    for calculator in calculator_pair:
        with pytest.raises(RuntimeError, match="prepare.*must be called"):
            calculator.query(group_by_cols=[])
