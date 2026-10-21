"""Test the shared matching seam against the pinned thesis implementations."""

# Execute only code read from the pinned local Git revision used as the oracle.
# ruff: noqa: S102

import ast
import json
import subprocess
from functools import reduce
from operator import and_, or_
from pathlib import Path

import ibis
import ibis.expr.types as ir
import pytest
from polars.testing import assert_frame_equal

from gpr_zero.constants import CATEGORY_MAPPING
from gpr_zero.gpr_index.dictionary import (
    THESIS_CATEGORY_ALIASES,
    DictionaryMatcher,
    normalize_categories,
)

THESIS = "290b7c4c8eb900a712384deea0b68e2e47009624"
ROOT = Path(__file__).resolve().parents[1]


def thesis_tree(path):
    result = subprocess.run(
        ["git", "show", f"{THESIS}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return ast.parse(result.stdout, filename=f"{THESIS}:{path}")


def thesis_functions(path, names):
    """Load only pure expression builders, avoiding script I/O and setup."""
    body = [
        node
        for node in thesis_tree(path).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in body} == set(names), (
        "The pinned oracle must contain every requested function"
    )
    namespace = {
        "ibis": ibis,
        "ir": ir,
        "reduce": reduce,
        "and_": and_,
        "or_": or_,
        "Dict": dict,
        "List": list,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def oracles():
    normalized = thesis_functions(
        "scripts/analyze_step2_spike_articles_normalized.py",
        ["build_word_set_expressions", "build_category_expressions"],
    )
    contribution = thesis_functions(
        "data_analysis/verify_step2_category_contributions_3outlets.py",
        ["build_category_expr"],
    )
    peak = thesis_functions(
        "scripts/analyze_category_peak_events.py",
        ["build_category_specific_expressions"],
    )
    # This script originally kept its expression construction inside main.
    # Execute its exact statements, rather than reimplementing an oracle here.
    tree = thesis_tree("scripts/compute_all_dictionary_timeseries_3outlets.py")
    initializer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "gpr_expr"
            for target in node.targets
        )
    )
    category_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "'categories' in dictionary"
    )
    code = compile(
        ast.Module(body=[initializer, category_loop], type_ignores=[]),
        "thesis:timeseries",
        "exec",
    )

    def timeseries(dictionary, table, columns):
        namespace = {
            "ibis": ibis,
            "dictionary": dictionary,
            "filtered_matrix": table,
            "word_column_mapping": columns,
        }
        exec(code, namespace)
        return namespace["gpr_expr"]

    return normalized, contribution, peak, timeseries


@pytest.fixture
def matrix():
    con = ibis.duckdb.connect()
    values = [True, False, None]
    rows = [
        (index, first, second)
        for index, (first, second) in enumerate((a, b) for a in values for b in values)
    ]
    table = con.create_table(
        "matrix",
        ibis.memtable(
            rows, schema={"id": "int64", "word_a": "boolean", "word_b": "boolean"}
        ),
    )
    yield table, {"a": "word_a", "b": "word_b", "unavailable": "missing_column"}
    con.disconnect()


@pytest.fixture
def dictionary():
    return {
        "word_sets": {
            "A": ["a", "a"],
            "B": ["b"],
            "empty": [],
            "unavailable": ["unavailable"],
            "unmapped": ["not_in_mapping"],
        },
        "categories": {
            "joint": "A AND B",
            "plus": "A + B",
            "proximity": "A N/2 AND B",
            "partial": "A AND missing",
            "empty": "A AND empty",
            "missing": "missing",
            "blank": "",
            "whitespace": "  ",
            "unavailable": "A AND unavailable",
            "unmapped": "A AND unmapped",
            "duplicate": "A AND B AND A",
        },
    }


def evaluated(table, expression):
    return table.select("id", expression.name("match")).order_by("id").to_polars()


def test_spike_matches_equal_thesis_for_missing_empty_null_and_legacy_syntax(
    oracles, matrix, dictionary
):
    table, columns = matrix
    normalized, _, _, _ = oracles
    word_sets = normalized["build_word_set_expressions"](
        dictionary["word_sets"], columns, table, "fixture"
    )
    expected = normalized["build_category_expressions"](
        dictionary["categories"], word_sets, "fixture"
    )
    actual = DictionaryMatcher(dictionary).spike_category_matches(table, columns)
    assert list(actual) == list(expected)
    for category in expected:
        assert_frame_equal(
            evaluated(table, actual[category]),
            evaluated(table, expected[category]),
            check_exact=True,
        )


def test_category_matches_equal_thesis_and_keep_nullable_results(
    oracles, matrix, dictionary
):
    table, columns = matrix
    _, contribution, _, _ = oracles
    matcher = DictionaryMatcher(dictionary)
    for category in [*dictionary["categories"], "absent"]:
        expected = contribution["build_category_expr"](
            dictionary, category, columns, table
        )
        actual = matcher.category_match(table, columns, category)
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            assert_frame_equal(
                evaluated(table, actual), evaluated(table, expected), check_exact=True
            )
    assert (
        evaluated(table, matcher.category_match(table, columns, "joint"))[
            "match"
        ].null_count()
        > 0
    )
    assert matcher.category_match(table, columns, "missing") is None


def test_peak_matches_equal_thesis_after_category_matching(oracles, matrix, dictionary):
    table, columns = matrix
    _, _, peak, _ = oracles
    for category in dictionary["categories"]:
        expected = peak["build_category_specific_expressions"](
            category, {"fixture": dictionary}, columns, table
        )
        actual = DictionaryMatcher(dictionary).category_match(table, columns, category)
        actual = actual.fill_null(False) if actual is not None else ibis.literal(False)
        assert_frame_equal(
            evaluated(table, actual), evaluated(table, expected), check_exact=True
        )


def test_timeseries_matches_equal_thesis_including_all_missing_categories(
    oracles, matrix, dictionary
):
    table, columns = matrix
    _, _, _, expected_match = oracles
    variants = [dictionary, {"word_sets": dictionary["word_sets"], "categories": {}}]
    variants.extend(
        {"word_sets": dictionary["word_sets"], "categories": {name: query}}
        for name, query in dictionary["categories"].items()
    )
    for variant in variants:
        actual = DictionaryMatcher(variant).timeseries_match(table, columns)
        expected = expected_match(variant, table, columns)
        assert_frame_equal(
            evaluated(table, actual), evaluated(table, expected), check_exact=True
        )
    missing = {"word_sets": {}, "categories": {"unknown": "missing"}}
    assert (
        evaluated(table, DictionaryMatcher(missing).timeseries_match(table, columns))[
            "match"
        ].to_list()
        == [True] * 9
    )


@pytest.mark.parametrize(
    "path,aliases",
    [
        ("scripts/analyze_step2_spike_articles_normalized.py", THESIS_CATEGORY_ALIASES),
        ("scripts/analyze_category_peak_events.py", THESIS_CATEGORY_ALIASES),
        ("scripts/analyze_category_peak_events_updated.py", CATEGORY_MAPPING),
    ],
)
def test_category_aliases_equal_their_thesis_workflow(path, aliases):
    assignment = next(
        node
        for node in thesis_tree(path).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CATEGORY_MAPPING"
            for target in node.targets
        )
    )
    assert aliases == ast.literal_eval(assignment.value)


def test_normalization_preserves_order_collision_behavior_and_input():
    dictionary = {
        "word_sets": {"A": ["a"]},
        "categories": {
            "軍事升級": "A",
            "other": "A",
            "Military Escalation": "B",
            "military_escalation": "C",
        },
    }
    actual = normalize_categories(dictionary)
    assert actual["categories"] == {
        "Military Escalation": "B",
        "other": "A",
        "military_escalation": "C",
    }
    assert list(actual["categories"]) == [
        "Military Escalation",
        "other",
        "military_escalation",
    ]
    assert normalize_categories(dictionary, THESIS_CATEGORY_ALIASES)["categories"] == {
        "Military Escalation": "C",
        "other": "A",
    }
    assert dictionary["categories"]["軍事升級"] == "A"


@pytest.mark.parametrize("query", ["missing", "A AND missing", "A and B", "AAND B"])
def test_binary_missing_references_preserve_thesis_errors(tmp_path, matrix, query):
    """The lenient analysis operations must never leak into binary labeling."""
    namespace = {"__name__": "thesis_binary_oracle"}
    exec(
        compile(
            thesis_tree("src/gpr_zero/gpr_index/calculator_binary_matrix.py"),
            "thesis:calculator_binary_matrix.py",
            "exec",
        ),
        namespace,
    )
    table, columns = matrix
    dictionary = {
        "word_sets": {"A": ["a"], "B": ["b"]},
        "categories": {"invalid": query},
    }
    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps(dictionary))
    thesis = namespace["GPRCalculatorBinaryMatrix"]([str(path)])
    thesis.labeled_articles_table = table
    thesis.word_column_mapping = columns
    with pytest.raises(Exception) as expected:
        thesis.query("dictionary", group_by_cols=["id"])
    with pytest.raises(type(expected.value)) as actual:
        DictionaryMatcher(dictionary).label_binary(table, columns)
    assert str(actual.value) == str(expected.value)
