"""Analysis adapters retain the pinned thesis expressions and output records."""

import ast
import importlib
import json
import subprocess
import types
from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gpr_zero.constants import STANDARD_CATEGORIES
from gpr_zero.project import Project, use_project

THESIS_REVISION = "290b7c4c8eb900a712384deea0b68e2e47009624"
ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    "analyze_step2_spike_articles_normalized",
    "analyze_category_peak_events",
    "analyze_category_peak_events_updated",
    "calculate_spike_word_lift_scores_v3",
)
CATEGORY = "Military Escalation"


@pytest.fixture(scope="module")
def workflows():
    """Load historical functions without their import-time filesystem writes."""
    pairs = {}
    for name in WORKFLOWS:
        source = subprocess.run(
            ["git", "show", f"{THESIS_REVISION}:scripts/{name}.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tree = ast.parse(source)
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (
                isinstance(node, ast.Assign)
                and all(
                    isinstance(target, ast.Name)
                    and (target.id.isupper() or target.id == "logger")
                    for target in node.targets
                )
            )
        ]
        historical = types.ModuleType(f"thesis_{name}")
        historical.__file__ = str(ROOT / "scripts" / f"{name}.py")
        exec(compile(tree, f"thesis:{name}.py", "exec"), historical.__dict__)
        pairs[name] = (
            importlib.import_module(f"gpr_zero.workflows.{name}"),
            historical,
        )
    return pairs


@pytest.fixture
def project(tmp_path):
    project = Project(
        tmp_path,
        tmp_path / "data",
        tmp_path / "articles.duckdb",
        tmp_path / "llm_config.yaml",
    )
    project.step2_dictionaries_dir.mkdir(parents=True)
    project.dictionaries_dir.mkdir(parents=True)
    definitions = {
        "step2_first": {
            "word_sets": {
                "risk": ["risk"],
                "action": ["action"],
                "empty": [],
                "unmapped": ["zulu"],
            },
            "categories": {
                "military_escalation": "risk AND action",
                "Military Escalation (軍事升級)": "risk AND empty",
                "軍事升級": "risk AND missing",
                "Diplomatic Isolation": "unmapped",
                "Unknown category": "missing",
                "Economic Coercion": "risk + action",
            },
        },
        "step2_second": {
            "word_sets": {"risk": ["risk"], "action": ["action"], "absent": ["zzz"]},
            "categories": {CATEGORY: "risk AND absent", "Regional Security": "action"},
        },
        "step2_meta": {"notes": "Metadata is included by the original workflows."},
    }
    for category in STANDARD_CATEGORIES:
        definitions["step2_second"]["categories"].setdefault(
            category, "risk AND action"
        )
    for name, definition in definitions.items():
        content = json.dumps(definition, ensure_ascii=False)
        (project.step2_dictionaries_dir / f"{name}.json").write_text(content)
        (project.dictionaries_dir / f"{name}.json").write_text(content)
    (project.dictionaries_dir / "step1_extra.json").write_text(
        json.dumps(
            {
                "word_sets": {"r": ["aaa", "risk"]},
                "categories": {"Legacy attribution": "risk"},
            }
        )
    )
    return project


@pytest.fixture
def dictionary_inputs():
    dictionaries = {
        "first": {
            "word_sets": {
                "risk": ["risk"],
                "action": ["action"],
                "empty": [],
                "absent": ["absent"],
            },
            "categories": {
                CATEGORY: "risk AND empty",
                "Diplomatic Isolation": "risk AND missing",
                "Economic Coercion": "action",
                "Regional Security": "missing",
                "Gray Zone Operations": "risk AND absent",
                "Unknown category": "risk + action",
            },
        },
        "second": {
            "word_sets": {"risk": ["risk"], "action": ["action"]},
            "categories": {
                CATEGORY: "action",
                "Diplomatic Isolation": "risk AND action",
            },
        },
        "metadata": {"notes": "no matches"},
    }
    for category in STANDARD_CATEGORIES:
        dictionaries["second"]["categories"].setdefault(category, "risk AND action")
    mapping = {
        "action": "contains_word_0000",
        "risk": "contains_word_0001",
        "absent": "contains_word_0002",
    }
    return dictionaries, mapping


@pytest.fixture
def connection_factory():
    connections = []

    def make():
        connection = ibis.duckdb.connect(threads=1)
        connections.append(connection)
        rows = [
            # End of year and the complete final week are intentional fixtures.
            (
                "before",
                "2019-12-29",
                "2019-12-29",
                "China Times (中國時報)",
                True,
                True,
            ),
            (
                "monday",
                "2019-12-30",
                "2019-12-30",
                "China Times (中國時報)",
                True,
                True,
            ),
            (
                "tuesday",
                "2019-12-31",
                "2019-12-31",
                "Liberty Times (自由時報)",
                False,
                True,
            ),
            (
                "sunday",
                "2020-01-05",
                "2020-01-05",
                "United Daily (聯合報)",
                True,
                False,
            ),
            ("next", "2020-01-06", "2020-01-06", "China Times (中國時報)", True, True),
            ("null", "2020-01-02", "2020-01-02", "China Times (中國時報)", None, None),
            ("other", "2020-01-02", "2020-01-02", "Other outlet", True, True),
            (
                "different_date",
                "2020-01-07",
                "2020-01-03",
                "China Times (中國時報)",
                True,
                True,
            ),
            (
                "variant_outlet",
                "2020-01-04",
                "2020-01-04",
                "China Times Archive",
                True,
                True,
            ),
        ]
        frame = pd.DataFrame(
            rows,
            columns=[
                "warc_id",
                "extracted_publish_date",
                "warc_date",
                "outlet",
                "contains_word_0000",
                "contains_word_0001",
            ],
        )
        frame["url"] = "https://example.test/" + frame.warc_id
        frame["title"] = "Title " + frame.warc_id
        frame["content"] = "Content " + frame.warc_id
        schema = {
            column: "boolean" if column.startswith("contains_word") else "string"
            for column in frame.columns
        }
        connection.create_table(
            "gpr_binary_matrix_all_words", ibis.memtable(frame, schema=schema)
        )
        connection.create_table(
            "articles", ibis.memtable(frame[["warc_id", "content"]])
        )
        return connection

    yield make
    for connection in connections:
        connection.disconnect()


@pytest.mark.parametrize("name", WORKFLOWS[:3])
def test_dictionary_loading_preserves_each_thesis_vocabulary_and_aliases(
    workflows, project, name
):
    current, thesis = workflows[name]
    thesis.__file__ = str(project.root / "scripts" / f"{name}.py")
    with use_project(project):
        actual = current.load_step2_dictionaries_and_word_mapping()
    expected = thesis.load_step2_dictionaries_and_word_mapping()
    assert actual == expected
    assert actual[1]["action"] == (
        "contains_word_0001" if name.endswith("updated") else "contains_word_0000"
    )
    assert ("step2_meta" in actual[0]) == (not name.endswith("updated"))
    categories = actual[0]["step2_first"]["categories"]
    assert ("military_escalation" in categories) == name.endswith("updated")


def test_word_metadata_retains_thesis_substring_attribution(workflows, project):
    current, thesis = workflows[WORKFLOWS[3]]
    thesis.__file__ = str(project.root / "scripts" / f"{WORKFLOWS[3]}.py")
    with use_project(project):
        actual = current.load_all_dictionaries_and_create_mapping()
    assert actual == thesis.load_all_dictionaries_and_create_mapping()
    # The legacy short set name 'r' matches query 'risk'.
    assert "Legacy attribution" in actual[1]["aaa"]


def test_spike_expression_weights_and_distribution_match_thesis(
    workflows, dictionary_inputs, connection_factory
):
    current, thesis = workflows[WORKFLOWS[0]]
    dictionaries, mapping = dictionary_inputs
    table = connection_factory().table("gpr_binary_matrix_all_words")
    outputs = []
    for implementation in (current, thesis):
        dictionary_matches, category_matches = implementation.build_all_expressions(
            dictionaries, mapping, table
        )
        weights = implementation.calculate_article_weights(table, dictionary_matches)
        category_weights = implementation.calculate_category_weights(
            table, category_matches
        )
        records = (
            table.mutate(weight=weights, **category_weights)
            .order_by("warc_id")
            .to_pandas()
        )
        distribution = implementation.calculate_category_distribution(
            table, category_matches, weights
        )
        outputs.append((records, distribution))
    assert_frame_equal(outputs[0][0], outputs[1][0], check_exact=True)
    assert outputs[0][1] == outputs[1][1]
    # Empty sets remain false for normalized spikes rather than being ignored.
    assert outputs[0][0].set_index("warc_id").loc["tuesday", CATEGORY] == 0


@pytest.mark.parametrize("name", WORKFLOWS[1:3])
@pytest.mark.parametrize("sample_size", [100, 2])
def test_peak_article_records_and_seeded_weighted_samples_match_thesis(
    workflows,
    dictionary_inputs,
    connection_factory,
    monkeypatch,
    name,
    sample_size,
):
    current, thesis = workflows[name]
    dictionaries, mapping = dictionary_inputs
    connection = connection_factory()
    frames = []
    for implementation in (current, thesis):
        monkeypatch.setattr(implementation, "ARTICLES_PER_PEAK", sample_size)
        np.random.seed(67)
        frames.append(
            implementation.sample_category_peak_articles(
                connection,
                CATEGORY,
                pd.Timestamp("2019-12-30"),
                dictionaries,
                mapping,
            )
            .sort_values("warc_id")
            .reset_index(drop=True)
        )
    assert_frame_equal(frames[0], frames[1], check_exact=True)
    assert frames[0].to_dict("records") == frames[1].to_dict("records")
    assert "before" not in set(frames[0].warc_id)
    assert "next" not in set(frames[0].warc_id)
    if sample_size == 100:
        expected = {"monday", "tuesday", "sunday", "variant_outlet"}
        if name.endswith("updated"):
            expected.add("different_date")
        assert set(frames[0].warc_id) == expected
        assert frames[0].set_index("warc_id").loc["tuesday", "category_weight"] == 1


def test_peak_category_expression_matches_thesis(
    workflows, dictionary_inputs, connection_factory
):
    current, thesis = workflows[WORKFLOWS[1]]
    dictionaries, mapping = dictionary_inputs
    table = connection_factory().table("gpr_binary_matrix_all_words")
    frames = [
        table.mutate(
            matches=implementation.build_category_specific_expressions(
                CATEGORY,
                dictionaries,
                mapping,
                table,
            )
        )
        .order_by("warc_id")
        .to_pandas()
        for implementation in (current, thesis)
    ]
    assert_frame_equal(frames[0], frames[1], check_exact=True)


def test_word_lift_records_and_week_coverage_match_thesis(
    workflows, dictionary_inputs, connection_factory
):
    current, thesis = workflows[WORKFLOWS[3]]
    _, mapping = dictionary_inputs
    connection = connection_factory()
    events = pd.DataFrame(
        [
            {
                "start_date": pd.Timestamp("2019-12-30"),
                "end_date": pd.Timestamp("2019-12-30"),
                "peak_date": pd.Timestamp("2019-12-30"),
            }
        ]
    )
    outputs = []
    for implementation in (current, thesis):
        baseline, total = implementation.calculate_baseline_word_frequencies(
            connection, mapping
        )
        words, spike_total = implementation.calculate_spike_word_frequencies(
            connection, events.iloc[0], mapping
        )
        records = implementation.analyze_spikes(
            connection, events, baseline, mapping, "Step2"
        )
        outputs.append((baseline, total, words, spike_total, records))
    for position in (0, 2, 4):
        assert_frame_equal(outputs[0][position], outputs[1][position], check_exact=True)
    assert outputs[0][1] == outputs[1][1] == 7
    assert outputs[0][3] == outputs[1][3] == 5
    assert outputs[0][4].to_dict("records") == outputs[1][4].to_dict("records")


def test_empty_spike_preserves_empty_output(
    workflows, dictionary_inputs, connection_factory
):
    current, thesis = workflows[WORKFLOWS[3]]
    _, mapping = dictionary_inputs
    event = {
        "start_date": pd.Timestamp("2024-01-01"),
        "end_date": pd.Timestamp("2024-01-01"),
    }
    connection = connection_factory()
    for implementation in (current, thesis):
        frame, count = implementation.calculate_spike_word_frequencies(
            connection, event, mapping
        )
        assert frame.empty and count == 0


def test_normalized_spike_json_records_match_thesis(
    workflows, project, connection_factory, monkeypatch
):
    current, thesis = workflows[WORKFLOWS[0]]
    thesis.__file__ = str(project.root / "scripts" / f"{WORKFLOWS[0]}.py")
    spike_directory = project.data_path("gpr_spikes", "spike_comparison")
    spike_directory.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "start_date": "2019-12-30",
                "end_date": "2019-12-30",
                "peak_date": "2019-12-30",
                "peak_value": 150,
                "duration_weeks": 1,
            },
            {
                "start_date": "2024-01-01",
                "end_date": "2024-01-01",
                "peak_date": "2024-01-01",
                "peak_value": 170,
                "duration_weeks": 1,
            },
        ]
    ).to_csv(spike_directory / "step2_unique_spikes.csv", index=False)
    expected_dir = project.root / "thesis_output"
    expected_dir.mkdir()
    monkeypatch.setattr(thesis, "OUTPUT_DIR", expected_dir)
    monkeypatch.setattr(ibis, "connect", lambda *args, **kwargs: connection_factory())
    monkeypatch.setattr(Project, "connect", lambda self, **kwargs: connection_factory())
    thesis.main()
    with use_project(project):
        current.main()
    actual_path = next(project.data_path("gpr_spikes", "spike_articles").glob("*.json"))
    expected_path = next(expected_dir.glob("*.json"))
    actual = json.loads(actual_path.read_text())
    assert actual == json.loads(expected_path.read_text())
    assert actual[0]["spike_info"]["article_end_date"] == "2020-01-05"
    assert actual[0]["articles"]
    assert actual[1]["total_articles"] == 0


def test_analysis_imports_create_no_files_or_directories(
    workflows, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    for current, _ in workflows.values():
        importlib.reload(current)
    assert list(tmp_path.iterdir()) == []
