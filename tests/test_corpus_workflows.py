"""End-to-end outputs and thesis-oracle checks for the shared cached corpus."""

import ast
import json
import subprocess
from pathlib import Path

import ibis
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gpr_zero.constants import TARGET_OUTLETS
from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory
from gpr_zero.project import Project
from gpr_zero.workflows.calculate_category_gpr_complete import (
    calculate_category_gpr_for_dictionary,
)
from gpr_zero.workflows.compute_all_dictionary_timeseries import main as compute_all
from gpr_zero.workflows.compute_all_dictionary_timeseries import rebuild_cache
from gpr_zero.workflows.compute_all_dictionary_timeseries_3outlets import (
    main as compute_three,
)

THESIS_COMMIT = "290b7c4c8eb900a712384deea0b68e2e47009624"


@pytest.fixture
def project(tmp_path):
    data = tmp_path / "data"
    dictionaries = data / "dictionaries" / "all_dictionaries"
    dictionaries.mkdir(parents=True)
    for name, word in [("step1_a", "apple"), ("step2_z", "zebra")]:
        definition = {
            "word_sets": {"word": [word]},
            "categories": {"Military Escalation": "word"},
        }
        (dictionaries / f"{name}.json").write_text(json.dumps(definition))
    project = Project(tmp_path, data, data / "articles.duckdb", tmp_path / "llm.yaml")
    conn = project.connect(read_only=False)
    conn.create_table(
        "articles",
        pd.DataFrame(
            {
                "article_id": [1, 2, 3, 4],
                "content": ["apple", "zebra", "zebra", "zebra"],
                "outlet": [
                    TARGET_OUTLETS[0],
                    TARGET_OUTLETS[1],
                    "Other",
                    TARGET_OUTLETS[2],
                ],
                "warc_date": ["2020-01-01", "2020-01-01", "2020-01-01", "2016-12-31"],
                "source_warc_file": ["fixture.warc"] * 4,
            }
        ),
    )
    conn.disconnect()
    rebuild_cache(project)
    return project


def test_time_series_outputs_keep_distinct_thesis_denominators(project):
    compute_all(project)
    compute_three(project)
    all_outlets = pd.read_parquet(
        project.data_path("gpr_timeseries", "step2_z_W_overall.parquet")
    )
    three = pd.read_parquet(
        project.data_path(
            "gpr_timeseries_3outlets", "step2_z_W_overall_3outlets.parquet"
        )
    )
    assert all_outlets["total_articles"].tolist() == [1, 3]
    assert all_outlets["total_gpr_count"].tolist() == [1, 2]
    assert all_outlets["total_gpr_share"].tolist() == pytest.approx(
        [1 / (1 + 1e-9), 2 / (3 + 1e-9)]
    )
    assert three["total_articles"].tolist() == [1, 2]
    assert three["gpr_articles"].tolist() == [1, 1]
    assert three["gpr_index"].tolist() == [1, 0.5]
    summary = json.loads(
        project.data_path(
            "gpr_timeseries_3outlets", "processing_summary_3outlets.json"
        ).read_text()
    )
    assert summary["successful"] == 2
    assert summary["failed"] == 0


@pytest.fixture(scope="module")
def thesis_category_calculation():
    path = "scripts/calculate_category_gpr_complete.py"
    source = subprocess.run(
        ["git", "show", f"{THESIS_COMMIT}:{path}"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "calculate_category_gpr_for_dictionary"
    )
    namespace = {"ibis": ibis}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), f"thesis:{path}", "exec"),
        namespace,
    )
    return namespace[function.name]


@pytest.mark.parametrize(
    "definition",
    [
        "word",
        "word AND missing",
        "word AND empty",
        "missing",
        "empty",
        "",
        "word + word",
    ],
)
def test_category_workflow_matches_thesis(
    project, thesis_category_calculation, definition
):
    inventory = DictionaryInventory.from_directory(project.dictionaries_dir)
    dictionary = {
        "word_sets": {"word": ["zebra"], "empty": []},
        "categories": {"category": definition},
    }
    conn = project.connect()
    try:
        expected = thesis_category_calculation(
            conn, dictionary, "category", inventory.word_columns
        )
        corpus = (
            ArticleCorpus.from_cache(conn, inventory).three_outlets().study_period()
        )
        actual = calculate_category_gpr_for_dictionary(corpus, dictionary, "category")
        if expected is None:
            assert actual is None
        else:
            assert_frame_equal(
                actual.reset_index(drop=True), expected.reset_index(drop=True)
            )
    finally:
        conn.disconnect()


def test_category_verification_uses_same_three_outlet_study_cohort(project, capsys):
    from gpr_zero.workflows.verify_step2_category_contributions_3outlets import (
        main as verify,
    )

    verify(project)
    output = capsys.readouterr().out
    assert "step2_z" in output
    assert "Military Escalation" in output
    assert "Min top contribution: 100.00%" in output
    assert "Median top contribution: 100.00%" in output


def test_all_dictionary_workflow_reports_invalid_file_without_shifting_cache(project):
    partial = project.dictionaries_dir / "partial.json"
    partial.write_text(json.dumps({"word_sets": {"noise": ["aaa"]}}))
    compute_all(project)
    summary = json.loads(
        project.data_path("gpr_timeseries", "processing_summary.json").read_text()
    )
    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["results"]["partial"]["status"] == "failed"
    series = pd.read_parquet(
        project.data_path("gpr_timeseries", "step2_z_W_overall.parquet")
    )
    assert series["total_gpr_count"].tolist() == [1, 2]
