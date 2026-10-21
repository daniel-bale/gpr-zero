"""Median artifacts compared against the pinned thesis workflow."""

import ast
import logging
import subprocess
import types
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from gpr_zero.project import Project
from gpr_zero.workflows.compute_step2_median_3outlets import main

THESIS_COMMIT = "290b7c4c8eb900a712384deea0b68e2e47009624"
THESIS_PATH = "scripts/compute_step2_median_3outlets.py"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_median_artifacts_and_reference_correlations_match_thesis(
    tmp_path, monkeypatch, caplog
):
    source = subprocess.run(
        ["git", "show", f"{THESIS_COMMIT}:{THESIS_PATH}"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(functions) == 1, "Required pinned thesis workflow is missing"
    oracle_root = tmp_path / "thesis"
    oracle = types.ModuleType("thesis_median_workflow")
    oracle.__dict__.update(
        Path=Path,
        pd=pd,
        np=np,
        logger=logging.getLogger("thesis_median_workflow"),
        __file__=str(oracle_root / THESIS_PATH),
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), THESIS_PATH, "exec"),
        oracle.__dict__,
    )

    # The current workflow must honor a data directory outside its project root.
    project = Project(
        root=tmp_path / "current",
        data_dir=tmp_path / "custom-data",
        database=tmp_path / "unused.duckdb",
        llm_config=tmp_path / "unused.yaml",
    )
    weeks = pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-27", "2020-02-03"])
    fixtures = {
        "step2_z": ([0, 1, 2, 3], [0.125, 0.375, 0.25, 0.5]),
        "step2_a": ([0, 2, 3], [0.25, 0.5, 0.75]),
        "step2_m": ([0, 1, 3], [0.375, 0.625, 0.25]),
        "step1_a": ([0, 1, 2, 3], [100.0, 120.0, 140.0, 160.0]),
        "step1_b": ([0, 1, 3], [25.0, 60.0, 120.0]),
        "lau_overall_taiwan_recalculated": ([0, 2, 3], [4.0, 3.0, 9.0]),
    }
    for directory in [
        oracle_root / "data" / "gpr_timeseries_3outlets",
        project.data_path("gpr_timeseries_3outlets"),
    ]:
        directory.mkdir(parents=True)
        for name, (indices, values) in fixtures.items():
            pd.DataFrame(
                {
                    "week": weeks.take(indices),
                    "gpr_index": values,
                    "total_articles": [100] * len(indices),
                }
            ).to_parquet(directory / f"{name}_W_overall_3outlets.parquet")

    caplog.set_level(logging.INFO)
    oracle.main()
    correlation_prefixes = (
        "Mean pairwise correlation among",
        "Correlation between Step2 median and Lau's index",
    )
    expected_correlations = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(correlation_prefixes)
    ]
    caplog.clear()
    monkeypatch.chdir(tmp_path)
    main(project)
    actual_correlations = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(correlation_prefixes)
    ]
    assert len(expected_correlations) == 3
    assert actual_correlations == expected_correlations

    expected_directory = oracle_root / "data" / "gpr_indices_3outlets"
    actual_directory = project.data_path("gpr_indices_3outlets")
    expected_files = {path.name for path in expected_directory.iterdir()}
    assert expected_files == {
        "step2_median_gpr_3outlets.csv",
        "step2_median_gpr_3outlets.parquet",
    }
    assert {path.name for path in actual_directory.iterdir()} == expected_files
    for name in expected_files:
        assert (actual_directory / name).read_bytes() == (
            expected_directory / name
        ).read_bytes()

    result = pd.read_parquet(actual_directory / "step2_median_gpr_3outlets.parquet")
    expected = pd.read_parquet(expected_directory / "step2_median_gpr_3outlets.parquet")
    assert_frame_equal(result, expected, check_exact=True)
    # Missing observations stay missing: each week's denominator is the number
    # of available Step2 dictionaries; Step1/Lau values cannot enter the median.
    assert result["count"].tolist() == [3, 2, 2, 3]
    assert result["gpr_index"].tolist() == [0.25, 0.5, 0.375, 0.5]
    assert result["week"].tolist() == weeks.tolist()
