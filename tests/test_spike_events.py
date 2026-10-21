"""Event behavior and CSV artifacts compared with the pinned thesis routines."""

import ast
import importlib
import json
import logging
import subprocess
import types
from datetime import timedelta
from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from gpr_zero.gpr_index.spikes import (
    article_window,
    compare_peak_events,
    detect_newspaper_events,
    detect_weekly_events,
    events_overlap,
    select_overlapping_events,
    unique_events_by_newspaper,
)
from gpr_zero.project import Project, use_project

THESIS_COMMIT = "290b7c4c8eb900a712384deea0b68e2e47009624"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def thesis_functions(path, names):
    """Load unchanged function bodies without thesis import-time plotting/I/O."""
    result = subprocess.run(
        ["git", "show", f"{THESIS_COMMIT}:{path}"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    tree = ast.parse(result.stdout)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(selected) == len(names), "Required pinned thesis routines are missing"
    module = types.ModuleType("thesis_spike_oracle")
    module.__dict__.update(
        pd=pd, np=np, logger=logging.getLogger(__name__), SPIKE_STD_THRESHOLD=1.0
    )
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), f"thesis:{path}", "exec"),
        module.__dict__,
    )
    return module


@pytest.fixture(scope="module")
def unified_oracle():
    return thesis_functions(
        "scripts/detect_gpr_spikes_unified.py",
        {"detect_spikes", "compare_spike_events"},
    )


@pytest.fixture(scope="module")
def newspaper_oracle():
    return thesis_functions(
        "scripts/calculate_newspaper_specific_step2_gpr.py",
        {"detect_newspaper_spikes", "analyze_unique_spikes"},
    )


VALUES = [
    [],
    [1.0],
    [1.0] * 8,
    [np.nan] * 4,
    [0.0] * 8 + [10.0, 10.0],
    [10.0, 11.0] + [0.0] * 8,
    [0.0, 20.0, np.nan, 20.0, 0.0, 0.0, 0.0, 0.0],
    [0.0] * 5 + [10.0, 0.0, 11.0, 11.0, 0.0, 0.0],
]


@pytest.mark.parametrize("values", VALUES)
def test_unified_detection_matches_thesis(unified_oracle, values):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-06", periods=len(values), freq="W-MON"),
            "share": pd.Series(values, dtype=float),
        }
    )
    expected_frame = frame.copy()
    expected, threshold = unified_oracle.detect_spikes(expected_frame, "share")
    actual, actual_threshold = detect_weekly_events(frame, "share")
    assert actual == expected
    if pd.isna(threshold):
        assert pd.isna(actual_threshold)
    else:
        assert actual_threshold == threshold
    assert_frame_equal(frame, expected_frame, check_exact=True)


def test_adjacent_rows_and_first_tied_peak_are_preserved(unified_oracle):
    dates = pd.date_range("2020-01-06", periods=10, freq="W-MON").tolist()
    dates[-1] += timedelta(weeks=4)
    frame = pd.DataFrame({"date": dates, "share": [0.0] * 8 + [10.0, 10.0]})
    events, threshold = detect_weekly_events(frame, "share")
    assert (events, threshold) == unified_oracle.detect_spikes(frame.copy(), "share")
    assert len(events) == 1
    assert events[0]["duration_weeks"] == 2
    assert events[0]["peak_date"] == dates[-2]
    assert events[0]["end_date"] == dates[-1]


@pytest.mark.parametrize("values", VALUES)
def test_newspaper_detection_and_export_match_thesis(
    newspaper_oracle, tmp_path, values
):
    dates = pd.date_range("2020-01-06", periods=len(values), freq="W-MON")
    frame = pd.DataFrame({"median_scaled": pd.Series(values, index=dates, dtype=float)})
    expected_frame = frame.copy()
    expected = newspaper_oracle.detect_newspaper_spikes({"A": expected_frame}, tmp_path)
    expected_bytes = (tmp_path / "newspaper_spikes.csv").read_bytes()
    assert detect_newspaper_events(frame) == expected["A"]
    assert_frame_equal(frame, expected_frame, check_exact=True)
    current = importlib.import_module(
        "gpr_zero.workflows.calculate_newspaper_specific_step2_gpr"
    )
    assert current.detect_newspaper_spikes({"A": frame.copy()}, tmp_path) == expected
    assert (tmp_path / "newspaper_spikes.csv").read_bytes() == expected_bytes


def event(start, end=None, peak=None, value=10.0):
    return {
        "start_date": pd.Timestamp(start),
        "end_date": pd.Timestamp(end or start),
        "peak_date": pd.Timestamp(peak or start),
        "peak_value": value,
        "duration_weeks": 1,
    }


@pytest.mark.parametrize("tolerance", [0, 6, 7, 8, 14])
def test_peak_matching_order_and_tolerance_match_thesis(unified_oracle, tolerance):
    new = [event("2020-01-13"), event("2020-01-06"), event("2020-02-10")]
    old = pd.DataFrame(
        [
            {**event("2020-01-06", "2020-01-20"), "peak_gpr_share": 2.0},
            {**event("2020-01-13"), "peak_gpr_share": 3.0},
            {**event("2020-06-01"), "peak_gpr_share": 4.0},
        ]
    )
    assert compare_peak_events(
        new, old, tolerance
    ) == unified_oracle.compare_spike_events(new, old, tolerance)


@pytest.mark.parametrize(
    "old,new",
    [
        (pd.DataFrame(), []),
        (pd.DataFrame(), [event("2020-01-06")]),
        (pd.DataFrame([event("2020-01-06")]), []),
    ],
)
def test_peak_matching_empty_and_missing_value_match_thesis(unified_oracle, old, new):
    assert compare_peak_events(new, old) == unified_oracle.compare_spike_events(
        new, old
    )


def test_inclusive_article_window_is_distinct_from_peak_and_week_overlap():
    spike = event("2020-02-24", "2020-03-02", "2020-03-02")
    start, end = article_window(spike)
    assert start == pd.Timestamp("2020-02-24")
    assert end == pd.Timestamp("2020-03-08")
    assert article_window({"start": start, "end": pd.Timestamp("2020-03-02")}) == (
        start,
        end,
    )
    assert events_overlap(spike, event("2020-03-02"))
    assert not events_overlap(spike, event("2020-03-03"))
    assert article_window(event("2020-12-28")) == (
        pd.Timestamp("2020-12-28"),
        pd.Timestamp("2021-01-03"),
    )


def test_newspaper_uniqueness_and_csv_match_thesis(newspaper_oracle, tmp_path):
    spikes = {
        "A": [event("2020-01-06", "2020-01-13"), event("2020-02-03")],
        "B": [event("2020-01-13", "2020-01-20"), event("2020-02-10")],
        "C": [],
    }
    expected = newspaper_oracle.analyze_unique_spikes(spikes, tmp_path)
    expected_bytes = (tmp_path / "newspaper_unique_spikes.csv").read_bytes()
    assert unique_events_by_newspaper(spikes) == expected
    current = importlib.import_module(
        "gpr_zero.workflows.calculate_newspaper_specific_step2_gpr"
    )
    assert current.analyze_unique_spikes(spikes, tmp_path) == expected
    assert (tmp_path / "newspaper_unique_spikes.csv").read_bytes() == expected_bytes


@pytest.mark.parametrize(
    "selected",
    [[event("2020-01-13")], [event("2020-01-27")], [event("2030-01-01")], []],
)
def test_summary_csv_matches_thesis_from_disk(tmp_path, selected):
    """Exercise actual CSV reads/writes, including the no-output behavior."""
    project = Project(
        tmp_path, tmp_path / "data", tmp_path / "news.duckdb", tmp_path / "llm.yaml"
    )
    directory = project.data_path("gpr_spikes")
    (directory / "spike_comparison").mkdir(parents=True)
    all_file = directory / "step2_all_spikes_summary.csv"
    unique_file = directory / "spike_comparison" / "step2_unique_spikes.csv"
    output = directory / "step2_unique_spikes_summary.csv"
    records = [
        {
            **event("2020-01-06", "2020-01-20"),
            "spike_number": 2,
            "event_summary": "Multi week",
            "keywords": "國防",
        },
        {
            **event("2020-01-27"),
            "spike_number": 1,
            "event_summary": "Single week",
            "keywords": "風險",
        },
    ]
    pd.DataFrame(records).to_csv(all_file, index=False)
    pd.DataFrame(selected, columns=list(event("2020-01-06"))).to_csv(
        unique_file, index=False
    )
    oracle = thesis_functions("scripts/create_step2_unique_spikes_summary.py", {"main"})
    oracle.ALL_SPIKES_FILE, oracle.UNIQUE_SPIKES_FILE, oracle.OUTPUT_FILE = (
        all_file,
        unique_file,
        output,
    )
    oracle.main()
    expected_bytes = output.read_bytes() if output.exists() else None
    output.unlink(missing_ok=True)
    current = importlib.import_module(
        "gpr_zero.workflows.create_step2_unique_spikes_summary"
    )
    with use_project(project):
        current.main()
    assert (output.read_bytes() if output.exists() else None) == expected_bytes


def test_summary_single_week_peak_and_daily_sampling_policy():
    rows = pd.DataFrame(
        [event("2020-01-06", peak="2020-01-13"), event("2020-02-01", "2020-02-03")]
    )
    assert select_overlapping_events(
        rows, pd.DataFrame([event("2020-01-13")])
    ).index.tolist() == [0]
    # A daily range beginning at noon does not contain midnight; retain the
    # historical discrete-date policy rather than changing it to interval overlap.
    selected = pd.DataFrame([event("2020-02-01 12:00", "2020-02-03 12:00")])
    assert select_overlapping_events(rows, selected).empty


def test_newspaper_duplicate_dates_and_row_endpoints_match_thesis(
    newspaper_oracle, tmp_path
):
    dates = pd.date_range("2020-01-06", periods=10, freq="W-MON").tolist()
    dates[-2], dates[-1] = dates[0], dates[0]
    frame = pd.DataFrame({"median_scaled": [0.0] * 8 + [10.0, 11.0]}, index=dates)
    assert (
        detect_newspaper_events(frame.copy())
        == newspaper_oracle.detect_newspaper_spikes({"A": frame}, tmp_path)["A"]
    )


@pytest.mark.parametrize("values", [[0.0] * 8 + [10.0, 10.0], [1.0] * 10])
def test_unified_workflow_csv_artifacts_match_thesis(tmp_path, values):
    project = Project(
        tmp_path, tmp_path / "data", tmp_path / "news.duckdb", tmp_path / "llm.yaml"
    )
    step2 = project.data_path("gpr_indices_3outlets", "step2_median_gpr_3outlets.csv")
    lau = project.data_path(
        "gpr_timeseries_3outlets",
        "lau_overall_taiwan_recalculated_W_overall_3outlets.parquet",
    )
    old = (
        tmp_path
        / "data_analysis/output/gpr_spikes/yeu_2024_20250715160933/detected_spike_events_with_llm_all_categories_2017_01_to_2024_12.csv"
    )
    for path in [step2, lau, old]:
        path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2020-01-06", periods=len(values), freq="W-MON")
    pd.DataFrame({"week": dates, "gpr_index": values}).to_csv(step2, index=False)
    pl.from_pandas(
        pd.DataFrame({"warc_date_bucketed": dates, "gpr_index": values})
    ).write_parquet(lau)
    pd.DataFrame(
        [
            {
                "gpr_category": "overall",
                "newspaper": "Overall Taiwan - Overall",
                "start_date": dates[-2],
                "end_date": dates[-1],
                "peak_date": dates[-2],
                "peak_gpr_share": 10.0,
            }
        ]
    ).to_csv(old, index=False)

    oracle = thesis_functions(
        "scripts/detect_gpr_spikes_unified.py",
        {
            "main",
            "load_gpr_timeseries",
            "detect_spikes",
            "compare_spike_events",
        },
    )
    oracle.__dict__.update(
        Path=Path,
        pl=pl,
        json=json,
        __file__=str(tmp_path / "scripts/detect_gpr_spikes_unified.py"),
    )
    oracle.main()
    output = project.data_path("gpr_spikes", "unified_detection_3outlets")
    expected = {path.name: path.read_bytes() for path in output.glob("*.csv")}
    expected_summary = json.loads((output / "spike_detection_summary.json").read_text())
    current = importlib.import_module("gpr_zero.workflows.detect_gpr_spikes_unified")
    with use_project(project):
        current.main()
    assert {path.name: path.read_bytes() for path in output.glob("*.csv")} == expected
    actual_summary = json.loads((output / "spike_detection_summary.json").read_text())
    del actual_summary["analysis_date"], expected_summary["analysis_date"]
    assert actual_summary == expected_summary


def test_newspaper_calculation_from_cached_articles_matches_thesis(
    tmp_path, monkeypatch
):
    """Compare the cache adapter migration through the actual calculation path."""
    project = Project(
        tmp_path, tmp_path / "data", tmp_path / "news.duckdb", tmp_path / "llm.yaml"
    )
    project.dictionaries_dir.mkdir(parents=True)
    for name, word in [
        ("step1_other", "A"),
        ("step2_war", "戰爭"),
        ("step2_risk", "風險"),
    ]:
        (project.dictionaries_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "word_sets": {"topic": [word]},
                    "categories": {"risk": "topic"},
                }
            ),
            encoding="utf-8",
        )

    source = subprocess.run(
        [
            "git",
            "show",
            f"{THESIS_COMMIT}:src/gpr_zero/gpr_index/calculator_binary_matrix.py",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    calculator_module = types.ModuleType("thesis_newspaper_calculator")
    exec(
        compile(source, "thesis:calculator_binary_matrix.py", "exec"),
        calculator_module.__dict__,
    )
    calculator_class = calculator_module.GPRCalculatorBinaryMatrix
    current = importlib.import_module(
        "gpr_zero.workflows.calculate_newspaper_specific_step2_gpr"
    )
    connection = ibis.duckdb.connect(str(project.database))
    rows = [
        {
            "outlet": outlet,
            "warc_date": date,
            "content": text,
            "source_warc_file": "fixture.warc",
        }
        for outlet in current.OUTLET_ENGLISH
        for date, texts in [
            ("2020-01-06", ["戰爭", "一般", "風險"]),
            ("2020-01-13", ["戰爭風險", "一般", "戰爭"]),
        ]
        for text in texts
    ]
    try:
        connection.create_table("articles", pl.DataFrame(rows))
        calculator_class(
            list(map(str, sorted(project.dictionaries_dir.glob("*.json"))))
        ).prepare_binary_matrix(
            connection.table("articles"),
            materialize=True,
        )
        oracle = thesis_functions(
            "scripts/calculate_newspaper_specific_step2_gpr.py",
            {"calculate_newspaper_gpr_indices"},
        )

        def oracle_path(path):
            if str(path) == "/home5/chuenho/data/news/stage2_enhanced_full.duckdb":
                return project.database
            return Path(path)

        oracle.__dict__.update(
            Path=oracle_path,
            __file__=str(tmp_path / "scripts/newspaper.py"),
            ibis=types.SimpleNamespace(connect=lambda _: connection),
            GPRCalculatorBinaryMatrix=calculator_class,
            tqdm=lambda rows, **kwargs: rows,
            NEWSPAPERS=current.NEWSPAPERS,
            OUTLET_ENGLISH=current.OUTLET_ENGLISH,
            create_newspaper_comparison_plot=lambda *args: None,
            calculate_newspaper_correlations=lambda *args: None,
        )
        expected = oracle.calculate_newspaper_gpr_indices()
        output = project.data_path("newspaper_gpr_step2")
        expected_csv = {path.name: path.read_bytes() for path in output.glob("*.csv")}
        expected_raw = {
            path.name: pd.read_parquet(path) for path in output.glob("*.parquet")
        }
        monkeypatch.setattr(Project, "connect", lambda *args, **kwargs: connection)
        monkeypatch.setitem(
            current.calculate_newspaper_gpr_indices.__globals__,
            "create_newspaper_comparison_plot",
            lambda *args: None,
        )
        monkeypatch.setitem(
            current.calculate_newspaper_gpr_indices.__globals__,
            "calculate_newspaper_correlations",
            lambda *args: None,
        )
        with use_project(project):
            actual = current.calculate_newspaper_gpr_indices()
        assert actual.keys() == expected.keys()
        assert len(actual) == 3
        for newspaper in expected:
            assert_frame_equal(actual[newspaper], expected[newspaper], check_exact=True)
        assert {
            path.name: path.read_bytes() for path in output.glob("*.csv")
        } == expected_csv
        for path in output.glob("*.parquet"):
            assert_frame_equal(
                pd.read_parquet(path), expected_raw[path.name], check_exact=True
            )
    finally:
        connection.disconnect()
