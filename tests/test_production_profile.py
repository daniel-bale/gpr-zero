"""Read-only checks against independently saved thesis profile artifacts.

Run with GPR_THESIS_PROFILE pointing to an exported profile directory. CSV
round-trip parsing preserves the supplied binary64 values; ordinary pandas CSV
parsing introduces extra rounding differences before the calculation begins.

Event records are checked exactly. The saved Lau threshold is deliberately
tested separately: the current and pinned thesis routines both produce one
binary64 ULP below that artifact. This is an observed discrepancy, not a claim
that the complete Lau artifact is identical.
"""

import ast
import logging
import os
import subprocess
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gpr_zero.gpr_index.spikes import (
    detect_weekly_events,
    events_overlap,
    select_overlapping_events,
)

THESIS_COMMIT = "290b7c4c8eb900a712384deea0b68e2e47009624"
THESIS_PATH = "scripts/detect_gpr_spikes_unified.py"
DATE_COLUMNS = ["start_date", "end_date", "peak_date"]
EVENT_COLUMNS = [
    *DATE_COLUMNS,
    "peak_value",
    "duration_weeks",
    "mean_spike_value",
]
VALUE_COLUMNS = {"step2": "median_raw", "lau": "gpr_share"}


@pytest.fixture(scope="module")
def production_profile():
    configured = os.environ.get("GPR_THESIS_PROFILE")
    if not configured:
        pytest.skip(
            "Set GPR_THESIS_PROFILE to run read-only saved thesis artifact checks"
        )
    path = Path(configured).expanduser()
    required = [
        "step2_weekly_index.csv",
        "lau_weekly_index.csv",
        "step2_spikes.csv",
        "lau_spikes.csv",
        "step2_unique_spikes.csv",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        pytest.fail(f"Configured thesis profile is incomplete: {path}: {missing}")
    return path


def read_weekly(profile, name):
    return pd.read_csv(
        profile / f"{name}_weekly_index.csv",
        float_precision="round_trip",
        parse_dates=["week"],
    ).rename(columns={"week": "date"})


def read_saved_events(profile, name):
    return pd.read_csv(
        profile / f"{name}_spikes.csv",
        float_precision="round_trip",
        parse_dates=DATE_COLUMNS,
    )


def calculate_events(profile, name):
    events, threshold = detect_weekly_events(
        read_weekly(profile, name), VALUE_COLUMNS[name]
    )
    result = pd.DataFrame(events, columns=EVENT_COLUMNS)
    result["threshold"] = threshold
    return result


@pytest.mark.parametrize("name", ["step2", "lau"])
def test_all_saved_event_fields_are_exact(production_profile, name):
    actual = calculate_events(production_profile, name)
    saved = read_saved_events(production_profile, name)
    # Check every produced event field, its order, date type and duration type.
    assert (
        actual.columns.tolist()
        == saved.columns.tolist()
        == [*EVENT_COLUMNS, "threshold"]
    )
    assert_frame_equal(actual[EVENT_COLUMNS], saved[EVENT_COLUMNS], check_exact=True)


def test_step2_saved_threshold_is_exact(production_profile):
    actual = calculate_events(production_profile, "step2")
    saved = read_saved_events(production_profile, "step2")
    assert_frame_equal(actual, saved, check_exact=True)


def test_lau_threshold_is_one_ulp_below_saved_artifact(production_profile):
    """Record the actual mismatch explicitly, without a tolerance comparison."""
    actual = calculate_events(production_profile, "lau")["threshold"]
    saved = read_saved_events(production_profile, "lau")["threshold"]
    assert len(actual) == len(saved)
    assert actual.nunique() == saved.nunique() == 1
    assert actual.iloc[0] != saved.iloc[0]
    assert actual.iloc[0] == np.nextafter(saved.iloc[0], -np.inf)


@pytest.mark.parametrize("policy", ["week_overlap", "summary_overlap"])
def test_unique_event_selection_matches_every_saved_field(production_profile, policy):
    step2 = calculate_events(production_profile, "step2")
    lau = calculate_events(production_profile, "lau")
    if policy == "week_overlap":
        keep = [
            not any(events_overlap(event, other) for _, other in lau.iterrows())
            for _, event in step2.iterrows()
        ]
        actual = step2.loc[keep]
    else:
        shared = select_overlapping_events(step2, lau)
        actual = step2.drop(index=shared.index)
    saved = read_saved_events(production_profile, "step2_unique")
    assert_frame_equal(actual.reset_index(drop=True), saved, check_exact=True)


@pytest.fixture(scope="module")
def pinned_thesis_detector():
    source = subprocess.run(
        ["git", "show", f"{THESIS_COMMIT}:{THESIS_PATH}"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "detect_spikes"
    ]
    assert len(functions) == 1, "Required pinned thesis detector is missing"
    oracle = types.ModuleType("pinned_production_detector")
    oracle.__dict__.update(
        pd=pd,
        logger=logging.getLogger(__name__),
        SPIKE_STD_THRESHOLD=1.0,
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), THESIS_PATH, "exec"),
        oracle.__dict__,
    )
    return oracle.detect_spikes


@pytest.mark.parametrize("name", ["step2", "lau"])
def test_current_detector_and_threshold_match_pinned_thesis_on_real_weeks(
    production_profile, pinned_thesis_detector, name
):
    """The Lau saved-threshold discrepancy also exists in the pinned routine."""
    expected_events, expected_threshold = pinned_thesis_detector(
        read_weekly(production_profile, name), VALUE_COLUMNS[name]
    )
    expected = pd.DataFrame(expected_events, columns=EVENT_COLUMNS)
    expected["threshold"] = expected_threshold
    assert_frame_equal(
        calculate_events(production_profile, name), expected, check_exact=True
    )
