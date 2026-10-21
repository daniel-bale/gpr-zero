"""The profile verifier rejects corruption and numeric drift without writing inputs."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from gpr_zero.constants import STANDARD_CATEGORIES
from gpr_zero.gpr_index.spikes import detect_weekly_events
from gpr_zero.verification import PROFILE_NAME, verify_profile
from gpr_zero.workflows.verify_profile import main


def fingerprint(path):
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


@pytest.fixture
def profile(tmp_path):
    """Three weeks with canonical dictionary counts; files total only a few KB."""
    root = tmp_path / "rebuild"
    profile = root / "thesis_profile"
    indices = root / "indices"
    profile.mkdir(parents=True)
    indices.mkdir()
    weeks = pd.date_range("2020-01-06", periods=3, freq="7D")
    omitted = "step2_gemini-2.5-pro_stable8cat_run8"
    step2 = [omitted] + [f"step2_dictionary_{i:02}" for i in range(39)]
    step1 = [f"step1_dictionary_{i:02}" for i in range(35)]
    rows = []
    for scope in ("all_outlets", "three_outlets"):
        for dictionary in step1 + step2 + ["lau_2024"]:
            counts = [1, 90, 10] if dictionary == "lau_2024" else [0, 10, 90]
            for week, count in zip(weeks, counts):
                rows.append(
                    {
                        "week": week,
                        "scope": scope,
                        "endpoint_policy": "full_cache",
                        "dictionary": dictionary,
                        "total_articles": 100,
                        "gpr_articles": count,
                        "gpr_share": count / 100,
                    }
                )
    dictionary_weekly = pd.DataFrame(rows)
    dictionary_weekly.to_parquet(indices / "dictionary_weekly.parquet", index=False)
    category_rows = [
        {
            "week": week,
            "scope": "three_outlets",
            "endpoint_policy": "legacy_end_midnight",
            "dictionary": dictionary,
            "category": category,
            "total_articles": 100,
            "category_articles": count,
            "category_share": count / 100,
        }
        for dictionary in step2
        for category in STANDARD_CATEGORIES
        for week, count in zip(weeks, [0, 10, 90])
    ]
    pd.DataFrame(category_rows).to_parquet(
        indices / "category_weekly.parquet", index=False
    )
    aggregate = {
        "status": "complete",
        "source_mode": "full",
        "complete": True,
        "input_complete": True,
        "source_complete": True,
        "calendar_timezone": "Asia/Taipei",
        "dictionary_catalog": {
            "dictionaries": [
                {"name": name, "family": family}
                for family, names in (
                    ("step1", step1),
                    ("step2", step2),
                    ("lau", ["lau_2024"]),
                )
                for name in names
            ]
        },
        "outputs": {
            name: {"parquet": f"{name}.parquet"}
            for name in ("dictionary_weekly", "category_weekly")
        },
    }
    aggregate_path = indices / "manifest.json"
    aggregate_path.write_text(json.dumps(aggregate))
    step2_weekly = pd.DataFrame(
        {"week": weeks, "median_raw": [0.0, 0.1, 0.9], "dictionary_count": 40}
    )
    step2_weekly["median_scaled"] = (
        step2_weekly.median_raw / step2_weekly.median_raw.mean() * 100
    )
    step2_weekly.to_csv(profile / "step2_weekly_index.csv", index=False)
    lau = pd.DataFrame(
        {
            "week": weeks,
            "dictionary": "lau",
            "total_articles": 100,
            "gpr_articles": [1, 90, 10],
            "gpr_share": [0.01, 0.9, 0.1],
        }
    )
    lau["mean100_index"] = lau.gpr_share / (1.01 / 3) * 100
    lau.to_csv(profile / "lau_weekly_index.csv", index=False)
    pd.DataFrame(
        [
            {"week": week, "category": category, "category_share": count / 100}
            for category in STANDARD_CATEGORIES
            for week, count in zip(weeks, [0, 10, 90])
        ]
    ).to_csv(profile / "category_medians.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": family,
                "scope": "all_outlets",
                "endpoint_policy": "full_cache_then_study_week_labels",
                "dictionary_count": count,
                "weeks": 3,
                "first_week": weeks[0],
                "last_week": weeks[-1],
                "pair_count": count * (count - 1) // 2,
                "expected_pair_count": count * (count - 1) // 2,
                "mean_pairwise_correlation": 1.0,
                "incomplete_weeks": 0,
            }
            for family, count in (("step1", 35), ("step2", 40))
        ]
    ).to_csv(profile / "pairwise_correlations.csv", index=False)
    for family, frame, column in (
        ("step2", step2_weekly, "median_raw"),
        ("lau", lau, "gpr_share"),
    ):
        events, threshold = detect_weekly_events(
            frame.rename(columns={"week": "date"}).copy(), column
        )
        result = pd.DataFrame(events).assign(threshold=threshold)
        result.to_csv(profile / f"{family}_spikes.csv", index=False)
        if family == "step2":
            result.to_csv(profile / "step2_unique_spikes.csv", index=False)
    for i in range(17):
        (profile / f"retained_{i:02}.txt").write_text(
            "retained figure/model provenance\n"
        )
    manifest = {
        "profile": PROFILE_NAME,
        "status": "complete",
        "complete": True,
        "calendar_timezone": "Asia/Taipei",
        "rebuild_dir": str(root),
        "all_thesis_claims_reproduced": False,
        "inputs": [fingerprint(aggregate_path)],
        "artifacts": {path.name: fingerprint(path) for path in profile.iterdir()},
    }
    manifest["artifacts"]["category_medians.csv"]["selection"] = (
        f"Three outlets; 39 dictionaries, excluding {omitted}; historical endpoint"
    )
    (profile / "manifest.json").write_text(json.dumps(manifest))
    return profile


def test_complete_small_profile_is_read_only_and_exact(profile):
    before = {
        str(path): fingerprint(path)
        for path in profile.parent.rglob("*")
        if path.is_file()
    }
    result = verify_profile(profile)
    assert result["status"] == "passed", [
        c for c in result["checks"] if not c["passed"]
    ]
    assert len(result["checks"]) == 34
    assert result["precision_notes"] == []
    assert result["all_thesis_claims_reproduced"] is False
    after = {
        str(path): fingerprint(path)
        for path in profile.parent.rglob("*")
        if path.is_file()
    }
    assert before == after


@pytest.mark.parametrize("damage", ["missing", "size", "same_size"])
def test_artifact_integrity_errors_return_failure(profile, damage):
    path = profile / "retained_00.txt"
    if damage == "missing":
        path.unlink()
    elif damage == "size":
        path.write_text("short")
    else:
        original = path.read_bytes()
        path.write_bytes(b"X" + original[1:])
    result = verify_profile(profile)
    assert result["status"] == "failed"
    check = next(c for c in result["checks"] if c["name"] == "artifact:retained_00.txt")
    assert check["passed"] is False
    assert "error" in check


def test_numeric_drift_is_not_hidden_by_tolerance(profile):
    path = profile / "step2_weekly_index.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[1, "median_raw"] = np.nextafter(frame.loc[1, "median_raw"], np.inf)
    frame.to_csv(path, index=False)
    manifest_path = profile / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][path.name] = fingerprint(path)
    manifest_path.write_text(json.dumps(manifest))
    result = verify_profile(profile)
    assert result["status"] == "failed"
    check = next(
        c for c in result["checks"] if c["name"] == "step2_weekly_from_integer_counts"
    )
    assert check["fields"]["median_raw"]["max_ulps"] == 1
    assert check["passed"] is False


def test_modified_upstream_manifest_is_rejected(profile):
    path = profile.parent / "indices" / "manifest.json"
    path.write_text(path.read_text() + "\n")
    result = verify_profile(profile)
    assert result["status"] == "failed"
    assert any(
        c["name"] == "aggregate_manifest_integrity" and not c["passed"]
        for c in result["checks"]
    )


def test_changed_aggregate_counts_fail_recomputation(profile):
    path = profile.parent / "indices" / "dictionary_weekly.parquet"
    frame = pd.read_parquet(path)
    mask = (
        (frame.scope == "three_outlets")
        & frame.dictionary.str.startswith("step2_")
        & (frame.week == pd.Timestamp("2020-01-13"))
    )
    frame.loc[mask, "gpr_articles"] += 1
    frame.to_parquet(path, index=False)
    result = verify_profile(profile)
    assert result["status"] == "failed"
    assert any(
        c["name"] == "step2_weekly_from_integer_counts" and not c["passed"]
        for c in result["checks"]
    )


def test_missing_or_incomplete_manifest_reports_json_failure(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["checks"][0]["name"] == "profile_contract"
    (tmp_path / "manifest.json").write_text(
        json.dumps({"profile": PROFILE_NAME, "complete": False})
    )
    result = verify_profile(tmp_path)
    assert result["status"] == "failed"
    assert "incomplete" in result["checks"][0]["error"]


def test_command_success_returns_zero_and_json(profile, capsys):
    assert main([str(profile)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
