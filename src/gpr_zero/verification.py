"""Read-only verification of the completed historical thesis result profile.

This checks small, already-aggregated inputs. It never opens the article database,
rebuilds the corpus, fits a model, invokes an LLM, or modifies supplied artifacts.
CSV floats are parsed with round-trip precision; numerical comparisons are exact.
The one documented exception is the saved Lau threshold's adjacent float, reported
separately rather than hidden behind a general numerical tolerance.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from gpr_zero.gpr_index.spikes import detect_weekly_events, events_overlap

PROFILE_NAME = "newer_four_chapter_thesis_historical_reproduction_v1"
FULL_CACHE = "full_cache"
THREE_OUTLETS = "three_outlets"
CATEGORY_ENDPOINT = "legacy_end_midnight"
STUDY_FIRST_WEEK = "2017-01-01"
STUDY_LAST_WEEK = "2024-12-31"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _csv(path: Path, dates=()) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", parse_dates=list(dates))


def _integrity_check(name: str, path: Path, expected: dict) -> dict:
    result = {"name": name, "path": str(path), "passed": False}
    try:
        result["expected_bytes"] = expected["bytes"]
        result["actual_bytes"] = path.stat().st_size
        if result["actual_bytes"] != result["expected_bytes"]:
            result["error"] = "File size differs from the manifest"
            return result
        result["expected_sha256"] = expected["sha256"]
        result["actual_sha256"] = _digest(path)
        result["passed"] = result["actual_sha256"] == result["expected_sha256"]
        if not result["passed"]:
            result["error"] = "File SHA-256 differs from the manifest"
    except (OSError, KeyError, TypeError) as error:
        result["error"] = str(error)
    return result


def _float_metrics(expected, actual) -> dict:
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    finite = np.isfinite(expected) & np.isfinite(actual)
    equal = (expected == actual) & finite
    result = {"exact": bool(equal.all()), "different_values": int((~equal).sum())}
    if not finite.all():
        result.update(
            max_absolute_error=None, max_ulps=None, error="Non-finite numeric value"
        )
        return result
    result["max_absolute_error"] = float(np.max(np.abs(expected - actual), initial=0.0))
    # Ordered IEEE-754 encodings give ULP distance for either sign. Equal signed
    # zeroes are treated as equal values, consistent with all other comparisons.
    mask = (1 << 64) - 1
    sign = 1 << 63

    def ordered(bits):
        bits = int(bits)
        return (~bits & mask) if bits & sign else bits | sign

    distances = [
        0 if same else abs(ordered(left) - ordered(right))
        for left, right, same in zip(
            expected.view(np.uint64), actual.view(np.uint64), equal
        )
    ]
    result["max_ulps"] = max(distances, default=0)
    return result


def _compare_frames(
    name: str, expected: pd.DataFrame, actual: pd.DataFrame, keys, fields
) -> dict:
    result = {
        "name": name,
        "passed": False,
        "expected_rows": len(expected),
        "actual_rows": len(actual),
    }
    columns = list(keys) + list(fields)
    if any(column not in expected or column not in actual for column in columns):
        result["error"] = "Comparison columns are missing"
        return result
    if expected.duplicated(list(keys)).any() or actual.duplicated(list(keys)).any():
        result["error"] = "Duplicate comparison keys"
        return result
    expected = expected.set_index(list(keys)).sort_index()
    actual = actual.set_index(list(keys)).sort_index()
    result["missing_keys"] = len(expected.index.difference(actual.index))
    result["unexpected_keys"] = len(actual.index.difference(expected.index))
    if not expected.index.equals(actual.index):
        result["error"] = "Comparison key sets differ"
        return result
    metrics = {}
    for field in fields:
        left, right = expected[field], actual[field]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if pd.api.types.is_integer_dtype(left) and pd.api.types.is_integer_dtype(
                right
            ):
                equal = left.to_numpy() == right.to_numpy()
                metrics[field] = {
                    "exact": bool(equal.all()),
                    "different_values": int((~equal).sum()),
                }
            else:
                metrics[field] = _float_metrics(left, right)
        else:
            equal = (left == right).fillna(False)
            metrics[field] = {
                "exact": bool(equal.all()),
                "different_values": int((~equal).sum()),
            }
    result["fields"] = metrics
    result["passed"] = all(metric["exact"] for metric in metrics.values())
    return result


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _dictionary_names(catalog: dict, family: str) -> list[str]:
    return [
        item["name"] for item in catalog["dictionaries"] if item["family"] == family
    ]


def _counts_to_shares(frame, numerator, share):
    result = frame.copy()
    _require(
        (result["total_articles"] > 0).all(), "Aggregate denominator must be positive"
    )
    _require((result[numerator] >= 0).all(), "Aggregate numerator must be nonnegative")
    _require(
        (result[numerator] <= result["total_articles"]).all(),
        "Aggregate numerator exceeds denominator",
    )
    result[share] = result[numerator] / result["total_articles"]
    return result


def _weekly_checks(profile, dictionary_weekly, catalog):
    selected = dictionary_weekly[
        (dictionary_weekly.scope == THREE_OUTLETS)
        & (dictionary_weekly.endpoint_policy == FULL_CACHE)
    ]
    selected = _counts_to_shares(selected, "gpr_articles", "gpr_share")
    step2_names = _dictionary_names(catalog, "step2")
    _require(
        len(step2_names) == 40, "The thesis weekly index requires 40 Step2 dictionaries"
    )
    step2_rows = selected[selected.dictionary.isin(step2_names)]
    _require(
        not step2_rows.duplicated(["week", "dictionary"]).any(),
        "Duplicate weekly dictionary counts",
    )
    grouped = step2_rows.groupby("week", sort=True)
    step2 = grouped.gpr_share.median().rename("median_raw").reset_index()
    step2["dictionary_count"] = grouped.dictionary.nunique().to_numpy()
    _require(
        (step2.dictionary_count == 40).all(),
        "A Step2 week does not include all 40 dictionaries",
    )
    step2["median_scaled"] = step2.median_raw / step2.median_raw.mean() * 100
    expected_step2 = _csv(profile / "step2_weekly_index.csv", ("week",))
    checks = [
        _compare_frames(
            "step2_weekly_from_integer_counts",
            expected_step2,
            step2,
            ["week"],
            ["median_raw", "median_scaled", "dictionary_count"],
        )
    ]

    lau_names = _dictionary_names(catalog, "lau")
    _require(len(lau_names) == 1, "The thesis profile requires one Lau dictionary")
    lau = selected[selected.dictionary == lau_names[0]].sort_values("week").copy()
    lau["dictionary"] = "lau"
    lau["mean100_index"] = lau.gpr_share / statistics.mean(lau.gpr_share) * 100
    expected_lau = _csv(profile / "lau_weekly_index.csv", ("week",))
    checks.append(
        _compare_frames(
            "lau_weekly_from_integer_counts",
            expected_lau,
            lau,
            ["week"],
            [
                "dictionary",
                "total_articles",
                "gpr_articles",
                "gpr_share",
                "mean100_index",
            ],
        )
    )
    return checks, {"step2": step2, "lau": lau}


def _category_check(profile, manifest, category_weekly, catalog):
    from gpr_zero.gpr_index.series import category_medians

    policy = manifest["artifacts"]["category_medians.csv"]["selection"]
    match = re.search(r"excluding\s+(step2_[^;,\s]+)", policy)
    _require(
        match is not None,
        "Category policy must identify the historically omitted dictionary",
    )
    omission = match.group(1)
    names = _dictionary_names(catalog, "step2")
    _require(
        omission in names, "The omitted category dictionary is missing from the catalog"
    )
    selected = category_weekly[
        (category_weekly.scope == THREE_OUTLETS)
        & (category_weekly.endpoint_policy == CATEGORY_ENDPOINT)
        & category_weekly.dictionary.isin(names)
    ]
    _require(
        not selected.duplicated(["week", "category", "dictionary"]).any(),
        "Duplicate category-week counts",
    )
    selected = _counts_to_shares(selected, "category_articles", "gpr_index")
    coverage = (
        selected[selected.dictionary != omission]
        .groupby(["week", "category"])
        .dictionary.nunique()
    )
    _require(
        (coverage == 39).all(),
        "Historical category weeks require 39 dictionary observations",
    )
    # The actual category workflow and this verifier share the scientific policy.
    # Confirm the declared manifest selection agrees with that named operation.
    expected_selection = (
        selected[selected.dictionary != omission]
        .groupby(["week", "category"])
        .gpr_index.median()
        .reset_index()
    )
    computed = category_medians(selected)
    policy_check = _compare_frames(
        "declared_category_policy",
        expected_selection,
        computed,
        ["week", "category"],
        ["gpr_index"],
    )
    _require(
        policy_check["passed"],
        "Manifest category omission differs from the canonical category operation",
    )
    computed = computed.rename(columns={"gpr_index": "category_share"})
    result = _compare_frames(
        "category_medians_from_integer_counts",
        _csv(profile / "category_medians.csv", ("week",)),
        computed,
        ["week", "category"],
        ["category_share"],
    )
    result["omitted_dictionary"] = omission
    result["dictionaries_per_category"] = 39
    result["endpoint_policy"] = CATEGORY_ENDPOINT
    return result


def _pairwise_check(profile, dictionary_weekly, catalog):
    selected = dictionary_weekly[
        (dictionary_weekly.scope == "all_outlets")
        & (dictionary_weekly.endpoint_policy == FULL_CACHE)
        & (dictionary_weekly.week >= STUDY_FIRST_WEEK)
        & (dictionary_weekly.week <= STUDY_LAST_WEEK)
    ]
    selected = _counts_to_shares(selected, "gpr_articles", "gpr_share")
    rows = []
    for family, count in (("step1", 35), ("step2", 40)):
        names = _dictionary_names(catalog, family)
        _require(
            len(names) == count,
            f"Pairwise correlations require {count} {family} dictionaries",
        )
        values = (
            selected[selected.dictionary.isin(names)]
            .pivot(index="week", columns="dictionary", values="gpr_share")
            .sort_index()
        )
        _require(
            set(values.columns) == set(names),
            "Missing dictionary in correlation inputs",
        )
        matrix = values.corr().to_numpy()
        pairs = matrix[np.triu_indices_from(matrix, k=1)]
        rows.append(
            {
                "step": family,
                "scope": "all_outlets",
                "endpoint_policy": "full_cache_then_study_week_labels",
                "dictionary_count": count,
                "weeks": len(values),
                "first_week": values.index.min(),
                "last_week": values.index.max(),
                "pair_count": int(np.isfinite(pairs).sum()),
                "expected_pair_count": count * (count - 1) // 2,
                "mean_pairwise_correlation": float(pairs.mean()),
                "incomplete_weeks": int(values.isna().any(axis=1).sum()),
            }
        )
    expected = _csv(profile / "pairwise_correlations.csv", ("first_week", "last_week"))
    return _compare_frames(
        "pairwise_correlations_from_integer_counts",
        expected,
        pd.DataFrame(rows),
        ["step"],
        [column for column in expected if column != "step"],
    )


def _spike_checks(profile, weekly):
    checks, notes, all_events, thresholds = [], [], {}, {}
    for family, value in (("step2", "median_raw"), ("lau", "gpr_share")):
        frame = weekly[family].rename(columns={"week": "date"}).copy()
        events, threshold = detect_weekly_events(frame, value)
        expected = _csv(
            profile / f"{family}_spikes.csv", ("start_date", "end_date", "peak_date")
        )
        actual = pd.DataFrame(events)
        all_events[family] = events
        thresholds[family] = threshold
        fields = [
            "end_date",
            "peak_date",
            "peak_value",
            "duration_weeks",
            "mean_spike_value",
        ]
        checks.append(
            _compare_frames(
                f"{family}_spike_events", expected, actual, ["start_date"], fields
            )
        )
        metrics = _float_metrics(expected.threshold, np.full(len(expected), threshold))
        threshold_check = {
            "name": f"{family}_spike_threshold",
            "passed": metrics["exact"],
            **metrics,
            "recomputed_threshold": float(threshold),
            "saved_thresholds": expected.threshold.unique().tolist(),
        }
        if (
            family == "lau"
            and not metrics["exact"]
            and metrics["max_ulps"] == 1
            and len(expected) == 22
            and len(frame) == 442
            and checks[-1]["passed"]
            and (expected.threshold == np.nextafter(threshold, np.inf)).all()
        ):
            threshold_check["passed"] = True
            threshold_check["precision_note"] = (
                "historical_lau_threshold_adjacent_float"
            )
            notes.append(
                {
                    "name": "historical_lau_threshold_adjacent_float",
                    "detail": "The saved Lau threshold is one representable float above the recomputation. The pinned thesis implementation gives the same recomputed value on these round-trip inputs; the historical source of the difference is unproven. Every event field matches exactly.",
                    "saved": float(expected.threshold.iloc[0]),
                    "recomputed": float(threshold),
                    "max_ulps": metrics["max_ulps"],
                    "max_absolute_error": metrics["max_absolute_error"],
                }
            )
        checks.append(threshold_check)
    unique = [
        event
        for event in all_events["step2"]
        if not any(events_overlap(event, other) for other in all_events["lau"])
    ]
    expected = _csv(
        profile / "step2_unique_spikes.csv", ("start_date", "end_date", "peak_date")
    )
    checks.append(
        _compare_frames(
            "step2_unique_spike_events",
            expected,
            pd.DataFrame(unique).assign(threshold=thresholds["step2"]),
            ["start_date"],
            [
                "end_date",
                "peak_date",
                "peak_value",
                "duration_weeks",
                "mean_spike_value",
                "threshold",
            ],
        )
    )
    return checks, notes


def verify_profile(profile_directory: str | Path) -> dict:
    """Verify a thesis profile using only its files and aggregate Parquet inputs."""
    profile = Path(profile_directory).expanduser().resolve()
    report = {
        "status": "failed",
        "profile_directory": str(profile),
        "checks": [],
        "precision_notes": [],
        "scope": "Artifact integrity and recomputation from existing integer weekly aggregates; no article database or corpus scan",
        "not_recomputed": [
            "article parsing and matching",
            "word tables",
            "newspaper word ratios",
            "VAR responses and confidence bands",
            "figures",
        ],
    }
    try:
        manifest = _json(profile / "manifest.json")
        report["profile"] = manifest.get("profile")
        _require(manifest.get("profile") == PROFILE_NAME, "Unsupported thesis profile")
        _require(
            manifest.get("complete") is True and manifest.get("status") == "complete",
            "Thesis profile is incomplete",
        )
        _require(
            manifest.get("calendar_timezone") == "Asia/Taipei",
            "Thesis profile requires the Asia/Taipei calendar",
        )
        report["all_thesis_claims_reproduced"] = manifest.get(
            "all_thesis_claims_reproduced", False
        )
        artifacts = manifest["artifacts"]
        _require(
            isinstance(artifacts, dict) and bool(artifacts),
            "Profile artifacts are missing",
        )
        for name, metadata in artifacts.items():
            _require(
                Path(name).name == name, "Profile artifact names must be filenames"
            )
            report["checks"].append(
                _integrity_check(f"artifact:{name}", profile / name, metadata)
            )
        _require(
            len(artifacts) == 24,
            "The completed thesis profile must declare all 24 artifacts",
        )
        if not all(check["passed"] for check in report["checks"]):
            return report
        rebuild = Path(manifest["rebuild_dir"])
        aggregate_path = rebuild / "indices" / "manifest.json"
        bindings = {
            str(Path(item["path"]).resolve()): item for item in manifest["inputs"]
        }
        _require(
            str(aggregate_path.resolve()) in bindings,
            "Aggregate manifest is not bound by the profile",
        )
        binding = _integrity_check(
            "aggregate_manifest_integrity",
            aggregate_path,
            bindings[str(aggregate_path.resolve())],
        )
        report["checks"].append(binding)
        _require(binding["passed"], "Aggregate manifest integrity check failed")
        aggregate = _json(aggregate_path)
        _require(
            aggregate.get("status") == "complete"
            and aggregate.get("source_mode") == "full"
            and all(
                aggregate.get(key) is True
                for key in ("complete", "input_complete", "source_complete")
            ),
            "A completed full aggregation is required",
        )
        _require(
            aggregate.get("calendar_timezone") == "Asia/Taipei",
            "Aggregate calendar differs from the thesis profile",
        )
        catalog = aggregate["dictionary_catalog"]
        directory = aggregate_path.parent
        dictionary_weekly = pd.read_parquet(
            directory / aggregate["outputs"]["dictionary_weekly"]["parquet"]
        )
        category_weekly = pd.read_parquet(
            directory / aggregate["outputs"]["category_weekly"]["parquet"]
        )
        checks, weekly = _weekly_checks(profile, dictionary_weekly, catalog)
        report["checks"].extend(checks)
        report["checks"].append(
            _category_check(profile, manifest, category_weekly, catalog)
        )
        report["checks"].append(_pairwise_check(profile, dictionary_weekly, catalog))
        checks, notes = _spike_checks(profile, weekly)
        report["checks"].extend(checks)
        report["precision_notes"].extend(notes)
        report["status"] = (
            "passed" if all(check["passed"] for check in report["checks"]) else "failed"
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        report["checks"].append(
            {"name": "profile_contract", "passed": False, "error": str(error)}
        )
    return report
