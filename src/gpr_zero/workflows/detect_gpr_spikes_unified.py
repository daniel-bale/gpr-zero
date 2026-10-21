#!/usr/bin/env python3
"""
Unified spike detection script for both Step2 median and Lau's GPR indices.
Uses the same methodology as capture_gpr_spikes.py (mean + 1*std threshold).
Includes validation against existing Lau spike results.
"""

import json
import logging

import pandas as pd
import polars as pl

from gpr_zero.gpr_index.spikes import compare_peak_events, detect_weekly_events
from gpr_zero.project import get_project

logger = logging.getLogger(__name__)


def load_gpr_timeseries(file_path):
    """Load GPR timeseries data from parquet or CSV."""
    if str(file_path).endswith(".parquet"):
        df = pl.read_parquet(file_path)
    else:
        df = pl.read_csv(file_path, try_parse_dates=True)

    # Ensure we have a date column
    if "warc_date_bucketed" in df.columns:
        df = df.rename({"warc_date_bucketed": "date"})
    elif "week" in df.columns:
        df = df.rename({"week": "date"})
    elif "date" not in df.columns:
        raise ValueError(f"No date column found in {file_path}")

    # Convert to pandas for easier manipulation
    return df.to_pandas()


def detect_spikes(df, value_column, name_prefix=""):
    """Compatibility entry point for unified weekly event detection."""
    events, threshold = detect_weekly_events(df, value_column)
    logger.info(
        "%sMean: %.6f, Std: %.6f, Threshold: %.6f",
        name_prefix,
        df[value_column].mean(),
        df[value_column].std(),
        threshold,
    )
    logger.info("%sDetected %s spike events", name_prefix, len(events))
    return events, threshold


# Existing imports keep working while the event policy has one owner.
compare_spike_events = compare_peak_events


def main():
    """Main execution function."""

    logger.info("=" * 80)
    logger.info("TASK 2: Unified GPR Spike Detection")
    logger.info("=" * 80)

    project = get_project()
    project_root = project.root

    # Load Step2 median GPR with 3-outlet filter
    logger.info("\n1. Loading Step2 median GPR index (3-outlet filtered)...")
    step2_file = project.data_path(
        "gpr_indices_3outlets", "step2_median_gpr_3outlets.csv"
    )
    step2_df = pd.read_csv(step2_file, parse_dates=["week"])
    step2_df.rename(columns={"week": "date"}, inplace=True)

    # Use the raw gpr_index for spike detection
    step2_df["overall_share"] = step2_df["gpr_index"]

    logger.info(f"Loaded Step2 data: {len(step2_df)} weeks")
    logger.info(f"Using 3-outlet filtered data from: {step2_file}")

    # Load Lau's GPR with 3-outlet filter
    logger.info("\n2. Loading Lau's GPR index (3-outlet filtered)...")
    lau_file = project.data_path(
        "gpr_timeseries_3outlets",
        "lau_overall_taiwan_recalculated_W_overall_3outlets.parquet",
    )
    lau_df = load_gpr_timeseries(lau_file)

    # For Lau data, use gpr_index as overall_share
    lau_df["overall_share"] = lau_df["gpr_index"]

    logger.info(f"Loaded Lau's data: {len(lau_df)} weeks")

    # Detect spikes for Step2
    logger.info("\n3. Detecting spikes in Step2 median GPR...")
    step2_spikes, step2_threshold = detect_spikes(
        step2_df,
        "overall_share",  # Use unscaled share
        name_prefix="Step2: ",
    )

    # Detect spikes for Lau
    logger.info("\n4. Detecting spikes in Lau's GPR...")
    lau_spikes, lau_threshold = detect_spikes(
        lau_df, "overall_share", name_prefix="Lau: "
    )

    # Load old Lau results for validation
    logger.info("\n5. Loading old Lau spike results for validation...")
    old_lau_file = (
        project_root
        / "data_analysis"
        / "output"
        / "gpr_spikes"
        / "yeu_2024_20250715160933"
        / "detected_spike_events_with_llm_all_categories_2017_01_to_2024_12.csv"
    )

    # Try to read with error handling for malformed CSV
    try:
        old_lau_df = pd.read_csv(old_lau_file, on_bad_lines="skip")
    except Exception as e:
        logger.warning(
            f"Failed to read CSV with skip, trying with error_bad_lines: {e}"
        )
        try:
            # Try reading line by line to find the issue
            old_lau_df = pd.read_csv(
                old_lau_file, encoding="utf-8", quoting=1
            )  # QUOTE_MINIMAL
        except:
            # Last resort - read specific columns only
            old_lau_df = pd.read_csv(
                old_lau_file,
                usecols=[
                    "gpr_category",
                    "newspaper",
                    "start_date",
                    "end_date",
                    "peak_date",
                    "peak_gpr_share",
                ],
                on_bad_lines="skip",
            )

    # Filter for overall category and Overall Taiwan
    # Based on exploration, the correct name is "Overall Taiwan - Overall"
    old_lau_overall = old_lau_df[
        (old_lau_df["gpr_category"] == "overall")
        & (old_lau_df["newspaper"] == "Overall Taiwan - Overall")
    ].copy()

    logger.info(f"Found {len(old_lau_overall)} old Lau spike events for validation")

    if len(old_lau_overall) == 0:
        logger.error("No validation data found! Checking full dataframe structure...")
        logger.info(f"Unique categories: {old_lau_df['gpr_category'].unique()}")
        logger.info(f"Unique newspapers: {old_lau_df['newspaper'].unique()}")
        logger.info("First few rows of old data:")
        logger.info(old_lau_df[["gpr_category", "newspaper"]].head(10))

    # Compare results
    logger.info("\n6. Comparing new Lau results with old results...")
    comparison = compare_spike_events(lau_spikes, old_lau_overall)

    # Print comparison summary
    matched = [c for c in comparison if c["status"] == "MATCHED"]
    new_only = [c for c in comparison if c["status"] == "NEW_ONLY"]
    old_only = [c for c in comparison if c["status"] == "OLD_ONLY"]

    logger.info("\nValidation Results:")
    logger.info(f"- Matched spikes: {len(matched)}")
    logger.info(f"- New only: {len(new_only)}")
    logger.info(f"- Old only: {len(old_only)}")

    if len(matched) > 0:
        logger.info("\nMatched spikes (first 5):")
        for match in matched[:5]:
            logger.info(
                f"  Peak: {match['new_peak_date'].strftime('%Y-%m-%d')} "
                f"(diff: {match['peak_date_diff_days']} days), "
                f"Value: {match['new_peak_value']:.6f} vs {match['old_peak_value']:.6f}"
            )

    # Save results
    output_dir = project.data_path("gpr_spikes", "unified_detection_3outlets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Step2 spikes
    step2_spikes_df = pd.DataFrame(step2_spikes)
    step2_spikes_df["threshold"] = step2_threshold
    step2_spikes_df["index_type"] = "Step2_median"
    step2_output = output_dir / "step2_median_spikes.csv"
    step2_spikes_df.to_csv(step2_output, index=False)
    logger.info(f"\nSaved Step2 spikes to {step2_output}")

    # Save Lau spikes
    lau_spikes_df = pd.DataFrame(lau_spikes)
    lau_spikes_df["threshold"] = lau_threshold
    lau_spikes_df["index_type"] = "Lau_2024"
    lau_output = output_dir / "lau_2024_spikes.csv"
    lau_spikes_df.to_csv(lau_output, index=False)
    logger.info(f"Saved Lau spikes to {lau_output}")

    # Save comparison results
    comparison_df = pd.DataFrame(comparison)
    comparison_output = output_dir / "lau_validation_comparison.csv"
    comparison_df.to_csv(comparison_output, index=False)
    logger.info(f"Saved validation comparison to {comparison_output}")

    # Create summary report
    summary = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "step2_spikes": {
            "count": len(step2_spikes),
            "threshold": float(step2_threshold),
            "mean": float(step2_df["overall_share"].mean()),
            "std": float(step2_df["overall_share"].std()),
        },
        "lau_spikes": {
            "count": len(lau_spikes),
            "threshold": float(lau_threshold),
            "mean": float(lau_df["overall_share"].mean()),
            "std": float(lau_df["overall_share"].std()),
        },
        "validation": {
            "matched": len(matched),
            "new_only": len(new_only),
            "old_only": len(old_only),
            "match_rate": len(matched) / len(old_lau_overall)
            if len(old_lau_overall) > 0
            else 0,
        },
    }

    summary_file = output_dir / "spike_detection_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSaved summary to {summary_file}")

    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("SPIKE DETECTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Step2 Median: {len(step2_spikes)} spikes detected")
    logger.info(f"Lau 2024: {len(lau_spikes)} spikes detected")
    logger.info(
        f"Validation: {len(matched)}/{len(old_lau_overall)} matched ({summary['validation']['match_rate']:.1%})"
    )

    if summary["validation"]["match_rate"] < 0.8:
        logger.warning(
            "\nWARNING: Low match rate with old results! Check for data or methodology differences."
        )
    else:
        logger.info("\n✓ Validation successful! New results match old results well.")

    logger.info("\n✓ Task 2 spike detection complete!")
