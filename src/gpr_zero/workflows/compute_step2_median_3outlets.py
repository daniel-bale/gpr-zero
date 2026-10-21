#!/usr/bin/env python3
"""
Compute median GPR index across Step2 dictionaries using 3-outlet data.
"""

import logging

import numpy as np
import pandas as pd

from gpr_zero.project import get_project

logger = logging.getLogger(__name__)


def main(project=None):
    """Compute median GPR index across Step2 dictionaries."""
    project = project or get_project()

    # Setup paths
    timeseries_dir = project.data_path("gpr_timeseries_3outlets")
    output_dir = project.data_path("gpr_indices_3outlets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all Step2 dictionary files
    step2_files = list(timeseries_dir.glob("step2_*_W_overall_3outlets.parquet"))
    logger.info(f"Found {len(step2_files)} Step2 dictionary files")

    # Load all Step2 time series
    all_series = []
    for file in sorted(step2_files):
        df = pd.read_parquet(file)
        # Keep only week and gpr_index columns
        df = df[["week", "gpr_index"]]
        df["dictionary"] = file.stem.replace("_W_overall_3outlets", "")
        all_series.append(df)

    # Combine all series
    combined_df = pd.concat(all_series, ignore_index=True)
    logger.info(
        f"Combined {len(combined_df)} rows from {len(step2_files)} dictionaries"
    )

    # Calculate median GPR index per week
    median_gpr = (
        combined_df.groupby("week")["gpr_index"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
    )
    median_gpr.rename(columns={"median": "gpr_index"}, inplace=True)

    # Scale to mean=100
    mean_value = median_gpr["gpr_index"].mean()
    median_gpr["gpr_index_scaled"] = (median_gpr["gpr_index"] / mean_value) * 100

    # Save results
    output_file = output_dir / "step2_median_gpr_3outlets.parquet"
    median_gpr.to_parquet(output_file)
    logger.info(f"Saved median GPR index to {output_file}")

    # Also save CSV for easy viewing
    csv_file = output_dir / "step2_median_gpr_3outlets.csv"
    median_gpr.to_csv(csv_file, index=False)
    logger.info(f"Saved CSV to {csv_file}")

    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")
    logger.info(
        f"Time period: {median_gpr['week'].min()} to {median_gpr['week'].max()}"
    )
    logger.info(f"Number of weeks: {len(median_gpr)}")
    logger.info(f"Mean GPR index (raw): {mean_value:.6f}")
    logger.info(f"Mean GPR index (scaled): {median_gpr['gpr_index_scaled'].mean():.2f}")
    logger.info(f"Min GPR index (scaled): {median_gpr['gpr_index_scaled'].min():.2f}")
    logger.info(f"Max GPR index (scaled): {median_gpr['gpr_index_scaled'].max():.2f}")

    # Calculate pairwise correlations
    logger.info("\n=== Pairwise Correlations ===")

    # Step2 pairwise correlations
    step2_pivot = combined_df[combined_df["dictionary"].str.contains("step2_")].pivot(
        index="week", columns="dictionary", values="gpr_index"
    )
    step2_corr_matrix = step2_pivot.corr()
    # Get upper triangle values (excluding diagonal)
    mask = np.triu(np.ones_like(step2_corr_matrix, dtype=bool), k=1)
    step2_correlations = step2_corr_matrix.where(mask).stack().values
    step2_mean_corr = step2_correlations.mean()
    logger.info(
        f"Mean pairwise correlation among {len(step2_files)} Step2 indices: {step2_mean_corr:.3f}"
    )

    # Step1 pairwise correlations
    step1_files = list(timeseries_dir.glob("step1_*_W_overall_3outlets.parquet"))
    if step1_files:
        step1_series = []
        for file in sorted(step1_files):
            df = pd.read_parquet(file)
            df = df[["week", "gpr_index"]]
            df["dictionary"] = file.stem.replace("_W_overall_3outlets", "")
            step1_series.append(df)

        step1_combined = pd.concat(step1_series, ignore_index=True)
        step1_pivot = step1_combined.pivot(
            index="week", columns="dictionary", values="gpr_index"
        )
        step1_corr_matrix = step1_pivot.corr()
        mask = np.triu(np.ones_like(step1_corr_matrix, dtype=bool), k=1)
        step1_correlations = step1_corr_matrix.where(mask).stack().values
        step1_mean_corr = step1_correlations.mean()
        logger.info(
            f"Mean pairwise correlation among {len(step1_files)} Step1 indices: {step1_mean_corr:.3f}"
        )

    # Correlation with Lau's index
    lau_file = (
        timeseries_dir / "lau_overall_taiwan_recalculated_W_overall_3outlets.parquet"
    )
    if lau_file.exists():
        lau_df = pd.read_parquet(lau_file)
        # Merge on week
        merged = median_gpr.merge(
            lau_df[["week", "gpr_index"]], on="week", suffixes=("_step2", "_lau")
        )
        lau_corr = merged["gpr_index_step2"].corr(merged["gpr_index_lau"])
        logger.info(f"Correlation between Step2 median and Lau's index: {lau_corr:.3f}")
