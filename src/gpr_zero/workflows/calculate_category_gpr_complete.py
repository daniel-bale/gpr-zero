#!/usr/bin/env python3
"""
Complete implementation: Calculate category-specific GPR indices.
For each Step2 dictionary and the 8 thesis categories:
1. Calculate weekly GPR index for that category
2. Take each category median using the selected 39-dictionary thesis profile
3. Scale to mean=100 and plot
"""

import logging

import pandas as pd
from tqdm import tqdm

from gpr_zero.constants import STANDARD_CATEGORIES
from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory
from gpr_zero.gpr_index.dictionary import DictionaryMatcher, normalize_categories
from gpr_zero.gpr_index.series import category_medians
from gpr_zero.project import get_project

logger = logging.getLogger(__name__)


def calculate_category_gpr_for_dictionary(corpus, dictionary, category):
    """Calculate a category's weekly share within the corpus article cohort."""
    table = corpus.table
    category_expr = DictionaryMatcher(dictionary).category_match(
        table, corpus.word_columns, category
    )
    if category_expr is None:
        return None
    with_category = table.mutate(is_category_match=category_expr.cast("int32"))

    # Group by week and calculate GPR
    # Cast string date to date type first
    weekly = with_category.group_by(
        with_category.warc_date.cast("date").truncate("W").name("week")
    ).aggregate(
        category_articles=with_category.is_category_match.sum(),
        total_articles=with_category.count(),
    )

    # Execute and convert to pandas
    result = weekly.execute()

    # Calculate GPR index
    result["gpr_index"] = result["category_articles"] / result["total_articles"]
    result["gpr_index"] = result["gpr_index"].fillna(0)

    # Sort by week
    result = result.sort_values("week")

    return result[["week", "gpr_index"]]


def main(project=None):
    project = project or get_project()
    output_dir = project.data_path("category_gpr_indices")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Calculate Category-Specific GPR Indices ===")
    inventory = DictionaryInventory.thesis_category_directory(project.dictionaries_dir)
    step2_inventory = DictionaryInventory.thesis_category_directory(
        project.step2_dictionaries_dir
    )
    step2_dicts = {
        name: normalize_categories(dictionary)
        for name, dictionary in step2_inventory.dictionaries.items()
    }
    conn = project.connect()
    corpus = ArticleCorpus.from_cache(conn, inventory).three_outlets().study_period()

    # Calculate GPR for each dictionary and category
    all_results = []

    for dict_name, dictionary in tqdm(
        step2_dicts.items(), desc="Processing dictionaries"
    ):
        for category in STANDARD_CATEGORIES:
            result = calculate_category_gpr_for_dictionary(corpus, dictionary, category)

            if result is not None:
                result["dictionary"] = dict_name
                result["category"] = category
                all_results.append(result)

    # Combine all results
    logger.info("Combining results...")
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save raw results
    combined_df.to_csv(output_dir / "all_category_gpr_indices.csv", index=False)
    logger.info(f"Saved raw results to {output_dir / 'all_category_gpr_indices.csv'}")

    # Calculate median for each category
    logger.info("Calculating median indices...")
    median_indices = category_medians(combined_df)

    # Pivot to get categories as columns
    pivot_df = median_indices.pivot(
        index="week", columns="category", values="gpr_index"
    )

    # Save unscaled median
    pivot_df.to_csv(output_dir / "category_median_gpr_unscaled.csv")

    # Scale each category to mean=100
    scaled_df = pivot_df.copy()
    for col in scaled_df.columns:
        col_mean = scaled_df[col].mean()
        if col_mean > 0:
            scaled_df[col] = (scaled_df[col] / col_mean) * 100

    # Save scaled median
    scaled_df.to_csv(output_dir / "category_median_gpr_scaled.csv")

    import matplotlib.pyplot as plt

    from gpr_zero.utils.font_setup import ensure_cjk_font_setup

    # Plot the results
    logger.info("Creating plots...")
    ensure_cjk_font_setup()

    # Plot 1: Scaled category indices
    fig, ax = plt.subplots(figsize=(14, 8))

    category_colors = {
        "Military Escalation": "#FF6B6B",
        "Diplomatic Isolation": "#4ECDC4",
        "Economic Coercion": "#45B7D1",
        "Regional Security": "#F7B801",
        "Internal Political Instability": "#95E1D3",
        "Gray Zone Operations": "#DDA0DD",
        "US-China-Taiwan Relations": "#98D8C8",
        "Cyber and Information Warfare": "#FFB6C1",
    }

    for category in STANDARD_CATEGORIES:
        if category in scaled_df.columns:
            ax.plot(
                scaled_df.index,
                scaled_df[category],
                label=category,
                color=category_colors[category],
                linewidth=2,
                alpha=0.8,
            )

    ax.axhline(100, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Category-Specific GPR Index (Mean = 100)")
    ax.set_title("Taiwan GPR Index by Category (Step2 Median, Scaled)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = output_dir / "category_gpr_indices_scaled.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved plot to %s", plot_path)

    # Find and save peak weeks
    logger.info("Finding peak weeks...")
    peak_weeks = []

    for category in STANDARD_CATEGORIES:
        if category in scaled_df.columns:
            # Find week with maximum scaled GPR
            max_idx = scaled_df[category].idxmax()
            max_value = scaled_df.loc[max_idx, category]

            # Also get unscaled value
            unscaled_value = pivot_df.loc[max_idx, category]

            peak_weeks.append(
                {
                    "category": category,
                    "peak_week": max_idx,
                    "scaled_peak_value": max_value,
                    "unscaled_peak_value": unscaled_value,
                    "scaled_mean": 100.0,
                }
            )

    peak_df = pd.DataFrame(peak_weeks)
    peak_df.to_csv(output_dir / "category_peak_weeks.csv", index=False)
    logger.info(f"Saved peak weeks to {output_dir / 'category_peak_weeks.csv'}")

    # Print summary
    logger.info("\n=== Peak Weeks Summary ===")
    for _, row in peak_df.iterrows():
        logger.info(
            f"{row['category']}: {row['peak_week'].strftime('%Y-%m-%d')} "
            f"(scaled: {row['scaled_peak_value']:.1f}, unscaled: {row['unscaled_peak_value']:.1f})"
        )

    conn.disconnect()
    logger.info("\nDone! Next steps:")
    logger.info("1. Update analyze_category_peak_events.py to use these peak weeks")
    logger.info("2. Re-run with proper category-specific article sampling")
