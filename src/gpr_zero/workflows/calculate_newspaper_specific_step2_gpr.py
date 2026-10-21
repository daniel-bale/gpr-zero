#!/usr/bin/env python3
"""
Calculate newspaper-specific GPR indices using Step2 dictionaries
and analyze unique spikes with top contributing words.
"""

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from gpr_zero.constants import OUTLET_CHINESE_TO_CODE as NEWSPAPERS
from gpr_zero.constants import OUTLET_ENGLISH_TO_CODE as OUTLET_ENGLISH
from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory
from gpr_zero.gpr_index.spikes import (
    detect_newspaper_events,
    unique_events_by_newspaper,
)
from gpr_zero.project import get_project

logger = logging.getLogger(__name__)


def calculate_newspaper_gpr_indices():
    """Calculate GPR indices for each newspaper using Step2 dictionaries."""

    project = get_project()
    output_dir = project.data_path("newspaper_gpr_step2")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not project.database.exists():
        logger.error("Database not found: %s", project.database)
        return

    # The inventory owns canonical cache column ordering for every dictionary.
    inventory = DictionaryInventory.from_directory(project.dictionaries_dir)
    step2_files = list(project.dictionaries_dir.glob("step2_*.json"))
    if not step2_files:
        logger.error("No Step2 dictionary files found in %s", project.dictionaries_dir)
        return
    conn = project.connect()
    corpus = ArticleCorpus.from_cache(conn, inventory)
    calculator = corpus.calculator()

    # Process all newspapers at once by grouping by outlet
    logger.info("Calculating GPR for all newspapers with outlet grouping...")

    all_newspaper_data = []

    # Process each dictionary
    for dict_file in tqdm(step2_files, desc="Processing dictionaries"):
        dict_name = dict_file.stem

        try:
            # Query for this dictionary grouped by outlet
            ts = calculator.query(
                dictionary_name=dict_name,
                group_by_cols=["outlet"],  # Group by outlet
                time_bucket="W",
                output_format="wide",
            )

            # Convert to pandas and add dictionary info
            ts_df = ts.to_pandas()
            ts_df["dictionary"] = dict_name

            all_newspaper_data.append(ts_df)

        except Exception as e:
            logger.error(f"Failed to process {dict_name}: {e}")
            continue

    if not all_newspaper_data:
        logger.error("No data collected")
        return {}

    # Combine all dictionary results
    combined_df = pd.concat(all_newspaper_data, ignore_index=True)

    # Map English outlet names to short codes
    combined_df["outlet"] = combined_df["outlet"].map(OUTLET_ENGLISH)

    # Rename columns from query output to match script expectations
    col_renames = {
        "warc_date_bucketed": "date_bucket",
        "total_gpr_share": "overall_share",
    }
    combined_df = combined_df.rename(
        columns={k: v for k, v in col_renames.items() if k in combined_df.columns}
    )

    # Map outlet codes to Chinese names
    outlet_to_chinese = {v: k for k, v in NEWSPAPERS.items()}
    combined_df["newspaper_zh"] = combined_df["outlet"].map(outlet_to_chinese)

    # Filter to keep only our target newspapers
    combined_df = combined_df[combined_df["newspaper_zh"].notna()]

    # Process each newspaper
    newspaper_results = {}

    for newspaper_zh, newspaper_code in NEWSPAPERS.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing newspaper: {newspaper_zh} ({newspaper_code})")
        logger.info(f"{'=' * 60}")

        # Filter data for this newspaper
        newspaper_df = combined_df[combined_df["outlet"] == newspaper_code]

        if newspaper_df.empty:
            logger.error(f"No data for {newspaper_zh}")
            continue

        # Calculate median and confidence intervals
        logger.info(f"Calculating median GPR for {newspaper_zh}...")

        # Group by date and calculate statistics
        weekly_stats = newspaper_df.groupby("date_bucket")["overall_share"].agg(
            [
                "mean",
                "median",
                "std",
                "count",
                lambda x: np.percentile(x, 16),  # 68% CI lower
                lambda x: np.percentile(x, 84),  # 68% CI upper
                lambda x: np.percentile(x, 2.5),  # 95% CI lower
                lambda x: np.percentile(x, 97.5),  # 95% CI upper
            ]
        )

        weekly_stats.columns = [
            "mean",
            "median",
            "std",
            "count",
            "ci68_lower",
            "ci68_upper",
            "ci95_lower",
            "ci95_upper",
        ]

        # Scale to mean=100
        scale_factor = 100 / weekly_stats["median"].mean()
        weekly_stats["median_scaled"] = weekly_stats["median"] * scale_factor
        weekly_stats["mean_scaled"] = weekly_stats["mean"] * scale_factor
        weekly_stats["ci68_lower_scaled"] = weekly_stats["ci68_lower"] * scale_factor
        weekly_stats["ci68_upper_scaled"] = weekly_stats["ci68_upper"] * scale_factor
        weekly_stats["ci95_lower_scaled"] = weekly_stats["ci95_lower"] * scale_factor
        weekly_stats["ci95_upper_scaled"] = weekly_stats["ci95_upper"] * scale_factor

        # Save results
        output_file = output_dir / f"{newspaper_code}_step2_median_gpr.csv"
        weekly_stats.to_csv(output_file)
        logger.info(f"Saved {newspaper_zh} median GPR to {output_file}")

        newspaper_results[newspaper_zh] = weekly_stats

        # Also save the raw data for this newspaper
        raw_output_file = (
            output_dir / f"{newspaper_code}_step2_all_dictionaries.parquet"
        )
        newspaper_df.to_parquet(raw_output_file)
        logger.info(f"Saved raw data to {raw_output_file}")

    # Create comparison plot
    create_newspaper_comparison_plot(newspaper_results, output_dir)

    # Calculate correlations between newspapers
    calculate_newspaper_correlations(newspaper_results, output_dir)

    return newspaper_results


def create_newspaper_comparison_plot(newspaper_results, output_dir):
    """Create a comparison plot of newspaper-specific GPR indices."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (newspaper_zh, stats_df) in enumerate(newspaper_results.items()):
        color = colors[idx % len(colors)]

        # Reset index to get date column
        plot_df = stats_df.reset_index()
        dates = pd.to_datetime(plot_df["date_bucket"])

        # Create complete date range for missing data handling
        date_range = pd.date_range(start=dates.min(), end=dates.max(), freq="W-MON")
        complete_df = pd.DataFrame({"Date": date_range})
        actual_df = pd.DataFrame(
            {
                "Date": dates,
                "median_scaled": plot_df["median_scaled"],
                "ci68_lower_scaled": plot_df["ci68_lower_scaled"],
                "ci68_upper_scaled": plot_df["ci68_upper_scaled"],
            }
        )
        plot_df_complete = complete_df.merge(actual_df, on="Date", how="left")

        # Plot median line with missing segments
        missing_segments = plot_df_complete["median_scaled"].isna().cumsum()

        for segment in plot_df_complete.groupby(missing_segments):
            if not segment[1]["median_scaled"].isna().all():
                ax.plot(
                    segment[1]["Date"],
                    segment[1]["median_scaled"],
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    label=newspaper_zh if segment[0] == 0 else "",
                )

        # Add confidence bands
        ax.fill_between(
            plot_df_complete["Date"],
            plot_df_complete["ci68_lower_scaled"],
            plot_df_complete["ci68_upper_scaled"],
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("GPR Index (Mean = 100)")
    ax.set_title("Newspaper-Specific GPR Indices with Step2 Median Approach")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "newspaper_step2_gpr_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison plot to {output_file}")


def calculate_newspaper_correlations(newspaper_results, output_dir):
    """Calculate correlation matrix between newspaper GPR indices."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a combined dataframe with all newspapers
    combined_data = {}

    for newspaper_zh, stats_df in newspaper_results.items():
        combined_data[newspaper_zh] = stats_df["median_scaled"]

    # Create correlation matrix
    corr_df = pd.DataFrame(combined_data)
    corr_matrix = corr_df.corr()

    # Save correlation matrix
    corr_file = output_dir / "newspaper_correlations.csv"
    corr_matrix.to_csv(corr_file)
    logger.info(f"Saved correlation matrix to {corr_file}")

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0.5,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix of Newspaper GPR Indices")

    plt.tight_layout()
    heatmap_file = output_dir / "newspaper_correlations_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved correlation heatmap to {heatmap_file}")

    # Print correlation summary
    logger.info("\nCorrelation Summary:")
    for i, newspaper1 in enumerate(corr_matrix.index):
        for j, newspaper2 in enumerate(corr_matrix.columns):
            if i < j:  # Upper triangle only
                corr_value = corr_matrix.loc[newspaper1, newspaper2]
                logger.info(f"{newspaper1} - {newspaper2}: {corr_value:.3f}")


def detect_newspaper_spikes(newspaper_results, output_dir):
    """Detect spikes for each newspaper's GPR index."""

    spike_results = {}

    for newspaper_zh, stats_df in newspaper_results.items():
        logger.info(f"\nDetecting spikes for {newspaper_zh}...")

        # Calculate mean and std
        mean_val = stats_df["median_scaled"].mean()
        std_val = stats_df["median_scaled"].std()
        threshold = mean_val + std_val

        logger.info(
            f"Mean: {mean_val:.2f}, Std: {std_val:.2f}, Threshold: {threshold:.2f}"
        )

        spike_groups = detect_newspaper_events(stats_df)

        spike_results[newspaper_zh] = spike_groups
        logger.info(f"Found {len(spike_groups)} spike periods for {newspaper_zh}")

    # Save spike results
    spike_summary = []
    for newspaper_zh, spikes in spike_results.items():
        for spike in spikes:
            spike_summary.append(
                {
                    "newspaper": newspaper_zh,
                    "start_date": spike["start_date"],
                    "end_date": spike["end_date"],
                    "peak_date": spike["peak_date"],
                    "peak_value": spike["peak_value"],
                    "duration_weeks": spike["weeks"],
                }
            )

    spike_df = pd.DataFrame(spike_summary)
    spike_file = output_dir / "newspaper_spikes.csv"
    spike_df.to_csv(spike_file, index=False)
    logger.info(f"Saved spike summary to {spike_file}")

    return spike_results


def analyze_unique_spikes(spike_results, output_dir):
    """Find unique spikes for each newspaper."""

    unique_spikes = unique_events_by_newspaper(spike_results)

    # Save unique spikes summary
    unique_summary = []
    for newspaper, spikes in unique_spikes.items():
        logger.info(f"\n{newspaper} has {len(spikes)} unique spikes:")
        for spike in spikes:
            logger.info(
                f"  {spike['start'].strftime('%Y-%m-%d')} to {spike['end'].strftime('%Y-%m-%d')} (peak: {spike['peak_value']:.2f})"
            )
            unique_summary.append(
                {
                    "newspaper": newspaper,
                    "start_date": spike["start"],
                    "end_date": spike["end"],
                    "peak_value": spike["peak_value"],
                }
            )

    unique_df = pd.DataFrame(unique_summary)
    unique_file = output_dir / "newspaper_unique_spikes.csv"
    unique_df.to_csv(unique_file, index=False)
    logger.info(f"\nSaved unique spikes to {unique_file}")

    return unique_spikes


def analyze_spike_words(unique_spikes, output_dir):
    """Analyze top contributing words for each newspaper's unique spikes."""

    logger.info("\n" + "=" * 80)
    logger.info("ANALYZING TOP WORDS FOR UNIQUE SPIKES")
    logger.info("=" * 80)

    project = get_project()
    conn = project.connect()
    inventory = DictionaryInventory.from_directory(project.dictionaries_dir)
    corpus = ArticleCorpus.newspaper_spike_cache(conn, inventory)
    binary_table = corpus.table

    # Process each newspaper's unique spikes
    word_analysis_results = {}

    for newspaper_zh, spikes in unique_spikes.items():
        if not spikes:
            logger.info(
                f"\nNo unique spikes for {newspaper_zh}, skipping word analysis"
            )
            continue

        logger.info(f"\nAnalyzing words for {newspaper_zh} unique spikes...")
        newspaper_code = NEWSPAPERS[newspaper_zh]

        # Combine all spike periods for this newspaper
        spike_words = {}
        total_spike_articles = 0

        for spike in spikes:
            # Get articles during this spike period
            # Map Chinese name to English outlet name used in binary matrix
            code_to_english = {v: k for k, v in OUTLET_ENGLISH.items()}
            outlet_english = code_to_english.get(newspaper_code, newspaper_zh)
            # The newspaper thesis workflow intentionally uses stored week
            # starts here. Other article workflows use article_window (+6 days).
            spike_articles = binary_table.filter(
                (binary_table.outlet == outlet_english)
                & (binary_table.warc_date >= spike["start"])
                & (binary_table.warc_date <= spike["end"])
            )

            # Count word occurrences
            word_cols = [
                col for col in binary_table.columns if col.startswith("contains_word_")
            ]

            for word_col in word_cols:
                word_count = spike_articles[word_col].sum().execute()
                if word_count > 0:
                    word = word_col.replace("word_", "")
                    spike_words[word] = spike_words.get(word, 0) + word_count

            total_spike_articles += spike_articles.total_articles.sum().execute()

        # Sort words by frequency
        top_words = sorted(spike_words.items(), key=lambda x: x[1], reverse=True)[:50]

        # Save word analysis
        word_df = pd.DataFrame(top_words, columns=["word", "count"])
        word_df["newspaper"] = newspaper_zh
        word_df["total_spike_articles"] = total_spike_articles
        word_df["frequency"] = word_df["count"] / total_spike_articles

        word_file = output_dir / f"{newspaper_code}_unique_spike_words.csv"
        word_df.to_csv(word_file, index=False)
        logger.info(f"Saved top words for {newspaper_zh} to {word_file}")

        # Print top 10 words
        logger.info(f"\nTop 10 words for {newspaper_zh} unique spikes:")
        for word, count in top_words[:10]:
            freq = count / total_spike_articles
            logger.info(f"  {word}: {count} ({freq:.1%})")

        word_analysis_results[newspaper_zh] = word_df

    # Create visualization of top words
    create_word_comparison_plot(word_analysis_results, output_dir)

    return word_analysis_results


def create_word_comparison_plot(word_analysis_results, output_dir):
    """Create visualization comparing top words across newspapers."""
    import matplotlib.pyplot as plt

    if not word_analysis_results:
        logger.warning("No word analysis results to plot")
        return

    # Create subplots for each newspaper
    n_newspapers = len(word_analysis_results)
    fig, axes = plt.subplots(1, n_newspapers, figsize=(6 * n_newspapers, 8))

    if n_newspapers == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (newspaper_zh, word_df) in enumerate(word_analysis_results.items()):
        ax = axes[idx]

        # Get top 20 words
        top_words = word_df.head(20)

        # Create horizontal bar chart
        y_pos = np.arange(len(top_words))
        ax.barh(y_pos, top_words["frequency"] * 100, color=colors[idx % len(colors)])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words["word"])
        ax.invert_yaxis()
        ax.set_xlabel("Frequency (%)")
        ax.set_title(f"{newspaper_zh}\nTop Words in Unique Spikes")
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (_, row) in enumerate(top_words.iterrows()):
            ax.text(
                row["frequency"] * 100,
                i,
                f" {row['frequency'] * 100:.1f}%",
                va="center",
                fontsize=8,
            )

    plt.suptitle("Top Contributing Words in Newspaper-Specific Spikes", fontsize=16)
    plt.tight_layout()

    output_file = output_dir / "newspaper_unique_spike_words_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved word comparison plot to {output_file}")


def main():
    """Main function to run newspaper-specific GPR analysis."""

    logger.info("=" * 80)
    logger.info("NEWSPAPER-SPECIFIC GPR ANALYSIS WITH STEP2 DICTIONARIES")
    logger.info("=" * 80)

    import matplotlib.pyplot as plt
    import seaborn as sns

    from gpr_zero.utils.font_setup import ensure_cjk_font_setup

    ensure_cjk_font_setup()
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # Calculate GPR indices for each newspaper
    newspaper_results = calculate_newspaper_gpr_indices()

    if not newspaper_results:
        logger.error("Failed to calculate newspaper GPR indices")
        return

    # Detect spikes for each newspaper
    output_dir = get_project().data_path("newspaper_gpr_step2")
    spike_results = detect_newspaper_spikes(newspaper_results, output_dir)

    # Find unique spikes
    unique_spikes = analyze_unique_spikes(spike_results, output_dir)

    # Analyze top words for unique spikes
    analyze_spike_words(unique_spikes, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
