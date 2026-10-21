#!/usr/bin/env python3
"""
Task 9: Word Frequency Analysis for Spikes (Both Step2 and Lau)
Calculates lift scores for GPR dictionary words during spike periods.
Based on analyze_word_level_contributions.py but adapted for all spikes.
"""

import logging

import pandas as pd
from tqdm import tqdm

from gpr_zero.constants import BINARY_MATRIX_TABLE as CACHED_TABLE_NAME
from gpr_zero.constants import TARGET_OUTLETS
from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory
from gpr_zero.gpr_index.spikes import article_window
from gpr_zero.project import get_project

logger = logging.getLogger(__name__)


def load_all_dictionaries_and_create_mapping():
    """Load the thesis word inventory and retain its substring category attribution."""
    inventory = DictionaryInventory.thesis_directory(get_project().dictionaries_dir)
    word_to_categories = {}
    word_to_dictionaries = {}
    for name, dictionary in inventory.dictionaries.items():
        for set_name, words in dictionary.get("word_sets", {}).items():
            for word in words:
                word_to_dictionaries.setdefault(word, set()).add(name)
                for category, query in dictionary.get("categories", {}).items():
                    # Thesis attribution uses substring membership, not query tokens.
                    if set_name in query:
                        word_to_categories.setdefault(word, set()).add(category)
    return inventory.word_columns, word_to_categories, word_to_dictionaries


def load_spike_data():
    """Load both Step2 and Lau's spikes."""
    spike_dir = get_project().data_path("gpr_spikes", "unified_detection_3outlets")

    # Load Step2 spikes
    step2_file = spike_dir / "step2_median_spikes.csv"
    step2_df = pd.read_csv(
        step2_file, parse_dates=["start_date", "end_date", "peak_date"]
    )
    step2_df["spike_type"] = "Step2"
    logger.info(f"Loaded {len(step2_df)} Step2 spikes")

    # Load Lau's spikes
    lau_file = spike_dir / "lau_2024_spikes.csv"
    lau_df = pd.read_csv(lau_file, parse_dates=["start_date", "end_date", "peak_date"])
    lau_df["spike_type"] = "Lau"
    logger.info(f"Loaded {len(lau_df)} Lau spikes")

    return step2_df, lau_df


def _positive_word_counts(table, word_column_mapping):
    """Aggregate a cohort once and retain vocabulary order in the result."""
    columns = [
        (word, column)
        for word, column in word_column_mapping.items()
        if column in table.columns
    ]
    result = table.aggregate(
        [table[column].sum().name(f"{column}_sum") for _, column in columns]
    ).execute()
    return [
        (word, count)
        for word, column in columns
        if (count := result[f"{column}_sum"].iloc[0]) > 0
    ]


def calculate_baseline_word_frequencies(conn, word_column_mapping):
    """Calculate baseline frequencies for the thesis's three exact outlet names."""
    cached_table = conn.table(CACHED_TABLE_NAME)
    articles = cached_table.filter(cached_table.outlet.isin(TARGET_OUTLETS))
    total = articles.count().execute()
    frame = pd.DataFrame(
        [
            {"word": word, "baseline_count": count, "baseline_frequency": count / total}
            for word, count in _positive_word_counts(articles, word_column_mapping)
        ]
    )
    return frame, total


def calculate_spike_word_frequencies(conn, spike_row, word_column_mapping):
    """Calculate frequencies across the complete final week of a spike."""
    table = conn.table(CACHED_TABLE_NAME)
    start, end = article_window(spike_row)
    articles = table.filter(
        (table.warc_date.cast("date") >= start.date())
        & (table.warc_date.cast("date") <= end.date())
        & table.outlet.isin(TARGET_OUTLETS)
    )
    total = articles.count().execute()
    if total == 0:
        return pd.DataFrame(), 0
    return pd.DataFrame(
        [
            {"word": word, "count": count, "frequency": count / total}
            for word, count in _positive_word_counts(articles, word_column_mapping)
        ]
    ), total


def calculate_lift_scores(spike_words_df, baseline_df):
    """Calculate lift scores comparing spike frequencies to baseline."""
    # Merge spike and baseline data
    merged = spike_words_df.merge(baseline_df, on="word", how="left")

    # Calculate lift score
    merged["lift_score"] = merged["frequency"] / merged["baseline_frequency"].fillna(
        0.0001
    )

    # Calculate contribution score (lift * count)
    merged["contribution_score"] = merged["lift_score"] * merged["count"]

    return merged.sort_values("contribution_score", ascending=False)


def analyze_spikes(conn, spikes_df, baseline_df, word_column_mapping, spike_type):
    """Analyze word frequencies for a set of spikes."""
    logger.info(f"\nAnalyzing {spike_type} spikes...")

    all_spike_words = []

    for idx, spike_row in tqdm(
        spikes_df.iterrows(),
        total=len(spikes_df),
        desc=f"Processing {spike_type} spikes",
    ):
        spike_id = f"{spike_type}_spike_{idx + 1:02d}"

        # Calculate word frequencies for this spike
        spike_words, total_articles = calculate_spike_word_frequencies(
            conn, spike_row, word_column_mapping
        )

        if len(spike_words) == 0:
            logger.warning(f"No words found for {spike_id}")
            continue

        # Calculate lift scores
        spike_words_with_lift = calculate_lift_scores(spike_words, baseline_df)
        spike_words_with_lift["spike_id"] = spike_id
        spike_words_with_lift["spike_type"] = spike_type
        spike_words_with_lift["spike_start"] = spike_row["start_date"]
        spike_words_with_lift["spike_end"] = spike_row["end_date"]
        spike_words_with_lift["total_articles_in_spike"] = total_articles

        all_spike_words.append(spike_words_with_lift)

        # Log top words for this spike
        logger.info(
            f"\n{spike_id}: {spike_row['start_date'].strftime('%Y-%m-%d')} to {spike_row['end_date'].strftime('%Y-%m-%d')}"
        )
        logger.info(f"Total articles: {total_articles}")
        logger.info("Top 5 words by lift score:")
        for _, row in spike_words_with_lift.head(5).iterrows():
            logger.info(
                f"  - {row['word']}: count={row['count']}, lift={row['lift_score']:.2f}"
            )

    # Combine all spike data
    if all_spike_words:
        combined_df = pd.concat(all_spike_words, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


def create_summary_statistics(step2_results, lau_results, output_dir):
    """Create summary statistics and save results."""
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)

    # Step2 summary
    if len(step2_results) > 0:
        logger.info("\nStep2 Spikes:")
        logger.info(
            f"Total unique words across all spikes: {step2_results['word'].nunique()}"
        )

        # Top words by average lift score
        step2_avg_lift = (
            step2_results.groupby("word")["lift_score"]
            .mean()
            .sort_values(ascending=False)
        )
        logger.info("\nTop 10 Step2 words by average lift score:")
        for word, lift in step2_avg_lift.head(10).items():
            logger.info(f"  - {word}: {lift:.2f}x baseline")

        # Save Step2 results
        step2_results.to_csv(
            output_dir / "step2_spike_word_lift_scores.csv", index=False
        )

        # Save top words summary
        step2_summary = (
            step2_results.groupby("word")
            .agg({"count": "sum", "lift_score": "mean", "spike_id": "count"})
            .rename(columns={"spike_id": "spike_count"})
            .sort_values("lift_score", ascending=False)
        )
        step2_summary.to_csv(output_dir / "step2_top_words_by_lift.csv")

    # Lau summary
    if len(lau_results) > 0:
        logger.info("\nLau Spikes:")
        logger.info(
            f"Total unique words across all spikes: {lau_results['word'].nunique()}"
        )

        # Top words by average lift score
        lau_avg_lift = (
            lau_results.groupby("word")["lift_score"]
            .mean()
            .sort_values(ascending=False)
        )
        logger.info("\nTop 10 Lau words by average lift score:")
        for word, lift in lau_avg_lift.head(10).items():
            logger.info(f"  - {word}: {lift:.2f}x baseline")

        # Save Lau results
        lau_results.to_csv(output_dir / "lau_spike_word_lift_scores.csv", index=False)

        # Save top words summary
        lau_summary = (
            lau_results.groupby("word")
            .agg({"count": "sum", "lift_score": "mean", "spike_id": "count"})
            .rename(columns={"spike_id": "spike_count"})
            .sort_values("lift_score", ascending=False)
        )
        lau_summary.to_csv(output_dir / "lau_top_words_by_lift.csv")

    # Compare Step2 vs Lau
    if len(step2_results) > 0 and len(lau_results) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON: Step2 vs Lau")
        logger.info("=" * 60)

        # Words unique to each
        step2_words = set(step2_results["word"].unique())
        lau_words = set(lau_results["word"].unique())

        logger.info(f"\nWords unique to Step2 spikes: {len(step2_words - lau_words)}")
        logger.info(f"Words unique to Lau spikes: {len(lau_words - step2_words)}")
        logger.info(f"Words in both: {len(step2_words & lau_words)}")

        # Top unique words for each
        step2_unique = step2_results[
            step2_results["word"].isin(step2_words - lau_words)
        ]
        if len(step2_unique) > 0:
            step2_unique_top = (
                step2_unique.groupby("word")["lift_score"]
                .mean()
                .sort_values(ascending=False)
            )
            logger.info("\nTop 5 words unique to Step2 spikes:")
            for word, lift in step2_unique_top.head(5).items():
                logger.info(f"  - {word}: {lift:.2f}x baseline")

        lau_unique = lau_results[lau_results["word"].isin(lau_words - step2_words)]
        if len(lau_unique) > 0:
            lau_unique_top = (
                lau_unique.groupby("word")["lift_score"]
                .mean()
                .sort_values(ascending=False)
            )
            logger.info("\nTop 5 words unique to Lau spikes:")
            for word, lift in lau_unique_top.head(5).items():
                logger.info(f"  - {word}: {lift:.2f}x baseline")


def main():
    project = get_project()
    output_dir = project.data_path("word_contributions")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Task 9: Word Frequency Analysis for Spikes ===")
    logger.info("Calculating lift scores for GPR dictionary words during spike periods")

    # Load word mappings
    word_column_mapping, word_to_categories, word_to_dictionaries = (
        load_all_dictionaries_and_create_mapping()
    )

    # Load spike data
    step2_spikes, lau_spikes = load_spike_data()

    # Connect to database (read-only)
    conn = project.connect()
    ArticleCorpus.from_cache(
        conn, DictionaryInventory.thesis_directory(project.dictionaries_dir)
    )

    # Calculate baseline word frequencies
    baseline_df, total_baseline_articles = calculate_baseline_word_frequencies(
        conn, word_column_mapping
    )

    # Save baseline for reference
    baseline_df.to_csv(output_dir / "baseline_word_frequencies.csv", index=False)
    logger.info(f"\nBaseline calculated from {total_baseline_articles} articles")

    # Analyze Step2 spikes
    step2_results = analyze_spikes(
        conn, step2_spikes, baseline_df, word_column_mapping, "Step2"
    )

    # Analyze Lau spikes
    lau_results = analyze_spikes(
        conn, lau_spikes, baseline_df, word_column_mapping, "Lau"
    )

    # Create summary statistics and save results
    create_summary_statistics(step2_results, lau_results, output_dir)

    conn.disconnect()

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)
