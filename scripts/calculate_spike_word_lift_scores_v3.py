#!/usr/bin/env python3
"""
Task 9: Word Frequency Analysis for Spikes (Both Step2 and Lau)
Calculates lift scores for GPR dictionary words during spike periods.
Based on analyze_word_level_contributions.py but adapted for all spikes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ibis
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

# Logging Configuration
log_file = Path('logs') / f'task9_word_lift_scores_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
CACHED_TABLE_NAME = "gpr_binary_matrix_all_words"
OUTPUT_DIR = Path("data/word_contributions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3-outlet filter
TARGET_OUTLETS = [
    'China Times (中國時報)',
    'Liberty Times (自由時報)', 
    'United Daily (聯合報)'
]


def load_all_dictionaries_and_create_mapping():
    """Load ALL 80 dictionaries to create the correct word column mapping."""
    project_root = Path(__file__).parent.parent
    all_dict_dir = project_root / "data" / "dictionaries" / "all_dictionaries"
    
    all_words = set()
    word_to_categories = {}  # Track which categories each word belongs to
    word_to_dictionaries = {}  # Track which dictionaries contain each word
    
    # Load all 80 dictionaries
    for dict_file in all_dict_dir.glob("*.json"):
        with open(dict_file, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            dict_name = dict_file.stem
            
            # Process word sets and categories
            word_sets = dict_data.get('word_sets', {})
            categories = dict_data.get('categories', {})
            
            # Collect all words from word sets
            for word_set_name, words in word_sets.items():
                for word in words:
                    all_words.add(word)
                    
                    # Track dictionary containing this word
                    if word not in word_to_dictionaries:
                        word_to_dictionaries[word] = set()
                    word_to_dictionaries[word].add(dict_name)
                    
                    # Track categories this word contributes to
                    for cat_name, cat_query in categories.items():
                        if word_set_name in cat_query:
                            if word not in word_to_categories:
                                word_to_categories[word] = set()
                            word_to_categories[word].add(cat_name)
    
    # Create word-to-column mapping using same ordering as binary matrix
    sorted_words = sorted(all_words)
    word_column_mapping = {}
    for i, word in enumerate(sorted_words):
        col_name = f"contains_word_{i:04d}"
        word_column_mapping[word] = col_name
    
    logger.info(f"Created mapping for {len(sorted_words)} unique words from {len(list(all_dict_dir.glob('*.json')))} dictionaries")
    return word_column_mapping, word_to_categories, word_to_dictionaries


def load_spike_data():
    """Load both Step2 and Lau's spikes."""
    spike_dir = Path(__file__).parent.parent / "data" / "gpr_spikes" / "unified_detection_3outlets"
    
    # Load Step2 spikes
    step2_file = spike_dir / "step2_median_spikes.csv"
    step2_df = pd.read_csv(step2_file, parse_dates=['start_date', 'end_date', 'peak_date'])
    step2_df['spike_type'] = 'Step2'
    logger.info(f"Loaded {len(step2_df)} Step2 spikes")
    
    # Load Lau's spikes
    lau_file = spike_dir / "lau_2024_spikes.csv"
    lau_df = pd.read_csv(lau_file, parse_dates=['start_date', 'end_date', 'peak_date'])
    lau_df['spike_type'] = 'Lau'
    logger.info(f"Loaded {len(lau_df)} Lau spikes")
    
    return step2_df, lau_df


def calculate_baseline_word_frequencies(conn, word_column_mapping):
    """Calculate baseline word frequencies across entire dataset."""
    logger.info("Calculating baseline word frequencies...")
    
    cached_table = conn.table(CACHED_TABLE_NAME)
    
    # Filter for 3 outlets only
    baseline_articles = cached_table.filter(
        cached_table.outlet.isin(TARGET_OUTLETS)
    )
    
    total_articles = baseline_articles.count().execute()
    logger.info(f"Total articles in baseline (3 outlets): {total_articles}")
    
    # Get word columns that exist in the table
    all_word_columns = list(word_column_mapping.values())
    existing_word_columns = [col for col in all_word_columns if col in cached_table.columns]
    logger.info(f"Found {len(existing_word_columns)} word columns in the cached table")
    
    # Calculate word frequencies for all columns at once
    logger.info("Creating aggregations for all word columns...")
    aggregations = []
    for col in existing_word_columns:
        aggregations.append(baseline_articles[col].sum().name(f"{col}_sum"))
    
    logger.info(f"Executing aggregation for {len(aggregations)} columns...")
    result = baseline_articles.aggregate(aggregations).execute()
    
    # Map results back to words
    word_counts = {}
    for col in existing_word_columns:
        count = result[f"{col}_sum"].iloc[0]
        if count > 0:
            # Find the word for this column
            for word, col_name in word_column_mapping.items():
                if col_name == col:
                    word_counts[word] = count
                    break
    
    # Convert to DataFrame
    baseline_df = pd.DataFrame([
        {'word': word, 'baseline_count': count, 'baseline_frequency': count / total_articles}
        for word, count in word_counts.items()
    ])
    
    logger.info(f"Calculated baseline frequencies for {len(baseline_df)} words")
    return baseline_df, total_articles


def calculate_spike_word_frequencies(conn, spike_row, word_column_mapping):
    """Calculate word frequencies for a specific spike period."""
    cached_table = conn.table(CACHED_TABLE_NAME)
    
    # Convert spike dates to article date range
    spike_start = spike_row['start_date']
    spike_end = spike_row['end_date']
    
    # For weekly spikes, expand to full week coverage
    article_start = spike_start
    article_end = spike_end + timedelta(days=6)
    
    # Filter for spike period and outlets
    spike_articles = cached_table.filter(
        (cached_table.warc_date.cast('date') >= article_start.date()) &
        (cached_table.warc_date.cast('date') <= article_end.date()) &
        (cached_table.outlet.isin(TARGET_OUTLETS))
    )
    
    total_articles = spike_articles.count().execute()
    
    if total_articles == 0:
        return pd.DataFrame(), 0
    
    # Get word columns that exist in the table
    all_word_columns = list(word_column_mapping.values())
    existing_word_columns = [col for col in all_word_columns if col in cached_table.columns]
    
    # Calculate word frequencies for all columns at once
    aggregations = []
    for col in existing_word_columns:
        aggregations.append(spike_articles[col].sum().name(f"{col}_sum"))
    
    # Execute aggregation
    result = spike_articles.aggregate(aggregations).execute()
    
    # Map results back to words
    word_counts = {}
    for col in existing_word_columns:
        count = result[f"{col}_sum"].iloc[0]
        if count > 0:
            # Find the word for this column
            for word, col_name in word_column_mapping.items():
                if col_name == col:
                    word_counts[word] = count
                    break
    
    # Convert to DataFrame
    spike_words_df = pd.DataFrame([
        {'word': word, 'count': count, 'frequency': count / total_articles}
        for word, count in word_counts.items()
    ])
    
    return spike_words_df, total_articles


def calculate_lift_scores(spike_words_df, baseline_df):
    """Calculate lift scores comparing spike frequencies to baseline."""
    # Merge spike and baseline data
    merged = spike_words_df.merge(baseline_df, on='word', how='left')
    
    # Calculate lift score
    merged['lift_score'] = merged['frequency'] / merged['baseline_frequency'].fillna(0.0001)
    
    # Calculate contribution score (lift * count)
    merged['contribution_score'] = merged['lift_score'] * merged['count']
    
    return merged.sort_values('contribution_score', ascending=False)


def analyze_spikes(conn, spikes_df, baseline_df, word_column_mapping, spike_type):
    """Analyze word frequencies for a set of spikes."""
    logger.info(f"\nAnalyzing {spike_type} spikes...")
    
    all_spike_words = []
    
    for idx, spike_row in tqdm(spikes_df.iterrows(), total=len(spikes_df), desc=f"Processing {spike_type} spikes"):
        spike_id = f"{spike_type}_spike_{idx+1:02d}"
        
        # Calculate word frequencies for this spike
        spike_words, total_articles = calculate_spike_word_frequencies(conn, spike_row, word_column_mapping)
        
        if len(spike_words) == 0:
            logger.warning(f"No words found for {spike_id}")
            continue
        
        # Calculate lift scores
        spike_words_with_lift = calculate_lift_scores(spike_words, baseline_df)
        spike_words_with_lift['spike_id'] = spike_id
        spike_words_with_lift['spike_type'] = spike_type
        spike_words_with_lift['spike_start'] = spike_row['start_date']
        spike_words_with_lift['spike_end'] = spike_row['end_date']
        spike_words_with_lift['total_articles_in_spike'] = total_articles
        
        all_spike_words.append(spike_words_with_lift)
        
        # Log top words for this spike
        logger.info(f"\n{spike_id}: {spike_row['start_date'].strftime('%Y-%m-%d')} to {spike_row['end_date'].strftime('%Y-%m-%d')}")
        logger.info(f"Total articles: {total_articles}")
        logger.info("Top 5 words by lift score:")
        for _, row in spike_words_with_lift.head(5).iterrows():
            logger.info(f"  - {row['word']}: count={row['count']}, lift={row['lift_score']:.2f}")
    
    # Combine all spike data
    if all_spike_words:
        combined_df = pd.concat(all_spike_words, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


def create_summary_statistics(step2_results, lau_results, output_dir):
    """Create summary statistics and save results."""
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    
    # Step2 summary
    if len(step2_results) > 0:
        logger.info("\nStep2 Spikes:")
        logger.info(f"Total unique words across all spikes: {step2_results['word'].nunique()}")
        
        # Top words by average lift score
        step2_avg_lift = step2_results.groupby('word')['lift_score'].mean().sort_values(ascending=False)
        logger.info("\nTop 10 Step2 words by average lift score:")
        for word, lift in step2_avg_lift.head(10).items():
            logger.info(f"  - {word}: {lift:.2f}x baseline")
        
        # Save Step2 results
        step2_results.to_csv(output_dir / "step2_spike_word_lift_scores.csv", index=False)
        
        # Save top words summary
        step2_summary = step2_results.groupby('word').agg({
            'count': 'sum',
            'lift_score': 'mean',
            'spike_id': 'count'
        }).rename(columns={'spike_id': 'spike_count'}).sort_values('lift_score', ascending=False)
        step2_summary.to_csv(output_dir / "step2_top_words_by_lift.csv")
    
    # Lau summary
    if len(lau_results) > 0:
        logger.info("\nLau Spikes:")
        logger.info(f"Total unique words across all spikes: {lau_results['word'].nunique()}")
        
        # Top words by average lift score
        lau_avg_lift = lau_results.groupby('word')['lift_score'].mean().sort_values(ascending=False)
        logger.info("\nTop 10 Lau words by average lift score:")
        for word, lift in lau_avg_lift.head(10).items():
            logger.info(f"  - {word}: {lift:.2f}x baseline")
        
        # Save Lau results
        lau_results.to_csv(output_dir / "lau_spike_word_lift_scores.csv", index=False)
        
        # Save top words summary
        lau_summary = lau_results.groupby('word').agg({
            'count': 'sum',
            'lift_score': 'mean',
            'spike_id': 'count'
        }).rename(columns={'spike_id': 'spike_count'}).sort_values('lift_score', ascending=False)
        lau_summary.to_csv(output_dir / "lau_top_words_by_lift.csv")
    
    # Compare Step2 vs Lau
    if len(step2_results) > 0 and len(lau_results) > 0:
        logger.info("\n" + "="*60)
        logger.info("COMPARISON: Step2 vs Lau")
        logger.info("="*60)
        
        # Words unique to each
        step2_words = set(step2_results['word'].unique())
        lau_words = set(lau_results['word'].unique())
        
        logger.info(f"\nWords unique to Step2 spikes: {len(step2_words - lau_words)}")
        logger.info(f"Words unique to Lau spikes: {len(lau_words - step2_words)}")
        logger.info(f"Words in both: {len(step2_words & lau_words)}")
        
        # Top unique words for each
        step2_unique = step2_results[step2_results['word'].isin(step2_words - lau_words)]
        if len(step2_unique) > 0:
            step2_unique_top = step2_unique.groupby('word')['lift_score'].mean().sort_values(ascending=False)
            logger.info("\nTop 5 words unique to Step2 spikes:")
            for word, lift in step2_unique_top.head(5).items():
                logger.info(f"  - {word}: {lift:.2f}x baseline")
        
        lau_unique = lau_results[lau_results['word'].isin(lau_words - step2_words)]
        if len(lau_unique) > 0:
            lau_unique_top = lau_unique.groupby('word')['lift_score'].mean().sort_values(ascending=False)
            logger.info("\nTop 5 words unique to Lau spikes:")
            for word, lift in lau_unique_top.head(5).items():
                logger.info(f"  - {word}: {lift:.2f}x baseline")


def main():
    logger.info("=== Task 9: Word Frequency Analysis for Spikes ===")
    logger.info("Calculating lift scores for GPR dictionary words during spike periods")
    
    # Load word mappings
    word_column_mapping, word_to_categories, word_to_dictionaries = load_all_dictionaries_and_create_mapping()
    
    # Load spike data
    step2_spikes, lau_spikes = load_spike_data()
    
    # Connect to database (read-only)
    conn = ibis.connect(f"duckdb://{DB_PATH}?access_mode=read_only")
    
    # Calculate baseline word frequencies
    baseline_df, total_baseline_articles = calculate_baseline_word_frequencies(conn, word_column_mapping)
    
    # Save baseline for reference
    baseline_df.to_csv(OUTPUT_DIR / "baseline_word_frequencies.csv", index=False)
    logger.info(f"\nBaseline calculated from {total_baseline_articles} articles")
    
    # Analyze Step2 spikes
    step2_results = analyze_spikes(conn, step2_spikes, baseline_df, word_column_mapping, "Step2")
    
    # Analyze Lau spikes
    lau_results = analyze_spikes(conn, lau_spikes, baseline_df, word_column_mapping, "Lau")
    
    # Create summary statistics and save results
    create_summary_statistics(step2_results, lau_results, OUTPUT_DIR)
    
    conn.disconnect()
    
    logger.info("\n" + "="*60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()