#!/usr/bin/env python3
"""
Complete implementation: Calculate category-specific GPR indices.
For each of 40 Step2 dictionaries and 8 categories:
1. Calculate weekly GPR index for that category
2. Take median across 40 dictionaries for each category
3. Scale to mean=100 and plot
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ibis
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
OUTPUT_DIR = Path("data/category_gpr_indices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all 80 dictionaries for word mapping (matching the binary matrix)
ALL_DICTS_DIR = Path("/home5/chuenho/gpr-zero/data/dictionaries/all_dictionaries")

# Standard 8 categories
STANDARD_CATEGORIES = [
    "Military Escalation",
    "Diplomatic Isolation", 
    "Economic Coercion",
    "Regional Security",
    "Internal Political Instability",
    "Gray Zone Operations",
    "US-China-Taiwan Relations",
    "Cyber and Information Warfare"
]

CATEGORY_MAPPING = {
    "Military Escalation": "Military Escalation",
    "Military Escalation (軍事升級)": "Military Escalation",
    "軍事升級": "Military Escalation",
    "Diplomatic Isolation": "Diplomatic Isolation",
    "Diplomatic Isolation (外交孤立)": "Diplomatic Isolation",
    "外交孤立": "Diplomatic Isolation",
    "Economic Coercion": "Economic Coercion",
    "Economic Coercion (經濟脅迫)": "Economic Coercion",
    "經濟脅迫": "Economic Coercion",
    "Regional Security": "Regional Security",
    "Regional Security (區域安全)": "Regional Security",
    "區域安全": "Regional Security",
    "Internal Political Instability": "Internal Political Instability",
    "Internal Political Instability (內部政治不穩定)": "Internal Political Instability",
    "內部政治不穩定": "Internal Political Instability",
    "Gray Zone Operations": "Gray Zone Operations",
    "Gray Zone Operations (灰色地帶行動)": "Gray Zone Operations",
    "灰色地帶行動": "Gray Zone Operations",
    "US-China-Taiwan Relations": "US-China-Taiwan Relations",
    "US-China-Taiwan Relations (美中台關係)": "US-China-Taiwan Relations",
    "美中台關係": "US-China-Taiwan Relations",
    "Cyber and Information Warfare": "Cyber and Information Warfare",
    "Cyber and Information Warfare (網路與資訊戰)": "Cyber and Information Warfare",
    "網路與資訊戰": "Cyber and Information Warfare"
}


def load_all_dictionaries_for_mapping():
    """Load all 80 dictionaries to create word mapping (matching binary matrix)."""
    all_words = set()
    
    for dict_file in ALL_DICTS_DIR.glob("*.json"):
        if "_meta.json" in dict_file.name:
            continue
            
        with open(dict_file, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            if 'word_sets' in dict_data:
                for word_set in dict_data['word_sets'].values():
                    all_words.update(word_set)
    
    # Create mapping
    sorted_words = sorted(all_words)
    word_to_column = {}
    for i, word in enumerate(sorted_words):
        col_name = f"contains_word_{i:04d}"
        word_to_column[word] = col_name
    
    return word_to_column


def load_step2_dictionaries():
    """Load Step2 dictionaries only."""
    step2_dir = Path("/home5/chuenho/gpr-zero/data/dictionaries/step2_batch")
    
    dictionaries = {}
    for dict_file in step2_dir.glob("*.json"):
        if "_meta.json" in dict_file.name:
            continue
            
        with open(dict_file, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            dict_name = dict_file.stem
            
            # Normalize category names
            if 'categories' in dict_data:
                normalized_categories = {}
                for cat_name, cat_def in dict_data['categories'].items():
                    normalized_name = CATEGORY_MAPPING.get(cat_name, cat_name)
                    normalized_categories[normalized_name] = cat_def
                dict_data['categories'] = normalized_categories
            
            dictionaries[dict_name] = dict_data
    
    return dictionaries


def calculate_category_gpr_for_dictionary(conn, dictionary, category, word_to_column):
    """Calculate GPR index for a specific category in a specific dictionary using ibis."""
    
    # Get word sets and category definition
    word_sets = dictionary.get("word_sets", {})
    categories = dictionary.get("categories", {})
    
    if category not in categories:
        return None
    
    # Connect to cached table
    cached_table = conn.table("gpr_binary_matrix_all_words")
    
    # Parse category query
    query_string = categories[category]
    word_set_names = [name.strip() for name in query_string.split(" AND ")]
    
    # Build expression for this category using ibis
    from functools import reduce
    from operator import and_, or_
    
    word_set_expressions = []
    
    for set_name in word_set_names:
        if set_name not in word_sets:
            continue
            
        words = word_sets[set_name]
        word_exprs = []
        
        for word in words:
            if word in word_to_column:
                col_name = word_to_column[word]
                if col_name in cached_table.columns:
                    word_exprs.append(cached_table[col_name])
        
        if word_exprs:
            # OR condition within word set
            set_expr = reduce(or_, word_exprs)
            word_set_expressions.append(set_expr)
    
    if not word_set_expressions:
        return None
    
    # AND condition between word sets
    category_expr = reduce(and_, word_set_expressions, ibis.literal(True))
    
    # Filter for target newspapers and date range
    filtered = cached_table.filter(
        cached_table.outlet.isin(['China Times (中國時報)', 'Liberty Times (自由時報)', 'United Daily (聯合報)']) &
        (cached_table.warc_date >= '2017-01-01') &
        (cached_table.warc_date <= '2024-12-31')
    )
    
    # Add category match column
    with_category = filtered.mutate(
        is_category_match=category_expr.cast('int32')
    )
    
    # Group by week and calculate GPR
    # Cast string date to date type first
    weekly = with_category.group_by(
        with_category.warc_date.cast('date').truncate('W').name('week')
    ).aggregate(
        category_articles=with_category.is_category_match.sum(),
        total_articles=with_category.count()
    )
    
    # Execute and convert to pandas
    result = weekly.execute()
    
    # Calculate GPR index
    result['gpr_index'] = (result['category_articles'] / result['total_articles'])
    result['gpr_index'] = result['gpr_index'].fillna(0)
    
    # Sort by week
    result = result.sort_values('week')
    
    return result[['week', 'gpr_index']]


def main():
    logger.info("=== Calculate Category-Specific GPR Indices ===")
    
    # Load dictionaries
    logger.info("Loading dictionaries...")
    word_to_column = load_all_dictionaries_for_mapping()
    step2_dicts = load_step2_dictionaries()
    logger.info(f"Loaded {len(step2_dicts)} Step2 dictionaries")
    logger.info(f"Total words in mapping: {len(word_to_column)}")
    
    # Connect to database
    conn = ibis.connect(f"duckdb://{DB_PATH}?access_mode=read_only")
    
    # Calculate GPR for each dictionary and category
    all_results = []
    
    for dict_name, dictionary in tqdm(step2_dicts.items(), desc="Processing dictionaries"):
        for category in STANDARD_CATEGORIES:
            result = calculate_category_gpr_for_dictionary(conn, dictionary, category, word_to_column)
            
            if result is not None:
                result['dictionary'] = dict_name
                result['category'] = category
                all_results.append(result)
    
    # Combine all results
    logger.info("Combining results...")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save raw results
    combined_df.to_csv(OUTPUT_DIR / "all_category_gpr_indices.csv", index=False)
    logger.info(f"Saved raw results to {OUTPUT_DIR / 'all_category_gpr_indices.csv'}")
    
    # Calculate median for each category
    logger.info("Calculating median indices...")
    median_indices = combined_df.groupby(['category', 'week'])['gpr_index'].median().reset_index()
    
    # Pivot to get categories as columns
    pivot_df = median_indices.pivot(index='week', columns='category', values='gpr_index')
    
    # Save unscaled median
    pivot_df.to_csv(OUTPUT_DIR / "category_median_gpr_unscaled.csv")
    
    # Scale each category to mean=100
    scaled_df = pivot_df.copy()
    for col in scaled_df.columns:
        col_mean = scaled_df[col].mean()
        if col_mean > 0:
            scaled_df[col] = (scaled_df[col] / col_mean) * 100
    
    # Save scaled median
    scaled_df.to_csv(OUTPUT_DIR / "category_median_gpr_scaled.csv")
    
    # Plot the results
    logger.info("Creating plots...")
    ensure_cjk_font_setup()
    
    # Plot 1: Scaled category indices
    fig, ax = plt.subplots(figsize=(14, 8))
    
    category_colors = {
        'Military Escalation': '#FF6B6B',
        'Diplomatic Isolation': '#4ECDC4',
        'Economic Coercion': '#45B7D1',
        'Regional Security': '#F7B801',
        'Internal Political Instability': '#95E1D3',
        'Gray Zone Operations': '#DDA0DD',
        'US-China-Taiwan Relations': '#98D8C8',
        'Cyber and Information Warfare': '#FFB6C1'
    }
    
    for category in STANDARD_CATEGORIES:
        if category in scaled_df.columns:
            ax.plot(scaled_df.index, scaled_df[category], 
                   label=category, 
                   color=category_colors[category], 
                   linewidth=2,
                   alpha=0.8)
    
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Category-Specific GPR Index (Mean = 100)')
    ax.set_title('Taiwan GPR Index by Category (Step2 Median, Scaled)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "category_gpr_indices_scaled.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to thesis figures
    import shutil
    thesis_path = Path("/home5/chuenho/thesis/figures/category_gpr_indices_scaled.png")
    shutil.copy(plot_path, thesis_path)
    logger.info(f"Saved plots to {plot_path} and {thesis_path}")
    
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
            
            peak_weeks.append({
                'category': category,
                'peak_week': max_idx,
                'scaled_peak_value': max_value,
                'unscaled_peak_value': unscaled_value,
                'scaled_mean': 100.0
            })
    
    peak_df = pd.DataFrame(peak_weeks)
    peak_df.to_csv(OUTPUT_DIR / "category_peak_weeks.csv", index=False)
    logger.info(f"Saved peak weeks to {OUTPUT_DIR / 'category_peak_weeks.csv'}")
    
    # Print summary
    logger.info("\n=== Peak Weeks Summary ===")
    for _, row in peak_df.iterrows():
        logger.info(f"{row['category']}: {row['peak_week'].strftime('%Y-%m-%d')} "
                   f"(scaled: {row['scaled_peak_value']:.1f}, unscaled: {row['unscaled_peak_value']:.1f})")
    
    logger.info("\nDone! Next steps:")
    logger.info("1. Update analyze_category_peak_events.py to use these peak weeks")
    logger.info("2. Re-run with proper category-specific article sampling")


if __name__ == "__main__":
    main()