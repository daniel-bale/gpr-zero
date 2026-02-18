#!/usr/bin/env python3
"""
Analyze word frequency ratios between Liberty Times and China Times.
Uses ALL GPR articles from each newspaper, not just spike periods.
"""

import ibis
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging
from collections import defaultdict
from matplotlib import font_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Newspaper mappings
NEWSPAPERS = {
    'chinatimes': 'China Times (中國時報)',
    'ltn': 'Liberty Times (自由時報)'
}

# Set up Chinese font
def setup_chinese_font():
    """Set up Chinese font for matplotlib"""
    try:
        # Try to find a Chinese font
        chinese_fonts = []
        for font in font_manager.fontManager.ttflist:
            if any(name in font.name.lower() for name in ['noto', 'chinese', 'cjk', 'tw', 'tc']):
                chinese_fonts.append(font.name)
        
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
            logger.info(f"Using Chinese font: {chinese_fonts[0]}")
        else:
            # Fallback
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            logger.warning("No Chinese font found, using default")
            
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        logger.error(f"Error setting up font: {e}")

def analyze_all_gpr_articles():
    """Analyze word frequencies for all GPR articles in each newspaper."""
    
    # Connect to database
    db_path = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
    conn = ibis.connect(f"duckdb:///{db_path}")
    
    # Use the cached binary matrix table
    CACHED_TABLE_NAME = "gpr_binary_matrix_all_words"
    
    try:
        cached_table = conn.table(CACHED_TABLE_NAME)
        logger.info(f"Using cached binary matrix table: {CACHED_TABLE_NAME}")
    except:
        logger.error(f"Could not find cached table: {CACHED_TABLE_NAME}")
        return {}
    
    # Load ALL dictionaries to match binary matrix creation
    dict_dir = Path("/home5/chuenho/gpr-zero/data/dictionaries/all_dictionaries")
    all_dict_files = list(dict_dir.glob("*.json"))
    
    # Collect all unique words from ALL dictionaries to match binary matrix
    all_words = set()
    for dict_file in all_dict_files:
        with open(dict_file, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            word_sets = dict_data.get('word_sets', {})
            for word_set_name, words in word_sets.items():
                for word in words:
                    all_words.add(word)
    
    # Create word-to-column mapping using same ordering as binary matrix
    sorted_words = sorted(all_words)
    word_column_mapping = {}
    for i, word in enumerate(sorted_words):
        col_name = f"contains_word_{i:04d}"
        word_column_mapping[word] = col_name
    
    logger.info(f"Created mapping for {len(word_column_mapping)} words from {len(all_dict_files)} dictionaries")
    
    # Get all word columns that exist in the table
    all_word_columns = list(word_column_mapping.values())
    existing_word_columns = [col for col in all_word_columns if col in cached_table.columns]
    logger.info(f"Found {len(existing_word_columns)} word columns in the cached table")
    
    # Date range for analysis (2017-2024)
    start_date = '2017-01-01'
    end_date = '2024-12-31'
    
    results = {}
    
    for outlet_code, outlet_name in NEWSPAPERS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {outlet_name} - ALL GPR articles")
        logger.info(f"{'='*60}")
        
        # Find matching outlets in the table
        all_outlets = cached_table.select('outlet').distinct().execute()['outlet'].tolist()
        matching_outlets = []
        
        outlet_mapping = {
            'chinatimes': ['China Times', '中國時報'],
            'ltn': ['Liberty Times', '自由時報']
        }
        
        for pattern in outlet_mapping.get(outlet_code, []):
            matching_outlets.extend([o for o in all_outlets if pattern in o])
        
        if not matching_outlets:
            logger.warning(f"No matching outlets found for {outlet_name}")
            continue
            
        logger.info(f"Found matching outlets: {matching_outlets}")
        
        # Filter for this outlet and date range
        outlet_articles = cached_table.filter(
            (cached_table.outlet.isin(matching_outlets)) &
            (cached_table.extracted_publish_date >= start_date) &
            (cached_table.extracted_publish_date <= end_date)
        )
        
        # Count total articles
        total_articles = outlet_articles.count().execute()
        logger.info(f"Total articles (2017-2024): {total_articles}")
        
        # Create GPR filter - use first 100 word columns for efficiency
        sample_cols = existing_word_columns[:100]
        
        gpr_expr = None
        for col in sample_cols:
            if gpr_expr is None:
                gpr_expr = outlet_articles[col]
            else:
                gpr_expr = gpr_expr | outlet_articles[col]
        
        # Add is_gpr column
        outlet_with_gpr = outlet_articles.mutate(is_gpr=gpr_expr)
        
        # Filter for GPR articles only
        gpr_articles = outlet_with_gpr.filter(outlet_with_gpr.is_gpr)
        
        # Count total GPR articles
        total_gpr_articles = gpr_articles.count().execute()
        logger.info(f"Total GPR articles: {total_gpr_articles} ({total_gpr_articles/total_articles*100:.1f}%)")
        
        if total_gpr_articles == 0:
            logger.warning(f"No GPR articles found for {outlet_name}")
            continue
        
        # Get word counts using ibis aggregation
        word_counts = {}
        
        # Process word columns in batches for aggregation
        batch_size = 50
        for i in range(0, len(existing_word_columns), batch_size):
            batch_columns = existing_word_columns[i:i+batch_size]
            
            # Create aggregations for this batch
            aggregations = []
            for col in batch_columns:
                aggregations.append(gpr_articles[col].sum().name(f"{col}_sum"))
            
            # Execute aggregation
            try:
                result = gpr_articles.aggregate(aggregations).execute()
                
                # Map results back to words
                for col in batch_columns:
                    count = result[f"{col}_sum"].iloc[0]
                    if count > 0:
                        # Find the word for this column
                        for word, col_name in word_column_mapping.items():
                            if col_name == col:
                                word_counts[word] = count
                                break
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(existing_word_columns) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.warning(f"Error processing batch {i//batch_size}: {e}")
                continue
        
        logger.info(f"Found {len(word_counts)} unique words in GPR articles")
        
        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate frequencies
        word_freq_data = []
        for word, count in sorted_words[:200]:  # Top 200 words
            freq = count / total_gpr_articles if total_gpr_articles > 0 else 0
            word_freq_data.append({
                'word': word,
                'count': count,
                'frequency': freq,
                'outlet': outlet_name,
                'total_gpr_articles': total_gpr_articles
            })
        
        results[outlet_code] = pd.DataFrame(word_freq_data)
        
        # Save results
        output_dir = Path("/home5/chuenho/gpr-zero/data/newspaper_gpr_step2")
        output_file = output_dir / f"{outlet_code}_all_gpr_word_frequencies.csv"
        results[outlet_code].to_csv(output_file, index=False)
        logger.info(f"Saved word analysis to {output_file}")
        
        # Print top 20 words
        logger.info(f"\nTop 20 words for {outlet_name} (ALL GPR articles):")
        for i, row in results[outlet_code].head(20).iterrows():
            logger.info(f"  {i+1:2d}. {row['word']:15s} - {row['count']:5d} ({row['frequency']*100:5.1f}%)")
    
    return results

def calculate_and_visualize_ratios(results):
    """Calculate frequency ratios and create visualizations."""
    if 'ltn' not in results or 'chinatimes' not in results:
        logger.error("Need both Liberty Times and China Times data")
        return
    
    lt_df = results['ltn']
    ct_df = results['chinatimes']
    
    # Create word-to-frequency mappings
    lt_freq = dict(zip(lt_df['word'], lt_df['frequency']))
    ct_freq = dict(zip(ct_df['word'], ct_df['frequency']))
    
    # Get all unique words
    all_words = set(lt_freq.keys()) | set(ct_freq.keys())
    
    # Calculate ratios
    ratio_data = []
    min_freq = 0.0001  # Minimum frequency to avoid division by zero
    
    for word in all_words:
        lt_f = lt_freq.get(word, 0)
        ct_f = ct_freq.get(word, 0)
        
        # Add small value to avoid division by zero
        lt_f_adj = max(lt_f, min_freq)
        ct_f_adj = max(ct_f, min_freq)
        
        # Calculate ratio
        ratio_lt_ct = lt_f_adj / ct_f_adj
        ratio_ct_lt = ct_f_adj / lt_f_adj
        
        ratio_data.append({
            'word': word,
            'lt_frequency': lt_f,
            'ct_frequency': ct_f,
            'ratio_lt_ct': ratio_lt_ct,
            'ratio_ct_lt': ratio_ct_lt,
            'log_ratio': np.log10(ratio_lt_ct)
        })
    
    ratio_df = pd.DataFrame(ratio_data)
    
    # Sort by absolute log ratio
    ratio_df['abs_log_ratio'] = abs(ratio_df['log_ratio'])
    ratio_df = ratio_df.sort_values('abs_log_ratio', ascending=False)
    
    # Create visualizations
    setup_chinese_font()
    output_dir = Path("/home5/chuenho/gpr-zero/data/newspaper_gpr_step2")
    
    # 1. Top distinctive words bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Liberty Times distinctive words
    lt_distinctive = ratio_df[ratio_df['log_ratio'] > 0].nlargest(20, 'ratio_lt_ct')
    ax1.barh(lt_distinctive['word'], lt_distinctive['ratio_lt_ct'], color='steelblue')
    ax1.set_xlabel('Frequency Ratio (Liberty Times / China Times)')
    ax1.set_title('Words More Frequent in Liberty Times (All GPR Articles)')
    ax1.invert_yaxis()
    
    # China Times distinctive words
    ct_distinctive = ratio_df[ratio_df['log_ratio'] < 0].nlargest(20, 'ratio_ct_lt')
    ax2.barh(ct_distinctive['word'], ct_distinctive['ratio_ct_lt'], color='firebrick')
    ax2.set_xlabel('Frequency Ratio (China Times / Liberty Times)')
    ax2.set_title('Words More Frequent in China Times (All GPR Articles)')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / "all_gpr_word_frequency_ratios.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed ratio analysis
    lt_top = ratio_df[ratio_df['log_ratio'] > 0].nlargest(100, 'ratio_lt_ct')
    ct_top = ratio_df[ratio_df['log_ratio'] < 0].nlargest(100, 'ratio_ct_lt')
    
    lt_top.to_csv(output_dir / "liberty_times_distinctive_words_all_gpr.csv", index=False)
    ct_top.to_csv(output_dir / "china_times_distinctive_words_all_gpr.csv", index=False)
    ratio_df.to_csv(output_dir / "word_frequency_ratios_all_gpr.csv", index=False)
    
    return ratio_df, lt_top, ct_top

def main():
    """Main function."""
    logger.info("="*80)
    logger.info("WORD FREQUENCY ANALYSIS: ALL GPR ARTICLES (2017-2024)")
    logger.info("="*80)
    
    # Analyze all GPR articles
    results = analyze_all_gpr_articles()
    
    if results:
        # Calculate ratios and create visualizations
        ratio_df, lt_top, ct_top = calculate_and_visualize_ratios(results)
        
        logger.info("\n" + "="*80)
        logger.info("TOP 20 LIBERTY TIMES DISTINCTIVE WORDS (ALL GPR)")
        logger.info("="*80)
        for _, row in lt_top.head(20).iterrows():
            logger.info(f"{row['word']:20s} - LT: {row['lt_frequency']*100:5.2f}%, CT: {row['ct_frequency']*100:5.2f}%, Ratio: {row['ratio_lt_ct']:,.1f}x")
        
        logger.info("\n" + "="*80)
        logger.info("TOP 20 CHINA TIMES DISTINCTIVE WORDS (ALL GPR)")
        logger.info("="*80)
        for _, row in ct_top.head(20).iterrows():
            logger.info(f"{row['word']:20s} - CT: {row['ct_frequency']*100:5.2f}%, LT: {row['lt_frequency']*100:5.2f}%, Ratio: {row['ratio_ct_lt']:,.1f}x")
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)

if __name__ == "__main__":
    main()