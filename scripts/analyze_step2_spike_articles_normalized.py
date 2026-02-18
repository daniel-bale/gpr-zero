#!/usr/bin/env python3
"""
Task 4: Analyze Step2-unique spike articles with normalized category names.
Maps all category variations to 8 standard English categories.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ibis
import ibis.expr.types as ir
from ibis import _
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
from functools import reduce
from operator import and_, or_
from tqdm import tqdm

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
OUTPUT_DIR = Path("data/gpr_spikes/spike_articles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Newspapers to analyze
NEWSPAPER_PATTERNS = ["China Times", "Liberty Times", "United Daily"]

# The cached table name from GPRCalculatorBinaryMatrix
CACHED_TABLE_NAME = "gpr_binary_matrix_all_words"

# Standard 8 categories (English only)
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

# Category normalization mapping
CATEGORY_MAPPING = {
    # Military Escalation
    "Military Escalation": "Military Escalation",
    "Military Escalation (軍事升級)": "Military Escalation",
    "軍事升級": "Military Escalation",
    "military_escalation": "Military Escalation",
    
    # Diplomatic Isolation
    "Diplomatic Isolation": "Diplomatic Isolation",
    "Diplomatic Isolation (外交孤立)": "Diplomatic Isolation",
    "外交孤立": "Diplomatic Isolation",
    "diplomatic_isolation": "Diplomatic Isolation",
    
    # Economic Coercion
    "Economic Coercion": "Economic Coercion",
    "Economic Coercion (經濟脅迫)": "Economic Coercion",
    "經濟脅迫": "Economic Coercion",
    "economic_coercion": "Economic Coercion",
    
    # Regional Security
    "Regional Security": "Regional Security",
    "Regional Security (區域安全)": "Regional Security",
    "區域安全": "Regional Security",
    "regional_security": "Regional Security",
    
    # Internal Political Instability
    "Internal Political Instability": "Internal Political Instability",
    "Internal Political Instability (內部政治不穩定)": "Internal Political Instability",
    "內部政治不穩定": "Internal Political Instability",
    "internal_political_instability": "Internal Political Instability",
    
    # Gray Zone Operations
    "Gray Zone Operations": "Gray Zone Operations",
    "Gray Zone Operations (灰色地帶行動)": "Gray Zone Operations",
    "灰色地帶行動": "Gray Zone Operations",
    "gray_zone_operations": "Gray Zone Operations",
    
    # US-China-Taiwan Relations
    "US-China-Taiwan Relations": "US-China-Taiwan Relations",
    "US-China-Taiwan Relations (美中台關係)": "US-China-Taiwan Relations",
    "美中台關係": "US-China-Taiwan Relations",
    "us_china_taiwan_relations": "US-China-Taiwan Relations",
    
    # Cyber and Information Warfare
    "Cyber and Information Warfare": "Cyber and Information Warfare",
    "Cyber and Information Warfare (網路與資訊戰)": "Cyber and Information Warfare",
    "網路與資訊戰": "Cyber and Information Warfare",
    "cyber_and_information_warfare": "Cyber and Information Warfare"
}


def convert_weekly_spike_to_date_range(spike_start, spike_end):
    """Convert weekly spike dates to actual date range for article sampling."""
    actual_start = spike_start
    actual_end = spike_end + timedelta(days=6)
    
    logger.info(f"Converting: {spike_start.date()} to {spike_end.date()} -> {actual_start.date()} to {actual_end.date()}")
    
    return actual_start.strftime('%Y-%m-%d'), actual_end.strftime('%Y-%m-%d')


def load_step2_dictionaries_and_word_mapping():
    """Load all Step2 dictionaries and build word-to-column mapping."""
    project_root = Path(__file__).parent.parent
    step2_batch_dir = project_root / "data" / "dictionaries" / "step2_batch"
    
    step2_files = list(step2_batch_dir.glob("*.json"))
    logger.info(f"Found {len(step2_files)} Step2 dictionaries")
    
    dictionaries = {}
    all_words = set()
    
    for dict_file in step2_files:
        with open(dict_file, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            dict_name = dict_file.stem
            
            # Normalize category names
            if 'categories' in dict_data:
                normalized_categories = {}
                for cat_name, cat_def in dict_data['categories'].items():
                    if cat_name in CATEGORY_MAPPING:
                        normalized_name = CATEGORY_MAPPING[cat_name]
                        normalized_categories[normalized_name] = cat_def
                    else:
                        logger.warning(f"Unknown category '{cat_name}' in {dict_name}")
                        normalized_categories[cat_name] = cat_def
                
                dict_data['categories'] = normalized_categories
            
            dictionaries[dict_name] = dict_data
            
            # Collect all words
            if 'word_sets' in dict_data:
                for word_set in dict_data['word_sets'].values():
                    all_words.update(word_set)
    
    # Create word-to-column mapping (matching GPRCalculatorBinaryMatrix logic)
    sorted_words = sorted(all_words)
    word_column_mapping = {}
    for i, word in enumerate(sorted_words):
        col_name = f"contains_word_{i:04d}"
        word_column_mapping[word] = col_name
    
    logger.info(f"Total unique words: {len(all_words)}")
    logger.info(f"Standard categories: {STANDARD_CATEGORIES}")
    
    return dictionaries, word_column_mapping


def build_word_set_expressions(word_sets: Dict[str, List[str]], word_column_mapping: Dict[str, str], table, dict_name: str) -> Dict[str, ir.BooleanColumn]:
    """Build word set expressions with dictionary-specific naming."""
    word_set_expressions = {}
    
    for set_name, words in word_sets.items():
        # Get columns for words that exist in this word set
        relevant_word_columns = []
        for word in words:
            if word in word_column_mapping and word_column_mapping[word] in table.columns:
                relevant_word_columns.append(word_column_mapping[word])
        
        if relevant_word_columns:
            # OR of all words in the set
            word_columns_exprs = [table[col] for col in relevant_word_columns]
            word_set_expr = reduce(or_, word_columns_exprs)
            # Use dictionary-specific naming
            word_set_expressions[f"{dict_name}_{set_name}"] = word_set_expr
        else:
            word_set_expressions[f"{dict_name}_{set_name}"] = ibis.literal(False)
    
    return word_set_expressions


def build_category_expressions(categories: Dict[str, str], word_set_expressions: Dict[str, ir.BooleanColumn], dict_name: str) -> Dict[str, ir.BooleanColumn]:
    """Build category expressions with simplified parsing (no + or N/ operators)."""
    category_expressions = {}
    
    for category_name, query_string in categories.items():
        # Simplified parsing - just AND of word sets
        word_set_names = [name.strip() for name in query_string.split(" AND ")]
        
        if not word_set_names:
            category_expr = ibis.literal(False)
        else:
            # AND of all required word sets
            has_expressions = []
            for set_name in word_set_names:
                # Look for dictionary-specific word set
                dict_set_name = f"{dict_name}_{set_name}"
                if dict_set_name in word_set_expressions:
                    has_expressions.append(word_set_expressions[dict_set_name])
            
            if has_expressions:
                category_expr = reduce(and_, has_expressions, ibis.literal(True)).fill_null(False)
            else:
                category_expr = ibis.literal(False)
        
        # Use normalized category name
        category_expressions[category_name] = category_expr
    
    return category_expressions


def build_all_expressions(dictionaries: Dict, word_column_mapping: Dict[str, str], table) -> Tuple[Dict, Dict]:
    """Build expressions for all dictionaries and their categories."""
    dict_expressions = {}
    dict_category_expressions = {}  # Key: (dict_name, category_name)
    
    for dict_name, dictionary in dictionaries.items():
        word_sets = dictionary.get("word_sets", {})
        categories = dictionary.get("categories", {})
        
        # Build word set expressions with dictionary-specific naming
        word_set_expressions = build_word_set_expressions(word_sets, word_column_mapping, table, dict_name)
        
        # Build category expressions
        category_expressions = build_category_expressions(categories, word_set_expressions, dict_name)
        
        # Dictionary matches if ANY category matches (OR of categories)
        if category_expressions:
            dict_expr = reduce(or_, category_expressions.values()).fill_null(False)
        else:
            dict_expr = ibis.literal(False)
        
        dict_expressions[dict_name] = dict_expr
        
        # Store category expressions with (dict, category) key
        for cat_name, cat_expr in category_expressions.items():
            dict_category_expressions[(dict_name, cat_name)] = cat_expr
    
    return dict_expressions, dict_category_expressions


def calculate_article_weights(table, dict_expressions: Dict[str, ir.BooleanColumn]) -> ir.IntegerColumn:
    """Calculate weight for each article (sum of dictionary matches)."""
    weight_expr = ibis.literal(0)
    for dict_expr in dict_expressions.values():
        weight_expr = weight_expr + dict_expr.cast('int32')
    
    return weight_expr


def calculate_category_weights(table, dict_category_expressions: Dict[Tuple[str, str], ir.BooleanColumn]) -> Dict[str, ir.IntegerColumn]:
    """Calculate category weights (sum of (dictionary, category) matches)."""
    category_weight_exprs = {}
    
    # Initialize all standard categories
    for category in STANDARD_CATEGORIES:
        category_weight_exprs[category] = ibis.literal(0)
    
    # Sum all (dict, category) pairs for each category
    for (dict_name, cat_name), cat_expr in dict_category_expressions.items():
        if cat_name in category_weight_exprs:
            category_weight_exprs[cat_name] = category_weight_exprs[cat_name] + cat_expr.cast('int32')
    
    return category_weight_exprs


def calculate_category_distribution(spike_articles, dict_category_expressions: Dict[Tuple[str, str], ir.BooleanColumn], article_weight_expr: ir.IntegerColumn) -> Dict[str, float]:
    """
    Calculate category distribution as:
    category_contribution = sum((dict, category) matches) / sum(dict matches)
    """
    # First calculate total article weight
    total_weight = spike_articles.aggregate(
        total=article_weight_expr.sum()
    ).execute()['total'].iloc[0]
    
    if total_weight == 0:
        return {}
    
    # Calculate category weights
    category_weight_exprs = calculate_category_weights(spike_articles, dict_category_expressions)
    
    category_distribution = {}
    for category, weight_expr in category_weight_exprs.items():
        cat_weight = spike_articles.aggregate(
            weight=weight_expr.sum()
        ).execute()['weight'].iloc[0]
        
        if cat_weight > 0:
            # Category contribution = category weight / total weight
            category_distribution[category] = (cat_weight / total_weight) * 100
    
    # Sort by percentage
    return dict(sorted(category_distribution.items(), key=lambda x: x[1], reverse=True))


def main():
    """Analyze Step2-unique spike articles with normalized categories."""
    
    # Load Step2-unique spikes
    project_root = Path(__file__).parent.parent
    spike_dir = project_root / "data" / "gpr_spikes" / "spike_comparison"
    
    step2_unique = pd.read_csv(
        spike_dir / "step2_unique_spikes.csv", 
        parse_dates=['start_date', 'end_date', 'peak_date']
    )
    
    logger.info(f"Analyzing {len(step2_unique)} Step2-unique spikes")
    
    # Load dictionaries and word mapping
    dictionaries, word_column_mapping = load_step2_dictionaries_and_word_mapping()
    
    # Connect to database
    logger.info("Connecting to database...")
    conn = ibis.connect(f"duckdb://{DB_PATH}")
    
    # Get the cached binary matrix table
    try:
        cached_table = conn.table(CACHED_TABLE_NAME)
        logger.info(f"Connected to cached table: {CACHED_TABLE_NAME}")
    except Exception as e:
        logger.error(f"Could not access cached table '{CACHED_TABLE_NAME}': {e}")
        return
    
    # Check columns
    logger.info(f"Table has {len(cached_table.columns)} columns")
    word_cols = [col for col in cached_table.columns if col.startswith('contains_word_')]
    logger.info(f"Found {len(word_cols)} word columns")
    
    # Get target outlets
    all_outlets = cached_table.select('outlet').distinct().to_pandas()['outlet'].tolist()
    target_outlets = [o for o in all_outlets if any(p in o for p in NEWSPAPER_PATTERNS)]
    logger.info(f"Target outlets: {target_outlets}")
    
    # Analyze each spike
    all_spike_analyses = []
    
    for spike_idx, spike_row in step2_unique.iterrows():
        spike_id = f"step2_unique_{spike_idx + 1:02d}"
        
        # Convert dates
        article_start_date, article_end_date = convert_weekly_spike_to_date_range(
            spike_row['start_date'], 
            spike_row['end_date']
        )
        
        spike_info = {
            'spike_id': spike_id,
            'start_date': spike_row['start_date'].strftime('%Y-%m-%d'),
            'end_date': spike_row['end_date'].strftime('%Y-%m-%d'),
            'article_start_date': article_start_date,
            'article_end_date': article_end_date,
            'peak_date': spike_row['peak_date'].strftime('%Y-%m-%d'),
            'peak_value': float(spike_row['peak_value']),
            'duration_weeks': int(spike_row['duration_weeks'])
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {spike_id}: {spike_info['start_date']} to {spike_info['end_date']}")
        
        # Filter articles for spike period
        spike_articles = cached_table.filter(
            (cached_table.extracted_publish_date >= article_start_date) &
            (cached_table.extracted_publish_date <= article_end_date) &
            (cached_table.outlet.isin(target_outlets))
        )
        
        # Count total articles
        total_count = spike_articles.count().execute()
        logger.info(f"Total articles in period: {total_count}")
        
        if total_count == 0:
            logger.warning(f"No articles found for {spike_id}")
            spike_analysis = {
                'spike_info': spike_info,
                'total_articles': 0,
                'gpr_articles': 0,
                'gpr_coverage_rate': 0,
                'category_distribution': {}
            }
        else:
            # Build all expressions
            logger.info("Building dictionary and category expressions...")
            dict_expressions, dict_category_expressions = build_all_expressions(
                dictionaries, word_column_mapping, spike_articles
            )
            
            # Build overall GPR expression (OR of all dictionaries)
            overall_gpr_expr = reduce(or_, dict_expressions.values()).fill_null(False)
            
            # Count GPR articles
            gpr_articles = spike_articles.filter(overall_gpr_expr)
            gpr_count = gpr_articles.count().execute()
            
            logger.info(f"GPR articles: {gpr_count}/{total_count} ({gpr_count/total_count*100:.1f}%)")
            
            # Calculate article weights
            weight_expr = calculate_article_weights(spike_articles, dict_expressions)
            
            # Get weight distribution
            articles_with_weight_col = spike_articles.mutate(weight=weight_expr)
            weight_stats = articles_with_weight_col.aggregate(
                avg_weight=articles_with_weight_col.weight.mean(),
                max_weight=articles_with_weight_col.weight.max(),
                articles_with_weight=articles_with_weight_col.weight.cast('int32').sum()
            ).execute()
            
            logger.info(f"Average weight: {weight_stats['avg_weight'].iloc[0]:.2f}")
            logger.info(f"Max weight: {weight_stats['max_weight'].iloc[0]}")
            
            # Calculate category distribution using the correct formula
            category_distribution = calculate_category_distribution(
                spike_articles, dict_category_expressions, weight_expr
            )
            
            if category_distribution:
                logger.info("Category contribution (% of total dictionary matches):")
                for cat, pct in list(category_distribution.items())[:5]:
                    logger.info(f"  {cat}: {pct:.1f}%")
            
            # If manageable count, get article details
            article_list = []
            if gpr_count <= 1000:
                logger.info(f"Loading {gpr_count} GPR articles with weights...")
                
                # Add weight column and filter
                articles_with_weight = spike_articles.mutate(
                    weight=weight_expr
                )
                articles_with_weight = articles_with_weight.filter(articles_with_weight.weight > 0)
                
                # Select relevant columns
                articles_df = articles_with_weight.select(
                    'warc_id', 'url', 'outlet', 'title', 'extracted_publish_date', 'weight'
                ).order_by(ibis.desc('weight')).limit(1000).to_pandas()
                
                # Convert to list of dicts
                for _, row in articles_df.iterrows():
                    article_list.append({
                        'warc_id': row['warc_id'],
                        'url': row['url'],
                        'outlet': row['outlet'],
                        'title': row['title'],
                        'publish_date': row['extracted_publish_date'].strftime('%Y-%m-%d') if pd.notna(row['extracted_publish_date']) else None,
                        'weight': int(row['weight'])
                    })
                
                logger.info(f"Loaded {len(article_list)} articles")
            else:
                logger.info(f"Too many articles ({gpr_count}), will need sampling")
            
            spike_analysis = {
                'spike_info': spike_info,
                'total_articles': total_count,
                'gpr_articles': gpr_count,
                'gpr_coverage_rate': (gpr_count / total_count * 100) if total_count > 0 else 0,
                'weight_statistics': {
                    'avg_weight': float(weight_stats['avg_weight'].iloc[0]),
                    'max_weight': int(weight_stats['max_weight'].iloc[0]),
                    'total_dictionaries': len(dictionaries)
                },
                'category_distribution': category_distribution,
                'articles': article_list if article_list else None,
                'needs_sampling': gpr_count > 1000
            }
        
        all_spike_analyses.append(spike_analysis)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"step2_spike_analysis_normalized_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_spike_analyses, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"\nAnalysis saved to: {output_file}")
    
    # Summary
    logger.info("\n=== Summary ===")
    for analysis in all_spike_analyses:
        spike_id = analysis['spike_info']['spike_id']
        total = analysis['total_articles']
        gpr = analysis['gpr_articles']
        rate = analysis['gpr_coverage_rate']
        
        logger.info(f"\n{spike_id}:")
        logger.info(f"  Total articles: {total}")
        logger.info(f"  GPR articles: {gpr} ({rate:.1f}%)")
        
        if 'weight_statistics' in analysis:
            logger.info(f"  Avg weight: {analysis['weight_statistics']['avg_weight']:.2f}")
            logger.info(f"  Max weight: {analysis['weight_statistics']['max_weight']}")
        
        if analysis.get('articles'):
            logger.info(f"  Articles loaded: {len(analysis['articles'])}")
        
        if analysis['category_distribution']:
            top_cats = list(analysis['category_distribution'].items())[:3]
            logger.info(f"  Top categories: {', '.join([f'{c}: {p:.1f}%' for c, p in top_cats])}")
    
    logger.info("\n=== Category Distribution Formula ===")
    logger.info("For each category:")
    logger.info("  contribution = sum((dict, category) matches) / sum(dict matches)")
    logger.info("This gives the percentage of total dictionary matches attributed to each category")
    
    conn.disconnect()
    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()