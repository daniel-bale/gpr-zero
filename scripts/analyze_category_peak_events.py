#!/usr/bin/env python3
"""
Task 5 (continued): Analyze peak events for each category.
Samples articles from each category's peak week and generates LLM summaries.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ibis
from ibis import _
import ibis.expr.types as ir
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from functools import reduce
from operator import and_, or_
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from gpr_zero.llm.prompt_manager import PromptManager

# Load environment variables
load_dotenv()

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
OUTPUT_DIR = Path("data/category_contributions/peak_events")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The cached table name
CACHED_TABLE_NAME = "gpr_binary_matrix_all_words"

# Newspapers to analyze
NEWSPAPER_PATTERNS = ["China Times", "Liberty Times", "United Daily"]

# Number of articles to sample per category peak
ARTICLES_PER_PEAK = 100

# Standard categories and mappings (same as before)
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
    "military_escalation": "Military Escalation",
    "Diplomatic Isolation": "Diplomatic Isolation",
    "Diplomatic Isolation (外交孤立)": "Diplomatic Isolation",
    "外交孤立": "Diplomatic Isolation",
    "diplomatic_isolation": "Diplomatic Isolation",
    "Economic Coercion": "Economic Coercion",
    "Economic Coercion (經濟脅迫)": "Economic Coercion",
    "經濟脅迫": "Economic Coercion",
    "economic_coercion": "Economic Coercion",
    "Regional Security": "Regional Security",
    "Regional Security (區域安全)": "Regional Security",
    "區域安全": "Regional Security",
    "regional_security": "Regional Security",
    "Internal Political Instability": "Internal Political Instability",
    "Internal Political Instability (內部政治不穩定)": "Internal Political Instability",
    "內部政治不穩定": "Internal Political Instability",
    "internal_political_instability": "Internal Political Instability",
    "Gray Zone Operations": "Gray Zone Operations",
    "Gray Zone Operations (灰色地帶行動)": "Gray Zone Operations",
    "灰色地帶行動": "Gray Zone Operations",
    "gray_zone_operations": "Gray Zone Operations",
    "US-China-Taiwan Relations": "US-China-Taiwan Relations",
    "US-China-Taiwan Relations (美中台關係)": "US-China-Taiwan Relations",
    "美中台關係": "US-China-Taiwan Relations",
    "us_china_taiwan_relations": "US-China-Taiwan Relations",
    "Cyber and Information Warfare": "Cyber and Information Warfare",
    "Cyber and Information Warfare (網路與資訊戰)": "Cyber and Information Warfare",
    "網路與資訊戰": "Cyber and Information Warfare",
    "cyber_and_information_warfare": "Cyber and Information Warfare"
}


def load_peak_weeks():
    """Load the most recent peak weeks analysis."""
    # Find the most recent peak weeks file
    peak_files = list(Path("data/category_contributions").glob("step2_category_peak_weeks_*.csv"))
    if not peak_files:
        raise FileNotFoundError("No peak weeks analysis found. Run calculate_step2_category_contributions.py first.")
    
    latest_file = max(peak_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading peak weeks from: {latest_file}")
    
    return pd.read_csv(latest_file, parse_dates=['peak_week'])


def load_step2_dictionaries_and_word_mapping():
    """Load all Step2 dictionaries with normalized categories."""
    project_root = Path(__file__).parent.parent
    step2_batch_dir = project_root / "data" / "dictionaries" / "step2_batch"
    
    step2_files = list(step2_batch_dir.glob("*.json"))
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
                        normalized_categories[cat_name] = cat_def
                
                dict_data['categories'] = normalized_categories
            
            dictionaries[dict_name] = dict_data
            
            # Collect all words
            if 'word_sets' in dict_data:
                for word_set in dict_data['word_sets'].values():
                    all_words.update(word_set)
    
    # Create word-to-column mapping
    sorted_words = sorted(all_words)
    word_column_mapping = {}
    for i, word in enumerate(sorted_words):
        col_name = f"contains_word_{i:04d}"
        word_column_mapping[word] = col_name
    
    return dictionaries, word_column_mapping


def build_category_specific_expressions(category: str, dictionaries: Dict, word_column_mapping: Dict[str, str], table):
    """Build expressions for a specific category across all dictionaries."""
    category_expressions = []
    
    for dict_name, dictionary in dictionaries.items():
        word_sets = dictionary.get("word_sets", {})
        categories = dictionary.get("categories", {})
        
        if category not in categories:
            continue
        
        # Build word set expressions for this dictionary
        word_set_expressions = {}
        for set_name, words in word_sets.items():
            relevant_word_columns = []
            for word in words:
                if word in word_column_mapping and word_column_mapping[word] in table.columns:
                    relevant_word_columns.append(word_column_mapping[word])
            
            if relevant_word_columns:
                word_columns_exprs = [table[col] for col in relevant_word_columns]
                word_set_expr = reduce(or_, word_columns_exprs)
                word_set_expressions[set_name] = word_set_expr
        
        # Build category expression
        query_string = categories[category]
        word_set_names = [name.strip() for name in query_string.split(" AND ")]
        
        has_expressions = []
        for set_name in word_set_names:
            if set_name in word_set_expressions:
                has_expressions.append(word_set_expressions[set_name])
        
        if has_expressions:
            category_expr = reduce(and_, has_expressions, ibis.literal(True)).fill_null(False)
            category_expressions.append(category_expr)
    
    # Combine all dictionary expressions for this category with OR
    if category_expressions:
        return reduce(or_, category_expressions).fill_null(False)
    else:
        return ibis.literal(False)


def sample_category_peak_articles(conn, category: str, peak_week, dictionaries, word_column_mapping):
    """Sample articles from a category's peak week weighted by category match strength."""
    # Handle peak_week formatting
    if hasattr(peak_week, 'strftime'):
        peak_week_str = peak_week.strftime('%Y-%m-%d')
        week_start = peak_week
    else:
        peak_week_str = str(peak_week)[:10]
        week_start = pd.to_datetime(peak_week_str)
    
    logger.info(f"Sampling articles for {category} peak week: {peak_week_str}")
    
    # Connect to tables
    cached_table = conn.table(CACHED_TABLE_NAME)
    articles_table = conn.table("articles")
    
    # Calculate week range
    week_end = week_start + timedelta(days=6)
    
    # Get target outlets
    all_outlets = cached_table.select('outlet').distinct().to_pandas()['outlet'].tolist()
    target_outlets = [o for o in all_outlets if any(p in o for p in NEWSPAPER_PATTERNS)]
    
    # Filter for peak week and target outlets
    week_articles = cached_table.filter(
        (cached_table.extracted_publish_date >= week_start.strftime('%Y-%m-%d')) &
        (cached_table.extracted_publish_date <= week_end.strftime('%Y-%m-%d')) &
        (cached_table.outlet.isin(target_outlets))
    )
    
    # Build category-specific expressions for all dictionaries
    category_weight_expr = ibis.literal(0)
    
    for dict_name, dictionary in dictionaries.items():
        word_sets = dictionary.get("word_sets", {})
        categories = dictionary.get("categories", {})
        
        if category not in categories:
            continue
        
        # Build word set expressions for this dictionary
        word_set_expressions = {}
        for set_name, words in word_sets.items():
            relevant_word_columns = []
            for word in words:
                if word in word_column_mapping and word_column_mapping[word] in week_articles.columns:
                    relevant_word_columns.append(word_column_mapping[word])
            
            if relevant_word_columns:
                word_columns_exprs = [week_articles[col] for col in relevant_word_columns]
                word_set_expr = reduce(or_, word_columns_exprs)
                word_set_expressions[set_name] = word_set_expr
        
        # Build category expression
        query_string = categories[category]
        word_set_names = [name.strip() for name in query_string.split(" AND ")]
        
        has_expressions = []
        for set_name in word_set_names:
            if set_name in word_set_expressions:
                has_expressions.append(word_set_expressions[set_name])
        
        if has_expressions:
            category_expr = reduce(and_, has_expressions, ibis.literal(True)).fill_null(False)
            # Add to weight (count how many dictionaries match this category)
            category_weight_expr = category_weight_expr + category_expr.cast('int32')
    
    # Filter articles matching this category (weight > 0)
    category_articles_with_weight = week_articles.mutate(
        category_weight=category_weight_expr
    ).filter(_.category_weight > 0)
    
    # Count matches
    total_matches = category_articles_with_weight.count().execute()
    logger.info(f"Found {total_matches} articles matching {category} in peak week")
    
    if total_matches == 0:
        return pd.DataFrame()
    
    # Get all articles with weights for sampling
    articles_with_weights = category_articles_with_weight.select(
        'warc_id', 'url', 'outlet', 'title', 'extracted_publish_date', 'category_weight'
    ).to_pandas()
    
    # Sample articles weighted by category_weight
    sample_size = min(ARTICLES_PER_PEAK, total_matches)
    
    if sample_size < total_matches:
        # Weighted sampling - articles with higher weights more likely to be selected
        weights = articles_with_weights['category_weight'].values
        weights = weights / weights.sum()  # Normalize to probabilities
        
        sampled_indices = np.random.choice(
            len(articles_with_weights), 
            size=sample_size, 
            replace=False, 
            p=weights
        )
        sampled_articles = articles_with_weights.iloc[sampled_indices]
    else:
        sampled_articles = articles_with_weights
    
    # Get full content
    warc_ids = sampled_articles['warc_id'].tolist()
    content_df = articles_table.filter(
        articles_table.warc_id.isin(warc_ids)
    ).select('warc_id', 'content').to_pandas()
    
    # Merge content
    result = sampled_articles.merge(content_df, on='warc_id', how='left')
    
    logger.info(f"Sampled {len(result)} articles (avg weight: {result['category_weight'].mean():.2f})")
    
    return result


# Function no longer needed - we'll use the template directly


# Define serialization function at module level
def serialize_data(obj):
    """Serialize various data types for JSON output."""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main():
    logger.info("=== Task 5 (continued): Analyze Category Peak Events ===")
    
    # Load peak weeks data
    peak_weeks_df = load_peak_weeks()
    logger.info(f"Loaded {len(peak_weeks_df)} category peak weeks")
    
    # Load dictionaries
    dictionaries, word_column_mapping = load_step2_dictionaries_and_word_mapping()
    logger.info(f"Loaded {len(dictionaries)} Step2 dictionaries")
    
    # Initialize PromptManager
    logger.info("Initializing PromptManager...")
    prompt_manager = PromptManager()
    
    # Connect to database
    conn = ibis.connect(f"duckdb://{DB_PATH}")
    
    # Process each category peak
    all_peak_analyses = []
    
    for _, peak_row in peak_weeks_df.iterrows():
        category = peak_row['category']
        peak_week = peak_row['peak_week']
        
        logger.info(f"\n{'='*60}")
        # Handle peak_week formatting - it might be string or datetime
        if hasattr(peak_week, 'strftime'):
            peak_week_str = peak_week.strftime('%Y-%m-%d')
        else:
            peak_week_str = str(peak_week)[:10]  # Get YYYY-MM-DD part
        logger.info(f"Processing {category} peak: {peak_week_str}")
        
        # Sample articles
        articles_df = sample_category_peak_articles(
            conn, category, peak_week, dictionaries, word_column_mapping
        )
        
        if len(articles_df) == 0:
            logger.warning(f"No articles found for {category} peak week")
            continue
        
        # Get LLM analysis
        try:
            # Format the prompt properly for the template
            # Create a summary of articles for the template
            articles_summary = ""
            for idx, (_, row) in enumerate(articles_df.head(30).iterrows()):
                weight = row['category_weight'] if 'category_weight' in row else 'N/A'
                articles_summary += f"""\nArticle {idx + 1} (Weight: {weight}):
- Date: {row['extracted_publish_date']}
- Outlet: {row['outlet']}
- Title: {row['title']}
- Content Preview: {row['content'][:300] if pd.notna(row['content']) else 'N/A'}...\n"""
            
            template_args = [
                {},  # Empty dict for system message (no variables)
                {
                    "category_name": category,
                    "dictionary_name": "Step2 (40 dictionaries)",
                    "peak_week": peak_week_str,
                    "peak_share": peak_row.get('peak_contribution_to_index', peak_row.get('peak_contribution', 0)) / 100.0,  # Convert to decimal
                    "contribution_percent": peak_row.get('peak_contribution_to_index', peak_row.get('peak_contribution', 0)),
                    "articles_summary": articles_summary
                }
            ]
            
            llm_response = prompt_manager.call_llm_with_prompt_file(
                prompt_template_file="prompts/analyze_category_peak_event.json",
                template_args_list=template_args
            )
            # Parse the JSON response
            if isinstance(llm_response["text"], str):
                try:
                    analysis_data = json.loads(llm_response["text"])
                    analysis = analysis_data
                except json.JSONDecodeError:
                    analysis = llm_response["text"]
            else:
                analysis = llm_response["text"]
            logger.info("LLM analysis generated successfully")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            analysis = f"Error generating analysis: {str(e)}"
        
        # Store results
        peak_analysis = {
            'category': category,
            'peak_week': peak_week,
            'peak_contribution_to_index': peak_row['peak_contribution_to_index'],
            'peak_proportion_in_gpr': peak_row['peak_proportion_in_gpr'],
            'articles_sampled': len(articles_df),
            'sampled_articles': articles_df.to_dict('records'),
            'llm_analysis': analysis
        }
        
        all_peak_analyses.append(peak_analysis)
        
        # Save individual analysis
        category_file = OUTPUT_DIR / f"{category.replace(' ', '_').lower()}_peak_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Use the module-level serialize_data function
        
        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump(peak_analysis, f, ensure_ascii=False, indent=2, default=serialize_data)
        
        logger.info(f"Saved analysis to: {category_file}")
    
    # Save comprehensive results
    final_output = {
        'analysis_date': datetime.now().isoformat(),
        'total_categories_analyzed': len(all_peak_analyses),
        'peak_analyses': all_peak_analyses
    }
    
    final_file = OUTPUT_DIR / f"all_category_peak_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Use serialize_data function for the final output as well
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2, default=serialize_data)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Analysis complete! Saved to: {final_file}")
    logger.info(f"Total categories analyzed: {len(all_peak_analyses)}")
    
    conn.disconnect()


if __name__ == "__main__":
    main()