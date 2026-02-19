#%% [markdown]
# # Category Peak Event Analysis
# 
# This script analyzes peak events for each individual GPR category by:
# 1. Loading peak week data from category contributions analysis
# 2. Sampling articles that specifically match each category during its peak week
# 3. Using LLM to analyze what geopolitical events occurred for each category
# 
# Unlike the overall GPR spike analysis, this focuses on category-specific peaks
# to understand what types of events drive each category's highest values.

#%%
import ibis
from ibis import _
import polars as pl
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from datetime import date, datetime, timedelta
import time
import argparse
import json
import os
from dotenv import load_dotenv

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from adjustText import adjust_text
import matplotlib.dates as mdates
import seaborn as sns

from gpr_zero.gpr_index.calculator import GPRCalculator
from gpr_zero.llm.prompt_manager import PromptManager
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#%% [markdown]
# ## Configuration
# 
# Set up paths, database connections, and analysis parameters.

#%%
# CJK Font Setup
ensure_cjk_font_setup()

# Database configuration
CC_NEWS_DB_FILES = [
    "/home5/chuenho/data/news/stage2_enhanced_full.duckdb",
]

# Dictionary configuration
DICTIONARIES = {
    "lau_2024": "data/dictionaries/yeu_tong_lau_2024.json",
    "president": "data/dictionaries/taiwan_gpr_dict_president.json"
}

# Output directory
OUTPUT_DIR = Path("data_analysis/output/category_peak_events")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Newspaper mapping
NEWSPAPER_COMPARISON_MAP = {
    "中國時報": "China Times (中國時報)",
    "自由時報": "Liberty Times (自由時報)",
    "聯合報": "United Daily (聯合報)"
}
TARGET_CHINESE_NEWSPAPERS = list(NEWSPAPER_COMPARISON_MAP.keys())

# Analysis parameters
ARTICLE_SAMPLE_SIZE = 150  # Number of articles to sample for each category peak
MAX_CONTENT_PREVIEW_LEN = 600  # Max length of content preview
PEAK_WINDOW_DAYS = 6  # Days after peak week to include in analysis

#%% [markdown]
# ## Helper Functions
# 
# Functions for loading peak data, sampling articles, and managing LLM analysis.

#%%
def load_peak_data(contributions_file: str) -> pl.DataFrame:
    """Load category contributions data with peak weeks."""
    try:
        peak_data = pl.read_csv(contributions_file)
        logger.info(f"Loaded peak data from {contributions_file}")
        logger.info(f"Found {len(peak_data)} category peaks to analyze")
        return peak_data
    except Exception as e:
        logger.error(f"Error loading peak data from {contributions_file}: {e}")
        return pl.DataFrame()

def get_category_articles_for_peak(
    conn_ccnews,
    gpr_calculator: GPRCalculator,
    category_name: str,
    peak_week: str,
    actual_ccnews_targets: list,
    cc_outlet_to_chinese_name: dict
) -> pl.DataFrame:
    """
    Get articles that match a specific category during its peak week.
    
    Args:
        conn_ccnews: Database connection
        gpr_calculator: GPR calculator instance
        category_name: Name of the category to analyze
        peak_week: Peak week date string (YYYY-MM-DD)
        actual_ccnews_targets: List of outlet names
        cc_outlet_to_chinese_name: Mapping of outlets to Chinese names
    
    Returns:
        DataFrame with articles matching the category during peak week
    """
    try:
        # Parse peak week date
        peak_date = datetime.strptime(peak_week, '%Y-%m-%d').date()
        end_date = peak_date + timedelta(days=PEAK_WINDOW_DAYS)
        
        logger.info(f"Fetching {category_name} articles from {peak_date} to {end_date}")
        
        # Filter articles for the peak week period
        articles_filtered = conn_ccnews.table("articles").filter(
            (_.warc_date.cast('date') >= ibis.literal(peak_date.isoformat()).cast('date')) &
            (_.warc_date.cast('date') <= ibis.literal(end_date.isoformat()).cast('date')) &
            (_.outlet.isin(actual_ccnews_targets))
        )
        
        # Map outlets to Chinese names
        case_tuples = []
        for outlet_pattern, zh_name in cc_outlet_to_chinese_name.items():
            actual_outlet = next((ao for ao in actual_ccnews_targets if outlet_pattern in ao), None)
            if actual_outlet:
                case_tuples.append((articles_filtered.outlet == actual_outlet, zh_name))
        
        if not case_tuples:
            logger.error("No outlet mapping found")
            return pl.DataFrame()
        
        # Prepare articles for GPR analysis
        articles_prepared = articles_filtered.mutate(
            newspaper_zh=ibis.cases(*case_tuples, else_=None),
            source_file=ibis.literal("CC-NEWS_CategoryPeak"),
            category=ibis.literal(""),
            date=_.warc_date.cast('timestamp')
        ).select(['date', 'content', 'newspaper_zh', 'source_file', 'category', 'title', 'outlet', 'warc_date', 'extracted_publish_date'])
        
        # Filter out null mappings
        articles_prepared = articles_prepared.filter(_.newspaper_zh.notnull())
        
        # Use GPR calculator to identify articles matching the specific category
        logger.info(f"Preparing articles for {category_name} analysis...")
        
        # Get the prepared table with category labels
        prepared_table = gpr_calculator.prepare(articles_prepared)
        
        # Convert to Polars for filtering
        prepared_pl = prepared_table.to_polars()
        
        # Filter articles that match the specific category
        if category_name == "overall":
            # For overall, get articles that match any category
            category_articles = prepared_pl.filter(pl.col("matches_any_category") == True)
        else:
            # For specific categories, check the category-specific column
            category_col = f"is_{category_name}"
            if category_col in prepared_pl.columns:
                category_articles = prepared_pl.filter(pl.col(category_col) == True)
            else:
                logger.warning(f"Category column '{category_col}' not found. Available columns: {prepared_pl.columns}")
                return pl.DataFrame()
        
        if category_articles.is_empty():
            logger.warning(f"No articles found matching category '{category_name}' during peak week")
            return pl.DataFrame()
        
        # Sample articles randomly
        sample_size = min(ARTICLE_SAMPLE_SIZE, len(category_articles))
        sampled_articles = category_articles.sample(n=sample_size, seed=42)
        
        logger.info(f"Found {len(category_articles)} {category_name} articles, sampled {len(sampled_articles)}")
        
        return sampled_articles
        
    except Exception as e:
        logger.error(f"Error getting {category_name} articles for peak {peak_week}: {e}")
        return pl.DataFrame()

def prepare_articles_summary(articles_df: pl.DataFrame) -> str:
    """Prepare a summary of articles for LLM analysis."""
    if articles_df.is_empty():
        return "No articles found for this category peak."
    
    summary_parts = []
    for article in articles_df.sort("warc_date").iter_rows(named=True):
        title = article.get('title', 'No Title')
        content_preview = article.get('content', '')
        if len(content_preview) > MAX_CONTENT_PREVIEW_LEN:
            content_preview = content_preview[:MAX_CONTENT_PREVIEW_LEN] + "..."
        
        pub_date = article.get('extracted_publish_date', article.get('warc_date', 'Unknown'))
        if hasattr(pub_date, 'strftime'):
            pub_date_str = pub_date.strftime('%Y-%m-%d')
        else:
            pub_date_str = str(pub_date)
        
        outlet = article.get('outlet', 'Unknown Outlet')
        
        summary_parts.append(
            f"Date: {pub_date_str}\\n"
            f"Outlet: {outlet}\\n"
            f"Title: {title}\\n"
            f"Content: {content_preview}\\n"
            f"---\\n"
        )
    
    return "\\n".join(summary_parts)

#%% [markdown]
# ## Main Analysis Function
# 
# Analyze category peak events using LLM to understand what happened.

#%%
def analyze_category_peaks(
    dictionary_name: str,
    contributions_file: str | None = None,
    max_llm_calls: int | None = None
):
    """
    Main function to analyze category peak events.
    
    Args:
        dictionary_name: Name of dictionary to analyze
        contributions_file: Path to category contributions CSV file
        max_llm_calls: Maximum number of LLM calls to make
    """
    
    # Create output directory for this dictionary
    dict_output_dir = OUTPUT_DIR / f"{dictionary_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dict_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing category peaks for {dictionary_name}")
    logger.info(f"Output directory: {dict_output_dir}")
    
    # Load peak data
    if contributions_file is None:
        contributions_file = f"data_analysis/output/category_contributions/category_contributions_{dictionary_name}.csv"
    
    peak_data = load_peak_data(contributions_file)
    if peak_data.is_empty():
        logger.error("No peak data loaded. Exiting.")
        return
    
    # Filter to exclude overall and ensure valid peak weeks
    dict_peaks = peak_data.filter(
        (pl.col("category") != "overall") &
        (pl.col("peak_week") != "N/A") &
        (pl.col("peak_week").is_not_null())
    )
    
    if dict_peaks.is_empty():
        logger.error(f"No valid category peaks found for {dictionary_name}")
        return
    
    logger.info(f"Found {len(dict_peaks)} category peaks to analyze")
    
    # Initialize GPR Calculator
    dict_path = DICTIONARIES.get(dictionary_name)
    if not dict_path or not Path(dict_path).exists():
        logger.error(f"Dictionary not found: {dict_path}")
        return
    
    try:
        gpr_calculator = GPRCalculator(dictionary_path=dict_path)
        logger.info(f"GPR Calculator initialized for {dictionary_name}")
    except Exception as e:
        logger.error(f"Error initializing GPR Calculator: {e}")
        return
    
    # Connect to database
    try:
        db_file_path = CC_NEWS_DB_FILES[0]
        conn_ccnews = ibis.connect(f"duckdb://{db_file_path}?access_mode=read_only")
        logger.info(f"Connected to database: {db_file_path}")
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return
    
    # Map outlet names
    try:
        articles_view = conn_ccnews.table("articles")
        all_outlets = articles_view.select('outlet').distinct().to_polars()['outlet'].to_list()
        
        cc_outlet_to_chinese_name = {}
        actual_ccnews_targets = []
        
        for chinese_name, pattern in NEWSPAPER_COMPARISON_MAP.items():
            matched_outlet = next((o for o in all_outlets if pattern in o), None)
            if matched_outlet:
                logger.info(f"Mapping outlet '{matched_outlet}' to '{chinese_name}'")
                actual_ccnews_targets.append(matched_outlet)
                cc_outlet_to_chinese_name[matched_outlet] = chinese_name
        
        if not actual_ccnews_targets:
            logger.error("No target newspapers found")
            return
            
    except Exception as e:
        logger.error(f"Error mapping outlet names: {e}")
        return
    
    # Initialize LLM
    prompt_manager = PromptManager(
        log_dir=dict_output_dir / "llm_logs", 
        model_name="google/gemini-2.5-flash-lite-preview-06-17"
    )
    
    # Prepare prompt template
    PROMPT_TEMPLATE_FILE = Path("prompts/analyze_category_peak_event.json")
    if not PROMPT_TEMPLATE_FILE.exists():
        logger.error(f"Prompt template not found: {PROMPT_TEMPLATE_FILE}")
        return
    
    # Analyze each category peak
    analysis_results = []
    llm_calls_made = 0
    
    for peak_row in dict_peaks.iter_rows(named=True):
        category_name = peak_row['category']
        peak_week = peak_row['peak_week']
        peak_share = peak_row['peak_share']
        contribution_percent = peak_row['contribution_percent']
        
        logger.info(f"\\nAnalyzing {category_name} peak on {peak_week} ({peak_share:.4f} share, {contribution_percent:.2f}% contribution)")
        
        # Check LLM call limit
        if max_llm_calls is not None and llm_calls_made >= max_llm_calls:
            logger.info(f"LLM call limit ({max_llm_calls}) reached. Skipping remaining analyses.")
            break
        
        # Get category-specific articles for the peak week
        category_articles = get_category_articles_for_peak(
            conn_ccnews, gpr_calculator, category_name, peak_week,
            actual_ccnews_targets, cc_outlet_to_chinese_name
        )
        
        if category_articles.is_empty():
            logger.warning(f"No articles found for {category_name} peak on {peak_week}")
            continue
        
        # Prepare articles summary for LLM
        articles_summary = prepare_articles_summary(category_articles)
        
        # Prepare LLM analysis
        result_entry = {
            "dictionary": dictionary_name,
            "category": category_name,
            "peak_week": peak_week,
            "peak_share": peak_share,
            "contribution_percent": contribution_percent,
            "articles_analyzed": len(category_articles),
            "llm_analysis": "N/A",
            "event_summary": "N/A",
            "keywords": "N/A"
        }
        
        # Call LLM if client is available
        if prompt_manager.client and articles_summary != "No articles found for this category peak.":
            template_args = [
                {},
                {
                    "category_name": category_name,
                    "peak_week": peak_week,
                    "peak_share": peak_share,
                    "contribution_percent": contribution_percent,
                    "articles_summary": articles_summary,
                    "dictionary_name": dictionary_name
                }
            ]
            
            logger.info(f"Calling LLM for {category_name} peak analysis (Call #{llm_calls_made + 1})")
            time.sleep(2)  # Rate limiting
            
            try:
                llm_response = prompt_manager.call_llm_with_prompt_file(
                    prompt_template_file=PROMPT_TEMPLATE_FILE,
                    template_args_list=template_args
                )
                llm_calls_made += 1
                
                if not llm_response.get("error") and llm_response.get("text"):
                    try:
                        parsed_response = json.loads(llm_response["text"])
                        result_entry["llm_analysis"] = parsed_response.get("analysis", "Parsing error")
                        result_entry["event_summary"] = parsed_response.get("event_summary", "Parsing error")
                        result_entry["keywords"] = parsed_response.get("keywords", "Parsing error")
                        logger.info(f"LLM analysis completed for {category_name}")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse LLM JSON response for {category_name}")
                        result_entry["llm_analysis"] = llm_response["text"]
                elif llm_response.get("error"):
                    logger.error(f"LLM API error for {category_name}: {llm_response['error']}")
                    result_entry["llm_analysis"] = f"API Error: {llm_response['error']}"
                elif llm_response.get("mock"):
                    logger.info(f"LLM call mocked for {category_name}")
                    result_entry["llm_analysis"] = "Mock Response (No API Key)"
                    
            except Exception as e:
                logger.error(f"Exception during LLM call for {category_name}: {e}")
                result_entry["llm_analysis"] = f"Call Exception: {e}"
        else:
            logger.warning(f"LLM client not available or no articles for {category_name}")
        
        analysis_results.append(result_entry)
    
    # Save results
    if analysis_results:
        results_df = pl.DataFrame(analysis_results)
        results_file = dict_output_dir / f"category_peak_analysis_{dictionary_name}.csv"
        results_df.write_csv(results_file)
        logger.info(f"Analysis results saved to {results_file}")
        
        # Log summary
        logger.info(f"\\n=== ANALYSIS SUMMARY for {dictionary_name} ===")
        for result in analysis_results:
            event_summary = result["event_summary"]
            if event_summary and event_summary != "N/A" and "error" not in event_summary.lower():
                logger.info(f"{result['category']} ({result['peak_week']}): {event_summary}")
            else:
                logger.info(f"{result['category']} ({result['peak_week']}): Analysis incomplete")
    
    # Cleanup
    conn_ccnews.disconnect()
    logger.info(f"Category peak analysis completed for {dictionary_name}")

#%% [markdown]
# ## Command Line Interface
# 
# Allow the script to be run with different parameters.

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze category peak events using LLM analysis"
    )
    parser.add_argument(
        "dictionary",
        choices=list(DICTIONARIES.keys()),
        help="Dictionary to analyze"
    )
    parser.add_argument(
        "--contributions-file",
        type=str,
        default=None,
        help="Path to category contributions CSV file (default: auto-detect)"
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=None,
        help="Maximum number of LLM calls to make (default: no limit)"
    )
    
    args = parser.parse_args()
    analyze_category_peaks(args.dictionary, args.contributions_file, args.max_llm_calls) 