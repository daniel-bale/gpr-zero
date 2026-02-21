#%% [markdown]
# # GPR Category Contribution Analysis
# 
# This script calculates the contribution of each category to the overall GPR index
# comparing Lau's 2024 dictionary and the Taiwan president dictionary.
# The analysis shows how much each category contributes to the total GPR index.
# 
# ## Usage
# 
# **Basic usage (run all configured dictionaries):**
# ```bash
# uv run data_analysis/calculate_category_contributions.py
# ```
# 
# **Run specific dictionaries:**
# ```bash
# uv run data_analysis/calculate_category_contributions.py --dictionaries lau_2024
# ```
# 
# **Add custom dictionaries:**
# ```bash
# uv run data_analysis/calculate_category_contributions.py --add-dictionary president /path/to/president_dict.json
# ```
# 
# **Compare both dictionaries:**
# ```bash
# uv run data_analysis/calculate_category_contributions.py --dictionaries lau_2024 president
# ```

#%%
import ibis
from ibis import _
import polars as pl
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from datetime import date, datetime
import argparse
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from gpr_zero.gpr_index.calculator import GPRCalculator
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup CJK font
ensure_cjk_font_setup()

#%% [markdown]
# ## Configuration
# 
# Set up the database connections, dictionaries, and analysis parameters.

#%%
# Database and dictionary configuration
CC_NEWS_DB_FILES = [
    "/home5/chuenho/data/news/stage2_enhanced_full.duckdb",
]

DICTIONARIES = {
    "lau_2024": "data/dictionaries/yeu_tong_lau_2024.json",
    "president": "data/dictionaries/taiwan_gpr_dict_president.json"
}

# Analysis parameters
APP_START_DATE_STR = "2017-01-01"
APP_END_DATE_STR = "2024-12-31"
APP_START_DATE = date.fromisoformat(APP_START_DATE_STR)
APP_END_DATE = date.fromisoformat(APP_END_DATE_STR)
TIME_BUCKET_GPR = 'W'  # Weekly analysis

# Target newspapers
NEWSPAPER_COMPARISON_MAP = {
    "中國時報": "China Times (中國時報)",
    "自由時報": "Liberty Times (自由時報)",
    "聯合報": "United Daily (聯合報)"
}
TARGET_CHINESE_NEWSPAPERS = list(NEWSPAPER_COMPARISON_MAP.keys())

# Output directory
OUTPUT_DIR = Path("data_analysis/output/category_contributions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#%% [markdown]
# ## Helper Functions
# 
# Define functions to calculate category contributions and create visualizations.

#%%
def check_dictionary_exists(dictionary_path: str) -> bool:
    """Check if dictionary file exists."""
    return Path(dictionary_path).exists()

def load_dictionary_info(dictionary_path: str) -> Dict:
    """Load dictionary and extract category information."""
    if not check_dictionary_exists(dictionary_path):
        raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")
    
    with open(dictionary_path, 'r', encoding='utf-8') as f:
        dict_data = json.load(f)
    
    categories = {}
    if 'categories' in dict_data:
        for category_name, category_data in dict_data['categories'].items():
            if isinstance(category_data, dict) and 'terms' in category_data:
                categories[category_name] = len(category_data['terms'])
            elif isinstance(category_data, list):
                categories[category_name] = len(category_data)
    
    return {
        'path': dictionary_path,
        'categories': categories,
        'total_terms': sum(categories.values())
    }

def calculate_category_contributions(
    gpr_calculator: GPRCalculator,
    conn_ccnews,
    actual_ccnews_targets: List[str],
    cc_outlet_to_chinese_name: Dict[str, str]
) -> pl.DataFrame:
    """Calculate GPR contributions for each category."""
    
    # Filter and prepare articles
    articles_ccnews_filtered = conn_ccnews.table("articles").filter(
        (_.warc_date.cast('date') >= ibis.literal(APP_START_DATE_STR).cast('date')) &
        (_.warc_date.cast('date') <= ibis.literal(APP_END_DATE_STR).cast('date')) &
        (_.outlet.isin(actual_ccnews_targets))
    )
    
    # Map outlets to Chinese names
    case_tuples = []
    for outlet_pattern, zh_name in cc_outlet_to_chinese_name.items():
        actual_outlet_for_pattern = next(
            (ao for ao in actual_ccnews_targets if outlet_pattern in ao), None
        )
        if actual_outlet_for_pattern:
            case_tuples.append(
                (articles_ccnews_filtered.outlet == actual_outlet_for_pattern, zh_name)
            )
    
    if not case_tuples:
        logger.error("No outlet mapping found")
        return pl.DataFrame()
    
    articles_ccnews_prepared = articles_ccnews_filtered.mutate(
        newspaper_zh=ibis.cases(*case_tuples, else_=None),
        source_file=ibis.literal("CC-NEWS_Combined"),
        category=ibis.literal(""),
        date=_.warc_date.cast('timestamp')
    ).select(['date', 'content', 'newspaper_zh', 'source_file', 'category'])
    
    # Filter out null mappings
    articles_ccnews_prepared = articles_ccnews_prepared.filter(_.newspaper_zh.notnull())
    
    # Prepare data for GPR calculation
    logger.info("Preparing data for GPR calculation...")
    prepared_table = gpr_calculator.prepare(articles_ccnews_prepared)
    
    # Calculate GPR shares
    logger.info("Calculating GPR shares...")
    gpr_results = gpr_calculator.query(
        group_by_cols=['date', 'newspaper_zh'],
        time_bucket=TIME_BUCKET_GPR,
        time_col='date',
        calculate_share=True
    )
    
    return gpr_results

def find_peak_weeks_for_categories(
    gpr_results: pl.DataFrame,
    categories: List[str]
) -> Dict[str, Dict[str, str | float]]:
    """
    Find the peak week for each category across all newspapers.
    
    Returns a dictionary with category names as keys and peak info as values.
    """
    
    if gpr_results.is_empty() or "date_bucketed" not in gpr_results.columns:
        return {}
    
    peak_weeks = {}
    
    # Process each category
    for category in categories:
        share_col = f"{category}_share"
        if share_col not in gpr_results.columns:
            continue
            
        # Aggregate across all newspapers for each week
        weekly_aggregated = gpr_results.group_by("date_bucketed").agg([
            # Calculate weighted average share for this week
            (pl.col(share_col) * pl.col("total_articles_in_group")).sum().alias("weighted_share"),
            pl.col("total_articles_in_group").sum().alias("total_articles_week")
        ]).with_columns(
            (pl.col("weighted_share") / pl.col("total_articles_week")).alias("avg_share_week")
        ).sort("date_bucketed")
        
        if not weekly_aggregated.is_empty():
            # Find the week with maximum average share
            max_week_idx = weekly_aggregated["avg_share_week"].arg_max()
            if max_week_idx is not None:
                peak_row = weekly_aggregated.row(max_week_idx, named=True)
                peak_weeks[category] = {
                    "peak_week": peak_row["date_bucketed"].strftime("%Y-%m-%d"),
                    "peak_share": peak_row["avg_share_week"],
                    "peak_articles": peak_row["weighted_share"]
                }
    
    # Handle overall category
    if "overall_share" in gpr_results.columns:
        overall_weekly = gpr_results.group_by("date_bucketed").agg([
            (pl.col("overall_share") * pl.col("total_articles_in_group")).sum().alias("weighted_overall"),
            pl.col("total_articles_in_group").sum().alias("total_articles_week")
        ]).with_columns(
            (pl.col("weighted_overall") / pl.col("total_articles_week")).alias("avg_overall_week")
        ).sort("date_bucketed")
        
        if not overall_weekly.is_empty():
            max_overall_idx = overall_weekly["avg_overall_week"].arg_max()
            if max_overall_idx is not None:
                peak_overall = overall_weekly.row(max_overall_idx, named=True)
                peak_weeks["overall"] = {
                    "peak_week": peak_overall["date_bucketed"].strftime("%Y-%m-%d"),
                    "peak_share": peak_overall["avg_overall_week"],
                    "peak_articles": peak_overall["weighted_overall"]
                }
    
    return peak_weeks

def calculate_overall_contributions(
    gpr_results: pl.DataFrame,
    categories: List[str]
) -> pl.DataFrame:
    """
    Calculate category contributions to GPR index and find peak weeks.
    
    Definition: The contribution to the index is the percent of articles in each category 
    satisfying the condition for inclusion in the GPR index, as a share of all articles 
    satisfying that condition.
    """
    
    if gpr_results.is_empty():
        return pl.DataFrame()
    
    # First, calculate total GPR articles (articles satisfying GPR inclusion condition)
    if "overall_share" not in gpr_results.columns:
        logger.error("Missing 'overall_share' column needed for contribution calculation")
        return pl.DataFrame()
    
    # Find peak weeks for each category
    peak_weeks = find_peak_weeks_for_categories(gpr_results, categories)
    
    gpr_results_with_counts = gpr_results.with_columns(
        # Calculate actual count of GPR articles for each category
        **{
            f"{category}_articles": (pl.col(f"{category}_share") * pl.col("total_articles_in_group")).round(0).cast(pl.Int64)
            for category in categories if f"{category}_share" in gpr_results.columns
        },
        # Calculate total GPR articles (overall articles satisfying GPR condition)
        total_gpr_articles=(pl.col("overall_share") * pl.col("total_articles_in_group")).round(0).cast(pl.Int64)
    )
    
    # Sum up all GPR articles across all time periods and newspapers
    total_gpr_articles_sum = gpr_results_with_counts["total_gpr_articles"].sum()
    
    if total_gpr_articles_sum == 0:
        logger.warning("No GPR articles found - cannot calculate contributions")
        return pl.DataFrame()
    
    # Calculate contributions for each category
    contributions = []
    
    for category in categories:
        articles_col = f"{category}_articles"
        if articles_col in gpr_results_with_counts.columns:
            # Sum articles for this category across all time periods and newspapers
            category_articles_sum = gpr_results_with_counts[articles_col].sum()
            
            # Calculate contribution as percentage of total GPR articles
            contribution_percent = (category_articles_sum / total_gpr_articles_sum) * 100
            
            # Get peak week info
            peak_info = peak_weeks.get(category, {})
            
            contributions.append({
                "category": category,
                "contribution_percent": contribution_percent,
                "category_articles": category_articles_sum,
                "total_gpr_articles": total_gpr_articles_sum,
                "contribution_ratio": category_articles_sum / total_gpr_articles_sum,
                "peak_week": peak_info.get("peak_week", "N/A"),
                "peak_share": peak_info.get("peak_share", 0.0),
                "peak_articles": peak_info.get("peak_articles", 0.0)
            })
    
    # Add overall as reference (should be 100% by definition)
    overall_peak_info = peak_weeks.get("overall", {})
    contributions.append({
        "category": "overall",
        "contribution_percent": 100.0,
        "category_articles": total_gpr_articles_sum,
        "total_gpr_articles": total_gpr_articles_sum,
        "contribution_ratio": 1.0,
        "peak_week": overall_peak_info.get("peak_week", "N/A"),
        "peak_share": overall_peak_info.get("peak_share", 0.0),
        "peak_articles": overall_peak_info.get("peak_articles", 0.0)
    })
    
    return pl.DataFrame(contributions)

def create_contribution_plot(
    contributions_df: pl.DataFrame,
    dict_name: str,
    dict_info: Dict
) -> None:
    """Create a bar plot showing category contributions."""
    
    if contributions_df.is_empty():
        logger.warning(f"No data to plot for {dict_name}")
        return
    
    # Sort by contribution percentage
    contributions_sorted = contributions_df.sort("contribution_percent", descending=True)
    
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    categories = contributions_sorted["category"].to_list()
    contributions = contributions_sorted["contribution_percent"].to_list()
    
    bars = plt.bar(categories, contributions, alpha=0.8, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars with peak week info
    for i, (bar, value) in enumerate(zip(bars, contributions)):
        # Main percentage label
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Peak week label (smaller, below the percentage)
        if i < len(contributions_sorted):
            peak_week = contributions_sorted.row(i, named=True).get('peak_week', 'N/A')
            if peak_week != 'N/A':
                # Format date to show only year-month for readability
                try:
                    from datetime import datetime
                    peak_date = datetime.strptime(peak_week, '%Y-%m-%d')
                    peak_label = peak_date.strftime('%Y-%m')
                except:
                    peak_label = peak_week[:7]  # Fallback to first 7 chars
                
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'Peak: {peak_label}', ha='center', va='bottom', 
                        fontsize=8, style='italic', alpha=0.7)
    
    plt.title(f'GPR Category Contributions - {dict_name.replace("_", " ").title()}\n'
              f'(Total Categories: {len(dict_info["categories"])}, Total Terms: {dict_info["total_terms"]})')
    plt.xlabel('Category')
    plt.ylabel('Contribution to GPR Index (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_filename = OUTPUT_DIR / f"category_contributions_{dict_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {plot_filename}")
    plt.close()

def create_peak_timeline_plot(
    contributions_df: pl.DataFrame,
    dict_name: str
) -> None:
    """Create a timeline plot showing when each category peaked."""
    
    if contributions_df.is_empty():
        logger.warning(f"No data to plot timeline for {dict_name}")
        return
    
    # Filter out categories without valid peak weeks
    valid_peaks = contributions_df.filter(
        (pl.col("peak_week") != "N/A") & 
        (pl.col("peak_week").is_not_null())
    )
    
    if valid_peaks.is_empty():
        logger.warning(f"No valid peak weeks found for {dict_name}")
        return
    
    # Convert peak weeks to datetime
    try:
        valid_peaks = valid_peaks.with_columns(
            pl.col("peak_week").str.to_date().alias("peak_date")
        ).sort("peak_date")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        categories = valid_peaks["category"].to_list()
        peak_dates = valid_peaks["peak_date"].to_pandas()
        peak_shares = valid_peaks["peak_share"].to_list()
        
        # Create scatter plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        for i, (category, date, share, color) in enumerate(zip(categories, peak_dates, peak_shares, colors)):
            ax.scatter(date, i, s=share*1000, c=[color], alpha=0.7, edgecolors='black')
            ax.text(date, i, f'{category}\n{share:.3f}', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_xlabel('Peak Week')
        ax.set_ylabel('Category')
        ax.set_title(f'Peak Weeks Timeline - {dict_name.replace("_", " ").title()}\n'
                    f'(Bubble size = peak share value)')
        
        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save timeline plot
        timeline_filename = OUTPUT_DIR / f"peak_timeline_{dict_name}.png"
        plt.savefig(timeline_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Peak timeline plot saved to {timeline_filename}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating timeline plot for {dict_name}: {e}")
        plt.close()

def create_comparison_plot(all_results: Dict[str, pl.DataFrame], all_dict_info: Dict) -> None:
    """Create a comparison plot showing contributions across dictionaries."""
    
    plt.figure(figsize=(14, 8))
    
    # Prepare data for comparison
    dict_names = list(all_results.keys())
    categories_union = set()
    
    for dict_name, df in all_results.items():
        if not df.is_empty():
            categories_union.update(df["category"].to_list())
    
    categories_union = sorted(list(categories_union))
    
    # Create grouped bar plot
    x = np.arange(len(categories_union))
    width = 0.35
    
    colors = ['skyblue', 'lightcoral']
    
    for i, (dict_name, df) in enumerate(all_results.items()):
        if df.is_empty():
            continue
            
        contributions = []
        for category in categories_union:
            category_data = df.filter(pl.col("category") == category)
            if category_data.is_empty():
                contributions.append(0)
            else:
                contributions.append(category_data["contribution_percent"].item())
        
        bars = plt.bar(x + i * width, contributions, width, 
                      label=dict_name.replace("_", " ").title(),
                      alpha=0.8, color=colors[i % len(colors)])
        
        # Add value labels
        for bar, value in zip(bars, contributions):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}%', ha='center', va='bottom', fontsize=8)
    
    plt.title('GPR Category Contributions Comparison')
    plt.xlabel('Category')
    plt.ylabel('Contribution to GPR Index (%)')
    plt.xticks(x + width/2, categories_union, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save comparison plot
    comparison_filename = OUTPUT_DIR / "category_contributions_comparison.png"
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {comparison_filename}")
    plt.close()

#%% [markdown]
# ## Main Analysis Function
# 
# Run the category contribution analysis for all specified dictionaries.

#%%
def main(dictionary_names: List[str] | None = None):
    """Main function to run category contribution analysis."""
    
    # Use all dictionaries if none specified
    if dictionary_names is None:
        dictionary_names = list(DICTIONARIES.keys())
    
    # Validate dictionary names
    invalid_names = [name for name in dictionary_names if name not in DICTIONARIES]
    if invalid_names:
        logger.error(f"Invalid dictionary names: {invalid_names}")
        logger.info(f"Available dictionaries: {list(DICTIONARIES.keys())}")
        return
    
    # Check if dictionary files exist
    missing_files = []
    for dict_name in dictionary_names:
        dict_path = DICTIONARIES[dict_name]
        if not check_dictionary_exists(dict_path):
            missing_files.append(f"{dict_name}: {dict_path}")
    
    if missing_files:
        logger.error("The following dictionary files are missing:")
        for missing_file in missing_files:
            logger.error(f"  - {missing_file}")
        logger.info("Please update the DICTIONARIES configuration with correct paths.")
        return
    
    # Connect to database
    logger.info("Connecting to CC-NEWS database...")
    try:
        db_file_path = CC_NEWS_DB_FILES[0]
        # Try read-only mode first to avoid lock conflicts
        conn_ccnews = ibis.connect(f"duckdb://{db_file_path}?access_mode=read_only")
        logger.info(f"Successfully connected to {db_file_path} in read-only mode")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
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
    
    # Store results for comparison
    all_results = {}
    all_dict_info = {}
    
    # Process each dictionary
    for dict_name in dictionary_names:
        dict_path = DICTIONARIES[dict_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dictionary: {dict_name}")
        logger.info(f"Path: {dict_path}")
        
        # Load dictionary info
        try:
            dict_info = load_dictionary_info(dict_path)
            all_dict_info[dict_name] = dict_info
            logger.info(f"Categories: {list(dict_info['categories'].keys())}")
            logger.info(f"Total terms: {dict_info['total_terms']}")
        except Exception as e:
            logger.error(f"Error loading dictionary info: {e}")
            continue
        
        # Initialize GPR calculator
        try:
            gpr_calculator = GPRCalculator(dictionary_path=dict_path)
            categories = list(gpr_calculator.category_definitions.keys())
            logger.info(f"GPR Calculator initialized with {len(categories)} categories")
        except Exception as e:
            logger.error(f"Error initializing GPR Calculator: {e}")
            continue
        
        # Calculate GPR results
        try:
            gpr_results = calculate_category_contributions(
                gpr_calculator, conn_ccnews, actual_ccnews_targets, cc_outlet_to_chinese_name
            )
            
            if gpr_results.is_empty():
                logger.warning(f"No GPR results for {dict_name}")
                continue
                
            logger.info(f"GPR calculation completed. Shape: {gpr_results.shape}")
            
        except Exception as e:
            logger.error(f"Error calculating GPR results: {e}")
            continue
        
        # Calculate overall contributions
        try:
            contributions_df = calculate_overall_contributions(gpr_results, categories)
            
            if contributions_df.is_empty():
                logger.warning(f"No contribution data for {dict_name}")
                continue
            
            all_results[dict_name] = contributions_df
            
            # Log results
            logger.info(f"\nCategory contributions for {dict_name}:")
            for row in contributions_df.sort("contribution_percent", descending=True).iter_rows(named=True):
                peak_week = row.get('peak_week', 'N/A')
                peak_share = row.get('peak_share', 0.0)
                logger.info(f"  {row['category']}: {row['contribution_percent']:.2f}% (Peak: {peak_week}, {peak_share:.4f})")
            
        except Exception as e:
            logger.error(f"Error calculating contributions: {e}")
            continue
        
        # Create individual plots
        try:
            create_contribution_plot(contributions_df, dict_name, dict_info)
            create_peak_timeline_plot(contributions_df, dict_name)
        except Exception as e:
            logger.error(f"Error creating plots for {dict_name}: {e}")
        
        # Save individual results
        try:
            csv_filename = OUTPUT_DIR / f"category_contributions_{dict_name}.csv"
            contributions_df.write_csv(csv_filename)
            logger.info(f"Results saved to {csv_filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    # Create comparison plot if we have multiple dictionaries
    if len(all_results) > 1:
        try:
            create_comparison_plot(all_results, all_dict_info)
        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}")
    
    # Save combined results
    try:
        combined_results = []
        for dict_name, df in all_results.items():
            if not df.is_empty():
                df_with_dict = df.with_columns(pl.lit(dict_name).alias("dictionary"))
                combined_results.append(df_with_dict)
        
        if combined_results:
            combined_df = pl.concat(combined_results, how="diagonal")
            combined_filename = OUTPUT_DIR / "category_contributions_combined.csv"
            combined_df.write_csv(combined_filename)
            logger.info(f"Combined results saved to {combined_filename}")
    
    except Exception as e:
        logger.error(f"Error saving combined results: {e}")
    
    # Cleanup
    conn_ccnews.disconnect()
    logger.info("Analysis completed!")

#%% [markdown]
# ## Command Line Interface
# 
# Allow the script to be run from command line with dictionary selection.

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate GPR category contributions for different dictionaries"
    )
    parser.add_argument(
        "--dictionaries",
        nargs="*",
        choices=list(DICTIONARIES.keys()),
        default=None,
        help="Dictionaries to analyze. If not specified, all dictionaries will be analyzed."
    )
    parser.add_argument(
        "--add-dictionary",
        nargs=2,
        metavar=("NAME", "PATH"),
        action="append",
        help="Add a dictionary with name and path. Can be used multiple times."
    )
    
    args = parser.parse_args()
    
    # Add custom dictionaries if provided
    if args.add_dictionary:
        for name, path in args.add_dictionary:
            DICTIONARIES[name] = path
            logger.info(f"Added dictionary: {name} -> {path}")
    
    main(args.dictionaries) 