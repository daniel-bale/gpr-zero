#!/usr/bin/env python3
"""
Create a simple ADIZ mention index by counting articles mentioning "防空識別區" 
(Air Defense Identification Zone) by week and newspaper, following Caldara's approach.
Plot the results together with actual ADIZ invasion data.
"""

import ibis
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import logging
from datetime import datetime
import seaborn as sns

# Import font setup for Chinese text
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seaborn theme first
sns.set_theme(style="whitegrid", palette="muted")

# Setup Chinese font support after theme
ensure_cjk_font_setup()

# Configuration
DB_PATH = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
ADIZ_INVASION_CSV = "data/external/ADIZ_invasion.csv"
OUTPUT_DIR = Path("data_analysis/output/adiz_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target newspapers mapping (like in capture_gpr_spikes.py)
NEWSPAPER_COMPARISON_MAP = {
    "中國時報": "China Times (中國時報)",
    "自由時報": "Liberty Times (自由時報)",
    "聯合報": "United Daily (聯合報)"
}
TARGET_CHINESE_NEWSPAPERS = list(NEWSPAPER_COMPARISON_MAP.keys())

# ADIZ keyword to search for
ADIZ_KEYWORD = "防空識別區"

# Date range for analysis
START_DATE = "2020-09-01"
END_DATE = "2024-12-31"


def load_adiz_invasion_data(csv_path: str) -> pd.DataFrame:
    """Load and process ADIZ invasion data from CSV."""
    logger.info(f"Loading ADIZ invasion data from {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Filter to 2020-2024 date range
    df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]
    
    # Create week start date (Monday) for aggregation
    df['week_start'] = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='D')
    
    # Aggregate by week (sum aircraft tracked per week)
    weekly_adiz = df.groupby('week_start').agg({
        'Total Aircraft Tracked in de facto ADIZ': 'sum'
    }).reset_index()
    
    weekly_adiz.columns = ['week_start', 'total_aircraft_tracked']
    
    logger.info(f"Loaded ADIZ data: {len(weekly_adiz)} weeks from {weekly_adiz['week_start'].min()} to {weekly_adiz['week_start'].max()}")
    
    return weekly_adiz

def calculate_adiz_share_by_week_newspaper(conn: ibis.BaseBackend) -> pl.DataFrame:
    """Calculate ADIZ article share by week and newspaper."""
    logger.info(f"Calculating ADIZ article share by week and newspaper for {START_DATE} to {END_DATE}")
    
    # Get articles table
    articles = conn.table("articles")
    
    # First, get all outlets and map them like in capture_gpr_spikes.py
    all_outlets_in_view = articles.select('outlet').distinct().to_polars()['outlet'].to_list()
    logger.info(f"Found {len(all_outlets_in_view)} distinct outlets in the database.")
    
    # Map outlets to Chinese names
    outlet_to_chinese_name = {}
    actual_targets = []
    
    for chinese_name_target, pattern in NEWSPAPER_COMPARISON_MAP.items():
        matched_outlet = next((o for o in all_outlets_in_view if pattern in o), None)
        if matched_outlet:
            logger.info(f"Mapping outlet '{matched_outlet}' to '{chinese_name_target}' for analysis.")
            actual_targets.append(matched_outlet)
            outlet_to_chinese_name[matched_outlet] = chinese_name_target
        else:
            logger.warning(f"No matching outlet found for pattern '{pattern}' (Chinese Name: {chinese_name_target}).")
    
    if not actual_targets:
        logger.error("No target newspapers found in the database after mapping.")
        return pl.DataFrame()
    
    logger.info(f"Will process ADIZ analysis for these actual outlets: {actual_targets}")
    
    # Filter for date range and target newspapers
    filtered_articles = articles.filter(
        (articles.outlet.isin(actual_targets)) &
        (articles.warc_date >= START_DATE) &
        (articles.warc_date <= END_DATE)
    )
    
    # Add week start date and map outlets to Chinese names
    case_tuples = []
    for outlet_pattern, zh_name in outlet_to_chinese_name.items():
        case_tuples.append((filtered_articles.outlet == outlet_pattern, zh_name))
    
    if not case_tuples:
        logger.error("Case tuples for newspaper mapping are empty.")
        return pl.DataFrame()
    
    articles_with_week = filtered_articles.mutate(
        week_start=filtered_articles.warc_date.cast('date').truncate('W'),
        has_adiz=filtered_articles.content.contains(ADIZ_KEYWORD),
        newspaper_zh=ibis.cases(*case_tuples, else_=None)
    )
    
    # Filter for non-null newspaper mappings
    articles_with_week = articles_with_week.filter(articles_with_week.newspaper_zh.notnull())
    
    # Calculate total articles and ADIZ articles by week and newspaper
    weekly_stats = articles_with_week.group_by(['week_start', 'newspaper_zh']).agg(
        total_articles=ibis._.count(),
        adiz_articles=ibis._.has_adiz.sum()
    )
    
    # Calculate ADIZ share
    weekly_shares = weekly_stats.mutate(
        adiz_share=weekly_stats.adiz_articles.cast('float64') / weekly_stats.total_articles.cast('float64')
    ).order_by(['week_start', 'newspaper_zh'])
    
    # Convert to Polars DataFrame
    result_df = weekly_shares.to_polars()
    
    logger.info(f"Found {len(result_df)} week-newspaper combinations")
    if len(result_df) > 0:
        logger.info(f"Date range: {result_df['week_start'].min()} to {result_df['week_start'].max()}")
        total_adiz = result_df['adiz_articles'].sum()
        total_articles = result_df['total_articles'].sum()
        logger.info(f"Total ADIZ articles: {total_adiz} out of {total_articles} total articles")
        
        # Log details by newspaper
        for newspaper in TARGET_CHINESE_NEWSPAPERS:
            newspaper_data = result_df.filter(pl.col('newspaper_zh') == newspaper)
            if not newspaper_data.is_empty():
                total_adiz_news = newspaper_data['adiz_articles'].sum()
                total_articles_news = newspaper_data['total_articles'].sum()
                logger.info(f"{newspaper}: {total_adiz_news} ADIZ articles out of {total_articles_news} total")
    
    return result_df

def create_combined_plot(adiz_shares_df: pl.DataFrame, adiz_invasion_df: pd.DataFrame):
    """Create a combined plot showing ADIZ article shares and actual ADIZ invasions."""
    logger.info("Creating combined ADIZ share and invasion plot")
    
    # Convert Polars to Pandas for plotting
    shares_pd = adiz_shares_df.to_pandas()
    shares_pd['week_start'] = pd.to_datetime(shares_pd['week_start'])
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax2 = ax1.twinx()
    
    # Colors for newspapers
    colors = {'中國時報': 'red', '自由時報': 'blue', '聯合報': 'green'}
    
    # Plot ADIZ shares for each newspaper (left y-axis)
    for newspaper in TARGET_CHINESE_NEWSPAPERS:
        newspaper_data = shares_pd[shares_pd['newspaper_zh'] == newspaper]
        if not newspaper_data.empty:
            ax1.plot(newspaper_data['week_start'], newspaper_data['adiz_share'],
                    color=colors[newspaper], marker='o', linewidth=2, markersize=4,
                    label=f'{newspaper}', alpha=0.8)
    
    # Plot actual ADIZ invasions (right y-axis)
    ax2.plot(adiz_invasion_df['week_start'], adiz_invasion_df['total_aircraft_tracked'],
            color='orange', linewidth=3, alpha=0.7, label='實際入侵架次')
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = OUTPUT_DIR / "adiz_shares_vs_invasions_weekly.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Combined plot saved to {plot_filename}")
    
    plt.show()

def create_newspaper_comparison_plot(adiz_shares_df: pl.DataFrame):
    """Create a plot comparing ADIZ article shares across newspapers."""
    logger.info("Creating newspaper comparison plot for ADIZ shares")
    
    # Convert to Pandas
    shares_pd = adiz_shares_df.to_pandas()
    shares_pd['week_start'] = pd.to_datetime(shares_pd['week_start'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Colors for newspapers
    colors = {'中國時報': 'red', '自由時報': 'blue', '聯合報': 'green'}
    
    # Plot each newspaper
    for newspaper in TARGET_CHINESE_NEWSPAPERS:
        newspaper_data = shares_pd[shares_pd['newspaper_zh'] == newspaper]
        if not newspaper_data.empty:
            ax.plot(newspaper_data['week_start'], newspaper_data['adiz_share'],
                   color=colors[newspaper], marker='o', linewidth=2, markersize=4,
                   label=newspaper, alpha=0.8)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = OUTPUT_DIR / "adiz_shares_by_newspaper_weekly.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Newspaper comparison plot saved to {plot_filename}")
    
    plt.show()

def save_data_to_csv(adiz_shares_df: pl.DataFrame, adiz_invasion_df: pd.DataFrame):
    """Save the processed data to CSV files."""
    logger.info("Saving data to CSV files")
    
    # Save ADIZ shares data
    shares_csv = OUTPUT_DIR / "adiz_shares_weekly_by_newspaper.csv"
    adiz_shares_df.write_csv(shares_csv)
    logger.info(f"ADIZ shares data saved to {shares_csv}")
    
    # Save ADIZ invasion data
    invasion_csv = OUTPUT_DIR / "adiz_invasions_weekly.csv"
    adiz_invasion_df.to_csv(invasion_csv, index=False)
    logger.info(f"ADIZ invasion data saved to {invasion_csv}")

def main():
    """Main function to run the ADIZ analysis."""
    logger.info(f"Starting ADIZ share analysis for {START_DATE} to {END_DATE}")
    
    # Connect to database
    logger.info(f"Connecting to database: {DB_PATH}")
    conn = ibis.connect(f"duckdb://{DB_PATH}", read_only=True)
    
    try:
        # Calculate ADIZ shares by week and newspaper
        adiz_shares_df = calculate_adiz_share_by_week_newspaper(conn)
        
        # Load ADIZ invasion data
        adiz_invasion_df = load_adiz_invasion_data(ADIZ_INVASION_CSV)
        
        # Save data to CSV
        save_data_to_csv(adiz_shares_df, adiz_invasion_df)
        
        # Create plots
        create_newspaper_comparison_plot(adiz_shares_df)
        create_combined_plot(adiz_shares_df, adiz_invasion_df)
        
        # Print summary statistics
        logger.info("\n=== SUMMARY STATISTICS ===")
        if len(adiz_shares_df) > 0:
            logger.info(f"Total weeks analyzed: {adiz_shares_df['week_start'].n_unique()}")
            logger.info(f"Total ADIZ articles: {adiz_shares_df['adiz_articles'].sum()}")
            logger.info(f"Total articles: {adiz_shares_df['total_articles'].sum()}")
            overall_share = adiz_shares_df['adiz_articles'].sum() / adiz_shares_df['total_articles'].sum()
            logger.info(f"Overall ADIZ share: {overall_share:.4f} ({overall_share*100:.2f}%)")
            
            for newspaper in TARGET_CHINESE_NEWSPAPERS:
                newspaper_data = adiz_shares_df.filter(pl.col('newspaper_zh') == newspaper)
                if not newspaper_data.is_empty():
                    total_adiz = newspaper_data['adiz_articles'].sum()
                    total_articles = newspaper_data['total_articles'].sum()
                    avg_share = total_adiz / total_articles if total_articles > 0 else 0
                    weeks_analyzed = len(newspaper_data)
                    logger.info(f"{newspaper}: {total_adiz}/{total_articles} articles ({avg_share:.4f} share), {weeks_analyzed} weeks")
        else:
            logger.warning("No data found for the specified date range and newspapers")
        
        logger.info(f"ADIZ invasion data: {len(adiz_invasion_df)} weeks")
        logger.info(f"Total aircraft tracked: {adiz_invasion_df['total_aircraft_tracked'].sum()}")
        logger.info(f"Average aircraft per week: {adiz_invasion_df['total_aircraft_tracked'].mean():.2f}")
        
    finally:
        conn.disconnect()
        logger.info("Database connection closed")
    
    logger.info("ADIZ analysis completed successfully")

if __name__ == "__main__":
    main()