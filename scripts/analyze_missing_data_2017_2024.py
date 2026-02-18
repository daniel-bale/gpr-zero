#!/usr/bin/env python3
"""
Analyze missing data patterns for each newspaper specifically for 2017-2024 period.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup matplotlib for Chinese text
from src.gpr_zero.utils.font_setup import ensure_cjk_font_setup
ensure_cjk_font_setup()

def analyze_missing_data_2017_2024():
    """Analyze missing data patterns for 2017-2024 period only."""
    
    # Load the existing Step2 median GPR data for each newspaper
    data_dir = Path("/home5/chuenho/gpr-zero/data/newspaper_gpr_step2")
    
    newspapers = {
        'China Times': 'china_step2_median_gpr.csv',
        'Liberty Times': 'liberty_step2_median_gpr.csv',
        'United Daily': 'united_step2_median_gpr.csv'
    }
    
    # Date range for analysis (2017-2024)
    start_date = pd.Timestamp('2017-01-01')
    end_date = pd.Timestamp('2024-12-31')
    
    # Create complete weekly date range
    complete_weeks = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    total_weeks = len(complete_weeks)
    
    logger.info(f"\nAnalyzing missing data for 2017-2024 period")
    logger.info(f"Total weeks in period: {total_weeks}")
    logger.info("="*60)
    
    missing_summary = []
    
    for newspaper_name, filename in newspapers.items():
        filepath = data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue
            
        # Load data
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index)
        
        # Filter to 2017-2024 period
        df_period = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Find missing weeks
        actual_weeks = set(df_period.index)
        expected_weeks = set(complete_weeks)
        missing_weeks = expected_weeks - actual_weeks
        missing_weeks_sorted = sorted(list(missing_weeks))
        
        # Calculate statistics
        missing_count = len(missing_weeks)
        missing_pct = (missing_count / total_weeks) * 100
        
        logger.info(f"\n{newspaper_name}:")
        logger.info(f"  Total weeks: {total_weeks}")
        logger.info(f"  Present weeks: {len(actual_weeks)}")
        logger.info(f"  Missing weeks: {missing_count} ({missing_pct:.1f}%)")
        
        # Find clusters of missing dates
        if missing_weeks_sorted:
            clusters = []
            current_cluster = [missing_weeks_sorted[0]]
            
            for i in range(1, len(missing_weeks_sorted)):
                if (missing_weeks_sorted[i] - missing_weeks_sorted[i-1]).days <= 7:
                    current_cluster.append(missing_weeks_sorted[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [missing_weeks_sorted[i]]
            clusters.append(current_cluster)
            
            # Report major missing periods (4+ weeks)
            major_gaps = [c for c in clusters if len(c) >= 4]
            if major_gaps:
                logger.info(f"  Major missing periods (4+ weeks):")
                for cluster in sorted(major_gaps, key=len, reverse=True)[:5]:  # Top 5
                    start = min(cluster).strftime('%Y-%m-%d')
                    end = max(cluster).strftime('%Y-%m-%d')
                    logger.info(f"    {start} to {end} ({len(cluster)} weeks)")
            
            # Analyze by year
            logger.info(f"  Missing weeks by year:")
            for year in range(2017, 2025):
                year_missing = [w for w in missing_weeks_sorted if w.year == year]
                if year_missing:
                    logger.info(f"    {year}: {len(year_missing)} weeks")
        
        # Add to summary
        missing_summary.append({
            'newspaper': newspaper_name,
            'total_weeks': total_weeks,
            'present_weeks': len(actual_weeks),
            'missing_weeks': missing_count,
            'missing_pct': missing_pct,
            'longest_gap_weeks': len(max(clusters, key=len)) if missing_weeks_sorted else 0
        })
    
    # Create visualization
    create_missing_data_visualization(missing_summary, complete_weeks, newspapers, data_dir)
    
    # Save summary
    summary_df = pd.DataFrame(missing_summary)
    summary_file = data_dir / "missing_data_summary_2017_2024.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"\nSaved summary to: {summary_file}")
    
    return missing_summary

def create_missing_data_visualization(missing_summary, complete_weeks, newspapers, data_dir):
    """Create visualization of missing data patterns."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Top plot: Bar chart of missing percentages
    summary_df = pd.DataFrame(missing_summary)
    colors = ['#d62728', '#1f77b4', '#ff7f0e']  # Red, Blue, Orange
    
    bars = ax1.bar(summary_df['newspaper'], summary_df['missing_pct'], color=colors)
    ax1.set_ylabel('Missing Data (%)', fontsize=12)
    ax1.set_title('Missing Data Percentage by Newspaper (2017-2024)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(summary_df['missing_pct']) * 1.2)
    
    # Add value labels on bars
    for bar, pct in zip(bars, summary_df['missing_pct']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11)
    
    # Bottom plot: Timeline of data availability
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_xlim(complete_weeks[0], complete_weeks[-1])
    
    newspaper_names = ['China Times', 'Liberty Times', 'United Daily']
    y_positions = [2, 1, 0]
    
    for idx, (newspaper_name, filename) in enumerate(newspapers.items()):
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
            df.index = pd.to_datetime(df.index)
            df_period = df[(df.index >= complete_weeks[0]) & (df.index <= complete_weeks[-1])]
            
            # Plot available weeks as lines
            for week in df_period.index:
                ax2.plot([week, week], [y_positions[idx] - 0.4, y_positions[idx] + 0.4], 
                        color=colors[idx], linewidth=1, alpha=0.7)
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(newspaper_names)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Data Availability Timeline (2017-2024)', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = data_dir / "missing_data_analysis_2017_2024.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to: {output_file}")

if __name__ == "__main__":
    analyze_missing_data_2017_2024()