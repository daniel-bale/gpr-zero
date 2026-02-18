#!/usr/bin/env python3
"""
Create clean article count comparison plot for thesis using existing GPR data.
Uses 3-outlet data from Step2 dictionaries.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import logging
from src.gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup CJK font support
ensure_cjk_font_setup()

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
NEWSPAPERS = {
    'China Times': 'china',
    'Liberty Times': 'liberty', 
    'United Daily': 'united'
}

DARK_BLUE = "#1f77b4"
FIGURE_SIZE = (12, 6)

def main():
    """Create article count comparison plot."""
    
    # Load newspaper-specific Step2 data that already has article counts
    data_dir = Path("/home5/chuenho/gpr-zero/data/newspaper_gpr_step2")
    output_dir = Path("/home5/chuenho/gpr-zero/data/gpr_indices_3outlets")
    output_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir = Path("/home5/chuenho/thesis/figures")
    
    # Load data for each newspaper from parquet files which have article counts
    newspaper_data = {}
    for name, code in NEWSPAPERS.items():
        file_path = data_dir / f"{code}_step2_all_dictionaries.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            # Get median across dictionaries for each week
            weekly_data = df.groupby('warc_date_bucketed').agg({
                'total_articles': 'median'  # median across 40 dictionaries
            })
            weekly_data.index = pd.to_datetime(weekly_data.index)
            newspaper_data[name] = weekly_data
            logger.info(f"Loaded data for {name}: {len(weekly_data)} weeks")
    
    # Create article count plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    colors = {
        'China Times': '#d62728',   # Red
        'Liberty Times': '#1f77b4', # Blue
        'United Daily': '#ff7f0e'   # Orange
    }
    
    for name, df in newspaper_data.items():
        # Filter to 2017-2024
        mask = (df.index >= '2017-01-01') & (df.index < '2025-01-01')
        df_filtered = df[mask]
        
        if 'total_articles' in df_filtered.columns:
            # Calculate daily average (divide weekly count by 7)
            daily_avg = df_filtered['total_articles'] / 7.0
            
            # Create complete date range for missing data handling
            date_range = pd.date_range(start='2017-01-01', end='2024-12-31', freq='W-MON')
            complete_df = pd.DataFrame(index=date_range)
            complete_df['daily_avg'] = daily_avg
            
            # Plot with gaps for missing data
            valid_mask = ~complete_df['daily_avg'].isna()
            
            # Find continuous segments
            segments = []
            current_segment = []
            
            for i, (date, is_valid) in enumerate(zip(complete_df.index, valid_mask)):
                if is_valid:
                    current_segment.append(i)
                else:
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = []
            
            if current_segment:
                segments.append(current_segment)
            
            # Plot each segment
            for segment in segments:
                segment_dates = complete_df.iloc[segment].index
                segment_values = complete_df.iloc[segment]['daily_avg']
                ax.plot(segment_dates, segment_values, 
                       color=colors[name], linewidth=2, alpha=0.9,
                       label=name if segment == segments[0] else "")
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    # Add legend
    ax.legend(fontsize=10, frameon=False, loc='upper right')
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / "article_count_daily_avg_comparison_clean.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved article count plot to {output_file}")
    plt.close()
    
    # Copy to thesis
    import shutil
    thesis_file = thesis_dir / "article_count_daily_avg_comparison_clean.png"
    shutil.copy(output_file, thesis_file)
    logger.info(f"Copied to thesis figures: {thesis_file}")
    
    # Print summary statistics
    logger.info("\n=== Article Count Statistics ===")
    for name, df in newspaper_data.items():
        mask = (df.index >= '2017-01-01') & (df.index < '2025-01-01')
        df_filtered = df[mask]
        if 'total_articles' in df_filtered.columns:
            daily_avg = df_filtered['total_articles'] / 7.0
            logger.info(f"{name}:")
            logger.info(f"  Mean daily articles: {daily_avg.mean():.1f}")
            logger.info(f"  Min daily articles: {daily_avg.min():.1f}")
            logger.info(f"  Max daily articles: {daily_avg.max():.1f}")

if __name__ == "__main__":
    main()