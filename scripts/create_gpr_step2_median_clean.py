#!/usr/bin/env python3
"""
Create a clean GPR Step2 median plot without labels and title for thesis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from src.gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup CJK font support
ensure_cjk_font_setup()

def main():
    """Create clean GPR Step2 median plot."""
    
    # Load the scaled Step2 median data
    data_file = Path("data/gpr_correlations/median_timeseries/median_gpr_step2_scaled.csv")
    if not data_file.exists():
        logger.error(f"File not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['warc_date_bucketed'])
    
    # Filter to ensure we end at 2024-12-31
    df = df[df['date'] <= '2024-12-31']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the median GPR with confidence bands
    # 95% confidence band
    ax.fill_between(df['date'], df['p025'], df['p975'], 
                    alpha=0.2, color='steelblue', label='95% CI')
    
    # 68% confidence band
    ax.fill_between(df['date'], df['p16'], df['p84'], 
                    alpha=0.3, color='steelblue', label='68% CI')
    
    # Median line
    ax.plot(df['date'], df['median'], color='darkblue', 
            linewidth=2, label='Median GPR')
    
    # Horizontal line at 100
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Remove title and labels
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, frameon=False)
    
    # Set y-axis limits to ensure good visualization
    ax.set_ylim(0, max(df['p975']) * 1.1)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    output_file = Path("data/gpr_correlations/gpr_step2_median_scaled_clean.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    plt.close()
    
    # Copy to thesis figures
    import shutil
    thesis_file = Path("/home5/chuenho/thesis/figures/gpr_step2_median_scaled_clean.png")
    shutil.copy(output_file, thesis_file)
    logger.info(f"Copied to thesis figures: {thesis_file}")

if __name__ == "__main__":
    main()