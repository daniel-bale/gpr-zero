#!/usr/bin/env python3
"""
Create a clean Lau vs Step2 comparison plot without titles and labels for thesis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup CJK font support
ensure_cjk_font_setup()

def main():
    """Create clean comparison plot."""
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "gpr_correlations" / "lau_comparison"
    
    # Load data
    lau_scaled = pd.read_csv(data_dir / "lau_gpr_scaled.csv")
    lau_scaled['date'] = pd.to_datetime(lau_scaled['date'])
    
    step2_median = pd.read_csv(project_root / "data" / "gpr_correlations" / "median_timeseries" / "median_gpr_step2_scaled.csv")
    step2_median['date'] = pd.to_datetime(step2_median['warc_date_bucketed'])
    
    # Merge dataframes
    merged_df = pd.merge(lau_scaled, step2_median, on='date', how='inner')
    
    # Create clean plot without titles/labels for thesis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot lines
    ax.plot(merged_df['date'], merged_df['median'], color='steelblue', 
            linewidth=2, label='Our GPR')
    ax.plot(merged_df['date'], merged_df['gpr_scaled'], color='darkred', 
            linewidth=2, label="Lau's GPR", linestyle='--')
    
    # Remove all labels and titles
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    # Keep legend but simplify
    ax.legend(loc='upper left', fontsize=11, frameon=False)
    
    # Keep grid for readability
    ax.grid(True, alpha=0.3)
    
    # Remove tick labels if needed (optional - comment out if you want to keep them)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_file = data_dir / "lau_vs_step2_comparison_thesis_clean.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved clean thesis plot to {output_file}")
    plt.close()
    
    # Copy to thesis figures directory
    import shutil
    thesis_fig_dir = Path("/home5/chuenho/thesis/figures")
    thesis_file = thesis_fig_dir / "lau_vs_step2_comparison_clean.png"
    shutil.copy(output_file, thesis_file)
    logger.info(f"Copied to thesis figures: {thesis_file}")

if __name__ == "__main__":
    main()