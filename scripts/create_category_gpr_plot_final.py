#!/usr/bin/env python3
"""
Create category GPR plot without scaling to mean=100, using raw proportions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import seaborn as sns
from src.gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style first
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Setup CJK font support AFTER style to avoid override
ensure_cjk_font_setup()

def main():
    """Create category GPR plot with actual GPR values (not scaled to mean=100)."""
    
    # Load category data - use unscaled version
    csv_file = Path("data/category_gpr_indices/category_median_gpr_unscaled.csv")
    if not csv_file.exists():
        logger.error(f"Category data not found: {csv_file}")
        return
    
    # Load data
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['week'])
    
    # Filter to 2017-2024
    df = df[(df['date'] >= '2017-01-01') & (df['date'] <= '2024-12-31')]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Category names and colors (consistent with other plots)
    categories = ['Military Escalation', 'Diplomatic Isolation', 'Economic Coercion', 
                  'Regional Security', 'Internal Political Instability', 'Gray Zone Operations',
                  'US-China-Taiwan Relations', 'Cyber and Information Warfare']
    
    # Use consistent color scheme
    category_colors = {
        'Military Escalation': '#FF6B6B',
        'Diplomatic Isolation': '#4ECDC4',
        'Economic Coercion': '#45B7D1',
        'Regional Security': '#F7B801',
        'Internal Political Instability': '#95E1D3',
        'Gray Zone Operations': '#DDA0DD',
        'US-China-Taiwan Relations': '#98D8C8',
        'Cyber and Information Warfare': '#FFB6C1'
    }
    
    # Plot each category
    for cat in categories:
        if cat in df.columns:
            ax.plot(df['date'], df[cat], label=cat, color=category_colors[cat], 
                   linewidth=1.5, alpha=0.8)
    
    # Remove title and labels (clean thesis figure)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    
    # Add legend inside the plot area with flat appearance
    ax.legend(loc='upper right', fontsize=9, frameon=True, 
              fancybox=False, shadow=False, ncol=1, framealpha=0.9)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    output_file = Path("data/category_gpr_indices/category_gpr_indices_clean_unscaled.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    plt.close()
    
    # Copy to thesis figures
    import shutil
    thesis_file = Path("/home5/chuenho/thesis/figures/category_gpr_indices_clean_unscaled.png")
    shutil.copy(output_file, thesis_file)
    logger.info(f"Copied to thesis figures: {thesis_file}")

if __name__ == "__main__":
    main()