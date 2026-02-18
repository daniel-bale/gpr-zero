#!/usr/bin/env python3
"""
Create a stacked bar plot showing category composition for unique spikes only.
Uses event labels on x-axis instead of spike numbers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Setup CJK font support
ensure_cjk_font_setup()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CATEGORY_DATA_DIR = Path("data/category_gpr_indices")
SPIKE_FILE = Path("data/gpr_spikes/unified_detection_3outlets/step2_median_spikes.csv")
UNIQUE_SPIKES_FILE = Path("data/gpr_spikes/step2_unique_spikes_summary_final.csv")
OUTPUT_DIR = Path("data/category_gpr_indices")
THESIS_DIR = Path("/home5/chuenho/thesis/figures")

# Categories in our standard order
CATEGORIES = [
    "Military Escalation",
    "Economic Coercion", 
    "Diplomatic Isolation",
    "Cyber and Information Warfare",
    "Gray Zone Operations",
    "Internal Political Instability",
    "Regional Security",
    "US-China-Taiwan Relations"
]

# Category colors (consistent with other plots - same as all spikes plot)
CATEGORY_COLORS = {
    'Military Escalation': '#FF6B6B',
    'Diplomatic Isolation': '#4ECDC4',
    'Economic Coercion': '#45B7D1',
    'Regional Security': '#F7B801',
    'Internal Political Instability': '#95E1D3',
    'Gray Zone Operations': '#DDA0DD',
    'US-China-Taiwan Relations': '#98D8C8',
    'Cyber and Information Warfare': '#FFB6C1'
}

def load_unique_spikes():
    """Load unique spike information."""
    df = pd.read_csv(UNIQUE_SPIKES_FILE)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date']) 
    df['peak_date'] = pd.to_datetime(df['peak_date'])
    return df

def load_category_data(spike_start, spike_end):
    """Load category-specific GPR data for a spike period."""
    # Load the consolidated category data file
    category_file = CATEGORY_DATA_DIR / "category_median_gpr_unscaled.csv"
    
    if not category_file.exists():
        logger.error(f"Category data not found: {category_file}")
        return {cat: 0 for cat in CATEGORIES}
        
    df = pd.read_csv(category_file)
    df['week'] = pd.to_datetime(df['week'])
    
    # Filter for spike period (extend end date by 6 days to include full week)
    spike_end_extended = spike_end + pd.Timedelta(days=6)
    spike_data = df[(df['week'] >= spike_start) & (df['week'] <= spike_end_extended)]
    
    if len(spike_data) == 0:
        logger.warning(f"No data found for spike period {spike_start} to {spike_end}")
        return {cat: 0 for cat in CATEGORIES}
    
    # Calculate median for each category
    category_values = {}
    for category in CATEGORIES:
        if category in spike_data.columns:
            category_values[category] = spike_data[category].median()
        else:
            category_values[category] = 0
            
    return category_values

def load_overall_gpr(spike_start, spike_end):
    """Load overall GPR for spike period."""
    file_path = Path("data/gpr_indices_3outlets/step2_median_gpr_3outlets.csv")
    df = pd.read_csv(file_path)
    df['week'] = pd.to_datetime(df['week'])
    
    # Extend end date by 6 days to include full week
    spike_end_extended = spike_end + pd.Timedelta(days=6)
    spike_data = df[(df['week'] >= spike_start) & (df['week'] <= spike_end_extended)]
    
    if len(spike_data) > 0:
        # Use unscaled GPR index
        return spike_data['gpr_index'].median()
    else:
        return 0

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load unique spikes
    unique_spikes = load_unique_spikes()
    logger.info(f"Processing {len(unique_spikes)} unique spikes")
    
    # Prepare data for plotting
    spike_data = []
    
    for idx, spike in unique_spikes.iterrows():
        spike_info = {
            'spike_number': spike['spike_number'],
            'peak_date': spike['peak_date'],
            'start_date': spike['start_date'], 
            'end_date': spike['end_date'],
            'event_label': spike['event_summary']  # Use event_summary from CSV
        }
        
        # Get category values
        category_values = load_category_data(spike['start_date'], spike['end_date'])
        overall_median = load_overall_gpr(spike['start_date'], spike['end_date'])
        
        # Calculate ratios (category median / overall median)
        for cat in CATEGORIES:
            cat_median = category_values.get(cat, 0)
            spike_info[cat] = cat_median / overall_median if overall_median > 0 else 0
            
        spike_data.append(spike_info)
        
    # Create DataFrame
    spike_df = pd.DataFrame(spike_data)
    
    # Sort by date
    spike_df = spike_df.sort_values('peak_date')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data for stacked bar chart
    x = np.arange(len(spike_df))
    width = 0.7
    
    # Create stacked bars - showing ratios (which can exceed 1.0)
    bottom = np.zeros(len(spike_df))
    
    for category in CATEGORIES:
        values = spike_df[category].values
        ax.bar(x, values, width, label=category, bottom=bottom, 
               color=CATEGORY_COLORS[category], alpha=0.8)
        bottom += values
    
    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(spike_df['event_label'], rotation=45, ha='right', fontsize=9)
    
    # Remove axis labels as requested for thesis
    ax.set_xlabel('')
    ax.set_ylabel('') 
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Legend outside plot area on the right
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
             fontsize=9, framealpha=0.9, title='Categories')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tighten layout
    plt.tight_layout()
    
    # Save the plot
    output_file = OUTPUT_DIR / "unique_spikes_category_stacked_bar.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    
    # Copy to thesis directory
    if THESIS_DIR.exists():
        thesis_file = THESIS_DIR / "unique_spikes_category_stacked_bar.png"
        plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to thesis directory: {thesis_file}")
    
    plt.close()
    
    # Print summary statistics
    print("\nCategory ratios across unique spikes:")
    print("=" * 50)
    
    for category in CATEGORIES:
        values = spike_df[category].values
        avg_ratio = np.mean(values)
        print(f"{category:30s}: {avg_ratio:.2f}")
    
    # Calculate total
    total_ratios = []
    for idx in range(len(spike_df)):
        spike_total = sum(spike_df.iloc[idx][cat] for cat in CATEGORIES)
        total_ratios.append(spike_total)
    avg_total = np.mean(total_ratios)
    
    print(f"\n{'Average total ratio':30s}: {avg_total:.2f}")
    
    print("\nNote: Each bar shows the category composition of a spike.")
    print("Values are ratios of category median to overall GPR median.")
    print("Ratios sum to less than 1.0 because the overall GPR includes")
    print("additional components beyond these 8 categories.")

if __name__ == "__main__":
    main()