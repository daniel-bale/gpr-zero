#!/usr/bin/env python3
"""
Create category stacked bar plot showing category ratios for each spike.
This shows the ratio of (category median / overall GPR median) for each individual spike using corrected 3-outlet data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging
from src.gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style first
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Setup CJK font support AFTER style to avoid override
ensure_cjk_font_setup()

# Standard 8 categories
CATEGORIES = [
    "Military Escalation",
    "Diplomatic Isolation", 
    "Economic Coercion",
    "Regional Security",
    "Internal Political Instability",
    "Gray Zone Operations",
    "US-China-Taiwan Relations",
    "Cyber and Information Warfare"
]

# Category colors (consistent with other plots)
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

def main():
    """Create category stacked bar plot showing ratios for each spike."""
    
    # Load the unscaled category data from Task 6
    category_file = Path("data/category_gpr_indices/category_median_gpr_unscaled.csv")
    if not category_file.exists():
        logger.error(f"Category data not found: {category_file}")
        return
    
    # Load spike data with corrected 3-outlet filter
    spike_file = Path("data/gpr_spikes/unified_detection_3outlets/step2_median_spikes.csv")
    if not spike_file.exists():
        logger.error(f"Spike data not found: {spike_file}")
        return
    
    # Load data
    category_df = pd.read_csv(category_file)
    category_df['week'] = pd.to_datetime(category_df['week'])
    
    spike_df = pd.read_csv(spike_file)
    spike_df['start_date'] = pd.to_datetime(spike_df['start_date'])
    spike_df['end_date'] = pd.to_datetime(spike_df['end_date'])
    spike_df['peak_date'] = pd.to_datetime(spike_df['peak_date'])
    
    logger.info(f"Loaded {len(spike_df)} spikes from corrected 3-outlet data")
    
    # Load overall GPR index data
    overall_file = Path("data/gpr_indices_3outlets/step2_median_gpr_3outlets.csv")
    overall_df = pd.read_csv(overall_file)
    overall_df['week'] = pd.to_datetime(overall_df['week'])
    
    # For each spike, calculate category ratios
    spike_data = []
    
    for idx, spike in spike_df.iterrows():
        # Get data for this spike period (including full weeks)
        spike_end_extended = spike['end_date'] + pd.Timedelta(days=6)
        spike_period = (category_df['week'] >= spike['start_date']) & (category_df['week'] <= spike_end_extended)
        spike_weeks = category_df[spike_period].copy()
        
        if len(spike_weeks) == 0:
            continue
        
        # Calculate median for each category during this spike
        spike_info = {
            'spike_id': f"S{idx+1:02d}",
            'peak_date': spike['peak_date'],
            'start_date': spike['start_date'],
            'end_date': spike['end_date']
        }
        
        # Get overall GPR for this spike period
        overall_spike = overall_df[(overall_df['week'] >= spike['start_date']) & 
                                   (overall_df['week'] <= spike_end_extended)]
        
        if len(overall_spike) == 0:
            continue
            
        # Calculate median overall GPR for this spike
        overall_median = overall_spike['gpr_index'].median()
        
        # Calculate ratios for each category
        for cat in CATEGORIES:
            if cat in spike_weeks.columns:
                cat_median = spike_weeks[cat].median()
                # Ratio = category median / overall GPR median
                spike_info[cat] = cat_median / overall_median if overall_median > 0 else 0
        
        spike_data.append(spike_info)
    
    # Convert to DataFrame for easier manipulation
    spike_props_df = pd.DataFrame(spike_data)
    spike_props_df = spike_props_df.sort_values('peak_date')
    
    logger.info(f"Analyzing {len(spike_props_df)} spikes with category breakdowns")
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data for stacking
    x_positions = np.arange(len(spike_props_df))
    bottom = np.zeros(len(spike_props_df))
    
    # Plot each category as a stacked layer
    for cat in CATEGORIES:
        if cat in spike_props_df.columns:
            values = spike_props_df[cat].values
            ax.bar(x_positions, values, bottom=bottom, 
                  label=cat, color=CATEGORY_COLORS[cat], alpha=0.8)
            bottom += values
    
    # Customize plot - no axis labels for thesis
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    # Don't limit y-axis since ratios can be > 1
    
    # Set x-axis labels (show every 3rd spike to avoid crowding)
    step = max(1, len(spike_props_df) // 7)
    tick_positions = range(0, len(spike_props_df), step)
    tick_labels = []
    
    for i in tick_positions:
        spike = spike_props_df.iloc[i]
        # Format: S01\n2018-04
        tick_labels.append(f"{spike['spike_id']}\n{spike['peak_date'].strftime('%Y-%m')}")
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
             fontsize=9, framealpha=0.9, title='Categories')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("data/category_gpr_indices")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "category_median_ratios_spike_periods_bar.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved spike period bar plot to: {output_file}")
    plt.close()
    
    # Copy to thesis
    import shutil
    thesis_file = Path("/home5/chuenho/thesis/figures/category_median_ratios_spike_periods_bar.png")
    shutil.copy(output_file, thesis_file)
    logger.info(f"Copied spike period bar plot to thesis: {thesis_file}")
    
    # Log summary statistics
    logger.info("\nCategory ratio summary across all spikes:")
    for cat in CATEGORIES:
        if cat in spike_props_df.columns:
            mean_ratio = spike_props_df[cat].mean()
            logger.info(f"  {cat}: {mean_ratio:.2f}")
    
    logger.info("\n" + "="*60)
    logger.info("THESIS CAPTION NOTE:")
    logger.info("Each bar represents the composition of a GPR spike, with height showing")
    logger.info("the ratio of category median to overall GPR median during that spike period.")
    logger.info("A ratio of 0.4 means the category's median was 40% of the overall GPR median.")
    logger.info("Ratios sum to less than 1.0 because the overall GPR includes additional components.")
    logger.info("="*60)

if __name__ == "__main__":
    main()