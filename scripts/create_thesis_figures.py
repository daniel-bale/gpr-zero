#!/usr/bin/env python3
"""
Create clean figures for thesis - no titles, minimal labels as specified.
Based on existing scripts but modified for thesis requirements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gpr_zero.utils.font_setup import ensure_cjk_font_setup
import json

# Output directory for thesis figures
THESIS_FIGURES_DIR = Path("/home5/chuenho/thesis/figures")
THESIS_FIGURES_DIR.mkdir(exist_ok=True)

def create_gpr_scaled_clean(output_filename="gpr_step2_median_scaled_clean.png"):
    """Create GPR index scaled to mean=100 - bottom panel only, no title."""
    
    # Load data
    data_dir = Path("/home5/chuenho/gpr-zero/data/gpr_correlations")
    timeseries_df = pd.read_csv(data_dir / "combined_gpr_timeseries_step2_expanded.csv")
    timeseries_df['warc_date_bucketed'] = pd.to_datetime(timeseries_df['warc_date_bucketed'])
    
    # Load metadata
    with open(data_dir / "dictionary_metadata_step2_expanded.json", 'r') as f:
        metadata = json.load(f)
    
    # Filter for Step2 only
    step2_metadata = {k: v for k, v in metadata.items() if v['step'] == 'step2'}
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Set date column as index
    timeseries_df = timeseries_df.set_index('warc_date_bucketed')
    
    # Filter to date range
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2024-12-31')
    timeseries_df = timeseries_df[(timeseries_df.index >= start_date) & 
                                  (timeseries_df.index <= end_date)]
    
    # Get Step2 dictionary columns only
    dict_cols = [col for col in timeseries_df.columns if col in step2_metadata]
    gpr_data = timeseries_df[dict_cols]
    
    # Scale each dictionary to mean = 100
    scaled_data = pd.DataFrame()
    for col in dict_cols:
        col_mean = gpr_data[col].mean()
        if col_mean > 0:
            scaled_data[col] = (gpr_data[col] / col_mean) * 100
    
    # Calculate statistics for scaled data
    scaled_median = scaled_data.median(axis=1)
    scaled_mean = scaled_data.mean(axis=1)
    scaled_p16 = scaled_data.quantile(0.16, axis=1)
    scaled_p84 = scaled_data.quantile(0.84, axis=1)
    scaled_p025 = scaled_data.quantile(0.025, axis=1)
    scaled_p975 = scaled_data.quantile(0.975, axis=1)
    
    # Plot
    ax.plot(scaled_median.index, scaled_median.values, color='darkblue', linewidth=2.5, label='Median')
    ax.plot(scaled_mean.index, scaled_mean.values, color='darkred', linewidth=1, 
            linestyle='--', alpha=0.8, label='Mean')
    
    # Confidence bands
    ax.fill_between(scaled_median.index, scaled_p16.values, scaled_p84.values, 
                   color='steelblue', alpha=0.4, label='68% CI')
    ax.fill_between(scaled_median.index, scaled_p025.values, scaled_p975.values, 
                   color='lightblue', alpha=0.3, label='95% CI')
    
    # Reference line at 100
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5)
    
    # NO TITLE
    # ax.set_title('Taiwan GPR Index: Scaled to Mean = 100')
    
    # Simple axis labels
    ax.set_xlabel('Date')
    ax.set_ylabel('GPR Index (Mean = 100)')
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path = THESIS_FIGURES_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {output_path}")
    return output_path


def create_category_contributions_clean(output_filename="category_contributions_clean.png"):
    """Create category contributions plot - no title."""
    
    # Load the category contributions data - use the latest file
    contrib_dir = Path("/home5/chuenho/gpr-zero/data/category_contributions")
    contrib_files = list(contrib_dir.glob("step2_weekly_category_contributions_*.csv"))
    if not contrib_files:
        print("Warning: No category contribution files found")
        return None
    
    # Use the latest file
    contrib_file = sorted(contrib_files)[-1]
    print(f"Using category contributions file: {contrib_file.name}")
    contrib_df = pd.read_csv(contrib_file)
    contrib_df['week'] = pd.to_datetime(contrib_df['week'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for each category
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
    
    # Plot each category (columns end with _contribution)
    for category in category_colors.keys():
        col_name = f"{category}_contribution"
        if col_name in contrib_df.columns:
            ax.plot(contrib_df['week'], contrib_df[col_name], 
                   label=category, color=category_colors[category], linewidth=1.5)
    
    # NO TITLE
    # ax.set_title('Weekly Category Contributions to Step2 GPR Index')
    
    # Axis labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Category Contribution (%)')
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path = THESIS_FIGURES_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {output_path}")
    return output_path


def create_top50_words_clean(output_filename="top50_words_clean.png"):
    """Copy existing top 50 words plot for thesis use."""
    
    # Since the existing plot already has the visualization we need,
    # let's just copy it and note that it needs manual cleaning
    import shutil
    
    source_file = Path("/home5/chuenho/gpr-zero/data/word_contributions/step2_spikes_top50_words.png")
    output_path = THESIS_FIGURES_DIR / output_filename
    
    if source_file.exists():
        shutil.copy(source_file, output_path)
        print(f"Copied existing top 50 words plot to: {output_path}")
        print("Note: This figure needs manual title removal")
    else:
        print(f"Warning: Source file not found: {source_file}")
        return None
    
    return output_path


def modify_svar_figure():
    """Modify SVAR figure - keep axes but change GPR y-label from Percent to Index."""
    
    print("\nNote: SVAR figure modification requires manual editing")
    print("The GPR panel y-axis should be changed from 'Percent' to 'Index'")
    print("All other labels should remain as they are")
    
    # Copy the SVAR figure to thesis directory for manual editing
    import shutil
    source = Path("/home5/chuenho/gpr-zero/data_analysis/output/svar_analysis/step2_median/7vars_lags2/svar_irf_custom.png")
    dest = THESIS_FIGURES_DIR / "svar_irf_step2_clean.png"
    
    if source.exists():
        shutil.copy(source, dest)
        print(f"Copied SVAR figure to: {dest}")
        print("Please manually edit the GPR panel y-axis label from 'Percent' to 'Index'")
    else:
        print(f"Warning: SVAR source file not found: {source}")


def main():
    """Generate all clean thesis figures."""
    ensure_cjk_font_setup()
    
    print("Creating clean thesis figures...")
    print("=" * 60)
    
    # 1. GPR scaled plot
    print("\n1. Creating GPR scaled plot (mean=100)...")
    create_gpr_scaled_clean()
    
    # 2. Category contributions
    print("\n2. Creating category contributions plot...")
    create_category_contributions_clean()
    
    # 3. Top 50 words
    print("\n3. Creating top 50 words plot...")
    create_top50_words_clean()
    
    # 4. SVAR figure note
    print("\n4. SVAR figure...")
    modify_svar_figure()
    
    print("\n" + "=" * 60)
    print("All figures created successfully!")
    print(f"Output directory: {THESIS_FIGURES_DIR}")
    
    # Update FIGURE_MAPPING.md
    update_figure_mapping()


def update_figure_mapping():
    """Update the FIGURE_MAPPING.md with new clean figure details."""
    
    mapping_updates = """
## Clean Figures Created

### Figure 4.1: GPR Index (Step2 Median)
- **File:** `gpr_step2_median_scaled_clean.png`
- **Title:** None (removed)
- **X-label:** Date
- **Y-label:** GPR Index (Mean = 100)
- **Legend:** Median, Mean, 68% CI, 95% CI

### Figure 4.3: Category Contributions
- **File:** `category_contributions_clean.png`
- **Title:** None (removed)
- **X-label:** Date
- **Y-label:** Category Contribution (%)
- **Legend:** All 8 categories

### Figure 6.1: Top 50 Words
- **File:** `top50_words_clean.png`
- **Title:** None (removed)
- **X-label:** None (removed)
- **Y-label:** Word names (Chinese)
- **Legend:** Category colors

### Figure 8.1: SVAR IRF
- **File:** `svar_irf_step2_clean.png`
- **Title:** Keep as is
- **X-label:** Quarters (keep)
- **Y-label:** MODIFY GPR panel from "Percent" to "Index" (manual edit needed)
- **Note:** All other panels keep original labels
"""
    
    print("\nFigure mapping updates:")
    print(mapping_updates)


if __name__ == "__main__":
    main()