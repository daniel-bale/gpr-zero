#!/usr/bin/env python3
"""
Create clean newspaper comparison plot with English names only and no confidence bands.
Also calculate and display correlations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style first before any other matplotlib operations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Reset to default font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Setup
output_dir = Path("/home5/chuenho/gpr-zero/data/newspaper_gpr_step2")
thesis_dir = Path("/home5/chuenho/thesis/figures")

# English names for newspapers
NEWSPAPERS = {
    'China Times': 'china',
    'Liberty Times': 'liberty', 
    'United Daily': 'united'
}

# Load data
newspaper_data = {}
for name, code in NEWSPAPERS.items():
    file_path = output_dir / f"{code}_step2_median_gpr.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        newspaper_data[name] = df

# Create clean plot without bands
fig, ax = plt.subplots(figsize=(14, 7))

colors = {
    'China Times': '#d62728',   # Red
    'Liberty Times': '#1f77b4', # Blue
    'United Daily': '#ff7f0e'   # Orange
}

# Plot each newspaper
for name, df in newspaper_data.items():
    # Filter to 2017-2024
    mask = (df.index >= '2017-01-01') & (df.index < '2025-01-01')
    df_filtered = df[mask]
    
    # Create segments for missing data
    date_range = pd.date_range(start='2017-01-01', end='2024-12-31', freq='W-MON')
    complete_df = pd.DataFrame(index=date_range)
    complete_df['value'] = df_filtered['median_scaled']
    
    # Plot with gaps for missing data
    valid_mask = ~complete_df['value'].isna()
    
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
        segment_values = complete_df.iloc[segment]['value']
        ax.plot(segment_dates, segment_values, 
               color=colors[name], linewidth=2, alpha=0.9,
               label=name if segment == segments[0] else "")

# Formatting
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('GPR Index', fontsize=12)
ax.set_title('Newspaper-Specific GPR Indices (Step2 Median)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Mean = 100')

# Set y-axis to start from 0
ax.set_ylim(bottom=0)

plt.tight_layout()

# Save
output_file = output_dir / "newspaper_comparison_clean.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved clean comparison plot to {output_file}")

# Create version without title for thesis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each newspaper (same as above but without title)
for name, df in newspaper_data.items():
    mask = (df.index >= '2017-01-01') & (df.index < '2025-01-01')
    df_filtered = df[mask]
    
    date_range = pd.date_range(start='2017-01-01', end='2024-12-31', freq='W-MON')
    complete_df = pd.DataFrame(index=date_range)
    complete_df['value'] = df_filtered['median_scaled']
    
    valid_mask = ~complete_df['value'].isna()
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
    
    for segment in segments:
        segment_dates = complete_df.iloc[segment].index
        segment_values = complete_df.iloc[segment]['value']
        ax.plot(segment_dates, segment_values, 
               color=colors[name], linewidth=2, alpha=0.9,
               label=name if segment == segments[0] else "")

ax.set_xlabel('Date')
ax.set_ylabel('GPR Index')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(bottom=0)

plt.tight_layout()

# Save thesis version
thesis_file = thesis_dir / "gpr_comparison_newspapers_clean_2017_01_to_2024_12.png"
plt.savefig(thesis_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved thesis figure to {thesis_file}")

# Calculate correlations on overlapping periods only
print("\n=== CORRELATION ANALYSIS ===")

# Align all data to common dates
common_dates = None
aligned_data = {}

for name, df in newspaper_data.items():
    # Filter to 2017-2024
    mask = (df.index >= '2017-01-01') & (df.index < '2025-01-01')
    df_filtered = df[mask]['median_scaled']
    
    if common_dates is None:
        common_dates = set(df_filtered.dropna().index)
    else:
        common_dates = common_dates.intersection(set(df_filtered.dropna().index))
    
    aligned_data[name] = df_filtered

# Create DataFrame with only common dates
common_df = pd.DataFrame()
for name, series in aligned_data.items():
    common_df[name] = series.loc[series.index.isin(common_dates)]

# Calculate correlation matrix
corr_matrix = common_df.corr()

print(f"\nCorrelation Matrix (based on {len(common_dates)} common weeks):")
print(corr_matrix)

# Save correlation matrix
corr_file = output_dir / "newspaper_correlations_updated.csv"
corr_matrix.to_csv(corr_file)

# Print specific correlations
print("\nPairwise Correlations:")
print(f"China Times - Liberty Times: {corr_matrix.loc['China Times', 'Liberty Times']:.3f}")
print(f"China Times - United Daily: {corr_matrix.loc['China Times', 'United Daily']:.3f}")
print(f"Liberty Times - United Daily: {corr_matrix.loc['Liberty Times', 'United Daily']:.3f}")

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0.5, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=0, vmax=1)
plt.title('Correlation Matrix of Newspaper GPR Indices')
plt.tight_layout()

heatmap_file = output_dir / "newspaper_correlations_heatmap_updated.png"
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSaved correlation heatmap to {heatmap_file}")

# Analyze periods of divergence
print("\n=== DIVERGENCE ANALYSIS ===")

# Calculate rolling correlation (52-week window)
window = 52
rolling_corr = {}

for i, name1 in enumerate(newspaper_data.keys()):
    for j, name2 in enumerate(newspaper_data.keys()):
        if i < j:
            pair_name = f"{name1}-{name2}"
            # Align the two series
            df1 = newspaper_data[name1]['median_scaled']
            df2 = newspaper_data[name2]['median_scaled']
            
            # Combine and forward fill to handle missing data
            combined = pd.DataFrame({
                name1: df1,
                name2: df2
            })
            
            # Calculate rolling correlation
            rolling_corr[pair_name] = combined[name1].rolling(window=window).corr(combined[name2])

# Plot rolling correlations
fig, ax = plt.subplots(figsize=(14, 6))

for pair, corr_series in rolling_corr.items():
    # Filter to 2017-2024
    mask = (corr_series.index >= '2017-01-01') & (corr_series.index < '2025-01-01')
    corr_filtered = corr_series[mask]
    ax.plot(corr_filtered.index, corr_filtered, label=pair, linewidth=2)

ax.set_xlabel('Date')
ax.set_ylabel('Rolling Correlation (52-week window)')
ax.set_title('Time-Varying Correlations Between Newspapers')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.2, 1.0)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()

rolling_file = output_dir / "newspaper_rolling_correlations.png"
plt.savefig(rolling_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved rolling correlation plot to {rolling_file}")

print("\n=== ANALYSIS COMPLETE ===")