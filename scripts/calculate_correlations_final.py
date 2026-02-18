#!/usr/bin/env python3
"""
Calculate final correlations for thesis:
1. Mean pairwise correlation among 40 Step2 indices
2. Mean pairwise correlation among 35 valid Step1 indices
3. Correlation between Step2 median and Lau's index
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
TIMESERIES_DIR = Path('data/gpr_timeseries_3outlets')
INDICES_DIR = Path('data/gpr_indices_3outlets')

# Problematic Step1 dictionaries to exclude
EXCLUDE_STEP1 = [
    'step1_claude-opus-4_defense_minister_run2',
    'step1_claude-opus-4_poli_professor_run2',
    'step1_gemini-2.5-pro_journalist_run1',
    'step1_gemini-2.5-pro_nsb_director_run2',
    'step1_gemini-2.5-pro_poli_professor_run2'
]

def calculate_pairwise_correlations(files, name):
    """Calculate mean pairwise correlation for a set of GPR index files."""
    print(f"\n=== {name} ===")
    print(f"Files: {len(files)}")
    
    # Load all time series
    data = {}
    for file in sorted(files):
        df = pd.read_parquet(file)
        dict_name = file.stem.replace('_W_overall_3outlets', '')
        # Create series with week as index, gpr_index as values
        data[dict_name] = df.set_index('week')['gpr_index']
    
    # Create DataFrame: rows=weeks, columns=dictionaries
    df_combined = pd.DataFrame(data)
    print(f"DataFrame shape: {df_combined.shape}")
    print(f"Date range: {df_combined.index.min()} to {df_combined.index.max()}")
    
    # Calculate correlation matrix
    corr_matrix = df_combined.corr()
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_triangle = corr_matrix.where(mask)
    
    # Get all correlation values
    correlations = upper_triangle.stack().values
    print(f"Number of pairwise correlations: {len(correlations)}")
    print(f"Expected: {len(files) * (len(files) - 1) // 2}")
    
    # Calculate statistics
    mean_corr = correlations.mean()
    std_corr = correlations.std()
    min_corr = correlations.min()
    max_corr = correlations.max()
    
    print(f"\nCorrelation statistics:")
    print(f"  Mean: {mean_corr:.6f}")
    print(f"  Std:  {std_corr:.6f}")
    print(f"  Min:  {min_corr:.6f}")
    print(f"  Max:  {max_corr:.6f}")
    
    return mean_corr

def main():
    """Calculate all correlations."""
    
    # 1. Step2 correlations (40 dictionaries)
    step2_files = list(TIMESERIES_DIR.glob('step2_*_W_overall_3outlets.parquet'))
    step2_corr = calculate_pairwise_correlations(step2_files, "STEP2 CORRELATIONS")
    
    # 2. Step1 correlations (35 valid dictionaries)
    all_step1_files = list(TIMESERIES_DIR.glob('step1_*_W_overall_3outlets.parquet'))
    valid_step1_files = [f for f in all_step1_files 
                         if f.stem.replace('_W_overall_3outlets', '') not in EXCLUDE_STEP1]
    step1_corr = calculate_pairwise_correlations(valid_step1_files, "STEP1 CORRELATIONS")
    
    # 3. Lau correlation
    print("\n=== LAU CORRELATION ===")
    median_file = INDICES_DIR / 'step2_median_gpr_3outlets.parquet'
    lau_file = TIMESERIES_DIR / 'lau_overall_taiwan_recalculated_W_overall_3outlets.parquet'
    
    median_df = pd.read_parquet(median_file)
    lau_df = pd.read_parquet(lau_file)
    
    print(f"Step2 median weeks: {len(median_df)}")
    print(f"Lau index weeks: {len(lau_df)}")
    
    # Merge on week column
    merged = median_df.merge(lau_df[['week', 'gpr_index']], on='week', suffixes=('_step2', '_lau'))
    print(f"Merged weeks: {len(merged)}")
    
    # Calculate correlation
    lau_corr = merged['gpr_index_step2'].corr(merged['gpr_index_lau'])
    print(f"\nCorrelation: {lau_corr:.6f}")
    
    # Summary statistics
    print(f"\nStep2 median statistics:")
    print(f"  Mean: {merged['gpr_index_step2'].mean():.6f}")
    print(f"  Std:  {merged['gpr_index_step2'].std():.6f}")
    
    print(f"\nLau index statistics:")
    print(f"  Mean: {merged['gpr_index_lau'].mean():.6f}")
    print(f"  Std:  {merged['gpr_index_lau'].std():.6f}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL CORRELATION VALUES FOR THESIS:")
    print("="*60)
    print(f"Mean pairwise correlation among 40 Step2 indices: {step2_corr:.3f}")
    print(f"Mean pairwise correlation among 35 Step1 indices: {step1_corr:.3f}")
    print(f"Correlation between Step2 median and Lau's index: {lau_corr:.3f}")

if __name__ == "__main__":
    main()