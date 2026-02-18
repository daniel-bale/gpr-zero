#!/usr/bin/env python3
"""
Compute pairwise correlations using the expanded Step2 dataset (35 Step1 + 40 Step2).
This version uses the balanced dataset with proper Step1 vs Step2 comparison.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

def main():
    """Compute correlations using expanded Step2 data with date filtering."""
    
    # Setup
    ensure_cjk_font_setup()
    data_dir = Path("/home5/chuenho/gpr-zero/data/gpr_correlations")
    
    print("Loading expanded GPR time series data...")
    
    # Load the expanded time series data
    timeseries_df = pd.read_csv(data_dir / "combined_gpr_timeseries_step2_expanded.csv")
    timeseries_df['warc_date_bucketed'] = pd.to_datetime(timeseries_df['warc_date_bucketed'])
    
    # Filter to date range 2017-01-01 to 2024-12-31
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2024-12-31')
    timeseries_df = timeseries_df[(timeseries_df['warc_date_bucketed'] >= start_date) & 
                                  (timeseries_df['warc_date_bucketed'] <= end_date)]
    
    # Set date as index and get dictionary columns
    timeseries_df = timeseries_df.set_index('warc_date_bucketed')
    
    # Load the expanded metadata
    with open(data_dir / "dictionary_metadata_step2_expanded.json", 'r') as f:
        metadata = json.load(f)
    
    # Get dictionary columns that exist in both data and metadata
    dict_cols = [col for col in timeseries_df.columns if col in metadata]
    combined_df = timeseries_df[dict_cols]
    
    print(f"Computing correlations for {len(dict_cols)} dictionaries")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Time periods: {len(combined_df)}")
    
    # Count by step and model
    step_counts = {}
    model_counts = {}
    
    for dict_name in dict_cols:
        meta = metadata[dict_name]
        if meta['step']:
            step_counts[meta['step']] = step_counts.get(meta['step'], 0) + 1
        if meta['model']:
            model_counts[meta['model']] = model_counts.get(meta['model'], 0) + 1
    
    print(f"\nDictionary distribution:")
    print(f"Step1: {step_counts.get('step1', 0)}")
    print(f"Step2: {step_counts.get('step2', 0)}")
    for model, count in model_counts.items():
        print(f"{model}: {count}")
    
    # Compute correlation matrix
    print("\nComputing correlation matrix...")
    correlation_matrix = combined_df.corr()
    
    # Save correlation matrix with step2_expanded suffix
    corr_output = data_dir / "gpr_correlation_matrix_step2_expanded.csv"
    correlation_matrix.to_csv(corr_output)
    print(f"Saved: {corr_output}")
    
    # Compute summary statistics
    print("Computing correlation statistics...")
    
    # Get upper triangle of correlation matrix (excluding diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    upper_triangle_corrs = correlation_matrix.where(mask).stack()
    
    stats = {
        'n_dictionaries': len(dict_cols),
        'n_time_periods': len(combined_df),
        'date_range': {
            'start': combined_df.index.min().isoformat(),
            'end': combined_df.index.max().isoformat()
        },
        'dictionary_distribution': {
            'by_step': step_counts,
            'by_model': model_counts
        },
        'correlation_stats': {
            'mean': float(upper_triangle_corrs.mean()),
            'median': float(upper_triangle_corrs.median()),
            'std': float(upper_triangle_corrs.std()),
            'min': float(upper_triangle_corrs.min()),
            'max': float(upper_triangle_corrs.max()),
            'q25': float(upper_triangle_corrs.quantile(0.25)),
            'q75': float(upper_triangle_corrs.quantile(0.75))
        }
    }
    
    # Save statistics with step2_expanded suffix
    stats_output = data_dir / "correlation_statistics_step2_expanded.json"
    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_output}")
    
    # Group analysis by model and step
    print("Computing group analysis...")
    
    # Group by model
    models = {}
    for dict_name, meta in metadata.items():
        if dict_name in dict_cols and meta['model'] and meta['model'] != 'taiwan':
            if meta['model'] not in models:
                models[meta['model']] = []
            models[meta['model']].append(dict_name)
    
    # Group by step
    steps = {}
    for dict_name, meta in metadata.items():
        if dict_name in dict_cols and meta['step']:
            if meta['step'] not in steps:
                steps[meta['step']] = []
            steps[meta['step']].append(dict_name)
    
    # Compute within-group correlations
    def compute_group_stats(groups, group_name):
        stats_dict = {'within_group': {}}
        
        for group, dicts in groups.items():
            if len(dicts) > 1:
                group_df = combined_df[dicts]
                group_corr = group_df.corr()
                
                # Get upper triangle
                mask = np.triu(np.ones_like(group_corr, dtype=bool), k=1)
                within_corrs = group_corr.where(mask).stack().dropna()
                
                stats_dict['within_group'][group] = {
                    'n_dicts': len(dicts),
                    'n_pairs': len(within_corrs),
                    'mean_corr': float(within_corrs.mean()),
                    'median_corr': float(within_corrs.median()),
                    'std_corr': float(within_corrs.std()),
                    'min_corr': float(within_corrs.min()),
                    'max_corr': float(within_corrs.max())
                }
        
        return stats_dict
    
    group_analysis = {
        'by_model': compute_group_stats(models, 'model'),
        'by_step': compute_group_stats(steps, 'step')
    }
    
    # Save group analysis with step2_expanded suffix
    group_output = data_dir / "group_analysis_step2_expanded.json"
    with open(group_output, 'w') as f:
        json.dump(group_analysis, f, indent=2)
    print(f"Saved: {group_output}")
    
    # Step1 vs Step2 analysis
    print("Computing step1 vs step2 analysis...")
    
    step1_dicts = steps.get('step1', [])
    step2_dicts = steps.get('step2', [])
    
    step_analysis = {
        'step1_count': len(step1_dicts),
        'step2_count': len(step2_dicts),
        'step1_summary_stats': {},
        'step2_summary_stats': {}
    }
    
    # Within-step1 correlations
    if len(step1_dicts) > 1:
        step1_df = combined_df[step1_dicts]
        step1_corr = step1_df.corr()
        mask = np.triu(np.ones_like(step1_corr, dtype=bool), k=1)
        step1_correlations = step1_corr.where(mask).stack().dropna()
        
        step_analysis['step1_summary_stats'] = {
            'mean_corr': float(step1_correlations.mean()),
            'median_corr': float(step1_correlations.median()),
            'std_corr': float(step1_correlations.std()),
            'min_corr': float(step1_correlations.min()),
            'max_corr': float(step1_correlations.max()),
            'mean_gpr': float(step1_df.mean().mean()),
            'std_gpr': float(step1_df.mean().std())
        }
    
    # Within-step2 correlations
    if len(step2_dicts) > 1:
        step2_df = combined_df[step2_dicts]
        step2_corr = step2_df.corr()
        mask = np.triu(np.ones_like(step2_corr, dtype=bool), k=1)
        step2_correlations = step2_corr.where(mask).stack().dropna()
        
        step_analysis['step2_summary_stats'] = {
            'mean_corr': float(step2_correlations.mean()),
            'median_corr': float(step2_correlations.median()),
            'std_corr': float(step2_correlations.std()),
            'min_corr': float(step2_correlations.min()),
            'max_corr': float(step2_correlations.max()),
            'mean_gpr': float(step2_df.mean().mean()),
            'std_gpr': float(step2_df.mean().std())
        }
    
    # Save step analysis with step2_expanded suffix
    step_output = data_dir / "step1_vs_step2_analysis_step2_expanded.json"
    with open(step_output, 'w') as f:
        json.dump(step_analysis, f, indent=2)
    print(f"Saved: {step_output}")
    
    # Placeholder for rolling correlations (simplified)
    rolling_stats = {}
    rolling_output = data_dir / "rolling_correlation_stats_step2_expanded.json"
    with open(rolling_output, 'w') as f:
        json.dump(rolling_stats, f, indent=2)
    print(f"Saved: {rolling_output}")
    
    print("\nSummary of Expanded Analysis:")
    print(f"Total dictionaries: {stats['n_dictionaries']}")
    print(f"Time periods: {stats['n_time_periods']}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"Mean correlation: {stats['correlation_stats']['mean']:.3f}")
    print(f"Median correlation: {stats['correlation_stats']['median']:.3f}")
    
    print(f"\nBalanced Step Comparison:")
    print(f"Step1 (Role-based): {step_analysis['step1_count']} dictionaries")
    print(f"Step2 (Stable prompt): {step_analysis['step2_count']} dictionaries")
    
    print(f"\nModel distribution:")
    for model, count in model_counts.items():
        if model != 'taiwan':
            print(f"  {model}: {count} dictionaries")
    
    print(f"\nExpanded correlation files saved to: {data_dir}")

if __name__ == "__main__":
    main()