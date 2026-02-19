#!/usr/bin/env python3
"""
Create median GPR index plots with confidence bands.
Shows the median GPR index across all dictionaries with 68% and 95% confidence intervals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gpr_zero.utils.font_setup import ensure_cjk_font_setup
import json
from scipy import stats

def create_gpr_median_plot(timeseries_df, metadata, output_path, data_dir=None):
    """Create median GPR index plot with confidence bands (scaled to mean=100)."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set date column as index
    timeseries_df = timeseries_df.set_index('warc_date_bucketed')
    
    # Filter to date range 2017-01-01 to 2024-12-31
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2024-12-31')
    timeseries_df = timeseries_df[(timeseries_df.index >= start_date) & 
                                  (timeseries_df.index <= end_date)]
    
    # Get all dictionary columns
    dict_cols = [col for col in timeseries_df.columns if col in metadata]
    gpr_data = timeseries_df[dict_cols]
    
    # Scale each dictionary series to have mean = 100
    scaled_data = pd.DataFrame()
    for col in dict_cols:
        col_mean = gpr_data[col].mean()
        if col_mean > 0:
            scaled_data[col] = (gpr_data[col] / col_mean) * 100
    
    # Calculate statistics for scaled data
    median = scaled_data.median(axis=1)
    mean = scaled_data.mean(axis=1)
    p16 = scaled_data.quantile(0.16, axis=1)  # ~1 std below (68% CI)
    p84 = scaled_data.quantile(0.84, axis=1)  # ~1 std above (68% CI)
    p025 = scaled_data.quantile(0.025, axis=1)  # ~2 std below (95% CI)
    p975 = scaled_data.quantile(0.975, axis=1)  # ~2 std above (95% CI)
    
    # Store the median timeseries data if data_dir is provided
    if data_dir:
        median_dir = data_dir / "median_timeseries"
        median_dir.mkdir(exist_ok=True)
        
        # Store scaled median data
        median_df = pd.DataFrame({
            'median': median,
            'mean': mean,
            'p16': p16,
            'p84': p84,
            'p025': p025,
            'p975': p975,
            'min': scaled_data.min(axis=1),
            'max': scaled_data.max(axis=1),
            'q25': scaled_data.quantile(0.25, axis=1),
            'q75': scaled_data.quantile(0.75, axis=1),
            'n_dictionaries': len(dict_cols)
        })
        median_df.to_csv(median_dir / "median_gpr_overall_scaled.csv")
        
        # Also store raw median data
        raw_median_df = pd.DataFrame({
            'median': gpr_data.median(axis=1),
            'mean': gpr_data.mean(axis=1),
            'p16': gpr_data.quantile(0.16, axis=1),
            'p84': gpr_data.quantile(0.84, axis=1),
            'p025': gpr_data.quantile(0.025, axis=1),
            'p975': gpr_data.quantile(0.975, axis=1),
            'min': gpr_data.min(axis=1),
            'max': gpr_data.max(axis=1),
            'q25': gpr_data.quantile(0.25, axis=1),
            'q75': gpr_data.quantile(0.75, axis=1),
            'n_dictionaries': len(dict_cols)
        })
        raw_median_df.to_csv(median_dir / "median_gpr_overall_raw.csv")
    
    # Plot median line
    ax.plot(median.index, median.values, color='darkblue', linewidth=2.5)
    
    # Plot confidence bands
    ax.fill_between(median.index, p16.values, p84.values, 
                   color='steelblue', alpha=0.4)
    ax.fill_between(median.index, p025.values, p975.values, 
                   color='lightblue', alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=45)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(start_date, end_date)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_gpr_median_by_step_plot(timeseries_df, metadata, output_path, data_dir=None):
    """Create median GPR index plot comparing Step1 vs Step2 (scaled to mean=100)."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set date column as index
    timeseries_df = timeseries_df.set_index('warc_date_bucketed')
    
    # Filter to date range 2017-01-01 to 2024-12-31
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2024-12-31')
    timeseries_df = timeseries_df[(timeseries_df.index >= start_date) & 
                                  (timeseries_df.index <= end_date)]
    
    # Separate dictionaries by step
    step1_cols = [col for col in timeseries_df.columns 
                  if col in metadata and metadata[col]['step'] == 'step1']
    step2_cols = [col for col in timeseries_df.columns 
                  if col in metadata and metadata[col]['step'] == 'step2']
    
    # Plot for each step
    for step_cols, label, color in [(step1_cols, 'Step1 (Role-based)', '#d62728'), 
                                     (step2_cols, 'Step2 (Stable prompt)', '#2ca02c')]:
        if step_cols:
            # Get values for dictionaries in this step
            step_data = timeseries_df[step_cols]
            
            # Scale each dictionary to mean = 100
            scaled_step_data = pd.DataFrame()
            for col in step_cols:
                col_mean = step_data[col].mean()
                if col_mean > 0:
                    scaled_step_data[col] = (step_data[col] / col_mean) * 100
            
            # Calculate statistics on scaled data
            median = scaled_step_data.median(axis=1)
            p16 = scaled_step_data.quantile(0.16, axis=1)
            p84 = scaled_step_data.quantile(0.84, axis=1)
            
            # Plot median line
            ax.plot(median.index, median.values, color=color, linewidth=2)
            
            # Plot confidence band
            ax.fill_between(median.index, p16.values, p84.values, 
                           color=color, alpha=0.2)
            
            # Store step-specific median data if data_dir is provided
            if data_dir:
                median_dir = data_dir / "median_timeseries"
                median_dir.mkdir(exist_ok=True)
                
                step_name = 'step1' if color == '#d62728' else 'step2'
                median_df = pd.DataFrame({
                    'median': median,
                    'mean': scaled_step_data.mean(axis=1),
                    'p16': p16,
                    'p84': p84,
                    'p025': scaled_step_data.quantile(0.025, axis=1),
                    'p975': scaled_step_data.quantile(0.975, axis=1),
                    'n_dictionaries': len(step_cols)
                })
                median_df.to_csv(median_dir / f"median_gpr_{step_name}_scaled.csv")
    
    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=45)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(start_date, end_date)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_gpr_median_by_model_plot(timeseries_df, metadata, output_path, data_dir=None):
    """Create median GPR index plot comparing different models (scaled to mean=100)."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set date column as index
    timeseries_df = timeseries_df.set_index('warc_date_bucketed')
    
    # Filter to date range 2017-01-01 to 2024-12-31
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2024-12-31')
    timeseries_df = timeseries_df[(timeseries_df.index >= start_date) & 
                                  (timeseries_df.index <= end_date)]
    
    # Define model colors (including GPT-4o)
    model_colors = {
        'claude-opus-4': '#1f77b4',
        'gemini-2.5-pro': '#ff7f0e',
        'gpt-4.1': '#2ca02c',
        'gpt-4o': '#d62728'  # Red for GPT-4o
    }
    
    # Plot for each model
    for model, color in model_colors.items():
        model_cols = [col for col in timeseries_df.columns 
                      if col in metadata and metadata[col]['model'] == model]
        
        if model_cols:
            # Get values for dictionaries from this model
            model_data = timeseries_df[model_cols]
            
            # Scale each dictionary to mean = 100
            scaled_model_data = pd.DataFrame()
            for col in model_cols:
                col_mean = model_data[col].mean()
                if col_mean > 0:
                    scaled_model_data[col] = (model_data[col] / col_mean) * 100
            
            # Calculate statistics on scaled data
            median = scaled_model_data.median(axis=1)
            p16 = scaled_model_data.quantile(0.16, axis=1)
            p84 = scaled_model_data.quantile(0.84, axis=1)
            
            # Plot median line
            ax.plot(median.index, median.values, color=color, linewidth=2, 
                   label=f'{model}')
            
            # Plot confidence band
            ax.fill_between(median.index, p16.values, p84.values, 
                           color=color, alpha=0.2)
            
            # Store model-specific median data if data_dir is provided
            if data_dir:
                median_dir = data_dir / "median_timeseries"
                median_dir.mkdir(exist_ok=True)
                
                model_filename = model.replace('-', '_').replace('.', '_')
                median_df = pd.DataFrame({
                    'median': median,
                    'mean': scaled_model_data.mean(axis=1),
                    'p16': p16,
                    'p84': p84,
                    'p025': scaled_model_data.quantile(0.025, axis=1),
                    'p975': scaled_model_data.quantile(0.975, axis=1),
                    'n_dictionaries': len(model_cols)
                })
                median_df.to_csv(median_dir / f"median_gpr_{model_filename}_scaled.csv")
    
    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(start_date, end_date)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_gpr_range_plot(timeseries_df, metadata, output_path, data_dir=None):
    """Create GPR index range plot showing min/max envelope (scaled to mean=100)."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set date column as index
    timeseries_df = timeseries_df.set_index('warc_date_bucketed')
    
    # Filter to date range 2017-01-01 to 2024-12-31
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2024-12-31')
    timeseries_df = timeseries_df[(timeseries_df.index >= start_date) & 
                                  (timeseries_df.index <= end_date)]
    
    # Get all dictionary columns
    dict_cols = [col for col in timeseries_df.columns if col in metadata]
    gpr_data = timeseries_df[dict_cols]
    
    # Scale each dictionary to mean = 100
    scaled_data = pd.DataFrame()
    for col in dict_cols:
        col_mean = gpr_data[col].mean()
        if col_mean > 0:
            scaled_data[col] = (gpr_data[col] / col_mean) * 100
    
    # Calculate statistics on scaled data
    median = scaled_data.median(axis=1)
    mean = scaled_data.mean(axis=1)
    minimum = scaled_data.min(axis=1)
    maximum = scaled_data.max(axis=1)
    q25 = scaled_data.quantile(0.25, axis=1)
    q75 = scaled_data.quantile(0.75, axis=1)
    
    # Plot range envelope
    ax.fill_between(median.index, minimum.values, maximum.values, 
                   color='lightgray', alpha=0.3)
    
    # Plot IQR
    ax.fill_between(median.index, q25.values, q75.values, 
                   color='steelblue', alpha=0.4)
    
    # Plot median
    ax.plot(median.index, median.values, color='darkblue', linewidth=2.5)
    
    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=45)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(start_date, end_date)
    
    # Add statistics text
    stats_text = f"""Dictionary Count: {len(dict_cols)}
Date Range: {timeseries_df.index.min().strftime('%Y-%m-%d')} to {timeseries_df.index.max().strftime('%Y-%m-%d')}
Observations: {len(timeseries_df)}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_gpr_scaled_plot(timeseries_df, metadata, output_path, data_dir=None):
    """Create scaled GPR index plot with mean = 100."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Set date column as index
    timeseries_df = timeseries_df.set_index('warc_date_bucketed')
    
    # Get all dictionary columns
    dict_cols = [col for col in timeseries_df.columns if col in metadata]
    gpr_data = timeseries_df[dict_cols]
    
    # Calculate statistics for original data
    median = gpr_data.median(axis=1)
    mean = gpr_data.mean(axis=1)
    p16 = gpr_data.quantile(0.16, axis=1)
    p84 = gpr_data.quantile(0.84, axis=1)
    p025 = gpr_data.quantile(0.025, axis=1)
    p975 = gpr_data.quantile(0.975, axis=1)
    
    # Upper panel: Original GPR index
    ax1.plot(median.index, median.values, color='darkblue', linewidth=2.5)
    ax1.fill_between(median.index, p16.values, p84.values, 
                   color='steelblue', alpha=0.4)
    ax1.fill_between(median.index, p025.values, p975.values, 
                   color='lightblue', alpha=0.3)
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, None)
    
    # Lower panel: Scaled GPR index (mean = 100)
    # Scale each dictionary series to have mean = 100
    scaled_data = pd.DataFrame()
    for col in dict_cols:
        col_mean = gpr_data[col].mean()
        if col_mean > 0:
            scaled_data[col] = (gpr_data[col] / col_mean) * 100
    
    # Calculate statistics for scaled data
    scaled_median = scaled_data.median(axis=1)
    scaled_p16 = scaled_data.quantile(0.16, axis=1)
    scaled_p84 = scaled_data.quantile(0.84, axis=1)
    scaled_p025 = scaled_data.quantile(0.025, axis=1)
    scaled_p975 = scaled_data.quantile(0.975, axis=1)
    
    ax2.plot(scaled_median.index, scaled_median.values, color='darkblue', linewidth=2.5)
    ax2.fill_between(scaled_median.index, scaled_p16.values, scaled_p84.values, 
                   color='steelblue', alpha=0.4)
    ax2.fill_between(scaled_median.index, scaled_p025.values, scaled_p975.values, 
                   color='lightblue', alpha=0.3)
    
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis
    ax2.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_gpr_overall_by_step(timeseries_df, metadata, output_path, data_dir=None):
    """Create two separate figures for overall GPR by step (scaled to mean=100)."""
    
    # Set date column as index
    timeseries_df_indexed = timeseries_df.set_index('warc_date_bucketed')
    
    # Filter to date range 2017-01-01 to 2024-12-31
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2024-12-31')
    timeseries_df_indexed = timeseries_df_indexed[(timeseries_df_indexed.index >= start_date) & 
                                                  (timeseries_df_indexed.index <= end_date)]
    
    # Separate dictionaries by step
    step1_cols = [col for col in timeseries_df_indexed.columns 
                  if col in metadata and metadata[col]['step'] == 'step1']
    step2_cols = [col for col in timeseries_df_indexed.columns 
                  if col in metadata and metadata[col]['step'] == 'step2']
    
    for step_cols, step_name, filename in [(step1_cols, 'Step1 (Role-based)', 'gpr_overall_step1.png'), 
                                           (step2_cols, 'Step2 (Stable prompt)', 'gpr_overall_step2.png')]:
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if step_cols:
            # Get values for dictionaries in this step
            step_data = timeseries_df_indexed[step_cols]
            
            # Scale each dictionary to mean = 100
            scaled_step_data = pd.DataFrame()
            for col in step_cols:
                col_mean = step_data[col].mean()
                if col_mean > 0:
                    scaled_step_data[col] = (step_data[col] / col_mean) * 100
            
            # Calculate statistics on scaled data
            median = scaled_step_data.median(axis=1)
            mean = scaled_step_data.mean(axis=1)
            p16 = scaled_step_data.quantile(0.16, axis=1)
            p84 = scaled_step_data.quantile(0.84, axis=1)
            p025 = scaled_step_data.quantile(0.025, axis=1)
            p975 = scaled_step_data.quantile(0.975, axis=1)
            
            # Plot median line
            ax.plot(median.index, median.values, color='darkblue', linewidth=2.5)
            
            # Plot confidence bands
            ax.fill_between(median.index, p16.values, p84.values, 
                           color='steelblue', alpha=0.4)
            ax.fill_between(median.index, p025.values, p975.values, 
                           color='lightblue', alpha=0.3)
        
        # Format plot
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(start_date, end_date)
        
        # Format x-axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path.parent / filename, dpi=300, bbox_inches='tight')
        plt.close()

def create_gpr_median_by_model_plot_with_gpt4o(output_path, data_dir=None):
    """Create median GPR index plot comparing different models INCLUDING GPT-4o (scaled to mean=100)."""
    
    # Load the full dataset INCLUDING GPT-4o
    data_dir = Path("/home5/chuenho/gpr-zero/data/gpr_correlations")
    timeseries_df = pd.read_csv(data_dir / "combined_gpr_timeseries.csv")
    timeseries_df['warc_date_bucketed'] = pd.to_datetime(timeseries_df['warc_date_bucketed'])
    
    # Load metadata with GPT-4o
    with open(data_dir / "dictionary_metadata.json", 'r') as f:
        metadata_full = json.load(f)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set date column as index
    timeseries_df = timeseries_df.set_index('warc_date_bucketed')
    
    # Filter to date range 2017-01-01 to 2024-12-31
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2024-12-31')
    timeseries_df = timeseries_df[(timeseries_df.index >= start_date) & 
                                  (timeseries_df.index <= end_date)]
    
    # Define model colors (including GPT-4o)
    model_colors = {
        'claude-opus-4': '#1f77b4',
        'gemini-2.5-pro': '#ff7f0e',
        'gpt-4.1': '#2ca02c',
        'gpt-4o': '#d62728'  # Red for GPT-4o
    }
    
    # Plot for each model
    for model, color in model_colors.items():
        model_cols = [col for col in timeseries_df.columns 
                      if col in metadata_full and metadata_full[col]['model'] == model]
        
        if model_cols:
            # Get values for dictionaries from this model
            model_data = timeseries_df[model_cols]
            
            # Scale each dictionary to mean = 100
            scaled_model_data = pd.DataFrame()
            for col in model_cols:
                col_mean = model_data[col].mean()
                if col_mean > 0:
                    scaled_model_data[col] = (model_data[col] / col_mean) * 100
            
            # Calculate statistics on scaled data
            median = scaled_model_data.median(axis=1)
            p16 = scaled_model_data.quantile(0.16, axis=1)
            p84 = scaled_model_data.quantile(0.84, axis=1)
            
            # Plot median line
            label = f'{model} Median (n={len(model_cols)})'
            ax.plot(median.index, median.values, color=color, linewidth=2, label=label)
            
            # Plot confidence band
            ax.fill_between(median.index, p16.values, p84.values, 
                           color=color, alpha=0.2)
    
    # Add reference line at 100
    ax.axhline(y=100, color='gray', linestyle='-', alpha=0.5, label='Reference (100)')
    
    # Format plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Scaled GPR Index (Mean = 100)', fontsize=12)
    ax.set_title('Taiwan GPR Index: Model Comparison (Including GPT-4o)\n(Scaled to Mean=100, 2017-2024)', 
                fontsize=14, pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(start_date, end_date)
    ax.set_ylim(0, None)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_plot_documentation(plots_dir, median_dir):
    """Create documentation for all plots and stored data."""
    
    doc_content = {
        "title": "GPR Index Visualization and Median Timeseries Documentation (Expanded Step2)",
        "description": "Documentation for GPR index plots with balanced Step1 vs Step2 comparison",
        "date_generated": pd.Timestamp.now().isoformat(),
        "date_range": "2017-01-01 to 2024-12-31",
        "total_dictionaries": 75,
        "model_counts": {
            "claude-opus-4": 18,
            "gemini-2.5-pro": 17,
            "gpt-4.1": 15,
            "gpt-4o": 20
        },
        "step_counts": {
            "step1": 35,
            "step2": 40
        },
        "plots": {
            "gpr_median_with_bands.png": {
                "description": "Overall median GPR index with 68% and 95% confidence bands",
                "details": "Shows the median GPR index across all 75 dictionaries, scaled to mean=100",
                "visualization": "Median line (dark blue) with confidence intervals",
                "data_stored": ["median_gpr_overall_scaled.csv", "median_gpr_overall_raw.csv"]
            },
            "gpr_median_by_step.png": {
                "description": "Comparison of Step1 (role-based) vs Step2 (stable prompt) median GPR",
                "details": "Compares the two dictionary generation approaches",
                "visualization": "Two median lines with confidence bands - Step1 (red) and Step2 (green)",
                "data_stored": ["median_gpr_step1_scaled.csv", "median_gpr_step2_scaled.csv"]
            },
            "gpr_median_by_model.png": {
                "description": "Model comparison showing median GPR for each LLM",
                "details": "Compares performance across Claude-Opus-4, Gemini-2.5-pro, GPT-4.1, and GPT-4o",
                "visualization": "Four median lines with confidence bands, one per model",
                "data_stored": [
                    "median_gpr_claude_opus_4_scaled.csv",
                    "median_gpr_gemini_2_5_pro_scaled.csv",
                    "median_gpr_gpt_4_1_scaled.csv",
                    "median_gpr_gpt_4o_scaled.csv"
                ]
            },
            "gpr_range_plot.png": {
                "description": "GPR index range showing min/max envelope and IQR",
                "details": "Displays the full range of GPR values across dictionaries",
                "visualization": "Min-max range (light gray), IQR (steel blue), and median line"
            },
            "gpr_scaled_mean100.png": {
                "description": "Two-panel plot showing original and scaled GPR indices",
                "details": "Upper panel: raw GPR values, Lower panel: scaled to mean=100",
                "visualization": "Side-by-side comparison of raw vs scaled data"
            },
            "gpr_overall_step1.png": {
                "description": "Step1-only median GPR with full confidence bands",
                "details": "Focused view of Step1 (role-based) dictionaries performance",
                "visualization": "Single median line with 68% and 95% confidence bands"
            },
            "gpr_overall_step2.png": {
                "description": "Step2-only median GPR with full confidence bands",
                "details": "Focused view of Step2 (stable prompt) dictionaries performance",
                "visualization": "Single median line with 68% and 95% confidence bands"
            }
        },
        "stored_data": {
            "median_gpr_overall_raw.csv": {
                "description": "Raw median GPR values across all dictionaries",
                "columns": ["median", "mean", "p16", "p84", "p025", "p975", "min", "max", "q25", "q75", "n_dictionaries"],
                "use_case": "For analysis requiring original GPR scale"
            },
            "median_gpr_overall_scaled.csv": {
                "description": "Scaled median GPR (mean=100) across all dictionaries",
                "columns": ["median", "mean", "p16", "p84", "p025", "p975", "min", "max", "q25", "q75", "n_dictionaries"],
                "use_case": "For normalized comparisons and visualization"
            },
            "median_gpr_step[1/2]_scaled.csv": {
                "description": "Step-specific median GPR scaled to mean=100",
                "columns": ["median", "mean", "p16", "p84", "p025", "p975", "n_dictionaries"],
                "use_case": "For comparing dictionary generation approaches"
            },
            "median_gpr_[model]_scaled.csv": {
                "description": "Model-specific median GPR scaled to mean=100",
                "columns": ["median", "mean", "p16", "p84", "p025", "p975", "n_dictionaries"],
                "use_case": "For comparing LLM performance"
            }
        },
        "column_definitions": {
            "median": "Median value across dictionaries at each time point",
            "mean": "Mean value across dictionaries",
            "p16": "16th percentile (~1 std below median for 68% CI)",
            "p84": "84th percentile (~1 std above median for 68% CI)",
            "p025": "2.5th percentile (~2 std below median for 95% CI)",
            "p975": "97.5th percentile (~2 std above median for 95% CI)",
            "min": "Minimum value across dictionaries",
            "max": "Maximum value across dictionaries",
            "q25": "First quartile (25th percentile)",
            "q75": "Third quartile (75th percentile)",
            "n_dictionaries": "Number of dictionaries used in calculation"
        },
        "notes": [
            "All plots are prepared for LaTeX integration (no labels/titles)",
            "Scaling to mean=100 allows fair comparison across dictionaries with different base levels",
            "Confidence intervals assume approximately normal distribution",
            "GPT-4.1 includes corrected taiwan dictionaries (renamed to proper GPT-4.1 run2)"
        ]
    }
    
    # Save documentation
    doc_path = plots_dir.parent / "plot_documentation.json"
    with open(doc_path, 'w') as f:
        json.dump(doc_content, f, indent=2)
    
    print(f"\nDocumentation saved to: {doc_path}")

def main():
    # Setup fonts
    ensure_cjk_font_setup()
    
    # Load data - NOW WITH EXPANDED STEP2 (35 Step1 + 40 Step2 = 75 total)
    data_dir = Path("/home5/chuenho/gpr-zero/data/gpr_correlations")
    plots_dir = data_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("Loading expanded data...")
    
    # Load the expanded time series data with Step2 expansion
    timeseries_df = pd.read_csv(data_dir / "combined_gpr_timeseries_step2_expanded.csv")
    timeseries_df['warc_date_bucketed'] = pd.to_datetime(timeseries_df['warc_date_bucketed'])
    
    # Load the expanded metadata
    with open(data_dir / "dictionary_metadata_step2_expanded.json", 'r') as f:
        metadata = json.load(f)
    
    print("Creating GPR median plot with confidence bands...")
    create_gpr_median_plot(
        timeseries_df, 
        metadata, 
        plots_dir / "gpr_median_with_bands.png",
        data_dir  # Pass data_dir to store median timeseries
    )
    
    print("Creating GPR median by step comparison...")
    create_gpr_median_by_step_plot(
        timeseries_df, 
        metadata, 
        plots_dir / "gpr_median_by_step.png",
        data_dir  # Pass data_dir to store step-specific median
    )
    
    print("Creating GPR median by model comparison...")
    create_gpr_median_by_model_plot(
        timeseries_df, 
        metadata, 
        plots_dir / "gpr_median_by_model.png",
        data_dir  # Pass data_dir to store model-specific median
    )
    
    print("Creating GPR range plot...")
    create_gpr_range_plot(
        timeseries_df, 
        metadata, 
        plots_dir / "gpr_range_plot.png",
        data_dir
    )
    
    print("Creating scaled GPR plot...")
    create_gpr_scaled_plot(
        timeseries_df,
        metadata,
        plots_dir / "gpr_scaled_mean100.png",
        data_dir
    )
    
    print("Creating overall GPR plots by step...")
    create_gpr_overall_by_step(
        timeseries_df,
        metadata,
        plots_dir / "dummy",  # Will create individual files
        data_dir
    )
    
    print(f"\nAll GPR median plots saved to: {plots_dir}")
    print("\nGPR plot files created:")
    gpr_plots = ['gpr_median_with_bands.png', 'gpr_median_by_step.png', 
                 'gpr_median_by_model.png', 'gpr_range_plot.png',
                 'gpr_scaled_mean100.png', 'gpr_overall_step1.png',
                 'gpr_overall_step2.png']
    for plot_file in gpr_plots:
        if (plots_dir / plot_file).exists():
            print(f"  - {plot_file}")
    
    # Print stored median timeseries data
    median_dir = data_dir / "median_timeseries"
    if median_dir.exists():
        print(f"\nMedian timeseries data saved to: {median_dir}")
        print("Median timeseries files created:")
        for csv_file in sorted(median_dir.glob("*.csv")):
            print(f"  - {csv_file.name}")
    
    # Create summary documentation
    create_plot_documentation(plots_dir, median_dir)

if __name__ == "__main__":
    main()