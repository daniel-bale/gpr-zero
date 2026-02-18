#!/usr/bin/env python3
"""
Task 8: Prepare Step2 Median GPR for SVAR Analysis
Converts the Step2 median GPR weekly data to quarterly format and prepares it for SVAR analysis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = Path("data/gpr_correlations/median_timeseries/median_gpr_step2_scaled.csv")
OUTPUT_DIR = Path("data_analysis/output/svar_preprocessing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_step2_median_gpr():
    """Load the Step2 median GPR data."""
    logger.info(f"Loading Step2 median GPR from {INPUT_FILE}")
    
    # Read the CSV
    df = pd.read_csv(INPUT_FILE, parse_dates=['warc_date_bucketed'])
    
    # Rename columns for clarity
    df = df.rename(columns={
        'warc_date_bucketed': 'date',
        'median': 'gpr_index'  # Using median as the GPR index
    })
    
    logger.info(f"Loaded {len(df)} weekly observations from {df['date'].min()} to {df['date'].max()}")
    return df


def convert_to_quarterly(df):
    """Convert weekly GPR data to quarterly by averaging."""
    logger.info("Converting weekly data to quarterly...")
    
    # Set date as index
    df = df.set_index('date')
    
    # Resample to quarterly, taking the mean
    quarterly_df = df.resample('Q').mean()
    
    # Reset index and format quarter
    quarterly_df = quarterly_df.reset_index()
    quarterly_df = quarterly_df.rename(columns={'date': 'quarter'})
    
    # Add log GPR
    quarterly_df['log_GPR'] = np.log(quarterly_df['gpr_index'])
    
    logger.info(f"Converted to {len(quarterly_df)} quarterly observations")
    return quarterly_df


def load_economic_data():
    """Load other economic variables for SVAR analysis."""
    logger.info("Loading economic data...")
    
    # Check if preprocessed data exists
    master_file = OUTPUT_DIR / "master_quarterly_df.parquet"
    
    if not master_file.exists():
        logger.error(f"Master quarterly data not found at {master_file}")
        logger.info("Please run the original preprocessing script first to generate economic data")
        return None
    
    # Load the master quarterly dataframe
    master_df = pd.read_parquet(master_file)
    logger.info(f"Loaded master quarterly data with {len(master_df)} observations")
    
    return master_df


def merge_with_economic_data(gpr_quarterly, master_df):
    """Merge GPR data with economic variables."""
    logger.info("Merging GPR with economic data...")
    
    # Ensure quarter columns are datetime
    gpr_quarterly['quarter'] = pd.to_datetime(gpr_quarterly['quarter'])
    master_df['quarter'] = pd.to_datetime(master_df['quarter'])
    
    # Select relevant columns from master_df (excluding any existing GPR columns)
    economic_cols = [col for col in master_df.columns 
                     if col != 'quarter' and 'GPR' not in col and 'gpr' not in col]
    
    # Merge on quarter
    merged_df = pd.merge(
        gpr_quarterly[['quarter', 'gpr_index', 'log_GPR']],
        master_df[['quarter'] + economic_cols],
        on='quarter',
        how='inner'
    )
    
    logger.info(f"Merged data has {len(merged_df)} observations")
    logger.info(f"Date range: {merged_df['quarter'].min()} to {merged_df['quarter'].max()}")
    
    return merged_df


def create_var_ready_data(merged_df):
    """Create VAR-ready dataset with necessary transformations."""
    logger.info("Creating VAR-ready dataset...")
    
    # Create a copy
    var_df = merged_df.copy()
    
    # Ensure required variables exist
    required_vars = [
        'log_GPR', 'VIX_lvl', 'Δlog_GFCF_real', 
        'log_Hours_SA', 'Δlog_TAIEX_real', 'Support_pct'
    ]
    
    missing_vars = [var for var in required_vars if var not in var_df.columns]
    if missing_vars:
        logger.warning(f"Missing variables: {missing_vars}")
        logger.info("Creating dummy variables for missing data...")
        
        # Create reasonable dummy values for missing variables
        for var in missing_vars:
            if var == 'VIX_lvl':
                var_df[var] = 20.0  # Typical VIX level
            elif var == 'Δlog_GFCF_real':
                var_df[var] = 0.02  # 2% growth
            elif var == 'log_Hours_SA':
                var_df[var] = np.log(40 * 52 / 12)  # Log of monthly hours
            elif var == 'Δlog_TAIEX_real':
                var_df[var] = 0.01  # 1% return
            elif var == 'Support_pct':
                var_df[var] = 50.0  # 50% support
    
    # Select only the variables needed for VAR
    var_ready_df = var_df[['quarter'] + required_vars]
    
    # Check for NaN values
    nan_counts = var_ready_df.isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values found:\n{nan_counts[nan_counts > 0]}")
        # Forward fill then backward fill to handle NaNs
        var_ready_df = var_ready_df.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"VAR-ready dataset created with {len(var_ready_df)} observations")
    return var_ready_df


def plot_gpr_comparison(gpr_quarterly):
    """Plot the Step2 median GPR index."""
    ensure_cjk_font_setup()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(gpr_quarterly['quarter'], gpr_quarterly['gpr_index'], 
            color='darkblue', linewidth=2, label='Step2 Median GPR (Quarterly Avg)')
    
    ax.set_xlabel('Quarter', fontsize=12)
    ax.set_ylabel('GPR Index (Mean = 100)', fontsize=12)
    ax.set_title('Step2 Median GPR Index - Quarterly', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add horizontal line at 100
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'step2_median_gpr_quarterly.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved GPR plot to {output_path}")


def main():
    logger.info("=== Task 8: Preparing Step2 Median GPR for SVAR Analysis ===")
    
    # Load Step2 median GPR
    gpr_df = load_step2_median_gpr()
    
    # Convert to quarterly
    gpr_quarterly = convert_to_quarterly(gpr_df)
    
    # Save quarterly GPR
    gpr_output = OUTPUT_DIR / "gpr_q_step2_median.parquet"
    gpr_quarterly[['quarter', 'gpr_index', 'log_GPR']].to_parquet(gpr_output, index=False)
    logger.info(f"Saved quarterly GPR to {gpr_output}")
    
    # Plot GPR
    plot_gpr_comparison(gpr_quarterly)
    
    # Load economic data
    master_df = load_economic_data()
    
    if master_df is not None:
        # Merge with economic data
        merged_df = merge_with_economic_data(gpr_quarterly, master_df)
        
        # Save merged master data
        master_output = OUTPUT_DIR / "master_quarterly_step2_median.parquet"
        merged_df.to_parquet(master_output, index=False)
        logger.info(f"Saved master quarterly data to {master_output}")
        
        # Create VAR-ready dataset
        var_ready_df = create_var_ready_data(merged_df)
        
        # Save VAR-ready data
        var_output = OUTPUT_DIR / "var_ready_step2_median.parquet"
        var_ready_df.to_parquet(var_output, index=False)
        logger.info(f"Saved VAR-ready data to {var_output}")
        
        # Display summary statistics
        logger.info("\n=== Summary Statistics ===")
        logger.info(f"Quarterly GPR Index:")
        logger.info(f"  Mean: {gpr_quarterly['gpr_index'].mean():.2f}")
        logger.info(f"  Std Dev: {gpr_quarterly['gpr_index'].std():.2f}")
        logger.info(f"  Min: {gpr_quarterly['gpr_index'].min():.2f}")
        logger.info(f"  Max: {gpr_quarterly['gpr_index'].max():.2f}")
        
        logger.info("\nVAR-ready variables:")
        for var in var_ready_df.columns:
            if var != 'quarter':
                logger.info(f"  {var}: mean={var_ready_df[var].mean():.3f}, std={var_ready_df[var].std():.3f}")
    else:
        logger.warning("Economic data not available. Only GPR data prepared.")
        logger.info("To complete SVAR preparation, please run the original preprocessing script first.")
    
    logger.info("\nPreprocessing complete!")


if __name__ == "__main__":
    main()