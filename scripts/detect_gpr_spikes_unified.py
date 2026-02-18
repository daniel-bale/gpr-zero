#!/usr/bin/env python3
"""
Unified spike detection script for both Step2 median and Lau's GPR indices.
Uses the same methodology as capture_gpr_spikes.py (mean + 1*std threshold).
Includes validation against existing Lau spike results.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl
from datetime import timedelta
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants from capture_gpr_spikes.py
SPIKE_STD_THRESHOLD = 1.0  # Number of standard deviations above mean
TIME_BUCKET_GPR = 'W'  # Weekly


def load_gpr_timeseries(file_path):
    """Load GPR timeseries data from parquet or CSV."""
    if str(file_path).endswith('.parquet'):
        df = pl.read_parquet(file_path)
    else:
        df = pl.read_csv(file_path, try_parse_dates=True)
    
    # Ensure we have a date column
    if 'warc_date_bucketed' in df.columns:
        df = df.rename({'warc_date_bucketed': 'date'})
    elif 'week' in df.columns:
        df = df.rename({'week': 'date'})
    elif 'date' not in df.columns:
        raise ValueError(f"No date column found in {file_path}")
    
    # Convert to pandas for easier manipulation
    return df.to_pandas()


def detect_spikes(df, value_column, name_prefix=""):
    """
    Detect spikes using mean + 1*std threshold.
    Groups consecutive weeks above threshold into spike events.
    """
    # Calculate statistics
    mean_val = df[value_column].mean()
    std_val = df[value_column].std()
    threshold = mean_val + (SPIKE_STD_THRESHOLD * std_val)
    
    logger.info(f"{name_prefix}Mean: {mean_val:.6f}, Std: {std_val:.6f}, Threshold: {threshold:.6f}")
    
    # Find candidate spikes
    df['is_spike'] = df[value_column] > threshold
    
    # Group consecutive spikes
    spike_events = []
    current_group = []
    
    for idx, row in df.iterrows():
        if row['is_spike']:
            current_group.append(row)
        else:
            if current_group:
                # Process the completed group
                group_df = pd.DataFrame(current_group)
                peak_idx = group_df[value_column].idxmax()
                peak_row = group_df.loc[peak_idx]
                
                spike_event = {
                    'start_date': group_df['date'].min(),
                    'end_date': group_df['date'].max(),
                    'peak_date': peak_row['date'],
                    'peak_value': peak_row[value_column],
                    'duration_weeks': len(group_df),
                    'mean_spike_value': group_df[value_column].mean()
                }
                spike_events.append(spike_event)
                current_group = []
    
    # Don't forget the last group if it ends at the last date
    if current_group:
        group_df = pd.DataFrame(current_group)
        peak_idx = group_df[value_column].idxmax()
        peak_row = group_df.loc[peak_idx]
        
        spike_event = {
            'start_date': group_df['date'].min(),
            'end_date': group_df['date'].max(),
            'peak_date': peak_row['date'],
            'peak_value': peak_row[value_column],
            'duration_weeks': len(group_df),
            'mean_spike_value': group_df[value_column].mean()
        }
        spike_events.append(spike_event)
    
    logger.info(f"{name_prefix}Detected {len(spike_events)} spike events")
    
    return spike_events, threshold


def compare_spike_events(new_spikes, old_spikes_df, tolerance_days=7):
    """
    Compare new spike events with old results.
    Allows for small date differences due to potential data updates.
    """
    comparison_results = []
    
    # Convert old spikes to list of dicts for easier comparison
    old_spikes = []
    for _, row in old_spikes_df.iterrows():
        old_spikes.append({
            'start_date': pd.to_datetime(row['start_date']),
            'end_date': pd.to_datetime(row['end_date']),
            'peak_date': pd.to_datetime(row['peak_date']),
            'peak_value': row['peak_gpr_share'] if 'peak_gpr_share' in row else None
        })
    
    # Match new spikes with old spikes
    matched_old = set()
    
    for new_spike in new_spikes:
        matched = False
        for i, old_spike in enumerate(old_spikes):
            if i in matched_old:
                continue
            
            # Check if peak dates are within tolerance
            peak_diff = abs((new_spike['peak_date'] - old_spike['peak_date']).days)
            if peak_diff <= tolerance_days:
                matched = True
                matched_old.add(i)
                
                comparison_results.append({
                    'status': 'MATCHED',
                    'new_peak_date': new_spike['peak_date'],
                    'old_peak_date': old_spike['peak_date'],
                    'peak_date_diff_days': peak_diff,
                    'new_peak_value': new_spike['peak_value'],
                    'old_peak_value': old_spike['peak_value'],
                    'new_duration': new_spike['duration_weeks'],
                    'old_duration': (old_spike['end_date'] - old_spike['start_date']).days // 7 + 1
                })
                break
        
        if not matched:
            comparison_results.append({
                'status': 'NEW_ONLY',
                'new_peak_date': new_spike['peak_date'],
                'new_peak_value': new_spike['peak_value'],
                'new_duration': new_spike['duration_weeks']
            })
    
    # Find old spikes not matched
    for i, old_spike in enumerate(old_spikes):
        if i not in matched_old:
            comparison_results.append({
                'status': 'OLD_ONLY',
                'old_peak_date': old_spike['peak_date'],
                'old_peak_value': old_spike['peak_value'],
                'old_duration': (old_spike['end_date'] - old_spike['start_date']).days // 7 + 1
            })
    
    return comparison_results


def main():
    """Main execution function."""
    
    logger.info("="*80)
    logger.info("TASK 2: Unified GPR Spike Detection")
    logger.info("="*80)
    
    project_root = Path(__file__).parent.parent
    
    # Load Step2 median GPR with 3-outlet filter
    logger.info("\n1. Loading Step2 median GPR index (3-outlet filtered)...")
    step2_file = project_root / "data" / "gpr_indices_3outlets" / "step2_median_gpr_3outlets.csv"
    step2_df = pd.read_csv(step2_file, parse_dates=['week'])
    step2_df.rename(columns={'week': 'date'}, inplace=True)
    
    # Use the raw gpr_index for spike detection
    step2_df['overall_share'] = step2_df['gpr_index']
    
    logger.info(f"Loaded Step2 data: {len(step2_df)} weeks")
    logger.info(f"Using 3-outlet filtered data from: {step2_file}")
    
    # Load Lau's GPR with 3-outlet filter
    logger.info("\n2. Loading Lau's GPR index (3-outlet filtered)...")
    lau_file = project_root / "data" / "gpr_timeseries_3outlets" / "lau_overall_taiwan_recalculated_W_overall_3outlets.parquet"
    lau_df = load_gpr_timeseries(lau_file)
    
    # For Lau data, use gpr_index as overall_share
    lau_df['overall_share'] = lau_df['gpr_index']
    
    logger.info(f"Loaded Lau's data: {len(lau_df)} weeks")
    
    # Detect spikes for Step2
    logger.info("\n3. Detecting spikes in Step2 median GPR...")
    step2_spikes, step2_threshold = detect_spikes(
        step2_df, 
        'overall_share',  # Use unscaled share
        name_prefix="Step2: "
    )
    
    # Detect spikes for Lau
    logger.info("\n4. Detecting spikes in Lau's GPR...")
    lau_spikes, lau_threshold = detect_spikes(
        lau_df, 
        'overall_share',
        name_prefix="Lau: "
    )
    
    # Load old Lau results for validation
    logger.info("\n5. Loading old Lau spike results for validation...")
    old_lau_file = project_root / "data_analysis" / "output" / "gpr_spikes" / "yeu_2024_20250715160933" / "detected_spike_events_with_llm_all_categories_2017_01_to_2024_12.csv"
    
    # Try to read with error handling for malformed CSV
    try:
        old_lau_df = pd.read_csv(old_lau_file, on_bad_lines='skip')
    except Exception as e:
        logger.warning(f"Failed to read CSV with skip, trying with error_bad_lines: {e}")
        try:
            # Try reading line by line to find the issue
            old_lau_df = pd.read_csv(old_lau_file, encoding='utf-8', quoting=1)  # QUOTE_MINIMAL
        except:
            # Last resort - read specific columns only
            old_lau_df = pd.read_csv(old_lau_file, 
                                    usecols=['gpr_category', 'newspaper', 'start_date', 
                                            'end_date', 'peak_date', 'peak_gpr_share'],
                                    on_bad_lines='skip')
    
    # Filter for overall category and Overall Taiwan
    # Based on exploration, the correct name is "Overall Taiwan - Overall"
    old_lau_overall = old_lau_df[
        (old_lau_df['gpr_category'] == 'overall') & 
        (old_lau_df['newspaper'] == 'Overall Taiwan - Overall')
    ].copy()
    
    logger.info(f"Found {len(old_lau_overall)} old Lau spike events for validation")
    
    if len(old_lau_overall) == 0:
        logger.error("No validation data found! Checking full dataframe structure...")
        logger.info(f"Unique categories: {old_lau_df['gpr_category'].unique()}")
        logger.info(f"Unique newspapers: {old_lau_df['newspaper'].unique()}")
        logger.info("First few rows of old data:")
        logger.info(old_lau_df[['gpr_category', 'newspaper']].head(10))
    
    # Compare results
    logger.info("\n6. Comparing new Lau results with old results...")
    comparison = compare_spike_events(lau_spikes, old_lau_overall)
    
    # Print comparison summary
    matched = [c for c in comparison if c['status'] == 'MATCHED']
    new_only = [c for c in comparison if c['status'] == 'NEW_ONLY']
    old_only = [c for c in comparison if c['status'] == 'OLD_ONLY']
    
    logger.info(f"\nValidation Results:")
    logger.info(f"- Matched spikes: {len(matched)}")
    logger.info(f"- New only: {len(new_only)}")
    logger.info(f"- Old only: {len(old_only)}")
    
    if len(matched) > 0:
        logger.info("\nMatched spikes (first 5):")
        for match in matched[:5]:
            logger.info(f"  Peak: {match['new_peak_date'].strftime('%Y-%m-%d')} "
                       f"(diff: {match['peak_date_diff_days']} days), "
                       f"Value: {match['new_peak_value']:.6f} vs {match['old_peak_value']:.6f}")
    
    # Save results
    output_dir = project_root / "data" / "gpr_spikes" / "unified_detection_3outlets"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Step2 spikes
    step2_spikes_df = pd.DataFrame(step2_spikes)
    step2_spikes_df['threshold'] = step2_threshold
    step2_spikes_df['index_type'] = 'Step2_median'
    step2_output = output_dir / "step2_median_spikes.csv"
    step2_spikes_df.to_csv(step2_output, index=False)
    logger.info(f"\nSaved Step2 spikes to {step2_output}")
    
    # Save Lau spikes
    lau_spikes_df = pd.DataFrame(lau_spikes)
    lau_spikes_df['threshold'] = lau_threshold
    lau_spikes_df['index_type'] = 'Lau_2024'
    lau_output = output_dir / "lau_2024_spikes.csv"
    lau_spikes_df.to_csv(lau_output, index=False)
    logger.info(f"Saved Lau spikes to {lau_output}")
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison)
    comparison_output = output_dir / "lau_validation_comparison.csv"
    comparison_df.to_csv(comparison_output, index=False)
    logger.info(f"Saved validation comparison to {comparison_output}")
    
    # Create summary report
    summary = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "step2_spikes": {
            "count": len(step2_spikes),
            "threshold": float(step2_threshold),
            "mean": float(step2_df['overall_share'].mean()),
            "std": float(step2_df['overall_share'].std())
        },
        "lau_spikes": {
            "count": len(lau_spikes),
            "threshold": float(lau_threshold),
            "mean": float(lau_df['overall_share'].mean()),
            "std": float(lau_df['overall_share'].std())
        },
        "validation": {
            "matched": len(matched),
            "new_only": len(new_only),
            "old_only": len(old_only),
            "match_rate": len(matched) / len(old_lau_overall) if len(old_lau_overall) > 0 else 0
        }
    }
    
    summary_file = output_dir / "spike_detection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSaved summary to {summary_file}")
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("SPIKE DETECTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Step2 Median: {len(step2_spikes)} spikes detected")
    logger.info(f"Lau 2024: {len(lau_spikes)} spikes detected")
    logger.info(f"Validation: {len(matched)}/{len(old_lau_overall)} matched ({summary['validation']['match_rate']:.1%})")
    
    if summary['validation']['match_rate'] < 0.8:
        logger.warning("\nWARNING: Low match rate with old results! Check for data or methodology differences.")
    else:
        logger.info("\n✓ Validation successful! New results match old results well.")
    
    logger.info("\n✓ Task 2 spike detection complete!")


if __name__ == "__main__":
    main()