#!/usr/bin/env python3
"""
Combine timeseries data with expanded Step2 dictionaries (now 40 instead of 20).
This creates the updated combined dataset for correlation analysis.
"""

import pandas as pd
import json
from pathlib import Path
import glob

def load_and_combine_timeseries():
    """Load and combine all available timeseries data."""
    
    # Path to timeseries data
    timeseries_dir = Path("/home5/chuenho/gpr-zero/data/gpr_timeseries")
    
    # Find all timeseries files
    files = list(timeseries_dir.glob("*_W_overall.parquet"))
    print(f"Found {len(files)} timeseries files")
    
    # Also check for new Step2 files
    step2_new = [f for f in files if 'step2' in f.name and any(f'run{i}' in f.name for i in range(6, 11))]
    print(f"Found {len(step2_new)} new Step2 files (runs 6-10)")
    
    # Debug: List all Step2 files
    step2_all = [f for f in files if 'step2' in f.name]
    print(f"\nAll Step2 files found ({len(step2_all)} total):")
    for f in sorted(step2_all)[:5]:
        print(f"  {f.name}")
    if len(step2_all) > 5:
        print(f"  ... and {len(step2_all) - 5} more")
    
    # Load and combine
    all_series = {}
    metadata_dict = {}
    
    for file_path in files:
        dict_name = file_path.stem.replace("_W_overall", "")
        
        try:
            # Load parquet file
            import polars as pl
            df = pl.read_parquet(file_path)
            df_pd = df.to_pandas()
            
            # Set date as index
            if 'warc_date_bucketed' in df_pd.columns:
                df_pd['warc_date_bucketed'] = pd.to_datetime(df_pd['warc_date_bucketed'])
                df_pd = df_pd.set_index('warc_date_bucketed')
                
                # Store total GPR share
                if 'total_gpr_share' in df_pd.columns:
                    all_series[dict_name] = df_pd['total_gpr_share']
                    
                    # Create metadata
                    metadata_dict[dict_name] = parse_dictionary_metadata(dict_name)
                    
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue
    
    print(f"Successfully loaded {len(all_series)} timeseries")
    
    # Combine into DataFrame
    combined_df = pd.DataFrame(all_series)
    combined_df = combined_df.fillna(0)
    
    # Remove rows where all values are 0
    combined_df = combined_df.loc[(combined_df != 0).any(axis=1)]
    
    return combined_df, metadata_dict

def parse_dictionary_metadata(dict_name):
    """Parse dictionary name to extract metadata."""
    metadata = {
        'name': dict_name,
        'step': None,
        'model': None,
        'role': None,
        'run': None
    }
    
    # Extract step (step1 or step2)
    if dict_name.startswith('step1_'):
        metadata['step'] = 'step1'
        remainder = dict_name[6:]
    elif dict_name.startswith('step2_'):
        metadata['step'] = 'step2'
        remainder = dict_name[6:]
    else:
        remainder = dict_name
    
    # Extract model and role
    parts = remainder.split('_')
    if len(parts) >= 2:
        metadata['model'] = parts[0]
        
        # Find role and run
        for i, part in enumerate(parts[1:], 1):
            if part.startswith('run'):
                metadata['run'] = part
                metadata['role'] = '_'.join(parts[1:i])
                break
        else:
            metadata['role'] = '_'.join(parts[1:])
    
    return metadata

def main():
    """Main function."""
    
    # Output directory
    output_dir = Path("/home5/chuenho/gpr-zero/data/gpr_correlations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading and combining timeseries data...")
    combined_df, metadata_dict = load_and_combine_timeseries()
    
    # Count by step
    step_counts = {}
    model_counts = {}
    
    for name, meta in metadata_dict.items():
        if meta['step']:
            step_counts[meta['step']] = step_counts.get(meta['step'], 0) + 1
        if meta['model']:
            model_counts[meta['model']] = model_counts.get(meta['model'], 0) + 1
    
    print(f"\nDictionary counts by step:")
    for step, count in step_counts.items():
        print(f"  {step}: {count}")
    
    print(f"\nDictionary counts by model:")  
    for model, count in model_counts.items():
        print(f"  {model}: {count}")
    
    # Save updated combined timeseries
    combined_output = output_dir / "combined_gpr_timeseries_step2_expanded.csv"
    combined_df.to_csv(combined_output)
    print(f"\nSaved: {combined_output}")
    
    # Save updated metadata
    metadata_output = output_dir / "dictionary_metadata_step2_expanded.json"
    with open(metadata_output, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"Saved: {metadata_output}")
    
    print(f"\nTotal dictionaries: {len(metadata_dict)}")
    print(f"Time periods: {len(combined_df)}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

if __name__ == "__main__":
    main()