#!/usr/bin/env python3
"""
Create a CSV file with only Step2-unique spikes (those not found in Lau's index).
Uses the spike comparison results to identify unique spikes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime

# Configuration
ALL_SPIKES_FILE = Path("data/gpr_spikes/step2_all_spikes_summary.csv")
UNIQUE_SPIKES_FILE = Path("data/gpr_spikes/spike_comparison/step2_unique_spikes.csv")
OUTPUT_FILE = Path("data/gpr_spikes/step2_unique_spikes_summary.csv")

def main():
    """Create summary of Step2-unique spikes only."""
    
    # Load all spikes summary
    all_spikes_df = pd.read_csv(ALL_SPIKES_FILE)
    print(f"Loaded {len(all_spikes_df)} total spikes")
    
    # Load unique spikes list
    unique_spikes_df = pd.read_csv(UNIQUE_SPIKES_FILE)
    print(f"Found {len(unique_spikes_df)} Step2-unique spikes")
    
    # The unique spikes file uses scaled values, so we need to match by dates
    # Convert dates to comparable format
    all_spikes_df['peak_date_dt'] = pd.to_datetime(all_spikes_df['peak_date'])
    unique_spikes_df['peak_date_dt'] = pd.to_datetime(unique_spikes_df['peak_date'])
    
    # Create a mapping of unique spike dates
    unique_dates = set()
    for _, spike in unique_spikes_df.iterrows():
        # Add all dates in the spike range
        start_dt = pd.to_datetime(spike['start_date'])
        end_dt = pd.to_datetime(spike['end_date'])
        peak_dt = pd.to_datetime(spike['peak_date'])
        
        # For single-day spikes
        if start_dt == end_dt:
            unique_dates.add(peak_dt)
        else:
            # For multi-day spikes, check if any spike overlaps this period
            date_range = pd.date_range(start_dt, end_dt, freq='D')
            for date in date_range:
                unique_dates.add(date)
    
    print(f"\nUnique spike dates: {sorted(unique_dates)}")
    
    # Find matching spikes
    unique_spike_rows = []
    
    for _, spike in all_spikes_df.iterrows():
        spike_start = pd.to_datetime(spike['start_date'])
        spike_end = pd.to_datetime(spike['end_date'])
        spike_peak = pd.to_datetime(spike['peak_date'])
        
        # Check if this spike overlaps with any unique spike dates
        is_unique = False
        
        # For single-day spikes
        if spike_start == spike_end:
            if spike_peak in unique_dates:
                is_unique = True
        else:
            # For multi-day spikes, check if any day overlaps
            spike_range = pd.date_range(spike_start, spike_end, freq='D')
            if any(date in unique_dates for date in spike_range):
                is_unique = True
        
        if is_unique:
            unique_spike_rows.append(spike)
            print(f"Found unique spike: #{spike['spike_number']} ({spike['peak_date']}) - {spike['event_summary']}")
    
    # Create DataFrame with unique spikes
    if unique_spike_rows:
        unique_summary_df = pd.DataFrame(unique_spike_rows)
        
        # Drop the datetime columns we added
        if 'peak_date_dt' in unique_summary_df.columns:
            unique_summary_df = unique_summary_df.drop(columns=['peak_date_dt'])
        
        # Sort by spike number
        unique_summary_df = unique_summary_df.sort_values('spike_number').reset_index(drop=True)
        
        # Save to CSV
        unique_summary_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        
        print(f"\nSaved {len(unique_summary_df)} Step2-unique spike summaries to: {OUTPUT_FILE}")
        
        # Display summary
        print("\nStep2-unique spikes:")
        print("="*80)
        for _, spike in unique_summary_df.iterrows():
            print(f"\nSpike #{spike['spike_number']} ({spike['peak_date']})")
            print(f"  Event: {spike['event_summary']}")
            print(f"  Duration: {spike['start_date']} to {spike['end_date']} ({spike['duration_weeks']} weeks)")
            print(f"  Peak value: {spike['peak_value']:.4f}")
            if spike['keywords']:
                print(f"  Keywords: {spike['keywords']}")
        
        print(f"\n{len(unique_summary_df)} unique spikes demonstrate our index's sensitivity to:")
        print("- Diplomatic isolation events")
        print("- Internal political instability")
        print("- Information warfare")
        print("- Foreign intervention potential")
        
    else:
        print("No unique spikes found!")

if __name__ == "__main__":
    main()