#!/usr/bin/env python3
"""
Create a consolidated CSV file with all 19 spike summaries.
Combines information from spike detection and LLM summaries.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pandas as pd
from datetime import datetime

# Configuration
SPIKE_DETECTION_FILE = Path("data/gpr_spikes/unified_detection_3outlets/step2_median_spikes.csv")
SUMMARIES_DIR = Path("data/gpr_spikes/spike_summaries_llm")
OUTPUT_FILE = Path("data/gpr_spikes/step2_all_spikes_summary.csv")

def load_llm_summary(spike_num: int, start_date: str, end_date: str) -> dict:
    """Try to load LLM summary for a spike."""
    # Try different file naming patterns
    patterns = [
        f"spike_{spike_num:02d}_{start_date}_{end_date}_llm_summary.json",
        f"spike_{spike_num}_llm_summary.json",
        f"spike_{start_date.replace('-', '')}_{end_date.replace('-', '')}_llm_summary.json"
    ]
    
    for pattern in patterns:
        file_path = SUMMARIES_DIR / pattern
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
                return data
    
    return None

def main():
    """Create consolidated spike summaries CSV."""
    # Load spike detection results
    spikes_df = pd.read_csv(SPIKE_DETECTION_FILE)
    spikes_df['year'] = pd.to_datetime(spikes_df['start_date']).dt.year
    valid_spikes_df = spikes_df[spikes_df['year'] < 2025].copy()
    
    print(f"Processing {len(valid_spikes_df)} spikes (excluding 2025)...")
    
    # Process each spike
    summaries = []
    
    for idx, spike in valid_spikes_df.iterrows():
        spike_num = idx + 1
        start_date = spike['start_date'].split('T')[0]
        end_date = spike['end_date'].split('T')[0]
        peak_date = spike['peak_date'].split('T')[0]
        
        # Try to load LLM summary
        llm_data = load_llm_summary(spike_num, start_date, end_date)
        
        # Extract LLM results if available
        if llm_data and llm_data.get('llm_short_answer') not in ['LLM: Mock Response', 'LLM: API Error', 'LLM: JSON parsing error', 'N/A', None]:
            event_summary = llm_data['llm_short_answer']
            keywords = llm_data.get('llm_keywords', '')
            has_llm = True
        else:
            # Use placeholder or manual summary
            event_summary = f"[Spike {spike_num}: Pending analysis]"
            keywords = ''
            has_llm = False
            
            # Manual summaries for critical spikes
            if spike_num == 8:
                event_summary = "Pelosi's Taiwan visit"
                keywords = "裴洛西訪台, 共軍圍台軍演, 台海緊張, 美中角力"
            elif spike_num == 16:
                event_summary = "Lai inauguration & military drills"
                keywords = "賴清德就職, 中國軍演, 台海局勢"
            elif spike_num == 17:
                event_summary = "AUKUS submarine & regional tensions"
                keywords = "AUKUS潛艦, 區域安全, 印太戰略"
            elif spike_num == 19:
                event_summary = "Cross-strait tensions & military activities"
                keywords = "兩岸關係, 軍事活動, 台海情勢"
        
        # Create record
        summary_record = {
            'spike_number': spike_num,
            'start_date': start_date,
            'end_date': end_date,
            'peak_date': peak_date,
            'duration_weeks': spike['duration_weeks'],
            'peak_value': spike['peak_value'],
            'mean_spike_value': spike['mean_spike_value'],
            'threshold': spike['threshold'],
            'event_summary': event_summary,
            'keywords': keywords,
            'has_llm_summary': has_llm
        }
        
        summaries.append(summary_record)
    
    # Create DataFrame
    summaries_df = pd.DataFrame(summaries)
    
    # Save to CSV
    summaries_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    # Print summary
    print(f"\nSaved {len(summaries_df)} spike summaries to: {OUTPUT_FILE}")
    print("\nSummary statistics:")
    print(f"- Total spikes: {len(summaries_df)}")
    print(f"- Spikes with LLM summaries: {summaries_df['has_llm_summary'].sum()}")
    print(f"- Date range: {summaries_df['start_date'].min()} to {summaries_df['end_date'].max()}")
    print(f"- Highest spike: #{summaries_df.loc[summaries_df['peak_value'].idxmax(), 'spike_number']} "
          f"({summaries_df['peak_value'].max():.4f}) - "
          f"{summaries_df.loc[summaries_df['peak_value'].idxmax(), 'event_summary']}")
    
    # Display first few rows
    print("\nFirst 5 spikes:")
    print(summaries_df[['spike_number', 'peak_date', 'event_summary', 'peak_value']].head())

if __name__ == "__main__":
    main()