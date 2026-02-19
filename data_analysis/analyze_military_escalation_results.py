#!/usr/bin/env python3
"""
Analyze the results of military escalation spike analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gpr_zero.utils.font_setup import ensure_cjk_font_setup

def analyze_military_escalation_results():
    """Analyze the military escalation spike analysis results"""
    
    # Setup Chinese fonts
    ensure_cjk_font_setup()
    
    # Load the results
    results_file = Path("data_analysis/output/military_escalation_analysis/military_escalation_spike_analysis.csv")
    df = pd.read_csv(results_file)
    
    print("=== Military Escalation Spike Analysis Results ===\n")
    
    # Filter for spikes that have military escalation data
    military_spikes = df[df['Overall Taiwan - Military escalation_count'] > 0].copy()
    
    print(f"Total spikes analyzed: {len(df)}")
    print(f"Spikes with military escalation content: {len(military_spikes)}")
    print(f"Percentage of spikes with military escalation: {len(military_spikes)/len(df)*100:.1f}%\n")
    
    if len(military_spikes) == 0:
        print("No spikes with military escalation content found.")
        return
    
    # Analyze newspaper distribution
    print("=== Newspaper Distribution in Military Escalation Spikes ===")
    china_spikes = military_spikes[military_spikes['中國時報_count'] > 0]
    udn_spikes = military_spikes[military_spikes['聯合報_count'] > 0]
    liberty_spikes = military_spikes[military_spikes['自由時報_count'] > 0]
    
    print(f"Spikes with 中國時報 military escalation: {len(china_spikes)}")
    print(f"Spikes with 聯合報 military escalation: {len(udn_spikes)}")
    print(f"Spikes with 自由時報 military escalation: {len(liberty_spikes)}")
    
    # Analyze total counts
    print(f"\nTotal military escalation words:")
    print(f"中國時報: {military_spikes['中國時報_count'].sum()}")
    print(f"聯合報: {military_spikes['聯合報_count'].sum()}")
    print(f"自由時報: {military_spikes['自由時報_count'].sum()}")
    
    # Analyze proportions
    print(f"\nAverage military escalation proportions:")
    china_avg_prop = military_spikes[military_spikes['中國時報_prop'] > 0]['中國時報_prop'].mean()
    udn_avg_prop = military_spikes[military_spikes['聯合報_prop'] > 0]['聯合報_prop'].mean()
    liberty_avg_prop = military_spikes[military_spikes['自由時報_prop'] > 0]['自由時報_prop'].mean()
    
    print(f"中國時報: {china_avg_prop:.6f}")
    print(f"聯合報: {udn_avg_prop:.6f}")
    print(f"自由時報: {liberty_avg_prop:.6f}")
    
    # Analyze ratios where both newspapers have data
    print("\n=== Ratio Analysis ===")
    
    # Liberty vs China ratios (where both have data)
    liberty_china_both = military_spikes[
        (military_spikes['中國時報_count'] > 0) & 
        (military_spikes['自由時報_count'] > 0)
    ]
    
    if len(liberty_china_both) > 0:
        print(f"\nSpikes with both 自由時報 and 中國時報 military escalation: {len(liberty_china_both)}")
        print(f"Average Liberty/China proportion ratio: {liberty_china_both['liberty_china_prop_ratio'].mean():.3f}")
        print(f"Average Liberty/China count ratio: {liberty_china_both['liberty_china_count_ratio'].mean():.3f}")
        
        # Check consistency
        prop_ratios = liberty_china_both['liberty_china_prop_ratio']
        count_ratios = liberty_china_both['liberty_china_count_ratio']
        
        print(f"Proportion ratio std dev: {prop_ratios.std():.3f}")
        print(f"Count ratio std dev: {count_ratios.std():.3f}")
        
        # Show some examples
        print(f"\nExample ratios (first 10 spikes with both newspapers):")
        for i, (idx, row) in enumerate(liberty_china_both.head(10).iterrows()):
            print(f"Spike {row['spike_date']}: Prop ratio = {row['liberty_china_prop_ratio']:.3f}, Count ratio = {row['liberty_china_count_ratio']:.3f}")
    
    # Time series analysis
    print("\n=== Time Series Patterns ===")
    military_spikes['year'] = pd.to_datetime(military_spikes['spike_date']).dt.year
    
    yearly_stats = military_spikes.groupby('year').agg({
        'Overall Taiwan - Military escalation_count': 'sum',
        '中國時報_count': 'sum',
        '聯合報_count': 'sum',
        '自由時報_count': 'sum'
    })
    
    print("Military escalation by year:")
    print(yearly_stats)
    
    # Major spikes
    print("\n=== Major Military Escalation Spikes ===")
    major_spikes = military_spikes.nlargest(10, 'Overall Taiwan - Military escalation_count')
    
    for _, spike in major_spikes.iterrows():
        print(f"Date: {spike['spike_date']}, Total: {spike['Overall Taiwan - Military escalation_count']}")
        print(f"  中國時報: {spike['中國時報_count']}, 聯合報: {spike['聯合報_count']}, 自由時報: {spike['自由時報_count']}")
        if spike['中國時報_count'] > 0 and spike['自由時報_count'] > 0:
            print(f"  Liberty/China ratio: {spike['liberty_china_count_ratio']:.3f}")
        print()

if __name__ == "__main__":
    analyze_military_escalation_results()