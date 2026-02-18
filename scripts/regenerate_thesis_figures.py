#!/usr/bin/env python3
"""
Regenerate all thesis figures after removing titles and labels.
This script runs all the modified scripts to generate clean figures.
"""

import subprocess
import sys
from pathlib import Path

# List of scripts to run in order
SCRIPTS_TO_RUN = [
    # 1. median_timeseries_with_bands.png
    ("scripts/create_enhanced_correlation_plots.py", "median_timeseries_with_bands.png"),
    
    # 2. gpr_scaled_mean100.png (already clean, but regenerate)
    ("scripts/create_gpr_median_plots.py", "gpr_scaled_mean100.png"),
    
    # 3. step2_category_contributions_timeseries.png
    ("scripts/calculate_step2_category_contributions.py", "step2_category_contributions_timeseries.png"),
    
    # 4. step2_spikes_top50_words.png
    ("scripts/analyze_word_level_contributions.py", "step2_spikes_top50_words.png"),
    
    # 5. category_dynamics_stacked_area.png and spike_vs_normal_comparison.png
    ("scripts/visualize_category_dynamics.py", "category_dynamics_stacked_area.png, spike_vs_normal_comparison.png"),
]

def run_script(script_path, output_desc):
    """Run a single script and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print(f"Generates: {output_desc}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print("SUCCESS ✓")
        if result.stdout:
            print("\nOutput:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR ✗: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def main():
    """Run all scripts to regenerate thesis figures."""
    print("Regenerating Thesis Figures Without Titles/Labels")
    print("="*60)
    
    success_count = 0
    failed_scripts = []
    
    for script_path, output_desc in SCRIPTS_TO_RUN:
        if run_script(script_path, output_desc):
            success_count += 1
        else:
            failed_scripts.append(script_path)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total scripts: {len(SCRIPTS_TO_RUN)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_scripts)}")
    
    if failed_scripts:
        print("\nFailed scripts:")
        for script in failed_scripts:
            print(f"  - {script}")
    else:
        print("\nAll figures regenerated successfully! ✓")
    
    print("\nNext steps:")
    print("1. Check the generated figures in their respective output directories")
    print("2. Copy clean figures to /home5/chuenho/thesis/figures/")
    print("3. Update FIGURE_MAPPING.md with the status")

if __name__ == "__main__":
    main()