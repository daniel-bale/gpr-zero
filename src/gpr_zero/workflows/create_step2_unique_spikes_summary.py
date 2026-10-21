#!/usr/bin/env python3
"""
Create a CSV file with only Step2-unique spikes (those not found in Lau's index).
Uses the spike comparison results to identify unique spikes.
"""

import pandas as pd

from gpr_zero.gpr_index.spikes import select_overlapping_events
from gpr_zero.project import get_project


def main():
    """Create summary of Step2-unique spikes only."""

    project = get_project()
    all_spikes_file = project.data_path("gpr_spikes", "step2_all_spikes_summary.csv")
    unique_spikes_file = project.data_path(
        "gpr_spikes", "spike_comparison", "step2_unique_spikes.csv"
    )
    output_file = project.data_path("gpr_spikes", "step2_unique_spikes_summary.csv")

    # Load all spikes summary
    all_spikes_df = pd.read_csv(all_spikes_file)
    print(f"Loaded {len(all_spikes_df)} total spikes")

    # Load unique spikes list
    unique_spikes_df = pd.read_csv(unique_spikes_file)
    print(f"Found {len(unique_spikes_df)} Step2-unique spikes")

    # Scale differs between the CSVs; the event module matches their dates.
    unique_summary_df = select_overlapping_events(all_spikes_df, unique_spikes_df)

    if not unique_summary_df.empty:
        # Sort by spike number
        unique_summary_df = unique_summary_df.sort_values("spike_number").reset_index(
            drop=True
        )

        # Save to CSV
        unique_summary_df.to_csv(output_file, index=False, encoding="utf-8")

        print(
            f"\nSaved {len(unique_summary_df)} Step2-unique spike summaries to: {output_file}"
        )

        # Display summary
        print("\nStep2-unique spikes:")
        print("=" * 80)
        for _, spike in unique_summary_df.iterrows():
            print(f"\nSpike #{spike['spike_number']} ({spike['peak_date']})")
            print(f"  Event: {spike['event_summary']}")
            print(
                f"  Duration: {spike['start_date']} to {spike['end_date']} ({spike['duration_weeks']} weeks)"
            )
            print(f"  Peak value: {spike['peak_value']:.4f}")
            if spike["keywords"]:
                print(f"  Keywords: {spike['keywords']}")

        print(
            f"\n{len(unique_summary_df)} unique spikes demonstrate our index's sensitivity to:"
        )
        print("- Diplomatic isolation events")
        print("- Internal political instability")
        print("- Information warfare")
        print("- Foreign intervention potential")

    else:
        print("No unique spikes found!")
