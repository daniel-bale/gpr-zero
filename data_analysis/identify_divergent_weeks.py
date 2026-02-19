#!/usr/bin/env python3
"""
Identify weeks where Liberty Times and China Times show divergent coverage patterns.

This script analyzes weekly GPR shares to find periods where the two newspapers
exhibit contrasting editorial emphasis using relative differences and period expansion.
"""

import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json


def combine_diplomatic_crisis_categories(df):
    """Combine 'Diplomatic Crisis' and 'Diplomatic Crisis manual' into single category"""
    # Check if both columns exist
    if "Diplomatic Crisis_share" in df.columns and "Diplomatic Crisis manual_share" in df.columns:
        df = df.with_columns(
            (pl.col("Diplomatic Crisis_share") + pl.col("Diplomatic Crisis manual_share")).alias("Diplomatic Crisis_share")
        )
        # Drop the manual column
        df = df.drop("Diplomatic Crisis manual_share")
    
    # Do the same for counts if they exist
    if "Diplomatic Crisis_count" in df.columns and "Diplomatic Crisis manual_count" in df.columns:
        df = df.with_columns(
            (pl.col("Diplomatic Crisis_count") + pl.col("Diplomatic Crisis manual_count")).alias("Diplomatic Crisis_count")
        )
        df = df.drop("Diplomatic Crisis manual_count")
    
    return df


def expand_period_dates(start_date, end_date, expansion_weeks=4):
    """Expand period by specified number of weeks on each side"""
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Expand by expansion_weeks on each side
    new_start = start_dt - timedelta(weeks=expansion_weeks)
    new_end = end_dt + timedelta(weeks=expansion_weeks)
    
    return new_start.strftime('%Y-%m-%d'), new_end.strftime('%Y-%m-%d')


def find_divergent_week_in_period(df, category, period_name, start_date, end_date, divergence_type, min_threshold=0.001, max_expansions=3):
    """
    Find the week with largest relative divergence in a period, expanding if necessary
    divergence_type: 'liberty_higher' or 'china_higher'
    """
    original_start, original_end = start_date, end_date
    
    for expansion in range(max_expansions + 1):
        if expansion > 0:
            start_date, end_date = expand_period_dates(original_start, original_end, expansion * 2)
            print(f"  Expansion {expansion}: {start_date} to {end_date}")
        
        # Filter data for this period
        period_df = df.filter(
            (pl.col("week") >= start_date) & 
            (pl.col("week") <= end_date)
        )
        
        if period_df.height == 0:
            print(f"  No data found for period {period_name} ({start_date} to {end_date})")
            continue
            
        # Pivot to compare newspapers
        try:
            pivot_df = period_df.pivot(
                values=f"{category}_share",
                index="week",
                columns="newspaper"
            )
        except Exception as e:
            print(f"  Error pivoting data for {period_name}: {e}")
            continue
        
        # Check if both newspapers have data
        if "自由時報" not in pivot_df.columns or "中國時報" not in pivot_df.columns:
            print(f"  Missing newspaper data for {period_name}")
            continue
        
        # Calculate relative difference (percentage difference)
        if divergence_type == "liberty_higher":
            # Relative difference: (Liberty - China) / China * 100
            pivot_df = pivot_df.with_columns(
                ((pl.col("自由時報") - pl.col("中國時報")) / (pl.col("中國時報") + 1e-10) * 100).alias("relative_difference")
            ).with_columns(
                (pl.col("自由時報") - pl.col("中國時報")).alias("absolute_difference")
            )
        else:  # china_higher
            # Relative difference: (China - Liberty) / Liberty * 100
            pivot_df = pivot_df.with_columns(
                ((pl.col("中國時報") - pl.col("自由時報")) / (pl.col("自由時報") + 1e-10) * 100).alias("relative_difference")
            ).with_columns(
                (pl.col("中國時報") - pl.col("自由時報")).alias("absolute_difference")
            )
        
        # Filter for positive differences (both relative and absolute must be positive)
        positive_diff = pivot_df.filter(
            (pl.col("relative_difference") > 0) & 
            (pl.col("absolute_difference") > min_threshold)
        )
        
        if positive_diff.height == 0:
            print(f"  No positive divergence found for {period_name} (expansion {expansion})")
            continue
        
        # Find week with largest relative difference
        best_week = positive_diff.sort("relative_difference", descending=True).head(1)
        
        if best_week.height > 0:
            week_data = best_week.row(0, named=True)
            print(f"  Found divergent week: {week_data['week']} (relative diff: {week_data['relative_difference']:.2f}%, absolute diff: {week_data['absolute_difference']:.4f})")
            
            return {
                "period": period_name,
                "category": category,
                "divergence_type": divergence_type,
                "week": week_data["week"],
                "liberty_times_share": week_data["自由時報"],
                "china_times_share": week_data["中國時報"],
                "relative_difference": week_data["relative_difference"],
                "absolute_difference": week_data["absolute_difference"],
                "expansion_level": expansion,
                "period_used": f"{start_date} to {end_date}"
            }
    
    print(f"  No divergent week found for {period_name} after {max_expansions} expansions")
    return None


def main():
    """Main function to identify divergent weeks"""
    
    # Load data
    data_path = "data_analysis/output/weekly_gpr_analysis/weekly_analysis_20250722_041207/weekly_gpr_shares.csv"
    print(f"Loading data from: {data_path}")
    
    try:
        shares_df = pl.read_csv(data_path)
        print(f"Loaded {shares_df.height} rows of data")
        print(f"Columns: {shares_df.columns}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Combine diplomatic crisis categories
    shares_df = combine_diplomatic_crisis_categories(shares_df)
    
    # Filter for target newspapers
    newspapers = ["自由時報", "中國時報"]
    df_filtered = shares_df.filter(pl.col("newspaper").is_in(newspapers))
    print(f"Filtered to {df_filtered.height} rows for target newspapers")
    
    # Define target periods
    diplomatic_periods = {
        "mid_2020": ("2020-06-01", "2020-08-31"),
        "early_2021": ("2021-01-01", "2021-03-31"),
        "early_2022": ("2022-01-01", "2022-03-31"),
        "early_2023": ("2023-01-01", "2023-03-31")
    }
    
    military_china_periods = {
        "mid_2018": ("2018-06-01", "2018-08-31"),
        "early_2019": ("2019-01-01", "2019-03-31"),
        "late_2020": ("2020-10-01", "2020-12-31"),
        "early_2021": ("2021-01-01", "2021-03-31"),
        "early_2022": ("2022-01-01", "2022-03-31"),
        "mid_2023": ("2023-06-01", "2023-08-31")
    }
    
    military_liberty_periods = {
        "early_2020": ("2020-01-01", "2020-03-31"),
        "late_2021": ("2021-10-01", "2021-12-31"),
        "late_2023": ("2023-09-01", "2023-11-30")
    }
    
    all_results = []
    
    # Analyze Diplomatic Crisis periods (Liberty Times > China Times) - Target: 4 weeks
    print("\n=== Analyzing Diplomatic Crisis periods (Liberty Times > China Times) ===")
    for period_name, (start_date, end_date) in diplomatic_periods.items():
        print(f"\nAnalyzing {period_name} ({start_date} to {end_date})...")
        result = find_divergent_week_in_period(
            df_filtered, "Diplomatic Crisis", period_name, start_date, end_date, "liberty_higher"
        )
        if result:
            all_results.append(result)
    
    # Analyze Military Escalation periods (China Times > Liberty Times) - Target: 6 weeks
    print("\n=== Analyzing Military Escalation periods (China Times > Liberty Times) ===")
    for period_name, (start_date, end_date) in military_china_periods.items():
        print(f"\nAnalyzing {period_name} ({start_date} to {end_date})...")
        result = find_divergent_week_in_period(
            df_filtered, "Military Escalation", period_name, start_date, end_date, "china_higher"
        )
        if result:
            all_results.append(result)
    
    # Analyze Military Escalation periods (Liberty Times > China Times) - Target: 3 weeks
    print("\n=== Analyzing Military Escalation periods (Liberty Times > China Times) ===")
    for period_name, (start_date, end_date) in military_liberty_periods.items():
        print(f"\nAnalyzing {period_name} ({start_date} to {end_date})...")
        result = find_divergent_week_in_period(
            df_filtered, "Military Escalation", period_name, start_date, end_date, "liberty_higher"
        )
        if result:
            all_results.append(result)
    
    # Create output directory
    output_dir = Path("data_analysis/output/divergent_coverage")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    json_output = output_dir / "divergent_weeks_detailed.json"
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nDetailed results saved to: {json_output}")
    
    # Create CSV summary
    if all_results:
        csv_df = pl.DataFrame(all_results)
        csv_output = output_dir / "divergent_weeks_summary.csv"
        csv_df.write_csv(csv_output)
        print(f"CSV summary saved to: {csv_output}")
        
        # Print summary statistics
        print(f"\nSummary: Found {len(all_results)} divergent weeks")
        
        # Show results sorted by relative difference
        print("\nResults sorted by relative difference:")
        sorted_results = csv_df.sort("relative_difference", descending=True)
        print(sorted_results)
        
        # Count by divergence type
        liberty_higher = csv_df.filter(pl.col("divergence_type") == "liberty_higher").height
        china_higher = csv_df.filter(pl.col("divergence_type") == "china_higher").height
        print(f"\nDivergence type counts:")
        print(f"Liberty Times higher: {liberty_higher}")
        print(f"China Times higher: {china_higher}")
        print(f"Total: {liberty_higher + china_higher}")
        
        # Show expansion statistics
        expansion_stats = csv_df.group_by("expansion_level").agg(pl.count().alias("count"))
        print(f"\nExpansion level statistics:")
        print(expansion_stats.sort("expansion_level"))
        
    else:
        print("No divergent weeks found!")


if __name__ == "__main__":
    main()