#!/usr/bin/env python3
"""
Visualize the identified divergent weeks on the original time series plots.

This script creates plots showing the GPR share time series for both newspapers
and highlights the identified divergent weeks to confirm they are correctly labeled.
"""

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
from pathlib import Path
import json

# Import font setup
import sys
sys.path.append('src')
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

def load_divergent_weeks():
    """Load the identified divergent weeks"""
    with open("data_analysis/output/divergent_coverage/divergent_weeks_detailed.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_category_with_divergent_weeks(df, category, divergent_weeks, output_dir):
    """Plot time series for a category with divergent weeks highlighted"""
    
    # Filter data for the category
    category_data = df.filter(
        (pl.col("newspaper").is_in(["自由時報", "中國時報"])) &
        (pl.col(f"{category}_share").is_not_null())
    ).sort("week")
    
    # Convert to pandas for easier plotting
    category_pd = category_data.to_pandas()
    category_pd['week'] = pd.to_datetime(category_pd['week'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot time series for each newspaper
    for newspaper in ["自由時報", "中國時報"]:
        newspaper_data = category_pd[category_pd['newspaper'] == newspaper]
        ax.plot(newspaper_data['week'], newspaper_data[f'{category}_share'], 
                label=newspaper, linewidth=1.5, alpha=0.8)
    
    # Highlight divergent weeks
    colors = {'liberty_higher': 'red', 'china_higher': 'blue'}
    markers = {'liberty_higher': '^', 'china_higher': 'v'}
    
    for result in divergent_weeks:
        if result['category'] == category:
            divergence_type = result['divergence_type']
            color = colors[divergence_type]
            marker = markers[divergence_type]
            
            # Handle single week format (new data structure)
            week_str = result['week']
            week_date = pd.to_datetime(week_str)
            
            # Get the actual values for both newspapers at this week
            week_data = category_pd[category_pd['week'] == week_date]
            
            if not week_data.empty:
                liberty_val = week_data[week_data['newspaper'] == '自由時報'][f'{category}_share'].iloc[0] if len(week_data[week_data['newspaper'] == '自由時報']) > 0 else 0
                china_val = week_data[week_data['newspaper'] == '中國時報'][f'{category}_share'].iloc[0] if len(week_data[week_data['newspaper'] == '中國時報']) > 0 else 0
                
                # Plot markers for both newspapers
                ax.scatter(week_date, liberty_val, color=color, marker=marker, s=100,
                          edgecolors='black', linewidth=1, zorder=5)
                ax.scatter(week_date, china_val, color=color, marker=marker, s=100,
                          edgecolors='black', linewidth=1, zorder=5, alpha=0.6)
                
                # Add text annotation
                max_val = max(liberty_val, china_val)
                ax.annotate(f'{week_str}\n({result["period"]})',
                           xy=(week_date, max_val),
                           xytext=(10, 20), textcoords='offset points',
                           fontsize=8, ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Formatting
    ax.set_title(f'{category} - GPR Share Time Series with Divergent Weeks', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('GPR Share', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 7]))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{category.replace(' ', '_')}_divergent_weeks.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_file}")

def create_summary_plot(df, divergent_weeks, output_dir):
    """Create a summary plot showing all categories with divergent weeks"""
    
    categories = ["Military Escalation", "Diplomatic Crisis"]
    
    fig, axes = plt.subplots(len(categories), 1, figsize=(15, 6*len(categories)))
    if len(categories) == 1:
        axes = [axes]
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        
        # Filter data for the category
        category_data = df.filter(
            (pl.col("newspaper").is_in(["自由時報", "中國時報"])) &
            (pl.col(f"{category}_share").is_not_null())
        ).sort("week")
        
        # Convert to pandas for easier plotting
        category_pd = category_data.to_pandas()
        category_pd['week'] = pd.to_datetime(category_pd['week'])
        
        # Plot time series for each newspaper
        for newspaper in ["自由時報", "中國時報"]:
            newspaper_data = category_pd[category_pd['newspaper'] == newspaper]
            ax.plot(newspaper_data['week'], newspaper_data[f'{category}_share'], 
                    label=newspaper, linewidth=1.5, alpha=0.8)
        
        # Highlight divergent weeks
        colors = {'liberty_higher': 'red', 'china_higher': 'blue'}
        markers = {'liberty_higher': '^', 'china_higher': 'v'}
        
        for result in divergent_weeks:
            if result['category'] == category:
                divergence_type = result['divergence_type']
                color = colors[divergence_type]
                marker = markers[divergence_type]
                
                # Handle single week format (new data structure)
                week_str = result['week']
                week_date = pd.to_datetime(week_str)
                
                # Get the actual values for both newspapers at this week
                week_data = category_pd[category_pd['week'] == week_date]
                
                if not week_data.empty:
                    liberty_val = week_data[week_data['newspaper'] == '自由時報'][f'{category}_share'].iloc[0] if len(week_data[week_data['newspaper'] == '自由時報']) > 0 else 0
                    china_val = week_data[week_data['newspaper'] == '中國時報'][f'{category}_share'].iloc[0] if len(week_data[week_data['newspaper'] == '中國時報']) > 0 else 0
                    
                    # Plot markers for both newspapers
                    ax.scatter(week_date, liberty_val, color=color, marker=marker, s=80,
                              edgecolors='black', linewidth=1, zorder=5)
                    ax.scatter(week_date, china_val, color=color, marker=marker, s=80,
                              edgecolors='black', linewidth=1, zorder=5, alpha=0.6)
        
        # Formatting
        ax.set_title(f'{category} - GPR Share with Divergent Weeks', fontsize=12, fontweight='bold')
        ax.set_ylabel('GPR Share', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 7]))
    
    plt.xlabel('Date', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "summary_divergent_weeks.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot: {output_file}")

def main():
    """Main function to create visualization"""
    
    # Setup Chinese fonts
    ensure_cjk_font_setup()
    
    # Load data
    print("Loading weekly GPR shares data...")
    shares_df = pl.read_csv("data_analysis/output/weekly_gpr_analysis/weekly_analysis_20250722_041207/weekly_gpr_shares.csv")
    
    # Combine diplomatic crisis categories
    if "Diplomatic Crisis_share" in shares_df.columns and "Diplomatic Crisis manual_share" in shares_df.columns:
        shares_df = shares_df.with_columns(
            (pl.col("Diplomatic Crisis_share") + pl.col("Diplomatic Crisis manual_share")).alias("Diplomatic Crisis_share")
        )
        shares_df = shares_df.drop("Diplomatic Crisis manual_share")
    
    # Load divergent weeks
    print("Loading divergent weeks data...")
    divergent_weeks = load_divergent_weeks()
    
    # Create output directory
    output_dir = Path("data_analysis/output/divergent_coverage/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create individual plots for each category
    categories_with_divergent_weeks = set()
    for result in divergent_weeks:
        categories_with_divergent_weeks.add(result['category'])
    
    print(f"Creating plots for categories: {categories_with_divergent_weeks}")
    
    for category in categories_with_divergent_weeks:
        print(f"Creating plot for {category}...")
        plot_category_with_divergent_weeks(shares_df, category, divergent_weeks, output_dir)
    
    # Create summary plot
    print("Creating summary plot...")
    create_summary_plot(shares_df, divergent_weeks, output_dir)
    
    # Print summary statistics
    print("\n=== Divergent Weeks Summary ===")
    for result in divergent_weeks:
        print(f"\nPeriod: {result['period']}")
        print(f"Category: {result['category']}")
        print(f"Divergence Type: {result['divergence_type']}")
        print(f"Week: {result['week']}")
        print(f"Liberty Times Share: {result['liberty_times_share']:.4f}")
        print(f"China Times Share: {result['china_times_share']:.4f}")
        print(f"Relative Difference: {result['relative_difference']:.2f}%")

if __name__ == "__main__":
    main()