#!/usr/bin/env python3
"""
Analyze Editorial Differences Between Newspapers During Divergent Coverage Periods

This script analyzes the LLM-generated summaries to identify patterns in editorial
emphasis between Liberty Times and China Times during geopolitical risk periods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict, Counter
import json
from datetime import datetime

# Import font setup for Chinese text
import sys
sys.path.append('src')
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

def load_divergent_summaries():
    """Load the LLM-generated summaries CSV."""
    csv_path = Path("data_analysis/output/divergent_coverage/article_samples/llm_summaries/divergent_coverage_summaries.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Summaries CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} summary records")
    print(f"Date range: {df['week_date'].min()} to {df['week_date'].max()}")
    print(f"Newspapers: {df['newspaper'].unique()}")
    
    return df

def extract_keywords_from_summaries(df):
    """Extract key themes and keywords from LLM summaries."""
    
    # Define keyword categories for analysis
    keyword_categories = {
        'us_support': [
            'US support', 'American support', 'US backing', 'US assistance',
            'US arms sales', 'US military', 'US intervention', 'US commitment',
            'Biden', 'Trump', 'Washington', 'Pentagon', 'State Department'
        ],
        'china_aggression': [
            'China threat', 'Chinese threat', 'PLA', 'military exercises',
            'Chinese aggression', 'Beijing threat', 'intimidation', 'provocation',
            'military pressure', 'coercion', 'Xi Jinping'
        ],
        'taiwan_sovereignty': [
            'Taiwan independence', 'sovereignty', 'democratic', 'freedom',
            'self-determination', 'Republic of China', 'Taiwan identity',
            'democratic values', 'Taiwan Strait', 'status quo'
        ],
        'economic_concerns': [
            'economic', 'trade', 'business', 'investment', 'sanctions',
            'ECFA', 'economic ties', 'supply chain', 'market access',
            'economic impact', 'financial'
        ],
        'cross_strait_dialogue': [
            'dialogue', 'peaceful', 'negotiation', 'communication',
            'cross-strait', 'exchange', 'cooperation', 'understanding',
            '1992 Consensus', 'One China'
        ],
        'military_conflict': [
            'military conflict', 'war', 'invasion', 'attack', 'defense',
            'military action', 'armed conflict', 'combat', 'battlefield',
            'military preparedness'
        ]
    }
    
    results = []
    
    for _, row in df.iterrows():
        summary = str(row['summary']).lower()
        
        keyword_counts = {}
        for category, keywords in keyword_categories.items():
            count = sum(summary.count(keyword.lower()) for keyword in keywords)
            keyword_counts[f'{category}_mentions'] = count
        
        # Calculate total words for normalization
        word_count = len(summary.split())
        keyword_counts['total_words'] = word_count
        
        # Add row data
        result = {
            'week_date': row['week_date'],
            'newspaper': row['newspaper'],
            'gpr_articles': row['gpr_articles'],
            'total_articles': row['total_articles'],
            **keyword_counts
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def analyze_editorial_emphasis(keyword_df):
    """Analyze editorial emphasis patterns between newspapers."""
    
    # Calculate normalized keyword frequencies (per 1000 words)
    keyword_cols = [col for col in keyword_df.columns if col.endswith('_mentions')]
    
    for col in keyword_cols:
        normalized_col = col.replace('_mentions', '_per_1000_words')
        keyword_df[normalized_col] = (keyword_df[col] / keyword_df['total_words']) * 1000
    
    # Group by newspaper and calculate means
    newspaper_emphasis = keyword_df.groupby('newspaper').agg({
        col: 'mean' for col in keyword_df.columns if col.endswith('_per_1000_words')
    }).round(2)
    
    print("\n=== Editorial Emphasis Analysis (mentions per 1000 words) ===")
    print(newspaper_emphasis)
    
    # Calculate differences between newspapers
    liberty_times = newspaper_emphasis.loc['自由時報']
    china_times = newspaper_emphasis.loc['中國時報']
    
    differences = liberty_times - china_times
    differences = differences.sort_values(ascending=False)
    
    print("\n=== Liberty Times vs China Times Differences ===")
    print("(Positive = Liberty Times emphasizes more, Negative = China Times emphasizes more)")
    for category, diff in differences.items():
        category_name = category.replace('_per_1000_words', '').replace('_', ' ').title()
        print(f"{category_name:25}: {diff:+6.2f}")
    
    return newspaper_emphasis, differences

def analyze_temporal_patterns(keyword_df):
    """Analyze how editorial emphasis changes over time."""
    
    # Convert week_date to datetime
    keyword_df['date'] = pd.to_datetime(keyword_df['week_date'])
    keyword_df['year'] = keyword_df['date'].dt.year
    
    # Calculate yearly averages
    yearly_patterns = keyword_df.groupby(['year', 'newspaper']).agg({
        col: 'mean' for col in keyword_df.columns if col.endswith('_per_1000_words')
    }).round(2)
    
    print("\n=== Temporal Patterns by Year ===")
    
    # Focus on key categories
    key_categories = [
        'us_support_per_1000_words',
        'china_aggression_per_1000_words', 
        'taiwan_sovereignty_per_1000_words',
        'cross_strait_dialogue_per_1000_words'
    ]
    
    for category in key_categories:
        print(f"\n{category.replace('_per_1000_words', '').replace('_', ' ').title()}:")
        category_data = yearly_patterns[category].unstack('newspaper')
        print(category_data)
    
    return yearly_patterns

def create_editorial_comparison_plots(newspaper_emphasis, differences, output_dir):
    """Create visualizations comparing editorial emphasis."""
    
    ensure_cjk_font_setup()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Editorial Emphasis Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Radar chart comparison
    categories = [col.replace('_per_1000_words', '').replace('_', '\n').title() 
                 for col in newspaper_emphasis.columns]
    
    # Bar chart comparison
    x_pos = np.arange(len(categories))
    width = 0.35
    
    liberty_values = newspaper_emphasis.loc['自由時報'].values
    china_values = newspaper_emphasis.loc['中國時報'].values
    
    ax1.bar(x_pos - width/2, liberty_values, width, label='自由時報 (Liberty Times)', 
            color='#2E86AB', alpha=0.8)
    ax1.bar(x_pos + width/2, china_values, width, label='中國時報 (China Times)', 
            color='#A23B72', alpha=0.8)
    
    ax1.set_xlabel('Editorial Themes')
    ax1.set_ylabel('Mentions per 1000 Words')
    ax1.set_title('Editorial Emphasis Comparison\nBetween Newspapers During GPR Periods')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference analysis
    diff_categories = [col.replace('_per_1000_words', '').replace('_', '\n').title() 
                      for col in differences.index]
    diff_values = differences.values
    
    colors = ['#2E86AB' if x > 0 else '#A23B72' for x in diff_values]
    bars = ax2.barh(diff_categories, diff_values, color=colors, alpha=0.8)
    
    ax2.set_xlabel('Difference (Liberty Times - China Times)')
    ax2.set_title('Editorial Emphasis Differences\n(Positive = Liberty Times emphasizes more)')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, diff_values):
        ax2.text(value + (0.1 if value > 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                f'{value:+.1f}', ha='left' if value > 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'editorial_emphasis_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved editorial comparison plot: {output_dir / 'editorial_emphasis_comparison.png'}")

def analyze_category_specific_patterns(df, keyword_df):
    """Analyze patterns specific to different GPR categories."""
    
    # Get category columns
    category_cols = [col for col in df.columns if col.startswith('count_')]
    
    print("\n=== Category-Specific Editorial Patterns ===")
    
    # For each category, analyze editorial emphasis
    for category_col in category_cols:
        category_name = category_col.replace('count_', '').replace('_', ' ')
        
        # Filter to weeks where this category has articles
        category_weeks = df[df[category_col] > 0].copy()
        
        if len(category_weeks) == 0:
            continue
            
        print(f"\n{category_name.title()} Weeks:")
        print(f"  Total weeks with articles: {len(category_weeks)}")
        
        # Merge with keyword analysis
        category_analysis = category_weeks.merge(
            keyword_df, on=['week_date', 'newspaper'], how='left'
        )
        
        # Calculate average emphasis by newspaper for this category
        emphasis_by_newspaper = category_analysis.groupby('newspaper').agg({
            'us_support_per_1000_words': 'mean',
            'china_aggression_per_1000_words': 'mean',
            'taiwan_sovereignty_per_1000_words': 'mean',
            'cross_strait_dialogue_per_1000_words': 'mean'
        }).round(2)
        
        print(f"  Editorial emphasis by newspaper:")
        for newspaper in emphasis_by_newspaper.index:
            print(f"    {newspaper}:")
            for theme, value in emphasis_by_newspaper.loc[newspaper].items():
                theme_name = theme.replace('_per_1000_words', '').replace('_', ' ').title()
                print(f"      {theme_name:20}: {value:5.1f}")

def generate_summary_report(df, keyword_df, differences, output_dir):
    """Generate a comprehensive summary report."""
    
    output_dir = Path(output_dir)
    report_path = output_dir / 'editorial_analysis_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Editorial Differences Analysis Report\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This analysis examines editorial differences between 自由時報 (Liberty Times) and 中國時報 (China Times) ")
        f.write(f"during {len(df)} divergent coverage periods from {df['week_date'].min()} to {df['week_date'].max()}.\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Top differences
        f.write("### Editorial Emphasis Differences\n\n")
        f.write("(Positive values indicate Liberty Times emphasizes more, negative values indicate China Times emphasizes more)\n\n")
        
        for category, diff in differences.head(3).items():
            category_name = category.replace('_per_1000_words', '').replace('_', ' ').title()
            direction = "Liberty Times" if diff > 0 else "China Times"
            f.write(f"- **{category_name}**: {direction} emphasizes {abs(diff):.1f} mentions per 1000 words more\n")
        
        f.write("\n### Coverage Volume\n\n")
        volume_stats = df.groupby('newspaper').agg({
            'gpr_articles': ['mean', 'sum'],
            'total_articles': ['mean', 'sum']
        }).round(1)
        
        for newspaper in volume_stats.index:
            gpr_mean = volume_stats.loc[newspaper, ('gpr_articles', 'mean')]
            gpr_total = volume_stats.loc[newspaper, ('gpr_articles', 'sum')]
            f.write(f"- **{newspaper}**: Average {gpr_mean} GPR articles per week, {gpr_total} total GPR articles\n")
        
        f.write("\n## Methodology\n\n")
        f.write("1. **Data Source**: LLM-generated summaries of article samples from 13 divergent coverage weeks\n")
        f.write("2. **Keyword Analysis**: Automated extraction of thematic keywords from summaries\n")
        f.write("3. **Normalization**: Keyword frequencies normalized per 1000 words for fair comparison\n")
        f.write("4. **Categories Analyzed**: US Support, China Aggression, Taiwan Sovereignty, Economic Concerns, Cross-Strait Dialogue, Military Conflict\n\n")
        
        f.write("## Data Quality\n\n")
        f.write(f"- **Total Articles Analyzed**: {df['gpr_articles'].sum():,}\n")
        f.write(f"- **Average Summary Length**: {keyword_df['total_words'].mean():.0f} words\n")
        f.write(f"- **Coverage Period**: {len(df['week_date'].unique())} distinct weeks\n")
    
    print(f"Generated summary report: {report_path}")

def main():
    """Main analysis function."""
    
    print("=== Editorial Differences Analysis ===")
    print("Analyzing LLM-generated summaries for editorial patterns...\n")
    
    # Load data
    df = load_divergent_summaries()
    
    # Extract keywords and themes
    print("Extracting keywords and themes from summaries...")
    keyword_df = extract_keywords_from_summaries(df)
    
    # Analyze editorial emphasis
    newspaper_emphasis, differences = analyze_editorial_emphasis(keyword_df)
    
    # Analyze temporal patterns
    temporal_patterns = analyze_temporal_patterns(keyword_df)
    
    # Analyze category-specific patterns
    analyze_category_specific_patterns(df, keyword_df)
    
    # Create output directory
    output_dir = Path("data_analysis/output/divergent_coverage/editorial_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    create_editorial_comparison_plots(newspaper_emphasis, differences, output_dir)
    
    # Generate summary report
    generate_summary_report(df, keyword_df, differences, output_dir)
    
    # Save detailed results
    keyword_df.to_csv(output_dir / 'keyword_analysis_detailed.csv', index=False)
    newspaper_emphasis.to_csv(output_dir / 'newspaper_emphasis_summary.csv')
    differences.to_csv(output_dir / 'editorial_differences.csv', header=['difference'])
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")
    print(f"- Editorial comparison plot: editorial_emphasis_comparison.png")
    print(f"- Summary report: editorial_analysis_report.md")
    print(f"- Detailed data: keyword_analysis_detailed.csv")

if __name__ == "__main__":
    main()