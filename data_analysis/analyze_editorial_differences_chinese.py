#!/usr/bin/env python3
"""
Analyze editorial differences between newspapers using actual Chinese articles.

This script reads the markdown files containing sampled Chinese articles,
analyzes them by category, and compares editorial patterns between newspapers
for the same weeks and same categories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
import sys
import json

# Add src to path for imports
sys.path.append('src')
from gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Ensure Chinese font support
ensure_cjk_font_setup()

def parse_markdown_file(filepath: Path) -> Dict[str, Any]:
    """Parse a markdown file to extract article information by category"""
    
    content = filepath.read_text(encoding='utf-8')
    
    # Extract basic info from filename
    filename = filepath.name
    parts = filename.replace('.md', '').split('_')
    week_date = parts[0]
    newspaper = parts[1]
    
    # Extract summary statistics
    stats = {}
    
    # Extract total articles
    total_match = re.search(r'\*\*Total Articles\*\*: ([\d,]+)', content)
    if total_match:
        stats['total_articles'] = int(total_match.group(1).replace(',', ''))
    
    # Extract sampled articles
    sampled_match = re.search(r'\*\*Sampled Articles\*\*: ([\d,]+)', content)
    if sampled_match:
        stats['sampled_articles'] = int(sampled_match.group(1).replace(',', ''))
    
    # Extract GPR articles in sample
    gpr_match = re.search(r'\*\*GPR Articles in Sample\*\*: ([\d,]+)', content)
    if gpr_match:
        stats['gpr_articles'] = int(gpr_match.group(1).replace(',', ''))
    
    # Extract category breakdown
    categories = {}
    category_pattern = r'\*\*([^*]+)\*\*: (\d+) articles \(([0-9.]+)% of sample\)'
    for match in re.finditer(category_pattern, content):
        category_name = match.group(1)
        article_count = int(match.group(2))
        percentage = float(match.group(3))
        categories[category_name] = {
            'count': article_count,
            'percentage': percentage
        }
    
    # Extract actual articles by category
    articles_by_category = {}
    
    # Split content by category sections
    category_sections = re.split(r'## ([^#\n]+) Articles \((\d+) samples\)', content)
    
    for i in range(1, len(category_sections), 3):
        if i + 2 < len(category_sections):
            category_name = category_sections[i].strip()
            sample_count = int(category_sections[i + 1])
            section_content = category_sections[i + 2]
            
            # Extract individual articles
            articles = []
            article_sections = re.split(r'### Article \d+', section_content)
            
            for article_section in article_sections[1:]:  # Skip first empty section
                article_data = {}
                
                # Extract title
                title_match = re.search(r'\*\*Title\*\*: (.+)', article_section)
                if title_match:
                    article_data['title'] = title_match.group(1).strip()
                
                # Extract date
                date_match = re.search(r'\*\*Date\*\*: (.+)', article_section)
                if date_match:
                    article_data['date'] = date_match.group(1).strip()
                
                # Extract content
                content_match = re.search(r'\*\*Content\*\*: (.+?)(?=\n---|\n\*\*|\Z)', article_section, re.DOTALL)
                if content_match:
                    article_data['content'] = content_match.group(1).strip()
                
                # Extract URL
                url_match = re.search(r'\*\*URL\*\*: (.+)', article_section)
                if url_match:
                    article_data['url'] = url_match.group(1).strip()
                
                if article_data:
                    articles.append(article_data)
            
            articles_by_category[category_name] = articles
    
    return {
        'week_date': week_date,
        'newspaper': newspaper,
        'stats': stats,
        'categories': categories,
        'articles': articles_by_category
    }

def load_gpr_dictionary() -> Dict[str, Any]:
    """Load the GPR dictionary from file"""
    dictionary_path = "data/dictionaries/taiwan_gpr_dict_president_updated.json"
    with open(dictionary_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_content_themes(articles: List[Dict], category: str, gpr_dict: Dict[str, Any]) -> Dict[str, int]:
    """Analyze themes in Chinese article content for a specific category using actual GPR dictionary"""
    
    # Get word sets from the dictionary
    word_sets = gpr_dict.get('word_sets', {})
    
    # Map categories to their constituent word sets based on dictionary structure
    category_word_sets = {
        'Military Escalation': ['military_threats', 'cross_strait_relations'],
        'Foreign Intervention Risk': ['foreign_intervention', 'military_threats'],
        'Economic Coercion': ['economic_coercion', 'cross_strait_relations'],
        'Cyber and Information Warfare': ['cyber_and_information_warfare', 'cross_strait_relations'],
        'Diplomatic Crisis': ['diplomatic_incidents', 'foreign_intervention', 'diplomatic_manual'],
        'Internal Political Instability': ['internal_political_risk', 'cross_strait_relations'],
        'Critical Infrastructure Threat': ['critical_infrastructure_threats', 'military_threats']
    }
    
    # Get word sets for this category
    category_sets = category_word_sets.get(category, [])
    
    # Count themes (word sets) for this category
    theme_counts = {}
    
    for word_set_name in category_sets:
        keywords = word_sets.get(word_set_name, [])
        if not keywords:
            continue
            
        count = 0
        for article in articles:
            content = article.get('content', '') + ' ' + article.get('title', '')
            # Check if any keyword from this word set appears in the article
            for keyword in keywords:
                if keyword in content:
                    count += 1
                    break  # Count each article only once per word set
        
        # Use a more readable name for the theme
        theme_name = word_set_name.replace('_', ' ').title()
        theme_counts[theme_name] = count
    
    return theme_counts

def compare_newspapers_by_category(data: Dict[str, Dict]) -> pd.DataFrame:
    """Compare newspaper coverage by category for the same weeks"""
    
    # Load GPR dictionary for theme analysis
    gpr_dict = load_gpr_dictionary()
    
    # Organize data by week and category
    comparison_data = []
    
    # Get all weeks that have both newspapers
    weeks_with_both = set()
    for newspaper_data in data.values():
        for week_data in newspaper_data.values():
            week = week_data['week_date']
            weeks_with_both.add(week)
    
    # Filter to weeks that actually have both newspapers
    valid_weeks = []
    for week in weeks_with_both:
        has_liberty = any(week in newspaper_data for newspaper_data in [data.get('自由時報', {})])
        has_china = any(week in newspaper_data for newspaper_data in [data.get('中國時報', {})])
        if has_liberty and has_china:
            valid_weeks.append(week)
    
    print(f"Found {len(valid_weeks)} weeks with both newspapers")
    
    # Compare each category for each valid week
    for week in valid_weeks:
        liberty_data = data.get('自由時報', {}).get(week)
        china_data = data.get('中國時報', {}).get(week)
        
        if not liberty_data or not china_data:
            continue
        
        # Get all categories present in either newspaper
        all_categories = set()
        all_categories.update(liberty_data['categories'].keys())
        all_categories.update(china_data['categories'].keys())
        
        for category in all_categories:
            liberty_cat = liberty_data['categories'].get(category, {'count': 0, 'percentage': 0})
            china_cat = china_data['categories'].get(category, {'count': 0, 'percentage': 0})
            
            # Calculate shares of total GPR articles
            liberty_gpr_total = liberty_data['stats'].get('gpr_articles', 1)
            china_gpr_total = china_data['stats'].get('gpr_articles', 1)
            
            liberty_share = (liberty_cat['count'] / liberty_gpr_total * 100) if liberty_gpr_total > 0 else 0
            china_share = (china_cat['count'] / china_gpr_total * 100) if china_gpr_total > 0 else 0
            
            # Analyze themes in articles
            liberty_articles = liberty_data['articles'].get(category, [])
            china_articles = china_data['articles'].get(category, [])
            
            liberty_themes = analyze_content_themes(liberty_articles, category, gpr_dict)
            china_themes = analyze_content_themes(china_articles, category, gpr_dict)
            
            comparison_data.append({
                'week': week,
                'category': category,
                'liberty_count': liberty_cat['count'],
                'china_count': china_cat['count'],
                'liberty_share': liberty_share,
                'china_share': china_share,
                'liberty_articles': len(liberty_articles),
                'china_articles': len(china_articles),
                'liberty_themes': liberty_themes,
                'china_themes': china_themes,
                'share_difference': liberty_share - china_share,
                'count_difference': liberty_cat['count'] - china_cat['count']
            })
    
    return pd.DataFrame(comparison_data)

def create_editorial_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """Create visualization comparing editorial patterns"""
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Ensure Chinese font support after style change
    ensure_cjk_font_setup()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Editorial Differences Analysis: Chinese Article Content', fontsize=16, fontweight='bold')
    
    # 1. Share differences by category (heatmap)
    pivot_shares = df.pivot_table(values='share_difference', index='week', columns='category', fill_value=0)
    
    sns.heatmap(pivot_shares, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                ax=axes[0,0], cbar_kws={'label': 'Share Difference (%)\n(自由時報 - 中國時報)'})
    axes[0,0].set_title('Share Differences by Week and Category')
    axes[0,0].set_xlabel('GPR Category')
    axes[0,0].set_ylabel('Week')
    
    # 2. Average share by newspaper and category
    avg_by_category = df.groupby('category').agg({
        'liberty_share': 'mean',
        'china_share': 'mean'
    }).reset_index()
    
    x = np.arange(len(avg_by_category))
    width = 0.35
    
    axes[0,1].bar(x - width/2, avg_by_category['liberty_share'], width, label='自由時報', alpha=0.8)
    axes[0,1].bar(x + width/2, avg_by_category['china_share'], width, label='中國時報', alpha=0.8)
    
    axes[0,1].set_title('Average Share by Category and Newspaper')
    axes[0,1].set_xlabel('GPR Category')
    axes[0,1].set_ylabel('Average Share of GPR Articles (%)')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(avg_by_category['category'], rotation=45, ha='right')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Count differences by category (box plot)
    df_melted = df.melt(id_vars=['week', 'category'], 
                       value_vars=['liberty_count', 'china_count'],
                       var_name='newspaper', value_name='count')
    df_melted['newspaper'] = df_melted['newspaper'].map({'liberty_count': '自由時報', 'china_count': '中國時報'})
    
    sns.boxplot(data=df_melted, x='category', y='count', hue='newspaper', ax=axes[1,0])
    axes[1,0].set_title('Article Count Distribution by Category')
    axes[1,0].set_xlabel('GPR Category')
    axes[1,0].set_ylabel('Article Count')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Share correlation between newspapers
    axes[1,1].scatter(df['liberty_share'], df['china_share'], alpha=0.6, s=50)
    
    # Add diagonal line
    max_val = max(df['liberty_share'].max(), df['china_share'].max())
    axes[1,1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal shares')
    
    # Add category labels for extreme points
    for _, row in df.iterrows():
        if abs(row['share_difference']) > 10:  # Label points with >10% difference
            axes[1,1].annotate(f"{row['category'][:10]}...\n{row['week']}", 
                             (row['liberty_share'], row['china_share']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.7)
    
    axes[1,1].set_title('Share Correlation Between Newspapers')
    axes[1,1].set_xlabel('自由時報 Share (%)')
    axes[1,1].set_ylabel('中國時報 Share (%)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'chinese_editorial_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Editorial comparison plot saved to: {plot_path}")

def analyze_theme_differences(df: pd.DataFrame, output_dir: Path):
    """Analyze differences in themes within categories"""
    
    theme_analysis = []
    
    for _, row in df.iterrows():
        week = row['week']
        category = row['category']
        liberty_themes = row['liberty_themes']
        china_themes = row['china_themes']
        
        # Get all themes for this category
        all_themes = set(liberty_themes.keys()) | set(china_themes.keys())
        
        for theme in all_themes:
            liberty_count = liberty_themes.get(theme, 0)
            china_count = china_themes.get(theme, 0)
            
            theme_analysis.append({
                'week': week,
                'category': category,
                'theme': theme,
                'liberty_count': liberty_count,
                'china_count': china_count,
                'difference': liberty_count - china_count
            })
    
    theme_df = pd.DataFrame(theme_analysis)
    
    # Save theme analysis
    theme_path = output_dir / 'chinese_theme_analysis.csv'
    theme_df.to_csv(theme_path, index=False, encoding='utf-8')
    print(f"Theme analysis saved to: {theme_path}")
    
    return theme_df

def generate_summary_report(df: pd.DataFrame, theme_df: pd.DataFrame, output_dir: Path):
    """Generate a summary report of editorial differences"""
    
    report = []
    report.append("# Chinese Editorial Differences Analysis Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall statistics
    report.append("## Overall Statistics")
    report.append(f"- Total week/category combinations analyzed: {len(df)}")
    report.append(f"- Weeks covered: {df['week'].nunique()}")
    report.append(f"- Categories analyzed: {df['category'].nunique()}")
    report.append("")
    
    # Category-level differences
    report.append("## Average Share Differences by Category")
    report.append("(Positive values indicate 自由時報 has higher share)")
    report.append("")
    
    category_summary = df.groupby('category').agg({
        'share_difference': ['mean', 'std', 'min', 'max'],
        'liberty_share': 'mean',
        'china_share': 'mean'
    }).round(2)
    
    for category in category_summary.index:
        mean_diff = category_summary.loc[category, ('share_difference', 'mean')]
        liberty_avg = category_summary.loc[category, ('liberty_share', 'mean')]
        china_avg = category_summary.loc[category, ('china_share', 'mean')]
        
        report.append(f"### {category}")
        report.append(f"- Average difference: {mean_diff:+.2f}%")
        report.append(f"- 自由時報 average share: {liberty_avg:.2f}%")
        report.append(f"- 中國時報 average share: {china_avg:.2f}%")
        report.append("")
    
    # Largest differences
    report.append("## Largest Share Differences")
    report.append("")
    
    # Top 10 largest positive differences (Liberty Times higher)
    top_liberty = df.nlargest(5, 'share_difference')
    report.append("### Top 5 cases where 自由時報 has higher share:")
    for _, row in top_liberty.iterrows():
        report.append(f"- {row['week']} - {row['category']}: +{row['share_difference']:.1f}% "
                     f"({row['liberty_share']:.1f}% vs {row['china_share']:.1f}%)")
    report.append("")
    
    # Top 10 largest negative differences (China Times higher)
    top_china = df.nsmallest(5, 'share_difference')
    report.append("### Top 5 cases where 中國時報 has higher share:")
    for _, row in top_china.iterrows():
        report.append(f"- {row['week']} - {row['category']}: {row['share_difference']:.1f}% "
                     f"({row['liberty_share']:.1f}% vs {row['china_share']:.1f}%)")
    report.append("")
    
    # Theme analysis summary
    if not theme_df.empty:
        report.append("## Theme Analysis Summary")
        report.append("")
        
        # Most divergent themes
        theme_summary = theme_df.groupby(['category', 'theme']).agg({
            'difference': ['mean', 'std']
        }).round(2)
        
        # Get themes with largest average differences
        theme_summary['abs_mean'] = abs(theme_summary[('difference', 'mean')])
        top_themes = theme_summary.nlargest(10, 'abs_mean')
        
        report.append("### Most Divergent Themes (by average difference):")
        for (category, theme), row in top_themes.iterrows():
            mean_diff = row[('difference', 'mean')]
            report.append(f"- {category} - {theme}: {mean_diff:+.1f} articles on average")
        report.append("")
    
    # Save report
    report_path = output_dir / 'chinese_editorial_analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to: {report_path}")

def main():
    """Main function to analyze editorial differences using Chinese articles"""
    
    print("Starting Chinese editorial differences analysis...")
    
    # Set up paths
    input_dir = Path("data_analysis/output/divergent_coverage/article_samples")
    output_dir = Path("data_analysis/output/divergent_coverage/chinese_editorial_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all markdown files
    markdown_files = list(input_dir.glob("*.md"))
    markdown_files = [f for f in markdown_files if f.name != 'sampling_summary.csv']
    
    print(f"Found {len(markdown_files)} markdown files to process")
    
    # Parse all markdown files
    parsed_data = {}
    for filepath in markdown_files:
        print(f"Parsing {filepath.name}...")
        try:
            data = parse_markdown_file(filepath)
            newspaper = data['newspaper']
            week = data['week_date']
            
            if newspaper not in parsed_data:
                parsed_data[newspaper] = {}
            
            parsed_data[newspaper][week] = data
            
        except Exception as e:
            print(f"Error parsing {filepath.name}: {e}")
            continue
    
    print(f"Successfully parsed data for {len(parsed_data)} newspapers")
    for newspaper, weeks in parsed_data.items():
        print(f"  {newspaper}: {len(weeks)} weeks")
    
    # Compare newspapers by category
    print("\nComparing newspapers by category...")
    comparison_df = compare_newspapers_by_category(parsed_data)
    
    if comparison_df.empty:
        print("No comparable data found!")
        return
    
    print(f"Generated {len(comparison_df)} category comparisons")
    
    # Save comparison data
    comparison_path = output_dir / 'chinese_category_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
    print(f"Comparison data saved to: {comparison_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_editorial_comparison_plots(comparison_df, output_dir)
    
    # Analyze themes
    print("\nAnalyzing themes...")
    theme_df = analyze_theme_differences(comparison_df, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(comparison_df, theme_df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()