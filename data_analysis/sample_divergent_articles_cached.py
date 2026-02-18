#!/usr/bin/env python3
"""
Cached article sampling for divergent coverage analysis.

This script uses materialized=True for caching GPR calculations, then applies time filtering
and samples 300 articles per week/newspaper combination, organizing them by category.
"""

import polars as pl
import pandas as pd
import ibis
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append('src')
from gpr_zero.gpr_index.calculator import GPRCalculator


def prepare_combined_materialized_table(conn, gpr_calculator: GPRCalculator):
    """Prepare a single materialized table containing all three newspapers"""
    
    print("Preparing combined materialized table for all newspapers...")
    
    # Get articles table
    articles_table = conn.table("articles")
    
    # Filter for all three newspapers and prepare articles
    print("  Filtering articles for all three newspapers...")
    articles_prepared = articles_table.filter(
        ibis._.outlet.contains("Liberty Times") |
        ibis._.outlet.contains("China Times") |
        ibis._.outlet.contains("United Daily")
    ).mutate(
        # Add newspaper column based on outlet
        newspaper=ibis.cases(
            (ibis._.outlet.contains("Liberty Times"), ibis.literal("自由時報")),
            (ibis._.outlet.contains("China Times"), ibis.literal("中國時報")),
            (ibis._.outlet.contains("United Daily"), ibis.literal("聯合報")),
            else_=ibis.literal("Unknown")
        ),
        date=ibis._.warc_date.cast('date'),
        content_lower=ibis._.content.lower(),
        source_file=ibis._.source_warc_file,
        category=ibis.literal("")  # Empty category for CC-NEWS data
    ).select(['date', 'content', 'content_lower', 'source_file', 'category', 'outlet', 'newspaper', 'warc_id', 'title', 'url'])
    
    # Calculate GPR labels with materialization
    print("  Running GPR classification with materialization...")
    labeled_articles = gpr_calculator.prepare(articles_prepared, materialize=True, force_recalculate=True, overwrite=True)
    
    print(f"  Materialized table created with {labeled_articles.count().to_polars()} total articles")
    return labeled_articles


def sample_articles_for_week_cached(
    labeled_articles_table,
    week_date: str,
    newspaper: str,
    output_dir: Path,
    sample_size: int = 300
) -> Dict[str, Any]:
    """Sample articles from the materialized table for a specific week/newspaper"""
    
    print(f"Processing {newspaper} for week {week_date}...")
    
    # Convert week date to date range
    week_start = datetime.strptime(week_date, '%Y-%m-%d').date()
    week_end = week_start + timedelta(days=6)
    
    # Filter by newspaper and time period
    print(f"  Filtering for {newspaper} in week {week_start} to {week_end}...")
    week_articles = labeled_articles_table.filter(
        (ibis._.newspaper == newspaper) &
        (ibis._.date >= ibis.literal(week_start).cast('date')) &
        (ibis._.date <= ibis.literal(week_end).cast('date'))
    )
    
    # Check total count for this week
    total_articles = week_articles.count().to_polars()
    if total_articles == 0:
        print(f"  No articles found for {newspaper} in week {week_date}")
        return None
    
    # Filter to only GPR articles before sampling
    print(f"  Found {total_articles} total articles, filtering to GPR articles...")
    gpr_articles = week_articles.filter(ibis._.matches_any_category == True)
    
    gpr_count = gpr_articles.count().to_polars()
    if gpr_count == 0:
        print(f"  No GPR articles found for {newspaper} in week {week_date}")
        return None
    
    print(f"  Found {gpr_count} GPR articles, sampling {min(sample_size, gpr_count)}...")
    
    # Sample GPR articles randomly
    sampled_articles = gpr_articles.order_by(ibis.random()).limit(sample_size).to_polars()
    
    # Convert to pandas for easier iteration
    sampled_articles = sampled_articles.to_pandas()
    
    print(f"  Organizing {len(sampled_articles)} articles by category...")
    
    # Initialize category containers
    gpr_categories = [
        "Military Escalation",
        "Foreign Intervention Risk", 
        "Economic Coercion",
        "Cyber and Information Warfare",
        "Diplomatic Crisis",
        "Internal Political Instability",
        "Critical Infrastructure Threat"
    ]
    
    category_samples = {cat: [] for cat in gpr_categories}
    category_counts = {cat: 0 for cat in gpr_categories}
    
    # Organize articles by category (each article assigned to only one category)
    for _, row in sampled_articles.iterrows():
        # Get article data
        article_data = {
            'warc_id': row['warc_id'],
            'date': str(row['date']),
            'title': row['title'] if pd.notna(row['title']) else '',
            'content': (row['content'] if pd.notna(row['content']) else '')[:2000],
            'url': row['url'] if pd.notna(row['url']) else '',
            'outlet': row['outlet'] if pd.notna(row['outlet']) else ''
        }
        
        # Find all categories this article belongs to
        article_categories = []
        for category in gpr_categories:
            # Handle Diplomatic Crisis specially - combine both categories
            if category == "Diplomatic Crisis":
                is_category = (row.get("is_Diplomatic Crisis", False) or
                             row.get("is_Diplomatic Crisis manual", False))
            else:
                is_category = row.get(f"is_{category}", False)
            
            if is_category:
                article_categories.append(category)
        
        # Assign article to first matching category (to avoid double counting)
        if article_categories:
            primary_category = article_categories[0]
            
            # Add multiple category label if applicable
            if len(article_categories) > 1:
                article_data['multiple_categories'] = True
                article_data['all_categories'] = ', '.join(article_categories)
                article_data['title'] = f"[MULTI: {', '.join(article_categories)}] {article_data['title']}"
            else:
                article_data['multiple_categories'] = False
                article_data['all_categories'] = primary_category
            
            category_samples[primary_category].append(article_data)
            category_counts[primary_category] += 1
    
    # Print category results
    total_gpr_articles = sum(category_counts.values())
    print(f"  GPR articles in sample: {total_gpr_articles}")
    for category, count in category_counts.items():
        if count > 0:
            print(f"    {category}: {count} articles")
    
    # Create markdown content
    markdown_content = create_week_markdown(
        week_date=week_date,
        newspaper=newspaper,
        total_articles=total_articles,
        sampled_articles=len(sampled_articles),
        category_counts=category_counts,
        category_samples=category_samples
    )
    
    # Save to file
    filename = f"{week_date}_{newspaper.replace(' ', '_')}.md"
    filepath = output_dir / filename
    filepath.write_text(markdown_content, encoding='utf-8')
    
    print(f"  Saved: {filepath}")
    
    return {
        'week': week_date,
        'newspaper': newspaper,
        'total_articles': total_articles,
        'sampled_articles': len(sampled_articles),
        'gpr_articles': total_gpr_articles,
        'category_counts': category_counts,
        'filename': filename
    }


def create_week_markdown(
    week_date: str,
    newspaper: str,
    total_articles: int,
    sampled_articles: int,
    category_counts: Dict[str, int],
    category_samples: Dict[str, List[Dict]]
) -> str:
    """Create markdown content for a week's articles"""
    
    week_start = datetime.strptime(week_date, '%Y-%m-%d').date()
    week_end = week_start + timedelta(days=6)
    
    # Calculate total GPR articles from category counts
    total_gpr_articles = sum(category_counts.values())
    
    content = f"""# {newspaper} - Week of {week_date}

## Summary Statistics

- **Week Period**: {week_start} to {week_end}
- **Total Articles**: {total_articles:,}
- **Sampled Articles**: {sampled_articles:,}
- **GPR Articles in Sample**: {total_gpr_articles:,}
- **GPR Share in Sample**: {(total_gpr_articles/sampled_articles*100):.2f}%

## Category Breakdown (from sample)

"""
    
    # Add category counts
    for category, count in category_counts.items():
        if count > 0:
            share = (count / sampled_articles * 100) if sampled_articles > 0 else 0
            content += f"- **{category}**: {count:,} articles ({share:.2f}% of sample)\n"
    
    content += "\n---\n\n"
    
    # Add article samples for each category
    for category, articles in category_samples.items():
        if articles:
            content += f"## {category} Articles ({len(articles)} samples)\n\n"
            
            for i, article in enumerate(articles, 1):
                content += f"### Article {i}\n\n"
                content += f"**Date**: {article['date']}\n\n"
                content += f"**Title**: {article['title']}\n\n"
                
                # Add multiple categories info if applicable
                if article.get('multiple_categories', False):
                    content += f"**Categories**: {article['all_categories']} (assigned to {category})\n\n"
                
                content += f"**URL**: {article['url']}\n\n"
                content += f"**Outlet**: {article['outlet']}\n\n"
                content += f"**Content**: {article['content']}\n\n"
                content += "---\n\n"
    
    return content


def main():
    """Main function to sample articles for all divergent weeks"""
    
    # Load divergent weeks data
    print("Loading divergent weeks data...")
    with open("data_analysis/output/divergent_coverage/divergent_weeks_detailed.json", 'r', encoding='utf-8') as f:
        divergent_weeks = json.load(f)
    
    print(f"Found {len(divergent_weeks)} divergent weeks")
    
    # Load GPR dictionary and initialize calculator
    print("Loading GPR dictionary...")
    dictionary_path = "data/dictionaries/taiwan_gpr_dict_president_updated.json"
    gpr_calculator = GPRCalculator(dictionary_path)
    
    # Get database connection (write mode for materialization)
    print("Connecting to database...")
    db_path = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
    conn = ibis.connect(f"duckdb://{db_path}")
    
    # Create output directory
    output_dir = Path("data_analysis/output/divergent_coverage/article_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare combined materialized table once for all newspapers
    print("\n=== PREPARING MATERIALIZED TABLE ===")
    labeled_articles_table = prepare_combined_materialized_table(conn, gpr_calculator)
    
    # Process each divergent week for the two main newspapers in the analysis
    results = []
    newspapers = ["自由時報", "中國時報"]  # Only these two for divergent coverage analysis
    
    total_combinations = len(divergent_weeks) * len(newspapers)
    current_combination = 0
    
    for week_data in divergent_weeks:
        week_date = week_data['week']
        
        for newspaper in newspapers:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] Processing {newspaper} for week {week_date}")
            
            try:
                result = sample_articles_for_week_cached(
                    labeled_articles_table=labeled_articles_table,
                    week_date=week_date,
                    newspaper=newspaper,
                    output_dir=output_dir
                )
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing {newspaper} for week {week_date}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save summary of results
    if results:
        # Flatten the results for CSV export
        flattened_results = []
        for result in results:
            # Create base record
            base_record = {
                'week': result['week'],
                'newspaper': result['newspaper'],
                'total_articles': result['total_articles'],
                'sampled_articles': result['sampled_articles'],
                'gpr_articles': result['gpr_articles'],
                'filename': result['filename']
            }
            
            # Add category counts as separate columns
            for category, count in result['category_counts'].items():
                base_record[f'count_{category.replace(" ", "_")}'] = count
            
            flattened_results.append(base_record)
        
        summary_df = pl.DataFrame(flattened_results)
        summary_path = output_dir / "sampling_summary.csv"
        summary_df.write_csv(summary_path)
        print(f"\nSampling summary saved to: {summary_path}")
        
        # Print summary statistics
        print(f"\nProcessed {len(results)} week/newspaper combinations")
        print(f"Total articles sampled: {summary_df['sampled_articles'].sum():,}")
        print(f"Total GPR articles: {summary_df['gpr_articles'].sum():,}")
        
        # Show results by newspaper
        by_newspaper = summary_df.group_by("newspaper").agg([
            pl.count().alias("weeks"),
            pl.sum("sampled_articles").alias("total_sampled"),
            pl.sum("gpr_articles").alias("total_gpr")
        ])
        print("\nBy newspaper:")
        print(by_newspaper)
    
    else:
        print("No articles were successfully sampled!")


if __name__ == "__main__":
    main()