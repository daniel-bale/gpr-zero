#!/usr/bin/env python3
"""
Sample articles for divergent coverage analysis.

This script samples articles for each identified divergent week and newspaper combination,
creating markdown files with article samples and statistics for LLM analysis.
Optimized for speed with early time and newspaper filtering.
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


def sample_articles_for_week(
    conn, 
    week_date: str, 
    newspaper: str, 
    gpr_calculator: GPRCalculator,
    output_dir: Path,
    max_articles_per_category: int = 50
) -> Dict[str, Any]:
    """Sample articles for a specific week and newspaper"""
    
    print(f"Processing {newspaper} for week {week_date}...")
    
    # Convert week date to date range
    week_start = datetime.strptime(week_date, '%Y-%m-%d').date()
    week_end = week_start + timedelta(days=6)
    
    # Map newspaper names to outlet patterns
    outlet_map = {
        "自由時報": "Liberty Times",
        "中國時報": "China Times", 
        "聯合報": "United Daily"
    }
    outlet_pattern = outlet_map.get(newspaper, newspaper)
    
    # Get articles table and apply ALL filters early for performance
    articles_table = conn.table("articles")
    
    print(f"  Filtering for {outlet_pattern} from {week_start} to {week_end}...")
    
    # Apply time and newspaper filtering FIRST to reduce dataset size
    articles_filtered = articles_table.filter(
        (ibis._.warc_date.cast('date') >= ibis.literal(week_start).cast('date')) &
        (ibis._.warc_date.cast('date') <= ibis.literal(week_end).cast('date')) &
        (ibis._.outlet.contains(outlet_pattern))
    )
    
    # Check if we have any articles before proceeding
    total_articles = articles_filtered.count().execute()
    if total_articles == 0:
        print(f"  No articles found for {newspaper} in week {week_date}")
        return None
    
    print(f"  Found {total_articles} articles after filtering")
    
    # Prepare for GPR calculation with minimal required columns
    articles_prepared = articles_filtered.mutate(
        newspaper=ibis.literal(newspaper),
        date=ibis._.warc_date.cast('date'),
        content_lower=ibis._.content.lower(),
        source_file=ibis._.source_warc_file,
        category=ibis.literal("")  # Empty category for CC-NEWS data
    ).select(['date', 'content', 'content_lower', 'source_file', 'category', 'outlet', 'newspaper', 'warc_id', 'title', 'url'])
    
    print(f"  Running GPR classification on {total_articles} articles...")
    
    # Calculate GPR labels - this is the expensive operation
    labeled_articles = gpr_calculator.prepare(articles_prepared)
    
    # Sample articles by category
    category_samples = {}
    category_counts = {}
    
    # Get all GPR categories from the dictionary
    gpr_categories = [
        "Military Escalation",
        "Foreign Intervention Risk", 
        "Economic Coercion",
        "Cyber and Information Warfare",
        "Diplomatic Crisis",
        "Internal Political Instability",
        "Critical Infrastructure Threat"
    ]
    
    print(f"  Sampling articles by category...")
    
    for category in gpr_categories:
        # Handle Diplomatic Crisis specially - combine both categories
        if category == "Diplomatic Crisis":
            category_articles = labeled_articles.filter(
                (labeled_articles["is_Diplomatic Crisis"] == True) |
                (labeled_articles["is_Diplomatic Crisis manual"] == True)
            )
        else:
            # Get articles for this category - use exact category name
            category_col = f"is_{category}"
            category_articles = labeled_articles.filter(
                labeled_articles[category_col] == True
            )
        
        category_count = category_articles.count().execute()
        category_counts[category] = category_count
        
        if category_count > 0:
            # Sample articles randomly (limit to max_articles_per_category)
            sampled = category_articles.order_by(ibis.random()).limit(max_articles_per_category).execute()
            
            # Convert to list of dicts for easier handling
            articles_list = []
            # Handle different DataFrame types (pandas vs polars)
            if hasattr(sampled, 'iter_rows'):
                # Polars DataFrame
                for row in sampled.iter_rows(named=True):
                    articles_list.append({
                        'warc_id': row['warc_id'],
                        'date': str(row['date']),
                        'title': row['title'] or '',
                        'content': (row['content'] or '')[:2000],  # Limit content length
                        'url': row['url'] or '',
                        'outlet': row['outlet'] or ''
                    })
            else:
                # Pandas DataFrame
                for _, row in sampled.iterrows():
                    articles_list.append({
                        'warc_id': row['warc_id'],
                        'date': str(row['date']),
                        'title': row['title'] if pd.notna(row['title']) else '',
                        'content': (row['content'] if pd.notna(row['content']) else '')[:2000],  # Limit content length
                        'url': row['url'] if pd.notna(row['url']) else '',
                        'outlet': row['outlet'] if pd.notna(row['outlet']) else ''
                    })
            
            category_samples[category] = articles_list
            print(f"    {category}: {category_count} articles, sampled {len(articles_list)}")
        else:
            category_samples[category] = []
    
    # Create markdown content
    markdown_content = create_week_markdown(
        week_date=week_date,
        newspaper=newspaper,
        total_articles=total_articles,
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
        'gpr_articles': sum(category_counts.values()),
        'category_counts': category_counts,
        'filename': filename
    }


def create_week_markdown(
    week_date: str,
    newspaper: str,
    total_articles: int,
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
- **GPR Articles**: {total_gpr_articles:,}
- **GPR Share**: {(total_gpr_articles/total_articles*100):.2f}%

## Category Breakdown

"""
    
    # Add category counts
    for category, count in category_counts.items():
        if count > 0:
            share = (count / total_articles * 100) if total_articles > 0 else 0
            content += f"- **{category}**: {count:,} articles ({share:.2f}%)\n"
    
    content += "\n---\n\n"
    
    # Add article samples for each category
    for category, articles in category_samples.items():
        if articles:
            content += f"## {category} Articles ({len(articles)} samples)\n\n"
            
            for i, article in enumerate(articles, 1):
                content += f"### Article {i}\n\n"
                content += f"**Date**: {article['date']}\n\n"
                content += f"**Title**: {article['title']}\n\n"
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
    
    # Get database connection
    print("Connecting to database...")
    db_path = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
    conn = ibis.connect(f"duckdb://{db_path}?access_mode=read_only")
    
    # Create output directory
    output_dir = Path("data_analysis/output/divergent_coverage/article_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each divergent week for both newspapers
    results = []
    newspapers = ["自由時報", "中國時報"]
    
    total_combinations = len(divergent_weeks) * len(newspapers)
    current_combination = 0
    
    for week_data in divergent_weeks:
        week_date = week_data['week']
        
        for newspaper in newspapers:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] Processing {newspaper} for week {week_date}")
            
            try:
                result = sample_articles_for_week(
                    conn=conn,
                    week_date=week_date,
                    newspaper=newspaper,
                    gpr_calculator=gpr_calculator,
                    output_dir=output_dir
                )
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing {newspaper} for week {week_date}: {e}")
                continue
    
    # Save summary of results
    if results:
        summary_df = pl.DataFrame(results)
        summary_path = output_dir / "sampling_summary.csv"
        summary_df.write_csv(summary_path)
        print(f"\nSampling summary saved to: {summary_path}")
        
        # Print summary statistics
        print(f"\nProcessed {len(results)} week/newspaper combinations")
        print(f"Total articles sampled: {summary_df['total_articles'].sum():,}")
        print(f"Total GPR articles: {summary_df['gpr_articles'].sum():,}")
        
        # Show results by newspaper
        by_newspaper = summary_df.group_by("newspaper").agg([
            pl.count().alias("weeks"),
            pl.sum("total_articles").alias("total_articles"),
            pl.sum("gpr_articles").alias("gpr_articles")
        ])
        print("\nBy newspaper:")
        print(by_newspaper)
    
    else:
        print("No articles were successfully sampled!")


if __name__ == "__main__":
    main()