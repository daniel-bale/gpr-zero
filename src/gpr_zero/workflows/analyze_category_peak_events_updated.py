#!/usr/bin/env python3
"""
Analyze peak events for each category using the new category-specific GPR peak weeks.
Samples articles from each category's peak week weighted by category match strength.
"""

import json
import logging
from datetime import datetime

import ibis
import numpy as np
import pandas as pd
from ibis import _ as ibis_row

from gpr_zero.constants import BINARY_MATRIX_TABLE as CACHED_TABLE_NAME
from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory
from gpr_zero.gpr_index.dictionary import DictionaryMatcher, normalize_categories
from gpr_zero.gpr_index.spikes import article_window
from gpr_zero.project import get_project

logger = logging.getLogger(__name__)
NEWSPAPER_PATTERNS = ["China Times", "Liberty Times", "United Daily"]
ARTICLES_PER_PEAK = 100


def load_category_peak_weeks():
    """Load the category-specific peak weeks from the new analysis."""
    peak_file = get_project().data_path(
        "category_gpr_indices", "category_peak_weeks.csv"
    )
    if not peak_file.exists():
        raise FileNotFoundError(
            "Category peak weeks file not found. Run calculate_category_gpr_complete.py first."
        )

    logger.info(f"Loading peak weeks from: {peak_file}")
    return pd.read_csv(peak_file, parse_dates=["peak_week"])


def load_step2_dictionaries_and_word_mapping():
    """Use Step2 definitions with the full cache vocabulary, as in the thesis."""
    project = get_project()
    dictionaries = {
        name: normalize_categories(dictionary)
        for name, dictionary in DictionaryInventory.thesis_category_directory(
            project.step2_dictionaries_dir
        ).dictionaries.items()
    }
    inventory = DictionaryInventory.thesis_category_directory(project.dictionaries_dir)
    return dictionaries, inventory.word_columns


def sample_category_peak_articles(
    conn, category: str, peak_week, dictionaries, word_column_mapping
):
    """Sample articles from a category's peak week weighted by category match strength."""
    # Handle peak_week formatting
    if hasattr(peak_week, "strftime"):
        peak_week_str = peak_week.strftime("%Y-%m-%d")
        week_start = peak_week
    else:
        peak_week_str = str(peak_week)[:10]
        week_start = pd.to_datetime(peak_week_str)

    logger.info(f"Sampling articles for {category} peak week: {peak_week_str}")

    # Connect to cached table
    cached_table = conn.table(CACHED_TABLE_NAME)

    # Calculate week range
    week_start, week_end = article_window(
        {"start_date": week_start, "end_date": week_start}
    )

    # Get target outlets
    all_outlets = (
        cached_table.select("outlet").distinct().to_pandas()["outlet"].tolist()
    )
    target_outlets = [o for o in all_outlets if any(p in o for p in NEWSPAPER_PATTERNS)]

    # Filter for peak week and target outlets using warc_date
    week_articles = cached_table.filter(
        (cached_table.warc_date >= week_start.strftime("%Y-%m-%d"))
        & (cached_table.warc_date <= week_end.strftime("%Y-%m-%d"))
        & (cached_table.outlet.isin(target_outlets))
    )

    # Build category-specific expressions for all dictionaries
    category_weight_expr = ibis.literal(0)

    for dictionary in dictionaries.values():
        match = DictionaryMatcher(dictionary).category_match(
            week_articles, word_column_mapping, category
        )
        if match is not None:
            category_weight_expr += match.fill_null(False).cast("int32")

    # Filter articles matching this category (weight > 0)
    category_articles_with_weight = week_articles.mutate(
        category_weight=category_weight_expr
    ).filter(ibis_row.category_weight > 0)

    # Count matches
    total_matches = category_articles_with_weight.count().execute()
    logger.info(f"Found {total_matches} articles matching {category} in peak week")

    if total_matches == 0:
        return pd.DataFrame()

    # Get all articles with weights for sampling
    articles_with_weights = category_articles_with_weight.select(
        "warc_id", "url", "outlet", "title", "warc_date", "content", "category_weight"
    ).to_pandas()

    # Sample articles weighted by category_weight
    sample_size = min(ARTICLES_PER_PEAK, total_matches)

    if sample_size < total_matches:
        # Weighted sampling - articles with higher weights more likely to be selected
        weights = articles_with_weights["category_weight"].values
        weights = weights / weights.sum()  # Normalize to probabilities

        sampled_indices = np.random.choice(
            len(articles_with_weights), size=sample_size, replace=False, p=weights
        )
        sampled_articles = articles_with_weights.iloc[sampled_indices]
    else:
        sampled_articles = articles_with_weights

    logger.info(
        f"Sampled {len(sampled_articles)} articles (avg weight: {sampled_articles['category_weight'].mean():.2f})"
    )

    return sampled_articles


# Define serialization function at module level
def serialize_data(obj):
    """Serialize various data types for JSON output."""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main():
    project = get_project()
    output_dir = project.data_path("category_contributions", "peak_events_updated")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "=== Analyze Category Peak Events (Updated with Category-Specific Indices) ==="
    )

    # Load category-specific peak weeks
    peak_weeks_df = load_category_peak_weeks()
    logger.info(f"Loaded {len(peak_weeks_df)} category peak weeks")

    # Load dictionaries
    dictionaries, word_column_mapping = load_step2_dictionaries_and_word_mapping()
    logger.info(f"Loaded {len(dictionaries)} Step2 dictionaries")
    logger.info(f"Total words in mapping: {len(word_column_mapping)}")

    # Initialize PromptManager
    logger.info("Initializing PromptManager...")
    prompt_manager = project.prompt_manager()

    # Connect to database
    conn = project.connect()
    inventory = DictionaryInventory.thesis_category_directory(project.dictionaries_dir)
    ArticleCorpus.from_cache(conn, inventory)

    # Process each category peak
    all_peak_analyses = []

    for _, peak_row in peak_weeks_df.iterrows():
        category = peak_row["category"]
        peak_week = peak_row["peak_week"]

        logger.info(f"\n{'=' * 60}")
        # Handle peak_week formatting
        if hasattr(peak_week, "strftime"):
            peak_week_str = peak_week.strftime("%Y-%m-%d")
        else:
            peak_week_str = str(peak_week)[:10]
        logger.info(f"Processing {category} peak: {peak_week_str}")

        # Sample articles
        articles_df = sample_category_peak_articles(
            conn, category, peak_week, dictionaries, word_column_mapping
        )

        if len(articles_df) == 0:
            logger.warning(f"No articles found for {category} peak week")
            continue

        # Get LLM analysis
        try:
            # Create a summary of articles for the template
            articles_summary = ""
            for idx, (_, row) in enumerate(articles_df.head(30).iterrows()):
                weight = row["category_weight"] if "category_weight" in row else "N/A"
                articles_summary += f"""\nArticle {idx + 1} (Weight: {weight}):
- Date: {row["warc_date"]}
- Outlet: {row["outlet"]}
- Title: {row["title"]}
- Content Preview: {row["content"][:300] if pd.notna(row["content"]) else "N/A"}...\n"""

            template_args = [
                {},  # Empty dict for system message (no variables)
                {
                    "category_name": category,
                    "dictionary_name": "Step2 (40 dictionaries)",
                    "peak_week": peak_week_str,
                    "peak_share": peak_row["scaled_peak_value"]
                    / 100.0,  # Normalized value
                    "contribution_percent": peak_row[
                        "scaled_peak_value"
                    ],  # Use scaled value as proxy
                    "articles_summary": articles_summary,
                },
            ]

            llm_response = prompt_manager.call_llm_with_prompt_file(
                prompt_template_file=str(
                    project.root / "prompts" / "analyze_category_peak_event.json"
                ),
                template_args_list=template_args,
            )
            # Parse the JSON response
            if isinstance(llm_response["text"], str):
                try:
                    analysis_data = json.loads(llm_response["text"])
                    analysis = analysis_data
                except json.JSONDecodeError:
                    analysis = llm_response["text"]
            else:
                analysis = llm_response["text"]
            logger.info("LLM analysis generated successfully")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            import traceback

            traceback.print_exc()
            analysis = f"Error generating analysis: {str(e)}"

        # Store results
        peak_analysis = {
            "category": category,
            "peak_week": peak_week,
            "scaled_peak_value": peak_row["scaled_peak_value"],
            "unscaled_peak_value": peak_row["unscaled_peak_value"],
            "articles_sampled": len(articles_df),
            "sampled_articles": articles_df.to_dict("records"),
            "llm_analysis": analysis,
        }

        all_peak_analyses.append(peak_analysis)

        # Save individual analysis
        category_file = (
            output_dir
            / f"{category.replace(' ', '_').lower()}_peak_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(
                peak_analysis, f, ensure_ascii=False, indent=2, default=serialize_data
            )

        logger.info(f"Saved analysis to: {category_file}")

    # Save comprehensive results
    final_output = {
        "analysis_date": datetime.now().isoformat(),
        "total_categories_analyzed": len(all_peak_analyses),
        "peak_analyses": all_peak_analyses,
    }

    final_file = (
        output_dir
        / f"all_category_peak_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2, default=serialize_data)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Analysis complete! Saved to: {final_file}")
    logger.info(f"Total categories analyzed: {len(all_peak_analyses)}")

    # Print summary of peak events
    logger.info("\n=== Summary of Category Peak Events ===")
    for analysis in all_peak_analyses:
        logger.info(f"\n{analysis['category']}:")
        logger.info(f"  Peak Week: {analysis['peak_week']}")
        logger.info(f"  Scaled Index Value: {analysis['scaled_peak_value']:.1f}")
        logger.info(f"  Articles Sampled: {analysis['articles_sampled']}")
        if (
            isinstance(analysis["llm_analysis"], dict)
            and "peak_event_description" in analysis["llm_analysis"]
        ):
            logger.info(
                f"  Event: {analysis['llm_analysis']['peak_event_description']}"
            )

    conn.disconnect()
