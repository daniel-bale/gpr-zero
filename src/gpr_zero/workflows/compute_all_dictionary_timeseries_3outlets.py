#!/usr/bin/env python3
"""
Compute GPR index time series for all dictionaries, LIMITED TO 3 MAIN OUTLETS.
This ensures consistency with category-specific calculations.
"""

import json
import logging
from datetime import datetime

import ibis
from ibis import _
from tqdm import tqdm

from gpr_zero.constants import TARGET_OUTLETS
from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory
from gpr_zero.gpr_index.dictionary import DictionaryMatcher
from gpr_zero.project import get_project

logger = logging.getLogger(__name__)


def main(project=None):
    """Compute weekly indices for the thesis three-outlet article cohort."""
    project = project or get_project()
    output_dir = project.data_path("gpr_timeseries_3outlets")
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = DictionaryInventory.thesis_directory(project.dictionaries_dir)
    conn = project.connect()
    corpus = ArticleCorpus.from_cache(conn, inventory).three_outlets()
    filtered_matrix = corpus.table
    dictionaries = inventory.dictionaries
    logger.info("Rows in three-outlet cohort: %s", filtered_matrix.count().execute())

    # Process each dictionary
    successful = 0
    failed = 0

    dict_progress = tqdm(
        sorted(dictionaries), desc="Processing dictionaries", unit="dict"
    )

    for dict_name in dict_progress:
        dict_progress.set_description(f"Processing {dict_name}")

        try:
            dictionary = dictionaries[dict_name]

            gpr_expr = DictionaryMatcher(dictionary).timeseries_match(
                filtered_matrix, corpus.word_columns
            )

            # Calculate weekly GPR
            weekly_gpr = (
                filtered_matrix.mutate(is_gpr=gpr_expr.cast("int32"))
                .mutate(week=_.warc_date.cast("date").truncate("W"))
                .group_by(["week"])
                .agg(gpr_articles=_.is_gpr.sum(), total_articles=ibis._.count())
                .mutate(
                    gpr_index=_.gpr_articles.cast("float64")
                    / _.total_articles.cast("float64")
                )
                .order_by("week")
            )

            # Convert to pandas
            df = weekly_gpr.to_pandas()

            # Save weekly overall time series
            output_file = output_dir / f"{dict_name}_W_overall_3outlets.parquet"
            df.to_parquet(output_file)
            logger.info(f"  ✓ Saved {dict_name}: {len(df)} weeks")

            successful += 1

        except Exception as e:
            logger.error(f"Failed to process {dict_name}: {e}")
            failed += 1
            continue

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("TIME SERIES COMPUTATION COMPLETE (3 OUTLETS)")
    logger.info(f"{'=' * 80}")
    logger.info(f"Target outlets: {', '.join(TARGET_OUTLETS)}")
    logger.info(f"Total dictionaries processed: {len(dictionaries)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {output_dir}")

    # Save summary
    summary = {
        "processing_complete_at": datetime.now().isoformat(),
        "target_outlets": TARGET_OUTLETS,
        "total_dictionaries": len(dictionaries),
        "successful": successful,
        "failed": failed,
        "output_directory": str(output_dir),
    }

    summary_file = output_dir / "processing_summary_3outlets.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    conn.disconnect()
    logger.info(f"Summary saved to: {summary_file}")
