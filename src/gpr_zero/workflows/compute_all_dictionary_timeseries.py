#!/usr/bin/env python3
"""
Aggregate word-presence flags into category-level indicators and compute index time series
for all 60 dictionaries. This script processes all dictionaries and generates time series data.
"""

import json
import logging
from datetime import datetime

from tqdm import tqdm

from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory
from gpr_zero.project import get_project

logger = logging.getLogger(__name__)


def rebuild_cache(project=None):
    """Explicitly rebuild the cache for the complete dictionary inventory."""
    project = project or get_project()
    inventory = DictionaryInventory.from_directory(project.dictionaries_dir)
    conn = project.connect(read_only=False)
    try:
        ArticleCorpus.rebuild(conn, inventory)
    finally:
        conn.disconnect()


def main(project=None):
    """Compute all dictionary time series from the existing canonical cache."""
    project = project or get_project()
    output_dir = project.data_path("gpr_timeseries")
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = DictionaryInventory.from_directory(project.dictionaries_dir)
    dict_files = list(project.dictionaries_dir.glob("*.json"))
    conn = project.connect()
    corpus = ArticleCorpus.from_cache(conn, inventory)
    calculator = corpus.calculator()

    # Process each dictionary and compute time series
    results_summary = {}
    successful = 0
    failed = 0

    time_buckets = ["W"]  # Weekly aggregations only

    # Create progress bar for dictionaries
    dict_progress = tqdm(
        sorted(dict_files), desc="Processing dictionaries", unit="dict"
    )

    for dict_file in dict_progress:
        dict_name = dict_file.stem

        # Update progress bar description
        dict_progress.set_description(f"Processing {dict_name}")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing dictionary: {dict_name}")
        logger.info(f"{'=' * 60}")

        try:
            dict_results = {}

            for time_bucket in time_buckets:
                logger.info(f"Computing {time_bucket} time series for {dict_name}...")

                # Query with different aggregation levels

                # 1. Overall time series (no additional grouping)
                overall_ts = calculator.query(
                    dictionary_name=dict_name,  # Use full name with prefix
                    group_by_cols=[],  # No additional grouping, time_bucket handles time
                    time_bucket=time_bucket,
                    output_format="wide",
                )

                # Save overall time series
                output_file = output_dir / f"{dict_name}_{time_bucket}_overall.parquet"
                overall_ts.write_parquet(str(output_file))
                logger.info(
                    f"  ✓ Saved overall {time_bucket} series: {overall_ts.shape[0]} periods"
                )

                # 2. By outlet time series
                outlet_ts = calculator.query(
                    dictionary_name=dict_name,  # Use full name with prefix
                    group_by_cols=[
                        "outlet"
                    ],  # Group by outlet, time_bucket handles time
                    time_bucket=time_bucket,
                    output_format="wide",
                )

                # Save outlet time series
                output_file = (
                    output_dir / f"{dict_name}_{time_bucket}_by_outlet.parquet"
                )
                outlet_ts.write_parquet(str(output_file))
                logger.info(
                    f"  ✓ Saved outlet {time_bucket} series: {outlet_ts.shape[0]} periods"
                )

                dict_results[time_bucket] = {
                    "overall_periods": overall_ts.shape[0],
                    "outlet_periods": outlet_ts.shape[0],
                    "categories": [
                        col for col in overall_ts.columns if col.endswith("_share")
                    ],
                }

            # Save metadata
            metadata = {
                "dictionary_name": dict_name,
                "processed_at": datetime.now().isoformat(),
                "results": dict_results,
                "status": "success",
            }

            metadata_file = output_dir / f"{dict_name}_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            results_summary[dict_name] = metadata
            successful += 1
            logger.info(f"✓ Successfully processed {dict_name}")

        except Exception as e:
            logger.error(f"✗ Failed to process {dict_name}: {e}")

            # Save error metadata
            error_metadata = {
                "dictionary_name": dict_name,
                "processed_at": datetime.now().isoformat(),
                "error": str(e),
                "status": "failed",
            }

            metadata_file = output_dir / f"{dict_name}_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(error_metadata, f, ensure_ascii=False, indent=2)

            results_summary[dict_name] = error_metadata
            failed += 1
            continue

    # Generate overall summary
    summary = {
        "processing_complete_at": datetime.now().isoformat(),
        "total_dictionaries": len(dict_files),
        "successful": successful,
        "failed": failed,
        "output_directory": str(output_dir),
        "time_buckets_processed": time_buckets,
        "results": results_summary,
    }

    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TIME SERIES COMPUTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total dictionaries processed: {len(dict_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time buckets: {', '.join(time_buckets)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Summary saved to: {summary_file}")

    # List some output files
    output_files = list(output_dir.glob("*.parquet"))
    logger.info(f"\nGenerated {len(output_files)} time series files:")
    for f in sorted(output_files)[:10]:
        logger.info(f"  - {f.name}")
    if len(output_files) > 10:
        logger.info(f"  ... and {len(output_files) - 10} more")

    if failed > 0:
        logger.warning(
            f"\n⚠️  {failed} dictionaries failed to process. Check individual metadata files for details."
        )

    conn.disconnect()
    logger.info("\n✓ All processing complete!")
