#!/usr/bin/env python3
"""
Aggregate word-presence flags into category-level indicators and compute index time series
for all 60 dictionaries. This script processes all dictionaries and generates time series data.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import ibis
from gpr_zero.gpr_index.calculator import GPRCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Compute GPR index time series for all dictionaries."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    dictionaries_dir = project_root / "data" / "dictionaries" / "all_dictionaries"
    output_dir = project_root / "data" / "gpr_timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Database connection
    db_path = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        logger.info("Please ensure the news database exists before running this script.")
        return
    
    logger.info(f"Connecting to database: {db_path}")
    conn = ibis.connect(f"duckdb:///{db_path}")
    
    # Get articles table
    try:
        articles_table = conn.table("articles")
        logger.info(f"Connected to articles table with columns: {articles_table.columns}")
    except Exception as e:
        logger.error(f"Failed to connect to articles table: {e}")
        return
    
    # Get all dictionary files
    dict_files = list(dictionaries_dir.glob("*.json"))
    logger.info(f"Found {len(dict_files)} dictionary files")
    
    if not dict_files:
        logger.error(f"No dictionary files found in {dictionaries_dir}")
        return
    
    # Initialize the binary matrix calculator with all dictionaries
    logger.info("Initializing GPR calculator with all dictionaries...")
    calculator = GPRCalculator([str(f) for f in dict_files])
    
    # Prepare the binary matrix (materialize for performance)
    logger.info("Preparing binary matrix for all words across all dictionaries...")
    try:
        calculator.prepare_binary_matrix(
            articles_table,
            materialize=True,
            force_recalculate=True,  # Force recalculation
            overwrite=True
        )
        logger.info("✓ Binary matrix preparation complete")
    except Exception as e:
        logger.error(f"Failed to prepare binary matrix: {e}")
        return
    
    # Process each dictionary and compute time series
    results_summary = {}
    successful = 0
    failed = 0
    
    time_buckets = ['W']  # Weekly aggregations only
    
    # Create progress bar for dictionaries
    dict_progress = tqdm(sorted(dict_files), desc="Processing dictionaries", unit="dict")
    
    for dict_file in dict_progress:
        dict_name = dict_file.stem
        
        # Update progress bar description
        dict_progress.set_description(f"Processing {dict_name}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dictionary: {dict_name}")
        logger.info(f"{'='*60}")
        
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
                    output_format='wide'
                )
                
                # Save overall time series
                output_file = output_dir / f"{dict_name}_{time_bucket}_overall.parquet"
                overall_ts.write_parquet(str(output_file))
                logger.info(f"  ✓ Saved overall {time_bucket} series: {overall_ts.shape[0]} periods")
                
                # 2. By outlet time series
                outlet_ts = calculator.query(
                    dictionary_name=dict_name,  # Use full name with prefix
                    group_by_cols=['outlet'],  # Group by outlet, time_bucket handles time
                    time_bucket=time_bucket,
                    output_format='wide'
                )
                
                # Save outlet time series
                output_file = output_dir / f"{dict_name}_{time_bucket}_by_outlet.parquet"
                outlet_ts.write_parquet(str(output_file))
                logger.info(f"  ✓ Saved outlet {time_bucket} series: {outlet_ts.shape[0]} periods")
                
                dict_results[time_bucket] = {
                    'overall_periods': overall_ts.shape[0],
                    'outlet_periods': outlet_ts.shape[0],
                    'categories': [col for col in overall_ts.columns if col.endswith('_share')],
                }
            
            # Save metadata
            metadata = {
                'dictionary_name': dict_name,
                'processed_at': datetime.now().isoformat(),
                'results': dict_results,
                'status': 'success'
            }
            
            metadata_file = output_dir / f"{dict_name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            results_summary[dict_name] = metadata
            successful += 1
            logger.info(f"✓ Successfully processed {dict_name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {dict_name}: {e}")
            
            # Save error metadata
            error_metadata = {
                'dictionary_name': dict_name,
                'processed_at': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
            
            metadata_file = output_dir / f"{dict_name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(error_metadata, f, ensure_ascii=False, indent=2)
            
            results_summary[dict_name] = error_metadata
            failed += 1
            continue
    
    # Generate overall summary
    summary = {
        'processing_complete_at': datetime.now().isoformat(),
        'total_dictionaries': len(dict_files),
        'successful': successful,
        'failed': failed,
        'output_directory': str(output_dir),
        'time_buckets_processed': time_buckets,
        'results': results_summary
    }
    
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TIME SERIES COMPUTATION COMPLETE")
    logger.info("="*80)
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
        logger.warning(f"\n⚠️  {failed} dictionaries failed to process. Check individual metadata files for details.")
    
    logger.info("\n✓ All processing complete!")

if __name__ == "__main__":
    main()