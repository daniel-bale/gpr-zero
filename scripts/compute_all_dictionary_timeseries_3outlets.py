#!/usr/bin/env python3
"""
Compute GPR index time series for all dictionaries, LIMITED TO 3 MAIN OUTLETS.
This ensures consistency with category-specific calculations.
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
from ibis import _
import pandas as pd
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target outlets (same as category calculation)
TARGET_OUTLETS = [
    'China Times (中國時報)',
    'Liberty Times (自由時報)', 
    'United Daily (聯合報)'
]

def main():
    """Compute GPR index time series for all dictionaries with outlet filtering."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    dictionaries_dir = project_root / "data" / "dictionaries" / "all_dictionaries"
    output_dir = project_root / "data" / "gpr_timeseries_3outlets"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Database connection
    db_path = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    logger.info(f"Connecting to database: {db_path}")
    conn = ibis.connect(f"duckdb:///{db_path}")
    
    # Use the binary matrix table
    try:
        binary_matrix = conn.table("gpr_binary_matrix_all_words")
        logger.info(f"Connected to binary matrix table")
        logger.info(f"Total rows before filtering: {binary_matrix.count().execute()}")
    except Exception as e:
        logger.error(f"Failed to connect to binary matrix table: {e}")
        return
    
    # Filter to target outlets
    filtered_matrix = binary_matrix.filter(
        binary_matrix.outlet.isin(TARGET_OUTLETS)
    )
    filtered_count = filtered_matrix.count().execute()
    logger.info(f"Rows after filtering to 3 outlets: {filtered_count}")
    
    # Get all dictionary files
    dict_files = list(dictionaries_dir.glob("*.json"))
    logger.info(f"Found {len(dict_files)} dictionary files")
    
    if not dict_files:
        logger.error(f"No dictionary files found in {dictionaries_dir}")
        return
    
    # Load all dictionaries and create word mapping
    all_words = set()
    dictionaries = {}
    
    for dict_file in dict_files:
        with open(dict_file, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            dictionaries[dict_file.stem] = dict_data
            
            if 'word_sets' in dict_data:
                for word_set in dict_data['word_sets'].values():
                    all_words.update(word_set)
    
    # Create word to column mapping
    sorted_words = sorted(all_words)
    word_column_mapping = {}
    for i, word in enumerate(sorted_words):
        col_name = f"contains_word_{i:04d}"
        word_column_mapping[word] = col_name
    
    logger.info(f"Total unique words across all dictionaries: {len(all_words)}")
    
    # Process each dictionary
    successful = 0
    failed = 0
    
    dict_progress = tqdm(sorted(dict_files), desc="Processing dictionaries", unit="dict")
    
    for dict_file in dict_progress:
        dict_name = dict_file.stem
        dict_progress.set_description(f"Processing {dict_name}")
        
        try:
            dictionary = dictionaries[dict_name]
            
            # Build GPR expression for this dictionary
            gpr_expr = ibis.literal(False)
            
            if 'categories' in dictionary:
                # Process each category
                for category_name, category_query in dictionary['categories'].items():
                    # Build category expression using word sets
                    word_set_names = [ws.strip() for ws in category_query.split(" AND ")]
                    category_expr = ibis.literal(True)
                    
                    for ws_name in word_set_names:
                        if ws_name in dictionary.get('word_sets', {}):
                            words = dictionary['word_sets'][ws_name]
                            ws_expr = ibis.literal(False)
                            
                            for word in words:
                                if word in word_column_mapping:
                                    col_name = word_column_mapping[word]
                                    if col_name in filtered_matrix.columns:
                                        ws_expr = ws_expr | filtered_matrix[col_name]
                            
                            category_expr = category_expr & ws_expr
                    
                    # Add to overall GPR expression
                    gpr_expr = gpr_expr | category_expr
            
            # Calculate weekly GPR
            weekly_gpr = (
                filtered_matrix
                .mutate(is_gpr=gpr_expr.cast('int32'))
                .mutate(week=_.warc_date.cast('date').truncate('W'))
                .group_by(['week'])
                .agg(
                    gpr_articles=_.is_gpr.sum(),
                    total_articles=ibis._.count()
                )
                .mutate(gpr_index=_.gpr_articles.cast('float64') / _.total_articles.cast('float64'))
                .order_by('week')
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
    logger.info(f"\n{'='*80}")
    logger.info("TIME SERIES COMPUTATION COMPLETE (3 OUTLETS)")
    logger.info(f"{'='*80}")
    logger.info(f"Target outlets: {', '.join(TARGET_OUTLETS)}")
    logger.info(f"Total dictionaries processed: {len(dict_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save summary
    summary = {
        'processing_complete_at': datetime.now().isoformat(),
        'target_outlets': TARGET_OUTLETS,
        'total_dictionaries': len(dict_files),
        'successful': successful,
        'failed': failed,
        'output_directory': str(output_dir)
    }
    
    summary_file = output_dir / "processing_summary_3outlets.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()