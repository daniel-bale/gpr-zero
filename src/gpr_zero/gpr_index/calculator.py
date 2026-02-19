"""Calculator for the Geopolitical Risk (GPR) index based on article counting or shares.

Handles labeling articles based on keywords and category definitions defined in a
structured dictionary, and provides a flexible query interface for aggregating results.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import polars as pl # Use Polars for return type

# Import only necessary ibis functions if needed, e.g. from ibis import _
from ibis import _

logger = logging.getLogger(__name__)


# Renamed from CustomGPRCalculator to GPRCalculator
class GPRCalculator:
    """
    Calculates Geopolitical Risk (GPR) indices using an ibis-based approach.

    Handles labeling articles based on keywords and category definitions,
    and provides a flexible query interface for aggregating results either as
    raw counts or as shares of total articles within a group.
    """
    def __init__(self, dictionary_path: str):
        """
        Initializes the calculator by loading the GPR dictionary.

        Args:
            dictionary_path: Path to the JSON GPR dictionary file.
                             Expected format: {"word_sets": {...}, "categories": {...}}
        """
        self.dictionary_path = dictionary_path
        self.word_sets: Dict[str, List[str]] = {}
        self.category_definitions: Dict[str, str] = {}
        self._load_structured_dictionary(dictionary_path)
        self.labeled_articles_table: Optional[ir.Table] = None # Keep ibis.Table type hint consistent
        logger.info(f"GPRCalculator initialized with dictionary: {dictionary_path}")
        logger.info(f"Found {len(self.word_sets)} word sets and {len(self.category_definitions)} category definitions.")

    def _load_structured_dictionary(self, dictionary_path: str):
        """Loads the structured GPR dictionary (word_sets and categories)."""
        path = Path(dictionary_path)
        if not path.exists():
            logger.error(f"GPR dictionary file not found: {dictionary_path}")
            raise FileNotFoundError(f"GPR dictionary file not found: {dictionary_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict) or "word_sets" not in data or "categories" not in data:
                raise ValueError("Dictionary must be a JSON object with 'word_sets' and 'categories' keys.")

            self.word_sets = data["word_sets"]
            self.category_definitions = data["categories"]

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {dictionary_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading GPR dictionary {dictionary_path}: {e}")
            raise

    def _build_regex_pattern(self, word_list: List[str]) -> str:
        """Builds a case-insensitive regex pattern for a list of words."""
        if not word_list:
            return r"a^" # Regex that never matches
        # Ensure words are treated literally, handle potential regex special chars within words
        escaped_words = [re.escape(word) for word in word_list if word]
        if not escaped_words:
            return r"a^"
        # Case-insensitive multi-word pattern using ibis/SQL compatible regex features
        return f"(?i)({'|'.join(escaped_words)})"

    def _get_labeled_table_name(self) -> str:
        """Generates a sanitized table name from the dictionary path."""
        dict_name = Path(self.dictionary_path).stem
        # Sanitize the name to be a valid SQL identifier (replace non-word chars with _)
        sanitized_name = re.sub(r'\W', '_', dict_name)
        return f"gpr_labeled_{sanitized_name}"

    def prepare(self,
                articles_table: ir.Table,
                materialize: bool = False,
                force_recalculate: bool = False,
                overwrite: bool = False) -> ir.Table:
        """
        Labels the input ibis table with GPR categories based on content and metadata.

        This version adds options to materialize the prepared table in the database
        for performance and reuse, and to force recalculation if needed.

        Args:
            articles_table: An ibis table containing article data. Must include
                            'content', 'source_file', 'category', and 'date' columns.
            materialize: If True, the labeled table will be saved as a new table
                         in the database. The table name is derived from the
                         GPR dictionary filename. Defaults to False.
            force_recalculate: If True, forces recalculation even if a materialized
                               table exists. Defaults to False.
            overwrite: If True, allows overwriting an existing materialized table.
                       Used in conjunction with `materialize=True`. If False,
                       an error will be raised on attempting to overwrite.
                       Defaults to False.

        Returns:
            An ibis table expression with added boolean columns for each word set
            (has_SETNAME), each category (is_CATEGORYNAME), and an overall match
            indicator (matches_any_category). If `materialize` is True, this
            will be a reference to the newly created/loaded table in the database.
        """
        if materialize:
            table_name = self._get_labeled_table_name()
            try:
                con = articles_table._find_backend()
                table_exists = table_name in con.list_tables()

                if table_exists and not force_recalculate:
                    logger.info(f"Loading existing materialized table: '{table_name}'")
                    self.labeled_articles_table = con.table(table_name)
                    return self.labeled_articles_table

                if table_exists and force_recalculate:
                    if overwrite:
                        logger.info(f"'{table_name}' exists but force_recalculate is True. It will be overwritten.")
                    else:
                        logger.warning(f"'{table_name}' exists and force_recalculate is True, but overwrite is False. The result will be calculated but not materialized to avoid an error.")

            except Exception as e:
                logger.warning(f"Could not check for materialized table. Proceeding with in-memory calculation. Error: {e}")
                materialize = False # Fallback to not materializing if backend check fails

        logger.info("Preparing article table for GPR calculation...")
        required_columns = ['content', 'source_file', 'category', 'date']
        if not all(col in articles_table.columns for col in required_columns):
             missing = [col for col in required_columns if col not in articles_table.columns]
             raise ValueError(f"Input table must contain required columns. Missing: {missing}")

        # Ensure date column is timestamp for potential truncation
        col_type = articles_table['date'].type()
        if not isinstance(col_type, (dt.Date, dt.Timestamp)):
             logger.warning(f"Casting 'date' column (type: {col_type}) to timestamp for compatibility.")
             articles_table = articles_table.mutate(date=articles_table['date'].cast('timestamp'))

        # Add necessary columns using mutate
        labeled_table = articles_table.mutate(
            content_lower=_.content.lower(),
            is_rtf=_.source_file.startswith('LT-')
        )

        # 1. Add boolean columns for each word set
        word_set_columns = {}
        for set_name, words in self.word_sets.items():
            col_name = f"has_{set_name}"
            pattern = self._build_regex_pattern(words)
            word_set_columns[col_name] = labeled_table.content_lower.re_search(pattern).fill_null(False)
            logger.debug(f"Adding column '{col_name}' for word set '{set_name}'")

        labeled_table = labeled_table.mutate(**word_set_columns)

        # 2. Add boolean columns for each category
        category_columns = {}
        all_category_match_expressions = []

        for category_name, query_string in self.category_definitions.items():
            col_name = f"is_{category_name}"
            
            # Revised parsing for category definitions to handle " AND " and then "N/"
            # Example query_string: "word_set_A AND word_set_B"
            # Or, if ProQuest was mixed (though not current use case): "word_set_A N/2 AND word_set_B"
            
            # Split by " AND " first
            components = [comp.strip() for comp in query_string.split(" AND ")]
            
            # Then, for each component, handle potential "N/" (ProQuest-like syntax)
            # and strip whitespace to get the actual word set name.
            # This also handles the case where the original category was just "+" separated,
            # assuming those are now represented as single components after an " AND " split
            # or if " AND " was not present at all.
            required_list_names = []
            for component in components:
                # If "+" is still present within a component (e.g. from an older format not yet updated),
                # split by "+" as a fallback for that component.
                # This makes it robust to "set_A + set_B AND set_C" if such a mixed case arises,
                # though ideally dictionaries should consistently use "AND".
                sub_components = [name.strip() for name in component.split("+")]
                for sub_comp_name in sub_components:
                    # Handle N/ operator part for each sub-component
                    actual_list_name = sub_comp_name.split("N/")[0].strip()
                    if actual_list_name: # Ensure not empty string
                        required_list_names.append(actual_list_name)
            
            if not required_list_names:
                logger.warning(f"No valid word set names found in query_string: '{query_string}' for category '{category_name}'. Skipping.")
                category_columns[col_name] = ibis.literal(False)
                all_category_match_expressions.append(ibis.literal(False))
                continue

            content_match_expr = None
            valid_definition = True
            for list_name in required_list_names:
                has_col_name = f"has_{list_name}"
                if has_col_name not in labeled_table.columns:
                    logger.warning(f"Word set column '{has_col_name}' needed for category '{category_name}' not found. Skipping category.")
                    valid_definition = False
                    break
                current_expr = getattr(labeled_table, has_col_name)
                if content_match_expr is None:
                    content_match_expr = current_expr
                else:
                    # Use boolean AND for joint occurrence
                    content_match_expr &= current_expr

            if not valid_definition or content_match_expr is None:
                 logger.warning(f"Could not build content match expression for category '{category_name}'. Assigning False.")
                 category_expr = ibis.literal(False) # Assign False if definition is invalid
            else:
                # Final expression considers RTF pre-assigned category
                category_expr = ibis.cases(
                    (labeled_table.is_rtf, labeled_table.category == category_name),
                    else_=content_match_expr # Use else_ keyword
                ).fill_null(False) # Use fill_null

            category_columns[col_name] = category_expr
            all_category_match_expressions.append(category_expr)
            logger.debug(f"Adding column '{col_name}' for category '{category_name}'")

        labeled_table = labeled_table.mutate(**category_columns)

        # 3. Add 'matches_any_category' column
        if not all_category_match_expressions:
            logger.warning("No valid category expressions generated. 'matches_any_category' will be False.")
            matches_any_category_expr = ibis.literal(False)
        else:
            # Use boolean OR to combine individual category matches
            matches_any_category_expr = all_category_match_expressions[0]
            for expr in all_category_match_expressions[1:]:
                matches_any_category_expr |= expr
            # Ensure final boolean result
            matches_any_category_expr = matches_any_category_expr.fill_null(False)


        labeled_table = labeled_table.mutate(matches_any_category=matches_any_category_expr)

        if materialize:
            table_name = self._get_labeled_table_name()
            logger.info(f"Attempting to materialize labeled table as '{table_name}' (overwrite={overwrite})...")
            try:
                con = articles_table._find_backend()
                # Use overwrite=True. It handles creation and replacement.
                con.create_table(table_name, labeled_table, overwrite=overwrite)
                self.labeled_articles_table = con.table(table_name)
                logger.info(f"Successfully materialized and loaded table '{table_name}'.")
            except Exception as e:
                logger.error(f"Failed to materialize table '{table_name}'. Returning in-memory expression. Error: {e}")
                self.labeled_articles_table = labeled_table # Fallback
        else:
            self.labeled_articles_table = labeled_table

        logger.info(f"Article table preparation complete. Labeled table schema: {list(self.labeled_articles_table.columns)}")
        return self.labeled_articles_table

    def query(self,
              group_by_cols: List[str],
              time_bucket: Optional[str] = None,
              time_col: str = 'date',
              calculate_share: bool = False) -> pl.DataFrame:
        """
        Queries the prepared GPR data, aggregating results by specified groups and time buckets.

        Can return either raw counts or shares (proportions) of articles.

        Args:
            group_by_cols: List of column names to group by (e.g., ['date'], ['date', 'newspaper']).
                           Must include `time_col` if `time_bucket` is used.
            time_bucket: Optional time unit for bucketing (e.g., 'W', 'M', 'Y', 'week', 'month', 'year').
                         Requires `time_col` to be a date/timestamp column.
            time_col: The name of the date/time column used for bucketing (default: 'date').
            calculate_share: If True, calculate the share of articles matching each category
                             (count / total articles in group). Defaults to False (raw counts).

        Returns:
            A Polars DataFrame with aggregated GPR results.
            If `calculate_share` is False: columns 'overall_index', 'CATEGORYNAME_count'.
            If `calculate_share` is True: columns 'overall_share', 'CATEGORYNAME_share', 'total_articles_in_group'.
        """
        if self.labeled_articles_table is None:
            raise RuntimeError("The 'prepare' method must be called before 'query'.")

        query_table = self.labeled_articles_table
        final_group_by_cols = list(group_by_cols) # Make a copy

        query_type = "shares" if calculate_share else "counts"
        logger.info(f"Querying GPR data ({query_type}): Group by {group_by_cols}, Time bucket: {time_bucket} on '{time_col}'")

        # --- Time Bucketing Logic (remains the same) ---
        bucketed_col_name = time_col # Default if no bucketing
        if time_bucket:
            if time_col not in query_table.columns:
                raise ValueError(f"Time column '{time_col}' not found in prepared table.")
            if time_col not in group_by_cols:
                 logger.warning(f"Time column '{time_col}' added implicitly to grouping for bucketing.")
                 final_group_by_cols.append(time_col)

            time_col_expr = query_table[time_col]
            if not isinstance(time_col_expr.type(), (dt.Date, dt.Timestamp)):
                 raise TypeError(f"Time column '{time_col}' (type: {time_col_expr.type()}) must be a Date or Timestamp type for bucketing.")

            bucketed_col_name = f"{time_col}_bucketed"
            ibis_time_unit = time_bucket.lower()
            if ibis_time_unit in ['w', 'week']: ibis_time_unit = 'W' # DuckDB uses 'W' for ISO week
            elif ibis_time_unit in ['m', 'month']: ibis_time_unit = 'month'
            elif ibis_time_unit in ['y', 'year']: ibis_time_unit = 'year'
            else: logger.warning(f"Unknown time bucket unit: {time_bucket}. Using as is.")

            logger.info(f"Applying time bucketing: '{time_col}' truncated to '{ibis_time_unit}' as '{bucketed_col_name}'")
            query_table = query_table.mutate(
                **{bucketed_col_name: query_table[time_col].truncate(ibis_time_unit)}
            )
            final_group_by_cols = [bucketed_col_name if col == time_col else col for col in final_group_by_cols]
            logger.info(f"Updated group_by columns for bucketing: {final_group_by_cols}")
        # --- End Time Bucketing ---

        # --- Aggregation Definitions ---
        aggregations = {}
        count_suffix = "_count"
        share_suffix = "_share"

        # Define count aggregations first
        overall_count_agg = query_table.matches_any_category.sum().cast('int64') # Ensure integer counts
        category_count_aggs = {}
        for category_name in self.category_definitions.keys():
            is_col_name = f"is_{category_name}"
            if is_col_name in query_table.columns:
                category_count_aggs[f"{category_name}{count_suffix}"] = getattr(query_table, is_col_name).sum().cast('int64')
            else:
                 logger.warning(f"Category column '{is_col_name}' not found during aggregation. Skipping count.")

        # Add unique date count when time bucketing is used
        if time_bucket:
            unique_date_count_agg = query_table[time_col].nunique().name("unique_dates_in_bucket")
            aggregations["unique_dates_in_bucket"] = unique_date_count_agg

        if not calculate_share:
            # Aggregate counts
            aggregations["overall_index"] = overall_count_agg
            aggregations.update(category_count_aggs)
        else:
            # Aggregate counts and total, then calculate shares
            total_count_agg = query_table.count().name("total_articles_in_group")
            # Add a small epsilon to prevent division by zero
            total_count_safe = (total_count_agg + 1e-9).cast('float64')

            aggregations["total_articles_in_group"] = total_count_agg # Include total count in output
            aggregations[f"overall{share_suffix}"] = (overall_count_agg / total_count_safe)

            for name, count_agg in category_count_aggs.items():
                 base_name = name.replace(count_suffix, "")
                 aggregations[f"{base_name}{share_suffix}"] = (count_agg / total_count_safe)
        # --- End Aggregation Definitions ---

        # --- Perform Grouping and Aggregation ---
        if not final_group_by_cols:
             logger.info("No grouping columns specified. Aggregating over the entire table.")
             result_expr = query_table.aggregate(**aggregations)
        else:
            logger.info(f"Grouping by: {final_group_by_cols}")
            grouped_table = query_table.group_by(final_group_by_cols)
            result_expr = grouped_table.aggregate(**aggregations)

        # Ensure sorting by group keys for predictable output
        if final_group_by_cols:
             # Use a list of ibis.asc() expressions for sorting
             sort_expressions = [ibis.asc(col) for col in final_group_by_cols]
             result_expr = result_expr.order_by(sort_expressions)
        # --- End Grouping and Aggregation ---

        # Execute query
        logger.info("Executing aggregation query...")
        try:
            result_df = result_expr.to_polars()
            logger.info(f"Query successful. Result shape: {result_df.shape}, Columns: {result_df.columns}")
            logger.debug(f"Result head:\n{result_df.head().to_pandas()}") # Log pandas for readability
            return result_df
        except Exception as e:
            logger.error(f"Error executing ibis query: {e}")
            # Consider logging the compiled SQL query for debugging
            # try:
            #     logger.error(f"Compiled SQL:\n{ibis.to_sql(result_expr)}")
            # except Exception as sql_e:
            #     logger.error(f"Could not compile query to SQL: {sql_e}")
            raise

# Example Usage (for documentation/testing - not run directly here)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     # Example assuming a duckdb database exists
#     db_path = "path/to/your/news.duckdb"
#     dict_path = "path/to/your/gpr_dict.json"
#     conn = ibis.connect(f"duckdb:///{db_path}")
#     articles = conn.table("articles")
#     calculator = GPRCalculator(dict_path)
#     calculator.prepare(articles)
#     daily_counts = calculator.query(group_by_cols=['date'])
#     print("\nDaily Counts:")
#     print(daily_counts.head())
#     weekly_shares = calculator.query(group_by_cols=['date', 'newspaper_en'], time_bucket='W', calculate_share=True)
#     print("\nWeekly Shares by Newspaper:")
#     print(weekly_shares.head())
