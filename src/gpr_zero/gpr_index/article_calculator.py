"""Calculator for the Geopolitical Risk (GPR) index based on article counting or shares.

Handles labeling articles based on keywords and category definitions defined in a
structured dictionary, and provides a flexible query interface for aggregating results.

Preserves the thesis article-labeling semantics: case-insensitive literal regex
matching and pre-assigned categories for LT- RTF articles. Keep this separate from
the binary-matrix calculator, whose matching and input contracts differ.
"""

import json
import logging
import re
from pathlib import Path

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import polars as pl

from gpr_zero.gpr_index.dictionary import DictionaryMatcher

logger = logging.getLogger(__name__)


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
        self.word_sets: dict[str, list[str]] = {}
        self.category_definitions: dict[str, str] = {}
        self._load_structured_dictionary(dictionary_path)
        self.labeled_articles_table: ir.Table | None = None
        logger.info(f"GPRCalculator initialized with dictionary: {dictionary_path}")
        logger.info(
            f"Found {len(self.word_sets)} word sets and {len(self.category_definitions)} category definitions."
        )

    def _load_structured_dictionary(self, dictionary_path: str):
        """Loads the structured GPR dictionary (word_sets and categories)."""
        path = Path(dictionary_path)
        if not path.exists():
            logger.error(f"GPR dictionary file not found: {dictionary_path}")
            raise FileNotFoundError(f"GPR dictionary file not found: {dictionary_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if (
                not isinstance(data, dict)
                or "word_sets" not in data
                or "categories" not in data
            ):
                raise ValueError(
                    "Dictionary must be a JSON object with 'word_sets' and 'categories' keys."
                )

            self.word_sets = data["word_sets"]
            self.category_definitions = data["categories"]

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {dictionary_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading GPR dictionary {dictionary_path}: {e}")
            raise

    def _get_labeled_table_name(self) -> str:
        """Generates a sanitized table name from the dictionary path."""
        dict_name = Path(self.dictionary_path).stem
        # Sanitize the name to be a valid SQL identifier (replace non-word chars with _)
        sanitized_name = re.sub(r"\W", "_", dict_name)
        return f"gpr_labeled_{sanitized_name}"

    def prepare(
        self,
        articles_table: ir.Table,
        materialize: bool = False,
        force_recalculate: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
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
                        logger.info(
                            f"'{table_name}' exists but force_recalculate is True. It will be overwritten."
                        )
                    else:
                        logger.warning(
                            f"'{table_name}' exists and force_recalculate is True, but overwrite is False. The result will be calculated but not materialized to avoid an error."
                        )

            except Exception as e:  # noqa: BLE001 -- Preserve thesis cache-check fallback.
                logger.warning(
                    f"Could not check for materialized table. Proceeding with in-memory calculation. Error: {e}"
                )
                materialize = (
                    False  # Fallback to not materializing if backend check fails
                )

        logger.info("Preparing article table for GPR calculation...")
        required_columns = ["content", "source_file", "category", "date"]
        if not all(col in articles_table.columns for col in required_columns):
            missing = [
                col for col in required_columns if col not in articles_table.columns
            ]
            raise ValueError(
                f"Input table must contain required columns. Missing: {missing}"
            )

        # Ensure date column is timestamp for potential truncation
        col_type = articles_table["date"].type()
        if not isinstance(col_type, (dt.Date, dt.Timestamp)):
            logger.warning(
                f"Casting 'date' column (type: {col_type}) to timestamp for compatibility."
            )
            articles_table = articles_table.mutate(
                date=articles_table["date"].cast("timestamp")
            )

        labeled_table = DictionaryMatcher(
            {"word_sets": self.word_sets, "categories": self.category_definitions}
        ).label_articles(articles_table)

        if materialize:
            table_name = self._get_labeled_table_name()
            logger.info(
                f"Attempting to materialize labeled table as '{table_name}' (overwrite={overwrite})..."
            )
            try:
                con = articles_table._find_backend()
                # Use overwrite=True. It handles creation and replacement.
                con.create_table(table_name, labeled_table, overwrite=overwrite)
                self.labeled_articles_table = con.table(table_name)
                logger.info(
                    f"Successfully materialized and loaded table '{table_name}'."
                )
            except Exception as e:  # noqa: BLE001 -- Preserve thesis materialization fallback.
                logger.error(
                    f"Failed to materialize table '{table_name}'. Returning in-memory expression. Error: {e}"
                )
                self.labeled_articles_table = labeled_table  # Fallback
        else:
            self.labeled_articles_table = labeled_table

        logger.info(
            f"Article table preparation complete. Labeled table schema: {list(self.labeled_articles_table.columns)}"
        )
        return self.labeled_articles_table

    def query(
        self,
        group_by_cols: list[str],
        time_bucket: str | None = None,
        time_col: str = "date",
        calculate_share: bool = False,
    ) -> pl.DataFrame:
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
        final_group_by_cols = list(group_by_cols)  # Make a copy

        query_type = "shares" if calculate_share else "counts"
        logger.info(
            f"Querying GPR data ({query_type}): Group by {group_by_cols}, Time bucket: {time_bucket} on '{time_col}'"
        )

        # Apply the historical article time-bucket aliases.
        bucketed_col_name = time_col  # Default if no bucketing
        if time_bucket:
            if time_col not in query_table.columns:
                raise ValueError(
                    f"Time column '{time_col}' not found in prepared table."
                )
            if time_col not in group_by_cols:
                logger.warning(
                    f"Time column '{time_col}' added implicitly to grouping for bucketing."
                )
                final_group_by_cols.append(time_col)

            time_col_expr = query_table[time_col]
            if not isinstance(time_col_expr.type(), (dt.Date, dt.Timestamp)):
                raise TypeError(
                    f"Time column '{time_col}' (type: {time_col_expr.type()}) must be a Date or Timestamp type for bucketing."
                )

            bucketed_col_name = f"{time_col}_bucketed"
            ibis_time_unit = time_bucket.lower()
            if ibis_time_unit in ["w", "week"]:
                ibis_time_unit = "W"  # DuckDB uses 'W' for ISO week
            elif ibis_time_unit in ["m", "month"]:
                ibis_time_unit = "month"
            elif ibis_time_unit in ["y", "year"]:
                ibis_time_unit = "year"
            else:
                logger.warning(f"Unknown time bucket unit: {time_bucket}. Using as is.")

            logger.info(
                f"Applying time bucketing: '{time_col}' truncated to '{ibis_time_unit}' as '{bucketed_col_name}'"
            )
            query_table = query_table.mutate(
                **{bucketed_col_name: query_table[time_col].truncate(ibis_time_unit)}
            )
            final_group_by_cols = [
                bucketed_col_name if col == time_col else col
                for col in final_group_by_cols
            ]
            logger.info(
                f"Updated group_by columns for bucketing: {final_group_by_cols}"
            )

        # Keep thesis count names, column ordering and epsilon denominators.
        aggregations = {}
        count_suffix = "_count"
        share_suffix = "_share"

        # Define count aggregations first
        overall_count_agg = query_table.matches_any_category.sum().cast(
            "int64"
        )  # Ensure integer counts
        category_count_aggs = {}
        for category_name in self.category_definitions:
            is_col_name = f"is_{category_name}"
            if is_col_name in query_table.columns:
                category_count_aggs[f"{category_name}{count_suffix}"] = (
                    getattr(query_table, is_col_name).sum().cast("int64")
                )
            else:
                logger.warning(
                    f"Category column '{is_col_name}' not found during aggregation. Skipping count."
                )

        # Add unique date count when time bucketing is used
        if time_bucket:
            unique_date_count_agg = (
                query_table[time_col].nunique().name("unique_dates_in_bucket")
            )
            aggregations["unique_dates_in_bucket"] = unique_date_count_agg

        if not calculate_share:
            # Aggregate counts
            aggregations["overall_index"] = overall_count_agg
            aggregations.update(category_count_aggs)
        else:
            # Aggregate counts and total, then calculate shares
            total_count_agg = query_table.count().name("total_articles_in_group")
            # Add a small epsilon to prevent division by zero
            total_count_safe = (total_count_agg + 1e-9).cast("float64")

            aggregations["total_articles_in_group"] = (
                total_count_agg  # Include total count in output
            )
            aggregations[f"overall{share_suffix}"] = (
                overall_count_agg / total_count_safe
            )

            for name, count_agg in category_count_aggs.items():
                base_name = name.replace(count_suffix, "")
                aggregations[f"{base_name}{share_suffix}"] = (
                    count_agg / total_count_safe
                )
        # --- End Aggregation Definitions ---

        # --- Perform Grouping and Aggregation ---
        if not final_group_by_cols:
            logger.info(
                "No grouping columns specified. Aggregating over the entire table."
            )
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
            logger.info(
                f"Query successful. Result shape: {result_df.shape}, Columns: {result_df.columns}"
            )
            logger.debug(
                f"Result head:\n{result_df.head().to_pandas()}"
            )  # Log pandas for readability
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
