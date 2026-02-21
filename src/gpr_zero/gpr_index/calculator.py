"""GPR Calculator using binary presence matrices for dictionary-based index construction.

Builds binary word-presence matrices from news articles and evaluates
dictionary coverage by matching word sets and categories at query time.
"""

import json
import logging
from pathlib import Path
from functools import reduce
from operator import and_, or_
from typing import Literal

import ibis
import ibis.expr.types as ir

logger = logging.getLogger(__name__)


class GPRCalculator:
    """GPR Calculator using binary word-presence matrices.

    Workflow:
    1. Load dictionaries and collect all unique words
    2. Build binary indicators for each word in articles (``prepare_binary_matrix``)
    3. Match word sets and categories at query time (``query``)
    """

    def __init__(self, dictionary_paths: list[str]):
        """Initialize with multiple dictionary paths to collect all words."""
        self.dictionary_paths = dictionary_paths
        self.all_words: set[str] = set()
        self.dictionaries: dict[str, dict] = {}
        self.labeled_articles_table: ir.Table | None = None
        self.word_column_mapping: dict[str, str] = {}
        self._load_all_dictionaries()

    def _load_all_dictionaries(self):
        """Load all dictionaries and collect unique words."""
        for dict_path in self.dictionary_paths:
            dict_name = Path(dict_path).stem

            try:
                with open(dict_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if (
                    not isinstance(data, dict)
                    or "word_sets" not in data
                    or "categories" not in data
                ):
                    logger.warning(f"Invalid dictionary format: {dict_path}")
                    continue

                self.dictionaries[dict_name] = data

                # Collect all words from this dictionary
                for word_set in data["word_sets"].values():
                    self.all_words.update(word_set)

                logger.info(
                    f"Loaded dictionary {dict_name} with {len(data['word_sets'])} word sets"
                )

            except Exception as e:
                logger.error(f"Failed to load dictionary {dict_path}: {e}")
                continue

        if not self.dictionaries:
            raise ValueError(
                "No valid dictionaries were loaded. Check that dictionary_paths "
                "are correct and each file contains 'word_sets' and 'categories' keys."
            )

        logger.info(
            f"Collected {len(self.all_words)} unique words from {len(self.dictionaries)} dictionaries"
        )

    def _get_table_name(self) -> str:
        """Generate table name for the binary matrix."""
        return "gpr_binary_matrix_all_words"

    def prepare_binary_matrix(
        self,
        articles_table: ir.Table,
        materialize: bool = False,
        force_recalculate: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """
        Prepare articles table with binary presence indicators for all words.

        Args:
            articles_table: Input ibis table with articles
            materialize: Whether to materialize the result table
            force_recalculate: Force recalculation even if materialized exists
            overwrite: Allow overwriting existing materialized table

        Returns:
            Ibis table with binary indicators for all words
        """
        if materialize:
            table_name = self._get_table_name()
            try:
                con = articles_table._find_backend()
                table_exists = table_name in con.list_tables()

                if table_exists and not force_recalculate:
                    logger.info(f"Loading existing materialized table: '{table_name}'")
                    self.labeled_articles_table = con.table(table_name)
                    # Build word_column_mapping to match the cached table's column indices
                    self.word_column_mapping = {}  # TODO: add type hint
                    sorted_words = sorted(self.all_words)
                    for i, word in enumerate(sorted_words):
                        col_name = f"contains_word_{i:04d}"
                        self.word_column_mapping[word] = col_name
                    logger.info(
                        f"Rebuilt word_column_mapping with {len(self.word_column_mapping)} entries"
                    )
                    return self.labeled_articles_table

                if table_exists and force_recalculate:
                    if overwrite:
                        logger.info(
                            f"'{table_name}' exists but force_recalculate is True. It will be overwritten."
                        )
                    else:
                        logger.warning(
                            f"'{table_name}' exists and force_recalculate is True, but overwrite is False."
                        )

            except Exception as e:
                logger.warning(
                    f"Could not check for materialized table. Proceeding with in-memory calculation. Error: {e}"
                )
                materialize = False

        logger.info(
            f"Building binary presence matrix for {len(self.all_words)} words..."
        )

        # Check required columns
        required_columns = ["content", "source_warc_file", "warc_date"]
        if not all(col in articles_table.columns for col in required_columns):
            missing = [
                col for col in required_columns if col not in articles_table.columns
            ]
            raise ValueError(
                f"Input table must contain required columns. Missing: {missing}"
            )

        # Create binary columns for all words in one mutate call
        word_columns = {}
        self.word_column_mapping = {}

        # Sort words for consistent indexing across runs
        sorted_words = sorted(self.all_words)

        for i, word in enumerate(sorted_words):
            # Create column name using index to avoid issues with Chinese characters
            col_name = f"contains_word_{i:04d}"
            word_columns[col_name] = articles_table.content.contains(word)
            self.word_column_mapping[word] = col_name

        # Add all word columns at once
        labeled_table = articles_table.mutate(**word_columns)

        # Materialize if requested
        if materialize:
            table_name = self._get_table_name()
            logger.info(
                f"Materializing table as '{table_name}' (overwrite={overwrite})..."
            )
            try:
                con = articles_table._find_backend()
                con.create_table(table_name, labeled_table, overwrite=overwrite)
                self.labeled_articles_table = con.table(table_name)
                logger.info(f"Successfully materialized table '{table_name}'.")
            except Exception as e:
                logger.error(
                    f"Failed to materialize table '{table_name}'. Returning in-memory expression. Error: {e}"
                )
                self.labeled_articles_table = labeled_table
        else:
            self.labeled_articles_table = labeled_table

        logger.info(
            f"Binary matrix preparation complete. Table has {len(labeled_table.columns)} columns"
        )
        logger.info(f"Word columns: {len(self.word_column_mapping)}")

        return self.labeled_articles_table

    # TODO: refactor query() — extract time bucketing, word-set expression building,
    # category expression building, aggregation construction, and query execution
    # into separate private methods (currently ~200 lines, target <50 each).
    def query(
        self,
        dictionary_name: str,
        group_by_cols: list[str],
        time_bucket: str | None = None,
        time_col: str = "warc_date",
    ):
        """
        Query GPR data for a specific dictionary, dynamically building word sets and categories.

        Args:
            dictionary_name: Name of dictionary to use for word sets and categories
            group_by_cols: List of column names to group by
            time_bucket: Optional time unit for bucketing ('W', 'M', 'Y', etc.)
            time_col: The name of the date/time column used for bucketing

        Returns:
            Polars DataFrame with aggregated GPR results (wide format)
        """
        if self.labeled_articles_table is None:
            raise RuntimeError("Must call prepare_binary_matrix() first")

        if dictionary_name not in self.dictionaries:
            raise ValueError(f"Dictionary '{dictionary_name}' not found")

        dictionary = self.dictionaries[dictionary_name]
        word_sets = dictionary["word_sets"]
        categories = dictionary["categories"]

        query_table = self.labeled_articles_table
        final_group_by_cols = list(group_by_cols)

        logger.info(
            f"Querying with dictionary '{dictionary_name}': {len(word_sets)} word sets, {len(categories)} categories"
        )

        # Time bucketing logic
        if time_bucket:
            if time_col not in query_table.columns:
                raise ValueError(
                    f"Time column '{time_col}' not found in prepared table."
                )

            bucketed_col_name = f"{time_col}_bucketed"
            # TODO: fix the type hint
            ibis_time_unit = time_bucket.lower()
            if ibis_time_unit in ["w", "week"]:
                ibis_time_unit = "W"
            elif ibis_time_unit in ["m", "month"]:
                ibis_time_unit = "M"
            elif ibis_time_unit in ["y", "year"]:
                ibis_time_unit = "Y"

            logger.info(
                f"Applying time bucketing: '{time_col}' truncated to '{ibis_time_unit}' as '{bucketed_col_name}'"
            )
            # Convert string date to date type if needed, then truncate
            query_table = query_table.mutate(
                **{
                    bucketed_col_name: query_table[time_col]
                    .cast(
                        "date"
                    )  # TODO: time_cols should be make sure to be in date format from table creation
                    .truncate(ibis_time_unit)
                }
            )

            # Replace time_col with bucketed_col_name in final_group_by_cols, or add if not present
            if time_col in final_group_by_cols:
                final_group_by_cols = [
                    bucketed_col_name if col == time_col else col
                    for col in final_group_by_cols
                ]
            else:
                # Add bucketed time column if time_bucket is specified but time_col not in group_by_cols
                final_group_by_cols.append(bucketed_col_name)
                logger.info(
                    f"Added '{bucketed_col_name}' to grouping for time bucketing"
                )

            logger.info(
                f"Updated group_by columns for bucketing: {final_group_by_cols}"
            )

        # Build word set expressions using reduce
        word_set_expressions = {}
        for set_name, words in word_sets.items():
            # Get columns for words that exist in this word set
            relevant_word_columns = [
                self.word_column_mapping.get(word)
                for word in words
                if word in self.word_column_mapping
            ]

            if relevant_word_columns:
                # Use reduce with OR operator: any word in the set matches
                word_columns_exprs = [query_table[col] for col in relevant_word_columns]
                word_set_expr = reduce(or_, word_columns_exprs)
                word_set_expressions[f"has_{set_name}"] = word_set_expr
            else:
                word_set_expressions[f"has_{set_name}"] = ibis.literal(False)

        # Add word set columns to query table
        query_table = query_table.mutate(**word_set_expressions)

        # Build category expressions using reduce
        category_expressions = {}
        all_category_match_expressions = []

        for category_name, query_string in categories.items():
            # Parse category definition (word_set_A AND word_set_B)
            # TODO: This parsing is so bad, need to implement a proper parser for more complex expressions in the future
            components = [comp.strip() for comp in query_string.split(" AND ")]
            required_list_names = []
            for component in components:
                sub_components = [name.strip() for name in component.split("+")]
                for sub_comp_name in sub_components:
                    actual_list_name = sub_comp_name.split("N/")[0].strip()
                    if actual_list_name:
                        required_list_names.append(actual_list_name)

            if not required_list_names:
                logger.warning(
                    f"No valid word set names found for category '{category_name}'. Skipping."
                )
                category_expr = ibis.literal(False)
            else:
                # Get word set expressions needed for this category
                has_columns = [
                    query_table[f"has_{list_name}"] for list_name in required_list_names
                ]
                # Use reduce with AND operator: all word sets must match
                category_expr = reduce(and_, has_columns, ibis.literal(True)).fill_null(
                    False
                )

            category_expressions[f"is_{category_name}"] = category_expr
            all_category_match_expressions.append(category_expr)

        # Add category columns to query table
        query_table = query_table.mutate(**category_expressions)

        # Add overall match indicator
        if all_category_match_expressions:
            matches_any_category_expr = reduce(
                or_, all_category_match_expressions
            ).fill_null(False)
        else:
            matches_any_category_expr = ibis.literal(False)

        query_table = query_table.mutate(matches_any_category=matches_any_category_expr)

        # Build aggregations - always compute both counts and shares
        aggregations = {}

        # Total article count
        total_count_agg = query_table.count().name("total_articles")
        aggregations["total_articles"] = total_count_agg
        total_count_safe = (total_count_agg + 1e-9).cast("float64")

        # Overall GPR index
        overall_count_agg = query_table.matches_any_category.sum().cast("int64")
        aggregations["total_gpr_count"] = overall_count_agg
        aggregations["total_gpr_share"] = overall_count_agg / total_count_safe

        # Category counts and shares
        category_count_aggs = {}
        category_share_aggs = {}
        for category_name in categories.keys():
            is_col_name = f"is_{category_name}"
            if is_col_name in query_table.columns:
                count_agg = getattr(query_table, is_col_name).sum().cast("int64")
                category_count_aggs[f"{category_name}_count"] = count_agg
                category_share_aggs[f"{category_name}_share"] = (
                    count_agg / total_count_safe
                )

        # Add all counts and shares
        aggregations.update(category_count_aggs)
        aggregations.update(category_share_aggs)

        # Perform aggregation
        if not final_group_by_cols:
            # TODO: raise earlier
            raise ValueError(
                "final_group_by_cols cannot be empty. Must specify at least time grouping or other grouping columns."
            )

        grouped_table = query_table.group_by(final_group_by_cols)
        result_expr = grouped_table.aggregate(**aggregations)

        # Sort by group keys
        sort_expressions = [ibis.asc(col) for col in final_group_by_cols]
        result_expr = result_expr.order_by(sort_expressions)

        # Execute query
        logger.info("Executing aggregation query...")
        try:
            result_df = result_expr.to_polars()
            logger.info(
                f"Query successful. Result shape: {result_df.shape}, Columns: {result_df.columns}"
            )
            return result_df

        except Exception as e:
            logger.error(f"Error executing ibis query: {e}")
            raise
