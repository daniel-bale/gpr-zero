"""GPR Calculator using binary presence matrices for dictionary-based index construction.

Builds binary word-presence matrices from news articles and evaluates
dictionary coverage by matching word sets and categories at query time.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import ibis
import ibis.expr.types as ir

from gpr_zero.gpr_index.dictionary import DictionaryMatcher

if TYPE_CHECKING:
    from gpr_zero.gpr_index.corpus import DictionaryInventory

logger = logging.getLogger(__name__)


class GPRCalculator:
    """GPR Calculator using binary word-presence matrices.

    Workflow:
    1. Load dictionaries and collect all unique words
    2. Build binary indicators for each word in articles (``prepare_binary_matrix``)
    3. Match word sets and categories at query time (``query``)
    """

    def __init__(self, dictionary_paths: "list[str] | DictionaryInventory"):
        """Initialize with multiple dictionary paths to collect all words."""
        from gpr_zero.gpr_index.corpus import DictionaryInventory

        self.all_words: set[str] = set()
        self.dictionaries: dict[str, dict] = {}
        self.labeled_articles_table: ir.Table | None = None
        self.word_column_mapping: dict[str, str] = {}
        if isinstance(dictionary_paths, DictionaryInventory):
            self.dictionary_paths = list(dictionary_paths.dictionary_paths)
            self.dictionaries = dict(dictionary_paths.dictionaries)
            self.all_words = set(dictionary_paths.words)
            self.word_column_mapping = dict(dictionary_paths.word_columns)
        else:
            self.dictionary_paths = dictionary_paths
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

            except Exception as e:  # noqa: BLE001 -- Thesis skips every failed dictionary load.
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

            except Exception as e:  # noqa: BLE001 -- Preserve thesis cache-check fallback.
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
            except Exception as e:  # noqa: BLE001 -- Preserve thesis materialization fallback.
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

    def query(
        self,
        dictionary_name: str,
        group_by_cols: list[str],
        time_bucket: str | None = None,
        time_col: str = "warc_date",
        output_format: Literal["wide", "long"] = "wide",
    ):
        """
        Query GPR data for a specific dictionary, dynamically building word sets and categories.

        Args:
            dictionary_name: Name of dictionary to use for word sets and categories
            group_by_cols: List of column names to group by
            time_bucket: Optional time unit for bucketing ('W', 'M', 'Y', etc.)
            time_col: The name of the date/time column used for bucketing
            output_format: Return counts and shares in 'wide' format, or
                non-zero category and overall counts in 'long' format.

        Returns:
            Polars DataFrame with aggregated GPR results in the requested format
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

        query_table = DictionaryMatcher(dictionary).label_binary(
            query_table, self.word_column_mapping
        )

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
        for category_name in categories:
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
            if output_format == "wide":
                return result_df
            if output_format == "long":
                return self._reshape_to_long_format(result_df, final_group_by_cols)
            raise ValueError(
                f"Unknown output_format: {output_format}. Use 'wide' or 'long'."
            )

        except Exception as e:
            logger.error(f"Error executing ibis query: {e}")
            raise

    def _reshape_to_long_format(self, wide_df, group_by_cols: list[str]):
        """Return the thesis long-format counts, omitting zero-count records."""
        import polars as pl

        base_cols = list(group_by_cols)
        if "date_bucketed" in wide_df.columns:
            base_cols.append("date_bucketed")
        if "total_articles" in wide_df.columns:
            base_cols.append("total_articles")

        count_columns = [col for col in wide_df.columns if col.endswith("_count")]
        category_count_cols = [
            col for col in count_columns if not col.startswith("total_")
        ]
        long_dfs = []
        for col in category_count_cols:
            cond_name = col.removesuffix("_count")
            df_subset = wide_df.select(base_cols + [col]).rename({col: "count"})
            df_subset = df_subset.with_columns(
                [
                    pl.lit("category").alias("cond_type"),
                    pl.lit(cond_name).alias("cond_name"),
                ]
            )
            long_dfs.append(df_subset)

        if "total_gpr_count" in wide_df.columns:
            df_subset = wide_df.select(base_cols + ["total_gpr_count"]).rename(
                {"total_gpr_count": "count"}
            )
            df_subset = df_subset.with_columns(
                [
                    pl.lit("overall").alias("cond_type"),
                    pl.lit("total_gpr").alias("cond_name"),
                ]
            )
            long_dfs.append(df_subset)

        if long_dfs:
            result_df = pl.concat(long_dfs, how="vertical")
            final_cols = base_cols + ["cond_type", "cond_name", "count"]
            result_df = result_df.select(
                [col for col in final_cols if col in result_df.columns]
            )
            result_df = result_df.filter(pl.col("count") > 0)
            logger.info(
                "Reshaped to long format: %s condition types, %s non-zero records",
                len(long_dfs),
                len(result_df),
            )
            return result_df

        logger.warning("No count columns found to reshape")
        return pl.DataFrame()
