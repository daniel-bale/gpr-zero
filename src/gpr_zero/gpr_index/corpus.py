"""Dictionary inventory and cached article cohorts with one vocabulary owner.

A cache's numbered word columns describe the complete inventory that built it.
Selecting dictionaries or articles never renumbers that vocabulary. Historical
thesis caches have no manifest and remain readable without rewriting them.
"""

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import ibis
import ibis.expr.types as ir

from gpr_zero.constants import (
    BINARY_MATRIX_TABLE,
    DATE_RANGE_END,
    DATE_RANGE_START,
    TARGET_OUTLETS,
)

logger = logging.getLogger(__name__)

_VOCABULARY_TABLE = f"{BINARY_MATRIX_TABLE}_vocabulary"


@dataclass(frozen=True)
class DictionaryInventory:
    """Loaded dictionaries and the canonical ordering of their cache vocabulary."""

    dictionary_paths: tuple[str, ...]
    dictionaries: Mapping[str, dict]
    words: tuple[str, ...]
    word_columns: Mapping[str, str]
    _thesis_step2_ordinals: bool = False

    @classmethod
    def from_directory(cls, directory: str | Path) -> "DictionaryInventory":
        return cls.from_paths(Path(directory).glob("*.json"))

    @classmethod
    def from_paths(cls, paths) -> "DictionaryInventory":
        """Load the calculator inventory, retaining its skip-and-log contract.

        Invalid JSON and definitions without both required keys never contribute
        vocabulary. File names do not determine validity in the calculator.
        """
        dictionaries = {}
        loaded_paths = []
        words = set()
        for path in map(Path, paths):
            try:
                with path.open(encoding="utf-8") as handle:
                    dictionary = json.load(handle)
                if (
                    not isinstance(dictionary, dict)
                    or "word_sets" not in dictionary
                    or "categories" not in dictionary
                ):
                    logger.warning("Invalid dictionary format: %s", path)
                    continue
                dictionaries[path.stem] = dictionary
                loaded_paths.append(str(path))
                for word_set in dictionary["word_sets"].values():
                    words.update(word_set)
            except Exception as exc:
                logger.error("Failed to load dictionary %s: %s", path, exc)
                continue
        if not dictionaries:
            raise ValueError(
                "No valid dictionaries were loaded. Check that dictionary_paths "
                "are correct and each file contains 'word_sets' and 'categories' keys."
            )
        return cls._from_loaded(loaded_paths, dictionaries, words)

    @classmethod
    def thesis_directory(cls, directory: str | Path) -> "DictionaryInventory":
        """Load every JSON file as the historical raw analysis scripts did."""
        return cls._load_raw(Path(directory).glob("*.json"))

    @classmethod
    def thesis_category_directory(cls, directory: str | Path) -> "DictionaryInventory":
        """Retain category workflows' raw definitions, excluding metadata files."""
        return cls._load_raw(
            path
            for path in Path(directory).glob("*.json")
            if "_meta.json" not in path.name
        )

    @classmethod
    def _load_raw(cls, paths) -> "DictionaryInventory":
        dictionaries = {}
        loaded_paths = []
        words = set()
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                dictionary = json.load(handle)
            if not isinstance(dictionary, dict):
                raise ValueError(f"Invalid dictionary: {path}")
            dictionaries[path.stem] = dictionary
            loaded_paths.append(str(path))
            for word_set in dictionary.get("word_sets", {}).values():
                words.update(word_set)
        if not dictionaries:
            raise ValueError("No dictionaries found in the requested inventory")
        return cls._from_loaded(loaded_paths, dictionaries, words)

    @classmethod
    def _from_loaded(cls, loaded_paths, dictionaries, words):
        ordered_words = tuple(sorted(words))
        return cls(
            tuple(loaded_paths),
            MappingProxyType(dictionaries),
            ordered_words,
            MappingProxyType(
                {word: f"contains_word_{i:04d}" for i, word in enumerate(ordered_words)}
            ),
        )

    @classmethod
    def thesis_step2_ordinal(cls, directory: str | Path) -> "DictionaryInventory":
        """Preserve thesis workflows that numbered Step2 words against a full cache.

        This intentional historical interpretation must not be used for new
        analyses or cache construction. Its mismatch is part of those outputs.
        """
        inventory = cls.thesis_directory(directory)
        return replace(inventory, _thesis_step2_ordinals=True)

    def select(self, prefix: str = "") -> dict[str, dict]:
        """Select definitions without changing the inventory's word columns."""
        return {
            name: dictionary
            for name, dictionary in self.dictionaries.items()
            if name.startswith(prefix)
        }


@dataclass(frozen=True)
class ArticleCorpus:
    """A cached table, its vocabulary, and an optional article cohort."""

    table: ir.Table
    inventory: DictionaryInventory

    @property
    def word_columns(self) -> Mapping[str, str]:
        return self.inventory.word_columns

    @classmethod
    def from_cache(cls, connection, inventory: DictionaryInventory) -> "ArticleCorpus":
        from gpr_zero.db import get_binary_matrix

        table = get_binary_matrix(connection)
        if (
            not inventory._thesis_step2_ordinals
            and _VOCABULARY_TABLE in connection.list_tables()
        ):
            rows = connection.table(_VOCABULARY_TABLE).execute()
            expected = dict(inventory.word_columns)
            actual = dict(zip(rows["word"], rows["column_name"]))
            if actual != expected or len(rows) != len(expected):
                raise ValueError(
                    "Cached article vocabulary does not match the full dictionary "
                    "inventory. Use the inventory that built this cache or explicitly "
                    "rebuild the cache."
                )
            missing = set(expected.values()).difference(table.columns)
            if missing:
                raise ValueError(
                    "Cached article matrix is missing manifested word columns"
                )
        return cls(table, inventory)

    @classmethod
    def newspaper_spike_cache(
        cls, connection, inventory: DictionaryInventory
    ) -> "ArticleCorpus":
        """Retain the newspaper workflow's historical cache-table preference."""
        try:
            table = connection.table("articles_word_presence_step2_dictionaries_W")
        except Exception:
            return cls.from_cache(connection, inventory)
        return cls(table, inventory)

    @classmethod
    def rebuild(cls, connection, inventory: DictionaryInventory) -> "ArticleCorpus":
        """Explicitly replace the full article cache and record its vocabulary."""
        if inventory._thesis_step2_ordinals:
            raise ValueError("Historical Step2 ordinals cannot build a canonical cache")
        articles = connection.table("articles")
        required = {"content", "source_warc_file", "warc_date"}
        missing = required.difference(articles.columns)
        if missing:
            raise ValueError(
                f"Input articles are missing required columns: {sorted(missing)}"
            )
        labeled = articles.mutate(
            **{
                column: articles.content.contains(word)
                for word, column in inventory.word_columns.items()
            }
        )
        # Both replacements are one DuckDB transaction: failed construction must
        # not pair an old manifest with a new matrix (or the reverse).
        database = connection.con
        database.execute("BEGIN TRANSACTION")
        try:
            connection.create_table(BINARY_MATRIX_TABLE, labeled, overwrite=True)
            vocabulary = ibis.memtable(
                {
                    "word": list(inventory.word_columns),
                    "column_name": list(inventory.word_columns.values()),
                },
                schema={"word": "string", "column_name": "string"},
            )
            connection.create_table(_VOCABULARY_TABLE, vocabulary, overwrite=True)
            database.execute("COMMIT")
        except Exception:
            database.execute("ROLLBACK")
            raise
        return cls.from_cache(connection, inventory)

    def three_outlets(self) -> "ArticleCorpus":
        return replace(
            self, table=self.table.filter(self.table.outlet.isin(TARGET_OUTLETS))
        )

    def study_period(self) -> "ArticleCorpus":
        return replace(
            self,
            table=self.table.filter(
                (self.table.warc_date >= DATE_RANGE_START)
                & (self.table.warc_date <= DATE_RANGE_END)
            ),
        )

    def attach(self, calculator):
        """Bind a legacy calculator to this corpus without reindexing its words."""
        missing = set(calculator.all_words).difference(self.word_columns)
        if missing:
            raise ValueError(
                "Selected dictionary words are absent from the cache inventory"
            )
        calculator.labeled_articles_table = self.table
        calculator.word_column_mapping = dict(self.word_columns)
        return calculator

    def calculator(self, dictionary_names=None):
        """Return a calculator for selected definitions with the full cache mapping."""
        from gpr_zero.gpr_index.calculator import GPRCalculator

        calculator = GPRCalculator(self.inventory)
        if dictionary_names is not None:
            calculator.dictionaries = {
                name: self.inventory.dictionaries[name] for name in dictionary_names
            }
        return self.attach(calculator)
