"""Interpret thesis dictionaries through article and binary-indicator adapters.

The thesis has three analysis conventions in addition to its two calculators.
Their differences are intentional here: named operations preserve how missing
word sets, empty sets and null indicators affected each published workflow.
Callers choose the operation they perform, rather than configure a grammar or a
collection of matching flags.
"""

import re
from collections.abc import Iterable, Mapping
from functools import reduce
from operator import and_, or_

import ibis
import ibis.expr.types as ir

from gpr_zero.constants import CATEGORY_MAPPING, STANDARD_CATEGORIES

THESIS_CATEGORY_ALIASES = {
    **CATEGORY_MAPPING,
    **{
        name.lower().replace(" ", "_").replace("-", "_"): name
        for name in STANDARD_CATEGORIES
    },
}


def normalize_categories(
    dictionary: Mapping, aliases: Mapping[str, str] = CATEGORY_MAPPING
) -> dict:
    """Copy a dictionary with thesis category aliases and last-definition wins.

    Older peak/spike workflows additionally accept snake-case labels through
    ``THESIS_CATEGORY_ALIASES``. Normalization never mutates the loaded inventory.
    """
    result = dict(dictionary)
    if "categories" in result:
        result["categories"] = {
            aliases.get(name, name): query
            for name, query in result["categories"].items()
        }
    return result


def _all(expressions: Iterable[ir.BooleanValue]) -> ir.BooleanValue:
    return reduce(and_, expressions, ibis.literal(True))


def _any(expressions: Iterable[ir.BooleanValue]) -> ir.BooleanValue:
    expressions = list(expressions)
    return reduce(or_, expressions) if expressions else ibis.literal(False)


def _required_sets(query: str) -> list[str]:
    """Calculator grammar: AND, legacy +, and ignored N/ proximity suffixes."""
    return [
        name
        for component in query.split(" AND ")
        for subcomponent in component.strip().split("+")
        if (name := subcomponent.strip().split("N/")[0].strip())
    ]


def _literal_pattern(words: Iterable[str]) -> str:
    """The article calculator's literal, case-insensitive regex convention."""
    if not words:
        return r"a^"
    escaped = [re.escape(word) for word in words if word]
    return f"(?i)({'|'.join(escaped)})" if escaped else r"a^"


class _BinaryIndicators:
    """Adapter for the stable word-to-column mapping of a binary matrix."""

    def __init__(self, table: ir.Table, word_columns: Mapping[str, str]):
        self.table = table
        self.word_columns = word_columns

    def word_set(self, words: Iterable[str]) -> ir.BooleanValue:
        # The binary calculator deliberately raises for a mapped column that is
        # absent from the table. Analysis operations consider only present ones.
        return _any(
            self.table[self.word_columns[word]]
            for word in words
            if word in self.word_columns
        )

    def available_sets(
        self, word_sets: Mapping[str, Iterable[str]]
    ) -> dict[str, ir.BooleanValue | None]:
        expressions = {}
        for name, words in word_sets.items():
            columns = [
                self.word_columns[word]
                for word in words
                if word in self.word_columns
                and self.word_columns[word] in self.table.columns
            ]
            expressions[name] = (
                _any(self.table[col] for col in columns) if columns else None
            )
        return expressions


class _ArticleText:
    """Adapter for literal regex matching and LT- RTF category assignments."""

    def __init__(self, table: ir.Table):
        self.table = table.mutate(
            content_lower=table.content.lower(),
            is_rtf=table.source_file.startswith("LT-"),
        )

    def word_set(self, words: Iterable[str]) -> ir.BooleanValue:
        return self.table.content_lower.re_search(_literal_pattern(words)).fill_null(
            False
        )


class DictionaryMatcher:
    """Build dictionary matches while preserving each thesis workflow's rules.

    Dictionary ordering is retained. This module does not load data, choose a
    vocabulary, aggregate counts, or change the calculators' share denominators.
    """

    def __init__(self, dictionary: Mapping):
        self.word_sets = dictionary.get("word_sets", {})
        self.categories = dictionary.get("categories", {})

    @staticmethod
    def _labels(table: ir.Table, categories: Mapping[str, ir.BooleanValue]) -> ir.Table:
        return table.mutate(
            **{f"is_{name}": expression for name, expression in categories.items()}
        ).mutate(matches_any_category=_any(categories.values()).fill_null(False))

    def label_binary(
        self, table: ir.Table, word_columns: Mapping[str, str]
    ) -> ir.Table:
        """Label a matrix using strict calculator references and AND/+ /N/."""
        adapter = _BinaryIndicators(table, word_columns)
        table = table.mutate(
            **{
                f"has_{name}": adapter.word_set(words)
                for name, words in self.word_sets.items()
            }
        )
        categories = {}
        for name, query in self.categories.items():
            required = _required_sets(query)
            categories[name] = (
                _all(table[f"has_{word_set}"] for word_set in required).fill_null(False)
                if required
                else ibis.literal(False)
            )
        return self._labels(table, categories)

    def label_articles(self, table: ir.Table) -> ir.Table:
        """Label article text, retaining regex, RTF and invalid-reference rules."""
        adapter = _ArticleText(table)
        table = adapter.table.mutate(
            **{
                f"has_{name}": adapter.word_set(words)
                for name, words in self.word_sets.items()
            }
        )
        categories = {}
        for name, query in self.categories.items():
            required = _required_sets(query)
            if not required or any(
                f"has_{word_set}" not in table.columns for word_set in required
            ):
                categories[name] = ibis.literal(False)
                continue
            content_match = _all(table[f"has_{word_set}"] for word_set in required)
            categories[name] = ibis.cases(
                (table.is_rtf, table.category == name), else_=content_match
            ).fill_null(False)
        return self._labels(table, categories)

    def category_match(
        self, table: ir.Table, word_columns: Mapping[str, str], category: str
    ) -> ir.BooleanValue | None:
        """Category contribution/peak convention: skip unavailable word sets.

        Only literal `` AND `` separates references. Return None for an absent
        category or when no referenced set has any available word columns.
        Preserve nulls; peak sampling callers explicitly fill them with False.
        """
        if category not in self.categories:
            return None
        available = _BinaryIndicators(table, word_columns).available_sets(
            self.word_sets
        )
        expressions = [
            expression
            for name in self.categories[category].split(" AND ")
            if (expression := available.get(name.strip())) is not None
        ]
        return _all(expressions) if expressions else None

    def spike_category_matches(
        self, table: ir.Table, word_columns: Mapping[str, str]
    ) -> dict[str, ir.BooleanValue]:
        """Normalized spike convention: retain empty sets, ignore missing names."""
        available = _BinaryIndicators(table, word_columns).available_sets(
            self.word_sets
        )
        sets = {
            name: expression if expression is not None else ibis.literal(False)
            for name, expression in available.items()
        }
        categories = {}
        for name, query in self.categories.items():
            expressions = [
                sets[part.strip()]
                for part in query.split(" AND ")
                if part.strip() in sets
            ]
            categories[name] = (
                _all(expressions).fill_null(False)
                if expressions
                else ibis.literal(False)
            )
        return categories

    def timeseries_match(
        self, table: ir.Table, word_columns: Mapping[str, str]
    ) -> ir.BooleanValue:
        """Three-outlet timeseries convention, including all-missing matches.

        Unlike spike matching, a category with no known references is True.
        Defined empty word sets are False and null indicators remain nullable.
        """
        available = _BinaryIndicators(table, word_columns).available_sets(
            self.word_sets
        )
        sets = {
            name: expression if expression is not None else ibis.literal(False)
            for name, expression in available.items()
        }
        return _any(
            _all(
                sets[part.strip()]
                for part in query.split(" AND ")
                if part.strip() in sets
            )
            for query in self.categories.values()
        )
