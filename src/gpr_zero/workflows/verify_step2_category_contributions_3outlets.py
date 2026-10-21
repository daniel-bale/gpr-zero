#!/usr/bin/env python3
"""
Verify Step2 category contributions with 3-outlet filtering.
For each of 40 Step2 dicts, compute:
  contribution = category_articles / total_gpr_articles
where total_gpr_articles = articles matching ANY category in that dict.

Uses binary matrix table for efficiency (same computation as GPRCalculator).
"""

import logging
from functools import reduce
from operator import or_

import numpy as np

from gpr_zero.constants import CATEGORY_MAPPING, STANDARD_CATEGORIES
from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory
from gpr_zero.gpr_index.dictionary import DictionaryMatcher, normalize_categories
from gpr_zero.project import get_project

logger = logging.getLogger(__name__)


def main(project=None):
    project = project or get_project()
    logger.info("=== Verify Step2 Category Contributions (3-outlet) ===")
    inventory = DictionaryInventory.thesis_category_directory(project.dictionaries_dir)
    # The verification workflow also recognized this one historical alias.
    aliases = {**CATEGORY_MAPPING, "diplomatic_isolation": "Diplomatic Isolation"}
    step2_dicts = {
        name: normalize_categories(dictionary, aliases=aliases)
        for name, dictionary in inventory.select("step2_").items()
    }
    conn = project.connect()
    corpus = ArticleCorpus.from_cache(conn, inventory).three_outlets().study_period()
    filtered = corpus.table

    results = []

    for dict_name, dictionary in sorted(step2_dicts.items()):
        logger.info(f"Processing {dict_name}...")

        # Build expressions for each category
        category_exprs = {}
        for cat in STANDARD_CATEGORIES:
            expr = DictionaryMatcher(dictionary).category_match(
                filtered, corpus.word_columns, cat
            )
            if expr is not None:
                category_exprs[cat] = expr

        if not category_exprs:
            logger.warning(f"  No categories found for {dict_name}")
            continue

        # Build overall GPR expression (matches ANY category)
        overall_expr = reduce(or_, category_exprs.values())

        # Build aggregation: count articles matching each category + overall
        agg_exprs = [filtered.count().name("total_articles")]
        agg_exprs.append(overall_expr.cast("int32").sum().name("overall_gpr"))
        for cat, expr in category_exprs.items():
            safe_name = cat.replace(" ", "_").replace("-", "_")
            agg_exprs.append(expr.cast("int32").sum().name(f"cat_{safe_name}"))

        # Execute single query for this dict
        result = filtered.aggregate(agg_exprs).execute()

        total_articles = int(result["total_articles"].iloc[0])
        overall_gpr = int(result["overall_gpr"].iloc[0])

        dict_result = {
            "dict": dict_name,
            "total_articles": total_articles,
            "overall_gpr": overall_gpr,
        }

        for cat in category_exprs:
            safe_name = cat.replace(" ", "_").replace("-", "_")
            cat_count = int(result[f"cat_{safe_name}"].iloc[0])
            contribution = (cat_count / overall_gpr * 100) if overall_gpr > 0 else 0
            dict_result[cat] = contribution
            dict_result[f"{cat}_count"] = cat_count

        # Find top category
        cat_contribs = {
            cat: dict_result.get(cat, 0)
            for cat in STANDARD_CATEGORIES
            if cat in dict_result
        }
        if cat_contribs:
            top_cat = max(cat_contribs, key=cat_contribs.get)
            dict_result["top_category"] = top_cat
            dict_result["top_pct"] = cat_contribs[top_cat]

        results.append(dict_result)

    conn.disconnect()

    # Sort by top_pct
    results.sort(key=lambda x: x.get("top_pct", 0), reverse=True)

    print("\n" + "=" * 100)
    print("Per-dict top category contribution (3-outlet, 2017-2024)")
    print("=" * 100)
    print(f"{'Dict':<55} {'Top Category':<28} {'Pct':>8} {'GPR':>10}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['dict']:<55} {r.get('top_category', '?'):<28} {r.get('top_pct', 0):>7.2f}% {r['overall_gpr']:>10}"
        )

    pcts = [r.get("top_pct", 0) for r in results]
    print(f"\nMin top contribution: {min(pcts):.2f}%")
    print(f"Max top contribution: {max(pcts):.2f}%")
    print(f"Median top contribution: {np.median(pcts):.2f}%")
    print(f"Mean top contribution: {np.mean(pcts):.2f}%")

    # Median per category across all dicts
    print("\n" + "=" * 100)
    print("Median contribution per category across 40 dicts (3-outlet)")
    print("=" * 100)
    for cat in STANDARD_CATEGORIES:
        values = [r.get(cat, 0) for r in results if cat in r]
        if values:
            print(
                f"  {cat:<35} median={np.median(values):>7.2f}%  mean={np.mean(values):>7.2f}%  max={max(values):>7.2f}%"
            )

    # Check for dicts near 42%
    print("\n=== Dicts with top category near 42% (±3%) ===")
    for r in results:
        if abs(r.get("top_pct", 0) - 42) < 3:
            print(
                f"  {r['dict']}: {r.get('top_category', '')} = {r.get('top_pct', 0):.2f}%"
            )
