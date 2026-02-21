#!/usr/bin/env python3
"""
Verify Step2 category contributions with 3-outlet filtering.
For each of 40 Step2 dicts, compute:
  contribution = category_articles / total_gpr_articles
where total_gpr_articles = articles matching ANY category in that dict.

Uses binary matrix table for efficiency (same computation as GPRCalculator).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ibis
import json
import logging
import numpy as np
from functools import reduce
from operator import and_, or_

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "/home5/chuenho/data/news/stage2_enhanced_full.duckdb"
ALL_DICTS_DIR = Path("/home5/chuenho/gpr-zero/data/dictionaries/all_dictionaries")

CATEGORY_MAPPING = {
    "Military Escalation": "Military Escalation",
    "Military Escalation (軍事升級)": "Military Escalation",
    "軍事升級": "Military Escalation",
    "Diplomatic Isolation": "Diplomatic Isolation",
    "Diplomatic Isolation (外交孤立)": "Diplomatic Isolation",
    "外交孤立": "Diplomatic Isolation",
    "diplomatic_isolation": "Diplomatic Isolation",
    "Economic Coercion": "Economic Coercion",
    "Economic Coercion (經濟脅迫)": "Economic Coercion",
    "經濟脅迫": "Economic Coercion",
    "Regional Security": "Regional Security",
    "Regional Security (區域安全)": "Regional Security",
    "區域安全": "Regional Security",
    "Internal Political Instability": "Internal Political Instability",
    "Internal Political Instability (內部政治不穩定)": "Internal Political Instability",
    "內部政治不穩定": "Internal Political Instability",
    "Gray Zone Operations": "Gray Zone Operations",
    "Gray Zone Operations (灰色地帶行動)": "Gray Zone Operations",
    "灰色地帶行動": "Gray Zone Operations",
    "US-China-Taiwan Relations": "US-China-Taiwan Relations",
    "US-China-Taiwan Relations (美中台關係)": "US-China-Taiwan Relations",
    "美中台關係": "US-China-Taiwan Relations",
    "Cyber and Information Warfare": "Cyber and Information Warfare",
    "Cyber and Information Warfare (網路與資訊戰)": "Cyber and Information Warfare",
    "網路與資訊戰": "Cyber and Information Warfare",
}

STANDARD_CATEGORIES = [
    "Military Escalation", "Diplomatic Isolation", "Economic Coercion",
    "Regional Security", "Internal Political Instability", "Gray Zone Operations",
    "US-China-Taiwan Relations", "Cyber and Information Warfare"
]


def load_all_dictionaries_for_mapping():
    """Load all 80 dictionaries to create word mapping (matching binary matrix)."""
    all_words = set()
    for dict_file in ALL_DICTS_DIR.glob("*.json"):
        if "_meta.json" in dict_file.name:
            continue
        with open(dict_file, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            if 'word_sets' in dict_data:
                for word_set in dict_data['word_sets'].values():
                    all_words.update(word_set)
    sorted_words = sorted(all_words)
    return {word: f"contains_word_{i:04d}" for i, word in enumerate(sorted_words)}


def load_step2_dictionaries():
    """Load Step2 dictionaries from all_dictionaries dir."""
    dictionaries = {}
    for dict_file in ALL_DICTS_DIR.glob("step2_*.json"):
        if "_meta.json" in dict_file.name:
            continue
        with open(dict_file, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            # Normalize category names
            if 'categories' in dict_data:
                normalized = {}
                for cat_name, cat_def in dict_data['categories'].items():
                    normalized_name = CATEGORY_MAPPING.get(cat_name, cat_name)
                    normalized[normalized_name] = cat_def
                dict_data['categories'] = normalized
            dictionaries[dict_file.stem] = dict_data
    return dictionaries


def build_category_expr(dictionary, category, word_to_column, table):
    """Build ibis boolean expression for a category."""
    word_sets = dictionary.get("word_sets", {})
    categories = dictionary.get("categories", {})

    if category not in categories:
        return None

    query_string = categories[category]
    word_set_names = [name.strip() for name in query_string.split(" AND ")]

    word_set_expressions = []
    for set_name in word_set_names:
        if set_name not in word_sets:
            continue
        words = word_sets[set_name]
        word_exprs = []
        for word in words:
            if word in word_to_column:
                col_name = word_to_column[word]
                if col_name in table.columns:
                    word_exprs.append(table[col_name])
        if word_exprs:
            set_expr = reduce(or_, word_exprs)
            word_set_expressions.append(set_expr)

    if not word_set_expressions:
        return None

    return reduce(and_, word_set_expressions, ibis.literal(True))


def main():
    logger.info("=== Verify Step2 Category Contributions (3-outlet) ===")

    word_to_column = load_all_dictionaries_for_mapping()
    step2_dicts = load_step2_dictionaries()
    logger.info(f"Loaded {len(step2_dicts)} Step2 dicts, {len(word_to_column)} words in mapping")

    conn = ibis.connect(f"duckdb://{DB_PATH}?access_mode=read_only")
    table = conn.table("gpr_binary_matrix_all_words")

    # Pre-filter for 3 outlets and date range
    filtered = table.filter(
        table.outlet.isin([
            'China Times (中國時報)',
            'Liberty Times (自由時報)',
            'United Daily (聯合報)'
        ]) &
        (table.warc_date >= '2017-01-01') &
        (table.warc_date <= '2024-12-31')
    )

    results = []

    for dict_name, dictionary in sorted(step2_dicts.items()):
        logger.info(f"Processing {dict_name}...")

        # Build expressions for each category
        category_exprs = {}
        for cat in STANDARD_CATEGORIES:
            expr = build_category_expr(dictionary, cat, word_to_column, filtered)
            if expr is not None:
                category_exprs[cat] = expr

        if not category_exprs:
            logger.warning(f"  No categories found for {dict_name}")
            continue

        # Build overall GPR expression (matches ANY category)
        overall_expr = reduce(or_, category_exprs.values())

        # Build aggregation: count articles matching each category + overall
        agg_exprs = [filtered.count().name('total_articles')]
        agg_exprs.append(overall_expr.cast('int32').sum().name('overall_gpr'))
        for cat, expr in category_exprs.items():
            safe_name = cat.replace(' ', '_').replace('-', '_')
            agg_exprs.append(expr.cast('int32').sum().name(f'cat_{safe_name}'))

        # Execute single query for this dict
        result = filtered.aggregate(agg_exprs).execute()

        total_articles = int(result['total_articles'].iloc[0])
        overall_gpr = int(result['overall_gpr'].iloc[0])

        dict_result = {
            'dict': dict_name,
            'total_articles': total_articles,
            'overall_gpr': overall_gpr,
        }

        for cat in category_exprs:
            safe_name = cat.replace(' ', '_').replace('-', '_')
            cat_count = int(result[f'cat_{safe_name}'].iloc[0])
            contribution = (cat_count / overall_gpr * 100) if overall_gpr > 0 else 0
            dict_result[cat] = contribution
            dict_result[f'{cat}_count'] = cat_count

        # Find top category
        cat_contribs = {cat: dict_result.get(cat, 0) for cat in STANDARD_CATEGORIES if cat in dict_result}
        if cat_contribs:
            top_cat = max(cat_contribs, key=cat_contribs.get)
            dict_result['top_category'] = top_cat
            dict_result['top_pct'] = cat_contribs[top_cat]

        results.append(dict_result)

    conn.disconnect()

    # Sort by top_pct
    results.sort(key=lambda x: x.get('top_pct', 0), reverse=True)

    print("\n" + "=" * 100)
    print("Per-dict top category contribution (3-outlet, 2017-2024)")
    print("=" * 100)
    print(f"{'Dict':<55} {'Top Category':<28} {'Pct':>8} {'GPR':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['dict']:<55} {r.get('top_category','?'):<28} {r.get('top_pct',0):>7.2f}% {r['overall_gpr']:>10}")

    pcts = [r.get('top_pct', 0) for r in results]
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
            print(f"  {cat:<35} median={np.median(values):>7.2f}%  mean={np.mean(values):>7.2f}%  max={max(values):>7.2f}%")

    # Check for dicts near 42%
    print("\n=== Dicts with top category near 42% (±3%) ===")
    for r in results:
        if abs(r.get('top_pct', 0) - 42) < 3:
            print(f"  {r['dict']}: {r.get('top_category','')} = {r.get('top_pct',0):.2f}%")


if __name__ == "__main__":
    main()
