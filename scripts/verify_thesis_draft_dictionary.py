#!/usr/bin/env python3
"""Compare the supplied thesis's printed dictionary with the recovered original."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis", type=Path, default=root.parent / "NTU-Thesis-LaTeX-Template")
    parser.add_argument("--dictionary", type=Path, default=root / "data/dictionaries/taiwan_gpr_dict_president.json")
    parser.add_argument("--cached-contributions", type=Path, default=root / "data_analysis/output/category_contributions/category_contributions_president.csv")
    parser.add_argument("--remote-audit", type=Path, default=root / "replication/remote_corpus_audit.json")
    parser.add_argument("--output", type=Path, default=root / "replication/draft_dictionary_check.json")
    args = parser.parse_args()
    chapter = args.thesis / "contents/chapter03.tex"
    source = chapter.read_text()
    dictionary = json.loads(args.dictionary.read_text())
    word_sets = {}
    categories = {}
    targets = []
    for line in source.splitlines():
        word_match = re.match(r"\\textit\{([^}]+)\}\s*&\s*(.*?)\\\\", line)
        if word_match:
            key, words = word_match.groups()
            word_sets[key.replace(" ", "_")] = [word.strip() for word in words.split(",")]
        cat_match = re.match(r"(.+?)\s*&\s*\\makecell\[l\]\{(.+?)\}\s*&\s*(.+?)\s*&\s*([\d.]+)\\\\", line)
        if cat_match:
            name, query, peak, percentage = cat_match.groups()
            name = name.strip().replace(r"\&", "and")
            parts = query.replace("\\\\", " ").split(" AND ")
            categories[name] = " AND ".join(part.strip().replace(" ", "_") for part in parts)
            targets.append({"category": name, "claimed_percent": float(percentage), "claimed_peak": peak.strip()})
    if len(word_sets) != 8 or len(categories) != 7:
        raise ValueError("Expected the complete eight-word-set, seven-category printed table")
    result = {
        "thesis_source": str(chapter.resolve()),
        "thesis_sha256": hashlib.sha256(chapter.read_bytes()).hexdigest(),
        "dictionary_source": str(args.dictionary.resolve()),
        "dictionary_sha256": hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
        "word_sets_match_exactly": word_sets == dictionary["word_sets"],
        "categories_match_exactly": categories == dictionary["categories"],
        "word_set_count": len(word_sets),
        "unique_words": len({word for words in word_sets.values() for word in words}),
        "category_count": len(categories),
        "targets": targets,
        "claimed_category_percentage_sum": round(sum(target["claimed_percent"] for target in targets), 2),
        "note": "The printed table matches a recovered original only if both checks are true. Category percentages can overlap; they are not a partition.",
    }
    if args.cached_contributions.exists():
        with args.cached_contributions.open(newline="") as stream:
            counts = {row["category"]: row for row in csv.DictReader(stream)}
        comparisons = []
        for target in targets:
            row = counts[target["category"]]
            value = 100 * int(row["category_articles"]) / int(row["total_gpr_articles"])
            expected_peak = re.search(r"\d{4}-\d{2}-\d{2}", target["claimed_peak"]).group()
            comparisons.append({
                **target, "category_articles": int(row["category_articles"]),
                "total_gpr_articles": int(row["total_gpr_articles"]),
                "recomputed_percent_from_cached_counts": value,
                "percent_matches_printed_rounding": round(value, 2) == target["claimed_percent"],
                "cached_peak_week": row["peak_week"],
                "peak_week_matches": row["peak_week"] == expected_peak,
            })
        result["cached_count_comparison"] = {
            "source": str(args.cached_contributions.resolve()),
            "sha256": hashlib.sha256(args.cached_contributions.read_bytes()).hexdigest(),
            "scope": "Ratios recomputed from archived category counts, not an independent article-level recount. Peak dates checked against archived peak weeks.",
            "rows": comparisons,
        }
    if args.remote_audit.exists():
        audit = json.loads(args.remote_audit.read_text())
        reconstructed = audit.get("original_president_labeled_reconstruction", {})
        weeks = reconstructed.get("weekly_original", [])
        if weeks:
            category_order = reconstructed["category_order"]
            denominator = sum(week["overall_gpr_articles"] for week in weeks)
            remote_checks = []
            for target in targets:
                field = f"category_{category_order.index(target['category'])}_count"
                numerator = sum(week[field] for week in weeks)
                peak = max(weeks, key=lambda week: week[field] / week["total_articles"])
                expected_peak = re.search(r"\d{4}-\d{2}-\d{2}", target["claimed_peak"]).group()
                remote_checks.append({
                    **target, "category_articles": numerator, "total_gpr_articles": denominator,
                    "recomputed_percent": 100 * numerator / denominator,
                    "percent_matches_printed_rounding": round(100 * numerator / denominator, 2) == target["claimed_percent"],
                    "recomputed_peak_week": peak["week"], "peak_week_matches": peak["week"] == expected_peak,
                })
            result["remote_reconstruction_comparison"] = {
                "source": str(args.remote_audit.resolve()),
                "scope": "Weekly article-level cached flags, seven original categories with explicit restoration of the removed economic term; no text reclassification.",
                "weeks": len(weeks), "total_articles": sum(week["total_articles"] for week in weeks),
                "rows": remote_checks,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("word_sets_match_exactly", "categories_match_exactly", "unique_words", "category_count")}, indent=2))
    if not all((result["word_sets_match_exactly"], result["categories_match_exactly"])):
        raise SystemExit("Printed dictionary differs from recovered original; see report")


if __name__ == "__main__":
    main()
