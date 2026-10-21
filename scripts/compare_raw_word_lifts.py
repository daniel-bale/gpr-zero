#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["duckdb==1.2.1", "pandas==2.2.3", "numpy==2.2.3"]
# ///
"""Reproduce historical spike word lifts from completed rebuilt article word IDs.

No result cache enters the calculation. Caches are comparison targets only.
The input aggregate database is opened read-only; temporary grouped tables live
only in this connection. Word IDs are resolved by vocabulary strings, never by
assuming that rebuilt IDs equal old contains_word_NNNN column ordinals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTLETS = ("China Times (中國時報)", "Liberty Times (自由時報)", "United Daily (聯合報)")
PERIODS = ("full_cache", "strict_study_week_labels")
HISTORICAL_PERIOD = "historical_20250802_log_ranges"
DETAIL_COLUMNS = ["word", "count", "frequency", "baseline_count", "baseline_frequency",
                  "lift_score", "contribution_score", "spike_id", "spike_start", "spike_end",
                  "total_articles_in_spike"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def completion_mode(manifest, allow_incomplete=False):
    # A benchmark must never be promoted to a full replication by its row count.
    complete = (manifest.get("status") == "complete" and manifest.get("source_mode") == "full"
                and all(manifest.get(key) is True for key in ("complete", "input_complete", "source_complete")))
    if manifest.get("status") != "complete":
        raise ValueError("Aggregate database is still running or failed; completed aggregation required")
    if not complete and not allow_incomplete:
        raise ValueError("Full completed aggregation required; --allow-incomplete labels a finished benchmark")
    return "full_raw_rebuild" if complete else "diagnostic_incomplete_or_sample"


def dictionary_words(path):
    document = json.loads(path.read_text())
    sets = document.get("word_sets")
    if not isinstance(sets, dict) or not sets:
        raise ValueError(f"Invalid dictionary word_sets: {path}")
    words = set()
    for members in sets.values():
        if not isinstance(members, list) or any(not isinstance(word, str) or not word for word in members):
            raise ValueError(f"Invalid dictionary words: {path}")
        words.update(members)
    return words


def load_word_families(dictionary_dir, lau_path, vocabulary):
    paths = sorted([*dictionary_dir.glob("step1_*.json"), *dictionary_dir.glob("step2_*.json")])
    if len(paths) != 80:
        raise ValueError(f"Expected exactly 80 historical dictionaries, found {len(paths)}")
    all_words = set().union(*(dictionary_words(path) for path in paths))
    families = {"step2": all_words, "lau": dictionary_words(lau_path)}
    unknown = set.union(*families.values()) - set(vocabulary)
    if unknown:
        raise ValueError(f"Words missing from rebuilt vocabulary: {sorted(unknown)}")
    return families, paths + [lau_path]


def read_spikes(path):
    frame = pd.read_csv(path)
    if not {"start_date", "end_date"}.issubset(frame.columns):
        raise ValueError(f"Spike range columns missing: {path}")
    frame = frame[["start_date", "end_date"]].copy()
    for key in frame:
        frame[key] = pd.to_datetime(frame[key], errors="raise")
        if frame[key].isna().any() or frame[key].dt.dayofweek.ne(0).any() or frame[key].ne(frame[key].dt.normalize()).any():
            raise ValueError(f"Spike boundaries must be non-null Monday labels: {path}")
    if frame.duplicated().any() or frame.start_date.gt(frame.end_date).any():
        raise ValueError(f"Duplicate or reversed spike range: {path}")
    frame = frame.sort_values(["start_date", "end_date"]).reset_index(drop=True)
    if any(frame.start_date.iloc[i] <= frame.end_date.iloc[i-1] + pd.Timedelta(days=6) for i in range(1, len(frame))):
        raise ValueError(f"Overlapping spike periods: {path}")
    return frame


def read_historical_step2_log(path):
    text = path.read_text()
    expected = re.search(r"Loaded (\d+) Step2 spikes", text)
    rows = re.findall(r"^Step2_spike_(\d+): (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})$", text, flags=re.MULTILINE)
    if expected is None or len(rows) != int(expected[1]) or [int(row[0]) for row in rows] != list(range(1, len(rows)+1)):
        raise ValueError("Historical log lacks a complete sequential Step2 range list")
    if "Analysis complete!" not in text:
        raise ValueError("Historical source log does not show completed analysis")
    frame = pd.DataFrame([{"start_date": pd.Timestamp(start), "end_date": pd.Timestamp(end)} for _, start, end in rows])
    if frame.duplicated().any() or frame.start_date.gt(frame.end_date).any() or frame.start_date.dt.dayofweek.ne(0).any() or frame.end_date.dt.dayofweek.ne(0).any():
        raise ValueError("Invalid historical log boundaries")
    return frame


def prepare_counts(connection, timezone, vocabulary_size):
    if timezone not in ("Asia/Taipei", "UTC"):
        raise ValueError(f"Unsupported calendar timezone: {timezone}")
    offset = 8 if timezone == "Asia/Taipei" else 0
    placeholders = ",".join("?" for _ in OUTLETS)
    # Read-only source retains UTC instants; the old cache's naive timestamps
    # represent Taipei wall time. Conversion must happen before DATE casting.
    sql = f"""CREATE TEMP TABLE scoped_articles AS
        SELECT CAST(warc_date + INTERVAL '{offset} hours' AS DATE) AS day, word_ids
        FROM articles WHERE outlet IN ({placeholders})"""
    connection.execute(sql, list(OUTLETS))
    invalid = connection.execute("""SELECT count(*) FROM scoped_articles WHERE word_ids IS NULL
        OR list_min(word_ids) < 0 OR list_max(word_ids) >= ?""", [vocabulary_size]).fetchone()[0]
    if invalid:
        raise ValueError(f"Invalid or null word IDs in {invalid} articles")
    # list_distinct guarantees binary per-article presence even in a synthetic
    # or externally supplied list that accidentally repeats an ID.
    connection.execute("""CREATE TEMP TABLE daily_words AS
        SELECT day, word_id, count(*)::BIGINT AS count
        FROM (SELECT day, unnest(list_distinct(word_ids)) AS word_id FROM scoped_articles)
        GROUP BY day, word_id""")
    connection.execute("""CREATE TEMP TABLE daily_articles AS
        SELECT day, count(*)::BIGINT AS total_articles FROM scoped_articles GROUP BY day""")
    total, undated = connection.execute("""SELECT coalesce(sum(total_articles),0),
        coalesce(sum(total_articles) FILTER (WHERE day IS NULL),0) FROM daily_articles""").fetchone()
    if not total:
        raise ValueError("No articles from the three target outlets")
    baseline = connection.execute("SELECT word_id, sum(count)::BIGINT AS baseline_count FROM daily_words GROUP BY word_id").fetchdf()
    baseline["baseline_frequency"] = baseline.baseline_count / total
    connection.execute("DROP TABLE scoped_articles")
    return baseline, {"baseline_articles": int(total), "undated_baseline_articles": int(undated),
                      "calendar_timezone": timezone, "scope_sql_sha256": hashlib.sha256(sql.encode()).hexdigest()}


def calculate_family(connection, spikes, family, words, vocabulary, baseline):
    ids = {index: word for index, word in enumerate(vocabulary) if word in words}
    base = baseline[baseline.word_id.isin(ids)].copy()
    base["word"] = base.word_id.map(ids)
    base = base[["word", "baseline_count", "baseline_frequency"]].sort_values("word")
    frames, coverage = [], []
    for index, spike in spikes.iterrows():
        start, end = spike.start_date.date(), (spike.end_date + pd.Timedelta(days=6)).date()
        total = connection.execute("SELECT coalesce(sum(total_articles),0) FROM daily_articles WHERE day BETWEEN ? AND ?", [start, end]).fetchone()[0]
        coverage.append({"spike_id": f"{family.capitalize()}_spike_{index+1:02}",
                         "spike_start": str(start), "spike_end": str(spike.end_date.date()),
                         "article_end_inclusive": str(end), "total_articles": int(total)})
        if not total:
            continue
        counts = connection.execute("SELECT word_id, sum(count)::BIGINT AS count FROM daily_words WHERE day BETWEEN ? AND ? GROUP BY word_id", [start, end]).fetchdf()
        counts = counts[counts.word_id.isin(ids)].copy()
        if counts.empty:
            continue
        counts["word"] = counts.word_id.map(ids)
        counts["frequency"] = counts["count"] / total
        counts = counts.merge(base, on="word", validate="one_to_one")
        counts["lift_score"] = counts.frequency / counts.baseline_frequency
        counts["contribution_score"] = counts.lift_score * counts["count"]
        counts["spike_id"] = coverage[-1]["spike_id"]
        counts["spike_start"] = str(start)
        counts["spike_end"] = str(spike.end_date.date())
        counts["total_articles_in_spike"] = int(total)
        frames.append(counts[DETAIL_COLUMNS])
    detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DETAIL_COLUMNS)
    summary = summarize_lifts(detail)
    top = summary[summary.spike_count >= math.ceil(len(spikes) / 2)].head(10).copy() if len(spikes) else summary.iloc[:0]
    return {"baseline": base, "detail": detail, "summary": summary, "top10": top,
            "coverage": pd.DataFrame(coverage), "spike_count": len(spikes),
            "minimum_positive_spikes": math.ceil(len(spikes) / 2),
            "spikes_without_articles": sum(not row["total_articles"] for row in coverage)}


def summarize_lifts(detail):
    if detail.empty:
        return pd.DataFrame(columns=["word", "count", "lift_score", "spike_count"])
    return (detail.groupby("word", as_index=False).agg(count=("count", "sum"),
            lift_score=("lift_score", "mean"), spike_count=("spike_id", "size"))
            .sort_values(["lift_score", "word"], ascending=[False, True]).reset_index(drop=True))


def compare_frames(reference, rebuilt, keys, metrics):
    for label, frame in (("reference", reference), ("rebuilt", rebuilt)):
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate {label} comparison keys: {keys}")
    merged = reference[keys + metrics].merge(rebuilt[keys + metrics], on=keys, how="outer",
                                             suffixes=("_reference", "_rebuilt"), indicator=True)
    both = merged._merge.eq("both")
    mismatch = ~both
    per_metric = {}
    for metric in metrics:
        left = pd.to_numeric(merged[metric + "_reference"], errors="raise")
        right = pd.to_numeric(merged[metric + "_rebuilt"], errors="raise")
        is_count = metric.endswith("count") or metric in ("count", "total_articles_in_spike")
        equal = left.eq(right) if is_count else pd.Series(np.isclose(left, right, rtol=1e-10, atol=1e-12), index=merged.index)
        equal |= left.isna() & right.isna()
        bad = both & ~equal
        mismatch |= bad
        merged[metric + "_delta"] = right - left
        per_metric[metric] = {"mismatched_rows": int(bad.sum()), "maximum_absolute_delta": float((right-left).abs().max()) if both.any() else None}
    merged["matched"] = ~mismatch
    result = {"status": "unavailable" if merged.empty else "matched" if not mismatch.any() else "different",
              "reference_rows": len(reference), "rebuilt_rows": len(rebuilt),
              "reference_only_rows": int(merged._merge.eq("left_only").sum()),
              "rebuilt_only_rows": int(merged._merge.eq("right_only").sum()),
              "mismatched_rows": int(mismatch.sum()), "metrics": per_metric}
    return merged, result


def read_printed_claims(path):
    text = path.read_text()
    block = text.split("Our GPR Index (21 spikes)", 1)
    if len(block) != 2:
        raise ValueError("Cannot locate original printed 21/22 spike table")
    block = block[1].split("\\bottomrule", 1)[0]
    pattern = r"^\s*([^&\\\n]+?)\s*&\s*([0-9.]+)\s*&\s*(\d+)\s*&\s*([^&\\\n]+?)\s*&\s*([0-9.]+)\s*&\s*(\d+)\s*\\\\"
    pairs = re.findall(pattern, block, flags=re.MULTILINE)
    if len(pairs) != 10:
        raise ValueError(f"Expected 10 paired printed rows, found {len(pairs)}")
    rows = []
    for rank, pair in enumerate(pairs, 1):
        for family, values in (("step2", pair[:3]), ("lau", pair[3:])):
            rows.append({"family": family, "printed_rank": rank, "word": values[0].strip(),
                         "printed_lift": float(values[1]), "printed_spike_count": int(values[2])})
    return pd.DataFrame(rows)


def compare_printed(claims, family, result):
    printed = claims[claims.family.eq(family)].copy()
    joined = printed.merge(result["summary"], on="word", how="left", validate="one_to_one")
    joined["rounded_lift_matches"] = (joined.lift_score - joined.printed_lift).abs().le(.005 + 1e-12)
    joined["spike_count_matches"] = joined.spike_count.eq(joined.printed_spike_count)
    ranks = {word: i for i, word in enumerate(result["top10"].word, 1)}
    joined["rebuilt_top10_rank"] = joined.word.map(ranks)
    joined["rank_matches"] = joined.rebuilt_top10_rank.eq(joined.printed_rank)
    joined["matched"] = joined.rounded_lift_matches & joined.spike_count_matches & joined.rank_matches
    return joined


def audit_saved_references(reference_dir, thesis, output):
    """Test internal saved-cache consistency separately from the raw rebuild."""
    output.mkdir(parents=True, exist_ok=True)
    claims = read_printed_claims(thesis)
    report = {"scope": "Saved-cache arithmetic and printed-thesis consistency only; no fresh corpus result",
              "families": [], "source_files": []}
    for family in ("step2", "lau"):
        folder = reference_dir if family == "step2" else reference_dir / "lau_specific"
        prefix = "step2" if family == "step2" else "lau_specific"
        detail_path = folder / f"{prefix}_spike_word_lift_scores.csv"
        summary_path = folder / f"{prefix}_top_words_by_lift.csv"
        detail = pd.read_csv(detail_path)
        summary = summarize_lifts(detail)
        ranges = detail[["spike_start", "spike_end"]].drop_duplicates()
        _, check = compare_frames(pd.read_csv(summary_path), summary, ["word"], ["count", "lift_score", "spike_count"])
        top = summary[summary.spike_count.ge(math.ceil(len(ranges) / 2))].head(10)
        printed = compare_printed(claims, family, {"summary": summary, "top10": top})
        printed.to_csv(output / f"{family}_saved_cache_vs_printed.csv", index=False)
        top.to_csv(output / f"{family}_saved_top10.csv", index=False)
        report["families"].append({"family": family, "cached_spike_count": len(ranges),
            "summary_recalculation": check, "printed_rows": len(printed),
            "printed_lift_matches": int(printed.rounded_lift_matches.sum()),
            "printed_spike_count_matches": int(printed.spike_count_matches.sum()),
            "printed_full_matches": int(printed.matched.sum())})
        report["source_files"].extend({"path": str(path.resolve()), "sha256": file_hash(path)} for path in (detail_path, summary_path))
    report["source_files"].append({"path": str(thesis.resolve()), "sha256": file_hash(thesis)})
    write_json(output / "summary.json", report)
    return report


def run(args):
    started = time.monotonic()
    manifest_path = args.aggregate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mode = completion_mode(manifest, args.allow_incomplete)
    vocabulary_path = Path(manifest["input_dir"]) / "word_vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text())
    if vocabulary["words"] != sorted(set(vocabulary["words"])) or object_hash(vocabulary["words"]) != vocabulary["sha256"] or vocabulary["sha256"] != manifest["vocabulary_sha256"]:
        raise ValueError("Rebuilt vocabulary hash/order differs from aggregate manifest")
    timezone = manifest.get("calendar_timezone")
    if manifest.get("corpus", {}).get("warc_date_timezone") != "UTC":
        raise ValueError("Expected aggregate articles.warc_date in UTC")
    definitions_path = args.spikes_dir / "spike_definitions.json"
    definitions = json.loads(definitions_path.read_text())
    if definitions.get("mode") != mode or definitions.get("calendar_timezone") != timezone:
        raise ValueError("Spike definitions and aggregate completeness/calendar do not match")
    families, dictionary_paths = load_word_families(args.dictionary_dir, args.lau_dictionary, vocabulary["words"])
    producer = json.loads((Path(manifest["input_dir"]) / "manifest.json").read_text())
    frozen = {str(Path(item["path"]).resolve()): item["sha256"] for item in producer["dependencies"]["dictionary_files"]}
    for path in dictionary_paths:
        if frozen.get(str(path.resolve())) != file_hash(path):
            raise ValueError(f"Dictionary differs from raw matching inputs: {path}")
    claims = read_printed_claims(args.thesis)
    source_paths = [manifest_path, vocabulary_path, definitions_path, args.thesis, *dictionary_paths]
    recovery_provenance = ROOT / "replication/raw_rebuild/word_lifts/additional_recovery_provenance.json"
    if recovery_provenance.exists():
        source_paths.append(recovery_provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_audit = audit_saved_references(args.reference_dir, args.thesis, args.output_dir / "reference_audit")
    summary = {"status": "running", "mode": mode, "calendar_timezone": timezone, "checks": [],
               "saved_cache_vs_thesis": saved_audit["families"],
               "definition": {"outlets": OUTLETS, "baseline": "All accepted deduplicated articles from three outlets, all dates including undated; title-only articles remain in denominator",
                 "frequency": "Number of articles containing literal case-sensitive word / number of articles; repeated occurrences within article count once",
                 "spike_dates": "Both endpoints are Monday labels, include through end + 6 days in selected calendar",
                 "lift_score": "(spike matching articles / spike articles) / (all-date matching articles / all-date articles)",
                 "summary_lift": "Arithmetic mean of per-event lift scores over events where word count is positive; absent events excluded",
                 "summary_count": "Sum of matching article counts across nonoverlapping spike events",
                 "top10": "spike_count >= ceil(number of detected spikes / 2), decreasing mean lift, lexical word tie break",
                 "step2_vocabulary": "Union of all 80 Step 1 and Step 2 dictionaries, matching saved source implementation",
                 "lau_vocabulary": "Original Lau dictionary only, matching saved lau_specific table",
                 "strict_study_variant": "Uses independently detected strict-study Monday labels, with same all-date baseline and full final week as historical method"},
               "provenance_notes": ["Step2 producer: scripts/calculate_spike_word_lift_scores_v3.py:115-231,296-301; recovered reference_code/get_top_words_min_spike_coverage.py confirms the intended Step2 full-vocabulary and Lau-specific inputs",
                 "Lau-specific producer recovered at replication/raw_rebuild/word_lifts/reference_code/calculate_lau_specific_word_lift_scores.py; uses lau_binary_matrix_3outlets and all-date baseline with the same event formula",
                 "Supplied older thesis chapter06 has unquantified top-ten words; this calculation audits recovered newer chapter03 printed ratios and spike counts",
                 "Newspaper relative-word table is a separate metric and is not checked by this script"],
               "script_sha256": file_hash(__file__), "aggregate_database": manifest["database"]}
    write_json(args.output_dir / "run_summary.json", summary)
    database = Path(manifest["database"])
    stat = database.stat()
    summary["aggregate_database_stat"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    connection = duckdb.connect(str(database), read_only=True)
    try:
        connection.execute(f"SET threads={args.threads}")
        connection.execute("SET memory_limit=?", [args.memory_limit])
        connection.execute("SET temp_directory=?", [str(args.output_dir / "duckdb_temp")])
        baseline, corpus = prepare_counts(connection, timezone, len(vocabulary["words"]))
        summary.update(corpus)
        summary["families"] = []
        periods = args.periods + ([HISTORICAL_PERIOD] if args.historical_step2_log else [])
        for period in periods:
            destination = args.output_dir / period
            destination.mkdir(parents=True, exist_ok=True)
            for family, words in families.items():
                if period == HISTORICAL_PERIOD:
                    if family != "step2":
                        continue
                    source_paths.append(args.historical_step2_log)
                    spikes = read_historical_step2_log(args.historical_step2_log)
                    spikes.to_csv(destination / "step2_ranges_from_completed_log.csv", index=False)
                else:
                    spike_path = args.spikes_dir / f"rebuilt_{family}_spikes_{period}.csv"
                    source_paths.append(spike_path)
                    spikes = read_spikes(spike_path)
                result = calculate_family(connection, spikes, family, words, vocabulary["words"], baseline)
                for label in ("baseline", "detail", "summary", "top10", "coverage"):
                    result[label].to_csv(destination / f"{family}_{label}.csv", index=False)
                printed = compare_printed(claims, family, result)
                printed.to_csv(destination / f"{family}_printed_thesis_comparison.csv", index=False)
                family_summary = {"period": period, "family": family, "vocabulary_words": len(words),
                                  **{key: result[key] for key in ("spike_count", "minimum_positive_spikes", "spikes_without_articles")},
                                  "printed_rows": len(printed), "printed_matches": int(printed.matched.sum()),
                                  "printed_lift_matches": int(printed.rounded_lift_matches.sum()),
                                  "printed_count_matches": int(printed.spike_count_matches.sum()),
                                  "printed_spike_total": 21 if family == "step2" else 22,
                                  "printed_spike_total_matches": len(spikes) == (21 if family == "step2" else 22)}
                if period == HISTORICAL_PERIOD:
                    # The recovered display script hard-codes >=11 based on
                    # the printed21 header. Keep this distinct from ceil(19/2).
                    printed_threshold_top = result["summary"][result["summary"].spike_count.ge(11)].head(10)
                    printed_threshold_top.to_csv(destination / "step2_top10_printed_threshold11.csv", index=False)
                    alternate = compare_printed(claims, family, {"summary": result["summary"], "top10": printed_threshold_top})
                    alternate.to_csv(destination / "step2_printed_threshold11_comparison.csv", index=False)
                    family_summary.update(event_selection="Archived completed2025-08-02 log ranges, not freshly detected; primary rebuilt21/22 results remain separate",
                        printed_threshold_11_matches=int(alternate.matched.sum()), printed_header_count_disagrees=len(spikes) != 21)
                summary["families"].append(family_summary)
                if period != "full_cache":
                    continue
                folder = args.reference_dir if family == "step2" else args.reference_dir / "lau_specific"
                prefix = "step2" if family == "step2" else "lau_specific"
                references = {"baseline": folder / ("baseline_word_frequencies.csv" if family == "step2" else "lau_specific_baseline_word_frequencies.csv"),
                              "detail": folder / f"{prefix}_spike_word_lift_scores.csv",
                              "summary": folder / f"{prefix}_top_words_by_lift.csv"}
                for label, path in references.items():
                    source_paths.append(path)
                    keys = ["word", "spike_start", "spike_end"] if label == "detail" else ["word"]
                    metrics = (["count", "frequency", "baseline_count", "baseline_frequency", "lift_score", "contribution_score", "total_articles_in_spike"] if label == "detail"
                               else ["count", "lift_score", "spike_count"] if label == "summary"
                               else ["baseline_count", "baseline_frequency"])
                    joined, check = compare_frames(pd.read_csv(path), result[label], keys, metrics)
                    joined.to_csv(destination / f"{family}_{label}_cache_comparison.csv", index=False)
                    summary["checks"].append({"family": family, "artifact": label, **check})
        summary.update(status="complete", elapsed_seconds=round(time.monotonic() - started, 3))
    except BaseException as error:
        summary.update(status="failed", error=str(error), elapsed_seconds=round(time.monotonic() - started, 3))
        raise
    finally:
        connection.close()
        summary["source_files"] = [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in dict.fromkeys(source_paths) if path.exists()]
        write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "mode", "baseline_articles", "elapsed_seconds", "families", "checks")}, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=ROOT / "replication/raw_rebuild/indices")
    parser.add_argument("--spikes-dir", type=Path, default=ROOT / "replication/raw_rebuild/comparison")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replication/raw_rebuild/word_lifts")
    parser.add_argument("--dictionary-dir", type=Path, default=ROOT / "data/dictionaries/all_dictionaries")
    parser.add_argument("--lau-dictionary", type=Path, default=ROOT / "data/dictionaries/yeu_tong_lau_2024.json")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data/word_contributions")
    parser.add_argument("--thesis", type=Path, default=ROOT / "replication/reference_thesis/contents/chapter03.tex")
    parser.add_argument("--periods", nargs="+", choices=PERIODS, default=list(PERIODS))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Audit saved table arithmetic and printed claims without opening an aggregate database")
    parser.add_argument("--historical-step2-log", type=Path, help="Additional conditional analysis on exact ranges recovered from a completed historical log; does not replace independently rebuilt ranges")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.reference_only:
        print(json.dumps(audit_saved_references(arguments.reference_dir, arguments.thesis, arguments.output_dir), ensure_ascii=False))
    else:
        run(arguments)
