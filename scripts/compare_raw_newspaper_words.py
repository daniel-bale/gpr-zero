#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["duckdb==1.2.1", "pandas==2.2.3", "numpy==2.2.3"]
# ///
"""Replay the actual thesis newspaper word-ratio calculation and outlet fix.

Both recovered historical and freshly detected spike ranges are analyzed. The
historical decomposition pooled three outlets even for events labeled as a
specific newspaper. The separate corrected variant filters articles to that
newspaper, while preserving the positive-event mean and all other definitions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import duckdb
import numpy as np
import pandas as pd

from compare_raw_word_lifts import ROOT, OUTLETS, completion_mode, dictionary_words, file_hash, object_hash, write_json, compare_frames

REFERENCES = ROOT / "replication/raw_rebuild/word_lifts/alternative_references"
NEWSPAPERS = {"中國時報": OUTLETS[0], "自由時報": OUTLETS[1]}
EVENT_KEYS = ["spike_id", "start_date", "end_date", "peak_date", "gpr_category", "newspaper"]


def load_decomposition(path):
    frame = pd.read_csv(path)
    frame = frame[frame.newspaper.isin(NEWSPAPERS)].copy()
    events = frame[EVENT_KEYS].drop_duplicates().sort_values("spike_id").reset_index(drop=True)
    if events.spike_id.duplicated().any():
        raise ValueError("Historical spike IDs have conflicting definitions")
    for key in ("start_date", "end_date", "peak_date"):
        events[key] = pd.to_datetime(events[key], errors="raise")
        if events[key].isna().any() or events[key].dt.dayofweek.ne(0).any():
            raise ValueError("Historical event boundaries must be Monday week labels")
    if events.start_date.gt(events.end_date).any():
        raise ValueError("Reversed historical event boundaries")
    words = frame[frame.cond_type.eq("word")][["spike_id", "newspaper", "cond_name", "cond_count", "spike_total_articles"]].rename(
        columns={"cond_name": "word", "cond_count": "count", "spike_total_articles": "total_articles"})
    if words.duplicated(["spike_id", "word"]).any() or (words["count"] <= 0).any() or (words.total_articles <= 0).any():
        raise ValueError("Historical word records must be unique positive-event counts")
    return events, words


def appearance_ratios(detail):
    positive = detail[detail["count"].gt(0) & detail.total_articles.gt(0)].copy()
    positive["appearance_rate"] = positive["count"] / positive.total_articles
    grouped = positive.groupby(["word", "newspaper"], as_index=False).agg(
        mean_appearance_rate=("appearance_rate", "mean"), total_word_count=("count", "sum"), positive_spike_count=("spike_id", "size"))
    sides = []
    for newspaper, prefix in (("中國時報", "china"), ("自由時報", "liberty")):
        side = grouped[grouped.newspaper.eq(newspaper)].drop(columns="newspaper").rename(
            columns={key: f"{prefix}_{key}" for key in ("mean_appearance_rate", "total_word_count", "positive_spike_count")})
        sides.append(side)
    result = sides[0].merge(sides[1], on="word", how="outer").fillna(0)
    # Preserves recovered producer's epsilon and direction, which contradicts
    # the thesis footnote's China/Liberty description.
    result["ratio_liberty_china"] = result.liberty_mean_appearance_rate / (result.china_mean_appearance_rate + 1e-6)
    result["ratio_china_liberty_plain"] = np.where(result.liberty_mean_appearance_rate.gt(0), result.china_mean_appearance_rate / result.liberty_mean_appearance_rate, np.nan)
    result["liberty_zero"] = result.liberty_mean_appearance_rate.eq(0)
    result["china_zero"] = result.china_mean_appearance_rate.eq(0)
    return result.sort_values("word").reset_index(drop=True)


def top_words(ratios):
    clean = ratios[~ratios.liberty_zero & ~ratios.china_zero].sort_values(["ratio_liberty_china", "word"])
    low = clean.head(10).copy().assign(favored_newspaper="China Times (中國時報)")
    high = clean.tail(10).copy().assign(favored_newspaper="Liberty Times (自由時報)")
    low["rank"] = range(1, len(low) + 1)
    high["rank"] = range(1, len(high) + 1)
    return pd.concat([low, high], ignore_index=True)


def printed_claims(thesis):
    text = thesis.read_text().split("\\label{tab:newspaper-ratios}", 1)[1].split("\\bottomrule", 1)[0]
    pairs = re.findall(r"^\s*([^&\\\n]+?)\s*&\s*([0-9.]+)\s*&\s*([^&\\\n]+?)\s*&\s*([0-9.]+)\s*\\\\", text, flags=re.MULTILINE)
    if len(pairs) != 10:
        raise ValueError(f"Expected 10 paired newspaper rows, found {len(pairs)}")
    return pd.DataFrame([{"word": word.strip(), "printed_ratio": float(value), "printed_side": side, "printed_rank": rank}
        for rank, pair in enumerate(pairs, 1) for side, word, value in
        (("China Times (中國時報)", pair[0], pair[1]), ("Liberty Times (自由時報)", pair[2], pair[3]))])


def compare_printed(claims, ratios):
    result = claims.merge(ratios, on="word", how="left", validate="one_to_one")
    result["producer_direction_matches"] = (result.ratio_liberty_china - result.printed_ratio).abs().le(.005 + 1e-12)
    result["footnote_direction_matches"] = (result.ratio_china_liberty_plain - result.printed_ratio).abs().le(.005 + 1e-12)
    ranks = top_words(ratios)[["word", "favored_newspaper", "rank"]]
    result = result.merge(ranks, left_on=["word", "printed_side"], right_on=["word", "favored_newspaper"], how="left", validate="one_to_one")
    result["rank_matches"] = result.printed_rank.eq(result["rank"]) & result.printed_side.eq(result.favored_newspaper)
    return result


def audit_reference(args, destination):
    destination.mkdir(parents=True, exist_ok=True)
    events, detail = load_decomposition(args.decomposition)
    ratios = appearance_ratios(detail)
    ratios.to_csv(destination / "ratios_recalculated_from_saved_counts.csv", index=False)
    claims = printed_claims(args.thesis)
    printed = compare_printed(claims, ratios)
    printed.to_csv(destination / "saved_counts_vs_printed.csv", index=False)
    saved = pd.read_csv(args.reference_dir / "word_appearance_ratios.csv")
    saved = saved[saved.cond_type.eq("word")].rename(columns={"cond_name": "word"})
    joined, check = compare_frames(saved, ratios, ["word"], ["ratio_liberty_china"])
    joined.to_csv(destination / "saved_ratio_arithmetic_comparison.csv", index=False)
    report = {"scope": "Saved aggregate counts only, independent of new raw corpus",
              "events": len(events), "positive_word_event_rows": len(detail), "ratio_rows": len(ratios),
              "saved_ratio_arithmetic": check, "printed_rows": len(printed),
              "producer_direction_matches": int(printed.producer_direction_matches.sum()),
              "footnote_direction_matches": int(printed.footnote_direction_matches.sum()),
              "printed_rank_matches": int(printed.rank_matches.sum()),
              "definition": "Unweighted mean of positive word-event appearance rates within each newspaper label; Liberty/(China+1e-6). Historical source drops actual article outlet before event join."}
    write_json(destination / "summary.json", report)
    return report, events, detail


def prepare_daily(connection, timezone, word_ids):
    if timezone not in ("Asia/Taipei", "UTC"):
        raise ValueError("Unsupported calendar timezone")
    hours = 8 if timezone == "Asia/Taipei" else 0
    dictionary_frame = pd.DataFrame({"word_id": list(word_ids), "word": list(word_ids.values())})
    connection.register("dictionary_words", dictionary_frame)
    connection.execute(f"""CREATE TEMP VIEW dated_articles AS
        SELECT CAST(warc_date + INTERVAL '{hours} hours' AS DATE) AS day, outlet, word_ids
        FROM articles WHERE outlet IN ({','.join(repr(o) for o in OUTLETS)})""")
    connection.execute("""CREATE TEMP TABLE daily_denominators AS
        SELECT day, outlet, count(*)::BIGINT AS total_articles FROM dated_articles GROUP BY day, outlet""")
    connection.execute("""CREATE TEMP TABLE daily_word_counts AS
        SELECT day, outlet, word, count(*)::BIGINT AS count
        FROM (SELECT day, outlet, unnest(list_distinct(word_ids)) AS word_id FROM dated_articles)
        JOIN dictionary_words USING(word_id) GROUP BY day, outlet, word""")


def calculate_variant(connection, events, condition_on_outlet):
    if events.empty:
        denominators = pd.DataFrame(columns=["spike_id", "newspaper", "total_articles"])
        return pd.DataFrame(columns=["spike_id", "newspaper", "word", "count", "total_articles"]), denominators
    selected = events.copy()
    selected["outlet"] = selected.newspaper.map(NEWSPAPERS)
    selected["article_end"] = selected.end_date + pd.Timedelta(days=6)
    connection.register("events", selected)
    condition = "AND d.outlet=e.outlet" if condition_on_outlet else ""
    denominators = connection.execute(f"""SELECT e.spike_id, e.newspaper, coalesce(sum(d.total_articles),0)::BIGINT AS total_articles
        FROM events e LEFT JOIN daily_denominators d ON d.day BETWEEN e.start_date AND e.article_end {condition}
        GROUP BY e.spike_id,e.newspaper ORDER BY e.spike_id""").fetchdf()
    counts = connection.execute(f"""SELECT e.spike_id, e.newspaper, d.word, sum(d.count)::BIGINT AS count
        FROM events e JOIN daily_word_counts d ON d.day BETWEEN e.start_date AND e.article_end {condition}
        GROUP BY e.spike_id,e.newspaper,d.word ORDER BY e.spike_id,d.word""").fetchdf()
    detail = counts.merge(denominators, on=["spike_id", "newspaper"], validate="many_to_one")
    return detail, denominators


def detect_events(weekly, outlet_names=None):
    """Original detector: observed weeks, sample SD, strictly >, 7-day groups."""
    events, definitions = [], []
    outlet_names = outlet_names or {label: name for name, label in NEWSPAPERS.items()}
    for (outlet, category), group in weekly.groupby(["outlet", "category"], sort=True):
        group = group.sort_values("week").copy()
        if group.week.duplicated().any() or group.week.isna().any() or group.total_articles.le(0).any():
            raise ValueError("Invalid/duplicate observed weekly source")
        group["share"] = group.matching_articles / (group.total_articles + 1e-9)
        mean, std = group.share.mean(), group.share.std(ddof=1)
        threshold = mean + std if pd.notna(std) and std > 0 else float("inf")
        selected = group[group.share.gt(threshold)]
        definitions.append({"outlet": outlet, "category": category, "observed_weeks": len(group),
            "mean": float(mean), "sample_std": float(std) if pd.notna(std) else None,
            "threshold": float(threshold) if np.isfinite(threshold) else None,
            "first_week": str(group.week.min().date()), "last_week": str(group.week.max().date())})
        blocks = selected.week.diff().ne(pd.Timedelta(days=7)).cumsum()
        for _, event in selected.groupby(blocks):
            peak = event.loc[event.share.idxmax()]
            events.append({"spike_id": len(events), "start_date": event.week.min(), "end_date": event.week.max(),
                "peak_date": peak.week, "gpr_category": category,
                "newspaper": outlet_names[outlet]})
    return pd.DataFrame(events, columns=EVENT_KEYS), definitions


def regenerate_events(connection, folder, manifest):
    frames, sources = [], []
    for artifact, value, label in (("category_weekly", "category_articles", "category"),
                                    ("dictionary_weekly", "gpr_articles", "'overall'")):
        path = Path(manifest["outputs"][artifact]["parquet"])
        if not path.is_absolute():
            path = folder / path
        # Date cast in capture_gpr_spikes.py includes every instant on Dec31.
        # The aggregate study_half_open policy is the exact same article scope.
        frame = connection.execute(f"""SELECT week, outlet, {label} AS category,
            {value} AS matching_articles, total_articles FROM read_parquet(?)
            WHERE dictionary='president_original7' AND scope='outlet'
              AND endpoint_policy='study_half_open' AND outlet IN (?,?)""",
            [str(path), *NEWSPAPERS.values()]).fetchdf()
        frames.append(frame)
        stat = path.stat()
        sources.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    weekly = pd.concat(frames, ignore_index=True)
    weekly["week"] = pd.to_datetime(weekly.week)
    if weekly.empty:
        raise ValueError("No original-president weekly counts available for newspaper event detection")
    events, definitions = detect_events(weekly)
    return events, definitions, sources


def old_draft_president_check(connection, args, manifest):
    destination = args.output_dir / "older_draft_president"
    destination.mkdir(parents=True, exist_ok=True)
    text = args.older_thesis.read_text()
    match = re.search(r"GPR index spikes are:\s*(.*?)\.\s*While", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("Cannot locate supplied older thesis president top-ten claim")
    claimed = [word.strip() for word in match[1].split("、")]
    if len(claimed) != 10 or len(set(claimed)) != 10:
        raise ValueError("Expected ten distinct words in supplied older thesis claim")
    path = Path(manifest["outputs"]["dictionary_weekly"]["parquet"])
    if not path.is_absolute():
        path = args.aggregate_dir / path
    weekly = connection.execute("""SELECT week,'three_outlets' AS outlet,'overall' AS category,
        gpr_articles AS matching_articles,total_articles FROM read_parquet(?)
        WHERE dictionary='president_original7' AND scope='three_outlets' AND endpoint_policy='study_half_open'""", [str(path)]).fetchdf()
    weekly["week"] = pd.to_datetime(weekly.week)
    events, definitions = detect_events(weekly, {"three_outlets": "Overall Taiwan - Overall"})
    events.to_csv(destination / "rebuilt_overall_event_ranges.csv", index=False)
    write_json(destination / "event_definitions.json", definitions)
    archived = pd.read_csv(args.decomposition)
    archived = archived[archived.newspaper.eq("Overall Taiwan - Overall") & archived.gpr_category.eq("overall")]
    reference_events = archived[EVENT_KEYS].drop_duplicates()
    for key in ("start_date", "end_date", "peak_date"):
        reference_events[key] = pd.to_datetime(reference_events[key])
    event_keys = [key for key in EVENT_KEYS if key != "spike_id"]
    compared, check = compare_frames(reference_events.assign(event_count=1), events.assign(event_count=1), event_keys, ["event_count"])
    compared.to_csv(destination / "archived_event_range_comparison.csv", index=False)
    detail, denominators = calculate_variant(connection, events, False)
    counts = detail.groupby("word", as_index=False)["count"].sum().sort_values(["count", "word"], ascending=[False, True])
    counts["rank"] = range(1, len(counts) + 1)
    counts["in_printed_wordset"] = counts.word.isin(claimed)
    detail.to_csv(destination / "rebuilt_word_event_counts.csv", index=False)
    counts.to_csv(destination / "rebuilt_word_counts.csv", index=False)
    denominators.to_csv(destination / "event_denominators.csv", index=False)
    actual = counts.head(10).word.tolist()
    report = {"claim_source": str(args.older_thesis.resolve()), "claim_sha256": file_hash(args.older_thesis),
        "claim_interpretation": "The prose lists ten most frequent words without explicitly asserting rank order; assess set membership separately from listed order",
        "counting": "Sum of binary matching-article counts across independently rebuilt original-president overall spike events; no repeated-occurrence weighting",
        "archived_event_count": len(reference_events), "rebuilt_event_count": len(events), "event_range_comparison": check,
        "events_without_articles": int(denominators.total_articles.eq(0).sum()), "claimed_words_in_prose_order": claimed,
        "rebuilt_top10_in_frequency_order": actual, "wordset_matches": set(actual) == set(claimed),
        "listed_order_matches_frequency_rank": actual == claimed,
        "missing_claimed_words": sorted(set(claimed) - set(actual)), "additional_rebuilt_top10": sorted(set(actual) - set(claimed))}
    write_json(destination / "summary.json", report)
    return report


def run(args):
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_audit, events, reference_detail = audit_reference(args, args.output_dir / "reference_audit")
    manifest_path = args.aggregate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mode = completion_mode(manifest, args.allow_incomplete)
    vocabulary_path = Path(manifest["input_dir"]) / "word_vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text())
    if object_hash(vocabulary["words"]) != vocabulary["sha256"] or vocabulary["sha256"] != manifest["vocabulary_sha256"]:
        raise ValueError("Vocabulary hash mismatch")
    words = dictionary_words(args.dictionary)
    if any(word.lower() != word or word.upper() != word for word in words):
        raise ValueError("Historical decomposition is case-insensitive; rebuilt presence IDs suffice only for case-invariant words")
    word_ids = {i: word for i, word in enumerate(vocabulary["words"]) if word in words}
    if set(word_ids.values()) != words:
        raise ValueError("Dictionary words missing from rebuilt vocabulary")
    if manifest.get("corpus", {}).get("warc_date_timezone") != "UTC":
        raise ValueError("Aggregate crawl timestamps must retain UTC semantics")
    producer_path = Path(manifest["input_dir"]) / "manifest.json"
    producer = json.loads(producer_path.read_text())
    frozen = {str(Path(row["path"]).resolve()): row["sha256"] for row in producer["dependencies"]["dictionary_files"]}
    if frozen.get(str(args.dictionary.resolve())) != file_hash(args.dictionary):
        raise ValueError("Dictionary differs from raw-match input")
    summary = {"status": "running", "mode": mode, "reference_audit": reference_audit,
               "calendar_timezone": manifest["calendar_timezone"], "dictionary_words": len(words),
               "event_selection": "Compare recovered historical ranges with independently regenerated original-president per-outlet category and overall ranges; analyze both sets separately",
               "event_detection": "Article calendar dates 2017-01-01 through 2024-12-31 inclusive (source casts timestamp to date); include partial endpoint weeks; observed weekly count/(total+1e-9); mean + sample SD; strictly greater; consecutive 7-day groups; earliest peak on ties",
               "counting": "Binary article word presence; denominator all accepted articles in window, including title-only; last Monday extended through Sunday; no GPR article filter",
               "averaging": "Positive-word events only, mean(count/total_articles) separately by newspaper event label; overlapping category events remain separate observations",
               "historical_pooled_windows": "All three outlets pooled in each labeled newspaper event, preserving decompose_spikes.py's dropped-outlet behavior",
               "corrected_outlet_conditioned": "Same event ranges and positive-event mean, articles filtered to event's named outlet",
               "ratio": "Liberty mean/(China mean+1e-6); also report plain China/Liberty to audit reversed thesis footnote",
               "script_sha256": file_hash(__file__),
               "comparison_helper_sha256": file_hash(Path(__file__).with_name("compare_raw_word_lifts.py")),
               "checks": []}
    write_json(args.output_dir / "run_summary.json", summary)
    events.to_csv(args.output_dir / "historical_event_ranges.csv", index=False)
    connection = duckdb.connect(manifest["database"], read_only=True)
    try:
        connection.execute(f"SET threads={args.threads}")
        connection.execute("SET memory_limit=?", [args.memory_limit])
        connection.execute("SET temp_directory=?", [str(args.output_dir / "duckdb_temp")])
        rebuilt_events, event_definitions, weekly_sources = regenerate_events(connection, args.aggregate_dir, manifest)
        rebuilt_events.to_csv(args.output_dir / "rebuilt_event_ranges.csv", index=False)
        write_json(args.output_dir / "event_detection_definitions.json", event_definitions)
        summary["weekly_source_artifacts"] = weekly_sources
        event_keys = [key for key in EVENT_KEYS if key != "spike_id"]
        event_comparison, event_check = compare_frames(events.assign(event_count=1), rebuilt_events.assign(event_count=1), event_keys, ["event_count"])
        event_comparison.to_csv(args.output_dir / "event_range_comparison.csv", index=False)
        summary["checks"].append({"artifact": "independently_detected_event_ranges", **event_check})
        prepare_daily(connection, manifest["calendar_timezone"], word_ids)
        claims = printed_claims(args.thesis)
        reference_ratios = appearance_ratios(reference_detail)
        summary["variants"] = []
        for range_source, selected_events in (("archived_ranges", events), ("rebuilt_ranges", rebuilt_events)):
            for variant, outlet_filter in (("historical_pooled_windows", False), ("corrected_outlet_conditioned", True)):
                destination = args.output_dir / range_source / variant
                destination.mkdir(parents=True, exist_ok=True)
                detail, denominators = calculate_variant(connection, selected_events, outlet_filter)
                ratios = appearance_ratios(detail)
                printed = compare_printed(claims, ratios)
                detail.to_csv(destination / "word_event_counts.csv", index=False)
                denominators.to_csv(destination / "event_denominators.csv", index=False)
                ratios.to_csv(destination / "word_ratios.csv", index=False)
                top_words(ratios).to_csv(destination / "top20_words.csv", index=False)
                printed.to_csv(destination / "printed_thesis_comparison.csv", index=False)
                summary["variants"].append({"range_source": range_source, "variant": variant,
                    "event_count": len(selected_events), "events_without_articles": int(denominators.total_articles.eq(0).sum()),
                    "positive_word_events": len(detail), "printed_ratio_matches": int(printed.producer_direction_matches.sum()),
                    "printed_footnote_direction_matches": int(printed.footnote_direction_matches.sum()),
                    "printed_rank_matches": int(printed.rank_matches.sum())})
                ratio_metrics = ["ratio_liberty_china", "china_mean_appearance_rate", "liberty_mean_appearance_rate",
                                 "china_total_word_count", "liberty_total_word_count", "china_positive_spike_count", "liberty_positive_spike_count"]
                ratio_comparison, ratio_check = compare_frames(reference_ratios, ratios, ["word"], ratio_metrics)
                ratio_comparison.to_csv(destination / "saved_ratios_comparison.csv", index=False)
                summary["checks"].append({"artifact": "all_word_ratios", "range_source": range_source, "variant": variant, **ratio_check})
                if not outlet_filter and range_source == "archived_ranges":
                    joined, check = compare_frames(reference_detail, detail, ["spike_id", "word"], ["count", "total_articles"])
                    joined.to_csv(destination / "saved_decomposition_comparison.csv", index=False)
                    summary["checks"].append({"artifact": "historical_positive_word_event_counts", **check})
        summary["older_draft_president"] = old_draft_president_check(connection, args, manifest)
        summary.update(status="complete", elapsed_seconds=round(time.monotonic() - started, 3))
    except BaseException as error:
        summary.update(status="failed", error=str(error), elapsed_seconds=round(time.monotonic() - started, 3))
        raise
    finally:
        connection.close()
        sources = [manifest_path, vocabulary_path, producer_path, args.dictionary, args.thesis, args.older_thesis, args.decomposition,
                   args.reference_dir / "word_appearance_ratios.csv", args.reference_dir / "top_words_analysis.csv"]
        recovery_provenance = ROOT / "replication/raw_rebuild/word_lifts/additional_recovery_provenance.json"
        if recovery_provenance.exists():
            sources.append(recovery_provenance)
        summary["sources"] = [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in sources]
        write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("sources",)}, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=ROOT / "replication/raw_rebuild/indices")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replication/raw_rebuild/newspaper_words")
    parser.add_argument("--decomposition", type=Path, default=REFERENCES / "spike_decomposition/spike_decomposition_taiwan_president_20250721001008.csv")
    parser.add_argument("--reference-dir", type=Path, default=REFERENCES / "cross_newspaper_analysis")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "data/dictionaries/taiwan_gpr_dict_president.json")
    parser.add_argument("--thesis", type=Path, default=ROOT / "replication/reference_thesis/contents/chapter03.tex")
    parser.add_argument("--older-thesis", type=Path, default=ROOT.parent / "NTU-Thesis-LaTeX-Template/contents/chapter06.tex")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.reference_only:
        print(json.dumps(audit_reference(arguments, arguments.output_dir)[0], ensure_ascii=False))
    else:
        run(arguments)
