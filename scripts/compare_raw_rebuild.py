#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pandas==2.2.3", "numpy==2.2.3", "pyarrow==20.0.0", "matplotlib==3.10.1"]
# ///
"""Compare a completed Stage1-Parquet article rebuild with preserved caches.

The input contract is documented in CONTRACT below. No source caches or rebuilt
aggregates are modified. Outer joins expose missing dates and dictionaries.
Incomplete/source-sample runs require --allow-incomplete and remain diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from replicate_thesis_indices import Replication as CachedAnalysis
from replicate_thesis_indices import json_value

ROOT = Path(__file__).resolve().parents[1]
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2025-01-01")
OUTLETS = ["China Times (中國時報)", "Liberty Times (自由時報)", "United Daily (聯合報)"]
MISSING_CATEGORY_DICTIONARY = "step2_gemini-2.5-pro_stable8cat_run8"
NEWSPAPERS = [(OUTLETS[0], "China Times", "china", 68), (OUTLETS[1], "Liberty Times", "liberty", 44),
              (OUTLETS[2], "United Daily", "united", 74)]
SPIKE_COLUMNS = ["start_date", "end_date", "peak_date", "peak_value", "duration_weeks", "mean_spike_value", "threshold"]
THESIS_UNIQUE_PEAKS = ["2021-10-04", "2021-12-06", "2022-05-23", "2022-10-10", "2022-10-24",
                      "2023-01-09", "2023-03-13", "2023-12-11", "2024-01-15", "2024-11-04", "2024-12-02"]
CONTRACT = {
    "manifest": {"status": "complete required for final comparison", "complete": "bool: aggregation finished",
                 "input_complete": "bool: all required parser inputs verified", "source_complete": "bool: parsing finished",
                 "source_mode": "full or pilot", "artifacts": "logical name to relative path"},
    "dictionary_weekly": ["week", "scope", "outlet", "dictionary", "total_articles", "gpr_articles", "endpoint_policy"],
    "category_weekly": ["week", "scope", "outlet", "dictionary", "category", "total_articles", "category_articles", "endpoint_policy"],
    "article_counts_daily_optional": ["day", "outlet", "total_articles"],
    "scope": ["all_outlets", "three_outlets", "outlet"],
    "endpoint_policy": ["full_cache", "study_half_open", "legacy_end_midnight"],
    "scope_rules": "full_cache includes all known parsed dates. study_half_open filters calendar-local article timestamps 2017<=t<2025. legacy_end_midnight filters 2017<=t<=2024-12-31T00:00:00. Calendar timezone comes from the aggregate manifest; raw article timestamps remain UTC. Dates in weekly files are Monday labels.",
    "dictionary_ids": "step1_* / step2_* original filenames; president_original7 or taiwan_gpr_dict_president; lau or yeu_tong_lau_2024",
}


def normalized_dictionary(name):
    aliases = {"original_president": "president_original7", "original7": "president_original7",
               "taiwan_gpr_dict_president": "president_original7", "yeu_tong_lau_2024": "lau",
               "lau_overall_taiwan_recalculated": "lau", "lau_2024": "lau"}
    return aliases.get(str(name), str(name))


def canonical_category(name):
    labels = {
        "Military Escalation": "軍事升級", "Diplomatic Isolation": "外交孤立",
        "Economic Coercion": "經濟脅迫", "Regional Security": "區域安全",
        "Internal Political Instability": "內部政治不穩定", "Gray Zone Operations": "灰色地帶行動",
        "US-China-Taiwan Relations": "美中台關係", "Cyber and Information Warfare": "網路與資訊戰",
    }
    for english, chinese in labels.items():
        if name in (english, chinese, english.lower().replace("-", "_").replace(" ", "_"),
                    f"{english} ({chinese})"):
            return english
    return name


def study_weeks(frame, partial=False):
    start = pd.Timestamp("2016-12-26") if partial else START
    return frame.loc[(frame.week >= start) & (frame.week < END)].copy()


def assess_completion(manifest, allow_incomplete=False):
    """Missing completion evidence never defaults to a full rebuild."""
    complete = (manifest.get("status") == "complete" and manifest.get("source_mode") == "full"
                and all(manifest.get(key) is True for key in ("complete", "input_complete", "source_complete")))
    if not complete and not allow_incomplete:
        raise ValueError("A full comparison requires status='complete', source_mode='full', and complete=input_complete=source_complete=true; use --allow-incomplete only for a labeled diagnostic.")
    return "full_raw_rebuild" if complete else "diagnostic_incomplete_or_sample"


def compare_frames(reference, rebuilt, keys, metrics, tolerance=1e-12):
    """Return a coverage-aware outer join and summary; never fill absent rows."""
    required = set(keys) | set(metrics)
    for name, frame in [("reference", reference), ("rebuilt", rebuilt)]:
        if not required <= set(frame):
            raise ValueError(f"{name} lacks {sorted(required - set(frame))}")
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate {name} comparison keys: {keys}")
    merged = reference[list(keys) + list(metrics)].merge(
        rebuilt[list(keys) + list(metrics)], on=keys, how="outer",
        suffixes=("_reference", "_rebuilt"), indicator="coverage", validate="one_to_one")
    metric_reports = {}
    mismatched_rows = pd.Series(False, index=merged.index)
    both = merged.coverage.eq("both")
    for metric in metrics:
        old, new = merged[metric + "_reference"], merged[metric + "_rebuilt"]
        merged[metric + "_delta"] = new - old
        unequal_nulls = both & old.isna().ne(new.isna())
        differences = merged.loc[both, metric + "_delta"].abs()
        is_count = metric.endswith("articles") or metric.endswith("count") or metric.endswith("weeks")
        threshold = 0 if is_count else tolerance
        mismatch = both & (merged[metric + "_delta"].abs().gt(threshold) | unequal_nulls)
        mismatched_rows |= mismatch
        metric_reports[metric] = {
            "max_absolute_delta": float(differences.max()) if differences.notna().any() else None,
            "paired_nonnull_rows": int((both & old.notna() & new.notna()).sum()),
            "unequal_null_rows": int(unequal_nulls.sum()), "mismatched_rows": int(mismatch.sum()),
            "sum_reference_matched_rows": float(old[both].sum(min_count=1)) if old[both].notna().any() else None,
            "sum_rebuilt_matched_rows": float(new[both].sum(min_count=1)) if new[both].notna().any() else None,
            "tolerance": threshold,
        }
    merged["numeric_mismatch"] = mismatched_rows
    missing = int(merged.coverage.eq("left_only").sum())
    added = int(merged.coverage.eq("right_only").sum())
    summary = {"reference_rows": len(reference), "rebuilt_rows": len(rebuilt),
               "paired_rows": int(both.sum()), "reference_only_rows": missing, "rebuilt_only_rows": added,
               "numeric_mismatch_rows": int(mismatched_rows.sum()), "metrics": metric_reports,
               "status": "unavailable" if len(merged) == 0 else
                         "matched" if not (missing or added or mismatched_rows.any()) else "different"}
    return merged, summary


def panel_for(frame, names):
    selected = frame[frame.dictionary.isin(names)]
    if selected.duplicated(["week", "dictionary"]).any():
        raise ValueError("Panel has duplicate dictionary/week observations")
    return selected.pivot(index="week", columns="dictionary", values="gpr_share").reindex(columns=names).sort_index()


def median_frame(panel, expected_count):
    counts = panel.notna().sum(axis=1)
    median = panel.median(axis=1).where(counts == expected_count)
    result = pd.DataFrame({"median_raw": median, "dictionary_count": counts})
    result["median_scaled"] = median / median.mean() * 100 if median.notna().any() else np.nan
    return result.rename_axis("week").reset_index()


def coverage_summary(series, calendar):
    """Absent observations, present nulls, and present zero indices are distinct."""
    if not series.index.is_unique:
        raise ValueError("Duplicate newspaper week labels")
    selected = series.loc[series.index.isin(calendar)]
    absent = ~calendar.isin(selected.index)
    flags = pd.Series(absent, index=calendar)
    groups = flags.ne(flags.shift(fill_value=False)).cumsum()
    gaps = [{"start": rows.index.min(), "end": rows.index.max(), "weeks": len(rows)}
            for _, rows in flags[flags].groupby(groups[flags])]
    return {"calendar_weeks": len(calendar), "observed_weeks": len(selected), "missing_weeks": int(absent.sum()),
            "missing_pct": float(absent.mean() * 100) if len(calendar) else None,
            "null_index_weeks": int(selected.isna().sum()), "zero_index_weeks": int(selected.eq(0).sum()),
            "longest_absent_run": max((gap["weeks"] for gap in gaps), default=0), "gaps": gaps}


def spike_overlap_pairs(step2, lau):
    """Inclusive overlap of event-label ranges; a Lau event may match several Step2 events."""
    columns = ["step2_idx", "lau_idx", "overlap_days", "overlap_pct_step2", "overlap_pct_lau", "peak_date_diff"]
    rows = []
    for left_id, left in step2.reset_index(drop=True).iterrows():
        for right_id, right in lau.reset_index(drop=True).iterrows():
            start, end = max(left.start_date, right.start_date), min(left.end_date, right.end_date)
            if start <= end:
                overlap = (end - start).days + 1
                rows.append([left_id, right_id, overlap, overlap / ((left.end_date-left.start_date).days+1)*100,
                             overlap / ((right.end_date-right.start_date).days+1)*100,
                             abs((left.peak_date-right.peak_date).days)])
    return pd.DataFrame(rows, columns=columns)


def add_undated_corpus_counts(daily, corpus):
    """Reconcile dated aggregate groups with complete article-table outlet counts."""
    daily = daily[["day", "outlet", "total_articles"]].copy()
    corpus = corpus.copy()
    corpus["outlet"] = corpus.outlet.fillna("__NULL__")
    if corpus.outlet.duplicated().any():
        raise ValueError("Duplicate corpus outlet counts")
    present_nulls = daily[daily.day.isna()].set_index("outlet").total_articles
    additions = []
    for row in corpus.itertuples(index=False):
        actual = int(present_nulls.get(row.outlet, 0))
        if actual and actual != row.undated_articles:
            raise ValueError("Null-date daily counts differ from complete corpus counts")
        if not actual and row.undated_articles:
            additions.append({"day": pd.NaT, "outlet": row.outlet, "total_articles": row.undated_articles})
    if additions:
        daily = pd.concat([daily, pd.DataFrame(additions)], ignore_index=True)
    total = daily.groupby("outlet", as_index=False).total_articles.sum()
    _, summary = compare_frames(corpus, total, ["outlet"], ["total_articles"])
    if summary["status"] != "matched":
        raise ValueError("Daily plus undated counts do not cover the complete article corpus")
    return daily


def pairwise_summary(panel):
    corr = panel.corr().to_numpy()
    pairs = corr[np.triu_indices(len(corr), 1)]
    return {"dictionary_count": len(panel.columns), "weeks": len(panel),
            "first_week": panel.index.min() if len(panel) else None,
            "last_week": panel.index.max() if len(panel) else None,
            "pair_count": int(np.isfinite(pairs).sum()), "expected_pair_count": len(pairs),
            "mean_pairwise_correlation": float(np.nanmean(pairs)) if np.isfinite(pairs).any() else None,
            "incomplete_weeks": int(panel.isna().any(axis=1).sum())}


class Comparison:
    def __init__(self, args):
        self.args = args
        self.folder = args.aggregate_dir.resolve()
        self.output = args.output_dir.resolve()
        self.data = args.data_root.resolve()
        self.sources = {}
        self.checks = []
        self.notes = []
        self.spike_frames = {}
        self.spike_definitions = []
        self.manifest = json.loads(self.source(self.folder / "manifest.json").read_text())
        self.mode = assess_completion(self.manifest, args.allow_incomplete)
        self.output.mkdir(parents=True, exist_ok=True)

    def source(self, path):
        path = path.resolve()
        if path not in self.sources:
            self.sources[path] = {"path": str(path), "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return path

    def write_json(self, name, data):
        (self.output / name).write_text(json.dumps(json_value(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def artifact(self, name, required=True):
        artifacts = self.manifest.get("artifacts", self.manifest.get("outputs", {}))
        logical_name = "article_daily" if name == "article_counts_daily" and "article_daily" in artifacts else name
        value = artifacts.get(logical_name, logical_name + ".csv")
        if isinstance(value, dict):
            value = value.get("path", value.get("file", value.get("parquet", value.get("csv"))))
        path = self.folder / value if value else self.folder / (name + ".csv")
        if not path.exists():
            if name == "article_counts_daily" and (self.folder / "article_daily.csv").exists():
                path = self.folder / "article_daily.csv"
            elif path.suffix == ".csv" and path.with_suffix(".parquet").exists():
                path = path.with_suffix(".parquet")
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing aggregate artifact {name}: {path}")
            return None
        return self.read_frame(path)

    def read_frame(self, path):
        path = self.source(path)
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    def normalized(self, frame, category=False):
        frame = frame.copy()
        frame["week"] = pd.to_datetime(frame.week).dt.as_unit("ns")
        frame["dictionary"] = frame.dictionary.map(normalized_dictionary)
        frame["outlet"] = frame.get("outlet", pd.Series("", index=frame.index)).fillna("")
        frame["endpoint_policy"] = frame.get("endpoint_policy", pd.Series("full_cache", index=frame.index))
        if category:
            frame["category"] = frame.category.map(canonical_category)
            frame["category_share"] = frame.category_articles / frame.total_articles.replace(0, np.nan)
        else:
            frame["gpr_share"] = frame.gpr_articles / frame.total_articles.replace(0, np.nan)
        return frame

    def compare(self, name, reference, rebuilt, keys, metrics, scope, note=None):
        merged, summary = compare_frames(reference, rebuilt, keys, metrics)
        merged.to_csv(self.output / (name + ".csv"), index=False)
        self.checks.append({"id": name, "scope": scope, "comparison_mode": self.mode,
                            "details_file": name + ".csv", **summary, "note": note})
        return merged, summary

    def dictionary_names(self):
        valid = {"step1": [], "step2": []}
        for path in sorted((self.data / "dictionaries/all_dictionaries").glob("step[12]_*.json")):
            data = json.loads(self.source(path).read_text())
            if all(all(term.strip() in data["word_sets"] for term in query.split(" AND ")) for query in data["categories"].values()):
                valid[path.stem[:5]].append(path.stem)
        if len(valid["step1"]) != 35 or len(valid["step2"]) != 40:
            raise ValueError("Expected exact valid35 Step1/40 Step2 reference dictionaries")
        return valid

    def reference_series(self, scope, valid):
        directory, suffix = (("gpr_timeseries", "_W_overall.parquet") if scope == "all_outlets"
                             else ("gpr_timeseries_3outlets", "_W_overall_3outlets.parquet"))
        names = valid["step1"] + valid["step2"]
        if scope == "three_outlets":
            names = names + ["lau_overall_taiwan_recalculated"]
        frames = []
        for name in names:
            df = self.read_frame(self.data / directory / (name + suffix))
            df = df.rename(columns={"warc_date_bucketed": "week", "total_gpr_count": "gpr_articles"})
            df["week"] = pd.to_datetime(df.week).dt.as_unit("ns")
            df["dictionary"] = normalized_dictionary(name)
            df["gpr_share"] = df.gpr_articles / df.total_articles.replace(0, np.nan)
            frames.append(df[["week", "dictionary", "total_articles", "gpr_articles", "gpr_share"]])
        return pd.concat(frames, ignore_index=True)

    def indices(self, rebuilt, valid):
        pairwise_rows = []
        for scope in ("all_outlets", "three_outlets"):
            reference = self.reference_series(scope, valid)
            raw = rebuilt[(rebuilt.scope == scope) & (rebuilt.endpoint_policy == "full_cache")]
            raw = raw[raw.dictionary.isin(reference.dictionary.unique())]
            for period, old, new in [("full_cache", reference, raw),
                                      ("strict_study_week_labels", study_weeks(reference), study_weeks(raw))]:
                self.compare(f"dictionary_weekly_{scope}_{period}", old, new,
                             ["week", "dictionary"], ["total_articles", "gpr_articles", "gpr_share"],
                             f"{scope}; {period}; counts and raw shares; same week-label filter")
                for step in ("step1", "step2"):
                    old_panel, new_panel = panel_for(old, valid[step]), panel_for(new, valid[step])
                    pairwise_rows.append({"scope": scope, "period": period, "step": step,
                                          "reference": pairwise_summary(old_panel), "rebuilt": pairwise_summary(new_panel)})
                    if step == "step2":
                        old_median, new_median = median_frame(old_panel, 40), median_frame(new_panel, 40)
                        self.compare(f"step2_median_{scope}_{period}", old_median, new_median, ["week"],
                                     ["median_raw", "median_scaled", "dictionary_count"],
                                     f"{scope}; {period}; 40 nonmissing dictionaries required at every week")
                        if scope == "three_outlets":
                            self.spikes(old_median, new_median, "step2", period)
                            if period == "strict_study_week_labels":
                                self.plot(old_median, new_median)
                                self.lau_correlations(old_median, new_median, old, new)
                if scope == "three_outlets":
                    old_lau = old.loc[old.dictionary.eq("lau"), ["week", "gpr_share"]].rename(columns={"gpr_share": "median_raw"})
                    new_lau = new.loc[new.dictionary.eq("lau"), ["week", "gpr_share"]].rename(columns={"gpr_share": "median_raw"})
                    self.spikes(old_lau, new_lau, "lau", period)
            # Article-date filtered indices are a distinct policy, not silently
            # equated with cached full-week counts at the two boundary weeks.
            article_filtered = rebuilt[(rebuilt.scope == scope) & (rebuilt.endpoint_policy == "study_half_open")]
            article_filtered = article_filtered[article_filtered.dictionary.isin(reference.dictionary.unique())]
            if len(article_filtered):
                self.compare(f"article_date_filter_boundary_effect_{scope}", study_weeks(reference, partial=True),
                             article_filtered, ["week", "dictionary"], ["total_articles", "gpr_articles", "gpr_share"],
                             f"{scope}; cached full-week counts vs rebuilt article-date half-open study selection",
                             "Differences at 2016-12-26 and 2024-12-30 can follow endpoint selection rather than classification changes.")
        self.write_json("pairwise_correlations.json", pairwise_rows)
        pd.json_normalize(pairwise_rows).to_csv(self.output / "pairwise_correlations.csv", index=False)

    def spikes(self, reference, rebuilt, name, period):
        frames = []
        for source, df in (("reference", reference), ("rebuilt", rebuilt)):
            values = df.set_index("week").median_raw
            detected, threshold = CachedAnalysis.detect_spikes(values) if len(values) else (pd.DataFrame(), None)
            if detected.empty:
                detected = pd.DataFrame(columns=SPIKE_COLUMNS)
            detected.to_csv(self.output / f"{source}_{name}_spikes_{period}.csv", index=False)
            self.spike_frames[(source, name, period)] = detected
            self.spike_definitions.append({"source": source, "index": name, "period": period,
                "threshold": threshold, "observed_weeks": len(values), "first_week": values.index.min() if len(values) else None,
                "last_week": values.index.max() if len(values) else None, "spike_count": len(detected)})
            frames.append(detected)
        self.compare(f"{name}_spikes_{period}", frames[0], frames[1],
                     ["start_date", "end_date", "peak_date"], ["peak_value", "duration_weeks"],
                     f"{period}; mean+sample std; absent calendar weeks split events")
        if period == "full_cache":
            cache_name = "step2_median_spikes.csv" if name == "step2" else "lau_2024_spikes.csv"
            original = self.read_frame(self.data / "gpr_spikes/unified_detection_3outlets" / cache_name)
            for col in ["start_date", "end_date", "peak_date"]:
                original[col] = pd.to_datetime(original[col])
            self.compare(f"{name}_spikes_vs_original_archived_events", original, frames[1],
                         ["start_date", "end_date", "peak_date"], ["peak_value", "duration_weeks"],
                         "Original archived event ranges versus full rebuilt series; archive includes events outside 2017–2024")

    def unique_spikes(self):
        provenance = "Inclusive event-label range overlap inferred from saved matched_spikes.csv; original comparison producer was not recovered. All archived pair IDs and overlap metrics are checked below. A shared endpoint overlaps; nearby nonoverlapping events do not; multiple Step2 events can match one Lau event."
        archived = {}
        for name, filename in [("step2", "step2_median_spikes.csv"), ("lau", "lau_2024_spikes.csv")]:
            frame = self.read_frame(self.data / "gpr_spikes/unified_detection_3outlets" / filename)
            for column in ["start_date", "end_date", "peak_date"]:
                frame[column] = pd.to_datetime(frame[column])
            archived[name] = frame
        pairs = spike_overlap_pairs(archived["step2"], archived["lau"])
        old_pairs = self.read_frame(self.data / "gpr_spikes/spike_comparison/matched_spikes.csv")
        _, evidence = self.compare("unique_spike_overlap_rule_archived_evidence", old_pairs, pairs,
            ["step2_idx", "lau_idx"], ["overlap_days", "overlap_pct_step2", "overlap_pct_lau", "peak_date_diff"],
            "Archived event ranges only; provenance validation, not a new raw-data result", provenance)
        archived_unique = self.read_frame(self.data / "gpr_spikes/spike_comparison/step2_unique_spikes.csv")
        for column in ["start_date", "end_date", "peak_date"]:
            archived_unique[column] = pd.to_datetime(archived_unique[column])
        self.compare("unique_spike_rule_archived_unique_events", archived_unique,
            archived["step2"].loc[~archived["step2"].index.isin(pairs.step2_idx)],
            ["start_date", "end_date", "peak_date"], ["peak_value", "duration_weeks"],
            "Archived unique-event selection reproduced from archived ranges only", provenance)
        summaries = []
        for period in ("full_cache", "strict_study_week_labels"):
            unique = {}
            for source in ("reference", "rebuilt"):
                step2, lau = (self.spike_frames[(source, name, period)] for name in ("step2", "lau"))
                matches = spike_overlap_pairs(step2, lau)
                matches.to_csv(self.output / f"{source}_spike_overlap_pairs_{period}.csv", index=False)
                unique[source] = step2.loc[~step2.index.isin(matches.step2_idx)].copy()
                unique[source].to_csv(self.output / f"{source}_step2_unique_spikes_{period}.csv", index=False)
                summaries.append({"source": source, "period": period, "step2_spikes": len(step2), "lau_spikes": len(lau),
                                  "overlap_pairs": len(matches), "step2_unique_count": len(unique[source]),
                                  "step2_unique_peak_dates": unique[source].peak_date.tolist()})
            self.compare(f"step2_unique_spikes_{period}", unique["reference"], unique["rebuilt"],
                         ["start_date", "end_date", "peak_date"], ["peak_value", "duration_weeks"], period, provenance)
            claimed = pd.DataFrame({"peak_date": pd.to_datetime(THESIS_UNIQUE_PEAKS), "peak_count": 1})
            for source in ("reference", "rebuilt"):
                actual = unique[source][["peak_date"]].assign(peak_count=1)
                self.compare(f"{source}_step2_unique_peaks_vs_thesis_{period}", claimed, actual, ["peak_date"], ["peak_count"],
                             source + "; " + period + "; thesis chapter03.tex:81–91 lists eleven peak dates", provenance)
        self.write_json("unique_spike_summary.json", {"mode": self.mode, "rule_provenance": provenance,
            "archived_pairs_rule_status": evidence["status"], "thesis_peak_dates": THESIS_UNIQUE_PEAKS, "results": summaries})
        self.write_json("spike_definitions.json", {"mode": self.mode, "calendar_timezone": self.manifest.get("calendar_timezone", "unspecified"),
            "method": "Raw shares; strictly greater than mean + sample standard deviation; missing calendar Mondays split events; thresholds computed separately for each source/period.",
            "event_end_semantics": "End date is the last Monday label; article-level word analyses must extend through that week's Sunday.",
            "series": self.spike_definitions})

    def newspapers(self, rebuilt, valid):
        calendar = pd.date_range(START, END-pd.Timedelta(days=1), freq="W-MON")
        panels = {"reference": {}, "rebuilt": {}}
        coverage = {"reference": [], "rebuilt": []}
        claims = []
        for outlet, name, stem, missing_claim in NEWSPAPERS:
            original = self.read_frame(self.data / f"newspaper_gpr_step2/{stem}_step2_median_gpr.csv")
            original = original.rename(columns={original.columns[0]: "week", "median": "median_raw", "count": "dictionary_count"})
            original.week = pd.to_datetime(original.week)
            raw = rebuilt[(rebuilt.scope == "outlet") & (rebuilt.endpoint_policy == "full_cache") & rebuilt.outlet.eq(outlet)]
            recomputed = median_frame(panel_for(raw, valid["step2"]), 40)
            old, new = study_weeks(original), study_weeks(recomputed)
            self.compare(f"newspaper_{stem}_median_strict_study", old, new, ["week"],
                         ["median_raw", "median_scaled", "dictionary_count"],
                         "Per-outlet median of 40 raw weekly dictionary shares; scaling uses each full observed series, then strict study week-label filter")
            for source, frame in [("reference", old), ("rebuilt", new)]:
                series = frame.set_index("week").median_scaled
                panels[source][name] = series
                coverage[source].append({"newspaper": name, **coverage_summary(series, calendar)})
            actual = coverage["rebuilt"][-1]
            claims.append({"id": "missing_weeks_"+stem, "claimed": missing_claim, "observed": actual["missing_weeks"],
                           "status": "reproduced" if missing_claim == actual["missing_weeks"] else "different"})
            if stem == "united":
                claims.append({"id": "united_longest_absent_run", "claimed": 47, "observed": actual["longest_absent_run"],
                               "status": "reproduced" if actual["longest_absent_run"] == 47 else "different"})
        fields = ["calendar_weeks", "observed_weeks", "missing_weeks", "null_index_weeks", "zero_index_weeks", "longest_absent_run"]
        self.compare("newspaper_coverage", pd.DataFrame(coverage["reference"]), pd.DataFrame(coverage["rebuilt"]),
                     ["newspaper"], fields, "418 Monday labels from2017-01-02 through2024-12-30; absent rows, null indices and zero indices kept distinct")
        correlations = {}
        for source, series in panels.items():
            common = pd.DataFrame(series).dropna()
            common.to_csv(self.output / f"{source}_newspaper_common_weeks.csv", index_label="week")
            corr = common.corr()
            corr.to_csv(self.output / f"{source}_newspaper_correlations.csv")
            rows = []
            for left, right, claim in [("China Times", "Liberty Times", .475), ("China Times", "United Daily", .780),
                                       ("Liberty Times", "United Daily", .559)]:
                value = float(corr.loc[left, right])
                rows.append({"first": left, "second": right, "correlation": value, "common_weeks": len(common)})
                if source == "rebuilt":
                    claims.append({"id": "correlation_"+left+"_"+right, "claimed": claim, "observed": value,
                                   "status": "reproduced" if abs(value-claim) <= .0005 else "different", "common_weeks": len(common)})
            correlations[source] = pd.DataFrame(rows)
        self.compare("newspaper_correlations", correlations["reference"], correlations["rebuilt"], ["first", "second"],
                     ["correlation", "common_weeks"], "Pearson on intersection of nonnull study weeks across all three newspapers, matching create_clean_newspaper_comparison.py:157–184")
        self.write_json("newspaper_statistics.json", {"mode": self.mode, "calendar_timezone": self.manifest.get("calendar_timezone", "unspecified"),
                        "calendar_weeks": len(calendar), "coverage": coverage, "thesis_claims": claims})

    def lau_correlations(self, old_median, new_median, old, new):
        rows = []
        for label, median, individual in [("reference", old_median, old), ("rebuilt", new_median, new)]:
            paired = median.merge(individual.loc[individual.dictionary.eq("lau"), ["week", "gpr_share"]], on="week").dropna(subset=["median_raw", "gpr_share"])
            rows.append({"source": label, "paired_weeks": len(paired), "correlation": paired.median_raw.corr(paired.gpr_share)})
        self.write_json("step2_lau_correlation.json", {"thesis_claim": .611, "period": "strict_study_week_labels", "rows": rows})

    def categories(self, rebuilt, valid):
        long = self.read_frame(self.data / "category_gpr_indices/all_category_gpr_indices.csv")
        long["week"] = pd.to_datetime(long.week)
        long["dictionary"] = "step2_" + long.dictionary.str.removeprefix("step2_")
        long = long.rename(columns={"gpr_index": "category_share"})
        raw = rebuilt[(rebuilt.scope == "three_outlets") & (rebuilt.endpoint_policy == "legacy_end_midnight") & rebuilt.dictionary.isin(valid["step2"])]
        self.compare("step2_category_weekly_legacy_endpoint", long, raw,
                     ["week", "dictionary", "category"], ["category_share"],
                     "Three outlets; article timestamps 2017-01-01 <= t <= 2024-12-31T00:00:00",
                     "The original cache omits Gemini run8; additional rebuilt rows for it are expected coverage differences.")
        reference = long.groupby(["week", "category"], as_index=False).category_share.median()
        category_metrics = []
        for policy in ["study_half_open", "legacy_end_midnight"]:
            selected = rebuilt[(rebuilt.scope == "three_outlets") & (rebuilt.endpoint_policy == policy) & rebuilt.dictionary.isin(valid["step2"])]
            for mode, names in [("canonical40", valid["step2"]), ("legacy39", [n for n in valid["step2"] if n != MISSING_CATEGORY_DICTIONARY])]:
                chosen = selected[selected.dictionary.isin(names)]
                counts = chosen.groupby(["week", "category"]).dictionary.nunique()
                medians = chosen.groupby(["week", "category"]).category_share.median().where(counts == len(names)).rename("category_share").reset_index()
                self.compare(f"category_median_{mode}_{policy}", reference, medians,
                             ["week", "category"], ["category_share"], f"Rebuilt {mode}; {policy}; original saved category medians use39 dictionaries and legacy endpoint")
                medians.to_csv(self.output / f"rebuilt_category_medians_{mode}_{policy}.csv", index=False)
                sums = medians.groupby("category").category_share.sum(min_count=1)
                category_metrics.extend({"mode": mode, "endpoint_policy": policy, "category": category,
                                         "sum_share": value, "share_of_category_sum_percent": value / sums.sum() * 100,
                                         "minimum_dictionary_count": int(counts.min()) if len(counts) else 0}
                                        for category, value in sums.items())
        pd.DataFrame(category_metrics).to_csv(self.output / "category_share_methods.csv", index=False)

    def original_and_lau(self, dictionaries, categories, audit):
        original = audit.get("original_president_labeled_reconstruction", {})
        if not original.get("weekly_original"):
            self.notes.append("No remote original-president weekly reference was provided.")
            return
        old = pd.DataFrame(original["weekly_original"]).rename(columns={"overall_gpr_articles": "gpr_articles"})
        old["week"] = pd.to_datetime(old.week)
        old["gpr_share"] = old.gpr_articles / old.total_articles
        selected = dictionaries[(dictionaries.scope == "three_outlets") & dictionaries.endpoint_policy.eq("study_half_open") & dictionaries.dictionary.eq("president_original7")]
        self.compare("original7_weekly_vs_historical_labeled_reconstruction", old, selected,
                     ["week"], ["total_articles", "gpr_articles", "gpr_share"],
                     "Three outlets; exact article-date study range; original7 dictionary",
                     "Reference is corrected historical labeled flags, not a previous independent article-text scan.")
        targets = {"Military Escalation": (33.48, "2022-08-01"), "Internal Political Instability": (32.90, "2024-01-08"),
                   "Diplomatic Crisis": (19.05, "2019-09-16"), "Foreign Intervention Risk": (12.35, "2022-08-01"),
                   "Economic Coercion": (10.81, "2018-08-20"), "Cyber and Information Warfare": (10.22, "2019-12-30"),
                   "Critical Infrastructure Threat": (.89, "2024-09-23")}
        raw_cats = categories[(categories.scope == "three_outlets") & categories.endpoint_policy.eq("study_half_open") & categories.dictionary.eq("president_original7")]
        reference_cats = []
        for i, category in enumerate(original["category_order"]):
            rows = old[["week", "total_articles", f"category_{i}_count"]].rename(columns={f"category_{i}_count": "category_articles"})
            rows["category"] = category
            rows["category_share"] = rows.category_articles / rows.total_articles
            reference_cats.append(rows)
        old_cats = pd.concat(reference_cats, ignore_index=True)
        self.compare("original7_category_weekly", old_cats, raw_cats,
                     ["week", "category"], ["total_articles", "category_articles", "category_share"],
                     "Original7 exact dictionary; study article-date selection")
        rates = []
        for category, (claimed, claimed_peak) in targets.items():
            rows = raw_cats[raw_cats.category.eq(category)]
            denominator = selected.gpr_articles.sum()
            observed = float(rows.category_articles.sum() / denominator * 100) if denominator else None
            peaks = rows.loc[rows.category_share.eq(rows.category_share.max()), "week"].dt.strftime("%Y-%m-%d").tolist()
            rates.append({"dictionary": "president_original7", "category": category, "claimed_percent": claimed,
                          "rebuilt_percent": observed, "matches_printed_rounding": observed is not None and round(observed, 2) == claimed,
                          "claimed_peak_week": claimed_peak, "rebuilt_peak_weeks": peaks, "peak_matches": claimed_peak in peaks,
                          "gpr_denominator": int(denominator)})
        lau_raw = categories[(categories.scope == "three_outlets") & categories.endpoint_policy.eq("study_half_open") & categories.dictionary.eq("lau")]
        lau_total = dictionaries[(dictionaries.scope == "three_outlets") & dictionaries.endpoint_policy.eq("study_half_open") & dictionaries.dictionary.eq("lau")].gpr_articles.sum()
        saved_lau = audit.get("lau", {}).get("scopes", {}).get("three_outlets_2017_2024", {})
        for category, group in lau_raw.groupby("category"):
            rates.append({"dictionary": "lau", "category": category,
                          "cached_category_count": saved_lau.get("category_counts", {}).get(category),
                          "rebuilt_category_count": int(group.category_articles.sum()), "gpr_denominator": int(lau_total),
                          "rebuilt_percent": float(group.category_articles.sum() / lau_total * 100) if lau_total else None,
                          "claimed_percent": 86 if category == "military_buildup" else None})
        self.write_json("original7_and_lau_claims.json", rates)

    def counts(self, daily, audit, corpus=None):
        if daily is None:
            self.notes.append("Daily corpus-count artifact unavailable; dictionary weekly denominators still compared.")
            return
        daily = daily.rename(columns={"date": "day"}).copy()
        if "scope" in daily:
            daily = daily[daily.scope.eq("outlet")]
        if "endpoint_policy" in daily:
            daily = daily[daily.endpoint_policy.eq("full_cache")]
        daily["day"] = pd.to_datetime(daily.day.replace("__NULL__", None))
        daily["outlet"] = daily.outlet.fillna("__NULL__")
        if daily.duplicated(["day", "outlet"]).any():
            raise ValueError("Duplicate day/outlet article counts")
        if corpus is not None:
            daily = add_undated_corpus_counts(daily, corpus)
        daily["year"] = daily.day.dt.year.astype("Int64").astype("string").fillna("__NULL__")
        daily["week"] = daily.day - pd.to_timedelta(daily.day.dt.weekday, unit="D")
        self.write_json("rebuilt_corpus_bounds.json", {
            "mode": self.mode, "rebuild_origin": "Local extracted Stage1 Parquet",
            "aggregate_scope": "Disjoint outlet groups, full_cache endpoint policy",
            "calendar_timezone": getattr(self, "manifest", {}).get("calendar_timezone", "unspecified"),
            "complete_corpus_outlet_counts_reconciled": corpus is not None,
            "total_articles": int(daily.total_articles.sum()),
            "first_observed_day": daily.day.min(), "last_observed_day": daily.day.max(),
            "observed_day_outlet_groups": len(daily), "outlet_count": int(daily.outlet.nunique()),
            "null_date_articles": int(daily.loc[daily.day.isna(), "total_articles"].sum()),
            "null_outlet_articles": int(daily.loc[daily.outlet.eq("__NULL__"), "total_articles"].sum()),
            "note": "Observed date bounds do not prove complete coverage; completion is determined by parser/aggregation manifests.",
        })
        for label, keys in [("day_outlet", ["day", "outlet"]), ("week_outlet", ["week", "outlet"]),
                            ("year_outlet", ["year", "outlet"]), ("outlet", ["outlet"])]:
            grouped = daily.groupby(keys, dropna=False, as_index=False).total_articles.sum()
            grouped.to_csv(self.output / f"rebuilt_article_counts_{label}.csv", index=False, na_rep="__NULL__")
            candidate = self.args.reference_counts_dir / f"article_counts_{label}.csv"
            if candidate.exists():
                old = self.read_frame(candidate)
                for key in ["day", "week"]:
                    if key in old:
                        old[key] = pd.to_datetime(old[key].replace("__NULL__", None))
                if "year" in old:
                    old["year"] = pd.to_numeric(old.year.replace("__NULL__", None)).astype("Int64").astype("string").fillna("__NULL__")
                old["outlet"] = old.outlet.fillna("__NULL__")
                self.compare(f"article_counts_{label}", old, grouped, keys, ["total_articles"], "Full corpus; day-level aggregate summed without dictionary repetition")
        selected = daily[(daily.day >= START) & (daily.day < END) & daily.outlet.isin(OUTLETS)]
        for label, query, keys in [("study_year_outlet", "three_outlet_year_counts_2017_2024", ["year", "outlet"]),
                                   ("study_outlet", "three_outlet_counts_2017_2024", ["outlet"])]:
            old = pd.DataFrame(audit.get("queries", {}).get(query, {}).get("rows", [])).rename(columns={"article_count": "total_articles"})
            if not old.empty:
                if "year" in old:
                    old["year"] = old.year.astype("Int64").astype("string")
                self.compare("article_counts_" + label, old, selected.groupby(keys, as_index=False).total_articles.sum(),
                             keys, ["total_articles"], "Three outlets; exact calendar-day study range; historical remote DB count reference")

    def plot(self, old, new):
        joined = old.set_index("week").join(new.set_index("week"), how="outer", lsuffix="_cached", rsuffix="_rebuilt").sort_index()
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        axes[0].plot(joined.index, joined.median_scaled_cached, label="Historical cache", color="#b56635", lw=1.5)
        axes[0].plot(joined.index, joined.median_scaled_rebuilt, label="Rebuilt from articles", color="#137b83", lw=1.2, alpha=.8, marker=".", markersize=3)
        axes[0].set_ylabel("Step2 median · mean = 100")
        axes[0].legend(frameon=False)
        axes[1].plot(joined.index, (joined.median_raw_rebuilt - joined.median_raw_cached) * 100,
                     color="#137b83", lw=1, marker=".", markersize=3)
        axes[1].axhline(0, color="gray", lw=.7)
        axes[1].set_ylabel("Raw share delta\npercentage points")
        for axis in axes:
            axis.grid(axis="y", alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
        prefix = "Full raw rebuild" if self.mode == "full_raw_rebuild" else "Diagnostic partial rebuild"
        fig.suptitle(prefix + ": comparison with the thesis cache", fontsize=16, weight="bold")
        fig.text(.08, .015, "Three outlets · strict 2017–2024 week labels · all 40 Step2 dictionaries required · gaps are not filled", fontsize=9)
        fig.tight_layout(rect=[0, .04, 1, .95])
        fig.savefig(self.output / "step2_raw_rebuild_overlay.png", dpi=180)
        fig.savefig(self.output / "step2_raw_rebuild_overlay.pdf")
        plt.close(fig)

    def run(self):
        self.write_json("input_contract.json", CONTRACT)
        parser_manifest_path = Path(self.manifest.get("input_dir", self.folder)) / "manifest.json"
        if parser_manifest_path.exists() and parser_manifest_path.resolve() != (self.folder / "manifest.json").resolve():
            parser_manifest = json.loads(self.source(parser_manifest_path).read_text())
            self.write_json("stage1_source_scope.json", {
                "mode": self.mode, "parser_mode": parser_manifest.get("mode"),
                "input_dir": parser_manifest.get("input_dir"),
                "source_inventory_files": len(parser_manifest.get("source_files", [])),
                "selected_files": len(parser_manifest.get("selected_files", [])),
                "parser_manifest": str(parser_manifest_path.resolve()),
                "aggregation_corpus": self.manifest.get("corpus"),
                "calendar_timezone": self.manifest.get("calendar_timezone", "unspecified"),
                "timestamp_semantics": self.manifest.get("timestamp_semantics"),
                "scope_note": "Rebuild starts at already extracted local Stage1 Parquet; WARC retrieval and extraction are not revalidated.",
            })
        dictionaries = self.normalized(self.artifact("dictionary_weekly"))
        categories = self.normalized(self.artifact("category_weekly"), category=True)
        valid = self.dictionary_names()
        audit = json.loads(self.source(self.args.remote_audit).read_text()) if self.args.remote_audit.exists() else {}
        self.counts(self.artifact("article_counts_daily", required=False), audit, self.artifact("article_corpus_counts", required=False))
        self.indices(dictionaries, valid)
        self.unique_spikes()
        self.newspapers(dictionaries, valid)
        self.categories(categories, valid)
        self.original_and_lau(dictionaries, categories, audit)
        self.write_json("comparison.json", {
            "generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(), "mode": self.mode,
            "rebuild_origin": "Local extracted Stage1 Parquet -> parsed articles -> dictionary matches -> indices. WARC extraction is outside this comparison.",
            "source_manifest": self.manifest, "checks": self.checks, "notes": self.notes,
            "interpretation": [
                "A matched cached aggregate does not by itself validate parser text; row-level audits are separate evidence.",
                "A numeric mismatch is evidence to investigate, not proof of a particular cause.",
                "Endpoint-policy comparisons are explicit; article timestamps and Monday labels are not interchangeable.",
                "Canonical40 and historical39 category medians are intentionally separate outputs.",
            ],
        })
        self.write_json("source_manifest.json", list(self.sources.values()))
        pd.json_normalize(self.checks).to_csv(self.output / "comparison_summary.csv", index=False)
        print(json.dumps({"mode": self.mode, "checks": len(self.checks), "output": str(self.output)}, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=ROOT / "replication/raw_rebuild/indices")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replication/raw_rebuild/comparison")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--remote-audit", type=Path, default=ROOT / "replication/remote_corpus_audit.json")
    parser.add_argument("--reference-counts-dir", type=Path, default=ROOT / "replication/raw_rebuild/reference_counts")
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly label results diagnostic; never a completed full rebuild")
    parser.add_argument("--print-contract", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.print_contract:
        print(json.dumps(CONTRACT, indent=2))
    else:
        Comparison(args).run()


if __name__ == "__main__":
    main()
