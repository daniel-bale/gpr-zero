#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "pandas==2.2.3",
#   "numpy==2.2.3",
#   "pyarrow==20.0.0",
#   "matplotlib==3.10.1",
# ]
# ///
"""Independently reproduce thesis index statistics from preserved cached inputs.

This does not re-extract or reclassify raw news articles. Weekly shares are
recomputed from cached numerator/denominator counts wherever those are available.
No existing pipeline code is imported and no source cache is modified.

Run from anywhere, with uv's project resolution disabled:
  UV_CACHE_DIR=/private/tmp/gpr-zero-uv-cache \
  UV_PYTHON_INSTALL_DIR=/private/tmp/gpr-zero-uv-python \
  MPLCONFIGDIR=/private/tmp/gpr-zero-mpl \
  uv run --no-project --python 3.12 scripts/replicate_thesis_indices.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2017-01-01")
STOP = pd.Timestamp("2025-01-01")
THESIS = "replication/reference_thesis/contents/"
STANDARD_CATEGORIES = [
    "Military Escalation", "Diplomatic Isolation", "Economic Coercion",
    "Regional Security", "Internal Political Instability", "Gray Zone Operations",
    "US-China-Taiwan Relations", "Cyber and Information Warfare",
]


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def window(data, start=START, stop=STOP):
    return data.loc[(data.index >= start) & (data.index < stop)]


class Replication:
    def __init__(self, data, output):
        self.data = data.resolve()
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.checks = []
        self.sources = {}

    def source(self, path):
        path = path.resolve()
        if path not in self.sources:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.sources[path] = {"path": str(path), "bytes": path.stat().st_size,
                                  "sha256": digest}
        return path

    def csv(self, relative):
        return pd.read_csv(self.source(self.data / relative))

    def record(self, id, claimed, observed, scope, source, status=None,
               tolerance=0, details=None):
        if status is None:
            if isinstance(claimed, (int, float)) and isinstance(observed, (int, float, np.number)):
                status = "reproduced" if abs(float(claimed) - float(observed)) <= tolerance else "discrepancy"
            else:
                status = "reproduced" if claimed == observed else "discrepancy"
        item = {"id": id, "claimed": claimed, "observed": observed,
                "status": status, "scope": scope, "source": source}
        if tolerance:
            item["absolute_tolerance"] = tolerance
        if details is not None:
            item["details"] = details
        self.checks.append(json_value(item))

    def dictionary_audit(self):
        results = []
        dictionaries = {}
        for file in sorted((self.data / "dictionaries/all_dictionaries").glob("step[12]_*.json")):
            content = json.loads(self.source(file).read_text())
            categories = content["categories"]
            word_sets = content["word_sets"]
            issues = []
            for category, query in categories.items():
                if re.search(r"\bOR\b", query):
                    issues.append({"category": category, "reason": "forbidden_OR_operator", "query": query})
                # Tokenizing operator words separately avoids misclassifying
                # OR syntax as a nonexistent word-set reference.
                tokens = re.split(r"\s+(?:AND|OR)\s+|[()+]", query)
                references = {re.sub(r"\s*N/\d+.*$", "", token).strip()
                              for token in tokens if token.strip()}
                missing = sorted(references - word_sets.keys())
                if missing:
                    issues.append({"category": category, "reason": "missing_word_set_reference",
                                   "references": missing, "query": query})
            results.append({"dictionary": file.stem, "step": file.stem[:5],
                            "valid": not issues, "category_count": len(categories),
                            "issues": issues})
            dictionaries[file.stem] = content
        step1 = [r for r in results if r["step"] == "step1"]
        step2 = [r for r in results if r["step"] == "step2"]
        valid1 = [r["dictionary"] for r in step1 if r["valid"]]
        valid2 = [r["dictionary"] for r in step2 if r["valid"]]
        self.record("step1_attempted_dictionary_count", 40, len(step1), "Original Step1 JSON including five step1_taiwan_* files", THESIS + "chapter02.tex:24")
        self.record("step1_valid_dictionary_count", 35, len(valid1), "AND-only queries with existing word-set references", THESIS + "chapter02.tex:24")
        self.record("step2_dictionary_count", 40, len(step2), "Original Step2 JSON", THESIS + "chapter02.tex:55")
        self.record("step2_valid_dictionary_count", 40, len(valid2), "AND-only queries with existing word-set references", THESIS + "chapter02.tex:55")
        reason_counts = {}
        for row in step1:
            for reason in {item["reason"] for item in row["issues"]}:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        self.record("step1_invalid_reason", {"forbidden_OR_operator": 5}, reason_counts,
                    "Count dictionaries once per validation failure type", THESIS + "chapter02.tex:24")
        category_sets = {
            "all40": set().union(*(set(dictionaries[r["dictionary"]]["categories"]) for r in step1)),
            "valid35": set().union(*(set(dictionaries[name]["categories"]) for name in valid1)),
        }
        for name, categories in category_sets.items():
            self.record(f"step1_unique_category_names_{name}", 172, len(categories),
                        f"Exact case-sensitive category names, {name}; no semantic consolidation",
                        THESIS + "chapter02.tex:35")
            (self.output / f"step1_category_names_{name}.txt").write_text("\n".join(sorted(categories)) + "\n")
        self.write_json("dictionary_validation.json", results)
        pd.DataFrame([{**r, "issues": json.dumps(r["issues"], ensure_ascii=False)} for r in results]).to_csv(self.output / "dictionary_validation.csv", index=False)
        if len(step1) != 40 or len(step2) != 40 or len(valid1) != 35 or len(valid2) != 40:
            raise ValueError("Dictionary inventory/validation differs from the required 35 + 40; inspect dictionary_validation.json")
        return valid1, valid2

    def load_timeseries(self, name, scope):
        suffix = "_W_overall.parquet" if scope == "all_outlets" else "_W_overall_3outlets.parquet"
        directory = "gpr_timeseries" if scope == "all_outlets" else "gpr_timeseries_3outlets"
        path = self.source(self.data / directory / (name + suffix))
        df = pd.read_parquet(path)
        date_col = "week" if "week" in df else "warc_date_bucketed"
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        if not df.index.is_unique:
            raise ValueError(f"Duplicate weeks in {path}")
        numerator = "gpr_articles" if "gpr_articles" in df else "total_gpr_count"
        saved = "gpr_index" if "gpr_index" in df else "total_gpr_share"
        share = df[numerator].astype(float).div(df["total_articles"].replace(0, np.nan))
        error = (share - df[saved]).abs().max()
        return share.rename(name), float(error), df

    def indices(self, valid1, valid2):
        panels = {}
        share_errors = []
        correlation_rows = []
        for scope in ["all_outlets", "three_outlets"]:
            for step, names, claimed in [("step1", valid1, .493), ("step2", valid2, .698)]:
                series = {}
                for name in names:
                    values, error, _ = self.load_timeseries(name, scope)
                    series[name] = values
                    share_errors.append({"scope": scope, "dictionary": name, "max_absolute_share_error": error})
                panel = pd.DataFrame(series).sort_index()
                panels[(scope, step)] = panel
                selected = window(panel)
                corr = selected.corr()
                upper = corr.to_numpy()[np.triu_indices(len(names), 1)]
                observed = float(np.nanmean(upper))
                row = {"scope": scope, "step": step, "dictionaries": len(names),
                       "date_window": "strict_2017_2024",
                       "weeks": len(selected), "pair_count": int(np.isfinite(upper).sum()),
                       "mean_pairwise_correlation": observed,
                       "first_week": selected.index.min(), "last_week": selected.index.max()}
                correlation_rows.append(row)
                for date_window, other in [
                    ("full_cache", panel),
                    ("include_2016_12_26_partial_week", window(panel, pd.Timestamp("2016-12-26"))),
                ]:
                    other_corr = other.corr().to_numpy()
                    other_upper = other_corr[np.triu_indices(len(names), 1)]
                    correlation_rows.append({
                        "scope": scope, "step": step, "date_window": date_window,
                        "dictionaries": len(names), "weeks": len(other),
                        "pair_count": int(np.isfinite(other_upper).sum()),
                        "mean_pairwise_correlation": float(np.nanmean(other_upper)),
                        "first_week": other.index.min(), "last_week": other.index.max(),
                    })
                self.record(f"{step}_pairwise_correlation_{scope}", claimed, observed,
                            f"{scope}; 2017 <= week < 2025; cached counts; pairwise complete observations",
                            THESIS + "chapter03.tex:18", tolerance=.0005, details=row)
                corr.to_csv(self.output / f"{step}_{scope}_correlation_matrix.csv")
                selected.to_csv(self.output / f"{step}_{scope}_weekly_shares.csv", index_label="week")
        pd.DataFrame(correlation_rows).to_csv(self.output / "pairwise_correlation_summary.csv", index=False)
        pd.DataFrame(share_errors).to_csv(self.output / "cached_share_validation.csv", index=False)
        self.record("weekly_cached_share_matches_count_ratio", 0, max(r["max_absolute_share_error"] for r in share_errors),
                    "All selected Step1/Step2 weekly cached numerators divided by denominators", "cached weekly parquets", tolerance=1e-12)
        full = panels[("three_outlets", "step2")]
        median = full.median(axis=1).rename("step2_median")
        saved = self.csv("gpr_indices_3outlets/step2_median_gpr_3outlets.csv")
        saved["week"] = pd.to_datetime(saved["week"])
        saved = saved.set_index("week").sort_index()
        missing_from_saved = median.index.difference(saved.index)
        extra_in_saved = saved.index.difference(median.index)
        same_dates = len(missing_from_saved) == 0 and len(extra_in_saved) == 0
        reconstructed = pd.DataFrame({"recomputed_raw": median, "saved_raw": saved["gpr_index"],
                                      "recomputed_scaled_full_cache": median / median.mean() * 100,
                                      "saved_scaled_full_cache": saved["gpr_index_scaled"],
                                      "dictionary_count": full.count(axis=1)})
        reconstructed["raw_absolute_error"] = (reconstructed.recomputed_raw - reconstructed.saved_raw).abs()
        reconstructed["scaled_absolute_error"] = (reconstructed.recomputed_scaled_full_cache - reconstructed.saved_scaled_full_cache).abs()
        reconstructed.to_csv(self.output / "median_reconstruction.csv", index_label="week")
        self.record("step2_median_reconstruction", 0, float(reconstructed.raw_absolute_error.max()),
                    "Three outlets, median of 40 raw weekly count ratios; full cached range",
                    "data/gpr_indices_3outlets/step2_median_gpr_3outlets.csv", tolerance=1e-12,
                    status="reproduced" if same_dates and reconstructed.raw_absolute_error.max() <= 1e-12 else "discrepancy",
                    details={"weeks": len(reconstructed), "min_dictionary_count": int(full.count(axis=1).min()),
                             "index_sets_equal": same_dates,
                             "missing_from_saved_dates": list(missing_from_saved),
                             "extra_in_saved_dates": list(extra_in_saved),
                             "recomputed_datetime_dtype": str(median.index.dtype),
                             "saved_datetime_dtype": str(saved.index.dtype),
                             "paired_nonmissing_weeks": len(reconstructed[["recomputed_raw", "saved_raw"]].dropna()),
                             "study_period_paired_nonmissing_weeks": len(window(reconstructed)[["recomputed_raw", "saved_raw"]].dropna()),
                             "max_scaled_absolute_error": float(reconstructed.scaled_absolute_error.max())})
        lau, _, _ = self.load_timeseries("lau_overall_taiwan_recalculated", "three_outlets")
        lau.name = "lau"
        paired = window(pd.concat([median, lau], axis=1)).dropna()
        self.record("step2_lau_correlation", .611, float(paired.corr().iloc[0, 1]),
                    "Three outlets; 2017 <= week < 2025; recomputed Step2 median and Lau count ratios",
                    THESIS + "chapter03.tex:20", tolerance=.0005, details={"paired_weeks": len(paired)})
        paired.to_csv(self.output / "step2_lau_weekly_raw.csv", index_label="week")
        self.plot_indices(paired)
        return median, lau

    @staticmethod
    def detect_spikes(values):
        values = values.sort_index()
        # Explicit calendar reindexing prevents bridging an absent week.
        values = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="W-MON"))
        threshold = float(values.mean() + values.std(ddof=1))
        above = values.gt(threshold)
        groups = above.ne(above.shift(fill_value=False)).cumsum()
        records = []
        for _, series in values[above].groupby(groups[above]):
            records.append({"start_date": series.index.min(), "end_date": series.index.max(),
                            "peak_date": series.idxmax(), "peak_value": float(series.max()),
                            "duration_weeks": len(series), "mean_spike_value": float(series.mean()),
                            "threshold": threshold})
        return pd.DataFrame(records), threshold

    def spikes(self, median, lau):
        rows = []
        for name, values, claimed, original_file in [
            ("step2", median, 21, "step2_median_spikes.csv"),
            ("lau", lau, 22, "lau_2024_spikes.csv"),
        ]:
            original = self.csv("gpr_spikes/unified_detection_3outlets/" + original_file)
            columns = ["start_date", "end_date", "peak_date"]
            original_dates = {tuple(pd.Timestamp(row[c]).strftime("%Y-%m-%d") for c in columns)
                              for _, row in original.iterrows()}
            windows = {"full_cache": values, "strict_2017_2024": window(values),
                       "include_2016_12_26_partial_week": window(values, pd.Timestamp("2016-12-26"))}
            for scope, selected in windows.items():
                detected, threshold = self.detect_spikes(selected)
                detected.to_csv(self.output / f"{name}_spikes_{scope}.csv", index=False)
                dates = {tuple(pd.Timestamp(row[c]).strftime("%Y-%m-%d") for c in columns)
                         for _, row in detected.iterrows()}
                row = {"index": name, "scope": scope, "count": len(detected), "threshold": threshold,
                       "observed_weeks": len(selected), "first_week": selected.index.min(),
                       "last_week": selected.index.max(), "exact_original_ranges": len(dates & original_dates),
                       "new_ranges": sorted(dates - original_dates),
                       "original_ranges_not_recovered": sorted(original_dates - dates)}
                row["original_events_with_peak_outside_study"] = [
                    {c: pd.Timestamp(original_row[c]).strftime("%Y-%m-%d") for c in columns}
                    for _, original_row in original.iterrows()
                    if not START <= pd.Timestamp(original_row["peak_date"]) < STOP
                ]
                rows.append(row)
                self.record(f"{name}_spike_count_{scope}", claimed, len(detected),
                            f"Three outlets; {scope}; mean + sample std, strict >; consecutive calendar weeks",
                            THESIS + "chapter03.tex:22,119", details=row)
                if scope == "full_cache":
                    self.record(f"{name}_original_spike_ranges", len(original_dates), len(dates & original_dates),
                                "Exact equality of start, end and peak dates against archived output",
                                "data/gpr_spikes/unified_detection_3outlets/" + original_file,
                                status="reproduced" if dates == original_dates else "discrepancy")
        self.write_json("spike_comparison.json", rows)

    def newspapers(self):
        mapping = {"China Times": "china", "Liberty Times": "liberty", "United Daily": "united"}
        claims = {"China Times": 68, "Liberty Times": 44, "United Daily": 74}
        panel = {}
        missing = []
        calendar = pd.date_range(START, STOP - pd.Timedelta(days=1), freq="W-MON")
        for name, stem in mapping.items():
            df = self.csv(f"newspaper_gpr_step2/{stem}_step2_median_gpr.csv")
            df.index = pd.to_datetime(df.iloc[:, 0])
            selected = window(df)
            panel[name] = selected["median_scaled"]
            absent = calendar.difference(selected.index)
            flags = pd.Series(calendar.isin(absent), index=calendar)
            groups = flags.ne(flags.shift(fill_value=False)).cumsum()
            gaps = [{"start": s.index.min(), "end": s.index.max(), "weeks": len(s)}
                    for _, s in flags[flags].groupby(groups[flags])]
            row = {"newspaper": name, "calendar_weeks": len(calendar), "observed_rows": len(selected),
                   "missing_weeks": len(absent), "missing_pct": len(absent) / len(calendar) * 100,
                   "null_index_weeks": int(selected.median_scaled.isna().sum()),
                   "zero_index_weeks": int(selected.median_scaled.eq(0).sum()),
                   "longest_absent_run": max((g["weeks"] for g in gaps), default=0), "gaps": gaps}
            missing.append(row)
            self.record(f"missing_weeks_{stem}", claims[name], len(absent),
                        "Absent rows on 418 Mondays from 2017-01-02 through 2024-12-30; zero indices are not missing",
                        THESIS + "chapter02.tex:11", details=row)
            if stem == "united":
                self.record("united_longest_missing_run", 47, row["longest_absent_run"],
                            "Calendar-consecutive absent weeks, 2017–2024", THESIS + "chapter02.tex:13")
        common = pd.DataFrame(panel).dropna()
        corr = common.corr()
        common.to_csv(self.output / "newspaper_common_weeks.csv", index_label="week")
        corr.to_csv(self.output / "newspaper_correlations.csv")
        for first, second, claimed in [("China Times", "Liberty Times", .475),
                                       ("China Times", "United Daily", .780),
                                       ("Liberty Times", "United Daily", .559)]:
            self.record("newspaper_correlation_" + mapping[first] + "_" + mapping[second],
                        claimed, float(corr.loc[first, second]),
                        "2017 <= week < 2025; intersection of nonnull weeks across all three newspapers, matching create_clean_newspaper_comparison.py",
                        THESIS + "chapter03.tex:164", tolerance=.0005,
                        details={"common_weeks": len(common)})
        self.write_json("newspaper_missing_weeks.json", missing)
        self.record("weekly_calendar_denominator", 418, len(calendar),
                    "W-MON dates inside [2017-01-01, 2025-01-01); partial starting week excluded",
                    THESIS + "chapter02.tex:11",
                    details={"first_monday": calendar[0], "last_monday": calendar[-1],
                             "including_partial_2016_12_26_week": len(calendar) + 1})

    def categories(self, median, valid2):
        saved = self.csv("category_gpr_indices/category_median_gpr_unscaled.csv")
        saved["week"] = pd.to_datetime(saved["week"])
        saved = saved.set_index("week")
        long = self.csv("category_gpr_indices/all_category_gpr_indices.csv")
        long["week"] = pd.to_datetime(long["week"])
        recalculated = long.pivot_table(index="week", columns="category", values="gpr_index", aggfunc="median")
        dictionary_counts = long.groupby(["week", "category"])["dictionary"].nunique().unstack()
        missing_dictionaries = sorted(set(n.removeprefix("step2_") for n in valid2) - set(long.dictionary.unique()))
        # Inspect the historical name mapping as data, without importing or
        # executing the existing pipeline. This explains coverage, not values.
        pipeline_path = self.source(ROOT / "scripts/calculate_category_gpr_complete.py")
        historical_mapping = next(ast.literal_eval(node.value) for node in ast.parse(pipeline_path.read_text()).body
                                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CATEGORY_MAPPING" for target in node.targets))
        omissions = []
        for name in missing_dictionaries:
            batch_path = self.source(self.data / "dictionaries/step2_batch" / (name + ".json"))
            dictionary = json.loads(batch_path.read_text())
            normalized = [historical_mapping.get(category, category) for category in dictionary["categories"]]
            omissions.append({"dictionary": name, "source": str(batch_path),
                              "original_category_names": list(dictionary["categories"]),
                              "category_names_after_historical_mapping": normalized,
                              "recognized_standard_categories": sorted(set(normalized) & set(STANDARD_CATEGORIES)),
                              "reason": "Historical mapping does not recognize lowercase snake_case names; its category-not-present guard returns None for all eight standard category requests.",
                              "pipeline_references": ["scripts/calculate_category_gpr_complete.py:47–72", "scripts/calculate_category_gpr_complete.py:112–118", "scripts/calculate_category_gpr_complete.py:132–133"]})
        self.write_json("category_cache_omissions.json", omissions)
        self.record("category_cache_dictionary_count", 40, int(long.dictionary.nunique()),
                    "Cached long-form category indices; distinct dictionaries contributing to category medians",
                    THESIS + "chapter03.tex:25–27", details={"dictionary_names": sorted(long.dictionary.unique()),
                    "min_week_category_count": int(dictionary_counts.min().min()),
                    "max_week_category_count": int(dictionary_counts.max().max()),
                    "missing_dictionaries": missing_dictionaries})
        shared_dates = saved.index.intersection(recalculated.index)
        shared_categories = saved.columns.intersection(recalculated.columns)
        difference = (saved.loc[shared_dates, shared_categories] - recalculated.loc[shared_dates, shared_categories]).abs()
        missing_dates = recalculated.index.difference(saved.index)
        extra_dates = saved.index.difference(recalculated.index)
        missing_categories = recalculated.columns.difference(saved.columns)
        extra_categories = saved.columns.difference(recalculated.columns)
        coverage_equal = not any(len(x) for x in [missing_dates, extra_dates, missing_categories, extra_categories])
        self.record("category_median_saved_reconstruction", 0, float(difference.max().max()),
                    "Recompute each weekly category median from cached individual category-index records",
                    "data/category_gpr_indices/category_median_gpr_unscaled.csv", tolerance=1e-12,
                    status="reproduced" if coverage_equal and difference.max().max() <= 1e-12 else "discrepancy",
                    details={"shared_weeks": len(shared_dates), "shared_categories": len(shared_categories),
                             "missing_from_saved_dates": list(missing_dates), "extra_in_saved_dates": list(extra_dates),
                             "missing_from_saved_categories": list(missing_categories), "extra_in_saved_categories": list(extra_categories)})
        recalculated.to_csv(self.output / "recomputed_category_median.csv", index_label="week")
        dictionary_counts.to_csv(self.output / "category_dictionary_counts.csv", index_label="week")
        selected = window(recalculated)
        overall = median.reindex(selected.index)
        category_sum = selected.sum()
        metrics = pd.DataFrame({
            "share_of_sum_of_category_indices_pct": category_sum / category_sum.sum() * 100,
            "ratio_of_category_sum_to_overall_sum_pct": category_sum / overall.sum() * 100,
            "mean_weekly_category_to_overall_ratio_pct": selected.div(overall.replace(0, np.nan), axis=0).mean() * 100,
            "mean_weekly_share_of_sum_of_categories_pct": selected.div(selected.sum(axis=1).replace(0, np.nan), axis=0).mean() * 100,
        })
        metrics.to_csv(self.output / "category_contribution_methods.csv", index_label="category")
        for method in metrics.columns:
            top = metrics[method].idxmax()
            self.record("largest_category_" + method, 42, float(metrics.loc[top, method]),
                        f"2017 <= week < 2025; recomputed cached category medians; {method}",
                        THESIS + "chapter02.tex:72", tolerance=.5, details={"category": top})
        provenance_file = "category_contributions/step2_overall_category_contributions_20250731_143405.csv"
        provenance = self.csv(provenance_file)
        self.write_json("legacy_42_percent_provenance.json", {
            "source": "data/" + provenance_file,
            "interpretation": "Legacy output explains the commented-out table numbers; it is not an independent reconstruction of the current median index.",
            "rows": provenance.to_dict("records"),
            "thesis_commented_source": THESIS + "chapter02.tex:125–146",
        })
        row = provenance.loc[provenance["category"] == "Diplomatic Isolation"].iloc[0]
        self.record("legacy_42_percent_provenance", 42, float(row["contribution_to_index"]) * 100,
                    "Historical cached output only: contribution_to_index, distinct from proportion_in_gpr",
                    "data/" + provenance_file, status="provenance_only",
                    details={"proportion_in_gpr_pct": row["proportion_in_gpr"],
                             "thesis_commented_table": THESIS + "chapter02.tex:138"})

    def plot_indices(self, paired):
        scaled = paired / paired.mean() * 100
        scaled.to_csv(self.output / "step2_lau_weekly_scaled100.csv", index_label="week")
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(12.5, 5.5))
        fig.set_facecolor("#fafbf9")
        ax.set_facecolor("#fafbf9")
        full_calendar = pd.date_range(START, STOP - pd.Timedelta(days=1), freq="W-MON")
        plotted = scaled.reindex(full_calendar)
        ax.plot(plotted.index, plotted.step2_median, label="Step2 median · 40 dictionaries", color="#116d75", lw=1.7)
        ax.plot(plotted.index, plotted.lau, label="Lau dictionary", color="#b6673b", lw=1.3, alpha=.8)
        ax.axhline(100, color="#8a9398", lw=.8, ls="--")
        ax.grid(axis="y", color="#dce0e0", lw=.6)
        ax.set_ylabel("Weekly GPR index · mean = 100")
        ax.set_xlim(START, STOP)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left", frameon=False, ncol=2)
        fig.suptitle("Taiwan geopolitical risk: replication from weekly counts", x=.085, ha="left", fontsize=16, weight="bold")
        fig.text(.085, .90, f"China Times, Liberty Times and United Daily | 2017–2024 | Pearson r = {paired.corr().iloc[0, 1]:.4f}", color="#5a666b")
        fig.text(.085, .025, "Shares recomputed from cached matched/total counts; each plotted series normalized within 2017–2024. Gaps remain missing.", fontsize=9, color="#5a666b")
        fig.subplots_adjust(left=.085, right=.98, top=.83, bottom=.12)
        fig.savefig(self.output / "step2_vs_lau.png", dpi=180)
        fig.savefig(self.output / "step2_vs_lau.pdf")
        plt.close(fig)

    def write_json(self, name, data):
        (self.output / name).write_text(json.dumps(json_value(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def run(self):
        valid1, valid2 = self.dictionary_audit()
        median, lau = self.indices(valid1, valid2)
        self.spikes(median, lau)
        self.newspapers()
        self.categories(median, valid2)
        self.write_json("checks.json", {
            "generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
            "method": "Independent calculations from preserved original dictionary JSON and cached weekly counts/category indices/newspaper medians; no raw-article reclassification.",
            "scope": {"start_inclusive": "2017-01-01", "stop_exclusive": "2025-01-01",
                      "date_field": "cached Monday week label", "spike_std_ddof": 1,
                      "correlation": "Pearson; pairwise nonmissing except all-newspaper intersection"},
            "limitations": [
                "Raw article extraction, dictionary matching, and exact article-date/outlet counts are not rebuilt by this script.",
                "Cached week labels before 2017 may contain 2017-01-01; strict week filtering excludes that partial week. Its article-level contents cannot be separated from weekly counts.",
                "Source CSVs describe category medians and newspaper indices; validating them does not establish raw-article provenance.",
                "The archived category-long cache contains the dictionary count reported in the checks; no missing dictionary is silently supplied.",
            ],
            "checks": self.checks,
        })
        pd.DataFrame([{**r, "claimed": json.dumps(r["claimed"], ensure_ascii=False),
                       "observed": json.dumps(r["observed"], ensure_ascii=False),
                       "details": json.dumps(r.get("details"), ensure_ascii=False)} for r in self.checks]).to_csv(self.output / "checks.csv", index=False)
        self.write_json("source_manifest.json", list(self.sources.values()))
        print(json.dumps({"output": str(self.output), "checks": len(self.checks),
                          "status_counts": pd.Series([r["status"] for r in self.checks]).value_counts().to_dict()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replication/indices")
    args = parser.parse_args()
    Replication(args.data_root, args.output_dir).run()


if __name__ == "__main__":
    main()
