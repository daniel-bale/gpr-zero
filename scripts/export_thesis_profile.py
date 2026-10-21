#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Export the newer four-chapter thesis's historical result selections.

This is a bounded, read-only consumer of completed rebuilds. It does not reparse
articles, fit models, silently substitute corrected variants, or claim that all
printed assertions agree. Each canonical artifact records its exact selection.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "newer_four_chapter_thesis_historical_reproduction_v1"
HISTORICAL = "historical_20250802_log_ranges"
VERSIONS = ("taiwan_gpr_dict_president", "gpr_taiwan_step2_median")
MAX_INPUT_BYTES = 32 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def object_hash(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def csv_bytes(rows, fields=None):
    require(bool(rows), "Cannot export an empty table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def unique(rows, keys):
    result = {tuple(row[key] for key in keys): row for row in rows}
    require(len(result) == len(rows), f"Duplicate keys: {keys}")
    return result


def close(left, right, tolerance=1e-12):
    left, right = float(left), float(right)
    return math.isfinite(left) and math.isfinite(right) and abs(left-right) <= tolerance


def complete_aggregate(manifest):
    require(manifest.get("status") == "complete" and manifest.get("source_mode") == "full"
            and all(manifest.get(key) is True for key in ("complete", "input_complete", "source_complete")),
            "Thesis profile requires a completed full rebuild; diagnostic/sample inputs are forbidden")
    require(manifest.get("calendar_timezone") == "Asia/Taipei", "Historical calendar must be Asia/Taipei")


def select_pairwise(rows):
    selected = [row for row in rows if row["scope"] == "all_outlets" and row["period"] == "strict_study_week_labels"]
    keyed = unique(selected, ["step"])
    require(set(keyed) == {("step1",), ("step2",)}, "Need exactly Step1 and Step2 all-outlet correlations")
    result = []
    for step, count in (("step1", 35), ("step2", 40)):
        item = keyed[(step,)]
        observed = item["rebuilt"]
        require(observed["dictionary_count"] == count and observed["weeks"] == 418
                and observed["pair_count"] == count*(count-1)//2 and observed["incomplete_weeks"] == 0,
                "Unexpected pairwise dictionary/week coverage")
        require(close(observed["mean_pairwise_correlation"], item["reference"]["mean_pairwise_correlation"]),
                "Selected pairwise correlation differs from historical reference")
        result.append({"step": step, "scope": "all_outlets", "endpoint_policy": "full_cache_then_study_week_labels", **observed})
    return result


def validate_comparison(rows, comparison, keys, metrics):
    """Bind selected rows to a successful outer-join comparison, not its label."""
    left, right = unique(rows, keys), unique(comparison, keys)
    require(left.keys() == right.keys(), "Selected artifact coverage differs from its comparison")
    for key, row in left.items():
        evidence = right[key]
        require(evidence["coverage"] == "both" and evidence["numeric_mismatch"] == "False",
                "Selected artifact has unmatched comparison rows")
        for metric in metrics:
            require(close(row[metric], evidence[metric + "_rebuilt"])
                    and close(evidence[metric + "_reference"], evidence[metric + "_rebuilt"]),
                    f"Selected artifact differs from comparison: {key}, {metric}")


def project_comparison(comparison, keys, metrics):
    rows = [{**{key: row[key] for key in keys}, **{metric: row[metric + "_rebuilt"] for metric in metrics}}
            for row in comparison]
    validate_comparison(rows, comparison, keys, metrics)
    return rows


def legacy_var_rows(rows, version):
    unique(rows, ["response", "impulse", "period"])
    require(len(rows) == 147 and {row["version"] for row in rows} == {version}, "Expected 147 VAR responses for one version")
    responses = {row["response"] for row in rows}
    require(len(responses) == 7 and {(r["response"], r["impulse"], int(r["period"])) for r in rows}
            == {(response, "log_GPR", period) for response in responses for period in range(21)},
            "Incomplete VAR response/horizon grid")
    result = []
    for row in rows:
        require(close(row["legacy_refit"], row["historical_cached"], 1e-10), "Historical VAR response mismatch")
        result.append({key: row[key] for key in ("version", "period", "response", "impulse")}
                      | {"point_estimate": row["legacy_refit"], "display_multiplier": row["legacy_display_multiplier"],
                         "display_unit": row["legacy_display_unit"]})
    return result


def category_spike_rows(events, categories, overall):
    """Historical figure: median within each event / median overall within it."""
    names = sorted({row["category"] for row in categories})
    require(len(names) == 8, "Category figure requires all 8 categories")
    unique(categories, ["week", "category"])
    unique(overall, ["week"])
    ratios, ranges = [], []
    for number, event in enumerate(events, 1):
        start = event["start_date"]
        end = (datetime.fromisoformat(event["end_date"]) + timedelta(days=6)).date().isoformat()
        selected = [row for row in categories if start <= row["week"] <= end]
        values = [float(row["median_raw"]) for row in overall if start <= row["week"] <= end]
        if not selected or not values:
            continue
        event_id = f"S{number:02d}"
        denominator = statistics.median(values)
        ranges.append({"spike_id": event_id, **{key: event[key] for key in ("start_date", "end_date", "peak_date")}})
        for category in names:
            values_for_category = [float(row["category_share"]) for row in selected if row["category"] == category]
            require(values_for_category, "Missing category within a plotted event")
            ratios.append({"spike_id": event_id, "category": category,
                           "ratio": statistics.median(values_for_category) / denominator if denominator > 0 else 0.0})
    return ratios, ranges


def corrected_target_checks(text, unique_spikes, correlations):
    table = text.split("label={tab:unique-spikes}", 1)[1].split("\\bottomrule", 1)[0]
    dates = re.findall(r"^(\d{4}-\d{2}-\d{2})\s*&", table, flags=re.MULTILINE)
    require(len(dates) == 11 and len(set(dates)) == 11, "Current thesis must have 11 unique peak labels")
    actual_dates = [row["peak_date"] for row in unique_spikes]
    require(sorted(dates) == sorted(actual_dates), "Current thesis peak dates differ from rebuilt full-cache events")
    patterns = [("China Times", "Liberty Times", r"between China Times and Liberty Times is ([0-9.]+)"),
                ("China Times", "United Daily", r"between China Times and United Daily is ([0-9.]+)"),
                ("Liberty Times", "United Daily", r"between Liberty Times and United Daily is ([0-9.]+)")]
    by_name = {row[""]: row for row in correlations}
    comparisons = []
    for first, second, pattern in patterns:
        match = re.search(pattern, text)
        require(match is not None, f"Missing current correlation claim: {first}/{second}")
        printed = float(match.group(1).rstrip("."))
        observed = float(by_name[first][second])
        require(close(printed, observed, .0005), f"Current newspaper correlation label differs: {first}/{second}")
        comparisons.append({"first": first, "second": second, "printed": printed, "rebuilt": observed, "matched": True})
    return {"unique_peak_dates": dates, "unique_peak_dates_match": True, "newspaper_correlations": comparisons,
            "historical_audit_preserved": True}


class Export:
    def __init__(self, rebuild, corrections):
        self.rebuild, self.corrections = rebuild.resolve(), corrections.resolve()
        self.inputs, self.artifacts, self.payloads = {}, {}, {}

    def read(self, path):
        path = Path(path).resolve()
        require(path.is_file() and path.stat().st_size <= MAX_INPUT_BYTES, f"Missing or oversized profile input: {path}")
        data = path.read_bytes()
        record = {"path": str(path), "bytes": len(data), "sha256": digest(data)}
        require(str(path) not in self.inputs or self.inputs[str(path)] == record, f"Input changed during export: {path}")
        self.inputs[str(path)] = record
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.read(path).decode())))

    def verify(self, path, expected):
        require(digest(self.read(path)) == expected, f"Stale or changed provenance input: {path}")

    def bound(self, records, path):
        selected = [r for r in records if Path(r.get("path", r.get("file", ""))).resolve() == Path(path).resolve()]
        require(len(selected) == 1, f"Upstream report must bind exactly one current input: {path}")
        self.verify(path, selected[0]["sha256"])

    def add(self, name, data, sources, selection, rows=None):
        require(name not in self.payloads, f"Duplicate export: {name}")
        self.payloads[name] = data
        self.artifacts[name] = {"sha256": digest(data), "bytes": len(data), "selection": selection,
                                "sources": [str(Path(path).resolve()) for path in sources]}
        if rows is not None:
            self.artifacts[name]["rows"] = rows

    def copy_csv(self, source, name, selection, count=None):
        data = self.read(source)
        rows = list(csv.DictReader(io.StringIO(data.decode())))
        require(rows and (count is None or len(rows) == count), f"Unexpected row count in {source}")
        self.add(name, data, [source], selection, len(rows))
        return rows

    def build(self):
        r = self.rebuild
        aggregate_path = r / "indices/manifest.json"
        aggregate = self.json(aggregate_path)
        complete_aggregate(aggregate)
        producer = self.json(Path(aggregate["input_dir"]) / "manifest.json")
        require(object_hash(producer) == aggregate["source_manifest_sha256"], "Aggregate parser-manifest object hash differs")
        comparison_path = r / "comparison/comparison.json"
        comparison = self.json(comparison_path)
        require(comparison.get("mode") == "full_raw_rebuild" and comparison["source_manifest"] == aggregate,
                "Comparison does not belong to the completed aggregate")
        self.bound(self.json(r / "comparison/source_manifest.json"), aggregate_path)
        checks = {item["id"]: item for item in comparison["checks"]}
        required_checks = ["dictionary_weekly_all_outlets_strict_study_week_labels", "step2_spikes_full_cache",
                           "lau_spikes_full_cache", "step2_unique_spikes_full_cache",
                           "category_median_legacy39_legacy_end_midnight", "newspaper_correlations",
                           "step2_median_three_outlets_full_cache", "dictionary_weekly_three_outlets_full_cache"]
        require(all(checks.get(name, {}).get("status") == "matched" for name in required_checks),
                "A selected historical comparison is missing or failed")
        word_path, newspaper_path, economic_path = (r / name for name in
            ("word_lifts/run_summary.json", "newspaper_words/run_summary.json", "economics/rebuild_summary.json"))
        word, newspaper, economics = (self.json(path) for path in (word_path, newspaper_path, economic_path))
        for report, sources in ((word, "source_files"), (newspaper, "sources")):
            require(report.get("status") == "complete" and report.get("mode") == "full_raw_rebuild"
                    and report.get("calendar_timezone") == "Asia/Taipei", "Incomplete downstream result")
            self.bound(report[sources], aggregate_path)
            for record in report[sources]:
                self.verify(record["path"], record["sha256"])
        require(economics.get("economic_only") is False and economics.get("raw_source_files_unchanged") is True
                and economics.get("all_economic_components_match") is True, "Economics/VAR rebuild is incomplete")
        self.bound([economics["news_manifest"]], aggregate_path)

        pair_path = r / "comparison/pairwise_correlations.json"
        pairs = select_pairwise(self.json(pair_path))
        self.add("pairwise_correlations.csv", csv_bytes(pairs), [pair_path],
                 "All outlets; full-cache weekly counts then 418 Monday labels in 2017–2024; 35 valid Step1 / 40 Step2 dictionaries", 2)
        weekly_step2_path = r / "comparison/step2_median_three_outlets_full_cache.csv"
        weekly_step2 = project_comparison(self.csv(weekly_step2_path), ["week"], ["median_raw", "median_scaled", "dictionary_count"])
        require(len(weekly_step2) == 442 and all(float(row["dictionary_count"]) == 40 for row in weekly_step2),
                "Primary Step2 index requires 442 complete 40-dictionary weeks")
        self.add("step2_weekly_index.csv", csv_bytes(weekly_step2), [weekly_step2_path],
                 "Three outlets; all 442 full-cache weeks; median_raw is raw share, median_scaled has full-cache mean 100", 442)
        weekly_lau_path = r / "comparison/dictionary_weekly_three_outlets_full_cache.csv"
        weekly_lau = project_comparison([row for row in self.csv(weekly_lau_path) if row["dictionary"] == "lau"],
                                       ["week", "dictionary"], ["total_articles", "gpr_articles", "gpr_share"])
        require(len(weekly_lau) == 442, "Primary Lau index requires all 442 full-cache weeks")
        lau_mean = statistics.mean(float(row["gpr_share"]) for row in weekly_lau)
        require(lau_mean > 0, "Cannot scale a nonpositive Lau mean")
        for row in weekly_lau:
            row["mean100_index"] = float(row["gpr_share"]) / lau_mean * 100
        self.add("lau_weekly_index.csv", csv_bytes(weekly_lau), [weekly_lau_path],
                 "Three outlets; all 442 full-cache weeks; gpr_share is the raw article fraction; mean100_index scales over all 442 weeks", 442)
        definitions_path = r / "comparison/spike_definitions.json"
        definitions = self.json(definitions_path)
        require(definitions["mode"] == "full_raw_rebuild" and definitions["calendar_timezone"] == "Asia/Taipei",
                "Wrong spike scope/calendar")
        spike_selection = "Three outlets; thresholds and events from all 442 observed full-cache weeks; mean + sample SD"
        for family, count in (("step2", 21), ("lau", 22), ("step2_unique", 11)):
            name = family + "_spikes"
            source = r / "comparison" / f"rebuilt_{name}_full_cache.csv"
            rows = self.copy_csv(source, name + ".csv", spike_selection, count)
            detail = r / "comparison" / checks[name + "_full_cache"]["details_file"]
            validate_comparison(rows, self.csv(detail), ["start_date", "end_date", "peak_date"], ["peak_value", "duration_weeks"])
            self.artifacts[name + ".csv"]["sources"] += [str(detail), str(definitions_path)]
        category_source = r / "comparison/rebuilt_category_medians_legacy39_legacy_end_midnight.csv"
        category_rows = self.copy_csv(category_source, "category_medians.csv",
            "Three outlets; 39 dictionaries, excluding step2_gemini-2.5-pro_stable8cat_run8; 2017-01-01 <= Taipei timestamp <= 2024-12-31T00:00:00", 3352)
        category_evidence = r / "comparison/category_median_legacy39_legacy_end_midnight.csv"
        validate_comparison(category_rows, self.csv(category_evidence), ["week", "category"], ["category_share"])
        self.artifacts["category_medians.csv"]["sources"].append(str(category_evidence))
        methods_path = r / "comparison/category_share_methods.csv"
        methods = [row for row in self.csv(methods_path) if row["mode"] == "legacy39" and row["endpoint_policy"] == "legacy_end_midnight"]
        require(len(methods) == 8 and all(int(row["minimum_dictionary_count"]) == 39 for row in methods), "Wrong category median variant")
        self.add("category_contribution_definition.csv", csv_bytes(methods), [methods_path],
                 "Sum of each selected category-median series / sum across all 8 series; does not validate the printed 42% claim", 8)

        families = unique(word["families"], ["period", "family"])
        historical = families[(HISTORICAL, "step2")]
        require(historical["spike_count"] == 19 and historical["printed_threshold_11_matches"] == 10,
                "Historical 19-event printed Step2 table has not reproduced")
        require(families[("full_cache", "lau")]["printed_matches"] == 10, "Printed Lau table has not reproduced")
        for family, folder, source_name, comparison_name in (
            ("step2", HISTORICAL, "step2_top10_printed_threshold11.csv", "step2_printed_threshold11_comparison.csv"),
            ("lau", "full_cache", "lau_top10.csv", "lau_printed_thesis_comparison.csv")):
            base = r / "word_lifts" / folder
            selected = self.copy_csv(base / source_name, family + "_top_words.csv",
                ("19 ranges from the completed 2025-08-02 log; positive-event cutoff 11" if family == "step2" else "22 independently rebuilt full-cache events; positive-event cutoff 11")
                + "; all-date three-outlet baseline; positive-event arithmetic mean lift", 10)
            evidence = self.csv(base / comparison_name)
            require(len(evidence) == 10 and all(row["matched"] == "True" for row in evidence), "Printed word table mismatch")
            selected_by_word = unique(selected, ["word"])
            require(set(selected_by_word) == {(row["word"],) for row in evidence}, "Word table rank/member mismatch")
            for row in evidence:
                got = selected_by_word[(row["word"],)]
                require(close(got["lift_score"], row["lift_score"]) and got["spike_count"] == row["spike_count"]
                        and selected[int(row["printed_rank"])-1]["word"] == row["word"], "Word table data differs from successful comparison")
            self.artifacts[family + "_top_words.csv"]["sources"].append(str(base / comparison_name))
        logged_ranges = self.copy_csv(r / "word_lifts" / HISTORICAL / "step2_ranges_from_completed_log.csv", "step2_word_table_event_ranges.csv",
                      "Historical 19-event selection, separate from the 21-event main spike table and the category figure's 19 events; conditional on archived log ranges", 19)
        logs = [Path(record["path"]) for record in word["source_files"] if Path(record["path"]).name == "task9_word_lift_scores_v3_20250802_220639.log"]
        require(len(logs) == 1, "The completed historical word-table log must be bound")
        logged = re.findall(r"^Step2_spike_\d+: (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})$", self.read(logs[0]).decode(), flags=re.MULTILINE)
        require([(row["start_date"], row["end_date"]) for row in logged_ranges] == logged and len(logged) == 19,
                "Historical word-table ranges differ from the completed log")
        self.artifacts["step2_word_table_event_ranges.csv"]["sources"].append(str(logs[0]))

        variants = unique(newspaper["variants"], ["range_source", "variant"])
        variant = variants[("rebuilt_ranges", "historical_pooled_windows")]
        require(variant["event_count"] == 351 and variant["printed_ratio_matches"] == 20 and variant["printed_rank_matches"] == 20,
                "Historical pooled newspaper table has not reproduced")
        selected_checks = [c for c in newspaper["checks"] if c["artifact"] == "independently_detected_event_ranges"
                           or (c["artifact"] == "all_word_ratios" and c.get("range_source") == "rebuilt_ranges"
                               and c.get("variant") == "historical_pooled_windows")]
        require(len(selected_checks) == 2 and all(c["status"] == "matched" for c in selected_checks), "Newspaper historical comparison failed")
        pooled = r / "newspaper_words/rebuilt_ranges/historical_pooled_windows"
        newspaper_selection = "351 independently regenerated newspaper-labeled events; article counts pooled over 3 outlets; positive-event mean; Liberty / (China + 1e-6)"
        top = self.copy_csv(pooled / "top20_words.csv", "newspaper_top_words.csv", newspaper_selection, 20)
        ratios = self.copy_csv(pooled / "word_ratios.csv", "newspaper_word_ratios.csv", newspaper_selection, 96)
        ratio_evidence = self.csv(pooled / "saved_ratios_comparison.csv")
        by_word = unique(ratios, ["word"])
        require(len(ratio_evidence) == 96 and all(row["matched"] == "True" for row in ratio_evidence), "Newspaper ratios differ")
        for row in ratio_evidence:
            require(close(by_word[(row["word"],)]["ratio_liberty_china"], row["ratio_liberty_china_rebuilt"]), "Newspaper ratio artifact changed")
        printed = self.csv(pooled / "printed_thesis_comparison.csv")
        require(len(printed) == 20 and all(row["producer_direction_matches"] == "True" and row["rank_matches"] == "True" for row in printed), "Printed newspaper rows differ")
        top_by_word = unique(top, ["word"])
        require(set(top_by_word) == {(row["word"],) for row in printed}, "Newspaper top 20 membership changed")
        for row in printed:
            got = top_by_word[(row["word"],)]
            require(got["rank"] == row["rank"] and got["favored_newspaper"] == row["favored_newspaper"]
                    and close(got["ratio_liberty_china"], row["ratio_liberty_china"]), "Newspaper top 20 data changed")
        self.copy_csv(r / "newspaper_words/rebuilt_event_ranges.csv", "newspaper_event_ranges.csv", newspaper_selection, 351)

        models = unique(economics["models"], ["version"])
        require(set(models) == {(version,) for version in VERSIONS}, "Both historical VARs required")
        for version in VERSIONS:
            model = models[(version,)]
            require(model["cached_point_count"] == 147 and all(value is True for value in model["checks"].values()), "VAR checks failed")
            self.verify(model["input_file"], model["input_sha256"])
            series = economics["news_series"][version]
            self.verify(series["file"], series["sha256"])
            source = r / "economics/var" / f"{version}_irf_comparison.csv"
            rows = legacy_var_rows(self.csv(source), version)
            self.add(f"var_{version}.csv", csv_bytes(rows), [source, economic_path],
                     "Historical 2 * sigma_GPR * orthogonal_IRF and original display multipliers; native-frequency quarterly GPR means; VAR(2)", 147)
            plot = r / "economics/var" / f"{version}_historical_replication.png"
            self.add(f"var_{version}.png", self.read(plot), [plot, economic_path], "Historical point-response comparison; corrected shock curves excluded")

        corrections_path = self.corrections / "change_manifest.json"
        corrections = self.json(corrections_path)
        require(corrections["status"] == "applied_and_verified" and corrections["original_reference_snapshot_preserved"] is True,
                "Current corrected thesis source must be verified")
        target = self.corrections / "project/contents/chapter03.tex"
        original = self.corrections / "original/contents/chapter03.tex"
        self.verify(target, corrections["after_sha256"])
        self.verify(original, corrections["before_sha256"])
        current = corrected_target_checks(self.read(target).decode(),
            self.csv(r / "comparison/rebuilt_step2_unique_spikes_full_cache.csv"),
            self.csv(r / "comparison/rebuilt_newspaper_correlations.csv"))
        current.update(target_source=str(target), target_sha256=corrections["after_sha256"],
                       correction_manifest=str(corrections_path), corrections=corrections["corrections"],
                       pdf_compilation_confirmed=corrections.get("compilation", {}).get("pdf_compilation_confirmed", False))
        self.add("current_thesis_label_checks.json", json_bytes(current),
                 [target, corrections_path, r / "comparison/rebuilt_newspaper_correlations.csv"],
                 "Current corrected chapter03: 11 peak labels and 3 newspaper correlations; original audit kept unchanged")

        category_audit_path = ROOT / "replication/inconsistency_audit/category_provenance/audit_report.json"
        category_audit = self.json(category_audit_path)
        require(category_audit.get("status") == "investigation_complete_no_remote_or_thesis_edits", "Category provenance audit is incomplete")
        findings = unique(category_audit["findings"], ["id"])
        require(("step1_172",) in findings and ("top_category_42_percent",) in findings, "Category provenance findings are missing")
        self.add("category_claim_provenance.json", self.read(category_audit_path), [category_audit_path],
                 "Bounded provenance audit of unsupported 172-name and unnamed top-category 42% claims; not selected index output")
        word_audit_dir = ROOT / "replication/inconsistency_audit/word_provenance"
        word_audit_path = word_audit_dir / "analysis_summary.json"
        word_audit = self.json(word_audit_path)
        figure = word_audit["category_figure"]
        require(figure["input_events"] == 21 and figure["plotted_events"] == 19
                and figure["common_ranges_with_old19"] == 7 and figure["raw_rebuild_validation"]["matched_cells"] == 152,
                "Category figure provenance/coverage differs")
        for source in (category_source, r / "indices/step2_median_weekly.csv", r / "comparison/rebuilt_step2_spikes_full_cache.csv"):
            self.verify(source, word_audit["inputs"][str(source)])
        ratio_path = word_audit_dir / "category_figure_raw_rebuild_ratio_validation.csv"
        ratio_evidence = self.csv(ratio_path)
        ratios, figure_ranges = category_spike_rows(self.csv(r / "comparison/rebuilt_step2_spikes_full_cache.csv"), category_rows, weekly_step2)
        require(len(ratios) == len(ratio_evidence) == 152 and len(figure_ranges) == 19, "Category figure requires 152 cells from 19 covered events")
        ratio_by_key = unique(ratios, ["spike_id", "category"])
        require(ratio_by_key.keys() == unique(ratio_evidence, ["spike_id", "category"]).keys(), "Category figure cells differ")
        for row in ratio_evidence:
            require(row["matched"] == "True" and close(row["saved_input_ratio"], row["raw_rebuilt_ratio"])
                    and close(ratio_by_key[(row["spike_id"], row["category"])]["ratio"], row["raw_rebuilt_ratio"]),
                    "Category figure ratio differs from validated rebuilt inputs")
        event_coverage_path = word_audit_dir / "category_figure_event_coverage.csv"
        coverage = self.csv(event_coverage_path)
        covered = {(row["spike_id"], row["start_date"], row["end_date"], row["peak_date"]) for row in coverage if row["plotted"] == "True"}
        require(len(coverage) == 21 and covered == {tuple(row[key] for key in ("spike_id", "start_date", "end_date", "peak_date")) for row in figure_ranges},
                "Category figure event coverage changed")
        self.add("category_spike_ratios.csv", csv_bytes(ratios),
                 [ratio_path, word_audit_path, category_source, weekly_step2_path],
                 "Historical figure reconstruction: within-event category median / within-event overall median; legacy39; 19 covered events from current 21, not the word-table 19", 152)
        self.add("category_figure_event_ranges.csv", csv_bytes(figure_ranges), [event_coverage_path, word_audit_path],
                 "Current 21 Step2 events minus two in 2025 with no category coverage; only 7 ranges overlap the older word-table 19", 19)
        self.add("word_and_category_figure_provenance.json", self.read(word_audit_path), [word_audit_path],
                 "Original numeric category-figure file was not recovered; producer reconstruction and preserved figure identity are distinct evidence")
        chapter_sources = []
        for number in range(1, 5):
            path = self.corrections / f"project/contents/chapter{number:02}.tex"
            self.read(path)
            chapter_sources.append(self.inputs[str(path)])
        category_image = self.corrections / "project/figures/category_median_ratios_spike_periods_bar.png"
        self.verify(category_image, figure["working_thesis_figure_sha256"])
        var_image = self.corrections / "project/figures/svar_irf_custom.png"
        self.read(var_image)
        unresolved = [
            {"id": "category_name_count_172", "status": "unresolved_printed_claim", "printed": 172,
             "observed": findings[("step1_172",)]["observed"],
             "action": "No retained artifact substantiates 172. A trailing-empty-line count is only a possible mechanism; choose the intended dictionary scope before correcting the prose."},
            {"id": "top_category_42_percent", "status": "unresolved_printed_claim", "printed": "Unnamed top contributor 42% in chapter02",
             "observed": "Old Diplomatic Isolation table value 42.09 equals 100 times an already-percent CSV value 0.420897. All eight old table values repeat this conversion. The producer also counted 40 dictionaries plus 20 metadata files and used incompatible word-column ordinals.",
             "action": "Do not attribute the prose to Military Escalation or substitute 0.4209% or 35.6918% as a validated target; the intended contribution denominator remains unresolved.",
             "evidence": str(category_audit_path)},
            {"id": "step2_word_table_event_header", "status": "unresolved_version_label", "printed": 21, "observed": 19,
             "action": "Printed word values use 19 logged event ranges and cutoff 11; main spike table separately uses 21 events."},
            {"id": "historical_word_ranges", "status": "conditional_historical_input", "observed": "19 event ranges recovered from a completed log",
             "action": "Word counts/lifts are freshly rebuilt; historical event ranges are fixed provenance inputs, not redetected from the current 21-event series."},
            {"id": "var_confidence_bands", "status": "stochastic_nonidentity", "observed": "Original Monte Carlo draws were unseeded; new deterministic bands need not match every endpoint.",
             "action": "Canonical VAR files export reproduced historical point estimates; regenerated bands remain separate upstream evidence."},
        ]
        self.add("unresolved_claims.json", json_bytes(unresolved), [comparison_path, word_path, economic_path, category_audit_path, word_audit_path],
                 "Remaining printed/version/provenance limitations; corrected peak/correlation/ratio labels are not listed as unresolved")
        return {"format_version": 1, "profile": PROFILE, "status": "complete", "complete": True,
                "all_thesis_claims_reproduced": False, "target": "Newer four-chapter thesis with verified label corrections",
                "calendar_timezone": "Asia/Taipei", "rebuild_dir": str(r), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "exporter_sha256": digest(Path(__file__).read_bytes()), "corpus": aggregate["corpus"],
                "purpose": "Historical reproduction defaults selected per artifact; corrected analyses remain outside this profile.",
                "var_targets": {"primary": "gpr_taiwan_step2_median", "secondary": "taiwan_gpr_dict_president",
                    "numeric_evidence": "Both named models reproduce their historical numeric caches.",
                    "target_image": self.inputs[str(var_image)],
                    "figure_evidence": "The thesis svar_irf_custom.png visually agrees with Step2 response shapes/scales; exact image-producer identity remains unproven."},
                "thesis_sources": chapter_sources,
                "historical_inconsistency": "The main spike table and printed word table use different historical event sets; this profile preserves and labels that distinction.",
                "event_sets": {"main_spike_table": "21 independently detected full-cache Step2 events",
                    "printed_word_table": "19 older ranges from the completed 2025-08-02 log",
                    "category_figure": "19 current events with category coverage: current 21 minus two events in 2025. This is a different set from the 19 word-table events; their ranges must not be interchanged.",
                    "category_figure_exported": True,
                    "category_figure_validation": "All 152 reconstructed ratios agree with saved-input reconstruction. Original per-event numeric plotting data was not recovered; this is not a pixel-height comparison.",
                    "category_figure_image": self.inputs[str(category_image)]},
                "historical_behaviors_requiring_disclosure": [
                    {"id": "correlation_outlet_scope", "behavior": "Printed pairwise correlations use all outlets, while the main index uses three outlets."},
                    {"id": "full_cache_spike_scope", "behavior": "Spike thresholds use all 442 full-cache weeks, including dates outside 2017–2024."},
                    {"id": "category_dictionary_omission", "behavior": "Category medians use 39 dictionaries because Gemini Step2 run 8 was omitted; overall Step2 medians use 40."},
                    {"id": "category_end_day", "behavior": "The historical category endpoint is 2024-12-31 at midnight, excluding the remainder of the last day."},
                    {"id": "pooled_newspaper_windows", "behavior": "Newspaper-labeled event windows pool article counts across three outlets; the figures are not outlet-conditioned word frequencies."},
                    {"id": "var_shock_and_units", "behavior": "Historical responses multiply Cholesky IRFs by 2 * sigma_GPR; investment/equity log changes are not multiplied by 100 despite original percentage labels."},
                ],
                "current_thesis_label_checks": current, "unresolved_claims": unresolved,
                "inputs": list(self.inputs.values()), "artifacts": self.artifacts,
                "guard": "Completed full-input guards, upstream manifest/hash bindings, selected result comparisons and fixed coverage; no corpus scan or large artifact copy"}

    def publish(self, output, manifest, overwrite=False):
        require(not output.is_symlink(), "Refusing symlink output")
        output = output.resolve()
        require(all(output != Path(path) and output not in Path(path).parents for path in self.inputs), "Output cannot contain profile inputs")
        if output.exists() and any(output.iterdir()):
            require(overwrite, "Output exists; use --overwrite for this derived profile")
            old = json.loads((output / "manifest.json").read_text())
            require(old.get("profile") == PROFILE, "Refusing to replace an unrelated output directory")
            require({p.name for p in output.iterdir()} == set(old["artifacts"]) | {"manifest.json"}, "Unexpected files in prior profile; refusing overwrite")
            require((output / "manifest.json").is_file() and not (output / "manifest.json").is_symlink(), "Invalid prior profile manifest")
            for name, evidence in old["artifacts"].items():
                path = output / name
                require(path.is_file() and not path.is_symlink() and path.stat().st_size == evidence["bytes"]
                        and digest(path.read_bytes()) == evidence["sha256"], "Prior profile artifact was modified; refusing overwrite")
        for path, evidence in self.inputs.items():
            self.verify(path, evidence["sha256"])
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".thesis-profile-", dir=output.parent))
        try:
            for name, data in self.payloads.items():
                (temporary / name).write_bytes(data)
            (temporary / "manifest.json").write_bytes(json_bytes(manifest))
            if output.exists():
                shutil.rmtree(output)
            temporary.replace(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-dir", type=Path, default=ROOT / "replication/raw_rebuild")
    parser.add_argument("--corrections-dir", type=Path, default=ROOT / "replication/thesis_edits/label_corrections")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replication/raw_rebuild/thesis_profile")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    exporter = Export(args.rebuild_dir, args.corrections_dir)
    manifest = exporter.build()
    exporter.publish(args.output_dir, manifest, args.overwrite)
    print(json.dumps({"status": "complete", "profile": PROFILE, "output": str(args.output_dir.resolve()),
                      "artifacts": len(manifest["artifacts"]), "bytes": sum(v["bytes"] for v in manifest["artifacts"].values()),
                      "all_thesis_claims_reproduced": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
