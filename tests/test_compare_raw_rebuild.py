# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pandas==2.2.3", "numpy==2.2.3", "pyarrow==20.0.0", "matplotlib==3.10.1"]
# ///
"""Coverage and endpoint tests for the full-rebuild comparison consumer."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_raw_rebuild as comparison


class CompletionTests(unittest.TestCase):
    def complete_manifest(self):
        return {"status": "complete", "complete": True, "input_complete": True,
                "source_complete": True, "source_mode": "full"}

    def test_missing_evidence_and_pilot_cannot_be_full(self):
        for manifest in [{}, {"complete": True}, {"complete": True, "source_complete": True, "source_mode": "pilot"}]:
            with self.assertRaises(ValueError):
                comparison.assess_completion(manifest)
            self.assertEqual(comparison.assess_completion(manifest, True), "diagnostic_incomplete_or_sample")
        self.assertEqual(comparison.assess_completion(self.complete_manifest()), "full_raw_rebuild")

    def test_each_completion_field_required_and_exact(self):
        alternatives = {"status": [None, "running", "failed"], "source_mode": [None, "benchmark"],
                        "complete": [None, False, 1], "input_complete": [None, False, 1],
                        "source_complete": [None, False, 1]}
        for field, values in alternatives.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    manifest = self.complete_manifest()
                    if value is None:
                        del manifest[field]
                    else:
                        manifest[field] = value
                    with self.assertRaisesRegex(ValueError, "status='complete'"):
                        comparison.assess_completion(manifest)
                    self.assertEqual(comparison.assess_completion(manifest, True), "diagnostic_incomplete_or_sample")


class FrameComparisonTests(unittest.TestCase):
    def test_outer_join_does_not_hide_missing_week_or_zero_fill(self):
        old = pd.DataFrame({"week": [1, 2], "gpr_articles": [10, 0]})
        new = pd.DataFrame({"week": [1, 3], "gpr_articles": [10, 0]})
        joined, summary = comparison.compare_frames(old, new, ["week"], ["gpr_articles"])
        self.assertEqual(summary["status"], "different")
        self.assertEqual(summary["reference_only_rows"], 1)
        self.assertEqual(summary["rebuilt_only_rows"], 1)
        self.assertTrue(np.isnan(joined.loc[joined.week.eq(2), "gpr_articles_rebuilt"].iloc[0]))

    def test_counts_exact_shares_tolerant_and_null_asymmetry_visible(self):
        old = pd.DataFrame({"k": [1, 2, 3], "total_articles": [100, 100, 100], "gpr_share": [.1, .2, np.nan]})
        new = pd.DataFrame({"k": [1, 2, 3], "total_articles": [100, 101, 100], "gpr_share": [.1 + 1e-14, .2, .0]})
        _, summary = comparison.compare_frames(old, new, ["k"], ["total_articles", "gpr_share"])
        self.assertEqual(summary["metrics"]["total_articles"]["mismatched_rows"], 1)
        self.assertEqual(summary["metrics"]["gpr_share"]["mismatched_rows"], 1)
        self.assertEqual(summary["metrics"]["gpr_share"]["unequal_null_rows"], 1)

    def test_duplicate_keys_fail_instead_of_cartesian_product(self):
        frame = pd.DataFrame({"k": [1, 1], "gpr_share": [.1, .2]})
        with self.assertRaisesRegex(ValueError, "Duplicate reference"):
            comparison.compare_frames(frame, frame.iloc[:1], ["k"], ["gpr_share"])

    def test_empty_comparison_not_a_match(self):
        frame = pd.DataFrame(columns=["k", "gpr_share"])
        _, summary = comparison.compare_frames(frame, frame, ["k"], ["gpr_share"])
        self.assertEqual(summary["status"], "unavailable")

    def test_count_delta_not_lost_when_both_sides_have_rows(self):
        old = pd.DataFrame({"k": [1], "gpr_articles": [4]})
        new = pd.DataFrame({"k": [1], "gpr_articles": [7]})
        merged, summary = comparison.compare_frames(old, new, ["k"], ["gpr_articles"])
        self.assertEqual(merged.gpr_articles_delta.iloc[0], 3)
        self.assertEqual(summary["numeric_mismatch_rows"], 1)


class PeriodAndDictionaryTests(unittest.TestCase):
    def test_newspaper_coverage_separates_absence_null_and_zero(self):
        calendar = pd.date_range("2017-01-02", periods=7, freq="W-MON")
        series = pd.Series([0., np.nan, 2., 0.], index=calendar[[0, 1, 4, 6]])
        summary = comparison.coverage_summary(series, calendar)
        self.assertEqual(summary["observed_weeks"], 4)
        self.assertEqual(summary["missing_weeks"], 3)
        self.assertEqual(summary["null_index_weeks"], 1)
        self.assertEqual(summary["zero_index_weeks"], 2)
        self.assertEqual(summary["longest_absent_run"], 2)
        self.assertEqual([gap["weeks"] for gap in summary["gaps"]], [2, 1])

    def test_overlap_uses_inclusive_ranges_and_allows_many_to_one(self):
        def events(rows):
            return pd.DataFrame([{key: pd.Timestamp(value) for key, value in zip(
                ["start_date", "end_date", "peak_date"], row)} for row in rows])
        step2 = events([("2025-03-03", "2025-03-03", "2025-03-03"),
                        ("2025-03-17", "2025-04-21", "2025-04-21"),
                        ("2025-04-07", "2025-04-07", "2025-04-07")])
        lau = events([("2025-03-03", "2025-03-31", "2025-03-24")])
        matches = comparison.spike_overlap_pairs(step2, lau)
        self.assertEqual(matches[["step2_idx", "lau_idx"]].values.tolist(), [[0, 0], [1, 0]])
        self.assertEqual(matches.overlap_days.tolist(), [1, 15])
        self.assertEqual(matches.peak_date_diff.tolist(), [21, 28])
        self.assertEqual(matches.overlap_pct_step2.iloc[0], 100)
        self.assertTrue(comparison.spike_overlap_pairs(step2.iloc[:0], lau).empty)

    def test_partial_week_distinct_from_strict_monday_window(self):
        frame = pd.DataFrame({"week": pd.to_datetime(["2016-12-26", "2017-01-02", "2024-12-30", "2025-01-06"])})
        self.assertEqual(len(comparison.study_weeks(frame)), 2)
        self.assertEqual(len(comparison.study_weeks(frame, partial=True)), 3)

    def test_missing_dictionary_makes_median_missing(self):
        panel = pd.DataFrame({"one": [1., 1.], "two": [3., np.nan]}, index=pd.to_datetime(["2017-01-02", "2017-01-09"]))
        result = comparison.median_frame(panel, 2)
        self.assertEqual(result.median_raw.iloc[0], 2.)
        self.assertTrue(np.isnan(result.median_raw.iloc[1]))
        self.assertEqual(result.dictionary_count.tolist(), [2, 1])

    def test_gemini_snake_case_names_and_original_id_preserved(self):
        self.assertEqual(comparison.canonical_category("cyber_and_information_warfare"), "Cyber and Information Warfare")
        self.assertEqual(comparison.canonical_category("us_china_taiwan_relations"), "US-China-Taiwan Relations")
        self.assertEqual(comparison.canonical_category("Diplomatic Crisis"), "Diplomatic Crisis")
        self.assertEqual(comparison.normalized_dictionary("taiwan_gpr_dict_president"), "president_original7")
        self.assertEqual(comparison.normalized_dictionary("step1_taiwan_gpr_dict_president_updated"), "step1_taiwan_gpr_dict_president_updated")
        self.assertEqual(comparison.normalized_dictionary("lau_2024"), "lau")


class AggregateContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = comparison.Comparison.__new__(comparison.Comparison)
        self.runner.sources = {}
        self.runner.checks = []
        self.runner.notes = []
        self.runner.mode = "diagnostic_incomplete_or_sample"
        self.runner.output = self.root / "output"
        self.runner.output.mkdir()
        self.runner.folder = self.root
        self.refs = self.root / "reference"
        self.refs.mkdir()
        self.runner.args = SimpleNamespace(reference_counts_dir=self.refs)

    def test_actual_outputs_manifest_parquet_and_daily_alias(self):
        expected = pd.DataFrame({"date": ["2017-01-01"], "total_articles": [3]})
        expected.to_parquet(self.root / "article_daily.parquet")
        self.runner.manifest = {"outputs": {"article_daily": {"parquet": "article_daily.parquet", "csv": "article_daily.csv", "rows": 1}}}
        pd.testing.assert_frame_equal(self.runner.artifact("article_counts_daily"), expected)

    def test_daily_counts_select_disjoint_scope_and_retain_null_groups(self):
        # Aggregation emits repeated pooled/period views. Counting all those
        # rows would multiply articles; only full_cache/outlet is disjoint.
        source = pd.DataFrame({
            "date": ["2017-01-01", "2017-01-01", "2017-01-01", None],
            "outlet": ["Example", "", "Example", None],
            "scope": ["outlet", "all_outlets", "outlet", "outlet"],
            "endpoint_policy": ["full_cache", "full_cache", "study_half_open", "full_cache"],
            "total_articles": [5, 5, 5, 2],
        })
        pd.DataFrame({"day": ["2017-01-01", "__NULL__"], "outlet": ["Example", "__NULL__"],
                      "total_articles": [5, 2]}).to_csv(self.refs / "article_counts_day_outlet.csv", index=False)
        pd.DataFrame({"year": ["2017", "__NULL__"], "outlet": ["Example", "__NULL__"],
                      "total_articles": [5, 2]}).to_csv(self.refs / "article_counts_year_outlet.csv", index=False)
        self.runner.counts(source, {})
        counts = pd.read_csv(self.runner.output / "rebuilt_article_counts_outlet.csv")
        self.assertEqual(counts.total_articles.sum(), 7)
        self.assertEqual(len(self.runner.checks), 2)
        self.assertTrue(all(row["status"] == "matched" for row in self.runner.checks))

    def test_complete_corpus_counts_restore_undated_rows_and_reject_lost_dated_rows(self):
        daily = pd.DataFrame({"day": pd.to_datetime(["2017-01-02"]), "outlet": ["Example"], "total_articles": [5]})
        corpus = pd.DataFrame({"outlet": ["Example", None], "total_articles": [7, 3], "undated_articles": [2, 3]})
        result = comparison.add_undated_corpus_counts(daily, corpus)
        self.assertEqual(result.total_articles.sum(), 10)
        self.assertEqual(result.loc[result.day.isna(), "total_articles"].sum(), 5)
        again = comparison.add_undated_corpus_counts(result, corpus)
        pd.testing.assert_frame_equal(result, again)
        corpus.loc[0, "total_articles"] = 8
        with self.assertRaisesRegex(ValueError, "do not cover"):
            comparison.add_undated_corpus_counts(daily, corpus)


if __name__ == "__main__":
    unittest.main()
