# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["duckdb==1.2.1", "pandas==2.2.3", "numpy==2.2.3"]
# ///
"""Word lift tests distinguish article presence, date scope, and event means."""
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_raw_word_lifts as lifts


class WordLiftTests(unittest.TestCase):
    def setUp(self):
        self.connection = duckdb.connect()
        self.addCleanup(self.connection.close)
        self.connection.execute("CREATE TABLE articles(warc_id VARCHAR, warc_date TIMESTAMP, outlet VARCHAR, word_ids INTEGER[])")
        rows = [
            ("a", "2016-01-01", lifts.OUTLETS[0], [0]),  # baseline outside study
            ("b", "2024-01-07 16:00:00", lifts.OUTLETS[0], [0, 0, 1]),  # Taipei Monday
            ("c", "2024-01-14 15:59:59", lifts.OUTLETS[1], []),  # Sunday title only
            ("d", "2024-01-14 16:00:00", lifts.OUTLETS[2], [0]),  # next Monday
            ("e", "2024-01-22", lifts.OUTLETS[0], [0, 1]),
            ("f", None, lifts.OUTLETS[0], [0]),  # undated baseline still counted
            ("g", "2024-01-08", "Other outlet", [0, 1]),
        ]
        self.connection.executemany("INSERT INTO articles VALUES (?,?,?,?)", rows)

    def events(self, starts):
        dates = pd.to_datetime(starts)
        return pd.DataFrame({"start_date": dates, "end_date": dates})

    def test_presence_full_baseline_title_only_and_timezone_endpoints(self):
        base, info = lifts.prepare_counts(self.connection, "Asia/Taipei", 2)
        self.assertEqual(info["baseline_articles"], 6)
        self.assertEqual(info["undated_baseline_articles"], 1)
        self.assertEqual(base.set_index("word_id").baseline_count.to_dict(), {0: 5, 1: 2})
        result = lifts.calculate_family(self.connection, self.events(["2024-01-08"]), "step2", {"A", "B"}, ["A", "B"], base)
        self.assertEqual(result["coverage"].total_articles.tolist(), [2])
        detail = result["detail"].set_index("word")
        self.assertEqual(detail["count"].to_dict(), {"A": 1, "B": 1})
        self.assertAlmostEqual(detail.loc["A", "lift_score"], .6)
        self.assertAlmostEqual(detail.loc["B", "lift_score"], 1.5)

    def test_positive_spike_mean_differs_from_pooled_and_zero_filled(self):
        base, _ = lifts.prepare_counts(self.connection, "Asia/Taipei", 2)
        result = lifts.calculate_family(self.connection, self.events(["2024-01-08", "2024-01-15", "2024-01-22"]), "step2", {"A", "B"}, ["A", "B"], base)
        summary = result["summary"].set_index("word")
        # B appears in 1/2 and 1/1 articles across two positive-hit events.
        # Baseline frequency=2/6: lifts 1.5 and 3, mean 2.25. The zero-hit
        # middle event is excluded (zero-filled=1.5, pooled=1.5).
        self.assertAlmostEqual(summary.loc["B", "lift_score"], 2.25)
        self.assertEqual(summary.loc["B", "spike_count"], 2)
        self.assertEqual(summary.loc["B", "count"], 2)
        self.assertEqual(result["minimum_positive_spikes"], 2)
        self.assertIn("B", result["top10"].word.tolist())

    def test_empty_spike_is_visible_and_not_zero_frequency(self):
        base, _ = lifts.prepare_counts(self.connection, "UTC", 2)
        result = lifts.calculate_family(self.connection, self.events(["2030-01-07"]), "lau", {"B"}, ["A", "B"], base)
        self.assertEqual(result["spikes_without_articles"], 1)
        self.assertTrue(result["detail"].empty)
        self.assertTrue(result["summary"].empty)

    def test_invalid_word_ids_rejected(self):
        self.connection.execute("INSERT INTO articles VALUES ('bad',NULL,?,[999])", [lifts.OUTLETS[0]])
        with self.assertRaisesRegex(ValueError, "Invalid or null"):
            lifts.prepare_counts(self.connection, "UTC", 2)


class ContractTests(unittest.TestCase):
    def test_running_database_rejected_even_for_diagnostic(self):
        with self.assertRaisesRegex(ValueError, "still running"):
            lifts.completion_mode({"status": "running"}, True)
        manifest = {"status": "complete", "source_mode": "full", "complete": True,
                    "input_complete": True, "source_complete": True}
        self.assertEqual(lifts.completion_mode(manifest), "full_raw_rebuild")
        manifest["source_complete"] = 1
        with self.assertRaises(ValueError):
            lifts.completion_mode(manifest)

    def test_spike_dates_are_monday_nonoverlapping_week_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "spikes.csv"
            for starts, ends in [(["2024-01-09"], ["2024-01-09"]),
                                 (["2024-01-08", "2024-01-15"], ["2024-01-15", "2024-01-15"])]:
                pd.DataFrame({"start_date": starts, "end_date": ends}).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    lifts.read_spikes(path)

    def test_outer_comparison_preserves_missing_rows(self):
        old = pd.DataFrame({"word": ["a", "b"], "count": [1, 0], "lift_score": [1., 0.]})
        new = pd.DataFrame({"word": ["a", "c"], "count": [1, 0], "lift_score": [1. + 1e-13, 0.]})
        joined, summary = lifts.compare_frames(old, new, ["word"], ["count", "lift_score"])
        self.assertEqual(summary["status"], "different")
        self.assertEqual(summary["reference_only_rows"], 1)
        self.assertEqual(summary["rebuilt_only_rows"], 1)
        self.assertTrue(pd.isna(joined.set_index("word").loc["b", "count_rebuilt"]))

    def test_parse_exact_twenty_printed_thesis_rows(self):
        claims = lifts.read_printed_claims(lifts.ROOT / "replication/reference_thesis/contents/chapter03.tex")
        self.assertEqual(len(claims), 20)
        blockade = claims[claims.word.eq("封鎖")].iloc[0]
        self.assertEqual(blockade.printed_lift, 4.62)
        self.assertEqual(blockade.printed_spike_count, 22)

    def test_completed_historical_log_preserves_nineteen_not_claimed_twentyone(self):
        path = lifts.ROOT / "replication/raw_rebuild/word_lifts/alternative_references/logs/task9_word_lift_scores_v3_20250802_220639.log"
        ranges = lifts.read_historical_step2_log(path)
        self.assertEqual(len(ranges), 19)
        self.assertEqual(ranges.start_date.iloc[0], pd.Timestamp("2018-03-19"))
        self.assertEqual(ranges.end_date.iloc[-1], pd.Timestamp("2024-12-09"))

    def test_actual_dictionary_mapping_resolves_by_words_not_old_ordinals(self):
        folder = lifts.ROOT / "data/dictionaries/all_dictionaries"
        lau = lifts.ROOT / "data/dictionaries/yeu_tong_lau_2024.json"
        vocabulary = sorted(set().union(*(lifts.dictionary_words(path) for path in
            [*folder.glob("step1_*.json"), *folder.glob("step2_*.json"), lau])))
        families, paths = lifts.load_word_families(folder, lau, vocabulary)
        self.assertEqual(len(families["step2"]), 3341)
        self.assertEqual(len(families["lau"]), 47)
        self.assertEqual(len(paths), 81)
        self.assertIn("封鎖", families["lau"])

    def test_saved_detail_formula_reconstructs_summary_and_exposes_printed_drift(self):
        # This verifies the inferred Lau-specific formula against its complete
        # saved per-spike table; it is independent of any new-corpus claim.
        claims = lifts.read_printed_claims(lifts.ROOT / "replication/reference_thesis/contents/chapter03.tex")
        for family, total in (("step2", 21), ("lau", 22)):
            folder = lifts.ROOT / "data/word_contributions"
            prefix = "step2"
            if family == "lau":
                folder /= "lau_specific"
                prefix = "lau_specific"
            detail = pd.read_csv(folder / f"{prefix}_spike_word_lift_scores.csv")
            actual = lifts.summarize_lifts(detail)
            reference = pd.read_csv(folder / f"{prefix}_top_words_by_lift.csv")
            _, check = lifts.compare_frames(reference, actual, ["word"], ["count", "lift_score", "spike_count"])
            self.assertEqual(check["status"], "matched")
            top = actual[actual.spike_count.ge(__import__("math").ceil(total / 2))].head(10)
            compared = lifts.compare_printed(claims, family, {"summary": actual, "top10": top})
            if family == "lau":
                self.assertTrue(compared.matched.all(), compared.to_string())
            else:
                # Existing saved Step2 tables differ from all ten printed
                # ratios. A cache match must not imply a thesis-number match.
                self.assertEqual(int(compared.rounded_lift_matches.sum()), 0)


if __name__ == "__main__":
    unittest.main()
