# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["duckdb==1.2.1", "pandas==2.2.3", "numpy==2.2.3"]
# ///
import sys
import unittest
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_raw_newspaper_words as words


class NewspaperWordTests(unittest.TestCase):
    def test_historical_pooling_and_corrected_outlet_condition(self):
        connection = duckdb.connect()
        self.addCleanup(connection.close)
        connection.execute("CREATE TABLE articles(warc_date TIMESTAMP, outlet VARCHAR, word_ids INTEGER[])")
        connection.executemany("INSERT INTO articles VALUES (?,?,?)", [
            ("2024-01-07 16:00:00", words.OUTLETS[0], [0, 0]),
            ("2024-01-14 15:59:59", words.OUTLETS[1], []),
            ("2024-01-08", words.OUTLETS[2], [0]),
            ("2024-01-14 16:00:00", words.OUTLETS[0], [0]),
        ])
        events = pd.DataFrame({"spike_id": [1, 2], "newspaper": ["中國時報", "自由時報"],
            "start_date": pd.to_datetime(["2024-01-08"] * 2), "end_date": pd.to_datetime(["2024-01-08"] * 2)})
        words.prepare_daily(connection, "Asia/Taipei", {0: "軍演"})
        pooled, pooled_denominators = words.calculate_variant(connection, events, False)
        corrected, corrected_denominators = words.calculate_variant(connection, events, True)
        self.assertEqual(pooled_denominators.total_articles.tolist(), [3, 3])
        self.assertEqual(pooled["count"].tolist(), [2, 2])
        self.assertEqual(corrected_denominators.total_articles.tolist(), [1, 1])
        self.assertEqual(corrected.spike_id.tolist(), [1])
        self.assertEqual(corrected["count"].tolist(), [1])

    def test_positive_event_mean_not_pooled_or_zero_filled_and_direction(self):
        detail = pd.DataFrame({"spike_id": [1, 2, 3, 4, 5], "newspaper": ["中國時報"] * 3 + ["自由時報"] * 2,
            "word": ["軍演"] * 5, "count": [1, 8, 0, 1, 0], "total_articles": [2, 10, 20, 4, 10]})
        result = words.appearance_ratios(detail).iloc[0]
        self.assertAlmostEqual(result.china_mean_appearance_rate, .65)
        self.assertAlmostEqual(result.liberty_mean_appearance_rate, .25)
        self.assertEqual(result.china_positive_spike_count, 2)
        self.assertAlmostEqual(result.ratio_liberty_china, .25 / (.65 + 1e-6))
        self.assertAlmostEqual(result.ratio_china_liberty_plain, 2.6)

    def test_missing_newspaper_presence_is_explicit_zero(self):
        detail = pd.DataFrame({"spike_id": [1], "newspaper": ["中國時報"], "word": ["軍演"], "count": [1], "total_articles": [2]})
        result = words.appearance_ratios(detail).iloc[0]
        self.assertTrue(result.liberty_zero)
        self.assertFalse(result.china_zero)
        self.assertTrue(pd.isna(result.ratio_china_liberty_plain))

    def test_small_sample_top_and_bottom_overlap_is_not_cartesian_join(self):
        detail = pd.DataFrame({"spike_id": [1, 2], "newspaper": ["中國時報", "自由時報"],
            "word": ["軍演", "軍演"], "count": [1, 1], "total_articles": [2, 4]})
        claims = pd.DataFrame({"word": ["軍演"], "printed_ratio": [.5],
            "printed_side": ["China Times (中國時報)"], "printed_rank": [1]})
        compared = words.compare_printed(claims, words.appearance_ratios(detail))
        self.assertEqual(len(compared), 1)
        self.assertTrue(compared.rank_matches.iloc[0])

    def test_independent_event_detection_uses_observed_weeks_sample_sd_and_first_peak(self):
        weekly = pd.DataFrame({"week": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-29",
            "2024-02-05", "2024-02-12", "2024-02-19", "2024-02-26", "2024-03-04", "2024-03-11"]),
            "outlet": [words.OUTLETS[0]] * 10, "category": ["overall"] * 10,
            "total_articles": [10] * 10, "matching_articles": [0, 10, 10, 10, 0, 0, 0, 0, 0, 0]})
        events, definitions = words.detect_events(weekly)
        self.assertEqual(len(events), 2)
        self.assertEqual(events.start_date.dt.strftime("%Y-%m-%d").tolist(), ["2024-01-08", "2024-01-29"])
        self.assertEqual(events.end_date.iloc[0], pd.Timestamp("2024-01-15"))
        self.assertEqual(events.peak_date.iloc[0], pd.Timestamp("2024-01-08"))
        self.assertEqual(definitions[0]["observed_weeks"], 10)
        self.assertAlmostEqual(definitions[0]["sample_std"], (weekly.matching_articles / (weekly.total_articles + 1e-9)).std(ddof=1))
        pooled, _ = words.detect_events(weekly.assign(outlet="three_outlets"), {"three_outlets": "Overall Taiwan - Overall"})
        self.assertEqual(pooled.newspaper.tolist(), ["Overall Taiwan - Overall"] * 2)

    def test_recovered_reference_recreates_all_twenty_numbers_in_opposite_direction_to_footnote(self):
        args = words.parse_args([])
        events, detail = words.load_decomposition(args.decomposition)
        self.assertEqual(len(events), 351)
        actual = words.appearance_ratios(detail)
        printed = words.compare_printed(words.printed_claims(args.thesis), actual)
        self.assertEqual(int(printed.producer_direction_matches.sum()), 20)
        self.assertEqual(int(printed.footnote_direction_matches.sum()), 0)
        self.assertEqual(int(printed.rank_matches.sum()), 20)


if __name__ == "__main__":
    unittest.main()
