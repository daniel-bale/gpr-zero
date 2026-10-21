# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "duckdb==1.2.1", "polars==1.30.0", "selectolax==0.3.29",
#   "tqdm==4.67.1", "pyarrow==20.0.0",
# ]
# ///
"""Synthetic checks; executable with uv run --no-project --python 3.12."""

import csv
import json
import math
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replicate_thesis_draft as replication


class ReplicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "input"
        (self.inputs / "2017").mkdir(parents=True)
        self.dictionary = self.root / "dictionary.json"
        self.dictionary.write_text(json.dumps({
            "word_sets": {"a": ["ALPHA", "a+b"], "b": ["beta"], "c": ["gamma"]},
            "categories": {"A_and_B": "a AND b", "A_and_C": "a AND c"},
        }))

    def row(self, identifier, content=None, title=None, timestamp="2017-01-02", outlet=None):
        html = (f"<h1>{title}</h1>" if title else "") + (f"<article>{content}</article>" if content else "")
        return {"warc_id": identifier, "warc_date": datetime.fromisoformat(timestamp),
                "url": "https://example.test/" + identifier, "domain": "example.test",
                "outlet": outlet or replication.THREE_OUTLETS[0], "is_english": False,
                "html_content_bytes": html.encode()}

    def parquet(self, name, rows):
        path = self.inputs / "2017" / name
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def args(self, *extra):
        return replication.parse_arguments([
            "--input-dir", str(self.inputs), "--dictionary", str(self.dictionary),
            "--output-dir", str(self.root / "results"), "--batch-size", "2", *extra,
        ])

    def test_counts_dates_title_only_literal_and_overlap(self):
        self.parquet("one.parquet", [
            self.row("and", "Alpha beta gamma"),
            self.row("literal", "a+b beta"),
            self.row("not_regex", "aaaab beta"),
            self.row("only_a", "alpha"),
            self.row("title_only", title="alpha beta"),
            self.row("empty"),
            self.row("early", "alpha beta", timestamp="2016-12-31"),
            self.row("late", "alpha beta", timestamp="2025-01-01"),
            self.row("other", "alpha beta", outlet="Other newspaper"),
            self.row("sunday", "alpha beta", timestamp="2017-01-08"),
            self.row("monday", "alpha beta", timestamp="2017-01-09"),
        ])
        result = replication.run(self.args())
        self.assertEqual(result["total_articles"], 7)
        self.assertEqual(result["matching_articles"], 4)
        self.assertEqual(result["title_only_articles"], 1)
        self.assertEqual(result["counts"]["outside_date_range"], 2)
        directory = self.root / "results/pilot/three"
        with (directory / "weekly_overall.csv").open() as stream:
            weeks = {row["week_start"]: row for row in csv.DictReader(stream)}
        self.assertEqual(int(weeks["2017-01-02"]["total_articles"]), 6)
        self.assertEqual(int(weeks["2017-01-02"]["matching_articles"]), 3)
        self.assertEqual(int(weeks["2017-01-09"]["matching_articles"]), 1)
        self.assertTrue(math.isnan(float(weeks["2017-01-16"]["gpr_percent"])))
        with (directory / "category_contributions.csv").open() as stream:
            contributions = list(csv.DictReader(stream))
        self.assertEqual([float(row["contribution_percent"]) for row in contributions], [100, 25])

    def test_cross_file_duplicates_and_resume_are_idempotent(self):
        self.parquet("one.parquet", [self.row("shared", "alpha beta")])
        self.parquet("two.parquet", [self.row("shared", "alpha beta"), self.row("new", "ordinary")])
        first = replication.run(self.args("--full"))
        resumed = replication.run(self.args("--full"))
        self.assertEqual(first["total_articles"], 2)
        self.assertEqual(first["matching_articles"], 1)
        self.assertEqual(first["counts"]["duplicate_accepted_ids"], 1)
        self.assertEqual(first, resumed)

    def test_interrupted_file_rolls_back_then_resumes(self):
        self.parquet("one.parquet", [self.row("first", "alpha beta")])
        self.parquet("two.parquet", [self.row("second", "alpha"), self.row("third", "alpha beta")])
        original = replication.insert_batch

        def fail_after_insert(conn, records):
            original(conn, records)
            if any(record["warc_id"] == "second" for record in records):
                raise RuntimeError("Synthetic interruption")

        with patch.object(replication, "insert_batch", fail_after_insert):
            with self.assertRaisesRegex(RuntimeError, "Synthetic interruption"):
                replication.run(self.args("--full"))
        conn = replication.setup_database(self.root / "results/full/three/articles.duckdb")
        self.assertEqual(conn.execute("SELECT count(*) FROM articles").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT count(*) FROM processed_files").fetchone()[0], 1)
        conn.close()
        result = replication.run(self.args("--full"))
        self.assertEqual(result["total_articles"], 3)
        self.assertEqual(result["matching_articles"], 2)

    def test_invalid_dictionary_and_changed_manifest_fail(self):
        self.parquet("one.parquet", [self.row("first", "alpha beta")])
        replication.run(self.args())
        document = json.loads(self.dictionary.read_text())
        document["word_sets"]["a"].append("changed")
        self.dictionary.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "Resume manifest differs"):
            replication.run(self.args())
        document["categories"]["bad"] = "a OR b"
        self.dictionary.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "Unsupported category expression"):
            replication.load_dictionary(self.dictionary)

    def test_all_outlets_includes_english_and_bounds_are_explicit(self):
        row = self.row("english", "Alpha beta", outlet="English paper")
        row["is_english"] = True
        self.parquet("one.parquet", [row])
        result = replication.run(self.args("--outlets", "all"))
        self.assertEqual(result["matching_articles"], 1)
        for options in (("--max-files", "0"), ("--full", "--max-files", "1")):
            with self.assertRaises(SystemExit):
                self.args(*options)


if __name__ == "__main__":
    unittest.main()
