# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "duckdb==1.2.1", "polars==1.30.0", "selectolax==0.3.29",
#   "tqdm==4.67.1", "pyarrow==20.0.0", "pyahocorasick==2.1.0",
# ]
# ///
"""Run using uv run --no-project --python 3.12; optional GPR_OLD_PILOT_DIR."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rebuild_raw_news as rebuild


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "source"
        self.inputs.mkdir()
        self.outputs = self.root / "results"
        self.dict_dir = self.root / "dictionaries"
        self.dict_dir.mkdir()
        self.dictionary = self.dict_dir / "test.json"
        self.dictionary.write_text(json.dumps({"word_sets": {"a": ["Alpha", "軍事", "軍事威脅", "a+b"]}}))
        self.words = ["Alpha", "a+b", "軍事", "軍事威脅"]

    def row(self, identifier, content=None, title=None, timestamp="2016-12-31"):
        html = (f"<h1>{title}</h1>" if title else "") + (f"<article>{content}</article>" if content else "")
        return {"warc_id": identifier, "warc_date": datetime.fromisoformat(timestamp),
                "url": "https://example.test/" + identifier, "domain": "example.test",
                "outlet": "Other paper", "is_english": True, "html_content_bytes": html.encode()}

    def source(self, name, rows):
        path = self.inputs / name
        pq.write_table(pa.Table.from_pylist(rows), path)
        return rebuild.source_stat(path, name)

    def init_worker(self):
        rebuild.worker_init(str(self.inputs), str(self.outputs), self.words,
                            {"batch_size": 1, "max_html_bytes": 1_000_000, "case_sensitive": True}, "test-dependency")

    def args(self, *extra):
        return rebuild.parse_arguments(["--input-dir", str(self.inputs), "--output-dir", str(self.outputs),
                                        "--dictionary-dir", str(self.dict_dir), "--extra-dictionary", str(self.dictionary),
                                        "--workers", "2", *extra])

    def test_aho_matches_exact_literal_substrings_and_overlaps(self):
        words = ["軍事", "軍事威脅", "威脅", "Alpha", "alpha", "a+b", "ß", "SS", "İ", "i\u0307"]
        automaton = rebuild.build_automaton(words)
        for content in [None, "", "軍事威脅軍事", "Alpha ALPHA alpha", "a+b aaaaab", "Straße SS", "İ i\u0307"]:
            expected = [index for index, word in enumerate(words) if word in (content or "")]
            self.assertEqual(rebuild.match_word_ids(content, automaton), expected)

    def test_shard_retains_duplicates_title_only_dates_and_provenance(self):
        item = self.source("one.parquet", [self.row("same", "軍事威脅 Alpha a+b"), self.row("same", "alpha"),
                                            self.row("title", title="軍事威脅", timestamp="2025-03-01"), self.row("empty")])
        self.init_worker()
        checkpoint = rebuild.process_source(item)
        shard, _ = rebuild.artifact_paths(self.outputs, item["path"])
        rows = pq.read_table(shard).to_pylist()
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["source_row"] for row in rows], [0, 1, 2])
        self.assertEqual([row["word_ids"] for row in rows], [[0, 1, 2, 3], [], []])
        self.assertEqual(rows[-1]["warc_date"].year, 2025)
        self.assertEqual(checkpoint["counts"]["title_only_articles"], 1)
        self.assertEqual(checkpoint["source_sha256"], rebuild.sha256(self.inputs / item["path"]))
        self.assertEqual(rebuild.read_checkpoint(self.outputs, item, "test-dependency"), checkpoint)
        with shard.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            rebuild.read_checkpoint(self.outputs, item, "test-dependency")

    def test_interrupted_shard_is_not_checkpointed(self):
        item = self.source("one.parquet", [self.row("one", "軍事"), self.row("two", "軍事")])
        self.init_worker()
        original = rebuild.historical_parser.extract_article_content_selectolax
        calls = 0

        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic interruption")
            return original(*args)

        with patch.object(rebuild.historical_parser, "extract_article_content_selectolax", fail_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                rebuild.process_source(item)
        self.assertIsNone(rebuild.read_checkpoint(self.outputs, item, "test-dependency"))
        shard, _ = rebuild.artifact_paths(self.outputs, item["path"])
        self.assertFalse(shard.exists())
        self.assertFalse(list(self.outputs.rglob("*.tmp-*")))
        self.assertEqual(rebuild.process_source(item)["counts"]["accepted_articles"], 2)

    def test_parallel_run_resume_runtime_workers_and_manifest_guard(self):
        (self.dict_dir / "word_column_mapping.json").write_text(json.dumps({"unrelated": "column mapping"}))
        self.source("one.parquet", [self.row("one", "軍事")])
        self.source("two.parquet", [self.row("two", "Alpha")])
        result = rebuild.run(self.args("--full"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_files"], 2)
        args = self.args("--full")
        args.workers = 1
        resumed = rebuild.run(args)
        self.assertEqual(resumed["resumed_files"], 2)
        self.assertEqual(resumed["counts"], result["counts"])
        self.assertGreater(result["max_worker_peak_rss_bytes"], 0)
        self.dictionary.write_text(json.dumps({"word_sets": {"a": ["changed"]}}))
        with self.assertRaisesRegex(ValueError, "Resume manifest differs"):
            rebuild.run(args)

    def test_selected_parser_record_context_and_hash_guard(self):
        self.source("one.parquet", [self.row("one", "ordinary")])
        parser = self.root / "selected_parser.py"
        parser.write_text("def extract_article_record(row):\n    return {'title': row['outlet'], 'content': 'Alpha', 'extracted_publish_date': None}\n")
        args = self.args("--full", "--parser-file", str(parser))
        result = rebuild.run(args)
        self.assertEqual(result["counts"]["articles_with_any_word"], 1)
        row = pq.read_table(self.outputs / "full/shards/one.parquet").to_pylist()[0]
        self.assertEqual(row["title"], "Other paper")
        self.assertEqual(row["content"], "Alpha")
        parser.write_text(parser.read_text() + "\n# Changed parser source\n")
        with self.assertRaisesRegex(ValueError, "Resume manifest differs"):
            rebuild.run(args)

    def test_historical_replay_keeps_rejects_and_limits_updates_to_eligible_sources(self):
        path = Path(__file__).resolve().parents[1] / "replication/reference_code/stage2_cache_replay.py"
        replay = rebuild.load_parser(path)
        row = self.row("one", None)
        row.update(domain="peoplenews.tw", outlet="People News (民報)",
                   _source_month="2019-12", _source_outlets=frozenset({"People News (民報)"}))
        row["html_content_bytes"] = ("<div id='main'>" + "Repair body " * 20 + "</div>").encode()
        rejected = replay.extract_article_record(row)
        self.assertIsNone(rejected["title"])
        self.assertIsNone(rejected["content"])
        row["html_content_bytes"] = b"<title>Original</title><h1>Repaired</h1>" + row["html_content_bytes"]
        repaired = replay.extract_article_record(row)
        self.assertEqual(repaired["title"], "Repaired")
        self.assertTrue(repaired["content"].startswith("Repair body"))
        row["_source_month"] = "2019-01"
        untouched = replay.extract_article_record(row)
        self.assertEqual(untouched["title"], "Original")
        self.assertIsNone(untouched["content"])

    def test_recovered_full_dictionary_vocabulary(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "data/dictionaries/all_dictionaries"
        if not directory.is_dir():
            self.skipTest("Recovered full dictionary directory unavailable")
        paths = sorted(path for path in directory.glob("*.json") if path.name != "word_column_mapping.json")
        extras = [root / "data/dictionaries/taiwan_gpr_dict_president.json", root / "data/dictionaries/yeu_tong_lau_2024.json"]
        if not all(path.is_file() for path in extras):
            self.skipTest("Recovered original dictionaries unavailable")
        words, sources = rebuild.collect_vocabulary(paths + extras)
        self.assertEqual(len(paths), 80)
        self.assertEqual(len(sources), 82)
        self.assertGreaterEqual(len(words), 3341)
        self.assertEqual(words, sorted(set(words)))
        rebuild.build_automaton(words)

    @unittest.skipUnless(os.environ.get("GPR_OLD_PILOT_DIR"), "Set GPR_OLD_PILOT_DIR for eight-file real-data equality check")
    def test_complete_old_pilot_rows_and_original_dictionary_counts(self):
        pilot = Path(os.environ["GPR_OLD_PILOT_DIR"])
        manifest = json.loads((pilot / "manifest.json").read_text())
        dictionary = json.loads(Path(manifest["dictionary"]).read_text())
        words = sorted({word for values in dictionary["word_sets"].values() for word in values})
        rebuild.worker_init(manifest["input_dir"], str(self.outputs), words,
                            {"batch_size": 128, "max_html_bytes": 128 * 1024 * 1024}, "pilot-check")
        for selected in manifest["selected_files"]:
            rebuild.process_source({key: selected[key] for key in ("path", "size", "mtime_ns")})
        conn = duckdb.connect()
        try:
            original_conn = duckdb.connect(str(pilot / "articles.duckdb"), read_only=True)
            original = original_conn.execute("SELECT warc_id,warc_date,url,domain,outlet,is_english,title,content,extracted_publish_date FROM articles ORDER BY warc_id").fetchall()
            original_conn.close()
            rebuilt = conn.execute("""SELECT warc_id,warc_date,url,domain,outlet,is_english,title,content,extracted_publish_date
                FROM read_parquet(?) WHERE warc_date >= TIMESTAMP '2017-01-01' AND warc_date < TIMESTAMP '2025-01-01'
                AND outlet IN ('China Times (中國時報)','Liberty Times (自由時報)','United Daily (聯合報)')
                QUALIFY row_number() OVER (PARTITION BY warc_id ORDER BY source_path,source_row)=1 ORDER BY warc_id""",
                [str(self.outputs / "shards/**/*.parquet")]).fetchall()
            self.assertEqual(rebuilt, original)
            self.assertEqual(len(rebuilt), 474)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
