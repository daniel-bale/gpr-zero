# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fastwarc==0.15.2", "brotli==1.1.0", "duckdb==1.2.1", "polars==1.30.0", "selectolax==0.3.29", "tqdm==4.67.1", "pyarrow==20.0.0", "pyahocorasick==2.1.0"]
# ///
import sys
from pathlib import Path
import unittest
import gzip
import tempfile
import uuid
import json
import copy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_warc_sample as validation


class RowComparisonTests(unittest.TestCase):
    columns = ["warc_id", "html_content_bytes", "content", "word_ids", "source_row"]

    def row(self, identity="a", html=b"<p>same</p>", content=None, source_row=0):
        return {"warc_id": identity, "html_content_bytes": html, "content": content,
                "word_ids": [0, 2], "source_row": source_row}

    def compare(self, left, right):
        return validation.compare_rows(iter(left), iter(right), self.columns)

    def test_exact_bytes_and_all_fields_match(self):
        row = self.row(html=b"\x00\xff\r\n")
        result = self.compare([row], [row.copy()])
        self.assertTrue(result["matched"])
        self.assertEqual(result["reference_html_bytes"], 4)

    def test_missing_extra_and_duplicate_rows_fail_without_dict_masking(self):
        row = self.row()
        for left, right, missing, extra in [([row], [], 1, 0), ([], [row], 0, 1), ([row], [row, row], 0, 1)]:
            with self.subTest(left=len(left), right=len(right)):
                result = self.compare(left, right)
                self.assertFalse(result["matched"])
                self.assertEqual(result["multiset_missing_rows"], missing)
                self.assertEqual(result["multiset_extra_rows"], extra)
                self.assertFalse(result["warc_id_multiplicity_matches"])

    def test_duplicate_ids_with_distinct_payloads_preserve_both_occurrences(self):
        first, second = self.row(), self.row(html=b"different", source_row=1)
        result = self.compare([first, second], [first, second])
        self.assertTrue(result["matched"])
        self.assertEqual(result["reference_duplicate_id_count"], 1)
        self.assertEqual(result["reference_duplicate_extra_rows"], 1)
        masked = self.compare([first, second], [second, second])
        self.assertFalse(masked["matched"])
        self.assertTrue(masked["warc_id_multiplicity_matches"])
        self.assertEqual(masked["multiset_missing_rows"], 1)

    def test_byte_mutation_and_null_empty_distinction_detected(self):
        result = self.compare([self.row()], [self.row(html=b"<p>Same</p>", content="")])
        self.assertFalse(result["matched"])
        self.assertEqual(result["field_mismatch_counts"], {"html_content_bytes": 1, "content": 1})

    def test_row_order_and_source_row_are_separate_checks(self):
        first, second = self.row("a"), self.row("b", source_row=1)
        result = self.compare([first, second], [second, first])
        self.assertFalse(result["matched"])
        self.assertEqual(result["multiset_missing_rows"], 0)
        self.assertEqual(result["ordered_mismatched_rows"], 2)
        changed_row = {**first, "source_row": 4}
        self.assertEqual(self.compare([first], [changed_row])["field_mismatch_counts"], {"source_row": 1})

    def test_word_id_order_content_and_duplicate_hits_are_exact(self):
        row = self.row()
        for ids in ([0], [2, 0], [0, 0, 2]):
            result = self.compare([row], [{**row, "word_ids": ids}])
            self.assertFalse(result["matched"])
            self.assertEqual(result["field_mismatch_counts"], {"word_ids": 1})


class UnmodifiedScannerTests(unittest.TestCase):
    def test_whole_warc_extracts_every_target_html_payload_without_decoding(self):
        payloads = [b"<p>first</p>", b"<p>\xff\x00\r\nsecond</p>", b"<p>last</p>"]
        records = []
        for index, body in enumerate([*payloads, b"other domain"]):
            url = f"https://chinatimes.com/test/{index}" if index < 3 else "https://example.org/"
            http = (f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
            header = (f"WARC/1.0\r\nWARC-Type: response\r\nWARC-Record-ID: <urn:uuid:{uuid.uuid4()}>\r\n"
                      f"WARC-Target-URI: {url}\r\nWARC-Date: 2024-01-01T00:00:00Z\r\n"
                      f"Content-Type: application/http; msgtype=response\r\nContent-Length: {len(http)}\r\n\r\n").encode()
            records.append(header + http + b"\r\n\r\n")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            warc = root / "CC-NEWS-20240101000000-00001.warc.gz"
            warc.write_bytes(gzip.compress(b"".join(records)))
            lookup = validation.stage1.load_domain_config(validation.stage1.TAIWANESE_DOMAINS_CONFIG)
            result = validation.stage1.process_warc_file_stage1((warc, None, *lookup, root / "stage1"))
            self.assertIsNone(result["error"])
            self.assertEqual(result["records_processed"], 4)
            self.assertEqual(result["target_html_extracted"], 3)
            with validation.pq.ParquetFile(result["parquet_file_path"]) as parquet:
                rows = list(validation.parquet_rows(parquet, validation.rebuild.SOURCE_COLUMNS))
            self.assertEqual([row["html_content_bytes"] for row in rows], payloads)


class DownloadProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.warc = self.root / "sample.warc.gz"
        self.warc.write_bytes(b"sample downloaded bytes")
        self.records = [{"relative_path": "2024/01/sample.warc.parquet", "warc_path": str(self.warc),
                         "url": "https://data.commoncrawl.org/sample.warc.gz"}]
        self.selection = self.root / "selection.json"
        self.selection.write_text(json.dumps({"records": self.records}))
        self.proof_path = self.root / "download_provenance.json"
        self.proof = {"complete": True, "selection": str(self.selection),
            "selection_sha256": validation.rebuild.sha256(self.selection),
            "total_bytes": self.warc.stat().st_size,
            "records": [{"relative_path": self.records[0]["relative_path"], "path": str(self.warc),
                         "url": self.records[0]["url"], "bytes": self.warc.stat().st_size,
                         "sha256": validation.rebuild.sha256(self.warc), "gzip_integrity": "passed"}]}

    def bind(self, proof=None):
        self.proof_path.write_text(json.dumps(self.proof if proof is None else proof))
        return validation.bind_download_provenance(self.selection, self.records, self.proof_path)

    def test_completed_proof_binds_exact_selection_and_actual_bytes(self):
        bound, provenance = self.bind()
        self.assertTrue(provenance["complete"])
        self.assertEqual(validation.verify_download_bytes(bound[0]), self.proof["records"][0]["sha256"])
        self.assertEqual(provenance["sha256"], validation.rebuild.sha256(self.proof_path))

    def test_missing_incomplete_and_wrong_selection_proof_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validation.bind_download_provenance(self.selection, self.records, self.proof_path)
        for field, value in (("complete", False), ("complete", 1), ("selection_sha256", "0" * 64),
                             ("selection", str(self.root / "other.json")), ("total_bytes", 1)):
            proof = copy.deepcopy(self.proof)
            proof[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.bind(proof)

    def test_file_set_duplicates_path_url_size_hash_and_gzip_proof_rejected(self):
        for field, value in (("relative_path", "extra.parquet"), ("path", str(self.root / "wrong.warc.gz")),
                             ("url", "https://example.org/wrong"), ("bytes", 1),
                             ("sha256", None), ("gzip_integrity", "failed")):
            proof = copy.deepcopy(self.proof)
            proof["records"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.bind(proof)
        for records in ([], self.proof["records"] * 2):
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                self.bind({**self.proof, "records": records})

    def test_actual_mutation_and_optional_selection_checksum_rejected(self):
        bound, _ = self.bind()
        with self.assertRaisesRegex(ValueError, "differs from selection"):
            validation.verify_download_bytes({**bound[0], "warc_sha256": "0" * 64})
        self.warc.write_bytes(b"Sample downloaded bytes")  # Same length, different bytes.
        with self.assertRaisesRegex(ValueError, "completed download provenance"):
            validation.verify_download_bytes(bound[0])


if __name__ == "__main__":
    unittest.main()
