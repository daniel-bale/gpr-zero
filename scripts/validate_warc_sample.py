#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "fastwarc==0.15.2", "brotli==1.1.0", "duckdb==1.2.1", "polars==1.30.0",
#   "selectolax==0.3.29", "tqdm==4.67.1", "pyarrow==20.0.0",
#   "pyahocorasick==2.1.0",
# ]
# ///
"""Validate complete, already downloaded WARC files against both frozen stages.

Selection JSON contains records (or files), each with relative_path (the exact
YYYY/MM/*.warc.parquet path), warc_path, and optional source_path. No networking,
record limit, source mutation, or global deduplication is performed. One WARC
file is processed at a time. Primary equality is direct field equality in source
row order, including bytes; multiplicity-preserving digest counters supplement
it to distinguish coverage, duplicate IDs, and order changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
from itertools import zip_longest
import json
import logging
from pathlib import Path
import re
import signal
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rebuild_raw_news as rebuild
import stage1_scan_warc as stage1

ROOT = Path(__file__).resolve().parents[1]
MISSING = object()


def canonical(value):
    """Canonical audit encoding; direct byte comparison does not use this hash."""
    if isinstance(value, bytes):
        return {"bytes_length": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, datetime):
        # Stage1 timestamps represent instants. Arrow unit/timezone metadata is
        # reported separately; aware equivalent instants compare equivalently.
        if value.tzinfo:
            value = value.astimezone(timezone.utc)
        return {"datetime": value.isoformat()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def row_digest(row, columns):
    return rebuild.object_hash([canonical(row[column]) for column in columns])


def compact(value):
    if isinstance(value, str) and len(value) > 256:
        return {"string_length": len(value), "utf8_sha256": hashlib.sha256(value.encode()).hexdigest()}
    return canonical(value)


def compare_rows(reference, actual, columns, sample_limit=25):
    """Consume each iterator once; never collapse duplicate WARC IDs to a dict."""
    ids = [Counter(), Counter()]
    fingerprints = [Counter(), Counter()]
    counts = [0, 0]
    fields = Counter()
    examples = []
    ordered_mismatches = 0
    html_bytes = [0, 0]
    for position, pair in enumerate(zip_longest(reference, actual, fillvalue=MISSING)):
        for side, row in enumerate(pair):
            if row is MISSING:
                continue
            counts[side] += 1
            ids[side][row["warc_id"]] += 1
            fingerprints[side][row_digest(row, columns)] += 1
            html_bytes[side] += len(row.get("html_content_bytes") or b"")
        left, right = pair
        if left is MISSING or right is MISSING:
            changed = ["__missing_row__"]
        else:
            changed = [column for column in columns if left[column] != right[column]]
        if changed:
            ordered_mismatches += 1
            fields.update(changed)
            if len(examples) < sample_limit:
                examples.append({"row": position, "fields": changed,
                    "reference": None if left is MISSING else {column: compact(left[column]) for column in set(changed) & set(columns) | {"warc_id"}},
                    "actual": None if right is MISSING else {column: compact(right[column]) for column in set(changed) & set(columns) | {"warc_id"}}})
    missing = sum((fingerprints[0] - fingerprints[1]).values())
    extra = sum((fingerprints[1] - fingerprints[0]).values())
    id_deltas = [{"warc_id": identity, "reference": ids[0][identity], "actual": ids[1][identity]}
                 for identity in sorted(ids[0].keys() | ids[1].keys(), key=str)
                 if ids[0][identity] != ids[1][identity]]
    return {"matched": ordered_mismatches == 0 and not missing and not extra,
        "reference_rows": counts[0], "actual_rows": counts[1],
        "ordered_mismatched_rows": ordered_mismatches, "field_mismatch_counts": dict(fields),
        "multiset_missing_rows": missing, "multiset_extra_rows": extra,
        "warc_id_multiplicity_matches": not id_deltas,
        "warc_id_multiplicity_differences": id_deltas,
        "reference_duplicate_id_count": sum(count > 1 for count in ids[0].values()),
        "actual_duplicate_id_count": sum(count > 1 for count in ids[1].values()),
        "reference_duplicate_extra_rows": sum(count - 1 for count in ids[0].values() if count > 1),
        "actual_duplicate_extra_rows": sum(count - 1 for count in ids[1].values() if count > 1),
        "reference_html_bytes": html_bytes[0], "actual_html_bytes": html_bytes[1],
        "mismatch_examples": examples,
        "comparison": "Direct equality of all fields at every row; SHA256 multiset and WARC-ID counters retain every duplicate occurrence"}


def parquet_rows(parquet, columns):
    for batch in parquet.iter_batches(batch_size=64, columns=columns, use_threads=False):
        yield from batch.to_pylist()


def compare_parquet(reference_path, actual_path, columns):
    with pq.ParquetFile(reference_path) as reference, pq.ParquetFile(actual_path) as actual:
        for label, parquet in (("reference", reference), ("actual", actual)):
            if set(parquet.schema_arrow.names) != set(columns):
                raise ValueError(f"Unexpected {label} schema: {parquet.schema_arrow.names}")
        result = compare_rows(parquet_rows(reference, columns), parquet_rows(actual, columns), columns)
        result.update(reference_schema=str(reference.schema_arrow), actual_schema=str(actual.schema_arrow),
                      arrow_schema_equal=reference.schema_arrow.equals(actual.schema_arrow))
        return result


class ErrorCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(self.format(record))


class ValidationTimeout(BaseException):
    pass


def deadline_handler(signum, frame):
    # BaseException avoids the historical scanner's broad per-record handlers.
    raise ValidationTimeout("Per-file validation wall-clock limit exceeded")


def bind_download_provenance(selection_path, records, proof_path):
    """Require one completed download proof per selected path, without masking duplicates."""
    if not proof_path.is_file():
        raise ValueError(f"Required download provenance is missing: {proof_path}")
    proof_bytes = proof_path.read_bytes()
    proof = json.loads(proof_bytes)
    if proof.get("complete") is not True:
        raise ValueError("Download provenance is not complete")
    selection_hash = rebuild.sha256(selection_path)
    if proof.get("selection_sha256") != selection_hash:
        raise ValueError("Download provenance selection SHA256 mismatch")
    if not proof.get("selection") or Path(proof["selection"]).resolve() != selection_path.resolve():
        raise ValueError("Download provenance selection path mismatch")
    downloaded = proof.get("records")
    if not isinstance(downloaded, list):
        raise ValueError("Download provenance records missing")
    expected_paths = [record["relative_path"] for record in records]
    actual_paths = [record.get("relative_path") for record in downloaded]
    if len(set(expected_paths)) != len(expected_paths) or len(set(actual_paths)) != len(actual_paths):
        raise ValueError("Duplicate selected/downloaded relative path")
    if set(expected_paths) != set(actual_paths):
        raise ValueError("Download provenance selected file set mismatch")
    by_path = {record["relative_path"]: record for record in downloaded}
    bound = []
    for record in records:
        download = by_path[record["relative_path"]]
        warc = Path(record["warc_path"]).resolve()
        if not download.get("path") or Path(download["path"]).resolve() != warc:
            raise ValueError("Download provenance WARC path mismatch")
        if not record.get("url") or download.get("url") != record["url"]:
            raise ValueError("Download provenance URL mismatch")
        if type(download.get("bytes")) is not int or download["bytes"] <= 0 or download["bytes"] != warc.stat().st_size:
            raise ValueError("Download provenance byte count mismatch")
        selected_length = record.get("head", {}).get("content_length")
        if selected_length is not None and selected_length != download["bytes"]:
            raise ValueError("Download provenance differs from selected HEAD byte count")
        if not isinstance(download.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", download["sha256"]):
            raise ValueError("Download provenance SHA256 is absent or malformed")
        if download.get("gzip_integrity") != "passed":
            raise ValueError("Download provenance gzip integrity did not pass")
        bound.append({**record, "download_proof": {key: download[key] for key in
                     ("path", "url", "bytes", "sha256", "gzip_integrity")}})
    if proof.get("total_bytes") != sum(record["download_proof"]["bytes"] for record in bound):
        raise ValueError("Download provenance total byte count mismatch")
    return bound, {"path": str(proof_path.resolve()), "sha256": hashlib.sha256(proof_bytes).hexdigest(),
                   "selection_sha256": selection_hash, "complete": True}


def verify_download_bytes(record):
    """Rehash the actual file immediately before scanning; also honor optional selection hashes."""
    path = Path(record["warc_path"])
    proof = record["download_proof"]
    if path.stat().st_size != proof["bytes"]:
        raise ValueError("Downloaded WARC byte count changed after provenance binding")
    digest = rebuild.sha256(path)
    if digest != proof["sha256"]:
        raise ValueError("Downloaded WARC SHA256 differs from completed download provenance")
    optional_digest = record.get("warc_sha256") or record.get("sha256")
    if optional_digest is not None and optional_digest != digest:
        raise ValueError("Downloaded WARC SHA256 differs from selection")
    return digest


def load_inputs(args):
    selection = json.loads(args.selection.read_text())
    records = selection.get("records", selection.get("files"))
    if not isinstance(records, list) or not records or len(records) > args.max_files:
        raise ValueError("Selection must contain 1..--max-files records")
    manifest = json.loads((args.rebuilt_dir / "manifest.json").read_text())
    progress = json.loads((args.rebuilt_dir / "progress.json").read_text())
    if manifest["mode"] != "full" or progress["status"] != "complete" or progress["completed_files"] != len(manifest["selected_files"]):
        raise ValueError("Completed full raw rebuild required")
    dependencies = manifest["dependencies"]
    if rebuild.object_hash(dependencies) != manifest["dependency_hash"]:
        raise ValueError("Rebuild dependency manifest hash mismatch")
    if rebuild.sha256(Path(rebuild.__file__)) != dependencies["runner_sha256"]:
        raise ValueError("Frozen rebuild runner changed")
    if rebuild.sha256(Path(dependencies["parser_file"])) != dependencies["parser_sha256"]:
        raise ValueError("Frozen reconstructed parser changed")
    words = json.loads((args.rebuilt_dir / "word_vocabulary.json").read_text())
    if rebuild.object_hash(words["words"]) != words["sha256"] or words["sha256"] != dependencies["vocabulary_sha256"]:
        raise ValueError("Frozen word vocabulary changed")
    for dictionary in dependencies["dictionary_files"]:
        if rebuild.sha256(Path(dictionary["path"])) != dictionary["sha256"]:
            raise ValueError(f"Frozen dictionary changed: {dictionary['path']}")
    inventory = {record["path"]: record for record in manifest["selected_files"]}
    if len(inventory) != len(manifest["selected_files"]):
        raise ValueError("Duplicate source path in full rebuild manifest")
    normalized = []
    for record in records:
        relative = Path(record["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3 or not relative.parts[0].isdigit() or not relative.parts[1].isdigit():
            raise ValueError("relative_path must preserve YYYY/MM/file.parquet")
        if str(relative) not in inventory:
            raise ValueError(f"Selected source not in complete rebuild: {relative}")
        source = Path(record.get("source_path", Path(manifest["input_dir"]) / relative)).resolve()
        if source != (Path(manifest["input_dir"]) / relative).resolve():
            raise ValueError("Source path does not agree with full rebuild inventory")
        warc = Path(record["warc_path"]).resolve()
        if not warc.is_file() or not warc.name.endswith(".warc.gz") or warc.stem + ".parquet" != relative.name:
            raise ValueError(f"Downloaded WARC filename/path mismatch: {warc}")
        if warc.stat().st_size > args.max_warc_bytes:
            raise ValueError(f"WARC exceeds explicit byte bound: {warc}")
        normalized.append({**record, "relative_path": str(relative), "warc_path": str(warc),
                           "source_path": str(source), "source_inventory": inventory[str(relative)]})
    if len({row["relative_path"] for row in normalized}) != len(normalized):
        raise ValueError("Repeated relative_path in selection")
    normalized, download_provenance = bind_download_provenance(args.selection, normalized, args.download_provenance)
    return normalized, manifest, words["words"], download_provenance


def validate_one(record, args, manifest, words):
    started = time.monotonic()
    relative = record["relative_path"]
    warc = Path(record["warc_path"])
    source = Path(record["source_path"])
    source_item = record["source_inventory"]
    if rebuild.source_stat(source, relative) != source_item:
        raise ValueError(f"Original Stage1 source stat changed: {source}")
    reference_checkpoint = rebuild.read_checkpoint(args.rebuilt_dir, source_item, manifest["dependency_hash"])
    if not reference_checkpoint or rebuild.sha256(source) != reference_checkpoint["source_sha256"]:
        raise ValueError("Original Stage1 bytes differ from the full rebuild checkpoint")
    warc_stat = {"size": warc.stat().st_size, "mtime_ns": warc.stat().st_mtime_ns}
    warc_hash = verify_download_bytes(record)
    stage1_dir = args.output_dir / "stage1"
    generated = stage1_dir / relative
    if generated.exists():
        raise ValueError(f"Refusing to overwrite prior extracted output: {generated}")
    generated.parent.mkdir(parents=True, exist_ok=True)
    errors = ErrorCapture()
    log_path = args.output_dir / "logs" / (relative + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(errors)
    logger.addHandler(handler)
    try:
        lookup = stage1.load_domain_config(stage1.TAIWANESE_DOMAINS_CONFIG)
        # Exactly the unchanged source worker; None means the entire WARC.
        scan = stage1.process_warc_file_stage1((warc, None, *lookup, generated.parent))
    finally:
        logger.removeHandler(errors)
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        handler.close()
    if scan["error"] or errors.messages:
        raise ValueError(f"Stage1 scan reported errors: {scan['error']}; {errors.messages[:5]}")
    if scan["parquet_file_path"] is None:
        # Historical scanner omits the empty file. Represent that absence as
        # zero rows explicitly so missing coverage cannot escape comparison.
        with pq.ParquetFile(source) as reference:
            pq.write_table(reference.schema_arrow.empty_table(), generated)
    elif Path(scan["parquet_file_path"]).resolve() != generated.resolve():
        raise ValueError("Stage1 generated an unexpected source-relative path")
    stage1_comparison = compare_parquet(source, generated, rebuild.SOURCE_COLUMNS)
    if stage1_comparison["actual_rows"] != scan["target_html_extracted"]:
        raise ValueError("Scanner extraction counter differs from emitted Parquet row count")
    dependencies = manifest["dependencies"]
    rebuilt_dir = args.output_dir / "stage2"
    rebuild.worker_init(str(stage1_dir), str(rebuilt_dir), words, dependencies["settings"],
                        manifest["dependency_hash"], dependencies["parser_file"])
    new_checkpoint = rebuild.process_source(rebuild.source_stat(generated, relative))
    reference_shard, _ = rebuild.artifact_paths(args.rebuilt_dir, relative)
    actual_shard, _ = rebuild.artifact_paths(rebuilt_dir, relative)
    stage2_comparison = compare_parquet(reference_shard, actual_shard, rebuild.SCHEMA.names)
    counts_equal = reference_checkpoint["counts"] == new_checkpoint["counts"]
    if warc_stat != {"size": warc.stat().st_size, "mtime_ns": warc.stat().st_mtime_ns}:
        raise ValueError("Downloaded WARC changed during validation")
    return {"relative_path": relative, "matched": stage1_comparison["matched"] and stage2_comparison["matched"] and counts_equal,
        "warc": {"path": str(warc), **warc_stat, "sha256": warc_hash},
        "download_proof": record["download_proof"],
        "source_parquet_sha256": reference_checkpoint["source_sha256"],
        "stage1": {"full_file_scan": True, "limit_records": None, "scan": scan, "comparison": stage1_comparison},
        "stage2": {"comparison": stage2_comparison, "checkpoint_counts_match": counts_equal,
                   "reference_counts": reference_checkpoint["counts"], "actual_counts": new_checkpoint["counts"]},
        "elapsed_seconds": time.monotonic() - started}


def run(args):
    args.output_dir = args.output_dir.resolve()
    records, manifest, words, download_provenance = load_inputs(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new empty validation output directory; prior evidence is never overwritten")
    source_root = Path(manifest["input_dir"]).resolve()
    if args.output_dir == source_root or source_root in args.output_dir.parents or args.output_dir == args.rebuilt_dir.resolve() or args.rebuilt_dir.resolve() in args.output_dir.parents:
        raise ValueError("Validation output cannot be inside either original source tree")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "scope": "Complete files, purposive bounded sample; not a corpus-wide WARC validation",
        "selection_sha256": rebuild.sha256(args.selection), "selected_files": len(records),
        "download_provenance": download_provenance,
        "script_sha256": rebuild.sha256(Path(__file__)), "stage1_script_sha256": rebuild.sha256(Path(stage1.__file__)),
        "frozen_rebuild_dependency_hash": manifest["dependency_hash"],
        "max_warc_bytes": args.max_warc_bytes, "timeout_seconds_per_file": args.timeout_seconds,
        "runtime_versions": {name: importlib.metadata.version(name) for name in
            ("fastwarc", "brotli", "duckdb", "polars", "selectolax", "tqdm", "pyarrow", "pyahocorasick")},
        "historical_stage1_runtime": {"fastwarc": "0.15.2", "brotli": "1.1.0", "polars": "1.28.1", "pyarrow": "19.0.1", "tqdm": "4.67.1"},
        "runtime_note": "Single process shares the frozen Stage2 Polars1.30/Arrow20 runtime. Historical Stage1 used Polars1.28.1/Arrow19; compare every logical field and direct HTML bytes, not Parquet serialization identity. Stage1 script itself is unchanged.",
        "record_limit": None, "files": []}
    path = args.output_dir / "run_summary.json"
    rebuild.atomic_json(path, summary)
    previous_handler = signal.signal(signal.SIGALRM, deadline_handler)
    try:
        for index, record in enumerate(records, 1):
            print(f"[{index}/{len(records)}] Full WARC scan: {record['relative_path']}", flush=True)
            signal.alarm(args.timeout_seconds)
            try:
                result = validate_one(record, args, manifest, words)
                summary["files"].append(result)
                rebuild.atomic_json(args.output_dir / "comparisons" / (record["relative_path"] + ".json"), result)
                print(f"[{index}/{len(records)}] matched={result['matched']} Stage1={result['stage1']['comparison']['actual_rows']} Stage2={result['stage2']['comparison']['actual_rows']}", flush=True)
            finally:
                signal.alarm(0)
            rebuild.atomic_json(path, summary)
            gc.collect()
        summary["status"] = "matched" if all(item["matched"] for item in summary["files"]) else "different"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        rebuild.atomic_json(path, summary)
    return 0 if summary["status"] == "matched" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--download-provenance", type=Path,
                        help="Required completed downloader proof (default: download_provenance.json beside selection)")
    parser.add_argument("--rebuilt-dir", type=Path, default=ROOT / "replication/raw_rebuild/news/full")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replication/warc_validation/validation")
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--max-warc-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args(argv)
    if min(args.max_files, args.max_warc_bytes, args.timeout_seconds) < 1:
        parser.error("All bounds must be positive")
    if args.download_provenance is None:
        args.download_provenance = args.selection.with_name("download_provenance.json")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
