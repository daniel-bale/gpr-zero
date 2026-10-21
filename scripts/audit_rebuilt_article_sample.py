#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow==20.0.0", "duckdb==1.2.1", "polars==1.30.0", "selectolax==0.3.29", "tqdm==4.67.1"]
# ///
"""Compare a bounded parsed-article sample with remote cached text fingerprints.

Only metadata, text hashes, lengths, and null/empty flags cross SSH. The remote
query is read-only with 512 MB memory, two threads, and a 60-second default cap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq


META_FIELDS = ["warc_id", "warc_date", "url", "domain", "outlet", "is_english", "extracted_publish_date"]
TEXT_FIELDS = [f"{field}_{suffix}" for field in ("title", "content") for suffix in ("is_null", "is_empty", "char_length", "sha256")]
COMPARE_FIELDS = META_FIELDS[1:] + TEXT_FIELDS
SQL = """SELECT warc_id,warc_date,url,domain,outlet,is_english,extracted_publish_date,
    title IS NULL AS title_is_null, title = '' AS title_is_empty,
    length(title) AS title_char_length, sha256(title) AS title_sha256,
    content IS NULL AS content_is_null, content = '' AS content_is_empty,
    length(content) AS content_char_length, sha256(content) AS content_sha256
FROM gpr_binary_matrix_all_words WHERE warc_id IN (SELECT unnest(?)) ORDER BY warc_id"""

REMOTE_CODE = r'''
import datetime, json, os, threading, time
import duckdb
config = json.loads(CONFIG_JSON)
stat = os.stat(config["database"])
source = {"path": config["database"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
connection = duckdb.connect(config["database"], read_only=True)
connection.execute("SET memory_limit='512MB'")
connection.execute("SET threads=2")
if config["table"] == "articles":
    constraints = connection.execute("SELECT constraint_type,constraint_column_names FROM duckdb_constraints() WHERE table_name='articles'").fetchall()
    if not any(kind == "PRIMARY KEY" and columns == ["warc_id"] for kind, columns in constraints):
        raise RuntimeError("Base articles fallback requires a primary-key index on warc_id")
timed_out = threading.Event()
def interrupt():
    timed_out.set()
    connection.interrupt()
timer = threading.Timer(config["timeout_seconds"], interrupt)
timer.daemon = True
started = time.monotonic()
timer.start()
try:
    result = connection.execute(config["sql"], config["parameters"])
    columns = [item[0] for item in result.description]
    rows = [dict(zip(columns, row)) for row in result.fetchall()]
    after = os.stat(config["database"])
    if source["size"] != after.st_size or source["mtime_ns"] != after.st_mtime_ns:
        raise RuntimeError("Source database metadata changed during query")
    text_bytes = 0
    if config.get("text_byte_limit"):
        text_bytes = sum(len((row.get(field) or "").encode()) for row in rows for field in ("title", "content"))
        if text_bytes > config["text_byte_limit"]:
            raise RuntimeError("Text byte cap exceeded; no article text returned")
    report = {"status": "complete", "records": rows, "text_bytes": text_bytes}
except Exception as error:
    report = {"status": "timeout" if timed_out.is_set() else "failed", "error": str(error), "records": []}
finally:
    timer.cancel()
    connection.close()
report.update({"source": source, "elapsed_seconds": time.monotonic() - started,
               "duckdb_version": duckdb.__version__, "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(json.dumps(report, ensure_ascii=False, default=lambda value: value.isoformat()))
'''


def hashed(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize(value):
    return value.isoformat() if isinstance(value, (datetime, date)) else value


def fingerprints(row: dict) -> dict:
    result = {key: normalize(row[key]) for key in META_FIELDS}
    for field in ("title", "content"):
        value = row[field]
        result.update({field + "_is_null": value is None,
                       field + "_is_empty": value == "" if value is not None else None,
                       field + "_char_length": len(value) if value is not None else None,
                       field + "_sha256": hashed(value.encode()) if value is not None else None})
    return result


def atomic_text(path: Path, value: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="")
    temporary.replace(path)


def write_csv(path: Path, fields: list[str], rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        # Distinguish null from the empty string and preserve boolean values.
        writer.writerow({key: "__NULL__" if row.get(key) is None else row[key] for key in fields})
    atomic_text(path, stream.getvalue())


def comparison_diagnostics(samples, records):
    """Describe systematic differences without changing either source."""
    remote = {row["warc_id"]: row for row in records}
    offsets, title_outlets, content_outlets = Counter(), Counter(), Counter()
    bucket_changes = Counter({"day": 0, "week": 0, "year": 0})
    comparable = exact_without_date = matching_title = matching_content = matching_both = 0
    for local in samples:
        cached = remote.get(local["warc_id"])
        if cached is None:
            continue
        comparable += 1
        title_equal = all(local[field] == cached[field] for field in TEXT_FIELDS if field.startswith("title_"))
        content_equal = all(local[field] == cached[field] for field in TEXT_FIELDS if field.startswith("content_"))
        matching_title += title_equal
        matching_content += content_equal
        matching_both += title_equal and content_equal
        exact_without_date += all(local[field] == cached[field] for field in COMPARE_FIELDS if field != "warc_date")
        if not title_equal:
            title_outlets[str(local["outlet"])] += 1
        if not content_equal:
            content_outlets[str(local["outlet"])] += 1
        if local["warc_date"] not in (None, "__NULL__") and cached["warc_date"] not in (None, "__NULL__"):
            left, right = datetime.fromisoformat(local["warc_date"]), datetime.fromisoformat(cached["warc_date"])
            offsets[str((right - left).total_seconds())] += 1
            bucket_changes["day"] += left.date() != right.date()
            bucket_changes["week"] += (left.date() - timedelta(days=left.weekday())) != (right.date() - timedelta(days=right.weekday()))
            bucket_changes["year"] += left.year != right.year
    return {"comparable_articles": comparable, "cache_minus_rebuild_timestamp_seconds": dict(offsets),
            "matching_title_fingerprints": matching_title, "matching_content_fingerprints": matching_content,
            "matching_both_text_fingerprints": matching_both, "exact_matches_excluding_warc_date": exact_without_date,
            "title_mismatches_by_outlet": dict(title_outlets), "content_mismatches_by_outlet": dict(content_outlets),
            "date_bucket_differences_in_sample": dict(bucket_changes),
            "interpretation": "A constant timestamp offset is observed evidence, not a correction applied to the raw rebuild; text mismatches remain independent of date handling."}


def choose_sample(directory: Path, per_year: int) -> tuple[list[dict], dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    if manifest["mode"] != "benchmark" or progress["status"] != "complete":
        raise ValueError("Audit requires a complete bounded benchmark, not an unfinished or full corpus")
    if len(manifest["selected_files"]) > 100:
        raise ValueError("Refusing to read more than100 local benchmark shards")
    unique = {}
    read_rows = 0
    for item in sorted(manifest["selected_files"], key=lambda value: value["path"]):
        path = directory / "shards" / item["path"]
        checkpoint = json.loads((directory / "checkpoints" / (item["path"] + ".json")).read_text())
        if path.stat().st_size != checkpoint["shard_size"] or hashed(path.read_bytes()) != checkpoint["shard_sha256"]:
            raise ValueError(f"Shard does not match checkpoint: {path}")
        columns = META_FIELDS + ["title", "content", "source_path", "source_row"]
        parquet = pq.ParquetFile(path)
        try:
            for batch in parquet.iter_batches(batch_size=128, columns=columns):
                for row in batch.to_pylist():
                    read_rows += 1
                    if row["warc_id"] in unique:
                        continue
                    year = str(row["warc_date"].year) if row["warc_date"] is not None else "__NULL__"
                    unique[row["warc_id"]] = {**fingerprints(row), "year": year,
                                               "source_path": row["source_path"], "source_row": row["source_row"]}
        finally:
            parquet.close()
    grouped = defaultdict(lambda: defaultdict(list))
    for row in unique.values():
        grouped[row["year"]][row["outlet"]].append(row)
    selected = []
    for year, outlets in sorted(grouped.items()):
        queues = {outlet: sorted(rows, key=lambda row: hashed(row["warc_id"].encode())) for outlet, rows in outlets.items()}
        # Stable pseudorandom outlet ordering avoids always favoring alphabetical outlets.
        order = sorted(queues, key=lambda outlet: hashed(str(outlet).encode()))
        year_rows = []
        while len(year_rows) < per_year and any(queues.values()):
            for outlet in order:
                if queues[outlet]:
                    year_rows.append(queues[outlet].pop(0))
                    if len(year_rows) == per_year:
                        break
        selected.extend(year_rows)
    return selected, {"benchmark_manifest_sha256": hashed((directory / "manifest.json").read_bytes()),
                      "benchmark_shards": len(manifest["selected_files"]), "read_rows": read_rows,
                      "unique_local_ids": len(unique), "sampled_by_year": dict(Counter(row["year"] for row in selected)),
                      "sampled_by_outlet": dict(Counter(str(row["outlet"]) for row in selected))}


def parse_arguments(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=root / "replication/raw_rebuild/news/benchmark-20")
    parser.add_argument("--output-dir", type=Path, default=root / "replication/raw_rebuild/article_sample_audit")
    parser.add_argument("--per-year", type=int, default=10)
    parser.add_argument("--host", default="bdsrc")
    parser.add_argument("--remote-python", default="/home5/chuenho/gpr-zero/.venv/bin/python")
    parser.add_argument("--database", default="/home5/chuenho/data/news/stage2_enhanced_full.duckdb")
    parser.add_argument("--table", choices=("gpr_binary_matrix_all_words", "articles"), default="gpr_binary_matrix_all_words")
    parser.add_argument("--prior-audit", type=Path, help="Required for indexed-base fallback; preserves the first timed-out audit")
    parser.add_argument("--inspect-five-text-examples", action="store_true",
                        help="Explicitly retrieve up to100KB of text for five existing mismatches, using indexed articles")
    parser.add_argument("--parser-file", type=Path, help="Historical parser module for local saved-sample validation")
    parser.add_argument("--reparse-saved-sample", action="store_true",
                        help="Parse the100saved IDs from source HTML using --parser-file; no remote query")
    parser.add_argument("--assert-disjoint-benchmark", type=Path, help="Reject overlap with source files in this prior benchmark")
    parser.add_argument("--assert-disjoint-sample", type=Path, help="Reject warc_id overlap with this prior local_sample.csv")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    if not 1 <= args.per_year <= 25 or not 1 <= args.timeout_seconds <= 60:
        parser.error("--per-year must be1..25; --timeout-seconds must be1..60")
    if args.table == "articles" and args.prior_audit is None:
        parser.error("--table articles requires --prior-audit pointing to the original timeout summary")
    return args


def reparse_saved_sample(args):
    if args.parser_file is None:
        raise ValueError("--reparse-saved-sample requires --parser-file")
    parser_path = args.parser_file.resolve()
    spec = importlib.util.spec_from_file_location("candidate_parser", parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    audit_dir = args.output_dir.resolve()
    with (audit_dir / "local_sample.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    with (audit_dir / "remote_fingerprints.csv").open() as stream:
        remote = {row["warc_id"]: row for row in csv.DictReader(stream)}
    manifest = json.loads((args.benchmark_dir / "manifest.json").read_text())
    input_dir = Path(manifest["input_dir"])
    expected_sources = {item["path"]: item for item in manifest["selected_files"]}
    wanted = defaultdict(dict)
    for row in saved:
        wanted[row["source_path"]][int(row["source_row"])] = row["warc_id"]
    results, differences = [], []
    for source, positions in wanted.items():
        path = input_dir / source
        metadata = path.stat()
        if metadata.st_size != expected_sources[source]["size"] or metadata.st_mtime_ns != expected_sources[source]["mtime_ns"]:
            raise ValueError(f"Source metadata changed: {source}")
        parquet = pq.ParquetFile(path)
        prepare_source = getattr(parser, "prepare_source", None)
        source_context = prepare_source(parquet, source) if prepare_source else {"_source_path": source}
        position = 0
        try:
            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    source_row = position
                    position += 1
                    if source_row not in positions:
                        continue
                    if row["warc_id"] != positions[source_row]:
                        raise ValueError("Sample source position no longer identifies the expected warc_id")
                    record_parser = getattr(parser, "extract_article_record", None)
                    extracted = (record_parser({**row, **source_context}) if record_parser else
                                 parser.extract_article_content_selectolax(row["html_content_bytes"], row["url"], row["domain"]))
                    timestamp = row["warc_date"]
                    if timestamp is not None and timestamp.tzinfo:
                        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
                    # Compatibility comparison only; stored rebuild dates remain UTC.
                    compared = fingerprints({**row, **extracted, "warc_date": timestamp + timedelta(hours=8) if timestamp else None})
                    serialized = {key: "__NULL__" if value is None else str(value) for key, value in compared.items()}
                    cached = remote[row["warc_id"]]
                    mismatch = [key for key in COMPARE_FIELDS if serialized[key] != cached[key]]
                    text_mismatch = [key for key in TEXT_FIELDS if serialized[key] != cached[key]]
                    for key in mismatch:
                        differences.append({"warc_id": row["warc_id"], "outlet": row["outlet"], "field": key,
                                            "candidate": serialized[key], "cached": cached[key]})
                    results.append({"warc_id": row["warc_id"], "outlet": row["outlet"],
                                    "accepted": bool(extracted.get("title") or extracted.get("content")),
                                    "exact_match_with_timestamp_shift": not mismatch, "text_match": not text_mismatch,
                                    "mismatched_fields": ";".join(mismatch)})
        finally:
            parquet.close()
    if len(results) != len(saved):
        raise ValueError("Not every saved sample row was recovered")
    output = audit_dir / "parser_candidates" / parser_path.stem
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison.csv", ["warc_id", "outlet", "accepted", "exact_match_with_timestamp_shift", "text_match", "mismatched_fields"], results)
    write_csv(output / "differences.csv", ["warc_id", "outlet", "field", "candidate", "cached"], differences)
    report = {"parser_file": str(parser_path), "parser_sha256": hashed(parser_path.read_bytes()),
              "sampled_articles": len(results), "accepted_articles": sum(row["accepted"] for row in results),
              "text_matches": sum(row["text_match"] for row in results),
              "exact_matches_with_timestamp_shift": sum(row["exact_match_with_timestamp_shift"] for row in results),
              "timestamp_comparison_shift_hours": 8, "differing_fields": dict(Counter(row["field"] for row in differences)),
              "remote_requery": False, "reference_fingerprints": str(audit_dir / "remote_fingerprints.csv"),
              "reference_fingerprints_sha256": hashed((audit_dir / "remote_fingerprints.csv").read_bytes())}
    atomic_text(output / "summary.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return report


def inspect_five_text_examples(args):
    """Targeted evidence recovery, only after the fingerprint audit found differences."""
    if args.table != "articles":
        raise ValueError("Text inspection requires the indexed base articles table")
    output = args.output_dir.resolve()
    with (output / "local_sample.csv").open() as stream:
        local = {row["warc_id"]: row for row in csv.DictReader(stream)}
    with (output / "remote_fingerprints.csv").open() as stream:
        cached = list(csv.DictReader(stream))
    selected = []
    for outlet, count in [("Liberty Times (自由時報)", 2), ("Central News Agency (中央社)", 1),
                          ("Now News (今日新聞)", 1), ("Taiwan News", 1)]:
        candidates = [row for row in cached if row["outlet"] == outlet and any(
            row[field] != local[row["warc_id"]][field] for field in ("title_sha256", "content_sha256"))]
        selected.extend(sorted(candidates, key=lambda row: (int(row["content_char_length"]), row["warc_id"]))[:count])
    if len(selected) != 5:
        raise ValueError("The expected five bounded mismatch examples are unavailable")
    if sum(4 * (int(row["title_char_length"]) + int(row["content_char_length"])) for row in selected) > 100_000:
        raise ValueError("Conservative UTF-8 upper bound exceeds the100KB transfer limit")
    sql = " UNION ALL ".join(["SELECT warc_id,title,content FROM articles WHERE warc_id = ?"] * len(selected))
    config = {"database": args.database, "sql": sql, "parameters": [row["warc_id"] for row in selected],
              "table": "articles", "timeout_seconds": args.timeout_seconds, "text_byte_limit": 100_000}
    code = "CONFIG_JSON = " + repr(json.dumps(config)) + "\n" + REMOTE_CODE
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host,
                               shlex.quote(args.remote_python) + " -"], input=code, text=True,
                              capture_output=True, check=True, timeout=args.timeout_seconds + 45)
    report = json.loads(response.stdout)
    records = report.pop("records")
    report.update({"sql": sql, "sql_sha256": hashed(sql.encode()), "text_byte_limit": 100_000,
                   "read_only": True, "memory_limit": "512MB", "threads": 2,
                   "timeout_seconds": args.timeout_seconds, "fingerprint_summary": str(output / "summary.json")})
    if report["status"] == "complete":
        remote = {row["warc_id"]: row for row in records}
        examples = []
        for index, expected in enumerate(selected, 1):
            identifier = expected["warc_id"]
            observed = remote[identifier]
            for field in ("title", "content"):
                if hashed(observed[field].encode()) != expected[field + "_sha256"]:
                    raise ValueError("Retrieved text differs from the previously captured fingerprint")
            row_metadata = local[identifier]
            shard = args.benchmark_dir / "shards" / row_metadata["source_path"]
            rows = pq.read_table(shard, columns=["warc_id", "source_row", "title", "content"]).to_pylist()
            parsed = next(row for row in rows if row["warc_id"] == identifier and row["source_row"] == int(row_metadata["source_row"]))
            example = {"case": index, "warc_id": identifier, "outlet": expected["outlet"],
                       "url": expected["url"], "source_path": row_metadata["source_path"],
                       "local_title": parsed["title"], "remote_title": observed["title"],
                       "local_content": parsed["content"], "remote_content": observed["content"],
                       "title_equal": parsed["title"] == observed["title"],
                       "content_equal": parsed["content"] == observed["content"],
                       "remote_content_is_substring_of_local": observed["content"] in (parsed["content"] or ""),
                       "remote_title_starts_with_local": observed["title"].startswith(parsed["title"] or "")}
            examples.append(example)
        report["examples"] = examples
    atomic_text(output / "text_examples.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "text_bytes": report.get("text_bytes"),
                      "elapsed_seconds": report.get("elapsed_seconds"), "examples": len(report.get("examples", []))}))
    return report


def run(args):
    samples, local_provenance = choose_sample(args.benchmark_dir.resolve(), args.per_year)
    if not samples or len(samples) > 300:
        raise ValueError("Expected1..300 sampled IDs")
    independence = {}
    if args.assert_disjoint_benchmark:
        prior_files = json.loads((args.assert_disjoint_benchmark / "manifest.json").read_text())["selected_files"]
        current_files = json.loads((args.benchmark_dir / "manifest.json").read_text())["selected_files"]
        overlap = {row["path"] for row in prior_files} & {row["path"] for row in current_files}
        if overlap:
            raise ValueError(f"Heldout benchmark overlaps prior source files: {sorted(overlap)}")
        independence.update({"prior_benchmark": str(args.assert_disjoint_benchmark.resolve()), "overlapping_source_files": 0})
    if args.assert_disjoint_sample:
        with args.assert_disjoint_sample.open() as stream:
            prior_ids = {row["warc_id"] for row in csv.DictReader(stream)}
        overlap = prior_ids & {row["warc_id"] for row in samples}
        if overlap:
            raise ValueError(f"Heldout sample overlaps prior warc_ids: {sorted(overlap)}")
        independence.update({"prior_sample": str(args.assert_disjoint_sample.resolve()), "overlapping_warc_ids": 0})
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "local_sample.csv", META_FIELDS + TEXT_FIELDS + ["year", "source_path", "source_row"], samples)
    identifiers = [row["warc_id"] for row in samples]
    sql, parameters, prior = SQL, [identifiers], None
    if args.table == "articles":
        prior = json.loads(args.prior_audit.read_text())
        if prior["status"] != "timeout" or prior.get("database") != args.database:
            raise ValueError("Indexed fallback requires a recorded timeout against the same database")
        if args.prior_audit.resolve().parent == output:
            raise ValueError("Fallback output must be separate to preserve the original timeout evidence")
        # Explicit equalities make each branch eligible for the warc_id ART index.
        projection = SQL.split("FROM gpr_binary_matrix_all_words")[0]
        point_query = projection + "FROM articles WHERE warc_id = ?"
        sql = " UNION ALL ".join([point_query] * len(identifiers))
        parameters = identifiers
    config = {"database": args.database, "sql": sql, "parameters": parameters, "table": args.table,
              "timeout_seconds": args.timeout_seconds}
    code = "CONFIG_JSON = " + repr(json.dumps(config)) + "\n" + REMOTE_CODE
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host, shlex.quote(args.remote_python) + " -"]
    try:
        response = subprocess.run(command, input=code, text=True, capture_output=True, check=True,
                                  timeout=args.timeout_seconds + 45)
        remote = json.loads(response.stdout)
    except subprocess.TimeoutExpired:
        remote = {"status": "timeout", "error": "SSH process exceeded bounded query+connection deadline", "records": []}
    records = remote.pop("records")
    summary = {**remote, "requested_articles": len(samples), "returned_records": len(records),
               "local_provenance": local_provenance, "host": args.host, "database": args.database,
               "table": args.table, "sql": sql, "sql_sha256": hashed(sql.encode()), "warc_ids_sha256": hashed(json.dumps(identifiers).encode()),
               "script_sha256": hashed(Path(__file__).read_bytes()), "read_only": True,
               "remote_memory_limit": "512MB", "remote_threads": 2, "query_timeout_seconds": args.timeout_seconds,
               "sampling": "Up to10 unique accepted articles per crawl year by default; round-robin across outlets ordered by SHA256; within outlet ordered by SHA256(warc_id). First local source_path/source_row wins duplicate IDs.",
               "scope_limit": "Benchmark sample verifies individual parser outputs; it does not prove equality of the entire corpus.",
               "null_token": "__NULL__", "text_transferred": False}
    summary["independence_checks"] = independence
    if prior is not None:
        summary["source_change"] = {"from_table": "gpr_binary_matrix_all_words", "to_table": "articles",
                                    "reason": "Binary-table fingerprint scan timed out; base articles has PRIMARY KEY(warc_id), matching metadata schema and corpus row count.",
                                    "prior_summary": str(args.prior_audit.resolve()),
                                    "prior_summary_sha256": hashed(args.prior_audit.read_bytes()),
                                    "prior_status": prior["status"], "prior_elapsed_seconds": prior.get("elapsed_seconds"),
                                    "limitation": "This fallback compares rebuilt text with the indexed base articles table; it does not independently establish binary-cache text equality."}
    if remote["status"] == "complete":
        by_id = defaultdict(list)
        for row in records:
            by_id[row["warc_id"]].append(row)
        comparisons, differences = [], []
        for local in samples:
            candidates = by_id[local["warc_id"]]
            mismatched = []
            if len(candidates) == 1:
                for field in COMPARE_FIELDS:
                    if local[field] != candidates[0][field]:
                        mismatched.append(field)
                        differences.append({"warc_id": local["warc_id"], "field": field,
                                            "local": json.dumps(local[field], ensure_ascii=False),
                                            "remote": json.dumps(candidates[0][field], ensure_ascii=False)})
            comparisons.append({"warc_id": local["warc_id"], "year": local["year"], "outlet": local["outlet"],
                                "remote_count": len(candidates), "exact_match": len(candidates) == 1 and not mismatched,
                                "mismatched_fields": ";".join(mismatched)})
        write_csv(output / "remote_fingerprints.csv", META_FIELDS + TEXT_FIELDS, records)
        write_csv(output / "comparison.csv", ["warc_id", "year", "outlet", "remote_count", "exact_match", "mismatched_fields"], comparisons)
        write_csv(output / "differences.csv", ["warc_id", "field", "local", "remote"], differences)
        summary.update({"exact_matches": sum(row["exact_match"] for row in comparisons),
                        "missing_remote_articles": sum(row["remote_count"] == 0 for row in comparisons),
                        "duplicate_remote_ids": sum(row["remote_count"] > 1 for row in comparisons),
                        "articles_with_field_differences": sum(bool(row["mismatched_fields"]) for row in comparisons),
                        "differing_fields": dict(Counter(row["field"] for row in differences))})
        summary["diagnostics"] = comparison_diagnostics(samples, records)
    atomic_text(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key in (
        "status", "requested_articles", "returned_records", "exact_matches", "missing_remote_articles",
        "duplicate_remote_ids", "articles_with_field_differences", "differing_fields", "error")}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.reparse_saved_sample:
        reparse_saved_sample(arguments)
    elif arguments.inspect_five_text_examples:
        inspect_five_text_examples(arguments)
    else:
        run(arguments)
