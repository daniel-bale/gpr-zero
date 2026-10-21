# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow==20.0.0"]
# ///
"""Audit every accepted article in the three fixed WARC-validation files.

Only bounded indexed point queries run remotely. Full title/content equality is
checked with SHA256, length, and null/empty flags; article text is not transferred.
This validates rebuilt Stage2 against the original indexed articles table. The
separate WARC extraction audit must establish WARC -> Stage1 -> rebuilt Stage2.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

import pyarrow.parquet as pq


FIXED_SOURCE_COUNTS = {
    "2018/06/CC-NEWS-20180625013240-00733.warc.parquet": 704,
    "2020/06/CC-NEWS-20200607060036-00418.warc.parquet": 305,
    "2024/09/CC-NEWS-20240911101042-06028.warc.parquet": 224,
}


PROJECTION = """SELECT warc_id,warc_date,url,domain,outlet,is_english,extracted_publish_date,
source_warc_file,stage2_processed_at,
title IS NULL AS title_is_null,title = '' AS title_is_empty,
length(title) AS title_char_length,sha256(title) AS title_sha256,
content IS NULL AS content_is_null,content = '' AS content_is_empty,
length(content) AS content_char_length,sha256(content) AS content_sha256
FROM articles WHERE warc_id = ?"""

REMOTE_CODE = r'''
import datetime, json, os, threading, time
import duckdb
config = json.loads(CONFIG_JSON)
if not 1 <= len(config["ids"]) <= 100 or len(set(config["ids"])) != len(config["ids"]):
    raise ValueError("Invalid bounded ID batch")
stat = os.stat(config["database"])
source = dict(path=config["database"], size=stat.st_size, mtime_ns=stat.st_mtime_ns)
if source != config["expected_source"]:
    raise RuntimeError("Database differs from prior successful held-out audit source")
connection = duckdb.connect(config["database"], read_only=True)
connection.execute("SET memory_limit='512MB'")
connection.execute("SET threads=2")
constraints = connection.execute("SELECT constraint_type,constraint_column_names FROM duckdb_constraints() WHERE table_name='articles'").fetchall()
if not any(kind == "PRIMARY KEY" and columns == ["warc_id"] for kind, columns in constraints):
    raise RuntimeError("Indexed articles PRIMARY KEY(warc_id) required")
timeout = threading.Event()
def interrupt():
    timeout.set()
    connection.interrupt()
timer = threading.Timer(config["timeout_seconds"], interrupt)
timer.daemon = True
started = time.monotonic()
timer.start()
try:
    result = connection.execute(config["sql"], config["ids"])
    columns = [item[0] for item in result.description]
    records = [dict(zip(columns, row)) for row in result.fetchall()]
    after = os.stat(config["database"])
    if after.st_size != source["size"] or after.st_mtime_ns != source["mtime_ns"]:
        raise RuntimeError("Database changed during read-only query")
    report = dict(status="complete", records=records)
except Exception as error:
    report = dict(status="timeout" if timeout.is_set() else "failed", error=str(error), records=[])
finally:
    timer.cancel()
    connection.close()
report.update(source=source, elapsed_seconds=time.monotonic()-started,
              duckdb_version=duckdb.__version__, primary_key_verified=True,
              captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
print(json.dumps(report, ensure_ascii=False, default=lambda value: value.isoformat()))
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_helper(root: Path):
    path = root / "scripts/audit_rebuilt_article_sample.py"
    spec = importlib.util.spec_from_file_location("prior_bounded_article_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def collect_rows(root: Path, selection_path: Path, full: Path, helper):
    selection = json.loads(selection_path.read_text())
    records = selection["records"]
    if len(records) != 3 or len({row["relative_path"] for row in records}) != 3:
        raise ValueError("Audit requires exactly the three fixed distinct selection paths")
    if {row["relative_path"]: row["rows"] for row in records} != FIXED_SOURCE_COUNTS:
        raise ValueError("Selection must contain the fixed 704 + 305 + 224 source rows")
    manifest_path, progress_path = full / "manifest.json", full / "progress.json"
    manifest, progress = json.loads(manifest_path.read_text()), json.loads(progress_path.read_text())
    if manifest["mode"] != "full" or progress["status"] != "complete":
        raise ValueError("Full Stage2 rebuild is not complete")
    parser_path = Path(manifest["dependencies"]["parser_file"])
    if sha256(parser_path) != manifest["dependencies"]["parser_sha256"]:
        raise ValueError("Frozen parser no longer matches the full-rebuild manifest")
    rows, provenance = [], []
    for selected in records:
        relative = selected["relative_path"]
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Selection path must be relative to the fixed Stage1 root")
        source = Path(manifest["input_dir"]) / relative
        if source.resolve() != Path(selected["source_path"]).resolve():
            raise ValueError("Selection source path and full-rebuild source disagree")
        shard = full / "shards" / relative
        checkpoint_path = full / "checkpoints" / (relative + ".json")
        checkpoint = json.loads(checkpoint_path.read_text())
        if sha256(source) != checkpoint["source_sha256"] or sha256(shard) != checkpoint["shard_sha256"]:
            raise ValueError(f"Source/shard hash differs from completed checkpoint: {relative}")
        if shard.stat().st_size != checkpoint["shard_size"]:
            raise ValueError("Shard size differs from checkpoint")
        source_ids = pq.read_table(source, columns=["warc_id"]).column("warc_id").to_pylist()
        if len(source_ids) != selected["rows"] or len(source_ids) != checkpoint["counts"]["source_rows"]:
            raise ValueError("Stage1 source row count differs from fixed selection/checkpoint")
        article_rows = pq.read_table(shard, columns=helper.META_FIELDS + ["title", "content", "source_path", "source_row"]).to_pylist()
        if len(article_rows) != checkpoint["counts"]["accepted_articles"]:
            raise ValueError("Not every accepted rebuilt article was loaded")
        if len(article_rows) != len(source_ids):
            raise ValueError("Fixed WARC sample requires every Stage1 row accepted; rejected rows cannot be omitted")
        positions = set()
        for row in article_rows:
            position = row["source_row"]
            if row["source_path"] != relative or not isinstance(position, int) or not 0 <= position < len(source_ids):
                raise ValueError("Unexpected source_path/source_row in rebuilt shard")
            if position in positions or source_ids[position] != row["warc_id"]:
                raise ValueError("Duplicate/misaligned source row or warc_id")
            positions.add(position)
            timestamp = row["warc_date"]
            if timestamp is not None and timestamp.tzinfo:
                timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
            local = helper.fingerprints({**row, "warc_date": timestamp})
            local.update(source_path=relative, source_row=position,
                         warc_date_taipei=(timestamp + timedelta(hours=8)).isoformat() if timestamp else None)
            rows.append(local)
        provenance.append({"source_path": relative, "source_rows": len(source_ids),
                           "accepted_rows": len(article_rows), "rejected_rows": len(source_ids)-len(article_rows),
                           "rejected_source_rows": sorted(set(range(len(source_ids))) - positions),
                           "source_sha256": checkpoint["source_sha256"], "shard_sha256": checkpoint["shard_sha256"],
                           "checkpoint_sha256": sha256(checkpoint_path),
                           "outlet_counts": dict(Counter(row["outlet"] for row in article_rows))})
    ids = [row["warc_id"] for row in rows]
    if sum(item["source_rows"] for item in provenance) != len(rows) or len(rows) != 1233:
        raise ValueError("Fixed WARC sample requires source_rows == accepted_rows == 1233")
    if not rows or len(rows) > 2000 or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("Expected 1..2000 accepted rows with distinct nonempty warc_ids")
    prior_paths = [root / "replication/raw_rebuild/article_sample_audit/base_articles/local_sample.csv",
                   root / "replication/raw_rebuild/article_sample_audit/cache_parser_holdout/local_sample.csv"]
    prior_ids, prior_files = set(), set()
    for path in prior_paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                prior_ids.add(row["warc_id"])
                prior_files.add(row["source_path"])
    if set(ids) & prior_ids or {row["source_path"] for row in rows} & prior_files:
        raise ValueError("Selected audit overlaps prior successful sampled articles/source files")
    return rows, {"selection_sha256": sha256(selection_path), "full_manifest_sha256": sha256(manifest_path),
                  "full_progress_sha256": sha256(progress_path), "parser_file": str(parser_path),
                  "parser_sha256": sha256(parser_path), "files": provenance,
                  "total_source_rows": sum(item["source_rows"] for item in provenance),
                  "total_accepted_rows": len(rows), "total_rejected_rows": 0,
                  "fixed_sample_all_source_rows_accepted": True,
                  "prior_unique_ids": len(prior_ids),
                  "prior_sample_files": [{"path": str(path), "sha256": sha256(path)} for path in prior_paths],
                  "overlapping_prior_ids": 0, "overlapping_prior_source_files": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--selection", type=Path, default=root / "replication/warc_validation/selection.json")
    parser.add_argument("--full-rebuild", type=Path, default=root / "replication/raw_rebuild/news/full")
    parser.add_argument("--output", type=Path, default=root / "replication/warc_validation/cache_comparison")
    parser.add_argument("--host", default="bdsrc")
    parser.add_argument("--remote-python", default="/home5/chuenho/gpr-zero/.venv/bin/python")
    parser.add_argument("--reuse-remote", action="store_true", help="Reuse exact saved ID batches after checking their provenance")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "remote_batches").mkdir(exist_ok=True)
    helper, helper_path = load_helper(root)
    local_rows, provenance = collect_rows(root, args.selection.resolve(), args.full_rebuild.resolve(), helper)
    fields = helper.META_FIELDS + helper.TEXT_FIELDS + ["source_path", "source_row", "warc_date_taipei"]
    helper.write_csv(output / "local_fingerprints.csv", fields, local_rows)
    prior_path = root / "replication/raw_rebuild/article_sample_audit/cache_parser_holdout/summary.json"
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "complete" or prior["table"] != "articles" or prior["host"] != args.host:
        raise ValueError("Expected successful prior held-out indexed-articles audit on the same host")
    expected_source = prior["source"]
    if prior["database"] != expected_source["path"]:
        raise ValueError("Prior database provenance is inconsistent")
    summary = {"status": "running", "scope": __doc__, "local_provenance": provenance,
               "host": args.host, "database": expected_source["path"], "table": "articles",
               "reference_source": expected_source, "prior_successful_audit": str(prior_path),
               "prior_successful_audit_sha256": sha256(prior_path), "helper_sha256": sha256(helper_path),
               "script_sha256": sha256(Path(__file__)), "selection": str(args.selection.resolve()),
               "requested_articles": len(local_rows), "remote_read_only": True,
               "remote_limits": {"ids_per_batch": 100, "memory": "512MB", "threads": 2, "query_timeout_seconds": 60},
               "text_transferred": False, "timestamp_comparison_shift_hours": 8,
               "source_filename_rule": "Historical loader: Path(source_path).name.removesuffix('.parquet') + '.warc.gz'; this preserves the original doubled '.warc.warc.gz' spelling.",
               "source_filename_rule_reference": str(root / "replication/reference_code/stage2_parse_html_load_db.2025-05.py"),
               "source_filename_rule_reference_sha256": sha256(root / "replication/reference_code/stage2_parse_html_load_db.2025-05.py"),
               "remote_requery": not args.reuse_remote,
               "uncompared_metadata": ["stage2_processed_at (historical processing provenance; no reproducible local equivalent)"]}
    helper.atomic_text(output / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    all_remote, batch_provenance = [], []
    for number, start in enumerate(range(0, len(local_rows), 100)):
        ids = [row["warc_id"] for row in local_rows[start:start+100]]
        ids_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        sql = " UNION ALL ".join([PROJECTION] * len(ids))
        sql_hash = hashlib.sha256(sql.encode()).hexdigest()
        batch_path = output / "remote_batches" / f"batch_{number:03}.json"
        if args.reuse_remote:
            remote = json.loads(batch_path.read_text())
            if remote["requested_ids_sha256"] != ids_hash or remote["sql_sha256"] != sql_hash or remote["source"] != expected_source:
                raise ValueError("Saved remote batch provenance differs from this selected audit")
        else:
            config = {"database": expected_source["path"], "expected_source": expected_source,
                      "ids": ids, "sql": sql, "timeout_seconds": 60}
            code = "CONFIG_JSON = " + repr(json.dumps(config)) + "\n" + REMOTE_CODE
            response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host,
                                       shlex.quote(args.remote_python) + " -"], input=code, text=True,
                                      capture_output=True, timeout=105, check=True)
            remote = json.loads(response.stdout)
            remote.update(requested_ids=ids, requested_ids_sha256=ids_hash, sql_sha256=sql_hash)
            helper.atomic_text(batch_path, json.dumps(remote, indent=2, ensure_ascii=False) + "\n")
        if remote["status"] != "complete":
            summary.update(status=remote["status"], error=remote.get("error"), failed_batch=number)
            helper.atomic_text(output / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            raise RuntimeError("Bounded reference query failed; no broader fallback attempted")
        returned = [row["warc_id"] for row in remote["records"]]
        if set(returned) - set(ids) or len(set(returned)) != len(returned):
            raise ValueError("Reference query returned unexpected or duplicate IDs")
        all_remote.extend(remote["records"])
        batch_provenance.append({"file": str(batch_path), "sha256": sha256(batch_path),
                                 "requested_ids_sha256": ids_hash, "requested_rows": len(ids), "returned_rows": len(returned),
                                 "query_elapsed_seconds": remote["elapsed_seconds"], "primary_key_verified": remote["primary_key_verified"]})
        print(f"Reference batch {number+1}: {len(returned)}/{len(ids)} rows, {remote['elapsed_seconds']:.2f}s", flush=True)
    by_id = {row["warc_id"]: row for row in all_remote}
    if len(by_id) != len(all_remote):
        raise ValueError("Duplicate remote IDs across bounded batches")
    comparisons, differences = [], []
    for local in local_rows:
        remote = by_id.get(local["warc_id"])
        raw_bad, shifted_bad = [], []
        if remote:
            for field in helper.COMPARE_FIELDS:
                if local[field] != remote[field]:
                    raw_bad.append(field)
                observed = local["warc_date_taipei"] if field == "warc_date" else local[field]
                if observed != remote[field]:
                    shifted_bad.append(field)
                    differences.append({"warc_id": local["warc_id"], "source_path": local["source_path"],
                                        "source_row": local["source_row"], "field": field,
                                        "rebuilt": json.dumps(observed, ensure_ascii=False), "cached": json.dumps(remote[field], ensure_ascii=False)})
        cached_file = remote.get("source_warc_file") if remote else None
        expected_cached_file = Path(local["source_path"]).name.removesuffix(".parquet") + ".warc.gz"
        source_file_matches = cached_file == expected_cached_file
        comparisons.append({"warc_id": local["warc_id"], "source_path": local["source_path"], "source_row": local["source_row"],
                            "outlet": local["outlet"], "cached_present": remote is not None,
                            "text_matches": remote is not None and not any(field in shifted_bad for field in helper.TEXT_FIELDS),
                            "exact_match_utc": remote is not None and not raw_bad,
                            "exact_match_taipei": remote is not None and not shifted_bad,
                            "cached_source_warc_file": cached_file, "expected_cached_source_warc_file": expected_cached_file,
                            "source_filename_matches": source_file_matches,
                            "raw_mismatched_fields": ";".join(raw_bad), "shifted_mismatched_fields": ";".join(shifted_bad)})
    helper.write_csv(output / "remote_fingerprints.csv", helper.META_FIELDS + helper.TEXT_FIELDS + ["source_warc_file", "stage2_processed_at"], all_remote)
    helper.write_csv(output / "comparison.csv", list(comparisons[0]), comparisons)
    helper.write_csv(output / "differences.csv", ["warc_id", "source_path", "source_row", "field", "rebuilt", "cached"], differences)
    per_file = {}
    for file in provenance["files"]:
        rows = [row for row in comparisons if row["source_path"] == file["source_path"]]
        per_file[file["source_path"]] = {"source_rows": file["source_rows"], "accepted_articles": len(rows),
                                        **{key: sum(row[key] for row in rows) for key in ["cached_present", "text_matches", "exact_match_utc", "exact_match_taipei", "source_filename_matches"]}}
    summary.update(status="complete", returned_articles=len(all_remote), remote_batches=batch_provenance,
                   exact_text_matches=sum(row["text_matches"] for row in comparisons),
                   exact_matches_utc=sum(row["exact_match_utc"] for row in comparisons),
                   exact_matches_taipei=sum(row["exact_match_taipei"] for row in comparisons),
                   source_filename_matches=sum(row["source_filename_matches"] for row in comparisons),
                   missing_reference_articles=sum(not row["cached_present"] for row in comparisons),
                   per_file=per_file, differing_fields=dict(Counter(row["field"] for row in differences)),
                   timestamp_diagnostics=helper.comparison_diagnostics(local_rows, all_remote),
                   source_database_matches_prior_audit=True,
                   fixed_sample_completion_check={"source_rows": provenance["total_source_rows"],
                                                  "accepted_rows": len(local_rows), "expected_rows": 1233,
                                                  "rejected_rows": provenance["total_rejected_rows"],
                                                  "all_source_rows_accepted": provenance["fixed_sample_all_source_rows_accepted"]},
                   all_article_text_and_metadata_match=all(row["exact_match_taipei"] and row["source_filename_matches"] for row in comparisons))
    helper.atomic_text(output / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in ["status", "requested_articles", "returned_articles", "exact_text_matches", "exact_matches_utc", "exact_matches_taipei", "source_filename_matches", "missing_reference_articles", "all_article_text_and_metadata_match"]}, indent=2))
    return 0 if summary["all_article_text_and_metadata_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
