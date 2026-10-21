#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Export small, read-only remote cached-corpus aggregates over SSH.

No raw articles, HTML, WARC files, or full databases are transferred. Remote
DuckDB queries use 512 MB memory, two threads, and a per-query interrupt timer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shlex
import subprocess
from pathlib import Path


TABLE = "gpr_binary_matrix_all_words"
NULL = "__NULL__"
QUERIES = {
    "day_outlet": "SELECT CAST(warc_date AS DATE) AS day, outlet, count(*) AS total_articles FROM gpr_binary_matrix_all_words GROUP BY 1,2 ORDER BY 1 NULLS LAST,2 NULLS LAST",
    "week_outlet": "SELECT CAST(date_trunc('week',warc_date) AS DATE) AS week, outlet, count(*) AS total_articles FROM gpr_binary_matrix_all_words GROUP BY 1,2 ORDER BY 1 NULLS LAST,2 NULLS LAST",
    "year_outlet": "SELECT CAST(EXTRACT(year FROM warc_date) AS INTEGER) AS year, outlet, count(*) AS total_articles FROM gpr_binary_matrix_all_words GROUP BY 1,2 ORDER BY 1 NULLS LAST,2 NULLS LAST",
    "outlet": "SELECT outlet, count(*) AS total_articles FROM gpr_binary_matrix_all_words GROUP BY 1 ORDER BY 1 NULLS LAST",
    "totals": "SELECT count(*) AS total_articles, count(*) FILTER (WHERE warc_date IS NULL) AS null_warc_date, count(*) FILTER (WHERE outlet IS NULL) AS null_outlet, count(*) FILTER (WHERE warc_date IS NULL AND outlet IS NULL) AS both_null, count(*) FILTER (WHERE outlet = '__NULL__') AS reserved_null_token_collision FROM gpr_binary_matrix_all_words",
}

REMOTE_CODE = r'''
import csv, datetime, hashlib, io, json, os, sys, threading, time
import duckdb
config = json.loads(CONFIG_JSON)
def metadata():
    stat = os.stat(config["database"])
    return {"path": config["database"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
before = metadata()
connection = duckdb.connect(config["database"], read_only=True)
connection.execute("SET memory_limit='512MB'")
connection.execute("SET threads=2")
def query(sql):
    started = time.monotonic()
    timer = threading.Timer(config["timeout_seconds"], connection.interrupt)
    timer.daemon = True
    timer.start()
    try:
        result = connection.execute(sql)
        fields = [item[0] for item in result.description]
        rows = result.fetchall()
        return fields, rows, time.monotonic() - started
    finally:
        timer.cancel()
try:
    exported = {}
    provenance = {}
    totals = None
    for name, sql in config["queries"].items():
        fields, rows, elapsed = query(sql)
        provenance[name] = {"sql": sql, "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
                            "elapsed_seconds": elapsed, "result_rows": len(rows)}
        if name == "totals":
            totals = dict(zip(fields, rows[0]))
        else:
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows([[config["null_token"] if value is None else value for value in row] for row in rows])
            exported[name] = stream.getvalue()
    after = metadata()
    if before != after:
        raise RuntimeError("Remote database metadata changed while aggregating; export rejected")
    if totals["reserved_null_token_collision"]:
        raise RuntimeError("Actual outlet name conflicts with reserved null token")
    print(json.dumps({"csv": exported, "totals": totals, "source": before,
                      "query_provenance": provenance, "duckdb_version": duckdb.__version__,
                      "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, ensure_ascii=False))
finally:
    connection.close()
'''


def write_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="")
    temporary.replace(path)


def parse_arguments(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="bdsrc")
    parser.add_argument("--remote-python", default="/home5/chuenho/gpr-zero/.venv/bin/python")
    parser.add_argument("--database", default="/home5/chuenho/data/news/stage2_enhanced_full.duckdb")
    parser.add_argument("--output-dir", type=Path, default=root / "replication/raw_rebuild/reference_counts")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def run(args):
    config = {"database": args.database, "queries": QUERIES, "null_token": NULL, "timeout_seconds": args.timeout_seconds}
    code = "CONFIG_JSON = " + repr(json.dumps(config)) + "\n" + REMOTE_CODE
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host,
               shlex.quote(args.remote_python) + " -"]
    completed = subprocess.run(command, input=code, text=True, capture_output=True,
                               check=True, timeout=(len(QUERIES) * args.timeout_seconds) + 45)
    result = json.loads(completed.stdout)
    totals = result["totals"]
    validated = {}
    for name, content in result.pop("csv").items():
        rows = list(csv.DictReader(io.StringIO(content)))
        total = sum(int(row["total_articles"]) for row in rows)
        null_outlet = sum(int(row["total_articles"]) for row in rows if row["outlet"] == NULL)
        if total != totals["total_articles"] or null_outlet != totals["null_outlet"]:
            raise ValueError(f"Aggregate total or null-outlet count failed validation: {name}")
        if name != "outlet":
            date_key = name.split("_")[0]
            null_date = sum(int(row["total_articles"]) for row in rows if row[date_key] == NULL)
            both_null = sum(int(row["total_articles"]) for row in rows if row[date_key] == NULL and row["outlet"] == NULL)
            if null_date != totals["null_warc_date"] or both_null != totals["both_null"]:
                raise ValueError(f"Null-date counts failed validation: {name}")
        validated[f"article_counts_{name}.csv"] = content
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename, content in validated.items():
        write_atomic(output / filename, content)
        files[filename] = {"sha256": hashlib.sha256(content.encode()).hexdigest(), "bytes": len(content.encode())}
    result.update({"host": args.host, "remote_python": args.remote_python, "table": TABLE,
                   "read_only": True, "remote_memory_limit": "512MB", "remote_threads": 2,
                   "query_timeout_seconds": args.timeout_seconds, "null_token": NULL,
                   "files": files, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "scope": "Entire cached binary-matrix corpus; no year, outlet, language, or content filter; no additional deduplication.",
                   "date_semantics": "Source warc_date; DATE cast for day, nonoverlapping Monday week, calendar year; explicit null groups preserved.",
                   "verification": "Each CSV total, null-outlet count, and applicable null-date/both-null counts equal an independent SQL aggregate."})
    write_atomic(output / "provenance.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "total_articles": totals["total_articles"],
                      "null_warc_date": totals["null_warc_date"], "null_outlet": totals["null_outlet"],
                      "csv_bytes": sum(value["bytes"] for value in files.values())}))
    return result


if __name__ == "__main__":
    run(parse_arguments())
