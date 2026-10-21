#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "duckdb==1.2.1", "polars==1.30.0", "selectolax==0.3.29",
#   "tqdm==4.67.1", "pyarrow==20.0.0",
# ]
# ///
"""Reconstruct the older thesis dictionary baseline from read-only stage-1 HTML.

The default is a deterministic, bounded pilot, never a full corpus replication.
Use --full explicitly for an independently checkpointed full run. No LLM calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_parse_html_load_db as historical_parser


START = datetime(2017, 1, 1)
END = datetime(2025, 1, 1)
THREE_OUTLETS = (
    "China Times (中國時報)",
    "Liberty Times (自由時報)",
    "United Daily (聯合報)",
)
SOURCE_COLUMNS = (
    "warc_id", "warc_date", "url", "domain", "outlet", "is_english",
    "html_content_bytes",
)
ARTICLE_SCHEMA = pa.schema([
    ("warc_id", pa.string()), ("warc_date", pa.timestamp("us")),
    ("url", pa.string()), ("domain", pa.string()), ("outlet", pa.string()),
    ("is_english", pa.bool_()), ("title", pa.string()), ("content", pa.string()),
    ("extracted_publish_date", pa.string()), ("source_file", pa.string()),
    ("is_gpr", pa.bool_()), ("matched_categories", pa.list_(pa.string())),
])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_dictionary(path: Path) -> tuple[dict, dict]:
    """Validate the deliberately small literal-OR / set-AND grammar."""
    document = json.loads(path.read_text())
    word_sets, categories = document.get("word_sets"), document.get("categories")
    if not isinstance(word_sets, dict) or not word_sets:
        raise ValueError("Dictionary must contain nonempty word_sets")
    for key, words in word_sets.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
            raise ValueError(f"Invalid word-set identifier: {key!r}")
        if not isinstance(words, list) or not words or any(
            not isinstance(word, str) or not word.strip() for word in words
        ):
            raise ValueError(f"Word set {key!r} must be a nonempty list of strings")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Dictionary must contain nonempty categories")
    parsed = {}
    for category, expression in categories.items():
        if not isinstance(category, str) or not category.strip() or not isinstance(expression, str):
            raise ValueError("Category names and expressions must be nonempty strings")
        operands = re.split(r"\s+AND\s+", expression.strip())
        if not operands or any(operand not in word_sets for operand in operands):
            raise ValueError(f"Unsupported category expression: {expression!r}")
        parsed[category] = operands
    return {key: tuple(word.lower() for word in words) for key, words in word_sets.items()}, parsed


def match_categories(content: str | None, word_sets: dict, categories: dict) -> list[str]:
    text = (content or "").lower()
    present = {key: any(word in text for word in words) for key, words in word_sets.items()}
    return [name for name, operands in categories.items() if all(present[key] for key in operands)]


def source_metadata(path: Path, input_dir: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.relative_to(input_dir)), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def discover_files(input_dir: Path) -> list[dict]:
    """Select year partitions 2017--2024; unpartitioned files are row-filtered."""
    files = []
    for path in sorted(input_dir.rglob("*.parquet")):
        relative = path.relative_to(input_dir)
        year = next((int(part) for part in relative.parts[:-1] if re.fullmatch(r"20\d{2}", part)), None)
        if year is None or 2017 <= year <= 2024:
            files.append({**source_metadata(path, input_dir), "partition_year": year})
    if not files:
        raise ValueError(f"No eligible Parquet files under {input_dir}")
    return files


def choose_files(files: list[dict], full: bool, maximum: int) -> list[dict]:
    if full:
        return files
    groups = defaultdict(list)
    for item in files:
        groups[item["partition_year"] or 0].append(item)
    budgets = {year: 0 for year in groups}
    for _ in range(min(maximum, len(files))):
        eligible = [year for year in sorted(groups) if budgets[year] < len(groups[year])]
        year = min(eligible, key=lambda key: (budgets[key], key))
        budgets[year] += 1
    selected = []
    for year, group in sorted(groups.items()):
        count = budgets[year]
        # Midpoints of equal file-count strata, deterministic and spread over the year.
        selected.extend(group[((2 * index + 1) * len(group)) // (2 * count)] for index in range(count))
    return sorted(selected, key=lambda item: item["path"])


def setup_database(path: Path):
    conn = duckdb.connect(str(path))
    conn.execute("""CREATE TABLE IF NOT EXISTS articles (
        warc_id VARCHAR PRIMARY KEY, warc_date TIMESTAMP, url VARCHAR, domain VARCHAR,
        outlet VARCHAR, is_english BOOLEAN, title VARCHAR, content VARCHAR,
        extracted_publish_date VARCHAR, source_file VARCHAR, is_gpr BOOLEAN,
        matched_categories VARCHAR[]
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS processed_files (
        path VARCHAR PRIMARY KEY, size BIGINT, mtime_ns BIGINT,
        counts_json VARCHAR, completed_at TIMESTAMP DEFAULT current_timestamp
    )""")
    return conn


def normalize_date(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time())
    if not isinstance(value, datetime):
        raise ValueError(f"Unsupported crawl date type: {type(value).__name__}")
    if value.tzinfo:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def insert_batch(conn, records: list[dict]) -> None:
    if not records:
        return
    conn.register("parsed_batch", pa.Table.from_pylist(records, schema=ARTICLE_SCHEMA))
    try:
        conn.execute("INSERT OR IGNORE INTO articles SELECT * FROM parsed_batch")
    finally:
        conn.unregister("parsed_batch")


def process_file(conn, input_dir: Path, item: dict, word_sets: dict, categories: dict,
                 outlets: str, batch_size: int, max_html_bytes: int) -> Counter:
    """Commit accepted articles and their file checkpoint in one transaction."""
    path = input_dir / item["path"]
    expected = {key: item[key] for key in ("path", "size", "mtime_ns")}
    if source_metadata(path, input_dir) != expected:
        raise ValueError(f"Source changed since inventory: {path}")
    parquet = pq.ParquetFile(path)
    missing = set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    counts = Counter()
    before = conn.execute("SELECT count(*) FROM articles").fetchone()[0]
    conn.execute("BEGIN TRANSACTION")
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(SOURCE_COLUMNS)):
            records = []
            for row in batch.to_pylist():
                counts["source_rows"] += 1
                if not all(row.get(field) for field in ("warc_id", "url", "domain")):
                    counts["missing_essential_fields"] += 1
                    continue
                timestamp = normalize_date(row["warc_date"])
                if timestamp is None:
                    counts["missing_crawl_date"] += 1
                    continue
                if not START <= timestamp < END:
                    counts["outside_date_range"] += 1
                    continue
                if outlets == "three" and row["outlet"] not in THREE_OUTLETS:
                    counts["outside_outlet_scope"] += 1
                    continue
                if row["html_content_bytes"] is None:
                    counts["missing_html"] += 1
                    continue
                if len(row["html_content_bytes"]) > max_html_bytes:
                    raise ValueError(f"HTML exceeds --max-html-bytes for {row['warc_id']}; raise the bound explicitly")
                parsed = historical_parser.extract_article_content_selectolax(
                    row["html_content_bytes"], row["url"], row["domain"]
                )
                if not (parsed.get("title") or parsed.get("content")):
                    counts["no_title_or_content"] += 1
                    continue
                matches = match_categories(parsed.get("content"), word_sets, categories)
                counts["accepted_candidates"] += 1
                records.append({
                    **{key: row[key] for key in ("warc_id", "url", "domain", "outlet", "is_english")},
                    "warc_date": timestamp, **parsed, "source_file": item["path"],
                    "is_gpr": bool(matches), "matched_categories": matches,
                })
            insert_batch(conn, records)
        if source_metadata(path, input_dir) != expected:
            raise ValueError(f"Source changed during parsing: {path}")
        counts["inserted_articles"] = conn.execute("SELECT count(*) FROM articles").fetchone()[0] - before
        counts["duplicate_accepted_ids"] = counts["accepted_candidates"] - counts["inserted_articles"]
        conn.execute("INSERT INTO processed_files (path, size, mtime_ns, counts_json) VALUES (?, ?, ?, ?)",
                     [item["path"], item["size"], item["mtime_ns"], json.dumps(counts)])
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return counts


def write_csv(path: Path, fields: list[str], rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_results(conn, output_dir: Path, manifest: dict, categories: dict, status: str) -> dict:
    completed = conn.execute("SELECT path, counts_json FROM processed_files ORDER BY path").fetchall()
    counts = Counter()
    for _, value in completed:
        counts.update(json.loads(value))
    totals = conn.execute("""SELECT count(*), count(*) FILTER (WHERE is_gpr),
        count(*) FILTER (WHERE content IS NULL OR content = ''),
        min(warc_date), max(warc_date) FROM articles""").fetchone()
    sample = manifest["mode"] == "pilot"
    fields = ["week_start", "total_articles", "matching_articles", "gpr_percent", "is_sample", "run_complete", "has_articles", "partial_boundary_week"]
    weeks = []
    week = START.date() - timedelta(days=START.weekday())
    while week < END.date():
        weeks.append(week)
        week += timedelta(days=7)
    weekly = conn.execute("""SELECT CAST(date_trunc('week', warc_date) AS DATE),
        count(*), count(*) FILTER (WHERE is_gpr) FROM articles GROUP BY 1""").fetchall()
    overall = {row[0]: row[1:] for row in weekly}

    def weekly_row(week, total, matching):
        return {"week_start": week.isoformat(), "total_articles": total,
                "matching_articles": matching, "gpr_percent": 100 * matching / total if total else math.nan,
                "is_sample": sample, "run_complete": status == "complete",
                "has_articles": bool(total),
                "partial_boundary_week": week < START.date() or week + timedelta(days=7) > END.date()}

    write_csv(output_dir / "weekly_overall.csv", fields,
              (weekly_row(week, *overall.get(week, (0, 0))) for week in weeks))
    outlets = list(THREE_OUTLETS) if manifest["outlets"] == "three" else [
        row[0] for row in conn.execute("SELECT DISTINCT outlet FROM articles ORDER BY outlet").fetchall()
    ]
    outlet_values = {(row[0], row[1]): row[2:] for row in conn.execute("""
        SELECT CAST(date_trunc('week', warc_date) AS DATE), outlet, count(*),
        count(*) FILTER (WHERE is_gpr) FROM articles GROUP BY 1, 2""").fetchall()}
    write_csv(output_dir / "weekly_outlet.csv", ["outlet", *fields],
              ({"outlet": outlet, **weekly_row(week, *outlet_values.get((week, outlet), (0, 0)))}
               for outlet in outlets for week in weeks))
    category_values = {(row[0], row[1]): row[2] for row in conn.execute("""
        SELECT week_start, category, count(*) FROM (
            SELECT CAST(date_trunc('week', warc_date) AS DATE) AS week_start,
                   unnest(matched_categories) AS category FROM articles
        ) GROUP BY 1, 2""").fetchall()}
    write_csv(output_dir / "weekly_category.csv", ["category", *fields],
              ({"category": category, **weekly_row(week, overall.get(week, (0, 0))[0],
                                                  category_values.get((week, category), 0))}
               for category in categories for week in weeks))
    contributions = Counter()
    for (_, category), number in category_values.items():
        contributions[category] += number
    write_csv(output_dir / "category_contributions.csv",
              ["category", "matching_articles", "overall_matching_articles", "contribution_percent", "is_sample"],
              ({"category": category, "matching_articles": contributions[category],
                "overall_matching_articles": totals[1],
                "contribution_percent": 100 * contributions[category] / totals[1] if totals[1] else math.nan,
                "is_sample": sample} for category in categories))
    summary = {
        "status": status, "mode": manifest["mode"], "is_sample": sample,
        "outlets": manifest["outlets"], "selected_files": len(manifest["selected_files"]),
        "completed_files": len(completed), "eligible_files": len(manifest["source_files"]),
        "counts": dict(sorted(counts.items())), "total_articles": totals[0],
        "matching_articles": totals[1], "title_only_articles": totals[2],
        "gpr_percent": 100 * totals[1] / totals[0] if totals[0] else None,
        "observed_crawl_start": totals[3].isoformat() if totals[3] else None,
        "observed_crawl_end": totals[4].isoformat() if totals[4] else None,
        "assumptions": manifest["assumptions"],
        "interpretation": "Pilot counts are sample diagnostics, not a reproduction of full-thesis estimates." if sample else "Full selected-input reconstruction; equivalence to the thesis still requires comparison against its original corpus and results.",
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def parse_arguments(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=root.parent / "thesis/data/news/stage1_parquet")
    parser.add_argument("--dictionary", type=Path, default=root / "data/dictionaries/taiwan_gpr_dict_president.json")
    parser.add_argument("--output-dir", type=Path, default=root / "replication/results")
    parser.add_argument("--outlets", choices=("three", "all"), default="three")
    parser.add_argument("--max-files", type=int, help="Pilot file cap; default 8, sampled across year partitions")
    parser.add_argument("--full", action="store_true", help="Explicitly process all eligible input files; resumable")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-html-bytes", type=int, default=32 * 1024 * 1024,
                        help="Fail rather than silently skip an oversized HTML record")
    args = parser.parse_args(argv)
    if args.full and args.max_files is not None:
        parser.error("--full and --max-files are mutually exclusive")
    if (args.max_files is not None and args.max_files <= 0) or args.batch_size <= 0 or args.max_html_bytes <= 0:
        parser.error("File, batch, and HTML bounds must be positive")
    return args


def run(args) -> dict:
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    dictionary = args.dictionary.resolve()
    word_sets, categories = load_dictionary(dictionary)
    files = discover_files(input_dir)
    selected = choose_files(files, args.full, args.max_files or 8)
    mode = "full" if args.full else "pilot"
    output_dir = args.output_dir.resolve() / mode / args.outlets
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output must be outside the read-only input tree")
    manifest = {
        "format_version": 1, "mode": mode, "outlets": args.outlets,
        "outlet_labels": list(THREE_OUTLETS) if args.outlets == "three" else "all stage-1 labels, including English",
        "input_dir": str(input_dir), "dictionary": str(dictionary),
        "dictionary_sha256": digest(dictionary),
        "parser_sha256": digest(Path(historical_parser.__file__)),
        "runner_sha256": digest(Path(__file__)),
        "source_files": files, "selected_files": selected,
        "batch_size": args.batch_size, "max_html_bytes": args.max_html_bytes,
        "assumptions": [
            "Date filter uses UTC crawl warc_date >= 2017-01-01 and < 2025-01-01, not extracted publication date.",
            "The historical Selectolax parser is reused unchanged, including its selectors, title handling, and 20000-character content truncation.",
            "Denominator includes unique accepted warc_id records with title OR content; title-only records cannot match content keywords.",
            "Matching is literal case-insensitive substring OR within each word set, AND across category operands, OR across categories.",
            "Three-outlet scope uses exact stage-1 labels; all-outlet scope includes English rows without language filtering.",
            "Weeks begin Monday and do not overlap; empty observed denominators produce NaN, not zero GPR. Boundary weeks are partial.",
            "Category contribution divides category matches by overall OR matches; overlapping categories can sum above 100 percent.",
            "Deduplication keeps the first accepted warc_id in deterministic source-file and row order; it does not deduplicate URL or syndicated stories.",
            "Pilot selection uses equal file-count strata within years, not a probability sample of articles; unselected records are not represented.",
            "Source guard is path/size/mtime_ns metadata plus parser/dictionary/runner SHA-256; source HTML is never written to outputs.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest differs: source files, code, dictionary, or settings changed; use a new --output-dir")
    else:
        if (output_dir / "articles.duckdb").exists():
            raise ValueError("Refusing existing database without a matching manifest")
        write_json(manifest_path, manifest)
    selected_paths = {item["path"] for item in selected}
    write_csv(output_dir / "inventory.csv", ["path", "size", "mtime_ns", "partition_year", "selected"],
              ({**item, "selected": item["path"] in selected_paths} for item in files))
    conn = setup_database(output_dir / "articles.duckdb")
    try:
        completed = {row[0] for row in conn.execute("SELECT path FROM processed_files").fetchall()}
        for item in selected:
            if item["path"] in completed:
                continue
            print(f"[{len(completed) + 1}/{len(selected)}] {item['path']}", flush=True)
            process_file(conn, input_dir, item, word_sets, categories, args.outlets,
                         args.batch_size, args.max_html_bytes)
            completed.add(item["path"])
            write_json(output_dir / "progress.json", {"status": "running", "completed_files": len(completed),
                       "selected_files": len(selected), "last_completed_file": item["path"]})
        summary = export_results(conn, output_dir, manifest, categories, "complete")
        write_json(output_dir / "progress.json", {"status": "complete", "completed_files": len(completed),
                   "selected_files": len(selected)})
        print(json.dumps({"output_dir": str(output_dir), "total_articles": summary["total_articles"],
                          "matching_articles": summary["matching_articles"], "mode": mode}, ensure_ascii=False))
        return summary
    except BaseException as error:
        write_json(output_dir / "progress.json", {"status": "interrupted_or_failed", "error": str(error),
                   "completed_files": conn.execute("SELECT count(*) FROM processed_files").fetchone()[0],
                   "selected_files": len(selected)})
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run(parse_arguments())
