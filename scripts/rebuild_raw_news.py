#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "duckdb==1.2.1", "polars==1.30.0", "selectolax==0.3.29",
#   "tqdm==4.67.1", "pyarrow==20.0.0", "pyahocorasick==2.1.0",
# ]
# ///
"""Rebuild accepted articles and exact dictionary-word presence from raw HTML.

No date, outlet, or language filtering; no LLM calls. Each source file produces
one atomic parsed Parquet shard and a completion checkpoint. Full processing
requires --full; --max-files creates a separate representative benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import ahocorasick
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_parse_html_load_db as historical_parser


SOURCE_COLUMNS = ["warc_id", "warc_date", "url", "domain", "outlet", "is_english", "html_content_bytes"]
SCHEMA = pa.schema([
    ("warc_id", pa.string()), ("warc_date", pa.timestamp("us")),
    ("url", pa.string()), ("domain", pa.string()), ("outlet", pa.string()),
    ("is_english", pa.bool_()), ("title", pa.string()), ("content", pa.string()),
    ("extracted_publish_date", pa.string()), ("source_path", pa.string()),
    ("source_row", pa.int64()), ("word_ids", pa.list_(pa.int32())),
])
WORKER = {}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_stat(path: Path, relative: str) -> dict:
    value = path.stat()
    return {"path": relative, "size": value.st_size, "mtime_ns": value.st_mtime_ns}


def collect_vocabulary(dictionary_paths: list[Path]) -> tuple[list[str], list[dict]]:
    words = set()
    dictionaries = []
    for path in sorted(set(path.resolve() for path in dictionary_paths)):
        document = json.loads(path.read_text())
        sets = document.get("word_sets")
        if not isinstance(sets, dict) or not sets:
            raise ValueError(f"Dictionary has no word_sets: {path}")
        for name, values in sets.items():
            if not isinstance(values, list) or not values or any(not isinstance(word, str) or not word.strip() for word in values):
                raise ValueError(f"Invalid word set {name!r} in {path}")
            words.update(values)
        dictionaries.append({"path": str(path), "sha256": sha256(path)})
    if not words:
        raise ValueError("No dictionary words were provided")
    return sorted(words), dictionaries


def build_automaton(words: list[str]):
    automaton = ahocorasick.Automaton()
    for identifier, word in enumerate(words):
        automaton.add_word(word, identifier)
    automaton.make_automaton()
    return automaton


def match_word_ids(content: str | None, automaton) -> list[int]:
    matches = set()
    for _, identifier in automaton.iter(content or ""):
        matches.add(identifier)
    return sorted(matches)


def select_files(files: list[dict], maximum: int | None) -> list[dict]:
    if maximum is None or maximum >= len(files):
        return files
    groups = defaultdict(list)
    for item in files:
        year = next((part for part in Path(item["path"]).parts[:-1] if part.isdigit() and len(part) == 4), "unknown")
        groups[year].append(item)
    allocations = {year: 0 for year in groups}
    for _ in range(maximum):
        year = min((year for year in groups if allocations[year] < len(groups[year])),
                   key=lambda year: (allocations[year], year))
        allocations[year] += 1
    selected = []
    for year, group in groups.items():
        count = allocations[year]
        selected.extend(group[(2 * index + 1) * len(group) // (2 * count)] for index in range(count))
    return sorted(selected, key=lambda item: item["path"])


def load_parser(parser_file: Path):
    if parser_file.resolve() == Path(historical_parser.__file__).resolve():
        return historical_parser
    spec = importlib.util.spec_from_file_location("selected_historical_parser", parser_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not any(callable(getattr(module, name, None)) for name in ("extract_article_content_selectolax", "extract_article_record")):
        raise ValueError(f"Parser module has no extraction function: {parser_file}")
    return module


def worker_init(input_dir: str, output_dir: str, words: list[str], settings: dict, dependency_hash: str,
                parser_file: str | None = None):
    # Avoid multiplying Arrow's internal thread pool by the process count.
    pa.set_cpu_count(1)
    WORKER.update(input_dir=Path(input_dir), output_dir=Path(output_dir),
                  automaton=build_automaton(words), settings=settings, dependency_hash=dependency_hash,
                  parser=load_parser(Path(parser_file) if parser_file else Path(historical_parser.__file__)))


def artifact_paths(output_dir: Path, relative: str) -> tuple[Path, Path]:
    return output_dir / "shards" / relative, output_dir / "checkpoints" / (relative + ".json")


def normalized_timestamp(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError(f"Unsupported warc_date type: {type(value).__name__}")
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def process_source(item: dict) -> dict:
    started = time.monotonic()
    path = WORKER["input_dir"] / item["path"]
    shard, checkpoint_path = artifact_paths(WORKER["output_dir"], item["path"])
    if source_stat(path, item["path"]) != item:
        raise ValueError(f"Source metadata changed: {path}")
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_name(shard.name + f".tmp-{os.getpid()}")
    counts = Counter()
    parquet = pq.ParquetFile(path)
    missing = set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing source columns {sorted(missing)}: {path}")
    prepare_source = getattr(WORKER["parser"], "prepare_source", None)
    source_context = prepare_source(parquet, item["path"]) if prepare_source else {"_source_path": item["path"]}
    try:
        with pq.ParquetWriter(temporary, SCHEMA, compression="zstd", compression_level=3) as writer:
            for batch in parquet.iter_batches(batch_size=WORKER["settings"]["batch_size"], columns=SOURCE_COLUMNS, use_threads=False):
                accepted = []
                for row in batch.to_pylist():
                    source_row = counts["source_rows"]
                    counts["source_rows"] += 1
                    if not all(row.get(key) for key in ("warc_id", "url", "domain")):
                        counts["missing_essential_fields"] += 1
                        continue
                    html = row["html_content_bytes"]
                    if html is None:
                        counts["missing_html"] += 1
                        continue
                    if len(html) > WORKER["settings"]["max_html_bytes"]:
                        raise ValueError(f"Oversized HTML at {item['path']} row {source_row}; increase --max-html-bytes explicitly")
                    record_parser = getattr(WORKER["parser"], "extract_article_record", None)
                    parsed = (record_parser({**row, **source_context}) if record_parser else
                              WORKER["parser"].extract_article_content_selectolax(html, row["url"], row["domain"]))
                    if not (parsed.get("title") or parsed.get("content")):
                        counts["no_title_or_content"] += 1
                        continue
                    timestamp = normalized_timestamp(row["warc_date"])
                    counts["missing_crawl_date"] += timestamp is None
                    counts["title_only_articles"] += not bool(parsed.get("content"))
                    word_ids = match_word_ids(parsed.get("content"), WORKER["automaton"])
                    counts["accepted_articles"] += 1
                    counts["articles_with_any_word"] += bool(word_ids)
                    accepted.append({**{key: row[key] for key in ("warc_id", "url", "domain", "outlet", "is_english")},
                                     "warc_date": timestamp, **parsed, "source_path": item["path"],
                                     "source_row": source_row, "word_ids": word_ids})
                if accepted:
                    writer.write_table(pa.Table.from_pylist(accepted, schema=SCHEMA))
        source_hash = sha256(path)
        if source_stat(path, item["path"]) != item:
            raise ValueError(f"Source changed during parsing: {path}")
        shard_hash = sha256(temporary)
        temporary.replace(shard)
        checkpoint = {"source": item, "source_sha256": source_hash,
                      "dependency_hash": WORKER["dependency_hash"],
                      "shard_size": shard.stat().st_size, "shard_sha256": shard_hash,
                      "counts": dict(counts), "elapsed_seconds": time.monotonic() - started}
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        checkpoint["worker_peak_rss"] = {"raw": peak_rss, "raw_unit": "bytes" if sys.platform == "darwin" else "KiB",
                                         "bytes": int(peak_rss if sys.platform == "darwin" else peak_rss * 1024),
                                         "meaning": "Process lifetime high-water resident set, not sum across workers"}
        atomic_json(checkpoint_path, checkpoint)
        return checkpoint
    finally:
        parquet.close()
        temporary.unlink(missing_ok=True)


def read_checkpoint(output_dir: Path, item: dict, dependency_hash: str) -> dict | None:
    shard, checkpoint_path = artifact_paths(output_dir, item["path"])
    if not checkpoint_path.exists():
        return None  # An orphan shard from an interrupted rename is safely rebuilt.
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint["source"] != item or checkpoint["dependency_hash"] != dependency_hash:
        raise ValueError(f"Checkpoint source or dependency mismatch: {checkpoint_path}")
    if not shard.exists() or shard.stat().st_size != checkpoint["shard_size"] or sha256(shard) != checkpoint["shard_sha256"]:
        raise ValueError(f"Checkpoint shard is missing or changed: {shard}")
    return checkpoint


def parse_arguments(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=root.parent / "thesis/data/news/stage1_parquet")
    parser.add_argument("--dictionary-dir", type=Path, default=root / "data/dictionaries/all_dictionaries")
    parser.add_argument("--extra-dictionary", type=Path, action="append", help="Repeatable; defaults to original president and Lau dictionaries")
    parser.add_argument("--output-dir", type=Path, default=root / "replication/raw_rebuild/news")
    parser.add_argument("--parser-file", type=Path, default=root / "replication/reference_code/stage2_cache_replay.py",
                        help="Historical extraction module; path and SHA256 are guarded on resume")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--full", action="store_true")
    scope.add_argument("--max-files", type=int, help="Representative benchmark file cap (default 10, roughly one/year)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-html-bytes", type=int, default=128 * 1024 * 1024)
    args = parser.parse_args(argv)
    if any(value <= 0 for value in (args.workers, args.batch_size, args.max_html_bytes)) or (args.max_files is not None and args.max_files <= 0):
        parser.error("Worker, file, batch, and HTML bounds must be positive")
    if args.extra_dictionary is None:
        args.extra_dictionary = [root / "data/dictionaries/taiwan_gpr_dict_president.json", root / "data/dictionaries/yeu_tong_lau_2024.json"]
    return args


def run(args) -> dict:
    started = time.monotonic()
    input_dir = args.input_dir.resolve()
    files = [source_stat(path, str(path.relative_to(input_dir))) for path in sorted(input_dir.rglob("*.parquet"))]
    if not files:
        raise ValueError(f"No source parquet files found: {input_dir}")
    maximum = None if args.full else (args.max_files or 10)
    selected = select_files(files, maximum)
    output_dir = args.output_dir.resolve() / ("full" if args.full else f"benchmark-{maximum}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output cannot be inside the read-only source tree")
    dictionary_files = sorted(path for path in args.dictionary_dir.glob("*.json")
                              if path.name != "word_column_mapping.json")
    if not dictionary_files:
        raise ValueError(f"No dictionaries found: {args.dictionary_dir}")
    words, dictionary_info = collect_vocabulary(dictionary_files + args.extra_dictionary)
    settings = {"batch_size": args.batch_size, "max_html_bytes": args.max_html_bytes, "case_sensitive": True}
    parser_file = args.parser_file.resolve()
    load_parser(parser_file)  # Fail before scheduling if the selected module is invalid.
    dependencies = {"parser_file": str(parser_file), "parser_sha256": sha256(parser_file),
                    "runner_sha256": sha256(Path(__file__)), "dictionary_files": dictionary_info,
                    "vocabulary_sha256": object_hash(words), "settings": settings}
    dependency_hash = object_hash(dependencies)
    manifest = {"format_version": 1, "input_dir": str(input_dir), "mode": "full" if args.full else "benchmark",
                "dependencies": dependencies, "dependency_hash": dependency_hash,
                "source_inventory_sha256": object_hash(files), "source_files": files, "selected_files": selected,
                "unique_words": len(words), "schema": str(SCHEMA),
                "semantics": [
                    "No date, outlet, or language filter; UTC warc_date retained; null dates remain visible.",
                    "Selected parser code and dependencies are guarded by hashes; historical selector defects and content truncation are preserved according to the selected parser's provenance.",
                    "Accepted denominator is title OR content; title-only articles have no word hits.",
                    "word_ids are sorted unique IDs into word_vocabulary.json, matched as case-sensitive literal word in content, following the historical binary-matrix calculator.",
                    "All accepted source rows are retained, including duplicate warc_id. Downstream global dedup must keep first by source_path then source_row.",
                    "Per-source SHA-256 recorded after parsing; resume guards source path/size/mtime_ns and verifies every completed shard SHA-256.",
                    "Aho-Corasick includes overlapping substring matches; repeated occurrences count once per article.",
                ]}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest differs: source inventory, parser, runner, vocabulary, or content settings changed; use a new output directory")
    else:
        if (output_dir / "shards").exists() or (output_dir / "checkpoints").exists():
            raise ValueError("Refusing existing artifacts without their manifest")
        atomic_json(manifest_path, manifest)
    atomic_json(output_dir / "word_vocabulary.json", {"words": words, "sha256": object_hash(words),
                "id_definition": "zero-based position in words; exact original spelling, sorted by Unicode code point"})
    totals = Counter()
    pending = []
    completed = 0
    completed_bytes = 0
    max_worker_rss_bytes = 0
    for item in selected:
        checkpoint = read_checkpoint(output_dir, item, dependency_hash)
        if checkpoint is None:
            pending.append(item)
        else:
            completed += 1
            completed_bytes += item["size"]
            totals.update(checkpoint["counts"])
            max_worker_rss_bytes = max(max_worker_rss_bytes, checkpoint.get("worker_peak_rss", {}).get("bytes", 0))
    resumed = completed
    source_total_bytes = sum(item["size"] for item in selected)

    def save_progress(status, last_file=None, error=None):
        report = {"status": status, "output_dir": str(output_dir), "workers": args.workers,
                  "selected_files": len(selected), "completed_files": completed, "resumed_files": resumed,
                  "source_total_bytes": source_total_bytes, "source_completed_bytes": completed_bytes,
                  "elapsed_seconds_this_invocation": time.monotonic() - started,
                  "counts": dict(totals), "last_completed_file": last_file, "error": error}
        report["max_worker_peak_rss_bytes"] = max_worker_rss_bytes
        atomic_json(output_dir / "progress.json", report)
        return report

    save_progress("running")
    print(json.dumps({"output_dir": str(output_dir), "selected_files": len(selected), "resumed_files": resumed,
                      "workers": args.workers, "unique_words": len(words)}, ensure_ascii=False), flush=True)
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=worker_init, initargs=(str(input_dir), str(output_dir), words, settings, dependency_hash, str(parser_file)))
    futures = {}
    iterator = iter(pending)

    def submit_next():
        item = next(iterator, None)
        if item is not None:
            futures[executor.submit(process_source, item)] = item

    try:
        for _ in range(min(len(pending), args.workers * 2)):
            submit_next()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                item = futures.pop(future)
                checkpoint = future.result()
                totals.update(checkpoint["counts"])
                max_worker_rss_bytes = max(max_worker_rss_bytes, checkpoint.get("worker_peak_rss", {}).get("bytes", 0))
                completed += 1
                completed_bytes += item["size"]
                save_progress("running", item["path"])
                print(f"[{completed}/{len(selected)}] {item['path']} accepted={checkpoint['counts'].get('accepted_articles', 0)}", flush=True)
                submit_next()
        report = save_progress("complete")
        atomic_json(output_dir / "run_summary.json", {**report, "unique_words": len(words),
                    "dictionary_files": len(dictionary_info), "deduplicated": False,
                    "interpretation": "Parsed source rows, before global warc_id deduplication or study-specific scope filters."})
        return report
    except BaseException as error:
        for future in futures:
            future.cancel()
        save_progress("interrupted_or_failed", error=str(error))
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    run(parse_arguments())
