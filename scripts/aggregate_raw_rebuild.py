#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "duckdb==1.2.1", "polars==1.30.0", "pyarrow==20.0.0",
#   "numpy==2.2.3", "pandas==2.2.3",
# ]
# ///
"""Aggregate freshly parsed article shards; historical result caches are never inputs.

Word IDs retain the parser's case-sensitive literal matching. Matching uses NumPy
bit masks for unique word sets, then reuses each unique category expression.
Only compact metadata, matched IDs, and word IDs are stored in DuckDB.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
TARGET_OUTLETS = (
    "China Times (中國時報)", "Liberty Times (自由時報)", "United Daily (聯合報)",
)
EXCLUDED_STEP1 = frozenset({
    "step1_claude-opus-4_defense_minister_run2",
    "step1_claude-opus-4_poli_professor_run2",
    "step1_gemini-2.5-pro_journalist_run1",
    "step1_gemini-2.5-pro_nsb_director_run2",
    "step1_gemini-2.5-pro_poli_professor_run2",
})
LEGACY_OMITTED_STEP2 = "step2_gemini-2.5-pro_stable8cat_run8"
STANDARD_CATEGORIES = (
    "Military Escalation", "Diplomatic Isolation", "Economic Coercion",
    "Regional Security", "Internal Political Instability", "Gray Zone Operations",
    "US-China-Taiwan Relations", "Cyber and Information Warfare",
)
CHINESE_CATEGORIES = (
    "軍事升級", "外交孤立", "經濟脅迫", "區域安全", "內部政治不穩定", "灰色地帶行動",
    "美中台關係", "網路與資訊戰",
)
POLICIES = ("full_cache", "study_half_open", "legacy_end_midnight")


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def normalize_step2_category(name: str) -> str:
    for english, chinese in zip(STANDARD_CATEGORIES, CHINESE_CATEGORIES):
        snake = re.sub(r"[^a-z0-9]+", "_", english.lower()).strip("_")
        if name in (english, chinese, f"{english} ({chinese})", snake):
            return english
    raise ValueError(f"Unrecognized Step 2 category: {name!r}")


def parse_category(query: str, word_sets: dict[str, list[str]]) -> list[str]:
    # Preserve the historical parser's treatment of + and N/n proximity notation.
    names = [part.split("N/")[0].strip() for part in re.split(r"\s+AND\s+|\+", query)]
    if not names or any(not name or name not in word_sets for name in names):
        raise ValueError(f"Invalid category expression {query!r}")
    return names


@dataclass
class Catalog:
    words: list[str]
    dictionaries: list[dict[str, Any]]
    word_sets: list[tuple[int, ...]]
    expressions: list[tuple[int, ...]]
    word_masks: np.ndarray
    expression_masks: list[tuple[np.ndarray, np.ndarray]]

    def manifest(self) -> dict[str, Any]:
        return {
            "dictionaries": self.dictionaries,
            "unique_word_sets": [list(item) for item in self.word_sets],
            "unique_category_expressions": [list(item) for item in self.expressions],
            "dictionary_count": len(self.dictionaries),
            "word_count": len(self.words),
            "unique_word_set_count": len(self.word_sets),
            "unique_expression_count": len(self.expressions),
            "excluded_step1": sorted(EXCLUDED_STEP1),
        }


def compile_catalog(words: list[str], documents: list[dict[str, Any]]) -> Catalog:
    if words != sorted(set(words)):
        raise ValueError("Vocabulary must be sorted and contain no duplicates")
    word_ids = {word: index for index, word in enumerate(words)}
    unique_sets: dict[tuple[int, ...], int] = {}
    expressions: dict[tuple[int, ...], int] = {}
    dictionaries = []
    for document in documents:
        definition = document["definition"]
        sets = definition["word_sets"]
        set_ids = {}
        for name, members in sets.items():
            missing = set(members) - word_ids.keys()
            if missing:
                raise ValueError(f"Vocabulary lacks words for {document['name']}: {sorted(missing)}")
            key = tuple(sorted({word_ids[word] for word in members}))
            set_ids[name] = unique_sets.setdefault(key, len(unique_sets))
        categories = []
        seen = set()
        for raw_name, query in definition["categories"].items():
            name = normalize_step2_category(raw_name) if document["family"] == "step2" else raw_name
            if name in seen:
                raise ValueError(f"Duplicate normalized category {name} in {document['name']}")
            seen.add(name)
            key = tuple(sorted({set_ids[item] for item in parse_category(query, sets)}))
            expression_id = expressions.setdefault(key, len(expressions))
            categories.append({"name": name, "raw_name": raw_name, "expression_id": expression_id})
        if document["family"] == "step2" and seen != set(STANDARD_CATEGORIES):
            raise ValueError(f"Step 2 dictionary does not cover all eight categories: {document['name']}")
        dictionaries.append({key: value for key, value in document.items() if key != "definition"} | {"categories": categories})
    set_list = list(unique_sets)
    expression_list = list(expressions)
    chunks = (len(set_list) + 63) // 64
    masks = np.zeros((len(words), chunks), dtype=np.uint64)
    for set_id, members in enumerate(set_list):
        if members:
            masks[np.asarray(members), set_id // 64] |= np.uint64(1) << np.uint64(set_id % 64)
    expression_masks = []
    for required in expression_list:
        grouped: dict[int, int] = {}
        for set_id in required:
            grouped[set_id // 64] = grouped.get(set_id // 64, 0) | (1 << (set_id % 64))
        expression_masks.append((np.array(list(grouped), dtype=np.int64), np.array(list(grouped.values()), dtype=np.uint64)))
    return Catalog(words, dictionaries, set_list, expression_list, masks, expression_masks)


def load_catalog(vocabulary: dict, dictionary_dir: Path, president: Path, lau: Path) -> Catalog:
    documents = []
    counts = {"step1": 0, "step2": 0}
    for path in sorted(dictionary_dir.glob("*.json")):
        definition = json.loads(path.read_text())
        if not isinstance(definition, dict) or "word_sets" not in definition or "categories" not in definition:
            continue
        family = "step1" if path.stem.startswith("step1_") else "step2" if path.stem.startswith("step2_") else None
        if family is None:
            raise ValueError(f"Unexpected dictionary filename {path}")
        counts[family] += 1
        if path.stem in EXCLUDED_STEP1:
            continue
        documents.append({"name": path.stem, "family": family, "path": str(path.resolve()), "sha256": file_hash(path), "definition": definition})
    if counts != {"step1": 40, "step2": 40}:
        raise ValueError(f"Expected original 40 Step 1 and 40 Step 2 dictionaries, found {counts}")
    for name, family, path in (("president_original7", "president", president), ("lau_2024", "lau", lau)):
        documents.append({"name": name, "family": family, "path": str(path.resolve()), "sha256": file_hash(path), "definition": json.loads(path.read_text())})
    catalog = compile_catalog(vocabulary["words"], documents)
    if len(catalog.dictionaries) != 77:
        raise ValueError(f"Expected 77 selected dictionaries, found {len(catalog.dictionaries)}")
    return catalog


def inspect_input(directory: Path, allow_incomplete: bool = False) -> dict[str, Any]:
    LOG.info("Validating raw completion, vocabulary, and shard provenance in %s", directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    vocabulary = json.loads((directory / "word_vocabulary.json").read_text())
    if object_hash(vocabulary["words"]) != vocabulary["sha256"]:
        raise ValueError("Vocabulary hash does not match its word list")
    complete = (
        manifest.get("mode") == "full"
        and progress.get("status") == "complete"
        and progress.get("completed_files") == progress.get("selected_files")
        and progress.get("selected_files") == len(manifest["selected_files"])
        and manifest["selected_files"] == manifest["source_files"]
        and (directory / "run_summary.json").exists()
    )
    if not complete and not allow_incomplete:
        raise ValueError("Full raw rebuild is not complete. Use --allow-incomplete only for a labeled benchmark.")
    if manifest.get("dependencies", {}).get("settings", {}).get("case_sensitive") is not True:
        raise ValueError("Raw matching must explicitly declare case_sensitive=true")
    if object_hash(manifest["dependencies"]) != manifest.get("dependency_hash"):
        raise ValueError("Raw dependency manifest hash is invalid")
    if vocabulary["sha256"] != manifest["dependencies"].get("vocabulary_sha256"):
        raise ValueError("Vocabulary differs from the parser dependency manifest")
    if object_hash(manifest["source_files"]) != manifest.get("source_inventory_sha256"):
        raise ValueError("Raw source inventory hash is invalid")
    paths = []
    checkpoint_manifest = []
    for item in manifest["selected_files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid source-relative path: {relative}")
        shard = directory / "shards" / relative
        checkpoint = directory / "checkpoints" / relative.with_suffix(relative.suffix + ".json")
        # Checkpoint names preserve the entire source name followed by .json.
        if not checkpoint.exists():
            alternate = directory / "checkpoints" / relative.with_suffix(".json")
            if alternate.exists():
                checkpoint = alternate
        if not shard.exists() or not checkpoint.exists():
            if complete:
                raise ValueError(f"Missing completed shard/checkpoint for {relative}")
            continue
        record = json.loads(checkpoint.read_text())
        if record.get("source") != item:
            raise ValueError(f"Checkpoint source differs from raw manifest: {shard}")
        if record.get("shard_size") != shard.stat().st_size:
            raise ValueError(f"Shard size differs from completed checkpoint: {shard}")
        if record.get("dependency_hash") != manifest.get("dependency_hash"):
            raise ValueError(f"Mixed parsing/matching dependencies: {shard}")
        if record.get("shard_sha256") != file_hash(shard):
            raise ValueError(f"Shard SHA-256 differs from completed checkpoint: {shard}")
        counts = record["counts"]
        classified_rows = sum(counts.get(key, 0) for key in ("accepted_articles", "missing_essential_fields", "missing_html", "no_title_or_content"))
        if classified_rows != counts.get("source_rows", 0):
            raise ValueError(f"Accepted and rejected counts do not reconcile with source rows: {shard}")
        paths.append(str(shard.resolve()))
        # Parser Counter omits accepted_articles for a valid empty shard.
        checkpoint_manifest.append({"path": str(relative), "source_sha256": record.get("source_sha256"), "shard_sha256": record.get("shard_sha256"), "shard_size": record["shard_size"], "accepted_articles": record["counts"].get("accepted_articles", 0)})
        if len(paths) % 5000 == 0:
            LOG.info("Verified %s/%s shard hashes", len(paths), len(manifest["selected_files"]))
    if not paths:
        raise ValueError("No verified completed shards available")
    if complete and len(paths) != len(manifest["source_files"]):
        raise ValueError("Full manifest has fewer verified shards than source files")
    if complete and sum(item["accepted_articles"] for item in checkpoint_manifest) != progress.get("counts", {}).get("accepted_articles", 0):
        raise ValueError("Accepted checkpoint totals differ from producer completion counts")
    if len(set(paths)) != len(paths):
        raise ValueError("Raw manifest lists duplicate source paths")
    return {"manifest": manifest, "progress": progress, "vocabulary": vocabulary, "shards": paths, "checkpoint_manifest": checkpoint_manifest, "complete": complete}


def evaluate_matches(word_lists: pa.Array, catalog: Catalog) -> tuple[np.ndarray, np.ndarray]:
    if word_lists.null_count:
        raise ValueError("Null word_ids list in parsed shard")
    offsets = word_lists.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    flat = word_lists.values.to_numpy(zero_copy_only=False)[offsets[0]:offsets[-1]]
    offsets = offsets - offsets[0]
    size = len(word_lists)
    if len(flat) and (flat.min() < 0 or flat.max() >= len(catalog.words)):
        raise ValueError("word_ids contains an ID outside the supplied vocabulary")
    set_bits = np.zeros((size, catalog.word_masks.shape[1]), dtype=np.uint64)
    nonempty = np.diff(offsets) > 0
    if len(flat):
        starts = offsets[:-1][nonempty]
        expanded_bytes = len(flat) * catalog.word_masks.shape[1] * 8
        if expanded_bytes <= 64 * 1024 * 1024:
            set_bits[nonempty] = np.bitwise_or.reduceat(catalog.word_masks[flat], starts, axis=0)
        else:
            # Keep highly keyword-dense batches from expanding into a large matrix.
            for chunk in range(catalog.word_masks.shape[1]):
                set_bits[nonempty, chunk] = np.bitwise_or.reduceat(catalog.word_masks[flat, chunk], starts)
    category_flags = np.empty((size, len(catalog.expressions)), dtype=np.bool_)
    for index, (chunks, required) in enumerate(catalog.expression_masks):
        category_flags[:, index] = np.all((set_bits[:, chunks] & required) == required, axis=1)
    dictionary_flags = np.empty((size, len(catalog.dictionaries)), dtype=np.bool_)
    for index, dictionary in enumerate(catalog.dictionaries):
        expression_ids = [category["expression_id"] for category in dictionary["categories"]]
        dictionary_flags[:, index] = category_flags[:, expression_ids].any(axis=1)
    return dictionary_flags, category_flags


def flags_to_lists(flags: np.ndarray) -> pa.ListArray:
    rows, columns = np.nonzero(flags)
    offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(flags)))].astype(np.int32)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(columns.astype(np.int32)))


def sum_groups(keys: np.ndarray, flags: np.ndarray, accumulator: dict[int, np.ndarray]) -> None:
    unique, inverse = np.unique(keys, return_inverse=True)
    counts = np.empty((len(unique), flags.shape[1] + 1), dtype=np.uint64)
    counts[:, 0] = np.bincount(inverse, minlength=len(unique))
    for index in range(flags.shape[1]):
        counts[:, index + 1] = np.bincount(inverse, weights=flags[:, index], minlength=len(unique))
    for key, row in zip(unique, counts):
        integer = int(key)
        if integer in accumulator:
            accumulator[integer] += row
        else:
            accumulator[integer] = row.copy()


def materialize_articles(connection: duckdb.DuckDBPyConnection, paths: list[str], *, staging_parent: Path | None = None, shard_batch_size: int = 512, dedup_partitions: int = 64) -> dict[str, Any]:
    columns = "warc_id, warc_date, outlet, is_english, extracted_publish_date, source_path, source_row, word_ids"
    # No HTML, title, URL, or article content is copied into the aggregate database.
    if not paths or shard_batch_size < 1 or not 1 <= dedup_partitions <= 256:
        raise ValueError("Materialization needs input paths, a positive shard batch, and 1–256 partitions")
    if staging_parent:
        staging_parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Partitioning compact metadata from %s shards into %s deduplication buckets", len(paths), dedup_partitions)
    source_count = 0
    with tempfile.TemporaryDirectory(prefix="dedup-", dir=staging_parent) as temporary:
        staging = Path(temporary)
        for offset in range(0, len(paths), shard_batch_size):
            batch_paths = paths[offset:offset + shard_batch_size]
            connection.read_parquet(batch_paths, union_by_name=True).project(f"{columns}, (hash(warc_id) % {dedup_partitions})::UTINYINT AS dedup_bucket").create_view("raw_articles")
            batch_count, missing_ids = connection.execute("SELECT count(*), count(*) FILTER (WHERE warc_id IS NULL OR warc_id='') FROM raw_articles").fetchone()
            if missing_ids:
                raise ValueError(f"Parsed input contains {missing_ids} missing warc IDs")
            if offset == 0:
                connection.execute("CREATE TABLE articles AS SELECT 0::BIGINT AS article_id, * EXCLUDE (dedup_bucket) FROM raw_articles WHERE FALSE")
            destination = str(staging / f"batch-{offset:08}").replace("'", "''")
            connection.execute(f"COPY raw_articles TO '{destination}' (FORMAT PARQUET, PARTITION_BY (dedup_bucket), COMPRESSION ZSTD)")
            connection.execute("DROP VIEW raw_articles")
            source_count += batch_count
            if offset == 0 or (offset // shard_batch_size + 1) % 10 == 0:
                LOG.info("Partitioned %s/%s source shards (%s article rows)", min(offset + shard_batch_size, len(paths)), len(paths), source_count)
        # Each ID has exactly one bucket, so first-source ordering remains global.
        # Wide word-ID lists are only sorted within one bounded disk partition.
        if source_count >= 2**32:
            raise ValueError("Article count exceeds deterministic ID partition stride")
        for bucket in range(dedup_partitions):
            partition_paths = sorted(str(path) for path in staging.glob(f"batch-*/dedup_bucket={bucket}/*.parquet"))
            if not partition_paths:
                continue
            connection.read_parquet(partition_paths, hive_partitioning=False).project(columns).create_view("partition_articles")
            connection.execute(f"""INSERT INTO articles
                SELECT ({bucket}::BIGINT * 4294967296 + row_number() OVER (ORDER BY source_path, source_row, warc_id))::BIGINT AS article_id, *
                FROM (SELECT * FROM partition_articles
                      QUALIFY row_number() OVER (PARTITION BY warc_id ORDER BY source_path, source_row)=1)
                ORDER BY article_id""")
            connection.execute("DROP VIEW partition_articles")
            if (bucket + 1) % 8 == 0 or bucket + 1 == dedup_partitions:
                LOG.info("Deduplicated %s/%s disk partitions", bucket + 1, dedup_partitions)
    connection.execute("CREATE TABLE outlets AS SELECT row_number() OVER (ORDER BY outlet NULLS FIRST)-1 AS outlet_id, outlet FROM (SELECT DISTINCT outlet FROM articles)")
    count, first, last, undated = connection.execute("SELECT count(*), min(warc_date), max(warc_date), count(*) FILTER (WHERE warc_date IS NULL) FROM articles").fetchone()
    return {"accepted_shard_rows": source_count, "deduplicated_rows": count, "duplicate_warc_id_rows_removed": source_count-count, "minimum_warc_date": first, "maximum_warc_date": last, "undated_rows": undated, "deduplication": "first accepted row by lexicographic source_path, then source_row, globally per warc_id", "materialization": {"method": "disk-backed hash partitions", "shard_batch_size": shard_batch_size, "dedup_partitions": dedup_partitions, "article_id": "DuckDB hash(warc_id) bucket * 2^32 + one-based position ordered by source_path/source_row/warc_id within bucket", "temporary_partitions_removed": True}}


def calendar_timestamps(timestamps: np.ndarray, calendar_timezone: str) -> np.ndarray:
    """Interpret stored naive timestamps as UTC, then select the analysis calendar.

    The historical host inserted aware UTC datetimes into DuckDB TIMESTAMP with
    session TimeZone=Asia/Taipei, retaining local wall time. Raw shards preserve
    UTC instants; this explicit conversion reproduces that calendar without
    making results depend on the current machine's timezone.
    """
    if calendar_timezone not in {"Asia/Taipei", "UTC"}:
        raise ValueError(f"Unsupported calendar timezone: {calendar_timezone}")
    return pd.DatetimeIndex(timestamps).tz_localize("UTC").tz_convert(calendar_timezone).tz_localize(None).to_numpy(dtype="datetime64[us]")


def evaluate_corpus(connection: duckdb.DuckDBPyConnection, catalog: Catalog, batch_size: int, calendar_timezone: str = "Asia/Taipei", *, staging_parent: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly: dict[int, np.ndarray] = {}
    daily: dict[int, np.ndarray] = {}
    outlets = connection.execute("SELECT outlet_id, outlet FROM outlets ORDER BY outlet_id").fetchall()
    outlet_count = len(outlets)
    president_index = next(i for i, item in enumerate(catalog.dictionaries) if item["family"] == "president")
    president = catalog.dictionaries[president_index]
    president_expressions = [item["expression_id"] for item in president["categories"]]
    processed = 0
    next_progress = 1
    started = time.monotonic()
    start = np.datetime64("2017-01-01", "us")
    end = np.datetime64("2025-01-01", "us")
    legacy = np.datetime64("2024-12-31", "us")
    if staging_parent:
        staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="matches-", dir=staging_parent) as temporary:
        matches_path = Path(temporary) / "article_matches.parquet"
        schema = pa.schema([("article_id", pa.int64()), ("dictionary_ids", pa.list_(pa.int32())), ("category_expression_ids", pa.list_(pa.int32()))])
        writer = pq.ParquetWriter(matches_path, schema, compression="zstd")
        reader_connection = connection.cursor()
        try:
            reader = reader_connection.execute("SELECT a.article_id, a.warc_date, o.outlet_id, a.word_ids FROM articles AS a JOIN outlets AS o ON a.outlet IS NOT DISTINCT FROM o.outlet").fetch_record_batch(batch_size)
            for batch in reader:
                dictionary_flags, category_flags = evaluate_matches(batch.column("word_ids"), catalog)
                match_table = pa.table({"article_id": batch.column("article_id"), "dictionary_ids": flags_to_lists(dictionary_flags), "category_expression_ids": flags_to_lists(category_flags)}, schema=schema)
                # Never write to DuckDB while its article reader is open: an
                # auto-checkpoint can otherwise spin waiting for that reader.
                writer.write_table(match_table)
                timestamps = calendar_timestamps(batch.column("warc_date").to_numpy(zero_copy_only=False), calendar_timezone)
                dated = ~np.isnat(timestamps)
                if dated.any():
                    stamps = timestamps[dated]
                    days = stamps.astype("datetime64[D]").astype(np.int64)
                    weeks = days - ((days + 3) % 7)
                    outlet_ids = batch.column("outlet_id").to_numpy(zero_copy_only=False)[dated]
                    eligibility = np.where((stamps >= start) & (stamps <= legacy), 1, np.where((stamps >= start) & (stamps < end), 2, 0))
                    week_keys = ((weeks * outlet_count + outlet_ids) * 3) + eligibility
                    flags = np.concatenate([dictionary_flags[dated], category_flags[dated]], axis=1)
                    sum_groups(week_keys, flags, weekly)
                    # Daily original-president output retains exact midnight endpoint eligibility.
                    daily_keys = ((days * outlet_count + outlet_ids) * 3) + eligibility
                    original_flags = np.column_stack([dictionary_flags[dated, president_index], category_flags[dated][:, president_expressions]])
                    sum_groups(daily_keys, original_flags, daily)
                processed += len(batch)
                if processed >= next_progress:
                    LOG.info("Matched %s unique articles in %.1fs", processed, time.monotonic()-started)
                    next_progress = processed + batch_size * 20
        finally:
            writer.close()
            reader_connection.close()
        LOG.info("Loading %s match rows after closing the article reader", processed)
        connection.execute("CREATE TABLE article_matches AS SELECT * FROM read_parquet(?)", [str(matches_path)])
        if connection.execute("SELECT count(*) FROM article_matches").fetchone()[0] != processed:
            raise ValueError("Match materialization count differs from evaluated articles")
    outlet_names = {int(index): name for index, name in outlets}
    def frame(accumulator, flag_columns, period):
        rows = []
        for key, counts in sorted(accumulator.items()):
            quotient, eligibility = divmod(key, 3)
            day, outlet_id = divmod(quotient, outlet_count)
            rows.append([pd.Timestamp(np.datetime64(day, "D")), outlet_names[outlet_id], eligibility, *counts.tolist()])
        return pd.DataFrame(rows, columns=[period, "outlet", "eligibility", "total_articles", *flag_columns])
    weekly_columns = [f"d_{i}" for i in range(len(catalog.dictionaries))] + [f"c_{i}" for i in range(len(catalog.expressions))]
    daily_columns = ["gpr_articles"] + [f"president_category_{i}" for i in range(len(president_expressions))]
    return frame(weekly, weekly_columns, "week"), frame(daily, daily_columns, "date")


def scope_frames(frame: pd.DataFrame, period: str, columns: list[str]):
    for policy in POLICIES:
        selected = frame if policy == "full_cache" else frame[frame.eligibility.isin([1, 2] if policy == "study_half_open" else [1])]
        by_outlet = selected.groupby([period, "outlet"], dropna=False, as_index=False)[columns].sum()
        yield policy, "outlet", by_outlet
        overall = by_outlet.groupby(period, as_index=False)[columns].sum()
        overall["outlet"] = ""
        yield policy, "all_outlets", overall
        three = by_outlet[by_outlet.outlet.isin(TARGET_OUTLETS)].groupby(period, as_index=False)[columns].sum()
        three["outlet"] = ""
        yield policy, "three_outlets", three


def save_frame(frame: pd.DataFrame, name: str, output: Path, paths: dict, csv: bool = True) -> None:
    parquet = output / f"{name}.parquet"
    temporary = parquet.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(parquet)
    paths[name] = {"parquet": parquet.name, "rows": len(frame)}
    if csv:
        csv_path = output / f"{name}.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(csv_path)
        paths[name]["csv"] = csv_path.name


def create_outputs(weekly: pd.DataFrame, daily: pd.DataFrame, catalog: Catalog, output: Path) -> dict:
    paths: dict[str, dict] = {}
    dictionary_columns = [f"d_{i}" for i in range(len(catalog.dictionaries))]
    flag_columns = dictionary_columns + [f"c_{i}" for i in range(len(catalog.expressions))]
    dictionary_parts, category_parts, weekly_article_parts = [], [], []
    dictionary_names = {f"d_{i}": document["name"] for i, document in enumerate(catalog.dictionaries)}
    for policy, scope, data in scope_frames(weekly, "week", ["total_articles", *flag_columns]):
        common = data[["week", "outlet", "total_articles"]].assign(endpoint_policy=policy, scope=scope)
        weekly_article_parts.append(common)
        melted = data[["week", "outlet", "total_articles", *dictionary_columns]].melt(id_vars=["week", "outlet", "total_articles"], var_name="dictionary", value_name="gpr_articles")
        melted["dictionary"] = melted.dictionary.map(dictionary_names)
        melted["gpr_share"] = melted.gpr_articles / melted.total_articles
        melted["scope"], melted["endpoint_policy"] = scope, policy
        dictionary_parts.append(melted)
        # Category analysis uses the three newspapers, either pooled or separately.
        if scope == "all_outlets":
            continue
        if scope == "outlet":
            data = data[data.outlet.isin(TARGET_OUTLETS)]
        for document in catalog.dictionaries:
            if document["family"] not in {"step2", "president", "lau"}:
                continue
            for category in document["categories"]:
                part = data[["week", "outlet", "total_articles", f"c_{category['expression_id']}"]].rename(columns={f"c_{category['expression_id']}": "category_articles"})
                part = part.assign(dictionary=document["name"], category=category["name"], scope=scope, endpoint_policy=policy)
                part["category_share"] = part.category_articles / part.total_articles
                category_parts.append(part)
    dictionary_weekly = pd.concat(dictionary_parts, ignore_index=True)
    category_weekly = pd.concat(category_parts, ignore_index=True)
    del dictionary_parts, category_parts
    save_frame(dictionary_weekly, "dictionary_weekly", output, paths)
    save_frame(category_weekly, "category_weekly", output, paths)
    save_frame(pd.concat(weekly_article_parts, ignore_index=True), "article_weekly", output, paths)
    step2_names = {item["name"] for item in catalog.dictionaries if item["family"] == "step2"}
    groups = ["week", "scope", "outlet", "endpoint_policy"]
    step2 = dictionary_weekly[dictionary_weekly.dictionary.isin(step2_names)]
    median = step2.groupby(groups, dropna=False).agg(gpr_index=("gpr_share", "median"), mean=("gpr_share", "mean"), std=("gpr_share", "std"), dictionary_count=("dictionary", "nunique"), total_articles=("total_articles", "first")).reset_index()
    if not (median.dictionary_count == len(step2_names)).all():
        raise ValueError("Incomplete dictionary coverage in Step 2 median")
    scale_groups = ["scope", "outlet", "endpoint_policy"]
    means = median.groupby(scale_groups, dropna=False).gpr_index.transform("mean")
    median["gpr_index_scaled"] = np.where(means > 0, median.gpr_index / means * 100, np.nan)
    save_frame(median, "step2_median_weekly", output, paths)
    category_medians = []
    step2_categories = category_weekly[category_weekly.dictionary.isin(step2_names)]
    for variant in ("canonical40", "legacy39"):
        selected = step2_categories if variant == "canonical40" else step2_categories[step2_categories.dictionary != LEGACY_OMITTED_STEP2]
        medians = selected.groupby(groups + ["category"], dropna=False).agg(gpr_index=("category_share", "median"), dictionary_count=("dictionary", "nunique"), total_articles=("total_articles", "first")).reset_index()
        expected = len(step2_names) if variant == "canonical40" else len(step2_names - {LEGACY_OMITTED_STEP2})
        if not (medians.dictionary_count == expected).all():
            raise ValueError(f"Incomplete category coverage for {variant}")
        means = medians.groupby(scale_groups + ["category"], dropna=False).gpr_index.transform("mean")
        medians["gpr_index_scaled"] = np.where(means > 0, medians.gpr_index / means * 100, np.nan)
        medians["variant"] = variant
        category_medians.append(medians)
    save_frame(pd.concat(category_medians, ignore_index=True), "category_medians", output, paths)
    daily_parts = []
    daily_columns = [column for column in daily.columns if column not in ("date", "outlet", "eligibility")]
    for policy, scope, data in scope_frames(daily, "date", daily_columns):
        data = data.assign(scope=scope, endpoint_policy=policy, dictionary="president_original7")
        data["gpr_share"] = data.gpr_articles / data.total_articles
        daily_parts.append(data)
    original_daily = pd.concat(daily_parts, ignore_index=True)
    save_frame(original_daily, "original_president_daily", output, paths)
    article_daily = original_daily[["date", "outlet", "scope", "endpoint_policy", "total_articles"]]
    save_frame(article_daily, "article_daily", output, paths)
    yearly = article_daily.assign(year=article_daily.date.dt.year).groupby(["year", "outlet", "scope", "endpoint_policy"], dropna=False, as_index=False).total_articles.sum()
    save_frame(yearly, "article_yearly", output, paths)
    # Native daily frequency for the original dictionary; native weekly frequency for Step 2.
    president_series = original_daily[(original_daily.scope == "three_outlets") & (original_daily.endpoint_policy == "full_cache") & (original_daily.date >= pd.Timestamp("2017-01-01"))][["date", "total_articles", "gpr_articles", "gpr_share"]].copy()
    # Preserve the original calculator's numerical denominator guard.
    president_series["gpr_index"] = president_series.gpr_articles / (president_series.total_articles + 1e-9)
    save_frame(president_series, "series_president_original7", output, paths, csv=False)
    step2_series = median[(median.scope == "three_outlets") & (median.endpoint_policy == "full_cache")][["week", "gpr_index", "gpr_index_scaled", "dictionary_count", "total_articles"]].rename(columns={"week": "date", "gpr_index": "raw_median_share", "gpr_index_scaled": "gpr_index"})
    save_frame(step2_series, "series_step2_median", output, paths, csv=False)
    return paths


def run(args: argparse.Namespace) -> dict:
    input_info = inspect_input(args.input_dir, args.allow_incomplete)
    catalog = load_catalog(input_info["vocabulary"], args.dictionary_dir, args.president_dictionary, args.lau_dictionary)
    producer_hashes = {str(Path(item["path"]).resolve()): item["sha256"] for item in input_info["manifest"]["dependencies"]["dictionary_files"]}
    for dictionary in catalog.dictionaries:
        if producer_hashes.get(dictionary["path"]) != dictionary["sha256"]:
            raise ValueError(f"Dictionary changed after raw matching: {dictionary['name']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    if args.database.exists() and not args.overwrite:
        raise FileExistsError(f"Database already exists; use --overwrite for an intentional fresh run: {args.database}")
    if args.database.exists():
        args.database.unlink()
        Path(str(args.database) + ".wal").unlink(missing_ok=True)
    manifest = {
        "format_version": 1, "status": "running", "complete": False,
        "input_complete": input_info["complete"], "input_dir": str(args.input_dir.resolve()),
        "source_complete": input_info["complete"], "source_mode": input_info["manifest"]["mode"],
        "database": str(args.database.resolve()), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregator_sha256": file_hash(Path(__file__)),
        "source_manifest_sha256": object_hash(input_info["manifest"]),
        "shard_manifest_sha256": object_hash(input_info["checkpoint_manifest"]),
        "vocabulary_sha256": input_info["vocabulary"]["sha256"],
        "shard_integrity": "every completed shard SHA-256 recomputed and matched to its parser checkpoint",
        "calendar_timezone": args.calendar_timezone,
        "timestamp_semantics": {"raw_shards": "timezone-naive UTC instants", "aggregate_database": "warc_date remains timezone-naive UTC", "extracted_publish_date": "unaltered nullable source text; never filled from warc_date", "date_week_year_and_endpoints": args.calendar_timezone, "historical_cache": "UTC aware Python datetimes inserted into TIMESTAMP under DuckDB session TimeZone=Asia/Taipei"},
        "matching": "case-sensitive literal substring; union within word sets, intersection across required sets, union across categories",
        "dictionary_catalog": catalog.manifest(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("duckdb", "polars", "pyarrow", "numpy", "pandas")},
        "endpoint_policies": {"full_cache": "all accepted article timestamps with known date", "study_half_open": "2017-01-01 <= calendar-local warc_date < 2025-01-01", "legacy_end_midnight": "2017-01-01 <= calendar-local warc_date <= 2024-12-31T00:00:00"},
        "economics_series": {"series_president_original7": {"frequency": "daily", "scale": "raw share with historical total_articles + 1e-9 denominator", "scope": "three_outlets", "endpoint_policy": "2017-01-01 onward, including 2025"}, "series_step2_median": {"frequency": "weekly", "scale": "mean 100 across observed full-cache weekly medians", "scope": "three_outlets", "endpoint_policy": "full_cache"}},
    }
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    started = time.monotonic()
    connection = duckdb.connect(str(args.database))
    try:
        connection.execute(f"SET threads={args.threads}")
        connection.execute("SET memory_limit=?", [args.memory_limit])
        connection.execute("SET max_temp_directory_size=?", [args.max_temp_size])
        connection.execute("SET temp_directory=?", [str(args.output_dir / "duckdb_temp")])
        manifest["corpus"] = materialize_articles(connection, input_info["shards"], staging_parent=args.output_dir / "dedup_staging")
        expected_rows = sum(item["accepted_articles"] for item in input_info["checkpoint_manifest"])
        if manifest["corpus"]["accepted_shard_rows"] != expected_rows:
            raise ValueError("Accepted article count differs from parser checkpoints")
        write_json(manifest_path, manifest)
        manifest["corpus"]["warc_date_timezone"] = "UTC"
        dated_bounds = calendar_timestamps(np.array([manifest["corpus"]["minimum_warc_date"], manifest["corpus"]["maximum_warc_date"]], dtype="datetime64[us]"), args.calendar_timezone)
        manifest["corpus"]["minimum_calendar_date"] = str(dated_bounds[0])
        manifest["corpus"]["maximum_calendar_date"] = str(dated_bounds[1])
        weekly, daily = evaluate_corpus(connection, catalog, args.batch_size, args.calendar_timezone, staging_parent=args.output_dir / "match_staging")
        paths = create_outputs(weekly, daily, catalog, args.output_dir)
        corpus_counts = connection.execute("SELECT outlet, count(*) AS total_articles, count(*) FILTER (WHERE warc_date IS NULL) AS undated_articles, min(warc_date) AS minimum_warc_date_utc, max(warc_date) AS maximum_warc_date_utc FROM articles GROUP BY outlet ORDER BY outlet NULLS FIRST").fetchdf()
        save_frame(corpus_counts, "article_corpus_counts", args.output_dir, paths)
        connection.execute("CHECKPOINT")
        manifest.update(status="complete", complete=input_info["complete"], outputs=paths, elapsed_seconds=round(time.monotonic()-started, 3))
        write_json(manifest_path, manifest)
        LOG.info("Raw aggregation complete: %s articles in %.1fs", manifest["corpus"]["deduplicated_rows"], time.monotonic()-started)
    except BaseException as error:
        manifest.update(status="failed", error=str(error), elapsed_seconds=round(time.monotonic()-started, 3))
        write_json(manifest_path, manifest)
        raise
    finally:
        connection.close()
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "replication/raw_rebuild/news/full")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replication/raw_rebuild/indices")
    parser.add_argument("--database", type=Path, default=ROOT / "replication/raw_rebuild/aggregate.duckdb")
    parser.add_argument("--dictionary-dir", type=Path, default=ROOT / "data/dictionaries/all_dictionaries")
    parser.add_argument("--president-dictionary", type=Path, default=ROOT / "data/dictionaries/taiwan_gpr_dict_president.json")
    parser.add_argument("--lau-dictionary", type=Path, default=ROOT / "data/dictionaries/yeu_tong_lau_2024.json")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--max-temp-size", default="100GB")
    parser.add_argument("--calendar-timezone", choices=("Asia/Taipei", "UTC"), default="Asia/Taipei", help="Calendar for dates, Monday weeks, and study endpoints; shards remain UTC. UTC is a diagnostic alternative.")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
