"""Independent truth tables and boundary tests for aggregation of fresh shards."""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.aggregate_raw_rebuild import (
    CHINESE_CATEGORIES, LEGACY_OMITTED_STEP2, STANDARD_CATEGORIES, TARGET_OUTLETS,
    calendar_timestamps, compile_catalog, create_outputs, evaluate_corpus, evaluate_matches,
    flags_to_lists, inspect_input, materialize_articles, normalize_step2_category,
    file_hash, object_hash, scope_frames,
)


def document(name, family, sets, categories):
    return {"name": name, "family": family, "definition": {"word_sets": sets, "categories": categories}}


def small_catalog():
    return compile_catalog(["A", "B", "US", "us", "軍事"], [
        document("president_original7", "president", {"a": ["A", "US"], "b": ["B"], "empty": []}, {"Joint": "a AND b", "Empty": "empty"}),
        document("lau_2024", "lau", {"lower": ["us"], "b": ["B"]}, {"Lower": "lower + b"}),
    ])


def test_vectorized_matching_agrees_with_independent_set_truth_table():
    catalog = small_catalog()
    rows = [[0, 1], [], [1, 2], [1, 3], [], [4], [1], []]
    lists = pa.array([[4], *rows, [0]], type=pa.list_(pa.int32())).slice(1, len(rows))
    dictionary_flags, category_flags = evaluate_matches(lists, catalog)
    expected = []
    for row in rows:
        actual_words = {catalog.words[index] for index in row}
        expected.append([bool(actual_words & {"A", "US"}) and "B" in actual_words, "us" in actual_words and "B" in actual_words])
    np.testing.assert_array_equal(dictionary_flags, expected)
    assert not category_flags[:, 1].any()  # Empty word sets never match.
    assert flags_to_lists(dictionary_flags).to_pylist() == [[i for i, match in enumerate(row) if match] for row in expected]


def test_empty_lists_and_invalid_word_ids():
    catalog = small_catalog()
    flags, _ = evaluate_matches(pa.array([[], [], []], type=pa.list_(pa.int32())), catalog)
    assert not flags.any()
    with pytest.raises(ValueError, match="outside"):
        evaluate_matches(pa.array([[len(catalog.words)]], type=pa.list_(pa.int32())), catalog)
    with pytest.raises(ValueError, match="Null word_ids"):
        evaluate_matches(pa.array([None], type=pa.list_(pa.int32())), catalog)


def test_matching_across_64_bit_word_set_boundary():
    words = [f"W{index:03}" for index in range(70)]
    sets = {f"set{index}": [word] for index, word in enumerate(words)}
    catalog = compile_catalog(words, [document("president_original7", "president", sets, {"Boundary": "set63 AND set64", "Alias": "set64 + set63"})])
    rows = [[63, 64], [0, 64], [63], [64], [63, 64, 69], []]
    dictionaries, categories = evaluate_matches(pa.array(rows, type=pa.list_(pa.int32())), catalog)
    expected = [[63 in row and 64 in row] for row in rows]
    np.testing.assert_array_equal(dictionaries, expected)
    np.testing.assert_array_equal(categories, expected)
    assert len(catalog.word_sets) == 70
    assert len(catalog.expressions) == 1


def test_category_mapping_handles_all_supported_spellings_and_rejects_unknown():
    for english, chinese in zip(STANDARD_CATEGORIES, CHINESE_CATEGORIES):
        assert normalize_step2_category(english) == english
        assert normalize_step2_category(chinese) == english
        assert normalize_step2_category(f"{english} ({chinese})") == english
        assert normalize_step2_category(english.lower().replace("-", "_").replace(" ", "_")) == english
    with pytest.raises(ValueError, match="Unrecognized"):
        normalize_step2_category("unsupported")


def write_shard(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("warc_id", pa.string()), ("warc_date", pa.timestamp("us")),
        ("outlet", pa.string()), ("is_english", pa.bool_()),
        ("extracted_publish_date", pa.string()),
        ("source_path", pa.string()), ("source_row", pa.int64()),
        ("word_ids", pa.list_(pa.int32())),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def article(identifier, timestamp, source, row, words, publish_date=None):
    return {"warc_id": identifier, "warc_date": timestamp, "outlet": TARGET_OUTLETS[0], "is_english": False, "extracted_publish_date": publish_date, "source_path": source, "source_row": row, "word_ids": words}


def test_global_dedup_and_article_date_endpoint_policies(tmp_path):
    first, last = tmp_path / "a.parquet", tmp_path / "z.parquet"
    write_shard(last, [article("duplicate", datetime(2024, 12, 31, 12), "z.parquet", 0, [1, 3])])
    write_shard(first, [
        article("duplicate", datetime(2024, 12, 31), "a.parquet", 5, []),
        article("duplicate", datetime(2024, 12, 31), "a.parquet", 1, [0, 1]),
        article("afternoon", datetime(2024, 12, 31, 12), "a.parquet", 6, [1, 3]),
        article("boundary_start", datetime(2017, 1, 1), "a.parquet", 7, [0, 1]),
        article("outside", datetime(2025, 1, 1), "a.parquet", 8, [1, 3]),
        article("undated", None, "a.parquet", 9, []),
    ])
    connection = duckdb.connect()
    info = materialize_articles(connection, [str(last), str(first)])
    assert info["accepted_shard_rows"] == 7
    assert info["deduplicated_rows"] == 5
    assert info["duplicate_warc_id_rows_removed"] == 2
    assert info["undated_rows"] == 1
    assert connection.execute("SELECT source_path, source_row, word_ids FROM articles WHERE warc_id='duplicate'").fetchone() == ("a.parquet", 1, [0, 1])
    assert "content" not in [row[0] for row in connection.execute("DESCRIBE articles").fetchall()]
    weekly, daily = evaluate_corpus(connection, small_catalog(), batch_size=2, calendar_timezone="UTC")
    selected = {(policy, scope): data for policy, scope, data in scope_frames(weekly, "week", ["total_articles", "d_0", "d_1"])}
    assert selected[("full_cache", "three_outlets")].total_articles.sum() == 4
    assert selected[("study_half_open", "three_outlets")].total_articles.sum() == 3
    assert selected[("legacy_end_midnight", "three_outlets")].total_articles.sum() == 2
    first_week = pd.Timestamp("2016-12-26")
    assert first_week in set(selected[("study_half_open", "three_outlets")].week)
    assert connection.execute("SELECT count(*) FROM article_matches").fetchone()[0] == 5
    connection.close()


def test_publication_date_retained_verbatim_without_crawl_fallback(tmp_path):
    shard = tmp_path / "raw.parquet"
    crawl_time = datetime(2024, 12, 31, 16)
    publication = "2016-12-31T23:30:00+08:00"
    write_shard(shard, [
        article("duplicate", crawl_time, "z.parquet", 0, [1], "2025-01-01"),
        article("duplicate", crawl_time, "a.parquet", 0, [0], publication),
        article("missing", crawl_time, "a.parquet", 1, [], None),
        article("unparsed", None, "a.parquet", 2, [], "更新時間：2017年1月1日"),
    ])
    connection = duckdb.connect()
    materialize_articles(connection, [str(shard)])
    rows = connection.execute("SELECT warc_id, warc_date, extracted_publish_date FROM articles ORDER BY warc_id").fetchall()
    assert rows == [
        ("duplicate", crawl_time, publication),
        ("missing", crawl_time, None),
        ("unparsed", None, "更新時間：2017年1月1日"),
    ]
    assert connection.execute("SELECT typeof(extracted_publish_date) FROM articles LIMIT 1").fetchone()[0] == "VARCHAR"
    connection.close()


def test_disk_partition_dedup_is_global_and_deterministic_across_source_batches(tmp_path):
    paths = [tmp_path / f"part-{i}.parquet" for i in range(4)]
    day = datetime(2020, 1, 1)
    write_shard(paths[0], [article("duplicate", day, "z-source", 0, [1], "later")])
    write_shard(paths[1], [article("duplicate", None, "a-source", 5, [0], "wrong-row")])
    write_shard(paths[2], [article("duplicate", None, "a-source", 1, [], "winner")])
    write_shard(paths[3], [article(f"unique-{i:02}", day, "b-source", i, [0, 1]) for i in range(100)])
    results = []
    for ordering in (paths, list(reversed(paths))):
        connection = duckdb.connect()
        connection.execute("SET memory_limit='64MB'")
        connection.execute("SET threads=1")
        info = materialize_articles(connection, [str(path) for path in ordering], staging_parent=tmp_path, shard_batch_size=1, dedup_partitions=8)
        assert info["accepted_shard_rows"] == 103
        assert info["deduplicated_rows"] == 101
        assert info["duplicate_warc_id_rows_removed"] == 2
        assert info["undated_rows"] == 1
        results.append(connection.execute("SELECT * FROM articles ORDER BY article_id").fetchall())
        assert connection.execute("SELECT source_path, source_row, word_ids, extracted_publish_date FROM articles WHERE warc_id='duplicate'").fetchone() == ("a-source", 1, [], "winner")
        assert connection.execute("SELECT count(DISTINCT article_id), count(DISTINCT warc_id) FROM articles").fetchone() == (101, 101)
        connection.close()
    assert results[0] == results[1]
    assert not list(tmp_path.glob("dedup-*"))


@pytest.mark.parametrize("identifier", [None, ""])
def test_disk_partition_dedup_rejects_missing_warc_id(tmp_path, identifier):
    path = tmp_path / "invalid.parquet"
    write_shard(path, [article(identifier, None, "invalid", 0, [])])
    connection = duckdb.connect()
    with pytest.raises(ValueError, match="missing warc IDs"):
        materialize_articles(connection, [str(path)], staging_parent=tmp_path)
    connection.close()
    assert not list(tmp_path.glob("dedup-*"))


def test_match_materialization_crosses_checkpoint_threshold_without_open_reader_lock(tmp_path):
    # A subprocess timeout bounds regressions in DuckDB's native checkpoint lock.
    # The historical failure spun when per-batch INSERT crossed its WAL threshold
    # while another cursor still streamed articles from the same database.
    code = textwrap.dedent("""
        import sys
        from pathlib import Path
        import duckdb
        from scripts.aggregate_raw_rebuild import compile_catalog, evaluate_corpus
        directory = Path(sys.argv[1])
        connection = duckdb.connect(str(directory / 'matches.duckdb'))
        connection.execute("SET threads=1")
        connection.execute("SET memory_limit='64MB'")
        connection.execute("CREATE TABLE articles AS SELECT i::BIGINT AS article_id, TIMESTAMP '2020-01-01' AS warc_date, 'outlet' AS outlet, [0,1]::INTEGER[] AS word_ids FROM range(5000) t(i)")
        connection.execute("CREATE TABLE outlets AS SELECT 0::BIGINT AS outlet_id, 'outlet' AS outlet")
        connection.execute("CHECKPOINT")
        connection.execute("SET checkpoint_threshold='1KB'")
        catalog = compile_catalog(['A','B'], [{'name':'president_original7','family':'president','definition':{'word_sets':{'a':['A'],'b':['B']},'categories':{'Joint':'a AND b'}}}])
        weekly, daily = evaluate_corpus(connection, catalog, 17, staging_parent=directory)
        assert daily.total_articles.sum() == 5000
        assert connection.execute("SELECT count(*) FROM articles a JOIN article_matches m USING(article_id) WHERE m.dictionary_ids=[0] AND m.category_expression_ids=[0]").fetchone()[0] == 5000
        connection.execute("CHECKPOINT")
        connection.close()
        assert not list(directory.glob('matches-*'))
    """)
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("calendar_timezone, expected_start, expected_end, expected_legacy", [
    ("UTC", 0, 1, 1),
    ("Asia/Taipei", 1, 0, 0),
])
def test_calendar_timezone_controls_midnight_week_and_study_boundaries(tmp_path, calendar_timezone, expected_start, expected_end, expected_legacy):
    shard = tmp_path / "raw.parquet"
    write_shard(shard, [
        article("start", datetime(2016, 12, 31, 16), "raw.parquet", 0, [0, 1]),
        article("end", datetime(2024, 12, 31, 16), "raw.parquet", 1, [0, 1]),
        article("legacy", datetime(2024, 12, 31), "raw.parquet", 2, [1, 3]),
        article("week", datetime(2017, 1, 1, 16), "raw.parquet", 3, [1, 3]),
        article("undated", None, "raw.parquet", 4, []),
    ])
    connection = duckdb.connect()
    materialize_articles(connection, [str(shard)])
    weekly, daily = evaluate_corpus(connection, small_catalog(), 2, calendar_timezone)
    frames = {(policy, scope): data for policy, scope, data in scope_frames(weekly, "week", ["total_articles", "d_0", "d_1"])}
    study = frames[("study_half_open", "three_outlets")]
    assert study.d_0.sum() == expected_start + expected_end
    assert study.total_articles.sum() == expected_start + expected_end + 2
    assert frames[("legacy_end_midnight", "three_outlets")].d_1.sum() == 1 + expected_legacy
    week = pd.Timestamp("2016-12-26" if calendar_timezone == "UTC" else "2017-01-02")
    assert weekly.loc[weekly.week == week, "d_1"].sum() == 1
    # UTC timestamp storage is unchanged even when analysis uses Taipei dates.
    assert connection.execute("SELECT warc_date FROM articles WHERE warc_id='start'").fetchone()[0] == datetime(2016, 12, 31, 16)
    connection.close()
    converted = calendar_timestamps(np.array(["2024-12-30T16:00:00", "NaT"], dtype="datetime64[us]"), calendar_timezone)
    assert converted[0] == np.datetime64("2024-12-31T00:00:00" if calendar_timezone == "Asia/Taipei" else "2024-12-30T16:00:00")
    assert np.isnat(converted[1])


def test_canonical40_and_legacy39_medians_and_native_frequencies(tmp_path):
    docs = []
    for index in range(40):
        name = LEGACY_OMITTED_STEP2 if index == 39 else f"step2_test_{index:02}"
        signal = ["A"] if index < 20 else ["B"]
        categories = {category: "signal" for category in STANDARD_CATEGORIES}
        if index == 39:
            categories = {category.lower().replace("-", "_").replace(" ", "_"): "signal" for category in STANDARD_CATEGORIES}
        docs.append(document(name, "step2", {"signal": signal}, categories))
    docs.extend([
        document("president_original7", "president", {"a": ["A"]}, {"Original": "a"}),
        document("lau_2024", "lau", {"b": ["B"]}, {"Lau": "b"}),
    ])
    catalog = compile_catalog(["A", "B"], docs)
    shard = tmp_path / "raw.parquet"
    write_shard(shard, [
        article("one", datetime(2016, 12, 30), "raw.parquet", 0, [0]),
        article("two", datetime(2017, 1, 2), "raw.parquet", 1, [0, 1]),
        article("three", datetime(2017, 1, 3), "raw.parquet", 2, [0]),
    ])
    connection = duckdb.connect()
    materialize_articles(connection, [str(shard)])
    weekly, daily = evaluate_corpus(connection, catalog, batch_size=2)
    create_outputs(weekly, daily, catalog, tmp_path)
    medians = pd.read_parquet(tmp_path / "category_medians.parquet")
    selected = medians[(medians.scope == "three_outlets") & (medians.endpoint_policy == "full_cache") & (medians.week == pd.Timestamp("2016-12-26"))]
    assert set(selected[selected.variant == "canonical40"].gpr_index) == {0.5}
    assert set(selected[selected.variant == "legacy39"].gpr_index) == {1.0}
    assert set(selected[selected.variant == "canonical40"].dictionary_count) == {40}
    assert set(selected[selected.variant == "legacy39"].dictionary_count) == {39}
    original = pd.read_parquet(tmp_path / "series_president_original7.parquet")
    assert list(original.date) == [pd.Timestamp("2017-01-02"), pd.Timestamp("2017-01-03")]
    assert list(original.gpr_index) == [1 / (1 + 1e-9), 1 / (1 + 1e-9)]
    step2 = pd.read_parquet(tmp_path / "series_step2_median.parquet")
    assert len(step2) == 2
    assert step2.gpr_index.mean() == pytest.approx(100)
    connection.close()


def test_full_completion_guard_and_vocabulary_hash(tmp_path):
    vocabulary = {"words": ["A"], "sha256": object_hash(["A"])}
    (tmp_path / "word_vocabulary.json").write_text(json.dumps(vocabulary))
    (tmp_path / "manifest.json").write_text(json.dumps({"mode": "benchmark", "selected_files": [], "source_files": [{"path": "x.parquet"}]}))
    (tmp_path / "progress.json").write_text(json.dumps({"status": "complete", "selected_files": 0, "completed_files": 0}))
    with pytest.raises(ValueError, match="not complete"):
        inspect_input(tmp_path)
    vocabulary["words"] = ["B"]
    (tmp_path / "word_vocabulary.json").write_text(json.dumps(vocabulary))
    with pytest.raises(ValueError, match="Vocabulary hash"):
        inspect_input(tmp_path, allow_incomplete=True)


def test_completed_checkpoint_identity_and_matching_provenance(tmp_path):
    source = {"path": "2020/x.parquet", "size": 100, "mtime_ns": 1}
    dependencies = {"settings": {"case_sensitive": True}, "dictionary_files": [], "vocabulary_sha256": object_hash(["A"])}
    vocabulary = {"words": ["A"], "sha256": object_hash(["A"])}
    manifest = {"mode": "full", "selected_files": [source], "source_files": [source], "dependencies": dependencies, "dependency_hash": object_hash(dependencies), "source_inventory_sha256": object_hash([source])}
    (tmp_path / "word_vocabulary.json").write_text(json.dumps(vocabulary))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "progress.json").write_text(json.dumps({"status": "complete", "selected_files": 1, "completed_files": 1, "counts": {"source_rows": 1, "accepted_articles": 1}}))
    (tmp_path / "run_summary.json").write_text(json.dumps({"status": "complete"}))
    shard = tmp_path / "shards/2020/x.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"test")
    checkpoint = tmp_path / "checkpoints/2020/x.parquet.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"source": source, "dependency_hash": object_hash(dependencies), "shard_size": 4, "shard_sha256": file_hash(shard), "counts": {"source_rows": 1, "accepted_articles": 1}}))
    assert inspect_input(tmp_path)["complete"] is True
    empty_counts = json.loads(checkpoint.read_text())
    empty_counts["counts"] = {"source_rows": 1, "no_title_or_content": 1}
    checkpoint.write_text(json.dumps(empty_counts))
    with pytest.raises(ValueError, match="producer completion counts"):
        inspect_input(tmp_path)
    (tmp_path / "progress.json").write_text(json.dumps({"status": "complete", "selected_files": 1, "completed_files": 1, "counts": {"source_rows": 1, "no_title_or_content": 1}}))
    assert inspect_input(tmp_path)["checkpoint_manifest"][0]["accepted_articles"] == 0
    invalid_counts = dict(empty_counts, counts={"source_rows": 1})
    checkpoint.write_text(json.dumps(invalid_counts))
    with pytest.raises(ValueError, match="do not reconcile"):
        inspect_input(tmp_path)
    checkpoint.write_text(json.dumps(empty_counts))
    (tmp_path / "word_vocabulary.json").write_text(json.dumps({"words": ["B"], "sha256": object_hash(["B"])}))
    with pytest.raises(ValueError, match="parser dependency"):
        inspect_input(tmp_path)
    (tmp_path / "word_vocabulary.json").write_text(json.dumps(vocabulary))
    shard.write_bytes(b"same")
    with pytest.raises(ValueError, match="Shard SHA-256"):
        inspect_input(tmp_path)
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Shard size"):
        inspect_input(tmp_path)
