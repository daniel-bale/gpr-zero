"""Bounded real-article checks against independently stored matching masks.

The sample is hash partition zero during the Taipei week beginning 2022-08-01,
not the entire production week. Source access is read-only. Storage statistics
must prove that the ID predicate prunes nearly all row groups before any article
data is read. Only the selected rows are registered in an ephemeral DuckDB view.
"""

import hashlib
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import duckdb
import ibis
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gpr_zero.constants import TARGET_OUTLETS
from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory
from gpr_zero.gpr_index.dictionary import DictionaryMatcher

DICTIONARIES = [
    "step2_claude-opus-4_stable8cat_run1",
    "step2_gemini-2.5-pro_stable8cat_run1",
    "step2_gpt-4.1_stable8cat_run1",
]
PARTITION_END = 2**32


def estimated_rows_in_id_range(connection, table, first_id, last_id):
    """Read only DuckDB storage metadata, never an article COUNT scan."""
    storage = connection.execute(f"PRAGMA storage_info('{table}')").fetchdf()
    segments = storage.loc[
        (storage.column_name == "article_id") & (storage.segment_type != "VALIDITY")
    ]
    groups = set()
    rows = 0
    for segment in segments.itertuples():
        bounds = re.search(r"Min: (\d+), Max: (\d+)", segment.stats)
        assert bounds is not None, "Cannot establish bounded source scan from metadata"
        lower, upper = map(int, bounds.groups())
        if lower <= last_id and upper >= first_id:
            groups.add(segment.row_group_id)
            rows += segment.count
    assert groups, f"No row groups cover the selected IDs in {table}"
    assert len(groups) <= 3 and rows <= 400_000, (
        f"Refusing expensive source scan: {table}: {len(groups)} groups, {rows} rows"
    )
    return len(groups), rows


@pytest.fixture(scope="module")
def production_sample():
    configured = os.environ.get("GPR_THESIS_PROFILE")
    if not configured:
        pytest.skip("Set GPR_THESIS_PROFILE to run bounded production matching checks")
    started = time.perf_counter()
    rebuild = Path(configured).expanduser().parent
    manifest = json.loads((rebuild / "indices" / "manifest.json").read_text())
    catalog = manifest["dictionary_catalog"]["dictionaries"]
    vocabulary = json.loads(
        (Path(manifest["input_dir"]) / "word_vocabulary.json").read_text()
    )
    words = vocabulary["words"]
    # The manifest records a semantic hash of the JSON word list, not the hash
    # of the enclosing vocabulary file. Validate rather than trusting ordinals.
    assert (
        hashlib.sha256(json.dumps(words, ensure_ascii=False).encode()).hexdigest()
        == (vocabulary["sha256"])
        == manifest["vocabulary_sha256"]
    )
    assert words == sorted(set(words))
    sparse_ids = {word: index for index, word in enumerate(words)}
    selected = {
        entry["name"]: (index, entry)
        for index, entry in enumerate(catalog)
        if entry["name"] in DICTIONARIES
    }
    assert set(selected) == set(DICTIONARIES)
    dictionary_directory = Path(selected[DICTIONARIES[0]][1]["path"]).parent
    inventory = DictionaryInventory.from_directory(dictionary_directory)
    assert set(inventory.words).issubset(sparse_ids)
    for name, (_, entry) in selected.items():
        path = Path(entry["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert inventory.dictionaries[name] == json.loads(path.read_text())
        assert set(inventory.dictionaries[name]["categories"]) == {
            category["raw_name"] for category in entry["categories"]
        }

    with duckdb.connect(str(rebuild / "aggregate.duckdb"), read_only=True) as source:
        source.execute("SET threads = 1")
        article_budget = estimated_rows_in_id_range(
            source, "articles", 1, PARTITION_END - 1
        )
        # Aggregate warc_date values are naive UTC. These are the UTC instants
        # corresponding to Taipei's Monday 00:00 through next Monday 00:00.
        articles = source.execute(
            "SELECT article_id, warc_date, outlet, word_ids FROM articles "
            "WHERE article_id >= 1 AND article_id < 4294967296 "
            "AND warc_date >= TIMESTAMP '2022-07-31 16:00:00' "
            "AND warc_date < TIMESTAMP '2022-08-07 16:00:00' "
            "ORDER BY article_id"
        ).fetch_arrow_table()
        frame = articles.to_pandas()
        assert 0 < len(frame) <= 2000
        assert frame.article_id.is_unique
        first_id, last_id = int(frame.article_id.min()), int(frame.article_id.max())
        match_budget = estimated_rows_in_id_range(
            source, "article_matches", first_id, last_id
        )
        stored = source.execute(
            "SELECT article_id, dictionary_ids, category_expression_ids "
            "FROM article_matches WHERE article_id >= ? AND article_id <= ? "
            "ORDER BY article_id",
            [first_id, last_id],
        ).fetchdf()
    assert stored.article_id.is_unique
    reference = frame.merge(stored, on="article_id", validate="one_to_one")
    assert len(reference) == len(frame), "Every sampled article needs a saved match row"

    memory = ibis.duckdb.connect()
    memory.con.register("bounded_articles", articles)
    table = memory.table("bounded_articles")
    table = table.mutate(warc_date=table.warc_date + ibis.interval(hours=8))
    needed_words = {
        word
        for name in DICTIONARIES
        for word_set in inventory.dictionaries[name]["word_sets"].values()
        for word in word_set
    }
    # Sparse IDs and contains_word ordinals belong to different inventories.
    # Translate by exact word spelling before using the real matching module.
    table = table.mutate(
        **{
            inventory.word_columns[word]: table.word_ids.contains(sparse_ids[word])
            for word in sorted(needed_words)
        }
    )
    corpus = ArticleCorpus(table, inventory)
    print(
        f"Production sample: week=2022-08-01 Taipei, hash partition=0, "
        f"rows={len(reference)}, ID range={first_id}..{last_id}, "
        f"articles scan budget={article_budget}, matches scan budget={match_budget}, "
        f"setup={time.perf_counter() - started:.3f}s"
    )
    try:
        yield corpus, reference, selected
    finally:
        memory.disconnect()


def saved_masks(reference, dictionary_id, catalog_entry):
    expected = pd.DataFrame(
        {
            "article_id": reference.article_id,
            "matches_any_category": reference.dictionary_ids.map(
                lambda identifiers: dictionary_id in identifiers
            ),
        }
    )
    for category in catalog_entry["categories"]:
        expected[f"is_{category['raw_name']}"] = reference.category_expression_ids.map(
            lambda identifiers: category["expression_id"] in identifiers
        )
    return expected.sort_values("article_id").reset_index(drop=True)


@pytest.mark.parametrize("name", DICTIONARIES)
def test_real_article_and_category_masks_match_saved_results(production_sample, name):
    corpus, reference, selected = production_sample
    dictionary_id, entry = selected[name]
    expected = saved_masks(reference, dictionary_id, entry)
    actual = (
        DictionaryMatcher(corpus.inventory.dictionaries[name])
        .label_binary(corpus.table, corpus.word_columns)
        .select(*expected.columns)
        .order_by("article_id")
        .to_pandas()
    )
    assert_frame_equal(actual, expected, check_exact=True)
    counts = expected.drop(columns="article_id").sum().to_dict()
    print(f"{name}: {len(expected)} exact article masks; saved counts={counts}")


@pytest.mark.parametrize("name", DICTIONARIES)
def test_real_calculator_counts_and_category_shares_match_saved_masks(
    production_sample, name
):
    corpus, reference, selected = production_sample
    dictionary_id, entry = selected[name]
    expected_masks = saved_masks(reference, dictionary_id, entry)
    expected_rows = reference[["article_id", "outlet"]].merge(
        expected_masks, on="article_id", validate="one_to_one"
    )
    expected_rows = expected_rows.loc[expected_rows.outlet.isin(TARGET_OUTLETS)]
    expected = []
    for outlet, group in expected_rows.groupby("outlet", sort=True):
        total = len(group)
        overall = int(group.matches_any_category.sum())
        row = {
            "outlet": outlet,
            "warc_date_bucketed": date(2022, 8, 1),
            "total_articles": total,
            "total_gpr_count": overall,
            "total_gpr_share": overall / (total + 1e-9),
        }
        for category in corpus.inventory.dictionaries[name]["categories"]:
            count = int(group[f"is_{category}"].sum())
            row[f"{category}_count"] = count
            row[f"{category}_share"] = count / (total + 1e-9)
        expected.append(row)
    assert len(expected) == 3
    actual = (
        corpus.three_outlets()
        .calculator([name])
        .query(name, group_by_cols=["outlet"], time_bucket="W")
        .to_dicts()
    )
    assert actual == expected
    print(
        f"{name}: exact calculator groups="
        f"{[(row['outlet'], row['total_articles'], row['total_gpr_count']) for row in actual]}"
    )
