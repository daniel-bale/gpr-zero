"""The cache vocabulary stays fixed when dictionaries and cohorts are selected."""

import json

import ibis
import pandas as pd
import pytest

from gpr_zero.constants import BINARY_MATRIX_TABLE, TARGET_OUTLETS
from gpr_zero.gpr_index.corpus import ArticleCorpus, DictionaryInventory


@pytest.fixture
def inventory(tmp_path):
    definitions = {
        "step1_early": {
            "word_sets": {"early": ["apple"]},
            "categories": {"early": "early"},
        },
        "step2_late": {
            "word_sets": {"late": ["zebra"]},
            "categories": {"late": "late"},
        },
    }
    for name, definition in definitions.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(definition))
    return DictionaryInventory.from_directory(tmp_path)


@pytest.fixture
def articles():
    return pd.DataFrame(
        {
            "article_id": [1, 2, 3, 4, 5],
            "content": ["apple", "zebra", None, "zebra", "zebra"],
            "outlet": [
                TARGET_OUTLETS[0],
                TARGET_OUTLETS[0],
                TARGET_OUTLETS[1],
                "Other",
                TARGET_OUTLETS[2],
            ],
            "source_warc_file": ["fixture.warc"] * 5,
            "warc_date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2016-12-31",
            ],
        }
    )


@pytest.fixture
def connection(articles):
    connection = ibis.duckdb.connect()
    connection.create_table("articles", articles)
    try:
        yield connection
    finally:
        connection.disconnect()


def test_selection_keeps_full_vocabulary_and_does_not_reload_files(
    connection, inventory
):
    corpus = ArticleCorpus.rebuild(connection, inventory).three_outlets().study_period()
    for path in inventory.dictionary_paths:
        from pathlib import Path

        Path(path).unlink()
    calculator = corpus.calculator(inventory.select("step2_"))
    assert calculator.word_column_mapping == {
        "apple": "contains_word_0000",
        "zebra": "contains_word_0001",
    }
    result = calculator.query("step2_late", [], time_bucket="W")
    assert result["total_articles"].to_list() == [3]
    assert result["total_gpr_count"].to_list() == [1]
    assert set(calculator.dictionaries) == {"step2_late"}


def test_manifest_rejects_same_width_different_vocabulary(
    connection, inventory, tmp_path
):
    ArticleCorpus.rebuild(connection, inventory)
    altered = tmp_path / "altered"
    altered.mkdir()
    (altered / "d.json").write_text(
        json.dumps(
            {
                "word_sets": {"other": ["apple", "yak"]},
                "categories": {"other": "other"},
            }
        )
    )
    with pytest.raises(
        ValueError, match="does not match the full dictionary inventory"
    ):
        ArticleCorpus.from_cache(
            connection, DictionaryInventory.from_directory(altered)
        )


def test_manifest_rejects_subset_inventory(connection, inventory):
    ArticleCorpus.rebuild(connection, inventory)
    subset = DictionaryInventory.from_paths([inventory.dictionary_paths[1]])
    with pytest.raises(ValueError, match="full dictionary inventory"):
        ArticleCorpus.from_cache(connection, subset)


def test_historical_unmanifested_cache_is_read_without_rewriting(connection, inventory):
    ArticleCorpus.rebuild(connection, inventory)
    connection.drop_table(f"{BINARY_MATRIX_TABLE}_vocabulary")
    tables_before = connection.list_tables()
    loaded = ArticleCorpus.from_cache(connection, inventory)
    assert loaded.table.count().execute() == 5
    assert connection.list_tables() == tables_before


def test_legacy_step2_ordinals_remain_explicit(connection, inventory, tmp_path):
    ArticleCorpus.rebuild(connection, inventory)
    step2 = tmp_path / "step2"
    step2.mkdir()
    (step2 / "step2_late.json").write_text(
        json.dumps(inventory.dictionaries["step2_late"])
    )
    historical = DictionaryInventory.thesis_step2_ordinal(step2)
    corpus = ArticleCorpus.from_cache(connection, historical)
    assert corpus.word_columns == {"zebra": "contains_word_0000"}
    with pytest.raises(ValueError, match="cannot build"):
        ArticleCorpus.rebuild(connection, historical)


def test_newspaper_cache_keeps_historical_preference(connection, inventory):
    ArticleCorpus.rebuild(connection, inventory)
    historic = "articles_word_presence_step2_dictionaries_W"
    connection.create_table(historic, connection.table(BINARY_MATRIX_TABLE).limit(1))
    assert (
        ArticleCorpus.newspaper_spike_cache(connection, inventory)
        .table.count()
        .execute()
        == 1
    )
    connection.drop_table(historic)
    assert (
        ArticleCorpus.newspaper_spike_cache(connection, inventory)
        .table.count()
        .execute()
        == 5
    )


def test_metadata_loading_contract_is_explicit(tmp_path):
    (tmp_path / "step2_dict.json").write_text(
        json.dumps({"word_sets": {"a": ["apple"]}, "categories": {}})
    )
    (tmp_path / "run_meta.json").write_text(
        json.dumps({"word_sets": {"b": ["banana"]}})
    )
    assert DictionaryInventory.from_directory(tmp_path).words == ("apple",)
    assert DictionaryInventory.thesis_directory(tmp_path).words == ("apple", "banana")


def test_failed_manifest_write_rolls_back_the_cache(connection, inventory, monkeypatch):
    ArticleCorpus.rebuild(connection, inventory)
    connection.con.execute("UPDATE articles SET content = 'changed'")
    create_table = connection.create_table

    def fail_manifest(name, *args, **kwargs):
        if name.endswith("_vocabulary"):
            raise RuntimeError("manifest write failed")
        return create_table(name, *args, **kwargs)

    monkeypatch.setattr(connection, "create_table", fail_manifest)
    with pytest.raises(RuntimeError, match="manifest write failed"):
        ArticleCorpus.rebuild(connection, inventory)
    preserved = ArticleCorpus.from_cache(connection, inventory)
    assert preserved.table.contains_word_0001.sum().execute() == 3


@pytest.mark.parametrize(
    "invalid_text",
    [
        '{"word_sets": {"noise": ["aaa"]}}',
        '{"categories": {"noise": "noise"}}',
        "[]",
        "{broken JSON",
    ],
)
def test_canonical_inventory_skips_invalid_files_like_calculator(
    tmp_path, invalid_text, caplog
):
    from gpr_zero.gpr_index.calculator import GPRCalculator

    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps({"word_sets": {"risk": ["risk"]}, "categories": {"risk": "risk"}})
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_text(invalid_text)
    paths = [str(invalid), str(tmp_path / "absent.json"), str(valid)]
    legacy = GPRCalculator(paths)
    inventory = DictionaryInventory.from_paths(paths)
    current = GPRCalculator(inventory)
    assert inventory.words == ("risk",)
    assert inventory.word_columns == {"risk": "contains_word_0000"}
    assert current.dictionaries == legacy.dictionaries
    assert current.all_words == legacy.all_words
    assert inventory.dictionary_paths == (str(valid),)
    assert "Failed to load dictionary" in caplog.text


def test_canonical_inventory_raises_when_no_valid_dictionary_remains(tmp_path):
    partial = tmp_path / "partial.json"
    partial.write_text('{"word_sets": {"noise": ["aaa"]}}')
    with pytest.raises(ValueError, match="No valid dictionaries were loaded"):
        DictionaryInventory.from_directory(tmp_path)
    assert DictionaryInventory.thesis_directory(tmp_path).words == ("aaa",)
    assert DictionaryInventory.thesis_category_directory(tmp_path).words == ("aaa",)


def test_metadata_name_does_not_override_calculator_validity(tmp_path):
    valid = {"word_sets": {"risk": ["risk"]}, "categories": {"risk": "risk"}}
    partial = {"word_sets": {"noise": ["aaa"]}}
    (tmp_path / "valid_meta.json").write_text(json.dumps(valid))
    (tmp_path / "partial.json").write_text(json.dumps(partial))
    assert DictionaryInventory.from_directory(tmp_path).words == ("risk",)
    assert DictionaryInventory.thesis_directory(tmp_path).words == ("aaa", "risk")
    assert DictionaryInventory.thesis_category_directory(tmp_path).words == ("aaa",)
