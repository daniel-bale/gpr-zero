"""Checks that rebuilding cannot hide missing periods or failed adjustment."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rebuild_raw_economics.py"
SPEC = importlib.util.spec_from_file_location("raw_economics", SCRIPT)
rebuild = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rebuild)


def test_x13_output_requires_exact_months_and_finite_values():
    dates = pd.date_range("2020-01-01", periods=2, freq="MS")
    actual = rebuild.parse_d11("date value\n202001 4.2\n202002 4.3\n", dates)
    assert actual.to_list() == [4.2, 4.3]
    with pytest.raises(ValueError, match="dates/length"):
        rebuild.parse_d11("202001 4.2\n202003 4.3\n", dates)
    with pytest.raises(ValueError, match="dates/length"):
        rebuild.parse_d11("202001 4.2\n202002 4.3\n202003 4.4\n", dates)
    with pytest.raises(ValueError, match="non-finite"):
        rebuild.parse_d11("202001 4.2\n202002 nan\n", dates)


def test_quarterly_aggregation_retains_native_weekly_observations():
    # Boundary week is assigned by its date, with no daily interpolation.
    data = pl.DataFrame({"date": [pd.Timestamp("2020-03-30"), pd.Timestamp("2020-04-06"),
                                  pd.Timestamp("2020-04-13")], "gpr_index": [10.0, 20.0, 40.0]})
    result = rebuild.quarterly_gpr(data)
    assert result.to_dicts() == [{"quarter": "2020-Q1", "gpr_q": 10.0},
                                {"quarter": "2020-Q2", "gpr_q": 30.0}]
    with pytest.raises(ValueError, match="unique"):
        rebuild.quarterly_gpr(pl.concat([data, data.head(1)]))


def test_comparison_does_not_hide_missing_quarters_or_changed_nulls():
    cached = pl.DataFrame({"quarter": ["2020-Q1", "2020-Q2"], "x": [1.0, 2.0]})
    assert rebuild.compare_frames(cached.reverse(), cached)["all_values_match"]
    missing = rebuild.compare_frames(cached.head(1), cached)
    assert not missing["all_values_match"]
    assert missing["cached_only_keys"] == ["2020-Q2"]
    changed = pl.DataFrame({"quarter": ["2020-Q1", "2020-Q2"], "x": [1.0, None]})
    assert not rebuild.compare_frames(changed, cached)["all_values_match"]


def test_unadjusted_fallback_is_rejected_and_cwd_restored(tmp_path):
    # Simulate original load_labor catching an X13 error and returning raw data.
    module = SimpleNamespace(run_x13_with_holiday_regressors=lambda *a, **k: None,
                             subprocess=SimpleNamespace(),
                             load_labor=lambda: pl.DataFrame({"quarter": ["2020-Q1"], "log_hours_sa_q": [4.2]}))
    transport = rebuild.RemoteX13(tmp_path / "x13", "unused-host", "/unused", {})
    original_cwd = Path.cwd()
    with pytest.raises(RuntimeError, match="unadjusted fallback rejected"):
        rebuild.strict_labor(module, transport, tmp_path / "runtime")
    assert Path.cwd() == original_cwd


def test_monte_carlo_seeds_are_distinct_and_reproduce_original_single_draw():
    from statsmodels.tsa.api import VAR
    data = np.random.default_rng(19).normal(size=(60, 2))
    model = VAR(data).fit(2, trend="c")
    # Demonstrate the installed original multi-draw seeding behavior explicitly.
    repeated = model.irf_resim(orth=True, repl=2, steps=3, seed=123, burn=10)
    assert np.array_equal(repeated[0], repeated[1])
    seeds = rebuild.mc_seeds(20261021, 0, 0, 4)
    assert len(set(seeds)) == 4
    assert seeds == rebuild.mc_seeds(20261021, 0, 0, 4)
    assert set(seeds).isdisjoint(rebuild.mc_seeds(20261021, 0, 1, 4))
    draws, _ = rebuild.monte_carlo_draws(model, seeds, burn=10, horizon=3)
    original = model.irf_resim(orth=True, repl=1, steps=3, seed=seeds[0], burn=10)
    assert np.array_equal(draws[0], original[0])
    assert not np.array_equal(draws[0], draws[1])
    repeated_draws, _ = rebuild.monte_carlo_draws(model, seeds, burn=10, horizon=3)
    assert np.array_equal(draws, repeated_draws)


def test_monte_carlo_quantiles_keep_original_discrete_order_statistics():
    draws = np.arange(400, dtype=float)[:, None, None, None]
    lower, upper, indices = rebuild.original_mc_quantiles(draws, 0.32)
    assert indices == [63, 335]
    assert float(lower.item()) == 63 and float(upper.item()) == 335
    lower, upper, indices = rebuild.original_mc_quantiles(draws, 0.05)
    assert indices == [9, 389]
    assert float(lower.item()) == 9 and float(upper.item()) == 389


def test_final_var_rejects_benchmark_news_manifest(tmp_path):
    manifest = {"status": "complete", "complete": False, "input_complete": False, "source_complete": False}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="benchmark or partial exports rejected"):
        rebuild.require_complete_news(tmp_path)
    manifest.update(complete=True, input_complete=True, source_complete=True)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert rebuild.require_complete_news(tmp_path)["complete"] is True


def test_model_link_creation_and_reuse_resolve_to_exact_target(tmp_path):
    target = tmp_path / "rebuilt_quarterly"
    target.mkdir()
    link = tmp_path / "svar_preprocessing"
    rebuild.ensure_model_link(link, target)
    assert link.is_symlink() and link.resolve(strict=True) == target.resolve(strict=True)
    rebuild.ensure_model_link(link, target)
    assert link.resolve(strict=True) == target.resolve(strict=True)


@pytest.mark.parametrize("existing_kind", ["wrong_target", "dangling", "directory"])
def test_model_link_rejects_existing_incorrect_paths_without_replacing_them(tmp_path, existing_kind):
    expected = tmp_path / "rebuilt_quarterly"
    expected.mkdir()
    link = tmp_path / "svar_preprocessing"
    if existing_kind == "directory":
        link.mkdir()
    else:
        wrong = tmp_path / "stale_quarterly"
        if existing_kind == "wrong_target":
            wrong.mkdir()
        link.symlink_to(wrong, target_is_directory=True)
    with pytest.raises(ValueError, match="Existing model workspace"):
        rebuild.ensure_model_link(link, expected)
    if existing_kind == "directory":
        assert link.is_dir() and not link.is_symlink()
    else:
        assert link.is_symlink() and link.readlink() == wrong
