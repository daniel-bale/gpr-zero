# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "numpy==2.2.3", "pandas==2.2.3", "pyarrow==20.0.0",
#   "polars==1.28.1", "ibis-framework==10.3.1", "statsmodels==0.14.4",
#   "scipy==1.14.1", "matplotlib==3.10.1", "pytest==8.3.5",
# ]
# ///
"""Rebuild historical economic transformations from raw source snapshots.

uv run --no-project scripts/rebuild_raw_economics.py --economic-only
uv run --no-project scripts/rebuild_raw_economics.py --news-indices-dir PATH

The original loaders and model conventions are preserved. X13 runs on bdsrc in
an automatically removed temporary directory; no original remote file changes.
Cached quarterly/model outputs are used only after rebuilding, for comparison.
Any seasonal-adjustment failure is fatal, including the historical loader's
otherwise-silent fallback to the unadjusted series.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import shlex
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl


RAW_FILES = ["CPI.csv", "TAIEX_daily.parquet", "invest.csv", "vixtwn_macromicro.csv",
             "美麗島民調.csv", "employment.csv", "workhour.csv", "interest.csv"]
LOADERS = {"cpi_q": "load_cpi", "taiex_q": "load_taiex", "gfcf_q": "load_gfcf",
           "vix_q": "load_vix", "support_q": "load_party_support", "rate_q": "load_interest_rate"}
NEWS_FILES = {"taiwan_gpr_dict_president": "series_president_original7.parquet",
              "gpr_taiwan_step2_median": "series_step2_median.parquet"}

REMOTE_RUNNER = r'''
import hashlib, json, pathlib, subprocess, sys, tempfile
request = json.load(sys.stdin)
sha = lambda data: hashlib.sha256(data).hexdigest()
remote_hashes = {}
for name, expected in request["raw_hashes"].items():
    actual = sha((pathlib.Path(request["remote_repo"]) / "data/external" / name).read_bytes())
    if actual != expected:
        raise RuntimeError("Remote/local raw source differs: " + name)
    remote_hashes[name] = actual
binary = pathlib.Path(request["binary"])
spec = request["spec"]
if sha(spec.encode()) != request["spec_sha256"]:
    raise RuntimeError("Input spec transport hash mismatch")
with tempfile.TemporaryDirectory(prefix="gpr-raw-economics-") as work:
    path = pathlib.Path(work)
    (path / "labor_sa.spc").write_text(spec)
    result = subprocess.run([str(binary), "-i", str(path / "labor_sa")],
                            capture_output=True, text=True, cwd=path, timeout=180)
    files = {p.name: p.read_text(errors="strict") for p in path.iterdir() if p.is_file()}
    response = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                    files=files, spec_sha256=sha(spec.encode()), raw_hashes=remote_hashes,
                    binary_sha256=sha(binary.read_bytes()), binary=str(binary),
                    scratch_directory=work, scratch_removed_after_execution=True)
print(json.dumps(response))
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_d11(text: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Reject missing, duplicate, or misaligned X13 monthly output."""
    observations = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and re.fullmatch(r"\d{6}", fields[0]):
            observations.append((pd.to_datetime(fields[0], format="%Y%m"), float(fields[1])))
    index = pd.DatetimeIndex([row[0] for row in observations])
    if not index.as_unit("ns").equals(pd.DatetimeIndex(dates).as_unit("ns")):
        raise ValueError("X13 output dates/length differ from input; truncation/padding forbidden")
    values = np.array([row[1] for row in observations])
    if not np.isfinite(values).all():
        raise ValueError("X13 returned non-finite seasonal adjustments")
    return pd.Series(values, index=dates, name="seasadj")


def compare_frames(rebuilt: pl.DataFrame, cached: pl.DataFrame, key: str = "quarter") -> dict:
    """Align by explicit unique keys; report missing rows and per-column errors."""
    a, b = rebuilt.to_pandas().set_index(key), cached.to_pandas().set_index(key)
    if not a.index.is_unique or not b.index.is_unique:
        raise ValueError(f"Duplicate comparison key: {key}")
    common = a.index.intersection(b.index)
    extra, missing = a.index.difference(b.index), b.index.difference(a.index)
    column_match = set(a.columns) == set(b.columns)
    comparisons = {}
    for column in sorted(set(a.columns).intersection(b.columns)):
        av, bv = a.loc[common, column], b.loc[common, column]
        if pd.api.types.is_numeric_dtype(av) and pd.api.types.is_numeric_dtype(bv):
            null_equal = bool(np.array_equal(av.isna(), bv.isna()))
            valid = av.notna() & bv.notna()
            diff = np.abs(av[valid].to_numpy() - bv[valid].to_numpy())
            comparisons[column] = {"max_absolute_error": float(diff.max()) if len(diff) else None,
                                   "null_pattern_matches": null_equal,
                                   "matches": null_equal and bool(np.allclose(av[valid], bv[valid], atol=1e-10, rtol=1e-10))}
        else:
            comparisons[column] = {"matches": bool(av.equals(bv))}
    return {"rebuilt_rows": len(a), "cached_rows": len(b), "matched_keys": len(common),
            "rebuilt_start": str(a.index.min()), "rebuilt_end": str(a.index.max()),
            "cached_start": str(b.index.min()), "cached_end": str(b.index.max()),
            "rebuilt_only_keys": [str(x) for x in extra], "cached_only_keys": [str(x) for x in missing],
            "column_names_match": column_match, "columns": comparisons,
            "all_values_match": column_match and len(extra) == 0 and len(missing) == 0 and all(v["matches"] for v in comparisons.values())}


class RemoteX13:
    def __init__(self, output: Path, host: str, remote_repo: str, raw_hashes: dict):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.host, self.remote_repo, self.raw_hashes = host, remote_repo, raw_hashes
        self.response = None
        self.dates = None
        self.adjusted = None
        self.completed = False

    def run(self, cmd, capture_output, text, cwd):
        """Subprocess-compatible adapter for the unchanged original X13 helper."""
        spec = Path(cmd[2] + ".spc").read_text()
        request = {"spec": spec, "spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
                   "raw_hashes": self.raw_hashes, "remote_repo": self.remote_repo,
                   "binary": self.remote_repo + "/x13as/x13as_ascii"}
        (self.output / "labor_sa.spc").write_text(spec)
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", self.host,
                                 "python3 -c " + shlex.quote(REMOTE_RUNNER)],
                                input=json.dumps(request), capture_output=True, text=True, timeout=240)
        if result.returncode:
            raise RuntimeError("Remote X13 transport/execution failed: " + result.stderr)
        response = json.loads(result.stdout)
        self.response = response
        if response["returncode"] != 0 or "labor_sa.d11" not in response["files"]:
            raise RuntimeError("X13 failed or did not emit d11: " + response["stdout"] + response["stderr"])
        if response["spec_sha256"] != request["spec_sha256"] or response["raw_hashes"] != self.raw_hashes:
            raise RuntimeError("Remote source/spec integrity check failed")
        self.adjusted = parse_d11(response["files"]["labor_sa.d11"], self.dates)
        for name, content in response["files"].items():
            if Path(name).name != name:
                raise RuntimeError("Unexpected remote output filename")
            (Path(cwd) / name).write_text(content)
            (self.output / name).write_text(content)
        (self.output / "execution_stdout.txt").write_text(response["stdout"])
        (self.output / "execution_stderr.txt").write_text(response["stderr"])
        metadata = {k: v for k, v in response.items() if k not in ["files", "stdout", "stderr"]}
        metadata["host"] = self.host
        metadata["file_sha256"] = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in response["files"].items()}
        (self.output / "execution_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.completed = True
        return subprocess.CompletedProcess(cmd, response["returncode"], response["stdout"], response["stderr"])


def strict_labor(module, transport: RemoteX13, runtime: Path) -> pl.DataFrame:
    """Keep original labor construction but reject its unadjusted fallback."""
    original_x13, original_subprocess = module.run_x13_with_holiday_regressors, module.subprocess
    original_helper_returned = False

    def capture_inputs(data, regressors, x13_path):
        nonlocal original_helper_returned
        if not np.isfinite(data).all() or not np.isfinite(regressors).all().all():
            raise ValueError("Non-finite X13 input")
        expected = pd.date_range(data.index.min(), data.index.max(), freq="MS")
        if not expected.equals(pd.DatetimeIndex(data.index).as_unit("ns")) or not data.index.equals(regressors.index):
            raise ValueError("Missing or misaligned monthly labor data")
        transport.dates = pd.DatetimeIndex(data.index).as_unit("ns")
        data.to_csv(transport.output / "monthly_log_hours_input.csv")
        regressors.to_csv(transport.output / "monthly_cny_regressors.csv")
        adjusted = original_x13(data, regressors, x13_path)
        original_helper_returned = True
        return adjusted

    runtime.mkdir(parents=True, exist_ok=True)
    placeholder = runtime / "x13as/x13as_ascii"
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text("Remote execution adapter marker; not an executable.\n")
    previous_cwd = Path.cwd()
    module.run_x13_with_holiday_regressors = capture_inputs
    module.subprocess = SimpleNamespace(run=transport.run)
    try:
        os.chdir(runtime)
        labor = module.load_labor()
    finally:
        os.chdir(previous_cwd)
        module.run_x13_with_holiday_regressors = original_x13
        module.subprocess = original_subprocess
    if not transport.completed or transport.adjusted is None or not original_helper_returned:
        raise RuntimeError("Seasonal adjustment failed; historical unadjusted fallback rejected")
    adjusted = transport.adjusted.rename_axis("date").rename("log_hours_sa_m").reset_index()
    adjusted.to_csv(transport.output / "monthly_log_hours_seasonally_adjusted.csv", index=False)
    expected = (pl.from_pandas(adjusted).group_by_dynamic("date", every="1q")
                .agg(pl.mean("log_hours_sa_m").alias("log_hours_sa_q"))
                .with_columns(pl.col("date").dt.to_string("%Y-Q%q").alias("quarter"))
                .select("quarter", "log_hours_sa_q"))
    if not compare_frames(labor, expected)["all_values_match"]:
        raise RuntimeError("Labor loader did not return verified X13-adjusted quarterly data")
    return labor


def quarterly_gpr(series: pl.DataFrame) -> pl.DataFrame:
    if not {"date", "gpr_index"}.issubset(series.columns):
        raise ValueError("Rebuilt index requires date and gpr_index")
    if series["date"].n_unique() != series.height or series["date"].null_count():
        raise ValueError("Rebuilt index dates must be unique and non-null")
    if series["gpr_index"].null_count() or not series["gpr_index"].is_finite().all():
        raise ValueError("Rebuilt index must be finite and non-null")
    return (series.sort("date").group_by_dynamic("date", every="1q")
            .agg(pl.mean("gpr_index").alias("gpr_q"))
            .with_columns(pl.col("date").dt.to_string("%Y-Q%q").alias("quarter"))
            .select("quarter", "gpr_q"))


def require_complete_news(indices: Path) -> dict:
    path = indices / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or any(manifest.get(key) is not True
            for key in ["complete", "input_complete", "source_complete"]):
        raise ValueError("Final economics/VAR run requires completed full-corpus news inputs; benchmark or partial exports rejected")
    return manifest


def ensure_model_link(link: Path, target: Path) -> None:
    """Reject stale workspace paths instead of fitting/reporting different inputs."""
    expected = target.resolve(strict=True)
    if not expected.is_dir():
        raise ValueError(f"Model workspace target is not a directory: {expected}")
    if link.is_symlink() or link.exists():
        if not link.is_symlink():
            raise ValueError(f"Existing model workspace path is not the expected symlink: {link}")
        try:
            actual = link.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise ValueError(f"Existing model workspace symlink is unresolved: {link}") from exc
        if actual != expected:
            raise ValueError(f"Existing model workspace symlink targets {actual}, expected {expected}: {link}")
    else:
        link.symlink_to(expected, target_is_directory=True)


def plot_raw_models(frame: pd.DataFrame, report: dict, output: Path, model_module) -> None:
    """Use the same plots but with the correct raw-rebuild provenance caption."""
    import matplotlib.pyplot as plt
    for corrected in (False, True):
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True)
        for i, ax in enumerate(axes.flat):
            if i == len(model_module.VARIABLES):
                ax.axis("off")
                ax.text(0, 0.95,
                        f"{report['nobs']} observations; {report['residual_degrees_of_freedom']} residual df\n"
                        f"Stable: {report['stable']}\n"
                        f"Companion radius: {report['companion_spectral_radius']:.6f}\n\n"
                        "Economic series rebuilt from raw snapshots.\n"
                        "News indices rebuilt from article matching.\n"
                        "Historical outputs used for comparison only.\n"
                        "Confidence bands are in a separate figure.",
                        transform=ax.transAxes, va="top", fontsize=10)
                continue
            d = frame.loc[frame["response"] == model_module.VARIABLES[i]]
            if corrected:
                ax.plot(d["period"], d["legacy_shock_consistent_display"], "--", color="#986900", label="Legacy shock")
                ax.plot(d["period"], d["two_sd_consistent_display"], color="#16656b", linewidth=2, label="Two-SD shock")
                units = model_module.CORRECTED_UNITS
            else:
                ax.plot(d["period"], d["legacy_refit"], color="#16656b", linewidth=2, label="Raw-data rebuild")
                ax.plot(d["period"], d["historical_cached"], linestyle="none", marker="o", markerfacecolor="none",
                        markersize=4, color="#986900", label="Historical cache")
                units = model_module.LEGACY_UNITS
            ax.axhline(0, color="0.5", linewidth=0.7)
            ax.set(title=model_module.TITLES[i], ylabel=units[i], xlabel="Quarters after shock")
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.grid(alpha=0.2)
            if i == 0:
                ax.legend(fontsize=8)
        title = "Shock normalization diagnostic" if corrected else "Historical point-response comparison"
        fig.suptitle(f"{model_module.VERSIONS[report['version']]['label']} · VAR(2) from rebuilt inputs\n{title}", fontsize=14)
        fig.text(0.02, 0.025,
                 "Recursive ordering and legacy specification retained. Log responses are log-point approximations. Small-sample causal interpretation requires additional assumptions.",
                 fontsize=9)
        fig.tight_layout(rect=[0, 0.06, 1, 0.9])
        suffix = "shock_normalization" if corrected else "historical_replication"
        fig.savefig(output / f"{report['version']}_{suffix}.png", dpi=180)
        plt.close(fig)


def mc_seeds(master_seed: int, model_number: int, band_number: int, count: int) -> list[int]:
    children = np.random.SeedSequence([master_seed, model_number, band_number]).spawn(count)
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    if len(set(seeds)) != count:
        raise RuntimeError("Derived Monte Carlo seeds collided")
    return seeds


def monte_carlo_draws(results, seeds: list[int], burn: int = 80, horizon: int = 20):
    """Same simulation/refit as statsmodels irf_resim, unique seed per draw."""
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.vector_ar.util import varsim
    draws, unstable = [], 0
    for seed in seeds:
        simulated = varsim(results.coefs, results.intercept, results.sigma_u,
                           steps=results.nobs + results.k_ar + burn, seed=seed)[burn:]
        fitted = VAR(simulated, exog=results.exog).fit(maxlags=results.k_ar, trend=results.trend)
        draws.append(fitted.orth_ma_rep(maxn=horizon))
        unstable += int(not fitted.is_stable())
    draws = np.array(draws)
    if not np.isfinite(draws).all():
        raise RuntimeError("Non-finite Monte Carlo response")
    if np.unique(draws.reshape(len(draws), -1), axis=0).shape[0] != len(seeds):
        raise RuntimeError("Monte Carlo draws are not distinct")
    return draws, unstable


def original_mc_quantiles(draws: np.ndarray, signif: float):
    """Preserve statsmodels' discrete order-statistic endpoints (no interpolation)."""
    count = len(draws)
    low_index = int(round(signif / 2 * count) - 1)
    high_index = int(round((1 - signif / 2) * count) - 1)
    sorted_draws = np.sort(draws, axis=0)
    return sorted_draws[low_index], sorted_draws[high_index], [low_index, high_index]


def regenerate_uncertainty(version: str, model_number: int, quarterly: Path, root: Path,
                           output: Path, point_frame: pd.DataFrame, model_module, master_seed: int) -> dict:
    from statsmodels.tsa.api import VAR
    import matplotlib.pyplot as plt
    data = pd.read_parquet(quarterly / f"var_ready_{version}.parquet")
    results = VAR(data[model_module.VARIABLES]).fit(2, trend="c")
    sigma = float(np.sqrt(results.sigma_u.iloc[0, 0]))
    cache_path = root / "data_analysis/output/svar_analysis" / version / model_module.VERSIONS[version]["cache_subdir"] / "svar_irf_data.parquet"
    cached = pd.read_parquet(cache_path).set_index(["response", "impulse", "period"])
    for level in [68, 95]:
        bounds = cached[[f"ci_{level}_lower", f"ci_{level}_upper"]].to_numpy()
        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Historical uncertainty endpoints are non-finite or unordered")
    metadata = {"master_seed": master_seed, "repl_per_confidence_level": 400, "burn": 80,
                "orth": True, "horizons": [0, 20], "levels": {},
                "seed_method": "SeedSequence([master_seed, model_number, band_number]).spawn(400), each child generates a distinct uint32 seed for one RandomState-based varsim call.",
                "seeding_reason": "statsmodels 0.14.4 irf_resim passes seed unchanged to varsim for each draw; supplying one fixed seed to the original multi-replication method repeats identical samples.",
                "comparison_limit": "Original cache used unseeded simulations. Exact confidence endpoints are not expected to reproduce; differences can reflect input differences as well as Monte Carlo randomness. Endpoint errors and zero-exclusion agreement are descriptive.",
                "unstable_simulation_policy": "Retain all simulated refits, including unstable ones, exactly as the original method does."}
    rows = []
    for band_number, (level, signif) in enumerate([(68, 0.32), (95, 0.05)]):
        seeds = mc_seeds(master_seed, model_number, band_number, 400)
        draws, unstable = monte_carlo_draws(results, seeds)
        lower, upper, indices = original_mc_quantiles(draws, signif)
        # One-draw identity verifies equivalence to the original implementation.
        builtin = results.irf_resim(orth=True, repl=1, steps=20, seed=seeds[0], burn=80)
        builtin_matches = bool(np.array_equal(builtin[0], draws[0]))
        if not builtin_matches:
            raise RuntimeError("Custom Monte Carlo draw differs from original statsmodels algorithm")
        metadata["levels"][str(level)] = {"signif": signif, "seed_namespace": [master_seed, model_number, band_number],
            "seeds": seeds, "unique_seeds": len(set(seeds)), "unique_draws": 400,
            "unstable_simulated_refits": unstable, "order_statistic_indices_zero_based": indices,
            "one_draw_matches_statsmodels_exactly": builtin_matches,
            "draw_array_sha256": hashlib.sha256(draws.tobytes()).hexdigest()}
        np.savez_compressed(output / f"{version}_mc{level}_draws.npz", orthogonal_irfs=draws, seeds=np.array(seeds, dtype=np.uint32))
        for i, variable in enumerate(model_module.VARIABLES):
            for h in range(21):
                scale = 2 * sigma * model_module.LEGACY_MULTIPLIERS[i]
                low, high = float(lower[h, i, 0] * scale), float(upper[h, i, 0] * scale)
                old = cached.loc[(variable, "log_GPR", h)]
                old_low, old_high = float(old[f"ci_{level}_lower"]), float(old[f"ci_{level}_upper"])
                signal = lambda lo, hi: "positive" if lo > 0 else ("negative" if hi < 0 else "includes_zero")
                rows.append({"version": version, "response": variable, "impulse": "log_GPR", "period": h,
                             "confidence_level": level, "rebuilt_lower_legacy": low, "rebuilt_upper_legacy": high,
                             "cached_lower_legacy": old_low, "cached_upper_legacy": old_high,
                             "lower_absolute_error": abs(low - old_low), "upper_absolute_error": abs(high - old_high),
                             "rebuilt_signal": signal(low, high), "cached_signal": signal(old_low, old_high),
                             "signal_matches": signal(low, high) == signal(old_low, old_high),
                             "two_sd_lower_consistent_display": float(lower[h, i, 0] * 2 * model_module.CORRECTED_MULTIPLIERS[i]),
                             "two_sd_upper_consistent_display": float(upper[h, i, 0] * 2 * model_module.CORRECTED_MULTIPLIERS[i])})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{version}_uncertainty_comparison.csv", index=False)
    for level in [68, 95]:
        subset = frame.loc[frame["confidence_level"] == level]
        metadata["levels"][str(level)].update({
            "max_lower_endpoint_absolute_error": float(subset["lower_absolute_error"].max()),
            "max_upper_endpoint_absolute_error": float(subset["upper_absolute_error"].max()),
            "significance_pattern_agreement": int(subset["signal_matches"].sum()),
            "significance_comparisons": len(subset),
            "rebuilt_zero_excluded_count": int((subset["rebuilt_signal"] != "includes_zero").sum()),
            "cached_zero_excluded_count": int((subset["cached_signal"] != "includes_zero").sum()),
            "significance_pattern_differences": subset.loc[~subset["signal_matches"], ["response", "period", "rebuilt_signal", "cached_signal"]].to_dict("records"),
        })
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True)
    for i, ax in enumerate(axes.flat):
        if i == 7:
            ax.axis("off")
            ax.text(0, 0.95, "Fresh raw-data inputs; original VAR(2).\n400 distinct simulations per CI level.\n80 burn-in observations; orthogonal IRFs.\nDeterministic seeds saved with results.\n\nOriginal cached bands were unseeded.\nEndpoint differences are expected.\nLegacy shock/display scaling retained.",
                    transform=ax.transAxes, va="top", fontsize=10)
            continue
        subset = frame.loc[frame["response"] == model_module.VARIABLES[i]]
        for level, color in [(95, "#cce3ea"), (68, "#699cac")]:
            bands = subset.loc[subset["confidence_level"] == level]
            ax.fill_between(bands["period"], bands["rebuilt_lower_legacy"], bands["rebuilt_upper_legacy"],
                            color=color, alpha=0.8, label=f"Regenerated {level}% CI")
        point = point_frame.loc[point_frame["response"] == model_module.VARIABLES[i]]
        ax.plot(point["period"], point["legacy_refit"], color="black", linewidth=1.5, label="Point response")
        ax.axhline(0, color="0.4", linewidth=0.7)
        ax.set(title=model_module.TITLES[i], ylabel=model_module.LEGACY_UNITS[i], xlabel="Quarters after shock")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(alpha=0.15)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"{model_module.VERSIONS[version]['label']} · regenerated Monte Carlo uncertainty", fontsize=14)
    fig.text(0.02, 0.025, "Recursive identification, 29 observations and 14 residual degrees of freedom. Pointwise intervals; all simulated refits retained as in the original algorithm.", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(output / f"{version}_regenerated_uncertainty.png", dpi=180)
    plt.close(fig)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--news-indices-dir", type=Path)
    parser.add_argument("--economic-only", action="store_true")
    parser.add_argument("--x13-host", default="bdsrc")
    parser.add_argument("--remote-repo", default="/home5/chuenho/gpr-zero")
    parser.add_argument("--mc-seed", type=int, default=20261021)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output.resolve() if args.output else root / "replication/raw_rebuild/economics"
    indices = args.news_indices_dir.resolve() if args.news_indices_dir else root / "replication/raw_rebuild/indices"
    news_manifest = None if args.economic_only else require_complete_news(indices)
    quarterly = output / "quarterly"
    quarterly.mkdir(parents=True, exist_ok=True)
    source = root / "data_preprocessing/build_master_df.py"
    raw_hashes = {name: sha256(root / "data/external" / name) for name in RAW_FILES}
    # Its import creates a relative OUTPUT_DIR before constants can be overridden.
    # Contain that side effect within this rebuild's isolated runtime directory.
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(runtime)
        module = import_file(source, "historical_economic_loaders")
    finally:
        os.chdir(previous_cwd)
    module.DATA_DIR = root / "data/external"
    module.OUTPUT_DIR = quarterly
    module.MASTER_DF_PATH = quarterly / "master_quarterly_df.parquet"
    components = {}
    reference = root / "data_analysis/output/svar_preprocessing"
    for name, loader in LOADERS.items():
        frame = getattr(module, loader)()
        frame.write_parquet(quarterly / f"{name}.parquet")
        frame.write_csv(quarterly / f"{name}.csv")
        components[name] = compare_frames(frame, pl.read_parquet(reference / f"{name}.parquet"))
        components[name].update(rebuilt_sha256=sha256(quarterly / f"{name}.parquet"),
                                reference_sha256=sha256(reference / f"{name}.parquet"))
    transport = RemoteX13(output / "x13", args.x13_host, args.remote_repo, raw_hashes)
    labor = strict_labor(module, transport, runtime)
    labor.write_parquet(quarterly / "labor_q.parquet")
    labor.write_csv(quarterly / "labor_q.csv")
    components["labor_q"] = compare_frames(labor, pl.read_parquet(reference / "labor_q.parquet"))
    components["labor_q"].update(rebuilt_sha256=sha256(quarterly / "labor_q.parquet"),
                                 reference_sha256=sha256(reference / "labor_q.parquet"))
    report = {
        "scope": "Original economic transformations rerun from historical raw snapshots; X13 recomputed remotely from locally generated spec, with no cached seasonal-adjustment inputs.",
        "source_file": str(source), "source_sha256": sha256(source), "raw_source_sha256": raw_hashes,
        "rebuild_script_sha256": sha256(Path(__file__)),
        "model_helper_sha256": sha256(root / "scripts/replicate_thesis_var.py"),
        "software": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "polars", "pyarrow", "statsmodels", "scipy", "matplotlib", "ibis-framework"]},
        "python": platform.python_version(),
        "component_comparisons": components,
        "all_economic_components_match": all(item["all_values_match"] for item in components.values()),
        "x13": {"host": args.x13_host, "binary_sha256": transport.response["binary_sha256"],
                "spec_sha256": transport.response["spec_sha256"], "raw_remote_hashes_verified": True,
                "input_months": len(transport.dates), "input_start": str(transport.dates.min().date()),
                "input_end": str(transport.dates.max().date()), "unadjusted_fallback_allowed": False,
                "warnings": [line.strip() for line in transport.response["stdout"].splitlines() if "WARNING" in line]},
        "transformations": {
            "quarter_aggregation": "VIX, CPI, TAIEX, support, adjusted log hours: arithmetic means; policy rate: quarter-end forward-filled level; investment: source quarterly nominal amount.",
            "labor": "log(employment * monthly average hours / working_age_population), monthly X13 automdl/outlier with CNY current/lag/lead holiday regressors, then quarterly mean.",
            "assembly": "Full outer joins by quarter; CPI deflation before logs; log1p(native-frequency quarterly GPR mean); filter 2017Q1-2024Q4 before differencing investment, TAIEX, policy rate; remove 2017Q1.",
            "legacy_details_preserved": "Historical investment loader multiplies nominal column7 by 1e9 despite its 1e8 comment; constant scale cancels from log growth. CNY lag/lead names/comments retained as implemented.",
        },
        "shock_normalization": {
            "orthogonal_irf": "Theta_h = Phi_h P, where P P' = Sigma_u and P[0,0] = sigma_GPR.",
            "legacy": "2 * sigma_GPR * Theta_h[:,0]; GPR own impact is 2 * sigma_GPR^2.",
            "two_residual_sd": "2 * Theta_h[:,0]; GPR own impact is 2 * sigma_GPR.",
            "display": "Historical plot-unit multipliers preserved for cache comparison; separate corrected-display curves multiply all log-based responses by 100.",
        },
        "models": [], "news_series": {}, "economic_only": args.economic_only,
    }
    if not args.economic_only:
        report["news_manifest"] = {"file": str(indices / "manifest.json"), "sha256": sha256(indices / "manifest.json"),
            **{key: news_manifest[key] for key in ["complete", "input_complete", "source_complete", "economics_series", "corpus", "dependencies"]}}
        model_workspace = output / "model_workspace/data_analysis/output"
        model_workspace.mkdir(parents=True, exist_ok=True)
        for link, target in [(model_workspace / "svar_preprocessing", quarterly),
                             (model_workspace / "svar_analysis", root / "data_analysis/output/svar_analysis")]:
            ensure_model_link(link, target)
        model_module = import_file(root / "scripts/replicate_thesis_var.py", "replicate_model_from_rebuilt_inputs")
        model_module.plot_model = lambda frame, report, destination: plot_raw_models(frame, report, destination, model_module)
        model_output = output / "var"
        model_output.mkdir(exist_ok=True)
        frames = []
        for model_number, (version, filename) in enumerate(NEWS_FILES.items()):
            path = indices / filename
            series = pl.read_parquet(path)
            if series.height != news_manifest["outputs"][path.stem]["rows"]:
                raise ValueError(f"News export row count differs from complete manifest: {filename}")
            gpr = quarterly_gpr(series)
            gpr.write_parquet(quarterly / f"gpr_q_{version}.parquet")
            gpr.write_csv(quarterly / f"gpr_q_{version}.csv")
            report["news_series"][version] = {"file": str(path), "sha256": sha256(path),
                "rows": series.height, "date_start": str(series["date"].min()), "date_end": str(series["date"].max()),
                "gpr_index_mean": float(series["gpr_index"].mean()),
                "gpr_index_min": float(series["gpr_index"].min()),
                "gpr_index_max": float(series["gpr_index"].max()),
                "aggregation": "Native-frequency row mean by calendar quarter; no interpolation or renormalization.",
                "date_gap_days": series.sort("date")["date"].diff().dt.total_days().drop_nulls().value_counts().to_dicts(),
                "quarterly_comparison": compare_frames(gpr, pl.read_parquet(reference / f"gpr_q_{version}.parquet"))}
            module.assemble_master_df(version)
            for kind in ["master_quarterly", "var_ready"]:
                name = f"{kind}_{version}"
                rebuilt = pl.read_parquet(quarterly / f"{name}.parquet")
                rebuilt.write_csv(quarterly / f"{name}.csv")
                components[name] = compare_frames(rebuilt, pl.read_parquet(reference / f"{name}.parquet"))
                components[name].update(rebuilt_sha256=sha256(quarterly / f"{name}.parquet"),
                                        reference_sha256=sha256(reference / f"{name}.parquet"))
            model_report, frame = model_module.replicate(version, output / "model_workspace", model_output)
            model_report["input_file"] = str(quarterly / f"var_ready_{version}.parquet")
            model_report["cautions"][0] = "Economic inputs were rebuilt from raw source snapshots and X13; news index inputs were rebuilt separately from article matching. Historical outputs are comparison references only."
            model_report["cautions"] = [c for c in model_report["cautions"] if "unseeded" not in c]
            model_report["cautions"].append("Monte Carlo bands are regenerated with deterministic distinct simulations; original unseeded endpoints cannot be exactly reproduced.")
            model_report["regenerated_uncertainty"] = regenerate_uncertainty(version, model_number, quarterly, root,
                                                                           model_output, frame, model_module, args.mc_seed)
            report["models"].append(model_report)
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(model_output / "all_irf_comparisons.csv", index=False)
    report["raw_source_files_unchanged"] = raw_hashes == {name: sha256(root / "data/external" / name) for name in RAW_FILES}
    if not report["raw_source_files_unchanged"]:
        raise RuntimeError("Raw source file changed during rebuild")
    (output / "rebuild_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "all_economic_components_match": report["all_economic_components_match"],
                      "economic_only": args.economic_only, "components": components}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
