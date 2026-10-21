# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "numpy==2.2.3",
#   "pandas==2.2.3",
#   "pyarrow==20.0.0",
#   "statsmodels==0.14.4",
#   "matplotlib==3.10.1",
#   "scipy==1.14.1",
# ]
# ///
"""Refit both thesis VARs from recovered, already preprocessed quarterly data.

Run from the repository root with:
    UV_CACHE_DIR=/private/tmp/gpr-zero-uv-cache \
    UV_PYTHON_INSTALL_DIR=/private/tmp/gpr-zero-uv-python \
    MPLCONFIGDIR=/private/tmp/gpr-zero-mpl \
    uv run --no-project scripts/replicate_thesis_var.py

The original calculation and historical caches are read-only. This reproduces
point estimates, not the original unseeded Monte Carlo uncertainty bands, and
does not claim independent replication of upstream economic-data preprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

VARIABLES = [
    "log_GPR", "VIX_lvl", "Δlog_GFCF_real", "log_Hours_SA",
    "Δlog_TAIEX_real", "Support_pct", "ΔRate_pct",
]
TITLES = [
    "GPR (log level)", "VIX", "Investment growth", "Hours (log level)",
    "TAIEX return", "Party support", "Policy-rate change",
]
# Faithfully preserve the legacy plotting transformation. The two delta-log
# series were NOT multiplied by 100 in the original code despite their labels.
LEGACY_MULTIPLIERS = np.array([100, 1, 1, 100, 1, 1, 1])
LEGACY_UNITS = [
    "100 × log points", "Index points", "Log change", "100 × log points",
    "Log change", "Percentage points", "Percentage points",
]
# Give every log-based response a consistent approximate-percent conversion
# in the separately labeled corrected display. These are not exact % changes.
CORRECTED_MULTIPLIERS = np.array([100, 1, 100, 100, 100, 1, 1])
CORRECTED_UNITS = [
    "100 × log points", "Index points", "100 × log change", "100 × log points",
    "100 × log change", "Percentage points", "Percentage points",
]
VERSIONS = {
    "taiwan_gpr_dict_president": {
        "label": "President dictionary",
        "cache_subdir": "lags_2",
    },
    "gpr_taiwan_step2_median": {
        "label": "Step2 Taiwan median",
        "cache_subdir": "7vars_lags2",
    },
}
LAGS = 2
HORIZON = 20
ATOL = 1e-9
RTOL = 1e-8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_model(frame: pd.DataFrame, report: dict, output: Path) -> None:
    """Separate faithful replication from the alternative shock normalization."""
    for corrected in (False, True):
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True)
        for i, ax in enumerate(axes.flat):
            if i == len(VARIABLES):
                ax.axis("off")
                info = (
                    f"{report['input_rows']} input quarters; {report['nobs']} fitted observations\n"
                    f"{report['parameters_per_equation']} coefficients/equation; "
                    f"{report['residual_degrees_of_freedom']} residual df\n"
                    f"Companion spectral radius: {report['companion_spectral_radius']:.6f}\n"
                    f"Stable: {report['stable']}\n\n"
                    "Recursive Cholesky order is inherited.\n"
                    "Point responses only; no confidence bands.\n"
                    "Preprocessing is taken from recovered caches."
                )
                ax.text(0.02, 0.95, info, transform=ax.transAxes, va="top", fontsize=10)
                continue
            data = frame.loc[frame["response"] == VARIABLES[i]]
            if corrected:
                ax.plot(data["period"], data["legacy_shock_consistent_display"],
                        color="#986900", linestyle="--", label="Legacy shock")
                ax.plot(data["period"], data["two_sd_consistent_display"],
                        color="#16656b", linewidth=2, label="Two-SD shock")
                unit = CORRECTED_UNITS[i]
            else:
                ax.plot(data["period"], data["legacy_refit"],
                        color="#16656b", linewidth=2, label="Independent refit")
                ax.plot(data["period"], data["historical_cached"],
                        color="#986900", linestyle="none", marker="o", markersize=4,
                        markerfacecolor="none", label="Historical cache")
                unit = LEGACY_UNITS[i]
            ax.axhline(0, color="0.45", linewidth=0.7)
            ax.set(title=TITLES[i], xlabel="Quarters after shock", ylabel=unit)
            ax.grid(alpha=0.18)
            ax.set_xticks([0, 5, 10, 15, 20])
            if i == 0:
                ax.legend(fontsize=8)
        label = VERSIONS[report["version"]]["label"]
        subtitle = (
            "Shock-normalization diagnostic: 2 × orthogonal IRF vs legacy 2σ × IRF"
            if corrected else
            "Historical point-response replication: original shock and display scaling"
        )
        fig.suptitle(f"{label} · seven-variable VAR(2)\n{subtitle}", fontsize=14)
        fig.text(0.02, 0.025,
                 "Log responses use linear log-point units, not exact percentage changes. "
                 "Small-sample responses depend on the recursive identification assumptions.",
                 fontsize=9)
        fig.tight_layout(rect=[0, 0.06, 1, 0.9])
        name = "shock_normalization" if corrected else "historical_replication"
        fig.savefig(output / f"{report['version']}_{name}.png", dpi=180)
        plt.close(fig)


def replicate(version: str, root: Path, output: Path) -> tuple[dict, pd.DataFrame]:
    input_path = root / "data_analysis/output/svar_preprocessing" / f"var_ready_{version}.parquet"
    cached_path = root / "data_analysis/output/svar_analysis" / version / VERSIONS[version]["cache_subdir"] / "svar_irf_data.parquet"
    input_hash, cached_hash = sha256(input_path), sha256(cached_path)
    raw = pd.read_parquet(input_path)
    if raw["quarter"].astype(str).str.fullmatch(r"\d{4}-Q[1-4]").all():
        raw["quarter"] = pd.PeriodIndex(raw["quarter"].str.replace("-", "", regex=False), freq="Q").to_timestamp()
    else:
        raw["quarter"] = pd.to_datetime(raw["quarter"], format="ISO8601")
    data = raw.set_index("quarter")[VARIABLES]
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError(f"{version}: require ordered, unique quarters; do not silently reorder")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError(f"{version}: missing or non-finite model inputs")
    if not np.array_equal(np.diff(data.index.to_period("Q").asi8), np.ones(len(data) - 1)):
        raise ValueError(f"{version}: input quarters are not contiguous")
    data.index.freq = pd.infer_freq(data.index)
    results = VAR(data).fit(LAGS, trend="c")
    # Direct construction of the same OLS system independently checks alignment,
    # intercept, lag order, coefficients, covariance, and MA recursion.
    values = data.to_numpy()
    x = np.column_stack([
        np.ones(len(data) - LAGS),
        *[values[LAGS - lag:len(values) - lag] for lag in range(1, LAGS + 1)],
    ])
    y = values[LAGS:]
    beta, _, rank, singular_values = np.linalg.lstsq(x, y, rcond=1e-15)
    residuals = y - x @ beta
    covariance = residuals.T @ residuals / (len(y) - x.shape[1])
    cholesky = np.linalg.cholesky(covariance)
    coefficients = np.stack([beta[1 + lag * 7:1 + (lag + 1) * 7].T for lag in range(LAGS)])
    ma = np.zeros((HORIZON + 1, 7, 7))
    ma[0] = np.eye(7)
    for h in range(1, HORIZON + 1):
        for lag in range(1, min(h, LAGS) + 1):
            ma[h] += ma[h - lag] @ coefficients[lag - 1]
    direct_orthogonal = ma @ cholesky
    # This is the same calculation stored as results.irf(HORIZON).orth_irfs.
    # Calling the point-response method directly avoids computing unused
    # asymptotic covariance estimates and their platform BLAS warnings.
    orthogonal = results.orth_ma_rep(maxn=HORIZON)
    sigma_gpr = float(np.sqrt(results.sigma_u.iloc[0, 0]))
    legacy_unscaled_display = orthogonal[:, :, 0] * (2 * sigma_gpr)
    corrected_model_units = orthogonal[:, :, 0] * 2
    cached = pd.read_parquet(cached_path)
    if cached.duplicated(["response", "impulse", "period"]).any():
        raise ValueError(f"{version}: duplicate cached IRF keys")
    expected_keys = {(variable, "log_GPR", period) for variable in VARIABLES for period in range(HORIZON + 1)}
    actual_keys = set(cached[["response", "impulse", "period"]].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        raise ValueError(f"{version}: unexpected/missing cached IRF keys")
    rows = []
    for i, variable in enumerate(VARIABLES):
        for h in range(HORIZON + 1):
            rows.append({
                "version": version, "period": h, "response": variable,
                "impulse": "log_GPR", "orthogonal_one_sd_model_units": orthogonal[h, i, 0],
                "legacy_refit": legacy_unscaled_display[h, i] * LEGACY_MULTIPLIERS[i],
                "legacy_display_multiplier": int(LEGACY_MULTIPLIERS[i]),
                "legacy_display_unit": LEGACY_UNITS[i],
                "two_sd_model_units": corrected_model_units[h, i],
                "two_sd_legacy_display": corrected_model_units[h, i] * LEGACY_MULTIPLIERS[i],
                "legacy_shock_consistent_display": legacy_unscaled_display[h, i] * CORRECTED_MULTIPLIERS[i],
                "two_sd_consistent_display": corrected_model_units[h, i] * CORRECTED_MULTIPLIERS[i],
                "consistent_display_unit": CORRECTED_UNITS[i],
            })
    frame = pd.DataFrame(rows).merge(
        cached[["response", "impulse", "period", "point_estimate"]].rename(columns={"point_estimate": "historical_cached"}),
        on=["response", "impulse", "period"], how="left", validate="one_to_one",
    )
    frame["absolute_error"] = np.abs(frame["legacy_refit"] - frame["historical_cached"])
    checks = {
        "design_matrix_matches_statsmodels": bool(np.array_equal(x, results.endog_lagged)),
        "all_model_and_response_values_finite": all(bool(np.isfinite(item).all()) for item in [beta, residuals, covariance, cholesky, orthogonal]),
        "direct_ols_coefficients_match": bool(np.allclose(beta, results.params, atol=ATOL, rtol=RTOL)),
        "direct_residual_covariance_matches": bool(np.allclose(covariance, results.sigma_u, atol=ATOL, rtol=RTOL)),
        "direct_ma_cholesky_irfs_match": bool(np.allclose(direct_orthogonal, orthogonal, atol=ATOL, rtol=RTOL)),
        "all_147_cached_points_match": bool(np.allclose(frame["legacy_refit"], frame["historical_cached"], atol=ATOL, rtol=RTOL)),
        "two_sd_impact_equals_two_sigma": bool(np.isclose(corrected_model_units[0, 0], 2 * sigma_gpr, atol=ATOL, rtol=RTOL)),
        "legacy_impact_equals_two_variance": bool(np.isclose(legacy_unscaled_display[0, 0], 2 * sigma_gpr ** 2, atol=ATOL, rtol=RTOL)),
        "input_files_unchanged": sha256(input_path) == input_hash and sha256(cached_path) == cached_hash,
    }
    companion = np.block([[coefficients[0], coefficients[1]], [np.eye(7), np.zeros((7, 7))]])
    eigenvalues = np.linalg.eigvals(companion)
    root_records = [{"real": float(z.real), "imag": float(z.imag), "modulus": float(abs(z))} for z in eigenvalues]
    report = {
        "version": version, "input_file": str(input_path.relative_to(root)),
        "input_sha256": input_hash, "cached_irf_file": str(cached_path.relative_to(root)),
        "cached_irf_sha256": cached_hash, "variables_in_cholesky_order": VARIABLES,
        "lags": LAGS, "trend": "constant", "horizon_inclusive": [0, HORIZON],
        "input_rows": len(data), "input_start": data.index[0].date().isoformat(),
        "input_end": data.index[-1].date().isoformat(),
        "estimation_start": data.index[LAGS].date().isoformat(),
        "estimation_end": data.index[-1].date().isoformat(), "nobs": int(results.nobs),
        "parameters_per_equation": int(x.shape[1]), "parameters_total_excluding_covariance": int(beta.size),
        "residual_degrees_of_freedom": int(results.df_resid), "design_rank": int(rank),
        "design_columns": int(x.shape[1]), "design_condition_number": float(singular_values[0] / singular_values[-1]),
        "residual_covariance_rank": int(np.linalg.matrix_rank(covariance)),
        "residual_covariance_eigenvalues": np.linalg.eigvalsh(covariance).tolist(),
        "stable": bool(results.is_stable()),
        "companion_spectral_radius": float(max(abs(eigenvalues))),
        "companion_eigenvalues": root_records,
        "min_inverse_root_modulus_statsmodels": float(min(abs(results.roots))),
        "sigma_gpr": sigma_gpr, "legacy_shock_multiplier": 2 * sigma_gpr,
        "legacy_gpr_impact_log_points": float(legacy_unscaled_display[0, 0]),
        "two_sd_gpr_impact_log_points": float(corrected_model_units[0, 0]),
        "legacy_to_two_sd_response_ratio": sigma_gpr,
        "two_sd_to_legacy_response_ratio": 1 / sigma_gpr,
        "max_absolute_cached_point_error": float(frame["absolute_error"].max()),
        "max_absolute_cached_point_error_by_response": frame.groupby("response")["absolute_error"].max().to_dict(),
        "direct_ols_max_coefficient_error": float(np.max(np.abs(beta - results.params.to_numpy()))),
        "direct_ma_max_irf_error": float(np.max(np.abs(direct_orthogonal - orthogonal))),
        "cached_point_count": len(frame), "checks": checks,
        "cautions": [
            "Recovered transformed quarterly data are the starting point, not independently rebuilt raw economic inputs.",
            "7-variable VAR(2) estimates 15 coefficients per equation from 29 observations; residual df is 14.",
            "Cholesky recursive identification depends on the stated variable ordering; numerical full rank is not causal validation.",
            "Original Monte Carlo confidence bands are unseeded and are not regenerated or validated here.",
            "Legacy plotting multiplies log GPR and log hours by 100, but leaves delta-log investment and TAIEX unmultiplied despite percentage-point labels.",
        ] + ([] if results.is_stable() else [
            "A companion eigenvalue exceeds one: the fitted VAR is unstable, limiting long-horizon economic interpretation."
        ]),
    }
    frame.to_csv(output / f"{version}_irf_comparison.csv", index=False)
    results.params.to_csv(output / f"{version}_coefficients.csv")
    results.sigma_u.to_csv(output / f"{version}_residual_covariance.csv")
    plot_model(frame, report, output)
    return report, frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output.resolve() if args.output else root / "replication/var"
    output.mkdir(parents=True, exist_ok=True)
    reports, frames = [], []
    for version in VERSIONS:
        report, frame = replicate(version, root, output)
        reports.append(report)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / "all_irf_comparisons.csv", index=False)
    result = {
        "scope": "Deterministic refit from recovered preprocessed quarterly inputs; original point-response convention preserved.",
        "original_analysis_source": "data_analysis/svar_analysis.py",
        "original_analysis_source_sha256": sha256(root / "data_analysis/svar_analysis.py"),
        "replication_script_sha256": sha256(Path(__file__)),
        "software": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow", "statsmodels", "matplotlib", "scipy"]},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numeric_tolerances": {"absolute": ATOL, "relative": RTOL},
        "shock_normalization": {
            "definition": "Sigma_u = P P'; orth_irf(h) = Phi_h P; GPR is first in the recursive ordering, hence P[0,0] = sigma_GPR.",
            "legacy": "orth_irf(h)[:,0] * (2 * sigma_GPR); own impact = 2 * sigma_GPR^2.",
            "two_sd": "orth_irf(h)[:,0] * 2; own impact = 2 * sigma_GPR.",
            "consequence": "Legacy responses equal sigma_GPR times a correctly normalized two-residual-SD response at every horizon, before optional display conversion.",
            "display": "Legacy display is preserved in comparison columns. Separate consistent display columns multiply every log-based response by 100, which is a log-point approximation rather than an exact percentage change.",
        },
        "models": reports,
        "all_numeric_checks_passed": all(all(report["checks"].values()) for report in reports),
    }
    (output / "replication_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output), "all_numeric_checks_passed": result["all_numeric_checks_passed"],
        "models": [{key: report[key] for key in ["version", "nobs", "stable", "companion_spectral_radius", "sigma_gpr", "max_absolute_cached_point_error", "checks"]} for report in reports],
    }, indent=2))
    return 0 if result["all_numeric_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
