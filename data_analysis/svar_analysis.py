import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import argparse

# --- CJK Font Setup ---
from gpr_zero.utils.font_setup import ensure_cjk_font_setup
ensure_cjk_font_setup()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Specification ---
# Variables to be included in the VAR model.
# Comment out variables to exclude them from the analysis.
MODEL_VARIABLES = [
    'log_GPR', 
    'VIX_lvl', 
    'Δlog_GFCF_real', 
    'log_Hours_SA', 
    'Δlog_TAIEX_real', 
    'Support_pct', 
    'ΔRate_pct'  # Removed for stability test
]
VARIABLE_COUNT = len(MODEL_VARIABLES)
VARIABLE_TAG = f"{VARIABLE_COUNT}vars"

# --- Configuration ---
BASE_INPUT_DIR = Path("data_analysis/output/svar_preprocessing")
BASE_OUTPUT_DIR = Path("data_analysis/output/svar_analysis")

def run_svar_analysis(gpr_version: str, lags: int):
    """
    Loads VAR-ready data, runs SVAR with a specific number of lags and variables,
    and saves results, plots, and IRF data.
    """
    variables = MODEL_VARIABLES
    # Define version- and lag-specific paths, now including variable count
    input_path = BASE_INPUT_DIR / f"var_ready_{gpr_version}.parquet"
    output_dir = BASE_OUTPUT_DIR / gpr_version / f"{VARIABLE_TAG}_lags{lags}"
    output_dir.mkdir(parents=True, exist_ok=True)
    irf_plot_path = output_dir / "svar_irf_custom.png"

    logger.info(f"--- Starting SVAR Analysis for GPR Version: {gpr_version} with {lags} lags and {VARIABLE_COUNT} variables ---")
    logger.info(f"Loading VAR-ready data from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
        df['quarter'] = pd.to_datetime(df['quarter'])
        df = df.set_index('quarter')
    except FileNotFoundError:
        logger.error(f"Input file not found at {input_path}. Please run the preprocessing script first.")
        return

    logger.info(f"Defining and fitting the VAR({lags}) model with variables: {variables}")
    model = sm.tsa.VAR(df[variables])
    results = model.fit(lags)
    logger.info(f"VAR({lags}) model fitted.")
    
    if not results.is_stable():
        logger.warning(f"VAR model with {lags} lags is NOT STABLE.")
    else:
        logger.info(f"VAR model stability: {results.is_stable()}")

    logger.info("Generating Impulse Response Functions (IRFs)...")
    irf = results.irf(20)
    errband_mc_1std = irf.errband_mc(orth=True, repl=400, signif=0.32, burn=80)
    errband_mc_2std = irf.errband_mc(orth=True, repl=400, signif=0.05, burn=80)
    
    Sigma_u = results.sigma_u
    Sigma_u_array = Sigma_u.values
    sigma_GPR = np.sqrt(Sigma_u_array[0, 0])
    shock_magnitude = 2 * sigma_GPR
    
    logger.info(f"GPR innovation std dev (from VAR residuals): {sigma_GPR:.4f}")
    logger.info(f"Using two standard deviation shock in GPR: {shock_magnitude:.4f}")

    # --- Custom Plotting & Data Collection ---
    logger.info("Generating custom IRF plot and collecting data...")
    plot_titles = {
        'log_GPR': 'GPR', 'VIX_lvl': 'VIX', 'Δlog_GFCF_real': 'Private Investment Growth',
        'log_Hours_SA': 'Hours (SA)', 'Δlog_TAIEX_real': 'TAIEX Returns', 
        'Support_pct': 'Party Support', 'ΔRate_pct': 'Change in Policy Rate'
    }
    ylabels = {
        'log_GPR': 'Percent', 'VIX_lvl': 'Index', 'Δlog_GFCF_real': 'Percentage Points',
        'log_Hours_SA': 'Percent', 'Δlog_TAIEX_real': 'Percentage Points', 
        'Support_pct': 'Percentage Points', 'ΔRate_pct': 'Percentage Points'
    }
    
    num_plots = len(variables)
    nrows = (num_plots + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(15, 4 * nrows), sharex=True)
    axes = axes.flatten()
    impulse_var = 'log_GPR'
    
    lower_bands_1std, upper_bands_1std = errband_mc_1std
    lower_bands_2std, upper_bands_2std = errband_mc_2std
    
    all_irfs_data = []

    for i, response_var in enumerate(variables):
        ax = axes[i]
        
        point_response = irf.orth_irfs[:, variables.index(response_var), variables.index(impulse_var)]
        lower_band_1std = lower_bands_1std[:, variables.index(response_var), variables.index(impulse_var)]
        upper_band_1std = upper_bands_1std[:, variables.index(response_var), variables.index(impulse_var)]
        lower_band_2std = lower_bands_2std[:, variables.index(response_var), variables.index(impulse_var)]
        upper_band_2std = upper_bands_2std[:, variables.index(response_var), variables.index(impulse_var)]
        
        point_response_scaled = point_response * shock_magnitude
        lower_band_1std_scaled = lower_band_1std * shock_magnitude
        upper_band_1std_scaled = upper_band_1std * shock_magnitude
        lower_band_2std_scaled = lower_band_2std * shock_magnitude
        upper_band_2std_scaled = upper_band_2std * shock_magnitude
        
        multiplier = 100 if ylabels[response_var] == 'Percent' else 1
        
        ax.plot(range(len(point_response_scaled)), point_response_scaled * multiplier, color='black', lw=2)
        ax.fill_between(range(len(point_response_scaled)), lower_band_2std_scaled * multiplier, upper_band_2std_scaled * multiplier, color='lightblue', alpha=0.4, label='95% CI')
        ax.fill_between(range(len(point_response_scaled)), lower_band_1std_scaled * multiplier, upper_band_1std_scaled * multiplier, color='steelblue', alpha=0.6, label='68% CI')
        
        ax.axhline(0, color='grey', linestyle='-', lw=1)
        ax.set_title(plot_titles[response_var])
        ax.set_xlabel('Quarters')
        ax.set_ylabel(ylabels.get(response_var, "Value"))
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        irf_data = {
            'period': range(len(point_response_scaled)),
            'response': response_var, 'impulse': impulse_var,
            'point_estimate': point_response_scaled * multiplier,
            'ci_68_lower': lower_band_1std_scaled * multiplier, 'ci_68_upper': upper_band_1std_scaled * multiplier,
            'ci_95_lower': lower_band_2std_scaled * multiplier, 'ci_95_upper': upper_band_2std_scaled * multiplier,
        }
        all_irfs_data.append(pd.DataFrame(irf_data))

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    # fig.suptitle(f'Impact of a Two SD GPR Shock ({gpr_version}, {VARIABLE_TAG}, {lags} lags)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    logger.info(f"Saving custom IRF plot to {irf_plot_path}...")
    fig.savefig(irf_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    irf_data_df = pd.concat(all_irfs_data, ignore_index=True)
    irf_data_path = output_dir / "svar_irf_data.parquet"
    irf_data_df.to_parquet(irf_data_path)
    logger.info(f"Saved IRF data to {irf_data_path}")
    
    results_path = output_dir / "svar_results_summary.txt"
    with open(results_path, 'w') as f:
        f.write(f"SVAR Analysis Results - {gpr_version} ({VARIABLE_TAG}, {lags} lags)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Variables: {variables}\n\n")
        f.write(f"Lags: {lags}\n\n")
        f.write(f"Model Stability: {results.is_stable()}\n\n")
        f.write("IRF Specification:\n")
        f.write(f"- Shock magnitude: Two standard deviation in GPR innovation = {shock_magnitude:.4f}\n")
        f.write(f"- GPR innovation std dev (from VAR residuals): {sigma_GPR:.4f}\n")
        f.write(str(results.summary()))
    logger.info(f"Model results summary saved to {results_path}")
    
    logger.info(f"--- SVAR analysis for {gpr_version} ({lags} lags, {VARIABLE_TAG}) complete. ---")

def run_robustness_check(gpr_version: str, base_lags: int, new_lags: int):
    """
    Compares the IRF of a VAR model with `new_lags` against a baseline
    model with `base_lags`, using the globally defined variables.
    """
    logger.info(f"--- Starting Robustness Check for {gpr_version}: Comparing {new_lags} lags vs {base_lags} lags ({VARIABLE_TAG}) ---")

    base_output_dir = BASE_OUTPUT_DIR / gpr_version / f"{VARIABLE_TAG}_lags{base_lags}"
    new_output_dir = BASE_OUTPUT_DIR / gpr_version / f"{VARIABLE_TAG}_lags{new_lags}"
    base_irf_path = base_output_dir / "svar_irf_data.parquet"
    new_irf_path = new_output_dir / "svar_irf_data.parquet"

    try:
        if not base_irf_path.exists():
            logger.info(f"Base IRF data for {base_lags} lags not found. Running analysis to generate it.")
            run_svar_analysis(gpr_version, base_lags)
        if not new_irf_path.exists():
            logger.info(f"New IRF data for {new_lags} lags not found. Running analysis to generate it.")
            run_svar_analysis(gpr_version, new_lags)
    except np.linalg.LinAlgError as e:
        logger.error(f"Robustness check failed: Could not generate IRF for one of the models. Error: {e}")
        return

    logger.info("Loading IRF data for comparison...")
    base_irf_df = pd.read_parquet(base_irf_path)
    new_irf_df = pd.read_parquet(new_irf_path)

    variables = base_irf_df['response'].unique()
    plot_titles = {
        'log_GPR': 'GPR', 'VIX_lvl': 'VIX', 'Δlog_GFCF_real': 'Private Investment Growth',
        'log_Hours_SA': 'Hours (SA)', 'Δlog_TAIEX_real': 'TAIEX Returns',
        'Support_pct': 'Party Support', 'ΔRate_pct': 'Change in Policy Rate'
    }

    num_plots = len(variables)
    nrows = (num_plots + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(15, 4 * nrows), sharex=True)
    axes = axes.flatten()

    for i, response_var in enumerate(variables):
        ax = axes[i]
        base_data = base_irf_df[base_irf_df['response'] == response_var]
        new_data = new_irf_df[new_irf_df['response'] == response_var]

        ax.fill_between(base_data['period'], base_data['ci_68_lower'], base_data['ci_68_upper'],
                        color='steelblue', alpha=0.6, label=f'{base_lags}-lag Model (68% CI)')
        
        ax.plot(base_data['period'], base_data['point_estimate'], color='black', lw=1.5, linestyle='--', label=f'{base_lags}-lag Model (Mean)')
        
        ax.plot(new_data['period'], new_data['point_estimate'], color='red', lw=2, label=f'{new_lags}-lag Model (Mean)')

        ax.axhline(0, color='grey', linestyle='-', lw=1)
        ax.set_title(plot_titles.get(response_var, response_var))
        ax.set_xlabel('Quarters')
        if i % 4 == 0:
            ax.set_ylabel("Response")
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    fig.suptitle(f'Robustness Check: {new_lags} lags vs. {base_lags} lags for {gpr_version} ({VARIABLE_TAG})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = new_output_dir / f"robustness_check_{base_lags}_vs_{new_lags}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Robustness check plot saved to {plot_path}")
    logger.info("--- Robustness Check Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Run SVAR analysis for a specific GPR index version.")
    parser.add_argument("--gpr-version", type=str, required=True, help="The name of the GPR index table to analyze.")
    parser.add_argument("--lags", type=int, default=2, help="Number of lags for the VAR model.")
    parser.add_argument("--robustness-check", action="store_true", help="Run a robustness check comparing two lag specifications.")
    parser.add_argument("--base-lags", type=int, default=2, help="The base number of lags for the robustness check.")
    parser.add_argument("--new-lags", type=int, default=4, help="The new number of lags to compare for the robustness check.")
    args = parser.parse_args()
    
    try:
        if args.robustness_check:
            run_robustness_check(args.gpr_version, args.base_lags, args.new_lags)
        else:
            run_svar_analysis(args.gpr_version, args.lags)
    except Exception as e:
        logger.error(f"An error occurred during SVAR analysis for {args.gpr_version}: {e}", exc_info=True)

if __name__ == "__main__":
    main()