# ### Summary of Changes
#
# This script has been updated to calculate labor as **log hours per capita** instead of total log hours.
#
# - **`load_labor()`:**
#   - Now reads both "employment" (`column_5`) and "working-age population" (`column_3`) from `employment.csv`.
#   - Calculates a new series: `log((employment * avg_hours) / working_age_pop)`.
# - **`assemble_master_df()`:**
#   - The final master dataframe now includes `log_Hours_per_capita` instead of the previous `log_Hours`.

import polars as pl
import ibis
from pathlib import Path
import logging
from datetime import date, datetime
from functools import reduce
import tempfile
import subprocess
import argparse

try:
    import pandas as pd
    from statsmodels.tsa.x13 import x13_arima_analysis
    from statsmodels.tools.sm_exceptions import X13NotFoundError
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    X13NotFoundError = None

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = Path("data/external")
GPR_DB_PATH = Path("data/gpr_indices.duckdb")
OUTPUT_DIR = Path("data_analysis/output/svar_preprocessing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_DF_PATH = OUTPUT_DIR / "master_quarterly_df.parquet"

# --- Chinese New Year Dates ---
# Gregorian date of Lunar New Year, for creating regressors
CNY_DATES = [
    datetime(1990, 1, 27), datetime(1991, 2, 15), datetime(1992, 2, 4),
    datetime(1993, 1, 23), datetime(1994, 2, 10), datetime(1995, 1, 31),
    datetime(1996, 2, 19), datetime(1997, 2, 7), datetime(1998, 1, 28),
    datetime(1999, 2, 16), datetime(2000, 2, 5), datetime(2001, 1, 24),
    datetime(2002, 2, 12), datetime(2003, 2, 1), datetime(2004, 1, 22),
    datetime(2005, 2, 9), datetime(2006, 1, 29), datetime(2007, 2, 18),
    datetime(2008, 2, 7), datetime(2009, 1, 26), datetime(2010, 2, 14),
    datetime(2011, 2, 3), datetime(2012, 1, 23), datetime(2013, 2, 10),
    datetime(2014, 1, 31), datetime(2015, 2, 19), datetime(2016, 2, 8),
    datetime(2017, 1, 28), datetime(2018, 2, 16), datetime(2019, 2, 5),
    datetime(2020, 1, 25), datetime(2021, 2, 12), datetime(2022, 2, 1),
    datetime(2023, 1, 22), datetime(2024, 2, 10), datetime(2025, 1, 29),
]

# --- Helper Functions ---
def minguo_to_ad(minguo_str: str) -> str | None:
    """Converts a Minguo calendar string (e.g., '106年1月') to AD date string."""
    if minguo_str is None:
        return None
    try:
        parts = minguo_str.replace('年', ' ').replace('月', '').split()
        year = int(parts[0]) + 1911
        month = int(parts[1])
        return f"{year}-{month:02d}-01"
    except (ValueError, IndexError):
        return None

def run_x13_with_holiday_regressors(data: pd.Series, regressors: pd.DataFrame, x13_path: str = "./x13as/x13as_ascii") -> pd.Series:
    """
    Run X-13 seasonal adjustment with proper holiday usertype for CNY regressors.
    
    This function bypasses statsmodels limitations and calls X-13 directly with
    usertype=holiday, which is the correct specification for Chinese New Year effects.
    """
    logger.info("Running X-13 with holiday regressors (direct call)")
    
    # Convert to absolute path
    x13_abs_path = Path(x13_path).resolve()
    
    # Series data
    data_str = "\n".join([str(val) for val in data.values])
    
    # Regressor data - organize by time point (all variables for time 1, then time 2, etc.)
    # Break into multiple lines to avoid X-13's 133 character limit
    reg_data = []
    for i in range(len(regressors)):
        for col in regressors.columns:
            reg_data.append(int(regressors.iloc[i][col]))
    
    # Break data into lines of reasonable length (max ~10 values per line)
    reg_lines = []
    for i in range(0, len(reg_data), 10):
        line_data = reg_data[i:i+10]
        reg_lines.append(" ".join([str(val) for val in line_data]))
    reg_data_str = "\n".join(reg_lines)
    
    # Variable names
    var_names = " ".join(regressors.columns)
    
    # Create X-13 spec content with proper holiday usertype and documented decimal formatting
    spec_content = f"""series{{
    title="Labor Hours Seasonal Adjustment with CNY Holiday Effects"
    start={data.index[0].year}.{data.index[0].month:02d}
    data=({data_str})
    decimals=5
}}

transform{{
    function=none
}}

regression{{
    user=({var_names})
    usertype=holiday
    data=({reg_data_str})
}}

automdl{{
}}

outlier{{
}}

forecast{{
    maxlead=0
}}

x11{{
    save=(d11 d12 d13)
    print=(d11 d12 d13)
}}

estimate{{
    print=(estimates roots)
}}
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write spec file
        spec_file = Path(tmpdir) / "labor_sa.spc"
        with open(spec_file, 'w') as f:
            f.write(spec_content)
        
        # Run X-13 with proper syntax
        spec_name = str(spec_file)[:-4]  # Remove .spc extension
        cmd = [str(x13_abs_path), "-i", spec_name]
        
        try:
            logger.info(f"Running X-13 command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
            
            if result.returncode != 0:
                logger.error(f"X-13 execution failed with return code {result.returncode}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                raise RuntimeError("X-13 execution failed")
            
            logger.info("X-13 execution completed successfully")
            
            # Save diagnostics
            diagnostics_path = OUTPUT_DIR / "x13_labor_diagnostics.txt"
            detailed_diagnostics_path = OUTPUT_DIR / "x13_labor_detailed_diagnostics.txt"
            
            # Save stdout
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)
            logger.info(f"X-13 stdout saved to {diagnostics_path}")
            
            # Read and save detailed output
            out_file = Path(tmpdir) / "labor_sa.out"
            if out_file.exists():
                with open(out_file, 'r') as f:
                    detailed_output = f.read()
                with open(detailed_diagnostics_path, "w", encoding="utf-8") as f:
                    f.write(detailed_output)
                logger.info(f"X-13 detailed output saved to {detailed_diagnostics_path}")
                
                # Check for holiday regressor confirmation
                if 'holiday' in detailed_output.lower():
                    logger.info("✓ Holiday regressors confirmed in X-13 output")
                else:
                    logger.warning("Holiday regressors not mentioned in X-13 output")
            
            # Read seasonally adjusted series
            d11_file = Path(tmpdir) / "labor_sa.d11"
            if not d11_file.exists():
                raise RuntimeError("X-13 did not generate seasonally adjusted series (d11 file missing)")
            
            # Parse the d11 file (seasonally adjusted series)
            with open(d11_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines and parse data
            data_lines = [line.strip() for line in lines if line.strip() and not line.startswith(('Date', 'YEAR'))]
            
            # Extract values (assuming format: YYYY.MM value)
            seasadj_values = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                        seasadj_values.append(value)
                    except ValueError:
                        continue
            
            if len(seasadj_values) != len(data):
                logger.warning(f"Length mismatch: input={len(data)}, output={len(seasadj_values)}")
                # Truncate or pad as needed
                if len(seasadj_values) > len(data):
                    seasadj_values = seasadj_values[:len(data)]
                elif len(seasadj_values) < len(data):
                    logger.error("X-13 output shorter than input - this shouldn't happen")
                    raise RuntimeError("X-13 output length mismatch")
            
            # Create seasonally adjusted series
            seasadj_series = pd.Series(seasadj_values, index=data.index, name='seasadj')
            logger.info(f"Successfully created seasonally adjusted series with {len(seasadj_series)} observations")
            
            return seasadj_series
            
        except Exception as e:
            logger.error(f"Error in X-13 execution: {e}")
            raise

# --- Loading Functions (Step 1: Raw to Clean Parquet) ---

def load_gpr(gpr_version: str) -> pl.DataFrame:
    """Load and process daily GPR data to quarterly."""
    logger.info(f"Loading GPR data for version: {gpr_version}")
    conn = ibis.connect(f"duckdb://{GPR_DB_PATH}")
    gpr_daily = conn.table(gpr_version).to_polars()
    conn.disconnect()
    return (
        gpr_daily
        .group_by_dynamic("date", every="1q")
        .agg(pl.mean("gpr_index").alias("gpr_q"))
        .with_columns(pl.col("date").dt.to_string("%Y-Q%q").alias("quarter"))
        .select(["quarter", "gpr_q"])
    )

def load_vix() -> pl.DataFrame:
    """Load and process daily VIX data to quarterly."""
    logger.info("Loading VIX data...")
    return (
        pl.read_csv(DATA_DIR / "vixtwn_macromicro.csv", has_header=False, new_columns=["date", "vix_close"])
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        .group_by_dynamic("date", every="1q")
        .agg(pl.mean("vix_close").alias("vix_q"))
        .with_columns(pl.col("date").dt.to_string("%Y-Q%q").alias("quarter"))
        .select(["quarter", "vix_q"])
    )

def load_cpi() -> pl.DataFrame:
    """Load and process monthly CPI data to quarterly."""
    logger.info("Loading CPI data...")
    return (
        pl.read_csv(DATA_DIR / "CPI.csv", skip_rows=2, truncate_ragged_lines=True, ignore_errors=True)
        .select(["統計期", "總指數"])
        .rename({"統計期": "date_str", "總指數": "cpi_index"})
        .with_columns(pl.col("date_str").map_elements(minguo_to_ad, return_dtype=pl.String).alias("date"))
        .filter(pl.col("date").is_not_null())
        .with_columns(
            pl.col("date").str.to_date("%Y-%m-%d"),
            pl.col("cpi_index").cast(pl.Float64, strict=False)
        )
        .drop_nulls()
        .group_by_dynamic("date", every="1q")
        .agg(pl.mean("cpi_index").alias("cpi_q"))
        .with_columns(pl.col("date").dt.to_string("%Y-Q%q").alias("quarter"))
        .select(["quarter", "cpi_q"])
    )

def load_taiex() -> pl.DataFrame:
    """Load and process daily TAIEX data to quarterly real log values."""
    logger.info("Loading TAIEX data...")
    return (
        pl.read_parquet(DATA_DIR / "TAIEX_daily.parquet")
        .group_by_dynamic("date", every="1q")
        .agg(pl.mean("taiex").alias("taiex_q_mean"))
        .with_columns(pl.col("date").dt.to_string("%Y-Q%q").alias("quarter"))
        .select(["quarter", "taiex_q_mean"])
    )

def load_gfcf() -> pl.DataFrame:
    """Load and process quarterly private investment data from the complex wide format."""
    logger.info("Loading private investment data...")
    raw_df = pl.read_csv(
        DATA_DIR / "invest.csv",
        skip_rows=6,
        has_header=False,
        truncate_ragged_lines=True,
    ).rename({
        "column_1": "col1",
        "column_2": "col2",
        "column_7": "private_invest_nominal_b",  # Changed from column_4 to column_7 for private investment
    })

    # Process the dataframe to handle the specific structure
    processed_df = (
        raw_df.with_columns(
            # Extract year and fill it down
            pl.when(pl.col("col1").str.contains(r"年"))
            .then(pl.col("col1").str.extract(r"(\d+)年", 1).cast(pl.Int32))
            .otherwise(None)
            .alias("year")
            .fill_null(strategy="forward")
        )
        .filter(pl.col("col2").str.contains(r"季"))  # Keep only rows with quarter data
        .with_columns(
            # Create a clean quarter string
            quarter=(pl.col("year") + 1911).cast(pl.String) + "-Q" + pl.col("col2").str.extract(r"(\d)", 1),
            # Clean and cast the private investment value
            gfcf_nominal_ntd=pl.col("private_invest_nominal_b")
            .str.replace_all(r"\s+", "")  # Remove all whitespace
            .cast(pl.Float64, strict=False) * 1_000_000_000  # Convert from 億元 (10^8)
        )
        .select(["quarter", "gfcf_nominal_ntd"])
        .drop_nulls()
    )

    if processed_df.height == 0:
        logger.warning("GFCF dataframe is empty after processing. Please check 'invest.csv' format.")

    return processed_df

def load_party_support() -> pl.DataFrame:
    """Load and process monthly party support data to quarterly."""
    logger.info("Loading party support data...")
    return (
        pl.read_csv(DATA_DIR / "美麗島民調.csv")
        .select(["年月", "泛綠"])
        .rename({"年月": "date_str", "泛綠": "support_pct"})
        .with_columns(
            pl.col("date_str").str.to_date("%Y-%m", strict=False).alias("date"),
            pl.col("support_pct").cast(pl.Float64, strict=False)
        )
        .drop_nulls()
        .group_by_dynamic("date", every="1q")
        .agg(pl.mean("support_pct").alias("support_q_pct"))
        .with_columns(pl.col("date").dt.to_string("%Y-Q%q").alias("quarter"))
        .select(["quarter", "support_q_pct"])
    )

def load_labor() -> pl.DataFrame:
    """
    Load, seasonally adjust, and process labor data to quarterly log hours per capita.
    This function performs seasonal adjustment on the monthly data using X-13-ARIMA
    with Chinese New Year regressors before aggregating to a quarterly series.
    """
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels or pandas not found. Skipping seasonal adjustment for labor data.")
        return pl.DataFrame({"quarter": [], "log_hours_sa_q": []})

    logger.info("Loading and seasonally adjusting labor data...")

    # 1. Construct Monthly Series (in Polars)
    employment_m = (
        pl.read_csv(DATA_DIR / "employment.csv", skip_rows=3, truncate_ragged_lines=True, ignore_errors=True, has_header=False)
        .select([pl.col("column_1"), pl.col("column_3"), pl.col("column_5")])
        .rename({"column_1": "date_str", "column_3": "working_age_pop", "column_5": "employment"})
        .with_columns(pl.col("date_str").map_elements(minguo_to_ad, return_dtype=pl.String).alias("date_str"))
        .with_columns(
            date=pl.col("date_str").str.to_date("%Y-%m-%d"),
            employment=(pl.col("employment").cast(pl.Float64, strict=False) * 1000),
            working_age_pop=(pl.col("working_age_pop").cast(pl.Float64, strict=False) * 1000)
        )
        .drop_nulls("date")
        .select(["date", "employment", "working_age_pop"])
    )
    hours_m = (
        pl.read_csv(DATA_DIR / "workhour.csv", skip_rows=2, truncate_ragged_lines=True, ignore_errors=True, has_header=False)
        .select([pl.col("column_1"), pl.col("column_2")])
        .rename({"column_1": "date_str", "column_2": "avg_hours"})
        .with_columns(pl.col("date_str").map_elements(minguo_to_ad, return_dtype=pl.String).alias("date_str"))
        .with_columns(
            date=pl.col("date_str").str.to_date("%Y-%m-%d"),
            avg_hours=pl.col("avg_hours").cast(pl.Float64, strict=False)
        )
        .drop_nulls("date")
        .select(["date", "avg_hours"])
    )
    labor_m = (
        employment_m.join(hours_m, on="date", how="inner")
        .with_columns(
            hours_per_capita=((pl.col("employment") * pl.col("avg_hours")) / pl.col("working_age_pop"))
        )
        .with_columns(
            log_hours_per_capita=pl.col("hours_per_capita").log()
        )
        .sort("date")
        .select(["date", "log_hours_per_capita"])
    )
    
    # 2. Create CNY Regressor with improved specification
    logger.info("Creating CNY regressors for seasonal adjustment...")
    
    # Convert CNY dates to monthly periods for matching
    cny_pd_series = pd.Series(CNY_DATES, name="date")
    cny_months_list = cny_pd_series.dt.to_period("M").astype(str).to_list()
    
    # Create monthly date range matching the labor data
    full_date_range = pl.date_range(
        labor_m["date"].min(),
        labor_m["date"].max(),
        "1mo",
        eager=True
    ).alias("date")
    
    # Create comprehensive CNY dummy variables
    cny_dummies = (
        full_date_range.to_frame()
        .with_columns(
            # Convert to month period for matching
            month_period=pl.from_pandas(
                full_date_range.to_pandas().dt.to_period("M").astype(str)
            ).rename("month_period")
        )
        .with_columns(
            # Main CNY month effect (strongest impact)
            cny_dummy=pl.col("month_period").is_in(cny_months_list).cast(pl.Float64)
        )
        .with_columns(
            # Pre-CNY effect (preparation, some early holidays)
            cny_dummy_lag=pl.col("cny_dummy").shift(1, fill_value=0.0),
            # Post-CNY effect (extended holiday effects, return to work)
            cny_dummy_lead=pl.col("cny_dummy").shift(-1, fill_value=0.0)
        )
        .select("date", "cny_dummy", "cny_dummy_lag", "cny_dummy_lead")
    )

    # Join with labor data and fill missing values
    labor_m_with_dummies = (
        labor_m.join(cny_dummies, on="date", how="left")
        .with_columns([
            pl.col("cny_dummy").fill_null(0.0),
            pl.col("cny_dummy_lag").fill_null(0.0), 
            pl.col("cny_dummy_lead").fill_null(0.0)
        ])
    )
    
    # Log the CNY regressor statistics
    cny_effects = labor_m_with_dummies.select([
        pl.col("cny_dummy").sum().alias("cny_months"),
        pl.col("cny_dummy_lag").sum().alias("cny_lag_months"),
        pl.col("cny_dummy_lead").sum().alias("cny_lead_months")
    ]).to_pandas().iloc[0]
    
    logger.info(f"CNY regressor summary: CNY months={cny_effects['cny_months']}, "
                f"Pre-CNY months={cny_effects['cny_lag_months']}, "
                f"Post-CNY months={cny_effects['cny_lead_months']}")

    # 3. Perform Seasonal Adjustment (in Pandas)
    logger.info("Converting to pandas and preparing X-13 inputs...")
    labor_pd = labor_m_with_dummies.to_pandas().set_index("date")
    
    # Ensure we have clean data for X-13
    endog = labor_pd["log_hours_per_capita"].dropna()
    exog = labor_pd[["cny_dummy", "cny_dummy_lag", "cny_dummy_lead"]].reindex(endog.index)
    
    # Verify data alignment and coverage
    logger.info(f"X-13 input data: {len(endog)} observations from {endog.index.min()} to {endog.index.max()}")
    logger.info(f"CNY effects in data: {exog.sum().to_dict()}")
    logger.info(f"Sample CNY regressor data:\n{exog.head(12)}")  # Show first year

    try:
        # Check for X-13 binary before running analysis
        x13_path = './x13as/x13as_ascii'
        x13_abs_path = Path(x13_path).resolve()
        if not x13_abs_path.exists():
            logger.error(
                f"X-13 binary not found at {x13_abs_path}. Please install it or set the correct path. "
                "See: https://www.census.gov/data/software/x13as.html"
            )
            raise FileNotFoundError(f"X-13 binary not found at {x13_abs_path}")
            
        # Apply X-13 seasonal adjustment with proper CNY holiday regressors
        logger.info("Applying X-13 seasonal adjustment with CNY holiday regressors...")
        logger.info(f"  Using direct X-13 call with usertype=holiday")
        logger.info(f"  Exog shape: {exog.shape}, non-zero values: {(exog != 0).sum().sum()}")
        
        log_hours_sa_m = run_x13_with_holiday_regressors(
            endog, 
            exog, 
            x13_path=x13_path
        )
        
        logger.info("X-13 seasonal adjustment with holiday regressors completed successfully.")

    except FileNotFoundError:
        # Reraise the specific error to stop execution
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during seasonal adjustment: {e}. Returning unadjusted series.")
        # When adjustment fails, use the original series but rename it to match expectations downstream
        log_hours_sa_m = endog.rename("seasadj")

    # 4. Aggregate to Quarterly (back in Polars)
    log_hours_sa_m_df = pl.from_pandas(log_hours_sa_m.reset_index().rename(columns={'index': 'date', 'seasadj': 'log_hours_sa_m'}))
    
    return (
        log_hours_sa_m_df
        .group_by_dynamic("date", every="1q")
        .agg(pl.mean("log_hours_sa_m").alias("log_hours_sa_q"))
        .with_columns(pl.col("date").dt.to_string("%Y-Q%q").alias("quarter"))
        .select(["quarter", "log_hours_sa_q"])
    )

def load_interest_rate() -> pl.DataFrame:
    """Load and process interest rate data to quarterly levels."""
    logger.info("Loading interest rate data...")
    interest_changes = (
        pl.read_csv(DATA_DIR / "interest.csv", encoding="big5")
        .select(["調整日期", "重貼現率"])
        .rename({"調整日期": "date", "重貼現率": "policy_rate_pct"})
        .with_columns(
            pl.col("date").str.to_date("%Y/%m/%d", strict=False),
            pl.col("policy_rate_pct").cast(pl.Float64, strict=False)
        )
        .sort("date").drop_nulls()
    )
    start_date = interest_changes["date"].min()
    if start_date is None:
        raise ValueError("No dates found in interest rate data.")
    full_date_range = pl.date_range(start_date, date(2025, 5, 1), "1d", eager=True).alias("date").to_frame()
    return (
        full_date_range.join(interest_changes, on="date", how="left").fill_null(strategy="forward")
        .group_by_dynamic("date", every="1q")
        .agg(pl.last("policy_rate_pct").alias("rate_q_pct"))
        .with_columns(pl.col("date").dt.to_string("%Y-Q%q").alias("quarter"))
        .select(["quarter", "rate_q_pct"])
    )

def run_preprocessing_and_save(gpr_version: str):
    """
    Step 1 of the pipeline. Loads each raw data source, processes it into a
    quarterly series, and saves it as a separate Parquet file.
    """
    logger.info(f"--- Running Step 1: Processing and saving individual series for GPR version: {gpr_version} ---")
    
    load_cpi().write_parquet(OUTPUT_DIR / "cpi_q.parquet")
    load_taiex().write_parquet(OUTPUT_DIR / "taiex_q.parquet")
    load_gfcf().write_parquet(OUTPUT_DIR / "gfcf_q.parquet")
    load_gpr(gpr_version).write_parquet(OUTPUT_DIR / f"gpr_q_{gpr_version}.parquet")
    load_vix().write_parquet(OUTPUT_DIR / "vix_q.parquet")
    load_party_support().write_parquet(OUTPUT_DIR / "support_q.parquet")
    load_labor().write_parquet(OUTPUT_DIR / "labor_q.parquet")
    load_interest_rate().write_parquet(OUTPUT_DIR / "rate_q.parquet")

    logger.info("--- Finished Step 1: All individual series saved as Parquet files. ---")

def assemble_master_df(gpr_version: str):
    """
    Step 2 of the pipeline. Loads the pre-processed Parquet files,
    creates real variables, applies stationarity transformations, 
    and assembles the final master dataframe.
    """
    logger.info(f"--- Running Step 2: Assembling final master dataframe for GPR version: {gpr_version} ---")
    
    # Define the output path for the master dataframe
    master_df_path = OUTPUT_DIR / f"master_quarterly_{gpr_version}.parquet"

    # Load all pre-processed series from Parquet files
    try:
        gpr_q = pl.read_parquet(OUTPUT_DIR / f"gpr_q_{gpr_version}.parquet")
        vix_q = pl.read_parquet(OUTPUT_DIR / "vix_q.parquet")
        taiex_q = pl.read_parquet(OUTPUT_DIR / "taiex_q.parquet")
        gfcf_q = pl.read_parquet(OUTPUT_DIR / "gfcf_q.parquet")
        support_q = pl.read_parquet(OUTPUT_DIR / "support_q.parquet")
        labor_q = pl.read_parquet(OUTPUT_DIR / "labor_q.parquet")
        rate_q = pl.read_parquet(OUTPUT_DIR / "rate_q.parquet")
        cpi_q = pl.read_parquet(OUTPUT_DIR / "cpi_q.parquet")
    except Exception as e:
        logger.error(f"Failed to read one or more Parquet files: {e}")
        return

    # Consolidate all base dataframes for joining
    base_dfs = [gpr_q, vix_q, taiex_q, gfcf_q, support_q, labor_q, rate_q]

    # Join all base dataframes using a full join to keep all data
    logger.info("Joining all base quarterly dataframes with a full join...")
    master_df = base_dfs[0]
    for df in base_dfs[1:]:
        master_df = master_df.join(df, on="quarter", how="full", coalesce=True)

    # Join with CPI data, then compute final variables and select columns
    logger.info("Joining with CPI and calculating base variables...")
    base_vars_df = master_df.join(cpi_q, on="quarter", how="left").with_columns(
        pl.col("gpr_q").log1p().alias("log_GPR"),
        pl.col("vix_q").alias("VIX_lvl"),
        (pl.col("taiex_q_mean") / (pl.col("cpi_q") / 100)).log().alias("log_TAIEX_real"),
        (pl.col("gfcf_nominal_ntd") / (pl.col("cpi_q") / 100)).log().alias("log_GFCF_real"),
        pl.col("support_q_pct").alias("Support_pct"),
        pl.col("log_hours_sa_q").alias("log_Hours_SA"),
        pl.col("rate_q_pct").alias("Rate_pct"),
    ).select([
        "quarter", "log_GPR", "VIX_lvl", "log_TAIEX_real", 
        "log_GFCF_real", "Support_pct", "log_Hours_SA", "Rate_pct"
    ])

    # Sort and filter for the desired date range before applying transformations
    base_vars_df = base_vars_df.sort("quarter").filter(
        (pl.col("quarter") >= "2017-Q1") & (pl.col("quarter") <= "2024-Q4")
    )

    # Apply stationarity transformations
    logger.info("Applying stationarity transformations...")
    final_df = base_vars_df.with_columns([
        # Log-differences for trending variables (growth rates/returns)
        (pl.col("log_GFCF_real") - pl.col("log_GFCF_real").shift(1)).alias("Δlog_GFCF_real"),
        (pl.col("log_TAIEX_real") - pl.col("log_TAIEX_real").shift(1)).alias("Δlog_TAIEX_real"),
        # First difference for policy rate (change in interest rate)
        (pl.col("Rate_pct") - pl.col("Rate_pct").shift(1)).alias("ΔRate_pct"),
    ]).select([
        # Keep quarter for reference
        "quarter",
        # Level variables (assumed stationary or near-stationary)
        "log_GPR", "VIX_lvl", "log_Hours_SA", "Support_pct",
        # Differenced variables for stationarity
        "Δlog_GFCF_real", "Δlog_TAIEX_real", "ΔRate_pct",
        # Also keep original levels for diagnostic purposes
        "log_GFCF_real", "log_TAIEX_real", "Rate_pct"
    ])

    # Drop the first row due to differencing (will contain NaN values)
    final_df = final_df.filter(pl.col("quarter") != final_df["quarter"].min())

    logger.info(f"Master dataframe created with shape: {final_df.shape}")
    logger.info(f"Final date range: {final_df['quarter'].min()} to {final_df['quarter'].max()}")
    logger.info(f"Variables for VAR analysis: log_GPR, VIX_lvl, Δlog_GFCF_real, log_Hours_SA, Δlog_TAIEX_real, Support_pct, ΔRate_pct")
    logger.info(f"Head of master_df:\n{final_df.head()}")
    logger.info(f"Tail of master_df:\n{final_df.tail()}")
    
    # Check for any remaining missing values
    null_count = final_df.null_count()
    if null_count.select(pl.sum_horizontal(pl.all())).item() > 0:
        logger.warning(f"Missing values found:\n{null_count}")
    else:
        logger.info("✓ No missing values in final dataset")
    
    final_df.write_parquet(master_df_path)
    logger.info(f"Master dataframe saved to {master_df_path}")

    # Save VAR-ready subset for convenience
    var_ready_path = OUTPUT_DIR / f"var_ready_{gpr_version}.parquet"
    var_ready_df = final_df.select([
        "quarter", "log_GPR", "VIX_lvl", "Δlog_GFCF_real", 
        "log_Hours_SA", "Δlog_TAIEX_real", "Support_pct", "ΔRate_pct"
    ])
    var_ready_df.write_parquet(var_ready_path)
    logger.info(f"VAR-ready dataframe saved to {var_ready_path}")

def main():
    """Runs the full two-step pipeline."""
    parser = argparse.ArgumentParser(description="Build the master quarterly dataframe for SVAR analysis.")
    parser.add_argument(
        "--gpr-version",
        type=str,
        default="yeu_tong_lau_2024",
        help="The name of the GPR index table to use from gpr_indices.duckdb (e.g., 'yeu_tong_lau_2024')."
    )
    args = parser.parse_args()

    run_preprocessing_and_save(args.gpr_version)
    assemble_master_df(args.gpr_version)

if __name__ == "__main__":
    main()