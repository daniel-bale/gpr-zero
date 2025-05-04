# %%
"""
Notebook to compare article counts between CC-NEWS extracted data 
and the dataset used in Lau's replication.
"""
import ibis
from ibis import _
import pandas as pd
import polars as pl
from pathlib import Path
import logging
import re
import json
from gpr_zero.gpr_index.calculator import GPRCalculator
import ibis.expr.datatypes as dt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np

# Set the backend to a non-interactive one
import matplotlib
matplotlib.use("Agg")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %% 
# --- GPR Calculation Setup ---

# Initialize the GPR calculator (using imported class)
try:
    gpr_calculator = GPRCalculator(
        dictionary_path="/home5/chuenho/gpr-zero/data/dictionaries/yeu_tong_lau_2024.json",
    )
    GPR_CALCULATOR_INITIALIZED = True
    logger.info("GPR Calculator initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize GPR Calculator: {e}")
    GPR_CALCULATOR_INITIALIZED = False
    # Depending on the flow, might want to exit or raise here
# --- End GPR Calculation Setup ---

# %% 
# --- Helper Function to Calculate GPR Category Counts ---

# %% 
# --- Font Setup for CJK --- 
font_path = Path("data/fonts/NotoSansCJKtc-Regular.otf")
if font_path.is_file():
    fm.fontManager.addfont(str(font_path))
    # Verify font registration and get the actual name
    try:
        registered_font_name = fm.FontProperties(fname=str(font_path)).get_name()
        logger.info(f"Font file {font_path} registered with name: '{registered_font_name}'")
        font_name = registered_font_name # Use the actual name found
    except Exception as e:
        logger.error(f"Could not get font properties for {font_path}: {e}")
        font_name = "Noto Sans CJK TC Regular" # Fallback to original guess
        logger.warning(f"Falling back to using font name: {font_name}")
        
    # Explicitly set sans-serif list with fallback
    plt.rcParams['font.family'] = 'sans-serif' # Ensure sans-serif is the default
    plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans'] # Add DejaVu Sans as fallback
    plt.rcParams['axes.unicode_minus'] = False 
    FONT_SETUP_SUCCESS = True
    print(f"Registered and set font: {font_name} from {font_path} with DejaVu Sans fallback")
else:
    FONT_SETUP_SUCCESS = False
    print(f"Font file not found at {font_path}. CJK characters may not display correctly.")
# --- End Font Setup ---

# %% 
# --- Configuration ---
CC_NEWS_DB_PATH = "/home5/chuenho/data/news/taiwan_news_2022_01-02.duckdb"
LAU_DB_PATH = "/home5/chuenho/gpr-zero/data/news/news.duckdb"

OUTPUT_DIR = Path("data_analysis/output/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Newspapers to compare (Chinese name from Lau DB -> CC-NEWS outlet name pattern)
# Keys are the actual names expected in Lau DB, Values are patterns for CC-NEWS
# These Chinese names will be used for merging and plotting.
NEWSPAPER_COMPARISON_MAP = {
    "中國時報": "China Times (中國時報)",
    "自由時報": "Liberty Times (自由時報)",
    "聯合報": "United Daily (聯合報)" 
}

TARGET_YEAR = 2022
TARGET_MONTHS = [1, 2, 3]
DATE_RANGE_STR = f"{TARGET_YEAR}_{TARGET_MONTHS[0]:02d}-{TARGET_MONTHS[-1]:02d}"

# %% 
# --- Helper Function to Get Daily Counts ---
def get_daily_counts(conn: ibis.BaseBackend, newspaper_col: str, target_newspapers: list[str], 
                     year: int, month: int, source_label: str, date_col_name: str = "datetime") -> pl.DataFrame:
    """Get daily article counts for specific newspapers and month."""
    logger.info(f"Getting daily counts for {source_label} source (date col: {date_col_name})...")
    try:
        if "articles" not in conn.list_tables():
            raise ValueError(f"'articles' table not found in {source_label} database.")
            
        articles = conn.table("articles")
        
        # Check if the date column exists
        if date_col_name not in articles.columns:
             raise ValueError(f"Date column '{date_col_name}' not found in {source_label} database table.")

        # Filter by date using the specified date column
        articles_filtered = articles.filter(
            (articles[date_col_name].year() == year) & 
            (articles[date_col_name].month() == month)
        )
        
        # Filter by newspapers
        articles_filtered = articles_filtered.filter(
            articles_filtered[newspaper_col].isin(target_newspapers)
        )
        
        # Group by date and newspaper, then count
        daily_counts = articles_filtered.group_by(
            date=articles_filtered[date_col_name].cast("date"), # Cast to date for grouping
            newspaper=articles_filtered[newspaper_col]
        ).agg(
            count=articles_filtered.count()
        ).order_by([ibis._.date, ibis._.newspaper])
        
        logger.info(f"Executing query for {source_label}...")
        df = daily_counts.to_polars()
        logger.info(f"Retrieved {len(df)} daily count records for {source_label}.")
        return df
        
    except Exception as e:
        logger.error(f"Error processing {source_label} source: {e}")
        raise

# %% 
# --- Connect to Databases ---
logger.info(f"Connecting to CC-NEWS DB: {CC_NEWS_DB_PATH}")
try:
    conn_ccnews = ibis.connect(f"duckdb://{CC_NEWS_DB_PATH}")
except Exception as e:
    logger.error(f"Failed to connect to CC-NEWS DB: {e}")
    raise

logger.info(f"Connecting to Lau DB: {LAU_DB_PATH}")
try:
    conn_lau = ibis.connect(f"duckdb://{LAU_DB_PATH}")
except Exception as e:
    logger.error(f"Failed to connect to Lau DB: {e}")
    conn_ccnews.disconnect()
    raise

# %% 
# --- Get Unique Outlet Names from CC-NEWS ---
logger.info("Fetching unique outlet names from CC-NEWS DB...")
cc_outlet_to_chinese_name = {} # Maps actual CC-NEWS outlet name to Chinese Name
actual_ccnews_targets = [] # List of actual CC-NEWS outlet names found

try:
    articles_ccnews = conn_ccnews.table("articles")
    ccnews_outlets = articles_ccnews.select('outlet').distinct().to_polars()["outlet"].to_list()
    logger.info(f"Found CC-NEWS outlets: {ccnews_outlets}")
    
    # Find matching CC-NEWS outlets and map them back to the Chinese name
    for chinese_name, pattern in NEWSPAPER_COMPARISON_MAP.items():
        matched_outlet = next((outlet for outlet in ccnews_outlets if pattern in outlet), None)
        if matched_outlet:
            logger.info(f"Mapping found CC-NEWS outlet '{matched_outlet}' to Chinese name '{chinese_name}' (from pattern '{pattern}')")
            actual_ccnews_targets.append(matched_outlet)
            cc_outlet_to_chinese_name[matched_outlet] = chinese_name
        else:
            logger.warning(f"Could not find a matching CC-NEWS outlet for pattern '{pattern}' (Chinese Name: '{chinese_name}') - skipping.")
    
    logger.info(f"Final CC-NEWS outlets to analyze: {actual_ccnews_targets}")
    logger.info(f"CC-NEWS Outlet to Chinese Name Map: {cc_outlet_to_chinese_name}")

except Exception as e:
    logger.error(f"Failed to get outlet names from CC-NEWS DB: {e}")
    actual_ccnews_targets = [] # Ensure it's empty if failed

# Define time column name used after bucketing
time_col_name = 'date_bucketed' # Expected output col from time_bucket

# %% 
# --- Calculate GPR Category Counts for CC-NEWS ---
gpr_counts_ccnews_df = pl.DataFrame() # Initialize empty
if GPR_CALCULATOR_INITIALIZED and actual_ccnews_targets:
    logger.info(f"Processing CC-NEWS source ({DATE_RANGE_STR})...")
    try:
        articles_ccnews_raw = conn_ccnews.table("articles")
        
        # 1. Filter Data (Date and Newspapers)
        logger.info("Filtering CC-NEWS articles...")
        articles_ccnews_filtered = articles_ccnews_raw.filter(
            (articles_ccnews_raw.datetime.year() == TARGET_YEAR) & 
            (articles_ccnews_raw.datetime.month().isin(TARGET_MONTHS)) & 
            (articles_ccnews_raw.outlet.isin(actual_ccnews_targets))    
        )
        
        # Log count after filtering
        try:
            # Execute the count expression via to_polars
            filtered_cc_count_series = articles_ccnews_filtered.count().to_polars()
            logger.info(f"Count of CC-NEWS articles after filtering: {filtered_cc_count_series}")
        except Exception as count_e:
            logger.warning(f"Could not get filtered count for CC-NEWS: {count_e}")
        
        # 2. Prepare Table for GPRCalculator (add newspaper_zh, required dummy cols)
        logger.info("Preparing CC-NEWS table for GPR Calculator...")
        
        # Build the list of (condition, result) tuples for ibis.cases
        case_tuples = []
        for outlet, zh_name in cc_outlet_to_chinese_name.items():
            condition = (articles_ccnews_filtered.outlet == outlet)
            case_tuples.append((condition, zh_name))
        
        articles_ccnews_prepared_for_gpr = articles_ccnews_filtered.mutate(
            newspaper_zh=ibis.cases(*case_tuples, else_=None), # Use ibis.cases with unpacked tuples
            # Add dummy columns required by prepare but maybe not present in CC-NEWS
            source_file=ibis.literal("CC-NEWS"), 
            category=ibis.literal(""), # Empty string for category
            date=articles_ccnews_filtered.datetime.cast('timestamp') # Ensure 'date' exists and is timestamp
        )
        # Select only columns needed by prepare + newspaper_zh
        cols_for_prepare = ['date', 'content', 'source_file', 'category', 'newspaper_zh']
        articles_ccnews_prepared_for_gpr = articles_ccnews_prepared_for_gpr.select(cols_for_prepare)

        # 3. Prepare using GPRCalculator
        logger.info("Running GPRCalculator.prepare() on CC-NEWS data...")
        # We need a new instance or reset prepare for each source if internal state is kept
        cc_calculator = GPRCalculator(dictionary_path="/home5/chuenho/gpr-zero/data/dictionaries/yeu_tong_lau_2024.json")
        prepared_ccnews_table = cc_calculator.prepare(articles_ccnews_prepared_for_gpr)

        # 4. Query for SHARES grouped by newspaper AND time bucket (Weekly)
        logger.info("Running GPRCalculator.query() on CC-NEWS data (Weekly Shares by Newspaper)...")
        gpr_results_ccnews_pl = cc_calculator.query(
            group_by_cols=['date', 'newspaper_zh'], 
            time_bucket='W', 
            time_col='date',
            calculate_share=True # <<< Calculate shares for CC-NEWS
        )
        
        # Log shape and head after query (shares)
        logger.info(f"Shape of CC-NEWS results after query (shares): {gpr_results_ccnews_pl.shape}")
        if not gpr_results_ccnews_pl.is_empty():
             logger.info(f"Head of CC-NEWS results after query (shares):\n{gpr_results_ccnews_pl.head()}") # Log head to see share values

        # 5. Transform results into [date_bucketed, newspaper_zh, category, share_ccnews] format
        logger.info("Transforming CC-NEWS results (Shares)...")
        if not gpr_results_ccnews_pl.is_empty():
            if time_col_name not in gpr_results_ccnews_pl.columns:
                 raise KeyError(f"Expected time bucket column '{time_col_name}' not found in CC-NEWS query results.")
            # Keep total_articles_in_group temporarily if needed, unpivot shares
            gpr_counts_ccnews_df = gpr_results_ccnews_pl.unpivot(
                index=[time_col_name, "newspaper_zh"], 
                on=[col for col in gpr_results_ccnews_pl.columns if col.endswith("_share")], # <<< Unpivot _share columns
                variable_name="category",
                value_name="share_ccnews" # <<< Value is share
            ).with_columns(
                pl.col("category").str.replace("_share", "") # <<< Remove _share suffix
            ).select([time_col_name, "newspaper_zh", "category", "share_ccnews"]
            ).filter(pl.col("newspaper_zh").is_not_null()
            ).with_columns(pl.col(time_col_name).cast(pl.Date)) # <<< Cast to Date
        else:
            gpr_counts_ccnews_df = pl.DataFrame({time_col_name: [], "newspaper_zh": [], "category": [], "share_ccnews": []},
                                                schema={time_col_name: pl.Date, "newspaper_zh": pl.Utf8, "category": pl.Utf8, "share_ccnews": pl.Float64})
        logger.info(f"Transformed CC-NEWS GPR counts shape: {gpr_counts_ccnews_df.shape}")

    except Exception as e:
        logger.error(f"Error processing CC-NEWS GPR counts: {e}")
        gpr_counts_ccnews_df = pl.DataFrame() # Ensure it's empty on error
else:
    logger.warning("Skipping CC-NEWS GPR calculation due to calculator init failure or no target outlets.")

# %% 
# --- Calculate GPR Category Counts for Lau's Data ---
gpr_counts_lau_df = pl.DataFrame() # Initialize empty
lau_original_targets = list(NEWSPAPER_COMPARISON_MAP.keys())
# Map: Chinese Name -> Chinese Name (for prepare step)
lau_target_map = {name: name for name in lau_original_targets}

if GPR_CALCULATOR_INITIALIZED and lau_original_targets:
    logger.info(f"Processing Lau source ({DATE_RANGE_STR})...")
    try:
        articles_lau_raw = conn_lau.table("articles")
        
        # 1. Filter Data (Date and Newspapers)
        logger.info("Filtering Lau articles...")
        articles_lau_filtered = articles_lau_raw.filter(
            (articles_lau_raw.date.year() == TARGET_YEAR) & 
            (articles_lau_raw.date.month().isin(TARGET_MONTHS)) & 
            (articles_lau_raw.newspaper.isin(lau_original_targets))
        )
        
        # Log count after filtering
        try:
            # Execute the count expression via to_polars
            filtered_lau_count_series = articles_lau_filtered.count().to_polars()
            logger.info(f"Count of Lau articles after filtering: {filtered_lau_count_series}")
        except Exception as count_e:
             logger.warning(f"Could not get filtered count for Lau: {count_e}")
        
        # 2. Prepare Table for GPRCalculator (add newspaper_zh)
        logger.info("Preparing Lau table for GPR Calculator...")
        articles_lau_prepared_for_gpr = articles_lau_filtered.mutate(
             newspaper_zh=articles_lau_filtered.newspaper # Use the original chinese name directly
        )
        # Ensure date is timestamp (prepare expects it)
        if not isinstance(articles_lau_prepared_for_gpr.date.type(), dt.Timestamp):
            articles_lau_prepared_for_gpr = articles_lau_prepared_for_gpr.mutate(date=_.date.cast('timestamp'))
            
        # Select only columns needed by prepare + newspaper_zh
        cols_for_prepare = ['date', 'content', 'source_file', 'category', 'newspaper_zh']
        articles_lau_prepared_for_gpr = articles_lau_prepared_for_gpr.select(cols_for_prepare)
        
        # 3. Prepare using GPRCalculator 
        logger.info("Running GPRCalculator.prepare() on Lau data...")
        # Need a new instance or reset prepare for each source if internal state is kept
        lau_calculator = GPRCalculator(dictionary_path="/home5/chuenho/gpr-zero/data/dictionaries/yeu_tong_lau_2024.json")
        prepared_lau_table = lau_calculator.prepare(articles_lau_prepared_for_gpr)

        # 4. Query for COUNTS grouped by newspaper AND time bucket (Weekly)
        logger.info("Running GPRCalculator.query() on Lau data (Weekly Counts by Newspaper)...")
        gpr_results_lau_pl = lau_calculator.query(
            group_by_cols=['date', 'newspaper_zh'], 
            time_bucket='W', 
            time_col='date',
            calculate_share=False # <<< Calculate counts for Lau
        )
        
        # Log shape after query
        logger.info(f"Shape of Lau results after query (counts): {gpr_results_lau_pl.shape}")
        if not gpr_results_lau_pl.is_empty():
             logger.info(f"Head of Lau results after query (counts):\n{gpr_results_lau_pl.head()}")

        # 5. Transform results into [date_bucketed, newspaper_zh, category, count_lau] format
        logger.info("Transforming Lau results (Counts)...")
        if not gpr_results_lau_pl.is_empty():
            if time_col_name not in gpr_results_lau_pl.columns: 
                 raise KeyError(f"Expected time bucket column '{time_col_name}' not found in Lau query results.")
            
            # Unpivot specific categories
            specific_cat_cols = [col for col in gpr_results_lau_pl.columns if col.endswith("_count")]
            if specific_cat_cols: # Check if there are any specific count columns
                lau_specific_cats_df = gpr_results_lau_pl.select(
                    [time_col_name, "newspaper_zh"] + specific_cat_cols
                ).unpivot(
                    index=[time_col_name, "newspaper_zh"],
                    on=specific_cat_cols,
                    variable_name="category",
                    value_name="count_lau"
                ).with_columns(
                    pl.col("category").str.replace("_count", "")
                )
            else:
                # Handle case where no specific categories were found/queried
                lau_specific_cats_df = pl.DataFrame({time_col_name: [], "newspaper_zh": [], "category": [], "count_lau": []},
                                                     schema={time_col_name: pl.Datetime, "newspaper_zh": pl.Utf8, "category": pl.Utf8, "count_lau": pl.Int64})

            # Extract 'overall' category
            if 'overall_index' in gpr_results_lau_pl.columns:
                lau_overall_cat_df = gpr_results_lau_pl.select(
                    [time_col_name, "newspaper_zh", "overall_index"]
                ).with_columns(
                    pl.lit("overall").alias("category"), # Add category column
                    pl.col("overall_index").alias("count_lau") # Rename overall_index to count_lau
                ).select([time_col_name, "newspaper_zh", "category", "count_lau"])
            else:
                 lau_overall_cat_df = pl.DataFrame({time_col_name: [], "newspaper_zh": [], "category": [], "count_lau": []},
                                                     schema={time_col_name: pl.Datetime, "newspaper_zh": pl.Utf8, "category": pl.Utf8, "count_lau": pl.Int64})

            # Combine specific and overall categories
            gpr_counts_lau_df = pl.concat(
                [lau_specific_cats_df, lau_overall_cat_df], 
                how="vertical_relaxed" # Allows combining potentially slightly different schemas if needed
            ).filter(
                pl.col("newspaper_zh").is_not_null()
            ).with_columns(
                pl.col(time_col_name).cast(pl.Date) # Cast time bucket column to Date
            )

        else:
            # Define schema explicitly for empty DataFrame case
            gpr_counts_lau_df = pl.DataFrame({time_col_name: [], "newspaper_zh": [], "category": [], "count_lau": []},
                                                schema={time_col_name: pl.Date, "newspaper_zh": pl.Utf8, "category": pl.Utf8, "count_lau": pl.Int64})
        
        # Updated log message
        logger.info(f"Transformed Lau GPR counts shape (incl. overall): {gpr_counts_lau_df.shape}")

    except Exception as e:
        logger.error(f"Error processing Lau GPR counts: {e}")
        gpr_counts_lau_df = pl.DataFrame() # Ensure it's empty on error
else:
    logger.warning("Skipping Lau GPR calculation due to calculator init failure or no target outlets.")

# %% 
# --- Join and Compare GPR Normalized Time Series --- 
# **Detailed explanations**
# - Re-introduced the `lau_normalize` function.
# - Explicitly calculated means and standard deviations for `share_ccnews` within the inner loop.
# - Applied the formula `((share_ccnews - mean_share_ccnews) / std_share_ccnews) + mean_count_lau` for `norm_share_ccnews`.
# - Applied the `lau_normalize` function to `count_lau` to calculate `norm_count_lau`.
# - Updated plot Y-axis label to "Normalized Value".
# - Updated plot series labels to "Norm Share (CC, Scaled+LauMean)" and "Norm Count (Lau, Scaled+Mean)".
# - Updated plot title suffix and directory names to `cc_scaled_lau_mean_vs_lau_norm`.
# - Updated correlation summary title and directory names to `cc_scaled_lau_mean_vs_lau_norm`.
# - Ensured correlation DataFrame column names are `pearson_norm` and `spearman_norm`.
logger.info("Comparing GPR time series (CC Scaled+LauMean vs Lau Scaled+Mean)...") # Updated log

gpr_comparison_long_df = pl.DataFrame() 

if not gpr_counts_ccnews_df.is_empty() or not gpr_counts_lau_df.is_empty():
    # Define schemas for empty DFs carefully, including types
    schema_ccnews = {time_col_name: pl.Date, "newspaper_zh": pl.Utf8, "category": pl.Utf8, "share_ccnews": pl.Float64}
    schema_lau = {time_col_name: pl.Date, "newspaper_zh": pl.Utf8, "category": pl.Utf8, "count_lau": pl.Int64}
    if gpr_counts_ccnews_df.is_empty():
         gpr_counts_ccnews_df = pl.DataFrame({k: [] for k in schema_ccnews.keys()}, schema=schema_ccnews)
    if gpr_counts_lau_df.is_empty():
         gpr_counts_lau_df = pl.DataFrame({k: [] for k in schema_lau.keys()}, schema=schema_lau)

    logger.info(f"Joining CC-NEWS shares ({gpr_counts_ccnews_df.shape}) and Lau counts ({gpr_counts_lau_df.shape}) weekly GPR...")
    gpr_comparison_long_df = gpr_counts_ccnews_df.join(
        gpr_counts_lau_df, 
        on=[time_col_name, "newspaper_zh", "category"],
        how="full"
    # Do not fill null here, handle during normalization
    ).sort(["newspaper_zh", "category", time_col_name]) 
    
    logger.info(f"Joined GPR Weekly Time Series DataFrame shape: {gpr_comparison_long_df.shape}")
    
    # Convert to pandas
    gpr_comparison_pd = gpr_comparison_long_df.to_pandas()
    gpr_comparison_pd[time_col_name] = pd.to_datetime(gpr_comparison_pd[time_col_name])
    
    # --- Perform Normalization and Analysis --- 
    correlation_results = []
    normalized_data_for_plotting = [] # Store normalized data if needed later
    
    unique_combinations = gpr_comparison_pd[['newspaper_zh', 'category']].drop_duplicates()
    logger.info(f"Found {len(unique_combinations)} unique newspaper/category combinations to analyze.") # Log unique combinations found
    logger.debug(f"Unique combinations:\\n{unique_combinations}")
    
    logger.info(f"Normalizing (CC Scaled+LauMean, Lau Scaled+Mean) and analyzing...") # Updated log
    
    # Create output subdirs
    plot_dir = OUTPUT_DIR / "timeseries_plots_cc_scaled_lau_mean_vs_lau_norm" # Updated dir name
    plot_dir.mkdir(parents=True, exist_ok=True)
    correlation_dir = OUTPUT_DIR / "correlations_cc_scaled_lau_mean_vs_lau_norm" # Updated dir name
    correlation_dir.mkdir(parents=True, exist_ok=True)
    
    # Re-introduce Normalization function (scale by std dev, shift back by mean)
    def lau_normalize(series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if std > 1e-9:
            # Correct formula: (v - mean) / std + mean
            return ((series - mean) / std) + mean
        else:
            # If std dev is zero, return the mean (constant series)
            return pd.Series(mean, index=series.index)
            
    # Plotting - One figure per newspaper
    for newspaper, group_np in gpr_comparison_pd.groupby("newspaper_zh"):
        categories = group_np["category"].unique()
        n_cats = len(categories)
        if n_cats == 0: continue
        
        fig_height = max(4, 3 * n_cats) # Adjust height based on number of categories
        fig, axes = plt.subplots(n_cats, 1, figsize=(12, fig_height), sharex=True)
        if n_cats == 1: axes = [axes] # Make iterable
            
        fig.suptitle(f"Weekly GPR Comparison (CC Scaled+LauMean vs Lau Scaled+Mean): {newspaper}", fontsize=16) # Updated title
        
        for i, category in enumerate(sorted(categories)):
            ax = axes[i]
            data_cat = group_np[group_np["category"] == category].copy() # Use copy to avoid SettingWithCopyWarning
            
            if not data_cat.empty:
                # Calculate means and std dev needed
                mean_share_ccnews = data_cat['share_ccnews'].mean()
                std_share_ccnews = data_cat['share_ccnews'].std()
                mean_count_lau = data_cat['count_lau'].mean() # Needed for CC norm
                # std_count_lau is calculated inside lau_normalize
                
                # Apply CC-NEWS Normalization: ((share - mean_share) / std_share) + mean_lau
                if std_share_ccnews > 1e-9:
                    data_cat['norm_share_ccnews'] = ((data_cat['share_ccnews'] - mean_share_ccnews) / std_share_ccnews) + mean_count_lau
                else:
                    # Handle zero std dev for shares - set normalized value to the Lau mean
                    data_cat['norm_share_ccnews'] = mean_count_lau 
                    
                # Apply Lau Count Normalization: ((count - mean_lau) / std_lau) + mean_lau
                data_cat['norm_count_lau'] = lau_normalize(data_cat['count_lau'])

                # Plotting
                ax.plot(data_cat[time_col_name], data_cat["norm_share_ccnews"], label="Norm Share (CC, Scaled+LauMean)", marker='.', linestyle='-', alpha=0.8) # Updated label
                ax.plot(data_cat[time_col_name], data_cat["norm_count_lau"], label="Norm Count (Lau, Scaled+Mean)", marker='x', linestyle='--', alpha=0.8) # Updated label
                ax.set_title(f"Category: {category}")
                ax.set_ylabel("Normalized Value") # Consistent label
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Correlation calculation on NORMALIZED series
                pearson_corr, spearman_corr = np.nan, np.nan 
                if len(data_cat) > 1: 
                    # Check variance AFTER normalization
                    var_cc_norm = data_cat["norm_share_ccnews"].var()
                    var_lau_norm = data_cat["norm_count_lau"].var()
                    
                    # Ensure both normalized series have variance before calculating correlation
                    if var_cc_norm > 1e-9 and var_lau_norm > 1e-9: 
                        temp_df_for_corr = data_cat[["norm_share_ccnews", "norm_count_lau"]].dropna() # Drop rows where either is NaN before corr
                        if len(temp_df_for_corr) > 1: # Check if enough points remain after dropna
                            pearson_matrix = temp_df_for_corr.corr(method='pearson')
                            spearman_matrix = temp_df_for_corr.corr(method='spearman')
                            pearson_corr = pearson_matrix.iloc[0, 1]
                            spearman_corr = spearman_matrix.iloc[0, 1]
                        else:
                             logger.warning(f"Skipping correlation for {newspaper} - {category} (normalized) due to insufficient non-NaN points after normalization.")
                    else:
                        logger.warning(f"Skipping correlation for {newspaper} - {category} (normalized) due to zero variance after normalization (Norm CC Var: {var_cc_norm:.2f}, Norm Lau Var: {var_lau_norm:.2f}).")
                        
                correlation_results.append({
                    "newspaper_zh": newspaper,
                    "category": category,
                    "pearson_norm": pearson_corr, # Use standard names
                    "spearman_norm": spearman_corr,
                    "n_points": len(data_cat) # Original points before dropna for correlation
                })
            else:
                ax.set_title(f"Category: {category} (No Data)")
                ax.text(0.5, 0.5, "No data for this category/newspaper", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        axes[-1].set_xlabel("Date (Week Ending)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
        plot_filename = plot_dir / f"weekly_cc_scaled_lau_mean_vs_lau_norm_{newspaper}_{DATE_RANGE_STR}.png" # Updated filename
        plt.savefig(plot_filename)
        logger.info(f"Time series plot saved to {plot_filename}")
        plt.close(fig)
        
    # --- Summarize and Save Correlations --- 
    if correlation_results:
        correlation_df = pd.DataFrame(correlation_results)
        logger.info("\\n--- Correlation Summary (CC Scaled+LauMean vs Lau Scaled+Mean) ---") # Updated title
        print(correlation_df)
        # Ensure column names are standard
        correlation_df = correlation_df.rename(columns={
            # Add potential renames here if the keys in correlation_results changed
        })
        corr_filename = correlation_dir / f"source_correlations_cc_scaled_lau_mean_vs_lau_norm_{DATE_RANGE_STR}.csv" # Updated filename
        correlation_df.to_csv(corr_filename, index=False)
        logger.info(f"Correlation summary saved to {corr_filename}")
        
        # Optional: Heatmap of correlations
        try:
            pivot_corr = correlation_df.pivot(index="category", columns="newspaper_zh", values="pearson_norm") # Use standard norm corr column
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
            plt.title(f"Pearson Corr (CC Scaled+LauMean vs Lau Scaled+Mean) - Weekly ({DATE_RANGE_STR})") # Updated title
            plt.tight_layout()
            heatmap_filename = correlation_dir / f"correlation_heatmap_cc_scaled_lau_mean_vs_lau_norm_{DATE_RANGE_STR}.png" # Updated filename
            plt.savefig(heatmap_filename)
            plt.close()
            logger.info(f"Correlation heatmap saved to {heatmap_filename}")
        except Exception as heatmap_e:
            logger.warning(f"Could not generate correlation heatmap: {heatmap_e}")
            
    else:
        logger.warning("No correlation results were generated.")
        
else:
    logger.warning("Cannot compare GPR time series as data from one or both sources is missing.")

# %% 
# --- Cleanup --- 
conn_ccnews.disconnect()
conn_lau.disconnect()
logger.info("Database connections closed.") 
logger.info("GPR Category Comparison Finished.")