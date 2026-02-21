# %%
import ibis
from ibis import _
import sys
import os
from pathlib import Path
import pandas as pd  # Add pandas import for later visualization

# Set the backend to a non-interactive one to avoid GUI requirements
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend for non-interactive plotting

# --- Font Setup for CJK ---
import matplotlib.pyplot as plt # Need plt for rcParams
import matplotlib.font_manager as fm

font_path = Path("data/fonts/NotoSansCJKtc-Regular.otf")
if font_path.is_file():
    fm.fontManager.addfont(str(font_path))
    # Specify the font family name matplotlib should recognize
    # This might require slight adjustment depending on the font's internal name
    font_name = "Noto Sans CJK TC Regular" 
    plt.rcParams['font.family'] = 'sans-serif' # Use sans-serif stack
    # Prepend the CJK font to the sans-serif list
    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
    # Ensure minus signs are handled correctly
    plt.rcParams['axes.unicode_minus'] = False 
    print(f"Registered and set font: {font_name} from {font_path}")
else:
    print(f"Font file not found at {font_path}. CJK characters may not display correctly.")
# --- End Font Setup ---

# Import gpr_zero modules directly (they are auto-installed by uv)
from gpr_zero.gpr_index.calculator import GPRCalculator
from gpr_zero.gpr_index.analyzer import GPRAnalyzer
# %%
import polars as pl
df = pl.read_ndjson("/home5/chuenho/gpr-zero/data/news/processed/LT-military_buildup_02.ndjson")
df.with_columns(
    pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d")
).sort("date").group_by_dynamic("date", every="1w").agg(pl.len()).head(20)
# %% [markdown]
# # Reproducing Yeu-Tong Lau's Work on GPR Index
#
# This notebook replicates the analysis from Yeu-Tong Lau (2024) on the Geopolitical Risk Index
# with a focus on Taiwan news data.
# %%
import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import argparse
import ibis.expr.datatypes as dt # Import datatypes module

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Create output directory for files
OUTPUT_DIR = "data_analysis/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map Chinese newspaper names to English
NEWSPAPER_MAP = {
    "中國時報": "China Times",
    "聯合報": "UDN",
    "自由時報": "Liberty Times",
    "工商時報": "Commercial Times",
}

# Correct GPR category names
CORRECT_CATEGORIES = ["war_threat", "peace_threat", "military_buildup", "war_begin"]

# %%
# Parse command line arguments
parser = argparse.ArgumentParser(description="Reproduce GPR analysis from Lau (2024)")
# parser.add_argument(
#     "--divide_by_total",
#     action="store_true",
#     help="Divide GPR by total article count (default: False)",
# )
# parser.add_argument(
#     "--total_count",
#     type=int,
#     default=None,
#     help="Override total article count for GPR calculation",
# )

# Handle both script and notebook contexts
# For this script, we always use the Lau normalization method
# if "ipykernel" in sys.modules:  # Running in a notebook
#     pass # No arguments needed in notebook context for this version
# else:  # Running as a script
#     try:
#         args, unknown_args = parser.parse_known_args()
#         # We are not using command-line flags for calculation type anymore
#     except:
#         # Fallback if argument parsing fails (though not expected now)
#         print(
#             "Argument parsing failed, proceeding with default Lau normalization"
#         )

# Set up logging
LOG_FILE = os.path.join(OUTPUT_DIR, "gpr_analysis.log")
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# %%
# Connect to the DuckDB database
conn = ibis.connect("duckdb:///home5/chuenho/gpr-zero/data/news/news.duckdb")

# %%
# List tables in the database
logger.info("Tables in the database:")
logger.info(conn.list_tables())

# Check if the articles table exists
if "articles" in conn.list_tables():
    # Get the articles table
    articles = conn.table("articles")
else:
    raise ValueError("articles table not found in database")

# %%
# Display the schema
logger.info("Articles table schema:")
logger.info(articles.schema())
articles.limit(5).to_polars()

# %%
# Get the first 5 rows of the articles table using to_polars instead of execute
logger.info("First 5 articles:")
first_5_articles = articles.limit(5).to_polars()
logger.info(first_5_articles)

# %%
# Basic statistics about the articles
articles_count = articles.count().to_polars()
if isinstance(articles_count, int):
    total_articles = articles_count
else:
    total_articles = articles_count.item()
logger.info(f"Total number of articles: {total_articles}")

# Count by newspaper
logger.info("\nArticles by newspaper:")
news_counts = (
    articles.group_by("newspaper")
    .aggregate(count=_.article_hash.count())
    .order_by(ibis.desc("count"))
    .to_polars()
)
logger.info(news_counts)

# Count by date (recent dates)
logger.info("\nArticles by date (last 10 days):")
date_counts = (
    articles.group_by("date")
    .aggregate(count=_.article_hash.count())
    .order_by(ibis.desc("date"))
    .limit(10)
    .to_polars()
)
logger.info(date_counts)

# Count by category
logger.info("\nTop 10 article categories:")
category_counts = (
    articles.group_by("category")
    .aggregate(count=_.article_hash.count())
    .order_by(ibis.desc("count"))
    .limit(10)
    .to_polars()
)
logger.info(category_counts)

# %%
# Loading the GPR dictionary
try:
    with open("data/dictionaries/yeu_tong_lau_2024.json", "r", encoding="utf-8") as f:
        gpr_dict_full = json.load(f)
    logger.info("Successfully loaded GPR dictionary")

    # Create a filtered dictionary with only the needed categories
    gpr_dict = {
        "metadata": gpr_dict_full.get("metadata", {}),
        "categories": gpr_dict_full.get("categories", {}),
    }

    # Only include the required categories
    if "keywords_lists" in gpr_dict_full:
        gpr_dict["keywords_lists"] = {}
        for category in CORRECT_CATEGORIES:
            if category in gpr_dict_full["keywords_lists"]:
                gpr_dict["keywords_lists"][category] = gpr_dict_full["keywords_lists"][
                    category
                ]
                logger.info(
                    f"Added {category} with {len(gpr_dict['keywords_lists'][category])} terms"
                )
            else:
                logger.warning(
                    f"Category {category} not found in the dictionary keywords_lists"
                )

    # Include translations if available
    if "translations" in gpr_dict_full:
        gpr_dict["translations"] = gpr_dict_full["translations"]

    # Display the structure of the filtered dictionary
    logger.info("\nFiltered dictionary structure:")
    if isinstance(gpr_dict, dict):
        for key, value in gpr_dict.items():
            if isinstance(value, dict):
                logger.info(f"{key}: {len(value)} items")
            elif isinstance(value, list):
                logger.info(f"{key}: {len(value)} terms")
            else:
                logger.info(f"{key}: {type(value)}")
    else:
        logger.info(f"Dictionary type: {type(gpr_dict)}")
except FileNotFoundError:
    raise FileNotFoundError("GPR dictionary file not found. Please check the path.")
except json.JSONDecodeError:
    raise ValueError("Error parsing the GPR dictionary JSON file.")

# %%        
# Initialize the calculator (Now using the imported GPRCalculator)
gpr_calculator = GPRCalculator(
    dictionary_path="data/dictionaries/yeu_tong_lau_2024.json",
)

# %% [markdown]
# ## Prepare Data and Calculate Daily GPR Index
# The `prepare` method labels the articles table in the database using ibis expressions.
# The `query` method then aggregates these labels to calculate daily GPR counts.

# %%
# Get the base articles table from ibis connection
logger.info("Getting articles table from database...")
articles_table = conn.table("articles") # Assuming 'conn' is the active ibis connection
logger.info(f"Articles table schema: {articles_table.schema()}")

# Add newspaper_en column using ibis for consistent mapping
logger.info("Adding 'newspaper_en' column...")
articles_with_en = articles_table.mutate(
    newspaper_en=ibis.cases(
        (articles_table.newspaper == "中國時報", "China Times"),
        (articles_table.newspaper == "聯合報", "UDN"),
        (articles_table.newspaper == "自由時報", "Liberty Times"),
        (articles_table.newspaper == "工商時報", "Commercial Times"),
        else_=articles_table.newspaper # Keep original if no match
    )
)
logger.info("Added 'newspaper_en'.")


# Prepare the data (label articles) using the calculator
logger.info("Preparing data using GPR calculator...")
gpr_calculator.prepare(articles_with_en) # Pass the ibis table with newspaper_en
logger.info("Data preparation complete.")


# Calculate daily GPR counts using the query method
logger.info("Calculating daily GPR index counts...")
daily_gpr_pl = gpr_calculator.query(group_by_cols=['date'])
logger.info(f"Calculated daily GPR counts. Shape: {daily_gpr_pl.shape}")

# Convert to pandas DataFrame and set index for downstream compatibility
logger.info("Converting results to pandas DataFrame and setting index...")
if not daily_gpr_pl.is_empty():
    # Rename columns to match downstream expectations (e.g., category counts)
    rename_mapping = {f"{cat}_count": cat for cat in gpr_calculator.category_definitions.keys()}
    daily_gpr_pl = daily_gpr_pl.rename(rename_mapping)

    # Convert to pandas, assuming 'date_bucketed' or 'date' is the time column name
    time_col_name = 'date' # Adjust if bucketing was used and renamed the column
    if time_col_name not in daily_gpr_pl.columns:
         # Handle potential renaming from bucketing if default 'date' isn't present
         potential_time_cols = [c for c in daily_gpr_pl.columns if c.startswith('date')]
         if potential_time_cols:
              time_col_name = potential_time_cols[0] # Use the first match
              logger.warning(f"Using inferred time column: {time_col_name}")
         else:
              raise KeyError("Could not find suitable date/time column in the results.")

    # Set index and sort (Polars sorts by default on query, but good practice)
    gpr_indices_df = daily_gpr_pl.to_pandas().set_index(time_col_name).sort_index()
    logger.info(f"Pandas DataFrame ready. Shape: {gpr_indices_df.shape}, Columns: {gpr_indices_df.columns}")
    logger.debug(f"GPR Indices DataFrame head:\n{gpr_indices_df.head()}")
else:
    logger.warning("GPR calculation resulted in an empty DataFrame.")
    # Create an empty DataFrame with expected columns if needed downstream
    expected_cols = ['overall_index'] + list(gpr_calculator.category_definitions.keys())
    gpr_indices_df = pd.DataFrame(columns=expected_cols)
    gpr_indices_df.index.name = 'date'


# %%
# --- No daily normalization needed for plots 1-4 ---

# %%
# --- No immediate weekly resampling here; done later for specific plots ---

# %%
# Plot overall GPR trend over time (Daily Counts - Plot 1)
# This section should now work correctly if gpr_indices_df was created successfully
logger.info("Plotting Daily GPR Counts (Plot 1)...")
plt.figure(figsize=(14, 7))
if "overall_index" in gpr_indices_df.columns:
    gpr_indices_df["overall_index"].plot(label="Overall GPR (Daily Count)")
    plt.title("Geopolitical Risk Index Over Time (Daily Article Counts)")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gpr_overall_trend_daily_counts.png"))
    plt.close()
else:
    logger.warning("Column 'overall_index' not found in gpr_indices_df. Skipping Plot 1.")

# %%
# Plot Weekly GPR (Counts - Plot 2)
# This section uses pandas resampling on gpr_indices_df, should work
logger.info("Calculating and Plotting Weekly GPR Counts (Plot 2)...")
if not gpr_indices_df.empty:
    gpr_indices_weekly_sum = gpr_indices_df.resample('W').sum()
    plt.figure(figsize=(14, 7))
    if "overall_index" in gpr_indices_weekly_sum.columns:
        gpr_indices_weekly_sum["overall_index"].plot(label="Overall GPR (Weekly Count)")
        plt.title("Geopolitical Risk Index Over Time (Weekly Article Counts)")
        plt.xlabel("Date (Week Ending)")
        plt.ylabel("Number of Articles")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "gpr_overall_trend_weekly_counts.png"))
        plt.close()
    else:
         logger.warning("Column 'overall_index' not found after weekly resampling. Skipping Plot 2.")
else:
    logger.warning("gpr_indices_df is empty. Skipping Plot 2.")


# %%
# ANALYSIS 1: Check for peak during Nancy Pelosi's visit (around 2022/8/5) - Daily Counts
# This uses the pandas gpr_indices_df, should work
logger.info("Analyzing Pelosi Visit peak using Daily Counts...")
pelosi_visit_date = pd.Timestamp("2022-08-05")
# Define window around the visit for daily data
window_start = pd.Timestamp("2022-07-25")
window_end = pd.Timestamp("2022-08-15")

# Filter the DAILY data for the Pelosi period
pelosi_period_daily = gpr_indices_df[
    (gpr_indices_df.index >= window_start) & (gpr_indices_df.index <= window_end)
]

if not pelosi_period_daily.empty:
    plt.figure(figsize=(10, 6))
    pelosi_period_daily["overall_index"].plot()
    plt.axvline(x=pelosi_visit_date, color="r", linestyle="--", label="Pelosi's Visit Date")
    plt.title("Daily GPR Counts Around Nancy Pelosi's Visit to Taiwan")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "pelosi_visit_gpr_daily_counts.png")
    )  
    plt.close()  

    # Find peak day around the visit
    peak_date = pelosi_period_daily["overall_index"].idxmax()
    peak_value = pelosi_period_daily["overall_index"].max()

    logger.info(
        f"Peak Daily GPR count around Pelosi's visit: {peak_date.date()} with value {peak_value}"
    )

    # Check if peak is close to Pelosi's visit date
    days_difference = abs((peak_date - pelosi_visit_date).days)
    logger.info(f"Days between peak and Pelosi's visit: {days_difference}")

    if days_difference <= 2: # Peak within 2 days before/after
        logger.info(
            "✓ CONFIRMED: The daily GPR count peaked around Nancy Pelosi's visit to Taiwan"
        )
    else:
        logger.info(
            "⚠ NOT CONFIRMED: The daily GPR count peak was not closely aligned with Pelosi's visit"
        )
else:
    logger.info("No data available for the period of Nancy Pelosi's visit")


# %% [markdown]
# ## Analysis 2: Newspaper Comparison (Weekly Counts)
# We query the calculator to get weekly counts grouped by newspaper.

# %%
# ANALYSIS 2: Newspaper Comparison (Weekly Counts - Plot 4)
logger.info("Analyzing GPR trends by newspaper (Weekly Counts - Plot 4)...")

# Calculate weekly GPR counts grouped by newspaper_en using the calculator
logger.info("Querying weekly GPR counts by newspaper...")
try:
    # Use 'newspaper_en' for grouping as added during prepare step
    newspaper_weekly_pl = gpr_calculator.query(
        group_by_cols=['date', 'newspaper_en'],
        time_bucket='W', # Get weekly buckets
        time_col='date'     # Specify the time column
    )
    logger.info(f"Weekly newspaper query successful. Shape: {newspaper_weekly_pl.shape}")

    # --- Convert Polars results to Pandas for existing plotting/analysis code ---
    if not newspaper_weekly_pl.is_empty():
        # Rename category columns (e.g., war_threat_count -> war_threat)
        rename_mapping = {f"{cat}_count": cat for cat in gpr_calculator.category_definitions.keys()}
        newspaper_weekly_pl = newspaper_weekly_pl.rename(rename_mapping)

        # Assume the time column is named 'date_bucketed' after query with time_bucket
        time_col_name = 'date_bucketed'
        if time_col_name not in newspaper_weekly_pl.columns:
             raise KeyError(f"Expected time bucket column '{time_col_name}' not found.")

        # Pivot the data to get newspapers as columns (suitable for comparison plot & correlation)
        # Overall Index Comparison DF
        comparison_df_weekly_pivoted_pl = newspaper_weekly_pl.pivot(
            index=time_col_name,
            columns='newspaper_en',
            values='overall_index'
        )
        # Convert to pandas and explicitly set the index
        comparison_df_weekly = comparison_df_weekly_pivoted_pl.to_pandas().set_index(time_col_name).sort_index()
        comparison_df_weekly.index.name = 'Date (Week Ending)' # Rename index for clarity
        logger.info(f"Created overall weekly comparison DataFrame. Shape: {comparison_df_weekly.shape}")
        logger.debug(f"Overall weekly comparison head:\\n{comparison_df_weekly.head()}")

        # Military Buildup Comparison DF (Handle potential missing category)
        if 'military_buildup' in newspaper_weekly_pl.columns:
            comparison_mb_weekly_pivoted_pl = newspaper_weekly_pl.pivot(
                index=time_col_name,
                columns='newspaper_en',
                values='military_buildup'
            )
            # Convert to pandas and explicitly set the index
            comparison_mb_weekly = comparison_mb_weekly_pivoted_pl.to_pandas().set_index(time_col_name).sort_index()
            comparison_mb_weekly.index.name = 'Date (Week Ending)'
            logger.info(f"Created military buildup weekly comparison DataFrame. Shape: {comparison_mb_weekly.shape}")
            logger.debug(f"Military buildup weekly comparison head:\\n{comparison_mb_weekly.head()}")
        else:
            logger.warning("'military_buildup' column not found in query result. Cannot create military buildup comparison.")
            comparison_mb_weekly = pd.DataFrame() # Empty DF

    else:
        logger.warning("Weekly newspaper query returned empty results.")
        comparison_df_weekly = pd.DataFrame()
        comparison_mb_weekly = pd.DataFrame()

except Exception as e:
    logger.error(f"Error during weekly newspaper analysis: {e}")
    comparison_df_weekly = pd.DataFrame()
    comparison_mb_weekly = pd.DataFrame()


# %%
# Prepare comparison dataframe for plotting (Weekly Counts) - This is now done above via pivot

# >>> Correlation Analysis (Overall Weekly Counts) <<<
# This part should work correctly with the pivoted comparison_df_weekly
if not comparison_df_weekly.empty and len(comparison_df_weekly.columns) > 1:
    logger.info("Calculating correlations based on Overall Weekly Counts...")

    # --- Pearson Correlation ---
    correlation_pearson = comparison_df_weekly.corr(method='pearson')
    logger.info("--- Pearson Correlation Matrix (Overall Weekly Counts) ---")
    logger.info(f"\n{correlation_pearson}")
    # Plot Pearson heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_pearson, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Pearson Correlation - Newspapers' Overall Weekly GPR Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "newspaper_correlation_pearson_weekly_overall.png"))
    plt.close()

    # --- Spearman Correlation ---
    correlation_spearman = comparison_df_weekly.corr(method='spearman')
    logger.info("--- Spearman Correlation Matrix (Overall Weekly Counts) ---")
    logger.info(f"\n{correlation_spearman}")
    # Plot Spearman heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_spearman, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spearman Correlation - Newspapers' Overall Weekly GPR Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "newspaper_correlation_spearman_weekly_overall.png"))
    plt.close()

else:
    logger.info("Not enough weekly data available to calculate overall newspaper correlations.")
# >>> End Correlation Analysis (Overall) <<<

# %%
# Plot the Weekly GPR Counts by newspaper (Plot 4)
# This should work with the pivoted comparison_df_weekly
if not comparison_df_weekly.empty:
    logger.info("Plotting Weekly GPR Counts by Newspaper (Plot 4)...")
    plt.figure(figsize=(14, 7))
    for newspaper_col in comparison_df_weekly.columns:
        comparison_df_weekly[newspaper_col].plot(label=newspaper_col)
    plt.title("Weekly GPR Counts by Newspaper")
    plt.xlabel("Date (Week Ending)")
    plt.ylabel("Number of Articles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "newspaper_gpr_trends_weekly_counts.png")
    )
    plt.close()
else:
    logger.info("No weekly data available to plot newspaper comparison.")

# %%
# Prepare comparison dataframe for Military Buildup (Weekly Counts) - Done above via pivot

# %%
# >>> Correlation Analysis (Military Buildup Weekly Counts) <<<
# This should work with the pivoted comparison_mb_weekly
if not comparison_mb_weekly.empty and len(comparison_mb_weekly.columns) > 1:
    logger.info("Calculating correlations based on Military Buildup Weekly Counts...")

    # --- Pearson Correlation ---
    mb_correlation_pearson = comparison_mb_weekly.corr(method='pearson')
    logger.info("--- Pearson Correlation Matrix (Military Buildup Weekly Counts) ---")
    logger.info(f"\n{mb_correlation_pearson}")
    # Plot Pearson heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mb_correlation_pearson, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Pearson Correlation - Newspapers' Weekly Military Buildup Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "newspaper_correlation_pearson_weekly_mb.png"))
    plt.close()

    # --- Spearman Correlation ---
    mb_correlation_spearman = comparison_mb_weekly.corr(method='spearman')
    logger.info("--- Spearman Correlation Matrix (Military Buildup Weekly Counts) ---")
    logger.info(f"\n{mb_correlation_spearman}")
    # Plot Spearman heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mb_correlation_spearman, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spearman Correlation - Newspapers' Weekly Military Buildup Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "newspaper_correlation_spearman_weekly_mb.png"))
    plt.close()

else:
    logger.info("Not enough weekly data available to calculate military buildup newspaper correlations.")
# >>> End Correlation Analysis (Military Buildup) <<<

# %%
# Plot the Weekly Military Buildup Counts by newspaper (New Plot)
# This should work with the pivoted comparison_mb_weekly
if not comparison_mb_weekly.empty:
    logger.info("Plotting Weekly Military Buildup Counts by Newspaper...")
    plt.figure(figsize=(14, 7))
    for newspaper_col in comparison_mb_weekly.columns:
        comparison_mb_weekly[newspaper_col].plot(label=newspaper_col)
    plt.title("Weekly Military Buildup Counts by Newspaper")
    plt.xlabel("Date (Week Ending)")
    plt.ylabel("Number of Articles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "newspaper_military_buildup_weekly_counts.png")
    )
    plt.close()
else:
    logger.info("No weekly military buildup data available to plot newspaper comparison.")

# %% [markdown]
# ## Analysis 3: Category Contribution (Weekly Counts)
# We query the calculator again to get weekly counts aggregated across all newspapers.

# %%
# ANALYSIS 3: Category Analysis (Weekly Counts - Plot 3)
logger.info("Analyzing category contributions (Weekly Counts - Plot 3)...")

# Query for weekly counts, grouped only by date
logger.info("Querying weekly GPR counts by category (all newspapers)...")
try:
    category_weekly_pl = gpr_calculator.query(
        group_by_cols=['date'],
        time_bucket='W', # Weekly buckets
        time_col='date'
    )
    logger.info(f"Weekly category query successful. Shape: {category_weekly_pl.shape}")

    if not category_weekly_pl.is_empty():
         # Rename category columns (e.g., war_threat_count -> war_threat)
         rename_mapping = {f"{cat}_count": cat for cat in gpr_calculator.category_definitions.keys()}
         category_weekly_pl = category_weekly_pl.rename(rename_mapping)

         # Assume time column is 'date_bucketed'
         time_col_name = 'date_bucketed'
         if time_col_name not in category_weekly_pl.columns:
              raise KeyError(f"Expected time bucket column '{time_col_name}' not found.")

         # Convert to pandas and set index
         category_weekly_sums = category_weekly_pl.to_pandas().set_index(time_col_name).sort_index()
         category_weekly_sums.index.name = 'Date (Week Ending)' # Rename index for clarity
         logger.info(f"Created weekly category sums DataFrame. Shape: {category_weekly_sums.shape}")
         logger.debug(f"Weekly category sums head:\n{category_weekly_sums.head()}")

    else:
         logger.warning("Weekly category query returned empty results.")
         category_weekly_sums = pd.DataFrame()


except Exception as e:
    logger.error(f"Error during weekly category analysis: {e}")
    category_weekly_sums = pd.DataFrame()


# Use the new category_weekly_sums DataFrame for analysis
if not category_weekly_sums.empty:
    # The rest of the dominance analysis and plotting code remains largely the same,
    # using the 'category_weekly_sums' pandas DataFrame.
    present_category_columns = [col for col in category_weekly_sums.columns if col != 'overall_index']

    if not present_category_columns:
        logger.warning("No category columns found in the weekly data for dominance analysis.")
    else:
        logger.info(f"Analyzing categories: {present_category_columns}")
        # category_weekly_sums = gpr_indices_weekly_sum[present_category_columns] # No longer needed

        # Calculate average weekly count for each category
        category_avg_weekly_counts = category_weekly_sums[present_category_columns].mean()
        sorted_categories = category_avg_weekly_counts.sort_values(ascending=False)

        logger.info("Categories by average weekly GPR count:")
        logger.info(sorted_categories)

        # Check if military categories dominate (based on average weekly counts)
        military_categories = [cat for cat in ["military_buildup"] if cat in present_category_columns]
        if military_categories:
            military_avg = category_avg_weekly_counts[military_categories].mean()
            non_military_categories = [
                cat for cat in present_category_columns if cat not in military_categories
            ]
            non_military_avg = (
                category_avg_weekly_counts[non_military_categories].mean()
                if non_military_categories
                else 0
            )

            logger.info(f"\nAverage weekly military count: {military_avg:.6f}")
            logger.info(f"Average weekly non-military count: {non_military_avg:.6f}")

            if military_avg > non_military_avg:
                ratio = military_avg / non_military_avg if non_military_avg > 1e-9 else float("inf")
                logger.info(f"Military/Non-military ratio: {ratio:.2f}")
                if ratio > 1.5:
                    logger.info("✓ CONFIRMED: Military buildup dominates other categories (based on avg weekly count)")
                else:
                    logger.info("⚠ PARTIALLY CONFIRMED: Military categories higher but not strongly dominant")
            else:
                logger.info("⚠ NOT CONFIRMED: Military categories do not dominate")
        else:
            logger.warning("Military category 'military_buildup' not found in data for dominance check.")

        # Plot categories over time (Weekly Counts)
        logger.info("Plotting Weekly GPR Counts by Category (Plot 3)...")
        plt.figure(figsize=(14, 7))
        for category in present_category_columns:
            plt.plot(category_weekly_sums.index, category_weekly_sums[category], label=category)
        plt.title("Weekly GPR Counts by Category")
        plt.xlabel("Date (Week Ending)")
        plt.ylabel("Number of Articles")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "category_trends_weekly_counts.png"))
        plt.close()

        # Plot stacked area chart (Weekly Counts)
        category_weekly_sums[present_category_columns].fillna(0).plot.area(stacked=True, figsize=(14, 7)) # Ensure only category cols are stacked
        plt.title("Composition of GPR Index Over Time (Weekly Counts)")
        plt.xlabel("Date (Week Ending)")
        plt.ylabel("Number of Articles")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "stacked_categories_weekly_counts.png"))
        plt.close()

        # Plot average weekly count by category
        plt.figure(figsize=(10, 6))
        category_avg_weekly_counts.sort_values(ascending=False).plot(kind="bar")
        plt.title("Average Weekly Count by Category")
        plt.ylabel("Average Weekly Article Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "category_contributions_weekly_counts.png"))
        plt.close()
else:
    logger.warning("Weekly category sums DataFrame is empty. Skipping category analysis.")

# %% [markdown]
# ## Analysis 4: Lau's Normalized Monthly Index
# We query the calculator for monthly counts and apply normalization.

# %%
# ANALYSIS 4: Prepare Lau's Normalized Monthly Index for later comparison (Plot 5)
logger.info("Calculating Lau's Normalized Monthly Index...")

# Query for monthly counts, grouped only by date
logger.info("Querying monthly GPR counts (all newspapers)...")
try:
    monthly_gpr_pl = gpr_calculator.query(
        group_by_cols=['date'],
        time_bucket='M', # Monthly buckets
        time_col='date'
    )
    logger.info(f"Monthly query successful. Shape: {monthly_gpr_pl.shape}")

    if not monthly_gpr_pl.is_empty():
        # Assume time column is 'date_bucketed'
        time_col_name = 'date_bucketed'
        if time_col_name not in monthly_gpr_pl.columns:
             raise KeyError(f"Expected time bucket column '{time_col_name}' not found.")

        # Convert to pandas Series (overall_index) with date index
        gpr_indices_monthly_sum_series = monthly_gpr_pl.to_pandas().set_index(time_col_name)['overall_index'].sort_index()
        gpr_indices_monthly_sum_series.index.name = 'Date (Month Ending)'
        logger.info(f"Created monthly sum series. Length: {len(gpr_indices_monthly_sum_series)}")

        # --- Apply Lau Normalization ---
        mean_monthly_count = gpr_indices_monthly_sum_series.mean()
        std_monthly_count = gpr_indices_monthly_sum_series.std()
        logger.info(f"Original Monthly Index (Sum): Mean={mean_monthly_count:.6f}, StdDev={std_monthly_count:.6f}")

        lau_normalized_monthly_index = pd.Series(dtype=float) # Initialize empty series
        if std_monthly_count > 1e-9:
            lau_normalized_monthly_index = (
                (gpr_indices_monthly_sum_series - mean_monthly_count) / std_monthly_count
            ) + mean_monthly_count # Add mean back

            final_mean = lau_normalized_monthly_index.mean()
            final_std = lau_normalized_monthly_index.std()
            logger.info(f"Lau's Normalized Monthly Index: Mean={final_mean:.6f}, StdDev={final_std:.6f}")
            logger.info("(Mean is preserved, StdDev is rescaled relative to original mean)") # Clarified description
        else:
            logger.warning("StdDev of monthly counts zero. Cannot normalize. Using original mean value.")
            lau_normalized_monthly_index = pd.Series(mean_monthly_count, index=gpr_indices_monthly_sum_series.index)
        # --- End Normalization ---

    else:
        logger.warning("Monthly GPR query returned empty results. Cannot calculate Lau normalized index.")
        lau_normalized_monthly_index = pd.Series(dtype=float)

except Exception as e:
    logger.error(f"Error during monthly analysis for Lau normalization: {e}")
    lau_normalized_monthly_index = pd.Series(dtype=float)


# --- Placeholder for Plot 5 (remains the same) ---
# logger.info("Plot 5 (Monthly Comparison) skipped: Caldara index not yet provided.")
# if not lau_normalized_monthly_index.empty:
# ... (plotting code remains the same) ...
# --- End Placeholder ---

# %% [markdown]
# ## Summary of Findings
#
# Based on the analysis, we can evaluate Yeu-Tong Lau's findings:
#
# 1. **Peak during Nancy Pelosi's visit (August 5, 2022)**: We examined the GPR index around this date to see if there was a significant peak.
#
# 2. **Similar trends across newspapers**: We calculated correlations between GPR indices from different newspapers to check for consistency.
#
# 3. **Military buildup dominance**: We compared the average contribution of military-related categories to other categories.
#
# ## Discussion on Articles in Multiple Categories
#
# When an article falls into multiple GPR categories, there are several possible approaches:
#
# 1. **Multiple counting**: Count the article once in each category it appears in. This approach emphasizes the multi-dimensional nature of geopolitical events but may inflate the overall counts.
#
# 2. **Primary category assignment**: Assign the article to its most relevant category based on criteria like:
#    - Keyword frequency
#    - Position of keywords in the text (e.g., title vs. body)
#    - Manual classification rules
#
# 3. **Fractional counting**: If an article is in N categories, count it as 1/N in each category. This preserves the total count but may dilute the strength of signals.
#
# 4. **Hierarchical classification**: Define a hierarchy of categories and assign articles to the highest-ranking category they match.
#
# In our implementation using the GPRCalculator, the approach is based on keyword frequency which provides a nuanced measure of category relevance. This allows articles to contribute to multiple categories with different weights based on the frequency of relevant terms.

logger.info(f"Analysis completed. All outputs saved to {OUTPUT_DIR}")
print(f"Analysis completed. All outputs saved to {OUTPUT_DIR}")
