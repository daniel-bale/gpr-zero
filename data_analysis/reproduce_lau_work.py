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

# Import gpr_zero modules directly (they are auto-installed by uv)
from gpr_zero.gpr_index.calculator import GPRCalculator
from gpr_zero.gpr_index.analyzer import GPRAnalyzer

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
parser.add_argument(
    "--divide_by_total",
    action="store_true",
    help="Divide GPR by total article count (default: False)",
)
parser.add_argument(
    "--total_count",
    type=int,
    default=None,
    help="Override total article count for GPR calculation",
)

# Handle both script and notebook contexts
if "ipykernel" in sys.modules:  # Running in a notebook
    DIVIDE_BY_TOTAL = False
    TOTAL_COUNT_OVERRIDE = None
else:  # Running as a script
    try:
        args, _ = parser.parse_known_args()
        DIVIDE_BY_TOTAL = args.divide_by_total
        TOTAL_COUNT_OVERRIDE = args.total_count
    except:
        # Fallback to defaults if argument parsing fails
        DIVIDE_BY_TOTAL = False
        TOTAL_COUNT_OVERRIDE = None
        print(
            "Argument parsing failed, using defaults: divide_by_total=False, total_count_override=None"
        )

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
conn = ibis.connect("duckdb://data/news/news.duckdb")

# %%
# List tables in the database
logger.info("Tables in the database:")
logger.info(conn.list_tables())

# Check if the news_articles table exists
if "news_articles" in conn.list_tables():
    # Get the news_articles table
    articles = conn.table("news_articles")
else:
    raise ValueError("news_articles table not found in database")

# %%
# Display the schema
logger.info("Articles table schema:")
logger.info(articles.schema())

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
    .aggregate(count=_.id.count())
    .order_by(ibis.desc("count"))
    .to_polars()
)
logger.info(news_counts)

# Count by date (recent dates)
logger.info("\nArticles by date (last 10 days):")
date_counts = (
    articles.group_by("date")
    .aggregate(count=_.id.count())
    .order_by(ibis.desc("date"))
    .limit(10)
    .to_polars()
)
logger.info(date_counts)

# Count by category
logger.info("\nTop 10 article categories:")
category_counts = (
    articles.group_by("category")
    .aggregate(count=_.id.count())
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
# Create a temporary dictionary file for the GPRCalculator
temp_dict_path = os.path.join(OUTPUT_DIR, "temp_dictionary.json")
with open(temp_dict_path, "w", encoding="utf-8") as f:
    json.dump(gpr_dict, f)


# Create the GPR calculator with option to disable division by total
class CustomGPRCalculator(GPRCalculator):
    def __init__(
        self,
        dictionary_path: str = None,
        divide_by_total: bool = True,
        total_count_override: int = None,
    ):
        super().__init__(dictionary_path=dictionary_path)
        self.divide_by_total = divide_by_total
        self.total_count_override = total_count_override

    def calculate_index_for_date(self, date: str, articles: dict) -> dict:
        """Override to optionally skip division by total article count"""
        # First calculate using the standard method
        result = super().calculate_index_for_date(date, articles)

        # If we don't want to divide by total, multiply back by the total count that was used
        if not self.divide_by_total:
            # If total_count_override is provided, use that instead
            total_count = (
                self.total_count_override
                if self.total_count_override
                else len(articles)
            )

            # Adjust the overall index
            if "overall_index" in result:
                result["overall_index"] *= total_count

            # Adjust category indices
            if "category_indices" in result and isinstance(
                result["category_indices"], dict
            ):
                for category in result["category_indices"]:
                    result["category_indices"][category] *= total_count

        return result


# Initialize the calculator with the division option
gpr_calculator = CustomGPRCalculator(
    dictionary_path=temp_dict_path,
    divide_by_total=DIVIDE_BY_TOTAL,
    total_count_override=TOTAL_COUNT_OVERRIDE,
)
logger.info(
    f"Using CustomGPRCalculator with divide_by_total={DIVIDE_BY_TOTAL}, total_count_override={TOTAL_COUNT_OVERRIDE}"
)

# %%
# Get all articles as a Polars DataFrame for processing
articles_df = articles.to_polars()

# Map newspaper names to English
articles_df = articles_df.with_columns(
    pl.col("newspaper")
    .map_dict(NEWSPAPER_MAP, default=pl.col("newspaper"))
    .alias("newspaper_en")
)

# Convert to dictionary format expected by GPRCalculator
articles_by_date = {}
for date in articles_df["date"].unique():
    date_str = date.strftime("%Y-%m-%d")
    articles_by_date[date_str] = {}

    date_articles = articles_df.filter(pl.col("date") == date)
    for row in date_articles.iter_rows(named=True):
        articles_by_date[date_str][row["id"]] = row["content"]

# %%
# Calculate GPR indices using the existing calculator
logger.info("Calculating GPR indices...")
# Add progress logging
total_dates = len(articles_by_date)
logger.info(f"Processing {total_dates} dates")
progress_step = max(1, total_dates // 10)  # Log progress every 10%

gpr_indices = {}
for i, (date, articles) in enumerate(articles_by_date.items()):
    result = gpr_calculator.calculate_index_for_date(date, articles)
    gpr_indices[date] = result

    # Log progress
    if (i + 1) % progress_step == 0 or i + 1 == total_dates:
        logger.info(
            f"Progress: {i + 1}/{total_dates} dates processed ({(i + 1) / total_dates * 100:.1f}%)"
        )

# Convert to pandas DataFrame
gpr_indices_df = pd.DataFrame.from_dict(gpr_indices, orient="index")
logger.info(f"Calculated indices for {len(gpr_indices_df)} dates")

# Ensure the index is a datetime type for proper plotting
gpr_indices_df.index = pd.to_datetime(gpr_indices_df.index)
gpr_indices_df = gpr_indices_df.sort_index()

# Replace the original gpr_indices with the dataframe
gpr_indices = gpr_indices_df

# %%
# Plot overall GPR trend over time
plt.figure(figsize=(14, 7))
gpr_indices["overall_index"].plot(label="Overall GPR")
plt.title("Geopolitical Risk Index Over Time")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "gpr_overall_trend.png")
)  # Save to output directory
plt.close()  # Close the figure to free memory

# %%
# ANALYSIS 1: Check for peak during Nancy Pelosi's visit (around 2022/8/5)
pelosi_visit = pd.Timestamp(
    "2022-08-05"
)  # Use pandas Timestamp instead of datetime.date
window_start = pd.Timestamp("2022-07-25")
window_end = pd.Timestamp("2022-08-15")

# Filter the data for the Pelosi period
pelosi_period = gpr_indices[
    (gpr_indices.index >= window_start) & (gpr_indices.index <= window_end)
]

if not pelosi_period.empty:
    plt.figure(figsize=(10, 6))
    pelosi_period["overall_index"].plot()
    plt.axvline(x=pelosi_visit, color="r", linestyle="--", label="Pelosi's Visit")
    plt.title("GPR Index Around Nancy Pelosi's Visit to Taiwan")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "pelosi_visit_gpr.png")
    )  # Save to output directory
    plt.close()  # Close the figure

    # Find peak around the visit
    peak_date = pelosi_period["overall_index"].idxmax()
    peak_value = pelosi_period["overall_index"].max()

    logger.info(
        f"Peak GPR around Pelosi's visit: {peak_date} with value {peak_value:.4f}"
    )

    # Check if peak is close to Pelosi's visit
    days_difference = abs((peak_date - pelosi_visit).days)
    logger.info(f"Days between peak and Pelosi's visit: {days_difference}")

    if days_difference <= 2:
        logger.info(
            "✓ CONFIRMED: The GPR index peaked around Nancy Pelosi's visit to Taiwan"
        )
    else:
        logger.info(
            "⚠ NOT CONFIRMED: The GPR peak was not closely aligned with Pelosi's visit"
        )
else:
    logger.info("No data available for the period of Nancy Pelosi's visit")

# %%
# ANALYSIS 2: Compare trends across different newspapers
logger.info("Analyzing GPR trends by newspaper...")
# Initialize GPR analyzer
gpr_analyzer = GPRAnalyzer(output_dir=os.path.join(OUTPUT_DIR, "newspaper_analysis"))

# Calculate GPR indices for each newspaper
newspapers = articles_df["newspaper"].unique().to_list()
newspaper_gpr = {}
newspaper_en_map = {}

for newspaper in newspapers:
    # Get English name
    en_name = NEWSPAPER_MAP.get(newspaper, newspaper)
    newspaper_en_map[newspaper] = en_name
    logger.info(f"Processing newspaper: {newspaper} (English: {en_name})")

    # Filter articles for this newspaper
    newspaper_articles = articles_df.filter(pl.col("newspaper") == newspaper)

    # Convert to dictionary format expected by GPRCalculator
    newspaper_articles_dict = {}
    for date in newspaper_articles["date"].unique():
        date_str = date.strftime("%Y-%m-%d")
        newspaper_articles_dict[date_str] = {}

        date_newspaper_articles = newspaper_articles.filter(pl.col("date") == date)
        for row in date_newspaper_articles.iter_rows(named=True):
            newspaper_articles_dict[date_str][row["id"]] = row["content"]

    # Skip if no articles
    if not any(newspaper_articles_dict.values()):
        logger.info(f"No articles for {newspaper}, skipping")
        continue

    # Calculate GPR indices for this newspaper
    newspaper_gpr[newspaper] = gpr_calculator.calculate_index_series(
        newspaper_articles_dict
    )
    logger.info(
        f"Calculated indices for {len(newspaper_gpr[newspaper])} dates for {newspaper}"
    )

# %%
# Prepare comparison dataframe for plotting with English names
comparison_df = None
for newspaper, gpr in newspaper_gpr.items():
    en_name = newspaper_en_map[newspaper]
    if comparison_df is None:
        comparison_df = gpr[["overall_index"]].rename(
            columns={"overall_index": en_name}
        )
    else:
        comparison_df[en_name] = gpr["overall_index"]

# %%
# Calculate correlations between newspapers to check for similar trends
if comparison_df is not None and len(comparison_df.columns) > 1:
    correlation_matrix = comparison_df.corr()

    # Calculate average correlation for each newspaper
    avg_correlations = correlation_matrix.mean()
    logger.info("Average correlation with other newspapers:")
    logger.info(avg_correlations)

    # Check if correlations are high (above 0.5)
    high_corr = avg_correlations[avg_correlations > 0.5]
    if len(high_corr) / len(avg_correlations) > 0.5:
        logger.info(
            "✓ CONFIRMED: Different newspapers show similar GPR trends (high correlation)"
        )
    else:
        logger.info(
            "⚠ NOT CONFIRMED: Newspapers show divergent GPR trends (low correlation)"
        )

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Newspapers' GPR Indices")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "newspaper_correlation.png")
    )  # Save to output directory
    plt.close()  # Close the figure

    # Plot the GPR indices by newspaper
    plt.figure(figsize=(14, 7))
    for newspaper in comparison_df.columns:
        # Resample to monthly for clearer visualization
        comparison_df[newspaper].resample("M").mean().plot(label=newspaper)
    plt.title("GPR Index by Newspaper")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "newspaper_gpr_trends.png")
    )  # Save to output directory
    plt.close()  # Close the figure
else:
    logger.info("Not enough data to compare newspapers")

# %%
# ANALYSIS 3: Check if military buildup dominates other categories
logger.info("Analyzing category dominance...")

# Extract category indices data with focus on the correct categories
category_indices = {}
for idx, row in gpr_indices.iterrows():
    # Handle potential missing data gracefully
    cat_indices = row.get("category_indices", {})
    if isinstance(cat_indices, dict):
        for category, value in cat_indices.items():
            # Only include the correct categories
            if category in CORRECT_CATEGORIES:
                if category not in category_indices:
                    category_indices[category] = []
                category_indices[category].append((idx, value))
    elif cat_indices is not None:
        logger.warning(
            f"Unexpected category_indices type for date {idx}: {type(cat_indices)}"
        )

# Log the categories found
found_categories = list(category_indices.keys())
logger.info(f"Found data for categories: {found_categories}")
missing_categories = [cat for cat in CORRECT_CATEGORIES if cat not in found_categories]
if missing_categories:
    logger.warning(f"No data found for categories: {missing_categories}")

# Convert to dataframes
category_dfs = {}
for category, values in category_indices.items():
    if values:
        df = pl.DataFrame(
            {"date": [v[0] for v in values], "value": [v[1] for v in values]}
        ).with_columns(pl.col("date").cast(pl.Date))
        category_dfs[category] = df
        logger.info(f"Created dataframe for {category} with {len(df)} rows")

# Calculate average index value for each category
category_avgs = {category: df["value"].mean() for category, df in category_dfs.items()}
sorted_categories = sorted(category_avgs.items(), key=lambda x: x[1], reverse=True)

logger.info("Categories by average GPR contribution:")
for category, avg in sorted_categories:
    logger.info(f"{category}: {avg:.6f}")

# Check if military categories dominate
military_categories = ["military_buildup"]
if military_categories:
    military_avg = sum(category_avgs.get(cat, 0) for cat in military_categories) / len(
        military_categories
    )
    non_military_categories = [
        cat for cat in category_avgs.keys() if cat not in military_categories
    ]
    non_military_avg = (
        sum(category_avgs[cat] for cat in non_military_categories)
        / len(non_military_categories)
        if non_military_categories
        else 0
    )

    logger.info(f"\nAverage military GPR: {military_avg:.6f}")
    logger.info(f"Average non-military GPR: {non_military_avg:.6f}")

    if military_avg > non_military_avg:
        ratio = (
            military_avg / non_military_avg if non_military_avg > 0 else float("inf")
        )
        logger.info(f"Military/Non-military ratio: {ratio:.2f}")

        if ratio > 1.5:
            logger.info("✓ CONFIRMED: Military buildup dominates other categories")
        else:
            logger.info(
                "⚠ PARTIALLY CONFIRMED: Military categories are higher but not strongly dominant"
            )
    else:
        logger.info("⚠ NOT CONFIRMED: Military categories do not dominate")

# Plot categories over time
plt.figure(figsize=(14, 7))
for category, df in category_dfs.items():
    if category == "military_buildup":
        # Highlight military categories
        plt.plot(df["date"], df["value"], linewidth=2, label=category)
    else:
        plt.plot(df["date"], df["value"], linewidth=1, alpha=0.7, label=category)
plt.title("GPR Categories Over Time")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "category_trends.png"))  # Save to output directory
plt.close()  # Close the figure

# %%
# Plot stacked area chart of category composition
# First, convert to pandas for easier plotting
category_data = {}
for category, df in category_dfs.items():
    pandas_df = df.to_pandas()
    pandas_df.set_index("date", inplace=True)
    category_data[category] = pandas_df["value"]

if category_data:
    category_panel = pd.DataFrame(category_data)

    # Fill NaN values with 0
    category_panel = category_panel.fillna(0)

    # Resample to monthly for clearer visualization
    category_monthly = category_panel.resample("M").mean()

    # Plot stacked area chart
    plt.figure(figsize=(14, 7))
    category_monthly.plot.area(stacked=True)
    plt.title("Composition of GPR Index Over Time")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "stacked_categories.png")
    )  # Save to output directory
    plt.close()  # Close the figure

    # Plot average contribution of each category
    plt.figure(figsize=(10, 6))
    category_panel.mean().sort_values(ascending=False).plot(kind="bar")
    plt.title("Average Contribution by Category")
    plt.ylabel("Average Index Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "category_contributions.png")
    )  # Save to output directory
    plt.close()  # Close the figure

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
