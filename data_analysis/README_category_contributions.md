# GPR Category Contribution Analysis

This directory contains a script to calculate and analyze how much each category contributes to the overall GPR index, comparing Lau's 2024 dictionary and the Taiwan president dictionary.

## Script Overview

The `calculate_category_contributions.py` script:

1. **Loads two key GPR dictionaries** (Lau's 2024 dictionary and Taiwan president dictionary)
2. **Calculates GPR shares** for each category across all newspapers and time periods
3. **Computes category contributions** following the definition: "percent of articles in each category satisfying GPR inclusion condition, as a share of all articles satisfying that condition"
4. **Identifies peak weeks** when each category had its highest share
5. **Creates visualizations** showing:
   - Individual category contributions per dictionary (with peak week annotations)
   - Timeline plots showing when each category peaked
   - Comparison plots across dictionaries
6. **Exports results** to CSV files with peak week information

## Usage

### Basic Usage

Run analysis for all configured dictionaries:
```bash
uv run data_analysis/calculate_category_contributions.py
```

### Run Specific Dictionaries

Analyze only the Lau dictionary:
```bash
uv run data_analysis/calculate_category_contributions.py --dictionaries lau_2024
```

### Available Dictionaries

The script focuses on two key dictionaries for comparison:

- `lau_2024`: Yeu-Tong Lau's 2024 GPR dictionary
- `president`: Taiwan GPR dictionary (President perspective)

### Run Specific Dictionaries

Compare Lau vs President dictionaries:
```bash
uv run data_analysis/calculate_category_contributions.py --dictionaries lau_2024 president
```

Analyze individual dictionaries:
```bash
# Lau dictionary only
uv run data_analysis/calculate_category_contributions.py --dictionaries lau_2024

# President dictionary only
uv run data_analysis/calculate_category_contributions.py --dictionaries president
```

### Add Custom Dictionaries

Add and analyze a custom dictionary:
```bash
uv run data_analysis/calculate_category_contributions.py \
  --add-dictionary custom /path/to/custom_dict.json \
  --dictionaries lau_2024 custom
```

## Output Files

The script generates several output files in `data_analysis/output/category_contributions/`:

### CSV Files
- `category_contributions_{dictionary_name}.csv` - Individual dictionary results
- `category_contributions_combined.csv` - Combined results from all dictionaries

### Visualizations
- `category_contributions_{dictionary_name}.png` - Bar charts for each dictionary (with peak week annotations)
- `peak_timeline_{dictionary_name}.png` - Timeline showing when each category peaked
- `category_contributions_comparison.png` - Side-by-side comparison of dictionaries

## Output Format

The CSV files contain the following columns:

| Column | Description |
|--------|-------------|
| `category` | Category name (e.g., "military", "political", "overall") |
| `contribution_percent` | Percentage contribution to GPR index |
| `category_articles` | Total articles in this category that satisfy GPR inclusion condition |
| `total_gpr_articles` | Total articles that satisfy GPR inclusion condition |
| `contribution_ratio` | Ratio of category articles to total GPR articles |
| `peak_week` | Week when this category had its highest share (YYYY-MM-DD format) |
| `peak_share` | Highest weekly share value for this category |
| `peak_articles` | Number of articles in the peak week for this category |
| `dictionary` | Dictionary name (in combined file only) |

## Analysis Method

The script calculates category contributions using these steps:

1. **Data Preparation**: Filters articles by date range and target newspapers
2. **GPR Calculation**: Uses the GPRCalculator to compute category shares for each time period
3. **Article Counting**: Converts shares to actual article counts for each category
4. **Contribution Calculation**: Calculates the percentage of articles in each category that satisfy the GPR inclusion condition, as a share of all articles satisfying that condition

**Mathematical Definition:**
- For each category: `contribution_percent = (category_articles_with_gpr_terms / total_articles_with_gpr_terms) × 100`
- Where articles can belong to multiple categories (overlapping counts are expected)
- The "overall" category represents 100% of all articles satisfying the GPR inclusion condition

**Peak Week Calculation:**
- Uses weekly time buckets (`TIME_BUCKET_GPR = 'W'`) following the same method as `capture_gpr_spikes.py`
- For each week, calculates weighted average share across all newspapers
- Peak week = week with the highest weighted average share for each category
- Peak weeks help identify when geopolitical tensions were highest for specific categories

## Configuration

### Adding New Dictionaries

Edit the `DICTIONARIES` configuration in the script:

```python
DICTIONARIES = {
    "lau_2024": "/home5/chuenho/gpr-zero/data/dictionaries/yeu_tong_lau_2024.json",
    "president": "/path/to/president_dict.json",
    "custom": "/path/to/custom_dict.json"
}
```

### Modifying Analysis Parameters

Key parameters you can adjust:

```python
# Date range for analysis
APP_START_DATE_STR = "2017-01-01"
APP_END_DATE_STR = "2024-12-31"

# Time aggregation (weekly analysis)
TIME_BUCKET_GPR = 'W'

# Target newspapers
NEWSPAPER_COMPARISON_MAP = {
    "中國時報": "China Times (中國時報)",
    "自由時報": "Liberty Times (自由時報)",
    "聯合報": "United Daily (聯合報)"
}
```

## Dependencies

The script requires:
- `ibis` for database operations
- `polars` for data manipulation
- `matplotlib` and `seaborn` for visualization
- `gpr_zero` package for GPR calculations

## Troubleshooting

### Database Lock Issues

If you encounter database lock errors, the script automatically tries to open the database in read-only mode. If this still fails, ensure no other processes are accessing the database.

### Missing Dictionary Files

The script will validate that all dictionary files exist before running. If files are missing, it will provide clear error messages with the expected paths.

### Memory Issues

For large datasets, consider:
- Reducing the date range
- Limiting the number of newspapers analyzed
- Running dictionaries individually rather than all at once

## Examples

### Comparing Lau vs President Dictionaries

```bash
uv run data_analysis/calculate_category_contributions.py --dictionaries lau_2024 president
```

This will generate:
- Individual contribution charts for each dictionary
- A comparison chart showing differences between dictionaries
- Combined CSV with all results

### Analyzing Individual Dictionaries

```bash
# Analyze Lau dictionary only
uv run data_analysis/calculate_category_contributions.py --dictionaries lau_2024

# Analyze President dictionary only  
uv run data_analysis/calculate_category_contributions.py --dictionaries president
```

This provides detailed analysis of individual dictionary characteristics.

### Analyzing a Single Dictionary

```bash
uv run data_analysis/calculate_category_contributions.py --dictionaries lau_2024
```

This provides detailed analysis of just the Lau dictionary categories. 