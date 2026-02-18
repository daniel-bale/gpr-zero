# Thesis Data Source Mapping

This document traces each data point, figure, and analytical result in the thesis (Chapters 1-4) back to its source script in the GPR-Zero project.

**Last verified**: 2026-02-16 (full pipeline re-run and comparison)

## Chapter 1: Introduction

### Data Claims
- **Line 46**: "3.3 million Mandarin articles (2017–2024)"
  - Source: Direct DB query on `data/news/stage2_enhanced_full.duckdb` with filters: 3 outlets, `warc_date` 2017-01-01 to 2024-12-31
  - Verification script: `scripts/analyze_domain_coverage.py` (operates on Stage 1 domain counts, not final DB)
  - **Verified**: 3,315,530 articles from 3 outlets with 2017-2024 date filter

## Chapter 2: Data and Dictionary Construction

### Section 2.1: Data Sources
- **Line 8**: "3,315,530 articles, consisting of 45% China Times, 38.5% United Daily, and 16.4% Liberty Times"
  - Source: Direct DB query with `warc_date >= '2017-01-01' AND warc_date < '2025-01-01'` on 3 outlets
  - Alternative: `scripts/create_article_count_comparison.py`
  - **Verified**: CT: 1,493,492 (45.0%), UD: 1,278,099 (38.5%), LT: 543,939 (16.4%)

### Section 2.1.1: Missing Data Patterns
- **Line 11**: "China Times has 68 missing weeks out of 418 (16.3%), Liberty Times has 44 missing weeks (10.5%), and United Daily has 74 missing weeks (17.7%)"
  - Source: `scripts/analyze_missing_data_2017_2024.py`
  - Input: `data/newspaper_gpr_step2/{chinatimes,ltn,udn}_step2_median_gpr.csv`
  - **Verified**: CT: 68 (16.3%), LT: 44 (10.5%), UD: 74 (17.7%)

### Section 2.2: LLM Dictionary Generation
- **Line 24**: "40 dictionary generations (4 models × 5 roles × 2 runs), but 5 failed validation by using forbidden OR operators, yielding 35 valid dictionaries"
  - Source: `scripts/batch_generate_step1.py`
  - Dictionary count verification: `scripts/count_dictionaries_by_model.py`
  - Excluded dicts: `EXCLUDE_STEP1` list in `scripts/calculate_correlations_final.py`
  - **Verified**: 35 batch (10 claude + 10 gemini + 5 gpt-4.1 + 10 gpt-4o) + 5 taiwan manual (gpt-4.1, generated 2025-05-09) = 40 Step1 dicts. 5 ill-format → 35 valid. Batch had `runs: 1` for gpt-4.1 because taiwan manual dicts already existed.
  - **DISCREPANCY**: Thesis says all 5 failed due to "forbidden OR operators". Actually: 3 gemini dicts have OR operators in category queries, 2 claude-opus-4 dicts reference non-existent word_set `international_relations`.

- **Line 35**: "35 dictionaries collectively generate 172 unique category names"
  - Source: `scripts/extract_categories_step1.py` (reads from `data/dictionaries/combined_step1/`, 40 dicts = 35 batch + 5 taiwan manual)
  - Output: `data/dictionaries/combined_step1/unique_categories.txt` (171 entries from all 40 dicts)
  - Analysis: `docs/category_mapping_analysis.md` (maps all categories to 8 representative groups)
  - **DISCREPANCY**: Correct category count is **150** (not 172). 35 valid dicts (30 valid batch + 5 taiwan manual) → 150 unique categories.

### Section 2.3: Result Dictionary
- **Table 1**: Dictionary example (Line 80-123)
  - Source dictionaries: `data/dictionaries/all_dictionaries/`
  - Dictionary generation: `scripts/batch_generate_step2.py`

## Chapter 3: Empirical Results

### Section 3.1: GPR Index
- **Line 18**: "mean pairwise correlation among the 40 Step2 indices is 0.698"
  - Source: `scripts/compute_gpr_correlations_step2_expanded.py`
  - Data: `data/gpr_correlations/combined_gpr_timeseries_step2_expanded.csv` filtered to 2017-2024
  - **Verified**: 0.6979 (rounds to 0.698)

- **Line 18**: "mean pairwise correlation among the 35 Step1 indices is 0.493"
  - Source: `scripts/compute_gpr_correlations_step2_expanded.py` (same script, step1 subset)
  - **Verified**: 0.4928 (rounds to 0.493)

- **Line 20**: "correlation between our Step2 median index and Lau's index is 0.611"
  - Source: Computed from `data/gpr_indices_3outlets/step2_median_gpr_3outlets.csv` merged with Lau parquet, filtered to 2017-2024
  - Alternative: `scripts/compute_gpr_correlations_fixed.py`
  - **Verified**: 0.6107 (rounds to 0.611)

### Figures
- **Figure 3.1** (Line 34): `gpr_step2_median_scaled_clean.png`
  - Source: `scripts/create_gpr_step2_median_clean.py`
  - Input: `data/gpr_correlations/median_timeseries/median_gpr_step2_scaled.csv`
  - **Verified**: Regenerated successfully

- **Figure 3.2** (Line 42): `gpr_step2_median_with_spikes_clean.png`
  - Source: `scripts/create_gpr_step2_median_with_spikes.py`
  - Input: `data/gpr_indices_3outlets/step2_median_gpr_3outlets.csv` + `data/gpr_spikes/unified_detection_3outlets/step2_median_spikes.csv` + 40 3-outlet parquets
  - **Verified**: Regenerated successfully

- **Figure 3.3** (Line 50): `lau_vs_step2_comparison_clean.png`
  - Source: `scripts/create_lau_comparison_plot_clean.py`
  - Input: `data/gpr_correlations/median_timeseries/median_gpr_step2_scaled.csv` + `data/gpr_correlations/lau_comparison/lau_gpr_scaled.csv`
  - **Verified**: Regenerated successfully

- **Figure 3.4** (Line 58): `category_gpr_indices_clean_unscaled.png`
  - Source: `scripts/create_category_gpr_plot_final.py`
  - Input: `data/category_gpr_indices/category_median_gpr_unscaled.csv`
  - **Verified**: Regenerated successfully
  - **FIX**: Was incorrectly mapped to `create_category_gpr_plot_fixed.py`

### Section 3.2: Unique-Spike Event Analysis
- **Table 2** (Line 72-94): Unique spikes table
  - Spike detection: `scripts/detect_gpr_spikes_unified.py`
  - Spike analysis: `scripts/analyze_step2_spike_articles_normalized.py`
  - Summaries generation: `scripts/generate_spike_summaries_llm.py`
  - Table creation: `scripts/create_step2_unique_spikes_summary.py`
  - **Verified**: 21 Step2 spikes detected, 90% validation match with Lau

### Section 3.3: Keyword & Category Decomposition
- **Table 3** (Line 111-136): Top words in spikes
  - Source: `scripts/calculate_spike_word_lift_scores_v3.py`
  - Input: binary matrix table + spike CSVs + all 80 dictionaries for word mapping
  - Output: `data/word_contributions/step2_top_words_by_lift.csv`
  - **Verified**: Regenerated successfully

- **Figure 3.5** (Line 142): `category_median_ratios_spike_periods_bar.png`
  - Source: `scripts/create_category_stacked_bar_medians.py`
  - Input: `data/category_gpr_indices/category_median_gpr_unscaled.csv` + spike CSV + `data/gpr_indices_3outlets/step2_median_gpr_3outlets.csv`
  - **Verified**: Regenerated successfully

- **Figure 3.6** (Line 150): `unique_spikes_category_stacked_bar.png`
  - Source: `scripts/create_unique_spikes_category_stacked_bar.py`
  - Input: `data/category_gpr_indices/category_median_gpr_unscaled.csv` + unique spikes summary + step2 median 3-outlets
  - **Verified**: Regenerated successfully

### Section 3.4: Source-Heterogeneity Examination
- **Line 164**: "Pearson correlation between China Times and Liberty Times is 0.475, between China Times and United Daily is 0.78 and between Liberty Times and United Daily is 0.559"
  - Source: `scripts/create_clean_newspaper_comparison.py`
  - **DISCREPANCY**: The script produces CT-LT: **0.559**, CT-UD: **0.780**, LT-UD: **0.475**. The thesis has CT-LT and LT-UD values swapped.
  - **FIX**: Was incorrectly mapped to `scripts/calculate_newspaper_specific_step2_gpr.py` (which produces different values: CT-LT: 0.581, CT-UD: 0.737, LT-UD: 0.420)

- **Figure 3.7** (Line 179): `gpr_comparison_newspapers_clean_2017_01_to_2024_12.png`
  - Source: `scripts/create_clean_newspaper_comparison.py`
  - Input: `data/newspaper_gpr_step2/{chinatimes,ltn,udn}_step2_median_gpr.csv`
  - **Verified**: Regenerated successfully

- **Figure 3.8** (Line 188): `article_count_daily_avg_comparison_clean.png`
  - Source: `scripts/create_article_count_comparison.py`
  - Input: `data/newspaper_gpr_step2/{chinatimes,ltn,udn}_step2_all_dictionaries.parquet`
  - **Verified**: Regenerated successfully

- **Table 4** (Line 196): Relative word usage between newspapers
  - Source: `scripts/analyze_newspaper_word_ratios_all_gpr.py`
  - Input: binary matrix table + all 80 dictionaries for word mapping
  - Output: `data/newspaper_gpr_step2/word_frequency_ratios_all_gpr.csv`
  - **Verified**: Regenerated successfully

- **Table 5** (Line 225): Cross-strait security framing
  - Source: `data_analysis/sample_divergent_articles.py`
  - Alternative: `data_analysis/sample_divergent_articles_cached.py`
  - **FIX**: Was incorrectly mapped to `scripts/sample_divergent_articles.py` (wrong directory)

- **Figure 3.9** (Line 259): `adiz_shares_vs_invasions_weekly.png`
  - Source: `data_analysis/create_adiz_mention_index.py`
  - Input: `data/news/stage2_enhanced_full.duckdb` + `external/ADIZ_invasion.csv`
  - **Verified**: Regenerated successfully (identical file size)

### Section 3.5: VAR Analysis
- **Figure 3.10** (Line 279): `svar_irf_custom.png`
  - Source: `data_analysis/svar_analysis.py --gpr-version gpr_taiwan_step2_median --lags 2`
  - Data preparation: `data_preprocessing/build_master_df.py --gpr-version gpr_taiwan_step2_median`
  - Input: `data_analysis/output/svar_preprocessing/var_ready_gpr_taiwan_step2_median.parquet`
  - GPR source: `data/gpr_indices.duckdb` table `gpr_taiwan_step2_median` (442 rows, weekly)
  - Economic data: `data_preprocessing/master_quarterly_df.parquet` (start-of-quarter dates)
  - **FIX**: Was incorrectly mapped to `scripts/prepare_step2_median_for_svar.py`, which is a broken simplified version that uses pandas `resample('Q')` producing end-of-quarter dates that don't match the master data's start-of-quarter dates. The actual working script is `data_preprocessing/build_master_df.py`, which uses Polars `group_by_dynamic` with string-based quarter labels for correct joining.
  - **Verified**: Regenerated successfully from existing preprocessed data

## Chapter 4: Conclusion
- No specific data claims requiring source mapping

## Data Pipeline Dependency Chain

```
Level 0: Articles DB
   data/news/stage2_enhanced_full.duckdb (articles table)
   Created by: stage1_scan_warc.py → stage2_parse_html_load_db.py

Level 1: Binary Matrix (CACHED TABLE)
   Table: gpr_binary_matrix_all_words (6,441,601 rows, 3,341 word columns)
   Created by: scripts/compute_all_dictionary_timeseries.py
   Uses: GPRCalculatorBinaryMatrix with 80 dicts from data/dictionaries/all_dictionaries/

Level 2A: Dictionary Timeseries (all outlets)
   data/gpr_timeseries/*_W_overall.parquet (~76 files)
   Created by: scripts/compute_all_dictionary_timeseries.py

Level 2B: Dictionary Timeseries (3 outlets only)
   data/gpr_timeseries_3outlets/*_W_overall_3outlets.parquet (~76 files)
   Created by: scripts/compute_all_dictionary_timeseries_3outlets.py

Level 3A: Combined Timeseries
   data/gpr_correlations/combined_gpr_timeseries_step2_expanded.csv
   Created by: scripts/combine_timeseries_step2_expanded.py
   Input: Level 2A parquets

Level 3B: Median Timeseries
   data/gpr_correlations/median_timeseries/median_gpr_step2_scaled.csv
   Created by: scripts/create_gpr_median_plots.py
   Input: Level 3A combined CSV

Level 3C: Step2 Median (3 outlets)
   data/gpr_indices_3outlets/step2_median_gpr_3outlets.csv
   Created by: scripts/compute_step2_median_3outlets.py
   Input: Level 2B (40 step2 parquets)

Level 3D: Category GPR
   data/category_gpr_indices/category_median_gpr_unscaled.csv
   Created by: scripts/calculate_category_gpr_complete.py
   Input: Level 1 binary matrix + 40 Step2 dicts

Level 3E: Newspaper GPR
   data/newspaper_gpr_step2/{chinatimes,ltn,udn}_step2_median_gpr.csv
   Created by: scripts/calculate_newspaper_specific_step2_gpr.py
   Input: DB articles table + Step2 dictionaries

Level 4: Spike Detection
   data/gpr_spikes/unified_detection_3outlets/step2_median_spikes.csv
   Created by: scripts/detect_gpr_spikes_unified.py
   Input: Level 3C step2 median + Lau parquet

Level 5: Thesis Figures & Tables (see Figure/Table entries above)
```

## Figure → Data Dependencies

| Figure | Primary Input | Level |
|--------|--------------|-------|
| 3.1 | median_gpr_step2_scaled.csv | 3B |
| 3.2 | step2_median_gpr_3outlets.csv + spikes.csv + 40 3-outlet parquets | 3C + 4 + 2B |
| 3.3 | median_gpr_step2_scaled.csv + lau_gpr_scaled.csv | 3B |
| 3.4 | category_median_gpr_unscaled.csv | 3D |
| 3.5 | category_median_gpr_unscaled.csv + spikes.csv + step2_median_gpr_3outlets.csv | 3D + 4 + 3C |
| 3.6 | category_median_gpr_unscaled.csv + unique_spikes_summary + step2_median_gpr_3outlets.csv | 3D + 4 + 3C |
| 3.7 | {chinatimes,ltn,udn}_step2_median_gpr.csv | 3E |
| 3.8 | {chinatimes,ltn,udn}_step2_all_dictionaries.parquet | 3E |
| 3.9 | stage2_enhanced_full.duckdb + ADIZ_invasion.csv | 0 + external |
| 3.10 | var_ready_gpr_taiwan_step2_median.parquet | SVAR preprocessing |

## Additional Scripts for Thesis Figure Generation
- Master script for all thesis figures: `scripts/create_thesis_figures.py`
- Regeneration script: `scripts/regenerate_thesis_figures.py`

## Data Pipeline Overview
1. **Stage 1**: WARC scanning → `scripts/stage1_scan_warc.py`
2. **Stage 2**: HTML parsing → `scripts/stage2_parse_html_load_db.py`
3. **Dictionary generation**: `scripts/batch_generate_step1.py`, `scripts/batch_generate_step2.py`
4. **Binary matrix**: `scripts/compute_all_dictionary_timeseries.py`
5. **GPR calculation**: `src/gpr_zero/gpr_index/calculator.py`, `src/gpr_zero/gpr_index/calculator_binary_matrix.py`
6. **Intermediate data**: See dependency chain above
7. **Analysis & visualization**: Various scripts in `scripts/` and `data_analysis/`

## Key Databases
- Main article database: `data/news/stage2_enhanced_full.duckdb`
- Binary matrix cache: `gpr_binary_matrix_all_words` table (in same DuckDB)
- GPR index store: `data/gpr_indices.duckdb` (tables: `gpr_taiwan_step2_median`, `yeu_tong_lau_2024`, etc.)

## Verification Notes (2026-02-16)

### Corrections Applied
1. **Figure 3.4**: Changed source from `create_category_gpr_plot_fixed.py` to `create_category_gpr_plot_final.py`
2. **Table 5**: Changed path from `scripts/sample_divergent_articles.py` to `data_analysis/sample_divergent_articles.py`
3. **Line 164 correlations**: Changed source from `calculate_newspaper_specific_step2_gpr.py` to `create_clean_newspaper_comparison.py`
4. **Figure 3.10 data prep**: Changed from `scripts/prepare_step2_median_for_svar.py` to `data_preprocessing/build_master_df.py --gpr-version gpr_taiwan_step2_median`
5. Added dependency chain documentation
6. Added intermediate data creators (create_gpr_median_plots.py, compute_step2_median_3outlets.py, calculate_category_gpr_complete.py)

### Known Discrepancies
1. **Ill-format reason (Line 24)**: Thesis says all 5 failed due to "forbidden OR operators". Actually: 3 gemini dicts have OR operators, 2 claude-opus-4 dicts reference non-existent word_set (`international_relations`).
2. **Category count (Line 35)**: Thesis says 172 unique categories. Correct: **150** (from 35 valid dicts).
3. **Newspaper correlations (Line 164)**: Thesis has CT-LT (0.475) and LT-UD (0.559) values swapped. Script produces CT-LT: 0.559, CT-UD: 0.780, LT-UD: 0.475

### 5 Ill-Format Step1 Dicts — Pipeline Impact
The 5 excluded dicts (3 gemini with OR operators, 2 claude-opus-4 with missing word_set refs) are in `all_dictionaries/` and were processed by the pipeline:
- **Binary matrix words**: OK — individual words correctly included in 3,341 columns
- **Original timeseries** (`calculator_binary_matrix.py`): Crashed with KeyError — no parquet output
- **3-outlets timeseries** (`compute_all_dictionary_timeseries_3outlets.py`): Produced parquets but with **incorrect GPR** — OR categories silently evaluate to False
- **Step2 median index** (main thesis index): **Not affected** — only uses step2_* dicts
- **Thesis correlations** (`calculate_correlations_final.py`): **Properly excluded** via `EXCLUDE_STEP1` list
- **Other Step1 scripts** (plotting, general correlations): May include incorrect data from these 5 dicts

### All Statistics Verified
- Article count: 3,315,530 (45% CT, 38.5% UD, 16.4% LT)
- Missing data: CT 68/16.3%, LT 44/10.5%, UD 74/17.7%
- Step2 correlation: 0.698
- Step1 correlation: 0.493
- Lau correlation: 0.611
- All Step2 median calculations use 40 dictionaries from 4 models × 10 runs
- Spike identification uses 1 standard deviation threshold above mean
- Correlation calculations filter to 2017-2024 (418 weeks)
- Word frequency ratios compare spike periods vs entire dataset
