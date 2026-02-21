# Thesis Data Source Mapping

This document traces each data point, figure, and analytical result in the thesis (Chapters 1-4) back to its source script in the GPR-Zero project.

**Last verified**: 2026-02-20 (section-by-section completeness audit)

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
  - **Note on 418 weeks**: Script uses `pd.date_range(start='2017-01-01', end='2024-12-31', freq='W-MON')` which yields 419 Monday-start weeks (2016-12-26 to 2024-12-30). Thesis rounds to 418.

- **Line 13**: "United Daily appears to have been added to the crawling list only after 2018" with 47 consecutive missing weeks in 2017
  - Source: `scripts/analyze_missing_data_2017_2024.py` (gap analysis output)
  - Input: `data/newspaper_gpr_step2/udn_step2_median_gpr.csv` (weekly GPR count by newspaper)
  - **Verified**: 47 consecutive zero-article weeks for UD in 2017 visible in weekly newspaper GPR data

- **Line 15**: "China Times currently blocks CCBot in their robots.txt" causing increasing missing data from late 2023 onwards
  - Source: Observational (robots.txt inspection), not script-generated

### Section 2.2: LLM Dictionary Generation
- **Line 24**: "four models (Claude-Opus-4, Gemini-2.5-Pro, GPT-4.1, GPT-4o)"
  - Actual model IDs (from `scripts/batch_generate_step1.py` lines 18-23):
    - `anthropic/claude-opus-4`
    - `google/gemini-2.5-pro`
    - `openai/gpt-4.1`
    - `openai/gpt-4o-2024-11-20`
  - **Verified**: Thesis display names match code model keys

- **Line 24**: "five distinct roles: president, defense minister, NSB director, political science professor, journalist"
  - Source: `scripts/batch_generate_step1.py` ROLE_MAP (lines 26-32)
  - Actual mapping:
    - `president` → "Taiwanese President"
    - `defense_minister` → "Taiwanese Minister of Defense"
    - `journalist` → "Senior Taiwanese Journalist specializing in Cross-Strait Relations"
    - `nsb_director` → "Director of Taiwan's National Security Bureau"
    - `poli_professor` → "Taiwanese Politics Professor"
  - **Verified**: All 5 roles match

- **Line 24**: "40 dictionary generations (4 models × 5 roles × 2 runs), but 5 failed validation by using forbidden OR operators, yielding 35 valid dictionaries"
  - Source: `scripts/batch_generate_step1.py`
  - Dictionary count verification: `scripts/count_dictionaries_by_model.py`
  - Excluded dicts: `EXCLUDE_STEP1` list in `scripts/calculate_correlations_final.py`
  - **Verified**: 35 batch (10 claude + 10 gemini + 5 gpt-4.1 + 10 gpt-4o) + 5 taiwan manual (gpt-4.1, generated 2025-05-09) = 40 Step1 dicts. 5 ill-format → 35 valid. Batch had `runs: 1` for gpt-4.1 because taiwan manual dicts already existed.
  - **DISCREPANCY**: Thesis says all 5 failed due to "forbidden OR operators". Actually: 3 gemini dicts have OR operators in category queries, 2 claude-opus-4 dicts reference non-existent word_set `international_relations`.

- **Line 35**: "35 dictionaries collectively generate 172 unique category names"
  - Source: `scripts/extract_categories_step1.py` (reads from `data/dictionaries/combined_step1/`, 40 dicts = 35 batch + 5 taiwan manual)
  - Output: `data/dictionaries/combined_step1/unique_categories.txt` (171 entries from all 40 dicts)
  - Analysis: `docs/category_mapping_analysis.md` (maps all categories to 8 representative groups, states 172)
  - **DISCREPANCY**: Thesis says "35 dictionaries collectively generate 172 unique category names" but 172 is from all 40 dicts (including 5 ill-format). From the 35 valid dicts only, the correct count is **150**.

### Section 2.2.2: Step 2 Fixed-Category Consolidation
- **Lines 42-50**: Eight fixed categories:
  1. Military Escalation
  2. Internal Political Instability
  3. Economic Coercion
  4. Diplomatic Isolation
  5. Cyber and Information Warfare
  6. US-China-Taiwan Relations
  7. Regional Security
  8. Gray Zone Operations
  - Source: `prompts/generate_taiwan_gpr_dictionary_stable_8categories.json`
  - Consolidation analysis: `docs/category_mapping_analysis.md`

- **Line 55**: "Each model runs 10 times, producing 40 dictionaries"
  - Source: `scripts/batch_generate_step2.py` (4 models × 10 runs = 40 Step2 dicts)
  - **Verified**: 40 Step2 dictionaries in `data/dictionaries/all_dictionaries/step2_*`

### Section 2.3: Result Dictionary
- **Table 1**: Dictionary example (Line 80-123)
  - Source dictionaries: `data/dictionaries/all_dictionaries/`
  - Dictionary generation: `scripts/batch_generate_step2.py`
  - Exact file: `data/dictionaries/all_dictionaries/step2_claude-opus-4_stable8cat_run1.json` (generated 2025-07-27)
  - Footnote (Line 86): "This example dictionary is from Claude-Opus-4, run 1 of the Step 2 generation"
  - Panel A: 8 categories with search query structures (AND operators between word sets)
  - Panel B: 7 example word sets with specific Chinese keywords (selected examples from full dictionary)

- **Line 70**: "Lau's GPR index containing one dominant category contributing 86% of the index"
  - Source: `data_analysis/calculate_category_contributions.py` → output `data_analysis/output/category_contributions/category_contributions_lau_2024.csv`
  - Lau has 4 categories: war_threat, peace_threat, military_buildup, war_begin
  - Formula: category_articles / total_gpr_articles, using `GPRCalculator` on `stage2_enhanced_full.duckdb`, filtered to 3 newspapers (CT, LT, UD), 2017-2024
  - **Verified**: military_buildup = 52,133 / 60,485 = **86.19%** (rounds to 86%)
  - Note: All-outlet data (moved to backup) gave 84.1% (104,403/124,077) — incorrect denominator

- **Line 71**: "LLM yields more balanced set of categories, with the top contributor accounting for only 42%"
  - Source: `data/category_gpr_indices/category_median_gpr_unscaled.csv` (3-outlet, sum of category weekly values / sum across all)
  - Verification script: `data_analysis/verify_step2_category_contributions_3outlets.py`
  - **Median index top contributor**: Military Escalation at 36.0% (sum-based GPR index share from `category_median_gpr_unscaled.csv`)
  - **Per-dict analysis (3-outlet)** (40 Step2 dicts, binary matrix with 3-outlet filter, `category_articles / total_gpr_articles`):
    - Max of max: 58.68% (gpt-4.1 run6, Military Escalation)
    - Median top contribution across 40 dicts: 34.87%
    - Mean: 38.37%
    - Dicts near 42%: gemini run10 (41.92%, Internal Political Instability), gpt-4.1 run3 (41.70%), gpt-4o run7 (40.74%)
    - Note: gemini run8 gives 100% (anomalous — all 7,468 GPR articles match Diplomatic Isolation only)
  - **Per-dict analysis (all-outlet)** (same formula, `data/gpr_timeseries/step2_*_W_overall.parquet`):
    - Max of max: 58.20%, Median: 34.93%, Mean: 37.18%
  - **Median per category (3-outlet)**: Military Escalation median=32.33%, Economic Coercion median=18.14%, US-China-Taiwan Relations median=15.75%
  - **DISCREPANCY — NO AGGREGATE METHOD YIELDS 42%**: The median index gives 36.0%, per-dict median gives 34.9% (3-outlet). Max-of-max is 58.7% — too high. Individual dicts can reach ~42% but no standard aggregate produces this value. Thesis may need correction to ~36%.

## Chapter 3: Empirical Results

### Section 3.1: GPR Index
- **Line 18**: "mean pairwise correlation among the 40 Step2 indices is 0.698"
  - Source: `scripts/compute_gpr_correlations_step2_expanded.py`
  - Data: `data/gpr_correlations/combined_gpr_timeseries_step2_expanded.csv` filtered to 2017-2024
  - **Verified**: 0.6979 (rounds to 0.698)
  - **DISCREPANCY — ALL-OUTLET vs 3-OUTLET**: This value (0.698) is computed from **all-outlet** timeseries (`data/gpr_timeseries/`), but the thesis states at the beginning that only 3-outlet data is used. The 3-outlet equivalent (`data/gpr_timeseries_3outlets/`) produces Step2 mean correlation = **0.700**, not 0.698. Either: (a) update thesis to use 3-outlet value 0.700, or (b) clarify that correlation analysis uses all-outlet data as an exception.

- **Line 18**: "mean pairwise correlation among the 35 Step1 indices is 0.493"
  - Source: `scripts/compute_gpr_correlations_step2_expanded.py` (same script, step1 subset)
  - **Verified**: 0.4928 (rounds to 0.493)
  - **DISCREPANCY — ALL-OUTLET vs 3-OUTLET**: Same issue as Step2 above. This value (0.493) is from **all-outlet** data. The 3-outlet equivalent is **0.440**, a substantially different value. This is a larger gap than Step2 and requires resolution.

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
- **`tab:unique-spikes`** (Line 72-94): Major unique spikes in the GPR index (11 events)
  - Spike detection: `scripts/detect_gpr_spikes_unified.py`
  - Spike analysis: `scripts/analyze_step2_spike_articles_normalized.py`
  - Summaries generation: `scripts/generate_spike_summaries_llm.py` (prompt: `prompts/analyze_gpr_spike_event.json`)
  - Summaries consolidation: `scripts/create_spike_summaries_csv.py` → `data/gpr_spikes/step2_all_spikes_summary.csv`
  - Table creation: `scripts/create_step2_unique_spikes_summary.py`
  - **Verified**: 21 Step2 spikes detected total (in `step2_median_spikes.csv`), 11 unique events shown in table
  - Event descriptions are LLM-generated summaries from sampled spike-period articles

### Section 3.3: Keyword & Category Decomposition
- **Table 2 / `tab:top-words`** (Line 111-136): Top words in spikes
  - Source: `scripts/calculate_spike_word_lift_scores_v3.py`
  - Input: binary matrix table + spike CSVs + all 80 dictionaries for word mapping
  - Output: `data/word_contributions/step2_top_words_by_lift.csv`
  - Header: "Our GPR Index (21 spikes)" vs "Lau's GPR Index (22 spikes)"
  - Shows 10 words per index with frequency ratio and spike count
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

- **Table 3 / `tab:word-ratios`** (Line 196): Relative word usage between newspapers
  - Source: `scripts/analyze_newspaper_word_ratios_all_gpr.py`
  - Input: binary matrix table + all 80 dictionaries for word mapping
  - Output: `data/newspaper_gpr_step2/word_frequency_ratios_all_gpr.csv`
  - **Verified**: Regenerated successfully

- **Table 4 / `tab:newspaper_stance`** (Line 225): Cross-strait security framing
  - Full chain:
    1. Divergent week identification: `data_analysis/identify_divergent_weeks.py` → `data_analysis/output/divergent_coverage/divergent_weeks_detailed.json`
    2. Article sampling: `data_analysis/sample_divergent_articles.py` (alt: `data_analysis/sample_divergent_articles_cached.py`)
    3. LLM summarization: `data_analysis/process_divergent_summaries.py` (prompt: `prompts/summarize_divergent_coverage.json`)
    4. Editorial analysis: `data_analysis/analyze_editorial_differences.py` → `editorial_differences.csv`
    5. Final visualization: `data_analysis/create_final_divergence_visualization.py` → `academic_summary_table.csv`
    6. Chinese analysis: `data_analysis/analyze_editorial_differences_chinese.py`
    7. Divergent week visualization: `data_analysis/visualize_divergent_weeks.py`
  - **FIX**: Was incorrectly mapped to `scripts/sample_divergent_articles.py` (wrong directory)

- **Line 165**: "We also estimate it with enterprise traditional newspaper data on a small date range and using Lau's approach. The correlation between China Times and Liberty Times is 0.6"
  - Source: `data_analysis/reproduce_lau_work.py` with Lau's dictionary on enterprise database `data/news/news.duckdb`
  - Enterprise data: 8,922 articles from 3 newspapers (CT: 1,493, UD: 2,262, LT: 5,167), date range 2022-01-01 to 2024-03-15
  - Data origin: `data/news/processed/` NDJSON files loaded into DuckDB
  - **Verified**: Pearson correlation on weekly overall GPR counts:
    - CT-LT: **0.5982** (rounds to 0.6)
    - CT-UD: 0.7403
    - LT-UD: 0.7126

- **Figure 3.9** (Line 259): `adiz_shares_vs_invasions_weekly.png`
  - Source: `data_analysis/create_adiz_mention_index.py`
  - Input: `data/news/stage2_enhanced_full.duckdb` + `data/external/ADIZ_invasion.csv`
  - ADIZ data: Daily count of PLA aircraft tracked in Taiwan's de facto ADIZ, 2020-09-09 to 2024-12-31
  - ADIZ data provenance: Compiled from Taiwan MoD daily ADIZ incursion tracking reports (exact compilation source not documented in codebase)
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

- **VAR Economic Variables** (Line 270, `data_preprocessing/build_master_df.py`):
  | Variable | Thesis Name | Raw Data File | Processing |
  |----------|-------------|---------------|------------|
  | `log_GPR` | GPR index | `data/gpr_indices.duckdb` table `gpr_taiwan_step2_median` | Weekly → quarterly, log level |
  | `VIX_lvl` | VIX | `data/external/vixtwn_macromicro.csv` | Daily → quarterly mean, level |
  | `Δlog_GFCF_real` | Private investment growth | `data/external/invest.csv` (column `private_invest_nominal_b`) | Nominal → real (CPI deflated), log-differenced |
  | `log_Hours_SA` | Work hours SA | `data/external/employment.csv` + `data/external/workhour.csv` | log(employment × avg_hours / working_age_pop), X-13 seasonal adjustment with Chinese New Year regressors |
  | `Δlog_TAIEX_real` | Stock market returns | `data/external/TAIEX_daily.parquet` | Daily → quarterly mean, CPI deflated, log-differenced |
  | `Support_pct` | Party support rate | `data/external/美麗島民調.csv` (column `泛綠`) | Monthly green coalition support → quarterly mean |
  | `ΔRate_pct` | Policy rate change | `data/external/interest.csv` (重貼現率/Rediscount Rate) | Forward-filled daily, quarterly level, first-differenced |

- **Cholesky Ordering** (`data_analysis/svar_analysis.py` lines 20-28):
  1. `log_GPR` (impulse variable, ordered first per Caldara methodology)
  2. `VIX_lvl`
  3. `Δlog_GFCF_real`
  4. `log_Hours_SA`
  5. `Δlog_TAIEX_real`
  6. `Support_pct`
  7. `ΔRate_pct`

- **IRF Details**: Two standard deviation GPR shock, quarterly data, 2 lags

## Chapter 4: Conclusion
- **Line 6**: "correlates 0.61 with Lau" → Cross-references Section 3.1, Line 20 (verified 0.6107)
- **Line 10**: "six-variable quarterly VAR" → Cross-references Section 3.5; note: actually 7 variables including GPR
- **Line 10**: "increase in investment, work hours, and equity returns, alongside a brief fall in implied volatility" → IRF results from Figure 3.10

## Appendix A: LLM Prompt Templates
- **Step 1 Prompt** (Lines 12-47): Role-based dictionary generation prompt
  - Source file: `prompts/generate_taiwan_gpr_dictionary_role_based.json`
  - Used by: `scripts/batch_generate_step1.py`
  - System message varies by role (5 roles documented in Section 2.2)
  - Output structure: `categories` (search queries) + `word_sets` (Chinese keyword lists)

- **Step 2 Prompt** (Lines 55-104): Fixed 8-category dictionary generation prompt
  - Source file: `prompts/generate_taiwan_gpr_dictionary_stable_8categories.json`
  - Used by: `scripts/batch_generate_step2.py`
  - Enforces exactly 8 categories with 15-25 Traditional Chinese terms per word set
  - JSON schema validation ensures structured output

## Appendix B: Category Mapping
- **Line 10-12**: "172 unique categories found across all GPR dictionaries"
  - Cross-references Section 2.2 Line 35 (same discrepancy: 172 is from all 40 dicts; 150 from 35 valid)
  - Consolidation analysis: `docs/category_mapping_analysis.md`

- **Lines 22-60**: Mapped subcategories per category
  - Source: `docs/category_mapping_analysis.md` + `prompts/generate_taiwan_gpr_dictionary_stable_8categories.json`
  - **Note**: Subcategory counts should be verified against actual `docs/category_mapping_analysis.md` content

- **Line 7-8**: Code repository link: `https://github.com/daniel-bale/gpr-zero`

## Data Pipeline Dependency Chain

```
Level 0: Articles DB
   Scripts: scripts/stage1_scan_warc.py → scripts/stage2_parse_html_load_db.py
   WARC input:      /home5/chuenho/data/crawl-data/CC-NEWS/{YYYY}/{MM}/*.warc.gz  (25,086 files, ~25TB)
   Stage 1 parquet: /home5/chuenho/data/news/stage1_parquet/{YYYY}/{MM}/*.parquet  (43,950 files, ~320GB)
   Stage 2 DB:      /home5/chuenho/data/news/stage2_enhanced_full.duckdb           (108GB, articles table)

Level 1: Binary Matrix (CACHED TABLE)
   Table: gpr_binary_matrix_all_words (6,441,601 rows, 3,341 word columns)
   Created by: scripts/compute_all_dictionary_timeseries.py
   Uses: GPRCalculatorBinaryMatrix with 80 dicts from data/dictionaries/all_dictionaries/

Level 2A: Dictionary Timeseries (all outlets)
   data/gpr_timeseries/*_W_overall.parquet (~76 files)
   Created by: scripts/compute_all_dictionary_timeseries.py
   NOTE: yeu_tong_lau_2024_W_overall.csv/.parquet MOVED to backup (all-outlet, 124,077 GPR articles → 84.1% military_buildup — incorrect for thesis)

Level 2B: Dictionary Timeseries (3 outlets only) ← CANONICAL for thesis
   data/gpr_timeseries_3outlets/*_W_overall_3outlets.parquet (~76 files)
   Created by: scripts/compute_all_dictionary_timeseries_3outlets.py
   Canonical Lau data: lau_overall_taiwan_recalculated_W_overall_3outlets.parquet (3-outlet, 60,485 GPR articles)

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

Level 5B: Category Contributions
   data_analysis/output/category_contributions/category_contributions_lau_2024.csv
   data_analysis/output/category_contributions/category_contributions_president.csv
   data_analysis/output/category_contributions/category_contributions_combined.csv
   Created by: data_analysis/calculate_category_contributions.py
   Input: Level 0 DB (stage2_enhanced_full.duckdb, 3-outlet filter) + Lau/Step2 dictionaries
   Uses: GPRCalculator (article-level matching via binary matrix cache)

Level 5C: Enterprise Correlation
   data_analysis/output/ (correlation heatmaps, inter-newspaper correlations)
   Created by: data_analysis/reproduce_lau_work.py
   Input: data/news/news.duckdb (enterprise data from data/news/processed/ NDJSON files)
   Data: 8,922 articles, 3 newspapers (CT/UD/LT), 2022-01 to 2024-03
   Uses: Lau's dictionary (yeu_tong_lau_2024.json) for GPR matching
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

## Planned/Unfinished Thesis Content

### Category Peak Event Analysis
- Scripts:
  - `scripts/analyze_category_peak_events.py` (prompt: `prompts/analyze_category_peak_event.json`)
  - `scripts/analyze_category_peak_events_updated.py`
  - `data_analysis/capture_category_peak_events.py`
- Output: `data/category_contributions/peak_events/*.json`
- Status: LLM-generated peak event descriptions exist but not yet incorporated into thesis

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

## Prompt Templates
| Prompt | Used By | Purpose |
|--------|---------|---------|
| `prompts/caldara_translation_combined.json` | `src/gpr_zero/llm/` | Caldara keyword translation |
| `prompts/generate_taiwan_gpr_dictionary_role_based.json` | `scripts/batch_generate_step1.py` | Step1 dictionary generation |
| `prompts/generate_taiwan_gpr_dictionary_stable_8categories.json` | `scripts/batch_generate_step2.py` | Step2 dictionary generation |
| `prompts/analyze_gpr_spike_event.json` | `scripts/generate_spike_summaries_llm.py` | Spike event LLM summaries (`tab:unique-spikes`) |
| `prompts/analyze_category_peak_event.json` | `scripts/analyze_category_peak_events.py` | Category peak event analysis (planned) |
| `prompts/summarize_divergent_coverage.json` | `data_analysis/process_divergent_summaries.py` | Divergent coverage LLM summaries (Table 4 / `tab:newspaper_stance`) |

## Key Databases
- Main article database: `data/news/stage2_enhanced_full.duckdb`
- Binary matrix cache: `gpr_binary_matrix_all_words` table (in same DuckDB)
- GPR index store: `data/gpr_indices.duckdb` (tables: `gpr_taiwan_step2_median`, `yeu_tong_lau_2024`, etc.)

## Verification Notes (2026-02-20)

### Corrections Applied (2026-02-16)
1. **Figure 3.4**: Changed source from `create_category_gpr_plot_fixed.py` to `create_category_gpr_plot_final.py`
2. **Table 5**: Changed path from `scripts/sample_divergent_articles.py` to `data_analysis/sample_divergent_articles.py`
3. **Line 164 correlations**: Changed source from `calculate_newspaper_specific_step2_gpr.py` to `create_clean_newspaper_comparison.py`
4. **Figure 3.10 data prep**: Changed from `scripts/prepare_step2_median_for_svar.py` to `data_preprocessing/build_master_df.py --gpr-version gpr_taiwan_step2_median`
5. Added dependency chain documentation
6. Added intermediate data creators (create_gpr_median_plots.py, compute_step2_median_3outlets.py, calculate_category_gpr_complete.py)

### Corrections Applied (2026-02-20)
7. **Table numbering**: Fixed table numbers — `tab:unique-spikes` is unnumbered longtable; Table 2 = `tab:top-words`, Table 3 = `tab:word-ratios`, Table 4 = `tab:newspaper_stance`
8. **Section 2.2**: Added model IDs, 5 roles, 8 categories enumeration
9. **Section 2.3**: Added Table 1 exact dictionary file, Lau 86% and LLM 42% category contribution claims with discrepancy notes
10. **Section 3.2**: Clarified 21 total spikes vs 11 unique events in table vs 19 in Figure 3.5
11. **Section 3.4**: Added ADIZ data provenance, flagged unmapped alternative correlation (0.6)
12. **Section 3.5**: Added full VAR variable documentation with data sources, Cholesky ordering
13. **Added Appendix A**: Prompt template references
14. **Added Appendix B**: Category mapping documentation
15. **Category count discrepancy**: Clarified 172 is from all 40 dicts, 150 is correct count from 35 valid dicts
16. **Lau 86% (Line 70)**: Resolved — copied `data_analysis/calculate_category_contributions.py` and output CSVs from backup. 3-outlet computation gives 86.19%. Moved all-outlet `yeu_tong_lau_2024_W_overall.csv/.parquet` to backup.
17. **Enterprise correlation (Line 165)**: Resolved — copied `data_analysis/reproduce_lau_work.py` from backup. CT-LT = 0.5982 on enterprise data (`data/news/news.duckdb`). Thesis value "0.6" is correct.
18. **Data pipeline chain**: Added Level 5B (category contributions) and Level 5C (enterprise correlation). Annotated Level 2A/2B with file move notes.
19. **LLM top contributor (Line 71)**: Exhaustive 3-outlet per-dict verification via `data_analysis/verify_step2_category_contributions_3outlets.py`. Confirmed no aggregate method produces 42%. Best aggregate: 36.0% (median index). Discrepancy documented.

### Known Discrepancies
1. **Ill-format reason (Line 24)**: Thesis says all 5 failed due to "forbidden OR operators". Actually: 3 gemini dicts have OR operators, 2 claude-opus-4 dicts reference non-existent word_set (`international_relations`).
2. **Category count (Line 35)**: Thesis says "35 dictionaries collectively generate 172 unique category names" but 172 is from all 40 dicts (including 5 ill-format). From 35 valid dicts only, the correct count is **150**.
3. **Newspaper correlations (Line 164)**: Thesis has CT-LT (0.475) and LT-UD (0.559) values swapped. Script produces CT-LT: 0.559, CT-UD: 0.780, LT-UD: 0.475
4. **LLM top contributor (Line 71)**: Thesis says "42%". Exhaustive 3-outlet verification via `data_analysis/verify_step2_category_contributions_3outlets.py` confirms no aggregate method yields 42%. Top contributor is Military Escalation: **36.0%** (median index share), **34.87%** (median of per-dict top contributions), **32.33%** (median per-category across 40 dicts). Max of max is 58.7% (too high). Individual dicts near 42% exist (gpt-4.1 run3 = 41.7%) but no standard aggregate matches. Thesis likely needs correction to ~36%.

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
- Correlation calculations filter to 2017-2024 (418 weeks, script yields 419 Monday-start weeks)
- Word frequency ratios compare spike periods vs entire dataset
- 5 roles verified: president, defense_minister, journalist, nsb_director, poli_professor
- 4 model IDs verified: anthropic/claude-opus-4, google/gemini-2.5-pro, openai/gpt-4.1, openai/gpt-4o-2024-11-20
- Table 1 dictionary file verified: step2_claude-opus-4_stable8cat_run1.json (generated 2025-07-27)
- 21 total Step2 spikes, 22 Lau spikes, 16 matched, 19 shown in Figure 3.5
- VAR: 7 variables (GPR + 6 economic), 2 lags, quarterly, Cholesky with GPR ordered first

### Statistics NOT Verified (needs thesis correction)
- LLM top contributor "42%" — exhaustive 3-outlet per-dict analysis confirms no aggregate method produces 42%. Closest: 36.0% (median index share). Individual dicts can reach ~42% but thesis refers to aggregate. Likely needs correction to ~36%.
- Category count "172" — thesis says 35 valid dicts produce 172, but 172 is from all 40 dicts; correct count from 35 valid dicts is **150**

### Statistics Resolved (2026-02-20 investigation)
- Lau's dominant category "86%" — RESOLVED: 86.19% from 3-outlet computation in `data_analysis/calculate_category_contributions.py` (52,133/60,485). Thesis value correct. Script and output moved from backup to main codebase.
- Alternative CT-LT correlation "0.6" — RESOLVED: CT-LT = 0.5982 (rounds to 0.6) on enterprise data (`data/news/news.duckdb`). Source: `data_analysis/reproduce_lau_work.py`. Thesis value correct.
