# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPR-Zero is a geopolitical risk assessment project that uses LLM-based methods to generate Chinese keyword dictionaries and calculate GPR indices from Taiwanese news sources. The project implements both dictionary-based keyword matching and zero-shot LLM classification approaches.

## Development Commands

### Python Environment
```bash
# Setup virtual environment with Python 3.12
uv venv

# Install dependencies
uv sync

# Install development tools
uv tool install ruff      # Linter/formatter
uv tool install pytest    # Testing framework
```

### Core Commands
```bash
# Dictionary generation
uv run main.py generate-dictionary --strategy direct --output my_dictionary

# GPR index calculation
uv run main.py calculate-index --dictionary data/dictionaries/my_dictionary.json --source chinatimes --normalize

# LLM classification
uv run main.py classify --source chinatimes --start-date 2020-01-01 --end-date 2020-12-31

# Index analysis
uv run main.py analyze --indices data/indices/gpr_index_chinatimes.csv data/indices/gpr_index_ltn.csv --names "China Times" "Liberty Times"
```

### Data Processing Pipeline
```bash
# Stage 1: WARC scanning and HTML extraction
python scripts/stage1_scan_warc.py --input path/to/warc/files --output-dir data/intermediate/stage1_parquet

# Stage 2: HTML parsing and database loading
python scripts/stage2_parse_html_load_db.py --input-dir data/intermediate/stage1_parquet --final-db data/news/taiwan_news_articles.duckdb
```

### Testing
```bash
# Run tests
uv run pytest

# Run specific test
uv run pytest tests/test_stage1.py
```

### Code Quality
```bash
# Format code
ruff format

# Lint code
ruff check

# Type checking (via pyright configuration in pyproject.toml)
# Uses pyright with exclusions for data/, logs/, docs/ directories
```

## Development Memories

- Always use `uv run` instead of directly running Python scripts to ensure consistent environment and dependency management
- Long running command should give command and let me run in tmux
- This is academic work, every check should be 100% correct, if you failed the checks and cannot proceed, stop and ask user
- When writing specific numbers in thesis (e.g., "3 to 15 categories"), always verify by writing a script to count/calculate the actual values
- Spikes is a date range, so matching, indexing, comparing should use start + end. start and end are both week_start, so when filtering in day level, it would be start to end + 6
- Always check for cached/materialized tables first before querying large source tables. The binary matrix table (`gpr_binary_matrix_all_words`) contains all article data including content, so use it instead of the articles table when possible
- The binary matrix table `gpr_binary_matrix_all_words` was created using 80 dictionaries from `/data/dictionaries/all_dictionaries/`, creating 3341 unique word columns. When analyzing specific dictionary subsets (e.g., Step2 only), the word_column_mapping must still be constructed using all 80 dictionaries to match the binary matrix column indices correctly

## Architecture Overview

### Core Components

1. **LLM Integration** (`src/gpr_zero/llm/`)
   - `prompt_manager.py`: Manages LLM API calls, supports OpenAI and OpenRouter providers
   - `dictionary_generator.py`: Generates GPR dictionaries using various strategies
   - `news_classifier.py`: Zero-shot classification of news articles

2. **GPR Index Calculation** (`src/gpr_zero/gpr_index/`)
   - `calculator.py`: Main GPR index calculator using ibis for data processing
   - `analyzer.py`: Analysis and comparison of GPR indices
   - `calculator_binary_matrix.py`: Binary matrix approach for calculations

3. **Data Processing Pipeline** (`scripts/`)
   - `stage1_scan_warc.py`: Extracts HTML from WARC files, outputs to Parquet
   - `stage2_parse_html_load_db.py`: Parses HTML content, loads to DuckDB
   - Domain-specific selectors for content extraction from Taiwanese news sites

4. **Utilities** (`src/gpr_zero/utils/`)
   - `data_loader.py`: Data loading utilities
   - `dictionary_utils.py`: Dictionary processing utilities
   - `font_setup.py`: Chinese font handling for visualizations

### Data Flow

1. **Raw Data**: WARC files from Common Crawl (CC-NEWS)
2. **Stage 1**: HTML extraction – Parquet files (intermediate storage)
3. **Stage 2**: Content parsing – DuckDB database (structured articles)
4. **Processing**: GPR dictionary generation + index calculation
5. **Analysis**: Time series analysis, correlation studies, visualizations

### Configuration

- **LLM Config** (`llm_config.yaml`): Model settings, API providers, task-specific parameters
- **Project Config** (`pyproject.toml`): Dependencies, Python settings, pyright configuration
- **Prompts** (`prompts/`): JSON templates for LLM interactions with structured schema support

### Key Data Structures

- **GPR Dictionary**: JSON format with `word_sets` and `categories` for structured keyword matching
- **Articles Database**: DuckDB with schema including `warc_id`, `url`, `domain`, `outlet`, `title`, `content`, `extracted_publish_date`
- **Time Series Data**: Parquet files with weekly/daily GPR index values by outlet and category

### Domain Knowledge

- Focuses on Taiwanese news outlets (China Times, Liberty Times, etc.)
- Implements Caldara & Iacoviello (2022) GPR methodology with Taiwan-specific enhancements
- Supports Chinese language processing with Traditional Chinese character handling
- Uses role-based dictionary generation (defense minister, president, journalist perspectives)

### Development Notes

- Uses `uv` for dependency management instead of pip/conda
- Ibis framework for cross-database analytics (primarily DuckDB backend)
- Polars for high-performance data processing
- FastWARC for efficient WARC file processing
- Selectolax for fast HTML parsing
- Font handling for Chinese characters in matplotlib visualizations

## Thesis Writing Style Guide

When modifying thesis content, follow these stylistic patterns:

1. **Concise and Direct**: Use short, declarative sentences that get straight to the point. No flowery language or unnecessary elaboration.

2. **Technical but Accessible**: Explain technical concepts clearly without oversimplifying. Terms like "Cholesky decomposition" or "hybrid warfare" should be used naturally.

3. **Structured Argumentation**: Present ideas in logical progression - statement, evidence, implication. Often use colons to introduce lists or explanations.

4. **Academic Tone with Occasional Informality**: Maintain academic rigor while occasionally using conversational elements (e.g., "The pipeline is essentially prompt, filter, count").

5. **Evidence-Based**: Heavy use of citations and specific data points (percentages, dates, statistics) to support claims.

6. **Clear Transitions**: Use explicit signposting ("First,", "Second,", "Thus,", "Consequently,") to guide readers through arguments.

7. **Active Voice Preference**: "We use", "We filtered", "This thesis answers" rather than passive constructions.
