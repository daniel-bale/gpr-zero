# GPR-Zero: Improving Geopolitical Risk Assessment

GPR-Zero generates Chinese geopolitical risk dictionaries with language models and
calculates GPR indices from Taiwanese news. See [project-goal.md](project-goal.md)
for the research motivation and [CONTEXT.md](CONTEXT.md) for domain terminology.

## Setup

Use Python 3.12 or newer with `uv`:

```bash
uv sync
uv run gpr --help
```

## One command and one project configuration

[gpr.toml](gpr.toml) owns runtime paths:

```toml
[paths]
data = "data"
# Set this only when the database lives outside the data directory:
# database = "/path/to/stage2_enhanced_full.duckdb"
llm_config = "llm_config.yaml"
```

Paths resolve beside this file. Dictionary locations, intermediate files, and
analysis outputs derive from `data`; the default database is
`data/news/stage2_enhanced_full.duckdb`. Model/provider settings belong in
`llm_config.yaml`; LLM workflows load the project's `.env` and write request logs
under the project's `logs/llm`.

`gpr` discovers the nearest `gpr.toml` from the working directory upward, falling
back to the source checkout's configuration. Select another project with
`--project /path/to/gpr.toml` or `GPR_PROJECT`, and inspect the resolved paths with
`uv run gpr config`. Database paths come exclusively from that project file.

## Commands

| Command | Result |
| --- | --- |
| `ingest warc --input /path/to/archives` | Extract article HTML into project Parquet files |
| `ingest html` | Parse intermediate HTML into the project article database |
| `dictionaries step1` | Role-based dictionary batches |
| `dictionaries step2` | Fixed eight-category dictionary batches |
| `cache rebuild` | Explicitly replace the binary matrix and its vocabulary manifest |
| `index all` | Weekly overall and outlet indices from the existing cache |
| `index three-outlets` | Weekly indices for the thesis three-outlet cohort |
| `index categories` | Category indices using the thesis category-median dictionary selection |
| `index verify-categories` | Three-outlet category contributions |
| `index newspapers` | Newspaper indices, comparisons, and spike analyses |
| `index median` | Step 2 median from the three-outlet time series |
| `spikes detect` | Weekly events and peak comparisons |
| `spikes summaries` | Unique-event summaries |
| `spikes articles` | Article samples and normalized category distributions |
| `spikes words` | Keyword lift scores |
| `spikes category-peaks` | Thesis category-peak analysis using Step 2 ordinals |
| `spikes category-peaks-updated` | Category-peak analysis using the complete vocabulary |
| `verify thesis-profile /path/to/thesis_profile` | Read-only verification of production reference artifacts |

Each command is prefixed with `uv run gpr`. Use `--help` on an ingestion or
verification command for its task-specific input options. Destinations come from
the project file. For example:

```bash
uv run gpr index three-outlets
uv run gpr index median
uv run gpr spikes detect
```

Index commands reuse cached articles. `uv run gpr cache rebuild` intentionally
replaces the cache from the source `articles` table. New caches record and validate
their vocabulary; existing unmanifested caches require the exact inventory that
built them. Full-corpus commands should run in your usual tmux session.

Dictionary generation and category-peak interpretation call the configured LLM
provider. The batch recipes preserve thesis prompts, roles, providers, and run
counts. LLM responses are not inherently deterministic.

`uv run python -m gpr_zero` uses the same dispatcher. This branch has no legacy
script trees, forwarding aliases, standalone `main.py`, or old
`generate-dictionary` command. Historical implementations and invocations remain
on the `thesis` branch.

## Production verification

The supplied production reference is:

```bash
uv run gpr verify thesis-profile \
  /Volumes/K/chuenho/gpr-zero/replication/raw_rebuild/thesis_profile
```

The verifier reads its manifest, exported results, and sibling aggregate Parquet
files. It checks artifact integrity and recomputes supported numeric results using
explicit artifact selections. It prints a JSON report and returns a nonzero status
on failed checks. It does not rewrite the reference directory or rebuild the corpus.

The profile records 40 dictionaries for overall Step 2 medians and 39 for category
medians, omitting Gemini Step 2 run 8 from the latter. It also records different
main-spike, printed-word-table, and category-figure event sets. These distinctions
are result definitions, not optional command-line switches. Its manifest reports
unresolved printed thesis claims, which remain unresolved by code cleanup.

The sibling `aggregate.duckdb` stores sparse word/match IDs and UTC timestamps;
it is not the text-containing binary cache consumed by `index` workflows. The
verification command reads the completed aggregate outputs directly. Full raw
corpus reconstruction and stochastic LLM/VAR outputs require separate execution.

## Architecture and tests

- `project.py` owns paths, database connections, and LLM setup.
- `gpr_index/corpus.py` owns dictionary inventories, cache validation, and article cohorts.
- `gpr_index/dictionary.py` owns dictionary interpretation and text/binary adapters.
- `gpr_index/spikes.py` owns event detection, overlap, peak comparison, and article windows.
- `gpr_index/series.py` owns category-median selection and aggregation.
- `workflows/` owns orchestration; `cli.py` owns command discovery.
- `verification.py` owns production profile checks and their evidence report.

```bash
uv run pytest
GPR_THESIS_PROFILE=/path/to/replication/raw_rebuild/thesis_profile uv run pytest
```

Tests use generated articles/WARC archives and recorded LLM responses. Differential
oracles load source from thesis commit `290b7c4c8eb900a712384deea0b68e2e47009624`
in Git; they do not require old scripts in this checkout. Production-only tests
skip explicitly when `GPR_THESIS_PROFILE` is unset. Reports distinguish exact
matches from floating-point representation differences rather than asserting
unqualified bit-for-bit equality.

## References

- Caldara, D., & Iacoviello, M. (2022). Measuring Geopolitical Risk.
- Yeu-Tong Lau (2024). Impact of Geopolitical Risk on Equity Prices of Taiwanese Listed Companies with Investment in China.
