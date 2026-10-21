# Repository guidance

GPR-Zero constructs geopolitical risk indices from Taiwanese news and LLM-generated
dictionaries. Read [CONTEXT.md](CONTEXT.md) when changing domain concepts; consult
[README.md](README.md) for commands, project configuration, and reproducibility rules.

## Development

Use `uv sync` for dependencies and `uv run` for Python commands. The installed
`gpr` CLI owns workflow selection; `uv run gpr --help` lists commands, and
`uv run gpr config` displays the resolved project paths. Long production-corpus
commands should be supplied to the user to run in tmux.

Run `uv run pytest` after changes to scientific behavior. Differential tests use
thesis commit `290b7c4c8eb900a712384deea0b68e2e47009624` as an oracle and need that
commit in local Git history. Use small generated fixtures and recorded LLM
responses. Production reference artifacts are at
`/Volumes/K/chuenho/gpr-zero/replication/raw_rebuild/thesis_profile`; run
`uv run gpr verify thesis-profile /path/to/thesis_profile` for a read-only evidence
report. The adjacent database uses sparse IDs and differs from the binary cache schema.

## Architecture

- Add or change orchestration in `src/gpr_zero/workflows/` and register commands in
  `cli.py`. Keep historical scripts and invocation aliases on the `thesis` branch.
- Resolve runtime paths and connections through `Project` in `project.py`.
  `gpr.toml` owns paths; `llm_config.yaml` owns provider/model settings.
- Use `DictionaryInventory` and `ArticleCorpus` for cache vocabulary and cohorts.
  Cache rebuilding is an explicit command; ordinary analysis uses cached articles.
- Use `DictionaryMatcher` for category expressions and text/binary adapters.
  Named operations preserve the distinct numerical definitions used in thesis artifacts.
- Use `gpr_index/spikes.py` for weekly event policies and article windows.
- Keep imports free of data reads, output-directory creation, and provider setup.

## Research constraints

- This is academic work, every check should be 100% correct, if you failed the checks and cannot proceed, stop and ask user
- When writing specific numbers in thesis (e.g., "3 to 15 categories"), always verify by writing a script to count/calculate the actual values
- Spikes is a date range, so matching, indexing, comparing should use start + end. start and end are both week_start, so when filtering in day level, it would be start to end + 6
- Always check for cached/materialized tables first before querying large source tables. The binary matrix table (`gpr_binary_matrix_all_words`) contains all article data including content, so use it instead of the articles table when possible
- The binary matrix table `gpr_binary_matrix_all_words` was created using 80 dictionaries from `/data/dictionaries/all_dictionaries/`, creating 3341 unique word columns. When analyzing specific dictionary subsets (e.g., Step2 only), the word_column_mapping must still be constructed using all 80 dictionaries to match the binary matrix column indices correctly

The recorded 80-dictionary/3341-word inventory above describes the original corpus;
verify the actual inventory before assuming those counts for another dataset.
Original thesis peak/sample workflows that used Step 2-only ordinals intentionally
retain that behavior through `DictionaryInventory.thesis_step2_ordinal`. New
analyses must use the full inventory. Preserve named thesis quirks while refactoring;
Use the supplied thesis profile to resolve artifact-specific dictionary selections:
overall Step 2 medians use 40 dictionaries, category medians omit Gemini run 8.
Record any remaining differences and unresolved claims explicitly.

## Thesis Writing Style Guide

When modifying thesis content, follow these stylistic patterns:

1. **Concise and Direct**: Use short, declarative sentences that get straight to the point. No flowery language or unnecessary elaboration.

2. **Technical but Accessible**: Explain technical concepts clearly without oversimplifying. Terms like "Cholesky decomposition" or "hybrid warfare" should be used naturally.

3. **Structured Argumentation**: Present ideas in logical progression - statement, evidence, implication. Often use colons to introduce lists or explanations.

4. **Academic Tone with Occasional Informality**: Maintain academic rigor while occasionally using conversational elements (e.g., "The pipeline is essentially prompt, filter, count").

5. **Evidence-Based**: Heavy use of citations and specific data points (percentages, dates, statistics) to support claims.

6. **Clear Transitions**: Use explicit signposting ("First,", "Second,", "Thus,", "Consequently,") to guide readers through arguments.

7. **Active Voice Preference**: "We use", "We filtered", "This thesis answers" rather than passive constructions.
