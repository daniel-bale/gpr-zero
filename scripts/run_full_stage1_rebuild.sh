#!/usr/bin/env bash
# Rebuild from existing extracted Stage 1 Parquet files, never from CC WARC.
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/gpr-zero-uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/private/tmp/gpr-zero-uv-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/gpr-zero-matplotlib}"

workers="${GPR_REBUILD_WORKERS:-4}"
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' 'GPR_REBUILD_WORKERS must be a positive integer.' >&2
  exit 2
fi

mkdir -p replication/raw_rebuild/logs
run_stage() {
  local stage="$1"
  shift
  printf '%s\n' "Running $stage"
  uv run --no-project --python 3.12 "$@" 2>&1 | tee "replication/raw_rebuild/logs/${stage}.log"
}

# Parsing resumes only after source metadata, code, dictionaries, and completed
# shard hashes pass the runner's guards. Later stages rebuild derived artifacts.
run_stage input_integrity_before scripts/verify_recovered_inputs.py
run_stage news scripts/rebuild_raw_news.py --full --workers "$workers"
run_stage indices scripts/aggregate_raw_rebuild.py --overwrite
run_stage reference_counts scripts/export_cached_corpus_counts.py
run_stage comparisons scripts/compare_raw_rebuild.py
run_stage word_lifts scripts/compare_raw_word_lifts.py \
  --historical-step2-log replication/raw_rebuild/word_lifts/alternative_references/logs/task9_word_lift_scores_v3_20250802_220639.log
run_stage newspaper_words scripts/compare_raw_newspaper_words.py
run_stage economics scripts/rebuild_raw_economics.py
run_stage thesis_profile scripts/export_thesis_profile.py --overwrite
run_stage input_integrity_after scripts/verify_recovered_inputs.py

printf '%s\n' 'Stage 1 rebuild and comparisons completed: replication/raw_rebuild/'
