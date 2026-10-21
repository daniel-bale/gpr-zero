#!/usr/bin/env bash
# The newer four-chapter thesis is this branch's default target.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

usage() {
  cat <<'EOF'
Usage: bash scripts/run_thesis_replication.sh [--rebuild | --cached-audit [--remote]]

Default: export the newer thesis profile from completed local Stage 1 results.
--rebuild: resume the local Stage 1 Parquet rebuild, comparisons, and export.
--cached-audit: repeat the earlier cached-input and older-draft diagnostics.
--remote: with --cached-audit, repeat bounded remote corpus checks too.
          Bare --remote retains the earlier cached-audit behavior.
EOF
}

mode=export
remote=false
case "${1:-}" in
  '') ;;
  --rebuild) mode=rebuild; shift ;;
  --cached-audit) mode=cached; shift ;;
  --remote) mode=cached; remote=true; shift ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if [[ "$mode" == cached && "${1:-}" == --remote ]]; then
  remote=true
  shift
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/gpr-zero-uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/private/tmp/gpr-zero-uv-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/gpr-zero-mpl}"

if [[ "$mode" == rebuild ]]; then
  exec bash scripts/run_full_stage1_rebuild.sh
fi

if [[ "$mode" == export ]]; then
  uv run --no-project --python 3.12 scripts/export_thesis_profile.py --overwrite
  printf 'Newer-thesis results: %s/replication/raw_rebuild/thesis_profile/manifest.json\n' "$PWD"
  exit 0
fi

if [[ "$remote" == true ]]; then
  uv run --no-project --python 3.12 scripts/audit_remote_corpus.py
fi

uv run --no-project --python 3.12 scripts/verify_thesis_draft_dictionary.py
uv run --no-project --python 3.12 scripts/replicate_thesis_indices.py
uv run --no-project --python 3.12 scripts/replicate_thesis_var.py
uv run --no-project --python 3.12 scripts/replicate_thesis_draft.py
uv run --no-project --python 3.12 tests/test_replicate_thesis_draft.py

printf 'Earlier cached-input diagnostics: %s/replication. The main thesis export uses completed Stage 1 results.\n' "$PWD"
