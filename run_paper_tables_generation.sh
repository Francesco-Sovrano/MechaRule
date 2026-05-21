#!/usr/bin/env bash
set -euo pipefail

source .env/bin/activate

DATA_ROOT="./data"
OUT_DIR="./paper_tables"

usage() {
  cat <<'EOF'
Usage:
  ./run_paper_tables_generation.sh [DATA_ROOT] [OUT_DIR]

Options:
  -h, --help                 Show help

This script only generates paper tables. Sensitivity-analysis summaries are
run by ./run_sensitivity_analysis.sh for the task/model pairs configured there.
EOF
}

if [ "$#" -ge 1 ] && [[ "${1:-}" != --* ]]; then
  DATA_ROOT="$1"
  shift
fi
if [ "$#" -ge 1 ] && [[ "${1:-}" != --* ]]; then
  OUT_DIR="$1"
  shift
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[warn] Ignoring unknown arg: $1" >&2
      shift
      ;;
  esac
done

python3 make_paper_tables.py --data_root "$DATA_ROOT" --out_dir "$OUT_DIR"
