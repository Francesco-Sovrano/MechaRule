#!/usr/bin/env bash
set -euo pipefail

source .env/bin/activate

DATA_ROOT="./data"
OUT_DIR="./paper_tables"
TASK_FILTER=""
MODEL_FILTER=""
RUN_TABLES=1
BASELINE_RUN_NAME="rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor"

usage() {
  cat <<'EOF'
Usage:
  ./run_paper_tables_generation.sh [DATA_ROOT] [OUT_DIR] [options]

Options:
  --task <name>              Optional task filter, e.g. arithmetic
  --model <path>             Optional model filter, e.g. Qwen/Qwen2-1.5B-Instruct
  --baseline-run-name <str>  Override baseline run directory name
  --skip-tables              Skip make_paper_tables.py
  -h, --help                 Show help

This script now supports three modes:

1) Exact model root:
   ./run_paper_tables_generation.sh ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct ./paper_tables --skip-tables

2) Task/model filters:
   ./run_paper_tables_generation.sh ./data ./paper_tables --task arithmetic --model Qwen/Qwen2-1.5B-Instruct --skip-tables

3) Broad root scan:
   ./run_paper_tables_generation.sh ./data ./paper_tables
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
    --task)
      TASK_FILTER="${2:-}"
      shift 2
      ;;
    --model)
      MODEL_FILTER="${2:-}"
      shift 2
      ;;
    --baseline-run-name)
      BASELINE_RUN_NAME="${2:-}"
      shift 2
      ;;
    --skip-tables)
      RUN_TABLES=0
      shift
      ;;
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

normalize_path() {
  local p="$1"
  p="${p#./}"
  p="${p%/}"
  printf '%s' "$p"
}

DATA_ROOT_NORM="$(normalize_path "$DATA_ROOT")"

if [ "$RUN_TABLES" -eq 1 ]; then
  python3 make_paper_tables.py --data_root "$DATA_ROOT" --out_dir "$OUT_DIR"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_SCRIPT="$SCRIPT_DIR/11_summarize_sensitivity_analysis.py"

if [ ! -f "$SUMMARY_SCRIPT" ]; then
  echo "[error] Missing summary script: $SUMMARY_SCRIPT" >&2
  exit 1
fi

summarize_one() {
  local stats_dir="$1"
  local task="$2"
  local model="$3"
  local safe_model out_dir

  safe_model="$(printf '%s' "$model" | sed 's#[/]#_#g')"
  out_dir="$OUT_DIR/sensitivity_summary/$task/$safe_model"

  echo "[info] Summarizing sensitivity analysis: $stats_dir -> $out_dir"
  python3 "$SUMMARY_SCRIPT" \
    --stats_root "$stats_dir" \
    --out_dir "$out_dir" \
    --baseline_run_name "$BASELINE_RUN_NAME"
}

FOUND_ANY=0

# Mode A: DATA_ROOT already points to an exact model root.
if [ -d "$DATA_ROOT/rule_extraction_results/neuron_flip_rules/stats" ]; then
  FOUND_ANY=1
  STATS_DIR="$DATA_ROOT/rule_extraction_results/neuron_flip_rules/stats"

  REL_MODEL_ROOT="$DATA_ROOT_NORM"
  TASK="$(printf '%s' "$REL_MODEL_ROOT" | cut -d/ -f2)"
  MODEL="$(printf '%s' "$REL_MODEL_ROOT" | cut -d/ -f3-)"

  if [ -z "$TASK" ] || [ -z "$MODEL" ]; then
    echo "[error] Could not infer task/model from model root: $DATA_ROOT" >&2
    exit 1
  fi

  summarize_one "$STATS_DIR" "$TASK" "$MODEL"

# Mode B: explicit task/model filters from a broader root.
elif [ -n "$TASK_FILTER" ] && [ -n "$MODEL_FILTER" ]; then
  TARGET_STATS_DIR="$DATA_ROOT/$TASK_FILTER/$MODEL_FILTER/rule_extraction_results/neuron_flip_rules/stats"
  if [ ! -d "$TARGET_STATS_DIR" ]; then
    echo "[error] Target stats directory not found: $TARGET_STATS_DIR" >&2
    exit 1
  fi
  FOUND_ANY=1
  summarize_one "$TARGET_STATS_DIR" "$TASK_FILTER" "$MODEL_FILTER"

# Mode C: scan all matching trees under DATA_ROOT.
else
  while IFS= read -r STATS_DIR; do
    [ -z "$STATS_DIR" ] && continue
    FOUND_ANY=1

    REL_PATH="${STATS_DIR#${DATA_ROOT%/}/}"
    TASK="$(printf '%s' "$REL_PATH" | cut -d/ -f1)"
    MODEL="$(printf '%s' "$REL_PATH" | cut -d/ -f2-)"
    MODEL="${MODEL%/rule_extraction_results/neuron_flip_rules/stats}"

    summarize_one "$STATS_DIR" "$TASK" "$MODEL"
  done < <(find "$DATA_ROOT" -type d -path '*/rule_extraction_results/neuron_flip_rules/stats' | sort)
fi

if [ "$FOUND_ANY" -eq 0 ]; then
  echo "[error] No sensitivity-analysis stats directories found under $DATA_ROOT." >&2
  exit 1
fi
