#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="./data"
OUT_DIR="./paper_tables"
PRIMARY_THRESHOLD="0.85"
N_BOOT="10000"
SAMPLE_STAT="mean"
STATS_SCORE_SCOPE="both"
MIN_RULE_DATASET_COVERAGE="0.005"

usage() {
  cat <<'EOH'
Usage:
  ./run_paper_tables_generation.sh [DATA_ROOT] [OUT_DIR]

Options:
  --primary-threshold VALUE   Predeclared MCC threshold for the exact paired test (default: 0.85)
  --sample-stat VALUE         Threshold-free sampled statistic: mean, median, prop_ge_threshold, or n_ge_threshold (default: mean)
  --n-boot N                  Bootstrap iterations for sampled stats (default: 10000)
  --stats-score-scope SCOPE   Scope for statistical comparison: test, all_fit, or both (default: both; test + all_fit)
  --min-rule-dataset-coverage VALUE  Minimum dataset coverage for HQ counts (default: 0.005)
  -h, --help                  Show help

This script generates paper tables and statistically valid E1/E3 comparison files.
By default it writes both TEST and ALL-FIT score-scope outputs. Threshold sweeps are descriptive only; inference uses one paired value per task/model unit.
EOH
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
    --primary-threshold)
      PRIMARY_THRESHOLD="${2:?missing value for --primary-threshold}"
      shift 2
      ;;
    --sample-stat)
      SAMPLE_STAT="${2:?missing value for --sample-stat}"
      shift 2
      ;;
    --n-boot)
      N_BOOT="${2:?missing value for --n-boot}"
      shift 2
      ;;
    --stats-score-scope)
      STATS_SCORE_SCOPE="${2:?missing value for --stats-score-scope}"
      shift 2
      ;;
    --min-rule-dataset-coverage)
      MIN_RULE_DATASET_COVERAGE="${2:?missing value for --min-rule-dataset-coverage}"
      shift 2
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

if [ -f .env/bin/activate ]; then
  source .env/bin/activate
fi

python3 make_paper_tables.py --data_root "$DATA_ROOT" --out_dir "$OUT_DIR" --min_rule_dataset_coverage "$MIN_RULE_DATASET_COVERAGE"

python3 10_compute_threshold_sweep_stats.py \
  --csv "$OUT_DIR/table2_threshold_sweep_per_run_long.csv" \
  --data-root "$DATA_ROOT" \
  --primary_threshold "$PRIMARY_THRESHOLD" \
  --sample-stat "$SAMPLE_STAT" \
  --score-scope "$STATS_SCORE_SCOPE" \
  --min-rule-dataset-coverage "$MIN_RULE_DATASET_COVERAGE" \
  --sample-size 0 \
  --n-boot "$N_BOOT" \
  --out_dir "$OUT_DIR/stats"
