#!/usr/bin/env bash

set -euo pipefail

# Activate virtual environment
source .env/bin/activate

# Optional: run Hugging Face in offline mode
# export HF_HUB_OFFLINE=1


EXPERIMENT_LIST=(
	"arithmetic"
	"bon_jailbreaking"
	# "hans_nli"
)

ANALYZED_LLM_LIST=(
	"Qwen/Qwen2-1.5B-Instruct"
	# "Qwen/Qwen2-7B-Instruct"
)

# Fallback if experiment not listed explicitly
DEFAULT_Z_THRESH=-1   # negative -> no MAD filtering
DEFAULT_BATCH_SIZE=32
DEFAULT_CIRCUIT_LEVEL=neuron
DEFAULT_CIRCUIT_SIZE=100000
DEFAULT_MIN_FLIP_RATE=0.2
CIRCUIT_SIZE_LIST=(25000 200000) # run 200000/85988 with DEFAULT_CIRCUIT_LEVEL=edge!
MIN_FLIP_RATE_LIST=(0.3 0.4 0.5)

SENSITIVITY_SUMMARY_ROOT="./paper_tables/sensitivity_summary"
BASELINE_RUN_NAME="rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_SCRIPT="$SCRIPT_DIR/11_summarize_sensitivity_analysis.py"

if [ ! -f "$SUMMARY_SCRIPT" ]; then
	echo "[error] Missing summary script: $SUMMARY_SCRIPT" >&2
	exit 1
fi

summarize_sensitivity_analysis() {
	local STATS_DIR="$1"
	local EXPERIMENT="$2"
	local ANALYZED_LLM="$3"
	local SAFE_MODEL
	local SUMMARY_OUT_DIR

	if [ ! -d "$STATS_DIR" ]; then
		echo "[error] Missing sensitivity-analysis stats directory: $STATS_DIR" >&2
		exit 1
	fi

	SAFE_MODEL="$(printf '%s' "$ANALYZED_LLM" | sed 's#[/]#_#g')"
	SUMMARY_OUT_DIR="$SENSITIVITY_SUMMARY_ROOT/$EXPERIMENT/$SAFE_MODEL"

	echo "Summarizing sensitivity analysis for $ANALYZED_LLM on $EXPERIMENT: $STATS_DIR -> $SUMMARY_OUT_DIR"
	python3 "$SUMMARY_SCRIPT" \
		--stats_root "$STATS_DIR" \
		--out_dir "$SUMMARY_OUT_DIR" \
		--baseline_run_name "$BASELINE_RUN_NAME" \
		--score_scope "${SENSITIVITY_SCORE_SCOPE:-all_fit}" \
		--quality_thresholds 0.50 0.60 0.70 0.75 0.80 0.85
}

for ANALYZED_LLM in "${ANALYZED_LLM_LIST[@]}"; do
	for EXPERIMENT in "${EXPERIMENT_LIST[@]}"; do
		EXP_DIR=./data/$EXPERIMENT

		# Experiment-level defaults
		BASE_Z_THRESH="$DEFAULT_Z_THRESH"
		MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=1
		BATCH_SIZE="$DEFAULT_BATCH_SIZE"
		CIRCUIT_LEVEL="$DEFAULT_CIRCUIT_LEVEL"
		EVAL_INTERVENTION="mean-positional"
		NEURONS_TYPE_FLAG=(--mlp_neurons_only)
		DECODE_ONLY_FLAG=(--decode_only)

		Z_THRESH="$BASE_Z_THRESH"
		case "$EXPERIMENT" in
			grammar_acceptability)
				BATCH_SIZE=256
				;;
			arithmetic)
				BATCH_SIZE=256
				case "$ANALYZED_LLM" in
					Qwen/Qwen2-1.5B-Instruct)
						Z_THRESH=10
						;;
					Qwen/*)
						Z_THRESH=10
						;;
					*)
						Z_THRESH=5
						;;
				esac
				;;
			bon_jailbreaking)
				case "$ANALYZED_LLM" in
					Qwen/Qwen2-1.5B-Instruct)
						;;
					Qwen/*)
						# MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=3
						BATCH_SIZE=256
						;;
					*)
						;;
				esac
				;;
			hans_nli)
				# NEURONS_TYPE_FLAG=()
				DECODE_ONLY_FLAG=()
				;;
		esac

		STATS_DIR=./data/$EXPERIMENT/$ANALYZED_LLM/rule_extraction_results/neuron_flip_rules/stats

		run_setting() {
			local CIRCUIT_SIZE="$1"
			local MIN_FLIP_RATE="$2"

			echo "Running $ANALYZED_LLM on $EXPERIMENT with Z_THRESH=$Z_THRESH, BATCH_SIZE=$BATCH_SIZE, CIRCUIT_LEVEL=$CIRCUIT_LEVEL, CIRCUIT_SIZE=$CIRCUIT_SIZE, MIN_FLIP_RATE=$MIN_FLIP_RATE, EVAL_INTERVENTION=$EVAL_INTERVENTION"

			local CMD=(
				bash _run_pipeline.sh
				"$EXPERIMENT"
				"$ANALYZED_LLM"
				--spectral_circuit_discovery
				--spectral_anchoring_plan
				--fast_anchoring
				--z_thresh "$Z_THRESH"
				--batch_size "$BATCH_SIZE"
				--circuit_level "$CIRCUIT_LEVEL"
				--min_flip_rate "$MIN_FLIP_RATE"
				--eval_intervention "$EVAL_INTERVENTION"
			)

			if [[ "$CIRCUIT_SIZE" == "200000" ]]; then
				CMD+=(--include_zero_scores)
				CMD+=(
					--circuit_size 80000
				)
			else
				CMD+=(
					--circuit_size "$CIRCUIT_SIZE"
				)
			fi

			if ((${#DECODE_ONLY_FLAG[@]} > 0)); then
				CMD+=("${DECODE_ONLY_FLAG[@]}")
			fi

			if ((${#NEURONS_TYPE_FLAG[@]} > 0)); then
				CMD+=("${NEURONS_TYPE_FLAG[@]}")
			fi

			CMD+=(
				--max_number_of_circuits_to_analyze "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"
			)

			"${CMD[@]}"
		}

		# Default point + one-factor-at-a-time sensitivity sweeps around it.
		# Unique settings: (100k,0.2), (50k,0.2), (200k,0.2), (100k,0.1), (100k,0.3), (100k,0.4), (100k,0.5)
		run_setting "$DEFAULT_CIRCUIT_SIZE" "$DEFAULT_MIN_FLIP_RATE"

		for CIRCUIT_SIZE in "${CIRCUIT_SIZE_LIST[@]}"; do
			if [[ "$CIRCUIT_SIZE" == "$DEFAULT_CIRCUIT_SIZE" ]]; then
				continue
			fi
			run_setting "$CIRCUIT_SIZE" "$DEFAULT_MIN_FLIP_RATE"
		done

		for MIN_FLIP_RATE in "${MIN_FLIP_RATE_LIST[@]}"; do
			if [[ "$MIN_FLIP_RATE" == "$DEFAULT_MIN_FLIP_RATE" ]]; then
				continue
			fi
			run_setting "$DEFAULT_CIRCUIT_SIZE" "$MIN_FLIP_RATE"
		done

		# Run script 11 only for the task/model pair configured in this loop.
		summarize_sensitivity_analysis "$STATS_DIR" "$EXPERIMENT" "$ANALYZED_LLM"

	done
done