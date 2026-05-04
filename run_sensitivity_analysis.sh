#!/usr/bin/env bash

# Optional, but recommended so failures surface:
# set -euo pipefail

# Activate virtual environment
source .env/bin/activate

# Optional: run Hugging Face in offline mode
# export HF_HUB_OFFLINE=1


EXPERIMENT_LIST=(
	"arithmetic"
	"hans_nli"
	"bon_jailbreaking"
)

ANALYZED_LLM_LIST=(
	### Coding
	# "microsoft/Phi-3-mini-4k-instruct"
	"Qwen/Qwen2-1.5B-Instruct"
	### Non-coding
	"Qwen/Qwen2-7B-Instruct"
	# "EleutherAI/gpt-j-6B"
	# "EleutherAI/pythia-6.9b"
)

# Fallback if experiment not listed explicitly
DEFAULT_Z_THRESH=-1   # negative -> no MAD filtering
DEFAULT_BATCH_SIZE=32
DEFAULT_CIRCUIT_LEVEL=neuron
DEFAULT_CIRCUIT_SIZE=100000
DEFAULT_MIN_FLIP_RATE=0.2
CIRCUIT_SIZE_LIST=(25000 200000) # run 200000 with DEFAULT_CIRCUIT_LEVEL=edge!
MIN_FLIP_RATE_LIST=(0.3 0.4 0.5)

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
				# MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=5
				BATCH_SIZE=256
				case "$ANALYZED_LLM" in
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
					Qwen/*)
						BATCH_SIZE=256
						;;
				esac
				;;
			hans_nli)
				# NEURONS_TYPE_FLAG=()
				DECODE_ONLY_FLAG=()
				;;
			grammar_acceptability)
				# NEURONS_TYPE_FLAG=()
				DECODE_ONLY_FLAG=()
				;;
			cognitive_bias_sensitivity)
				# NEURONS_TYPE_FLAG=()
				DECODE_ONLY_FLAG=()
				;;
		esac

		STATS_DIR=./data/$EXPERIMENT/$ANALYZED_LLM/rule_extraction_results/neuron_flip_rules/stats

		run_setting() {
			local CIRCUIT_SIZE="$1"
			local MIN_FLIP_RATE="$2"

			echo "Running $ANALYZED_LLM on $EXPERIMENT with Z_THRESH=$Z_THRESH, BATCH_SIZE=$BATCH_SIZE, CIRCUIT_LEVEL=$CIRCUIT_LEVEL, CIRCUIT_SIZE=$CIRCUIT_SIZE, MIN_FLIP_RATE=$MIN_FLIP_RATE, EVAL_INTERVENTION=$EVAL_INTERVENTION"

			bash _run_pipeline.sh \
				"$EXPERIMENT" \
				"$ANALYZED_LLM" \
				--spectral_circuit_discovery \
				--spectral_anchoring_plan \
				--fast_anchoring \
				--z_thresh "$Z_THRESH" \
				--batch_size "$BATCH_SIZE" \
				--circuit_level "$CIRCUIT_LEVEL" \
				--circuit_size "$CIRCUIT_SIZE" \
				--min_flip_rate "$MIN_FLIP_RATE" \
				--eval_intervention "$EVAL_INTERVENTION" \
				"${DECODE_ONLY_FLAG[@]}" \
				"${NEURONS_TYPE_FLAG[@]}" \
				--max_number_of_circuits_to_analyze "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"
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

	done
done