#!/usr/bin/env bash

# Optional, but recommended so failures surface:
# set -euo pipefail

# Activate virtual environment
source .env/bin/activate


EXPERIMENT_LIST=(
	# "hans_nli"
	"arithmetic"
	"bon_jailbreaking"
)

ANALYZED_LLM_LIST=(
	"Qwen/Qwen2-1.5B-Instruct"
	"Qwen/Qwen2-7B-Instruct"
	"EleutherAI/gpt-j-6B"
	# "EleutherAI/pythia-6.9b"
	### Extra
	# "mistralai/Mistral-7B-Instruct-v0.1"
	# "swiss-ai/Apertus-8B-Instruct-2509" # not supported by TransformerLens
	# "mistralai/Mistral-7B-Instruct-v0.3" # not supported by TransformerLens
)

# Fallback if experiment not listed explicitly
DEFAULT_Z_THRESH=-1   # negative -> no MAD filtering
DEFAULT_BATCH_SIZE=32
DEFAULT_CIRCUIT_LEVEL=neuron
DEFAULT_CIRCUIT_SIZE=100000

for EXPERIMENT in "${EXPERIMENT_LIST[@]}"; do
	EXP_DIR=./data/$EXPERIMENT

	# Experiment-level defaults
	BASE_Z_THRESH="$DEFAULT_Z_THRESH"
	BATCH_SIZE="$DEFAULT_BATCH_SIZE"
	CIRCUIT_LEVEL="$DEFAULT_CIRCUIT_LEVEL"
	CIRCUIT_SIZE="$DEFAULT_CIRCUIT_SIZE"
	EVAL_INTERVENTION="mean-positional"
	NEURONS_TYPE_FLAG=(--mlp_neurons_only)

	for ANALYZED_LLM in "${ANALYZED_LLM_LIST[@]}"; do
		# Reset per-model defaults so flags/settings do not leak between LLMs.
		Z_THRESH="$BASE_Z_THRESH"
		BATCH_SIZE="$DEFAULT_BATCH_SIZE"
		MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=1
		DECODE_ONLY_FLAG=(--decode_only)

		case "$EXPERIMENT" in
			arithmetic)
				BATCH_SIZE=256
				case "$ANALYZED_LLM" in
					Qwen/Qwen2-1.5B-Instruct)
						Z_THRESH=10
						;;
					Qwen/*)
						MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=5
						Z_THRESH=10
						;;
					*)
						MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=5
						Z_THRESH=5
						;;
				esac
				;;
			bon_jailbreaking)
				case "$ANALYZED_LLM" in
					Qwen/Qwen2-1.5B-Instruct)
						;;
					Qwen/*)
						MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE=3
						BATCH_SIZE=256
						;;
					*)
						;;
				esac
				;;
			hans_nli)
				DECODE_ONLY_FLAG=()
				;;
		esac

		STATS_DIR=./data/$EXPERIMENT/$ANALYZED_LLM/rule_extraction_results/neuron_flip_rules/stats

		echo "Running $EXPERIMENT with Z_THRESH=$Z_THRESH, BATCH_SIZE=$BATCH_SIZE, CIRCUIT_LEVEL=$CIRCUIT_LEVEL, CIRCUIT_SIZE=$CIRCUIT_SIZE, EVAL_INTERVENTION=$EVAL_INTERVENTION"

		# "Spectral circuit discovery" experiment
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
			--eval_intervention "$EVAL_INTERVENTION" \
			"${DECODE_ONLY_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}" \
			--max_number_of_circuits_to_analyze "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"

		# "Random anchoring plan" experiment
		bash _run_pipeline.sh \
			"$EXPERIMENT" \
			"$ANALYZED_LLM" \
			--spectral_circuit_discovery \
			--random_anchoring_plan \
			--fast_anchoring \
			--z_thresh "$Z_THRESH" \
			--batch_size "$BATCH_SIZE" \
			--circuit_level "$CIRCUIT_LEVEL" \
			--circuit_size "$CIRCUIT_SIZE" \
			--eval_intervention "$EVAL_INTERVENTION" \
			"${DECODE_ONLY_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}" \
			--max_number_of_circuits_to_analyze "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"

		# "Spectral splits" experiment
		bash _run_pipeline.sh \
			"$EXPERIMENT" \
			"$ANALYZED_LLM" \
			--spectral_splits \
			--fast_anchoring \
			--z_thresh "$Z_THRESH" \
			--batch_size "$BATCH_SIZE" \
			--circuit_level "$CIRCUIT_LEVEL" \
			--circuit_size "$CIRCUIT_SIZE" \
			--eval_intervention "$EVAL_INTERVENTION" \
			"${DECODE_ONLY_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}" \
			--max_number_of_circuits_to_analyze "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"

		# Incorrect-rules control (permute targets before rule extraction; outputs isolated via suffix)
		bash _run_pipeline.sh \
			"$EXPERIMENT" \
			"$ANALYZED_LLM" \
			--incorrect_rules \
			--spectral_circuit_discovery \
			--spectral_anchoring_plan \
			--fast_anchoring \
			--z_thresh "$Z_THRESH" \
			--batch_size "$BATCH_SIZE" \
			--circuit_level "$CIRCUIT_LEVEL" \
			--circuit_size "$CIRCUIT_SIZE" \
			--eval_intervention "$EVAL_INTERVENTION" \
			"${DECODE_ONLY_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}" \
			--max_number_of_circuits_to_analyze "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"

		# # "Random correct rule-splits" experiment
		# bash _run_pipeline.sh \
		# 	"$EXPERIMENT" \
		# 	"$ANALYZED_LLM" \
		# 	--random_circuit_discovery \
		# 	--spectral_anchoring_plan \
		# 	--fast_anchoring \
		# 	--z_thresh "$Z_THRESH" \
		# 	--batch_size "$BATCH_SIZE" \
		# 	--circuit_level "$CIRCUIT_LEVEL" \
		# 	--circuit_size "$CIRCUIT_SIZE" \
		# 	--eval_intervention "$EVAL_INTERVENTION" \
		# 	"${DECODE_ONLY_FLAG[@]}" \
		# 	"${NEURONS_TYPE_FLAG[@]}" \
		# 	--max_number_of_circuits_to_analyze "$MAX_NUMBER_OF_CIRCUITS_TO_ANALYZE"

		# "Slow anchoring" experiment
		bash _run_pipeline.sh \
			"$EXPERIMENT" \
			"$ANALYZED_LLM" \
			--spectral_circuit_discovery \
			--spectral_anchoring_plan \
			--slow_anchoring \
			--z_thresh "$Z_THRESH" \
			--batch_size "$BATCH_SIZE" \
			--circuit_level "$CIRCUIT_LEVEL" \
			--circuit_size "$CIRCUIT_SIZE" \
			--eval_intervention "$EVAL_INTERVENTION" \
			"${DECODE_ONLY_FLAG[@]}" \
			"${NEURONS_TYPE_FLAG[@]}" \
			--max_number_of_circuits_to_analyze 1
	done

	python3 8_compare_experiments.py \
		--stats_path "$EXP_DIR" \
		--out_dir ./data/aggregated_visualizations \
		--rule_quality_metric mcc --best_mode tail@0.9 \
		--thr_min 0.85 --thr_max 0.99 --thr_step 0.01

	python3 9_compare_models.py \
		--task_dir "$EXP_DIR" \
		--out_dir ./data/aggregated_visualizations/$EXPERIMENT \
		--tau 0.2 --eps 0.2
done

python3 8_compare_experiments.py \
	--stats_path ./data/ \
	--out_dir ./data/aggregated_visualizations \
	--rule_quality_metric mcc --best_mode tail@0.9 \
	--thr_min 0.85 --thr_max 0.99 --thr_step 0.01

python3 10_compute_threshold_sweep_stats.py --csv data/aggregated_visualizations/aggregated_by_task/threshold_sweep_all_tasks_all_llms.csv
