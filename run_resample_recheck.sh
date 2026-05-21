#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment
source .env/bin/activate

# Recheck scripts 6 and 7 with the SAME effective config used by
# _run_pipeline.sh / run_mecharule_experiments.sh for:
#   arithmetic + Qwen/Qwen2-7B-Instruct + spectral discovery + spectral anchoring plan
# while changing only the intervention from the pipeline's effective `mean`
# to `mean-donor`.

# Run from the directory that contains this script, so all relative paths are project-local.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"



EXPERIMENT_NAME="arithmetic"
MODEL_ID="Qwen/Qwen2-7B-Instruct"
MODEL_SLUG="${MODEL_ID//\//__}"

# Match the effective pipeline config.
BATCH_SIZE=256
POINTS_TO_USE_FOR_MEAN_ABLATION=8192
MAX_POINTS_PER_ABLATION=64
MIN_FLIP_RATE=0.2
TASK_MODULE="lib.tasks.${EXPERIMENT_NAME}_task"

DATA_DIR="${SCRIPT_DIR}/data/${EXPERIMENT_NAME}/${MODEL_ID}"
CACHE_DIR="${SCRIPT_DIR}/cache/${EXPERIMENT_NAME}"
FEATURES_SCORES_DIR="${DATA_DIR}/feature_report"
RULES_DIR="${DATA_DIR}/rule_extraction_results"

# This matches _run_pipeline.sh for spectral discovery + decode_only.
CIRCUIT_LABEL="rule_split-spectral_sample-decode_only"
DISCOVERY_OUT_DIR="${DATA_DIR}/neural_circuit_discovery_results/eap_ig_inputs/${CIRCUIT_LABEL}"
DISCOVERY_INPUT_AUTODISCOVERY_DIR="${DISCOVERY_OUT_DIR}/neural_circuits"

# Baseline labels as produced by _run_pipeline.sh.
BASELINE_INTERVENTION="mean"
BASELINE_BAG_LABEL="agonist_neurons-fast-spectral_anchor"
BASELINE_STATS_DIRNAME="${CIRCUIT_LABEL}-${BASELINE_BAG_LABEL}"
BASELINE_STATS_DIR="${RULES_DIR}/neuron_flip_rules/stats/${BASELINE_STATS_DIRNAME}"

# Candidate labels for the recheck run.
# Keep separate outputs, but preserve the same naming scheme.
INTERVENTION="mean-donor"
CANDIDATE_BAG_LABEL="${BASELINE_BAG_LABEL}-${INTERVENTION}"
CANDIDATE_STATS_DIRNAME="${CIRCUIT_LABEL}-${CANDIDATE_BAG_LABEL}"
CANDIDATE_RULES_DIR="${RULES_DIR}/neuron_flip_rules_${INTERVENTION}"
CANDIDATE_STATS_DIR="${CANDIDATE_RULES_DIR}/stats/${CANDIDATE_STATS_DIRNAME}"

# Put all intervention-shift results inside the main project directory.
INTERVENTION_SHIFT_DIR="${SCRIPT_DIR}/intervention_shift/${EXPERIMENT_NAME}/${MODEL_SLUG}/${BASELINE_INTERVENTION}_vs_${INTERVENTION}"
mkdir -p "${INTERVENTION_SHIFT_DIR}"

SPECTRAL_FLAGS=(
  --spectral_space hidden
  --rep_hook_name ln_final.hook_normalized
  --rep_pooling last
  --spectral_dim 32
)

run_analyze() {
  local baseline_subset="$1"
  local out_dir="$2"
  local plan_path=""

  if [[ "${baseline_subset}" == "positive" ]]; then
    plan_path="${DISCOVERY_OUT_DIR}/spectral_sampling_plan_qwen2_hidden_for_neuron_ablation_baseline_positive.json"
  else
    plan_path="${DISCOVERY_OUT_DIR}/spectral_sampling_plan_qwen2_hidden_for_neuron_ablation_baseline_negative.json"
  fi

  python3 6_analyze_bag_of_rules.py \
    --input_data_dir "${DISCOVERY_INPUT_AUTODISCOVERY_DIR}" \
    --output_data_dir "${out_dir}" \
    --task_module "${TASK_MODULE}" \
    --n_associated "${MAX_POINTS_PER_ABLATION}" \
    --n_unrelated "${MAX_POINTS_PER_ABLATION}" \
    --batch_size "${BATCH_SIZE}" \
    --search_epsilon "${MIN_FLIP_RATE}" \
    --points_to_use_for_mean_ablation "${POINTS_TO_USE_FOR_MEAN_ABLATION}" \
    --intervention "${INTERVENTION}" \
    --fast_ablation \
    --baseline_subset "${baseline_subset}" \
    --sampling_strategy plan \
    --sampling_plan_path "${plan_path}" \
    --sign_split_first \
    --decode_only
}

run_analyze "positive" "${DISCOVERY_OUT_DIR}/${CANDIDATE_BAG_LABEL}/positive_baseline"
run_analyze "negative" "${DISCOVERY_OUT_DIR}/${CANDIDATE_BAG_LABEL}/negative_baseline"

python3 7_refine_neuron_anchored_rules.py \
  --task_module "${TASK_MODULE}" \
  --ai_model "${MODEL_ID}" \
  --rules_dir "${CANDIDATE_RULES_DIR}" \
  --features_scores_dir "${FEATURES_SCORES_DIR}" \
  --circuit_agonists_path "${DISCOVERY_OUT_DIR}/${CANDIDATE_BAG_LABEL}" \
  --search_epsilon "${MIN_FLIP_RATE}" \
  --batch_size "${BATCH_SIZE}" \
  --stats_dirname "${CANDIDATE_STATS_DIRNAME}" \
  --use_spectral_sampling \
  --sampling_max_points 10000 \
  --spectral_cache_dir "${CACHE_DIR}" \
  "${SPECTRAL_FLAGS[@]}" \
  --global_n_clusters "${MAX_POINTS_PER_ABLATION}" \
  --points_to_use_for_mean_ablation "${POINTS_TO_USE_FOR_MEAN_ABLATION}" \
  --intervention "${INTERVENTION}" \
  --extract_rules \
  --only_unique_datapoints_in_shap \
  --decode_only \
  --summarize_rule_metrics

python3 summarize_intervention_shift.py \
  --baseline_dir "${BASELINE_STATS_DIR}" \
  --candidate_dir "${CANDIDATE_STATS_DIR}" \
  --output_csv "${INTERVENTION_SHIFT_DIR}/intervention_shift_summary.csv" \
  --output_dir "${INTERVENTION_SHIFT_DIR}" \
  --baseline_label "${BASELINE_INTERVENTION}" \
  --candidate_label "${INTERVENTION}" \
  --task_name "${TASK_MODULE}" \
  --model_name "${MODEL_ID}"
