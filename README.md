# MechaRule: Neuron-Anchored Rule Extraction for Large Language Models

This repository contains the code for **MechaRule**, the pipeline introduced in the KDD 2026 paper **Neuron-Anchored Rule Extraction for Large Language Models via Contrastive Hierarchical Ablation**.

MechaRule connects symbolic rule extraction with mechanistic interpretability. It starts from task-level model behaviour, extracts human-readable rules over prompt or dataset features, and then searches for model components, especially MLP neurons, whose ablation selectively disrupts the behaviour covered by a rule.

The end-to-end method is **MechaRule**. The adaptive neuron-localization stage is **Contrastive Hierarchical Ablation (CHA)**. In this repository, an **agonist neuron** means a neuron whose suppression changes a rule-aligned behaviour on covered datapoints while mostly preserving unrelated datapoints.

## Paper

**Neuron-Anchored Rule Extraction for Large Language Models via Contrastive Hierarchical Ablation**  
Francesco Sovrano, Gabriele Dominici, and Marc Langheinrich  
Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2026  
DOI: `10.1145/3770855.3818091`

## What the pipeline does

MechaRule has three main stages:

1. **Behaviour measurement**: generate or load prompts, run an analysed LLM, and score task-level behaviour such as arithmetic correctness, jailbreak success, or NLI correctness.
2. **Symbolic rule extraction**: generate interpretable feature functions, score each datapoint, and extract rules that predict the target behaviour.
3. **Neuron anchoring**: use circuit attribution, grouped ablations, and neuron-level refinement to identify neurons whose suppression selectively disrupts the rule-aligned behaviour.

The pipeline is designed for auditing learned behaviours in open-weight LLMs. Some steps can use hosted APIs for feature generation or judging, but the circuit-discovery and ablation stages require local model access.

## Repository layout

### Pipeline scripts

| Script | Purpose |
| --- | --- |
| `1_generate_prompts_and_answers.py` | Generate or load task prompts, run the analysed LLM, and cache prompt-answer data. |
| `2_generate_features.py` | Use a feature LLM to propose interpretable Python feature functions, execute them safely, and write `scores.csv` plus `features.json`. |
| `3_extract_rules.py` | Extract symbolic rules from feature columns with RuleSHAP-style rule induction. |
| `4_spectral_sample_datapoints.py` | Build rule-specific or baseline-specific sampling plans with spectral coverage and optional length-matched pairing. |
| `5_discover_circuits.py` | Discover rule-associated circuits using EAP or EAP-IG-style attribution. |
| `6_analyze_bag_of_rules.py` | Run grouped ablations to find promising agonist-rich neuron groups. |
| `7_refine_neuron_anchored_rules.py` | Refine grouped candidates to single-neuron candidates and optionally re-extract neuron-anchored rules. |
| `8_compare_experiments.py` | Aggregate run-mode results within a task or data tree. |
| `9_compare_models.py` | Compare results across analysed LLMs for one task. |
| `10_compute_threshold_sweep_stats.py` | Compute threshold-sweep summary statistics from aggregated results. |
| `11_summarize_sensitivity_analysis.py` | Summarize sensitivity-analysis runs. |

### Orchestration and utility scripts

| Script | Purpose |
| --- | --- |
| `_run_pipeline.sh` | Runs the full 1-7 pipeline for one task and analysed model. |
| `run_mecharule_experiments.sh` | Runs the main task and model grid, then aggregates outputs. |
| `run_sensitivity_analysis.sh` | Runs threshold and configuration sensitivity checks. |
| `run_resample_recheck.sh` | Rechecks selected runs under an alternative intervention configuration. |
| `run_paper_tables_generation.sh` | Generates paper-oriented summary tables. |
| `make_paper_tables.py` | Builds paper summary tables from completed experiment outputs. |
| `clean_results_for_export.py` | Removes or normalizes bulky generated artifacts before exporting results. |

### Core library modules

| Module | Purpose |
| --- | --- |
| `lib/task_spec.py` | Defines the task interface used by all pipeline steps. |
| `lib/tasks/` | Built-in task specifications: arithmetic, BON jailbreaking, and HANS NLI. |
| `lib/caching_and_prompting.py` | Caching, deterministic seeding, and unified model-call wrappers for Ollama, OpenAI, and Groq. |
| `lib/feature_representation.py` | Feature dataclass plus sandboxed execution of LLM-proposed feature functions. |
| `lib/feature_extraction_runner.py` | Feature proposal, scoring, filtering, and report generation. |
| `lib/ruleshap.py` and `lib/data_model_for_shap.py` | RuleSHAP-style rule extraction and orchestration. |
| `lib/text_and_rules.py` | Rule loading, parsing, and application to feature tables. |
| `lib/spectral_analysis.py` | LLM representation extraction, PCA, spectral coverage, and sampling utilities. |
| `lib/modeling_and_ablation.py` | Hugging Face and TransformerLens model loading, generation, and ablation hooks. |
| `lib/neuron_intervention.py` | Single-neuron intervention helpers. |
| `lib/eap/` | EAP and EAP-IG attribution utilities. |

## Pipeline overview

```text
1. Generate prompts and answers
   -> cache/<task>/<model>/llm_io_data.pkl

2. Generate interpretable features
   -> data/<task>/<model>/feature_report/scores.csv
   -> data/<task>/<model>/feature_report/features.json

3. Extract symbolic rules
   -> data/<task>/<model>/rule_extraction_results/association_rules_*.csv
   -> data/<task>/<model>/rule_extraction_results/rule_combo_*.csv
   -> data/<task>/<model>/rule_extraction_results/optimal_rule_set_*.csv

4. Build sampling plans
   -> spectral_sampling_plan_*.json
   -> cached representation files under cache/<task>/<model>/

5. Discover circuits
   -> neural_circuit_discovery_results*/.../manifest.json
   -> neural_circuit_discovery_results*/.../dataset_info.json
   -> neural_circuit_discovery_results*/.../neural_circuits/

6. Analyse rule-aligned neuron groups
   -> per_rule/<target>/rule_*.json
   -> rule_knockout.json
   -> neuron_bucket_stats.json

7. Refine to neuron-anchored rules
   -> neuron_flip_rules*/stats/<run_mode>/scores.csv
   -> neuron_flip_rules*/stats/<run_mode>/flip_stats_*.csv/json/pdf
   -> optional re-extracted rules for neuron flip targets
```

## External APIs and services

The repository can use local models and hosted APIs. The required services depend on the task and configuration.

| Service | Used for | Configuration |
| --- | --- | --- |
| **Hugging Face Hub** | Loading analysed LLMs, tokenizers, sentence-transformer models, and TransformerLens-compatible weights. | Model IDs are passed through `--ai_model`; cache paths use standard Hugging Face settings. |
| **Ollama** | Default local feature-generation LLM, for example `gemma3:27b`. | Install Ollama, start the server, and pull the configured models. |
| **OpenAI API** | Optional feature-generation or judge backend. | Set `OPENAI_API_KEY` in your shell. |
| **Groq API** | Optional feature-generation or judge backend. | Set `GROQ_API_KEY` in your shell. |
| **HANS dataset download** | `lib/tasks/hans_nli_task.py` can download HANS if no local file is provided. | Use `HANS_LOCAL_FILE` for offline runs. |

Do not hardcode API keys in scripts. Export credentials in your shell instead:

```bash
export OPENAI_API_KEY="..."   # only if using OpenAI-backed calls
export GROQ_API_KEY="..."     # only if using Groq-backed calls
```

## Requirements

The provided setup script assumes:

- Python 3.12
- A working C/C++ build environment for scientific Python packages if wheels are unavailable
- OpenMP runtime support for XGBoost and numerical dependencies
- A GPU for practical circuit-discovery and ablation runs
- Ollama if you want to use the default local feature-generation configuration

On macOS, install OpenMP support if needed:

```bash
brew install libomp
```

## Installation

```bash
bash setup.sh
. .env/bin/activate
```

`setup.sh` creates `.env`, installs `requirements.txt`, and pulls the default Ollama models if the `ollama` executable is available. If Ollama is not installed, setup continues and prints a skip message.

For manual Ollama setup:

```bash
ollama serve > ollama.log 2>&1 &
ollama pull gemma3:27b
```

## Quickstart

### Run the main experiment grid

```bash
. .env/bin/activate
export GROQ_API_KEY="..."      # only if the selected task or backend needs Groq
export OPENAI_API_KEY="..."    # only if the selected task or backend needs OpenAI
bash run_mecharule_experiments.sh
```

The default grid in `run_mecharule_experiments.sh` covers:

- `arithmetic`
- `bon_jailbreaking`
- `hans_nli`

The default analysed models include Qwen2 and GPT-J variants. The full grid is compute-heavy and may require substantial GPU memory, disk cache, and runtime.

### Run one task and model

```bash
. .env/bin/activate
bash _run_pipeline.sh arithmetic Qwen/Qwen2-1.5B-Instruct \
  --spectral_circuit_discovery \
  --spectral_anchoring_plan \
  --fast_anchoring \
  --decode_only \
  --mlp_neurons_only \
  --z_thresh 10 \
  --batch_size 256 \
  --circuit_level neuron \
  --circuit_size 100000 \
  --max_number_of_circuits_to_analyze 5
```

The task name maps to `lib.tasks.<task>_task`. For example, `arithmetic` maps to `lib.tasks.arithmetic_task`.

## Manual step-by-step usage

The examples below show the main data flow for the arithmetic task with `Qwen/Qwen2-1.5B-Instruct`. Adjust paths, task modules, and model IDs for other experiments.

### 1. Generate prompts and answers

```bash
python 1_generate_prompts_and_answers.py \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --task_module lib.tasks.arithmetic_task \
  --prompts_answers_pkl_file ./cache/arithmetic/Qwen/Qwen2-1.5B-Instruct/llm_io_data.pkl \
  --batch_size 256
```

### 2. Generate interpretable features

```bash
python 2_generate_features.py \
  --ai_model gemma3:27b \
  --task_module lib.tasks.arithmetic_task \
  --prompts_answers_pkl_file ./cache/arithmetic/Qwen/Qwen2-1.5B-Instruct/llm_io_data.pkl \
  --features_scores_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report \
  --cache_dir ./cache/arithmetic/Qwen/Qwen2-1.5B-Instruct \
  --num_correct_example_prompts 32 \
  --num_incorrect_example_prompts 32 \
  --drop_near_duplicate_features \
  --near_duplicate_features_threshold 0.9999 \
  --drop_low_predictive_power_features \
  --min_delta 0.2
```

### 3. Extract symbolic rules

```bash
python 3_extract_rules.py \
  --features_scores_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report \
  --rules_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/rule_extraction_results \
  --task_module lib.tasks.arithmetic_task \
  --npermutations 5 \
  --only_unique_datapoints_in_shap \
  --use_shap_in_xgb \
  --use_shap_in_lasso
```

Use `--fake_targets` for the incorrect-rules control. `_run_pipeline.sh` isolates fake-target outputs with a separate suffix.

### 4. Build a spectral sampling plan

```bash
python 4_spectral_sample_datapoints.py \
  --features_scores_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report \
  --rules_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/rule_extraction_results \
  --rules_glob "optimal_rule_set" \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --spectral_cache_dir ./cache/arithmetic \
  --spectral_space hidden \
  --rep_hook_name ln_final.hook_normalized \
  --rep_pooling last \
  --spectral_dim 32 \
  --baseline_subset all \
  --pair_by_similarity_len_matched \
  --pair_similarity_metric euclidean \
  --min_points_per_ablation 128 \
  --max_points_per_ablation 128 \
  --use_global_clusters \
  --global_n_clusters 32 \
  --batch_size 256 \
  --task_module lib.tasks.arithmetic_task \
  --output_path ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/spectral_sampling_plan.json
```

### 5. Discover circuits

The main paper uses EAP-IG input attribution as a high-recall candidate reducer with a top-M export budget of 100,000. Smaller `--circuit_size` values are useful for debugging only.

```bash
python 5_discover_circuits.py \
  --rules_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/rule_extraction_results \
  --rules_glob "optimal_rule_set" \
  --features_scores_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report \
  --output_data_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/spectral_plan/eap_ig_inputs/neural_circuits \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --cache_dir ./cache/arithmetic/Qwen/Qwen2-1.5B-Instruct \
  --method EAP-IG-inputs \
  --intervention patching \
  --eval_intervention mean-positional \
  --circuit_level neuron \
  --circuit_size 100000 \
  --max_pairs_per_circuit 128 \
  --sampling_strategy plan \
  --sampling_plan_path ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/spectral_sampling_plan.json \
  --task_module lib.tasks.arithmetic_task
```

### 6. Run grouped ablations

The main grouped-ablation runs use decode-only MLP-write interventions, strength-based CHA pruning, and an optional root split by signed discovery score. Selectivity is recorded after localization rather than used as an inclusion filter.

```bash
python 6_analyze_bag_of_rules.py \
  --input_data_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/spectral_plan/eap_ig_inputs/neural_circuits \
  --output_data_dir ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/spectral_plan/eap_ig_inputs/agonist_neurons-fast-spectral_anchor \
  --scores_path ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/feature_report/scores.csv \
  --ai_model Qwen/Qwen2-1.5B-Instruct \
  --intervention mean-positional \
  --points_to_use_for_mean_ablation 256 \
  --sampling_strategy plan \
  --sampling_plan_path ./data/arithmetic/Qwen/Qwen2-1.5B-Instruct/neural_circuit_discovery_results/spectral_sampling_plan.json \
  --n_associated 64 \
  --n_unrelated 64 \
  --search_epsilon 0.2 \
  --fast_ablation \
  --decode_only \
  --mlp_neurons_only \
  --sign_split_first \
  --task_module lib.tasks.arithmetic_task
```

### 7. Refine neuron-anchored rules

```bash
python 7_refine_neuron_anchored_rules.py --help
```

This step reads the step-6 per-rule outputs, filters single-neuron candidates by strength/effect (`max_effect`), writes flip-target columns, and can optionally run rule extraction again on those flip targets. Selectivity diagnostics are retained for reporting.

## Built-in tasks

### Arithmetic

`lib/tasks/arithmetic_task.py` probes exact arithmetic on prompts such as `a+b=`, `a-b=`, `a*b=`, or `a/b=`. The default target is `is_correct`. Seed features include operators, operand properties, digit features, and binned numeric ranges.

### BON jailbreaking

`lib/tasks/bon_jailbreaking_task.py` evaluates whether an analysed model produces harmful behaviour for obfuscated jailbreak prompts. It uses a classifier-style judge backend through `instruct_model`, so this task may require Groq, OpenAI, Ollama, or another configured backend depending on the selected model.

### HANS NLI

`lib/tasks/hans_nli_task.py` probes natural-language inference heuristics using the HANS dataset. It can download HANS automatically or use a local file through `HANS_LOCAL_FILE`. The default target is entailment-label correctness.

## Output layout

Typical outputs are written under `data/<task>/<analysed_model>/`:

```text
data/<task>/<model>/
  feature_report/
    scores.csv
    features.json
  rule_extraction_results*/
    association_rules_*.csv
    rule_combo_*.csv
    optimal_rule_set_*.csv
    shap_plots/
    neuron_flip_rules*/
      stats/<run_mode>/
        scores.csv
        rule_combo_metrics_*.csv
        flip_stats_*.csv
        flip_stats_*.json
  neural_circuit_discovery_results*/
    <method_or_plan>/
      manifest.json
      dataset_info.json
      neural_circuits/
      agonist_neurons*/
        per_rule/<target>/rule_*.json
        rule_knockout.json
        neuron_bucket_stats.json

cache/<task>/<model>/
  llm_io_data.pkl
  spectral_reps_*.pt or .npy
  model-call caches
```

## Reproducibility and performance notes

- Most scripts expose `--seed` or `--random_seed` and call deterministic seeding utilities.
- Spectral representations can be cached with `--spectral_cache_dir`.
- Circuit discovery and ablation are compute-heavy. Start with smaller `--circuit_size`, `--max_points_per_ablation`, and `--max_number_of_circuits_to_analyze` values when debugging.
- `EAP` is faster and useful for debugging. The main paper settings use `EAP-IG-inputs` with `--circuit_size 100000`; EAP-IG and input-gradient variants are slower but can give higher-quality attribution.
- The HANS task downloads data unless `HANS_LOCAL_FILE` points to an existing local file.
- Set `HF_HUB_OFFLINE=1` only after the required Hugging Face models and datasets are cached locally.

## Troubleshooting

| Symptom | Suggested fix |
| --- | --- |
| XGBoost or OpenMP import error on macOS | Install `libomp` with Homebrew. |
| Ollama connection failure | Start Ollama with `ollama serve` and pull the configured feature model. |
| Hugging Face model download failure | Check credentials, model access, cache path, and offline mode. |
| Graph visualization errors | GraphViz and pygraphviz are optional. Circuit discovery can still run without rendered graphs. |
| Feature execution errors | LLM-proposed features are sandboxed. Rejected features are dropped by design. |
| Very slow ablation runs | Reduce `--circuit_size`, `--max_number_of_circuits_to_analyze`, `--max_points_per_ablation`, or switch to `--fast_anchoring`. |
| Large stdout from step 1 | Redirect output to a log file when generating large caches. |

## Development notes

- Keep generated outputs out of version control. Use `data/`, `cache/`, and experiment-specific output directories for generated artifacts.
- Do not commit API keys, local credentials, generated model outputs, or large cache files.
- Run syntax checks before publishing changes:

```bash
find . -name '*.py' -not -path './__MACOSX/*' -not -path '*/__pycache__/*' -print0 | xargs -0 python -m py_compile
find . -maxdepth 1 -name '*.sh' -print0 | xargs -0 -I{} bash -n '{}'
```

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{sovrano2026neuronanchored,
  author    = {Sovrano, Francesco and Dominici, Gabriele and Langheinrich, Marc},
  title     = {Neuron-Anchored Rule Extraction for Large Language Models via Contrastive Hierarchical Ablation},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  series    = {KDD '26},
  year      = {2026},
  address   = {Jeju, Republic of Korea},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3770855.3818091}
}
```
