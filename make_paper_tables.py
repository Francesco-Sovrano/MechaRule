#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RUN_CONFIGS = {
    "rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor": {
        "display": "Rule split + spectral coverage",
        "split_dir": "rule_split-spectral_sample-decode_only",
        "anchor_dir": "agonist_neurons-fast-spectral_anchor",
        "fake": False,
    },
    "rule_split-spectral_sample-decode_only-agonist_neurons-fast-random_anchor": {
        "display": "Rule split + random coverage",
        "split_dir": "rule_split-spectral_sample-decode_only",
        "anchor_dir": "agonist_neurons-fast-random_anchor",
        "fake": False,
    },
    "rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor-fake_targets": {
        "display": "Fake rule split + spectral coverage",
        "split_dir": "rule_split-spectral_sample-decode_only",
        "anchor_dir": "agonist_neurons-fast-spectral_anchor",
        "fake": True,
    },
    "spectral_split-decode_only-agonist_neurons-fast-random_anchor": {
        "display": "Spectral split (no rule)",
        "split_dir": "spectral_split-decode_only",
        "anchor_dir": "agonist_neurons-fast-random_anchor",
        "fake": False,
    },
    "rule_split-spectral_sample-decode_only-agonist_neurons-spectral_anchor": {
        "display": "Rule split + bruteforce search",
        "split_dir": "rule_split-spectral_sample-decode_only",
        "anchor_dir": "agonist_neurons-spectral_anchor",
        "fake": False,
    },
}

MAIN_RUN = "rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor"
BF_RUN = "rule_split-spectral_sample-decode_only-agonist_neurons-spectral_anchor"

TABLE2_RUNS = [
    MAIN_RUN,
    "spectral_split-decode_only-agonist_neurons-fast-random_anchor",
    "rule_split-spectral_sample-decode_only-agonist_neurons-fast-random_anchor",
    "rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor-fake_targets",
]

DEFAULT_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_MIN_RULE_DATASET_COVERAGE = 0.005
DEFAULT_TABLE3_BINS = [0.2, 0.3, 0.5, 1.0]

# Main paper tables intentionally cover arithmetic and jailbreaking.
# HANS/NLI is an appendix task and can be included with the CLI flag.
DEFAULT_MAIN_TASKS = ("arithmetic", "bon_jailbreaking", "jailbreaking")

SENSITIVITY_ARTIFACT_PATTERNS = (
    re.compile(r"-M\d+"),
    re.compile(r"-tau0\.\d"),
)

# Fallback architecture sizes used only when result artifacts do not expose
# the brute-force candidate count N directly.  The budget denominator is
# 2*N - 1, where N is the matched brute-force singleton candidate count.
# The script first tries to read N from artifacts, then combines the observed
# hidden width with these layer counts, and finally falls back to these
# complete tuples.  The cap limits N for larger sampled brute-force runs.
MODEL_ARCHITECTURE_FALLBACKS = {
    "gpt-j-6B": {"n_layers": 28, "hidden_size": 4096},
    "GPT-J": {"n_layers": 28, "hidden_size": 4096},
    "Qwen2-7B-Instruct": {"n_layers": 28, "hidden_size": 3584},
    "Qwen2-7B": {"n_layers": 28, "hidden_size": 3584},
    "Qwen2-1.5B-Instruct": {"n_layers": 28, "hidden_size": 1536},
    "Qwen2-1.5B": {"n_layers": 28, "hidden_size": 1536},
}

DEFAULT_BRUTEFORCE_CANDIDATE_CAP = 100000


def _is_sensitivity_artifact_path(path: Path) -> bool:
    """Return True when any path component looks like a sensitivity artefact."""
    return any(
        pattern.search(part)
        for part in path.parts
        for pattern in SENSITIVITY_ARTIFACT_PATTERNS
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _fmt_pct_from_unit(x: float, nd: int = 1) -> str:
    if x != x:
        return ""
    return f"{100.0 * float(x):.{nd}f}%"


def _fmt_pct_from_pct(x: float, nd: int = 1) -> str:
    if x != x:
        return ""
    return f"{float(x):.{nd}f}%"


def _fmt_num(x: Any, nd: int = 1) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return ""
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return f"{xf:.{nd}f}"


def _fmt_range(lo: Any, hi: Any) -> str:
    return f"{_fmt_num(lo)}–{_fmt_num(hi)}"


def _parse_bins(s: Optional[str]) -> List[float]:
    if not s:
        return list(DEFAULT_TABLE3_BINS)
    vals = [float(v.strip()) for v in str(s).split(",") if v.strip()]
    if len(vals) < 2:
        raise ValueError("Need at least two bin edges")
    return vals


def _parse_manifest_neuron_label(label: str) -> Tuple[Optional[str], Optional[int]]:
    try:
        obj = ast.literal_eval(str(label))
        if isinstance(obj, tuple) and len(obj) == 2:
            return str(obj[0]), int(obj[1])
    except Exception:
        pass
    return None, None


def _layer_num(layer_label: str) -> int:
    digits = "".join(ch for ch in str(layer_label) if ch.isdigit())
    return int(digits) if digits else 10**9


def _pretty_task(task: str) -> str:
    t = str(task)
    mapping = {
        "arithmetic": "Arithmetic",
        "bon_jailbreaking": "Jailbreaking",
        "jailbreaking": "Jailbreaking",
        "hans_nli": "HANS NLI",
    }
    return mapping.get(t, t.replace("_", " ").title())


def _pretty_model(org: str, model: str) -> str:
    name = str(model)
    if name.startswith("Qwen2-"):
        core = name.replace("-Instruct", "")
        return core
    if name.startswith("gpt-j") or name.startswith("GPT-J"):
        return "GPT-J"
    if name.startswith("qwen") or name.startswith("Qwen"):
        return name.replace("-Instruct", "")
    return name


def _find_model_roots(data_root: Path, tasks: Optional[Sequence[str]] = None) -> List[Path]:
    data_root = data_root.resolve()
    roots: List[Path] = []
    task_filter = {t.strip() for t in tasks} if tasks else None
    for stats_root in data_root.rglob("rule_extraction_results/neuron_flip_rules/stats"):
        model_root = stats_root.parent.parent.parent
        try:
            rel = model_root.relative_to(data_root)
        except Exception:
            continue
        if _is_sensitivity_artifact_path(rel):
            continue
        if len(rel.parts) < 3:
            continue
        task = rel.parts[0]
        if task_filter and task not in task_filter:
            continue
        roots.append(model_root)
    unique = sorted({p.resolve() for p in roots})
    return unique


def _find_feature_model_roots(data_root: Path, tasks: Optional[Sequence[str]] = None) -> List[Path]:
    data_root = data_root.resolve()
    roots: List[Path] = []
    task_filter = {t.strip() for t in tasks} if tasks else None
    for feature_dir in data_root.rglob("feature_report"):
        model_root = feature_dir.parent
        try:
            rel = model_root.relative_to(data_root)
        except Exception:
            continue
        if _is_sensitivity_artifact_path(rel):
            continue
        if len(rel.parts) < 3:
            continue
        task = rel.parts[0]
        if task_filter and task not in task_filter:
            continue
        roots.append(model_root)
    return sorted({p.resolve() for p in roots})


def _find_model_or_feature_roots(data_root: Path, tasks: Optional[Sequence[str]] = None) -> List[Path]:
    return sorted({*map(Path.resolve, _find_model_roots(data_root, tasks=tasks)), *map(Path.resolve, _find_feature_model_roots(data_root, tasks=tasks))})


def _task_org_model(data_root: Path, model_root: Path) -> Tuple[str, str, str]:
    rel = model_root.resolve().relative_to(data_root.resolve())
    if len(rel.parts) < 3:
        raise ValueError(f"Unexpected model root layout: {model_root}")
    return rel.parts[0], rel.parts[1], rel.parts[2]


def _model_root_from_spec(data_root: Path, task: str, model_spec: str) -> Path:
    if "/" not in model_spec:
        raise ValueError(f"Model spec must look like ORG/MODEL, got: {model_spec}")
    org, model = model_spec.split("/", 1)
    return (data_root / task / org / model).resolve()


def _unique_preserve_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _path_aliases(name: str) -> List[str]:
    """Return compatible artifact-directory aliases, exact name first.

    Some result trees use `...-decode_only...`, while older appendix/HANS
    artifacts omit `-decode_only`.  The exact path is always preferred so the
    helper does not change results when both layouts exist.
    """
    aliases = [str(name)]
    if "-decode_only" in str(name):
        aliases.append(str(name).replace("-decode_only", ""))
    return _unique_preserve_order(aliases)


def _stats_dir(model_root: Path, run_name: str) -> Path:
    return model_root / "rule_extraction_results" / "neuron_flip_rules" / "stats" / run_name


def _stats_dir_candidates(model_root: Path, run_name: str) -> List[Path]:
    return [_stats_dir(model_root, cand) for cand in _path_aliases(run_name)]


def _resolve_stats_dir(model_root: Path, run_name: str) -> Path:
    for path in _stats_dir_candidates(model_root, run_name):
        if path.exists():
            return path
    return _stats_dir(model_root, run_name)


def _localization_dir_candidates(model_root: Path, run_name: str, baseline_subset: str = "positive_baseline") -> List[Path]:
    cfg = RUN_CONFIGS[run_name]
    base_name = "neural_circuit_discovery_results_fake_targets" if cfg.get("fake") else "neural_circuit_discovery_results"
    base = model_root / base_name / "eap_ig_inputs"
    candidates = []
    for split_dir in _path_aliases(cfg["split_dir"]):
        candidates.append(base / split_dir / cfg["anchor_dir"] / baseline_subset)
    return _unique_preserve_order(candidates)


def _localization_dir(model_root: Path, run_name: str, baseline_subset: str = "positive_baseline") -> Path:
    # Backwards-compatible canonical path; use _resolve_localization_dir for reads.
    cfg = RUN_CONFIGS[run_name]
    base_name = "neural_circuit_discovery_results_fake_targets" if cfg.get("fake") else "neural_circuit_discovery_results"
    return model_root / base_name / "eap_ig_inputs" / cfg["split_dir"] / cfg["anchor_dir"] / baseline_subset


def _resolve_localization_dir(model_root: Path, run_name: str, baseline_subset: str = "positive_baseline") -> Path:
    for path in _localization_dir_candidates(model_root, run_name, baseline_subset=baseline_subset):
        if path.exists():
            return path
    return _localization_dir(model_root, run_name, baseline_subset=baseline_subset)


def _discovery_manifest_path(model_root: Path, run_name: str) -> Path:
    cfg = RUN_CONFIGS[run_name]
    base_name = "neural_circuit_discovery_results_fake_targets" if cfg.get("fake") else "neural_circuit_discovery_results"
    base = model_root / base_name / "eap_ig_inputs"

    candidates: List[Path] = []
    for split_dir in _path_aliases(cfg["split_dir"]):
        split_base = base / split_dir
        candidates.extend([
            split_base / "neural_circuits" / "manifest.json",
            split_base / "manifest.json",
            split_base / cfg["anchor_dir"] / "neural_circuits" / "manifest.json",
            split_base / cfg["anchor_dir"] / "manifest.json",
        ])

    for path in _unique_preserve_order(candidates):
        if path.exists():
            return path

    searched = "\n".join(str(p) for p in _unique_preserve_order(candidates))
    raise FileNotFoundError(
        f"Manifest not found for run '{run_name}'. Looked in:\n{searched}"
    )


def _feature_dataset_stats_path(model_root: Path) -> Optional[Path]:
    path = model_root / "feature_report" / "dataset_stats.json"
    return path if path.exists() else None


def _normalise_unit_rate(value: Any) -> float:
    val = _safe_float(value)
    if val != val:
        return val
    # Most files store fractions in [0, 1].  Accept 0-100 percentages defensively.
    if 1.0 < val <= 100.0:
        return val / 100.0
    return val


def _metric_from_dataset_stats(model_root: Path, task: str) -> Optional[Tuple[str, float]]:
    path = _feature_dataset_stats_path(model_root)
    if path is None:
        return None
    stats = _read_json(path)
    task_l = str(task).lower()

    if "jailbreak" in task_l:
        candidates = [
            ("Jailbreak rate", "jailbreak_rate"),
            ("Jailbreak rate", "pct_is_jailbroken"),
            ("Jailbreak rate", "is_jailbroken_rate"),
        ]
    else:
        candidates = [
            ("Accuracy", "accuracy"),
            ("Accuracy", "pct_is_correct"),
            ("Accuracy", "is_correct_rate"),
        ]

    # Also handle files where only one obvious task metric is present.
    candidates.extend([
        ("Jailbreak rate", "jailbreak_rate"),
        ("Jailbreak rate", "pct_is_jailbroken"),
        ("Accuracy", "accuracy"),
        ("Accuracy", "pct_is_correct"),
    ])

    for metric, key in candidates:
        if key in stats:
            value = _normalise_unit_rate(stats[key])
            if value == value:
                return metric, value
    return None


def _feature_scores_path(model_root: Path) -> Optional[Path]:
    feature_dir = model_root / "feature_report"
    if not feature_dir.exists():
        return None
    for name in ("scores.csv", "scores.parquet"):
        cand = feature_dir / name
        if cand.exists():
            return cand
    hits = sorted(feature_dir.glob("scores.*"))
    return hits[0] if hits else None


def _load_scores_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _markdown_fallback(df: pd.DataFrame, index: bool = False) -> str:
    if index:
        render_df = df.copy()
        render_df.insert(0, "index", render_df.index)
    else:
        render_df = df
    headers = [str(c) for c in render_df.columns]
    rows = [["" if pd.isna(v) else str(v) for v in row] for row in render_df.to_numpy().tolist()]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(values):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    header = fmt_row(headers)
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = [fmt_row(row) for row in rows]
    return "\n".join([header, sep] + body)


def _write_table_artifacts(df: pd.DataFrame, out_dir: Path, stem: str, caption: Optional[str] = None, index: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    tex_path = out_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=index)
    try:
        md_text = df.to_markdown(index=index)
    except ImportError:
        md_text = _markdown_fallback(df, index=index)
    md_path.write_text(md_text + "\n", encoding="utf-8")
    tex = df.to_latex(index=index, escape=True)
    if caption:
        tex = f"% {caption}\n" + tex
    tex_path.write_text(tex + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path), "latex": str(tex_path)}


def _write_empty_table_artifacts(out_dir: Path, stem: str, columns: Sequence[str], caption: Optional[str] = None) -> pd.DataFrame:
    """Overwrite outputs with an empty table so stale artifacts are not reused."""
    df = pd.DataFrame(columns=list(columns))
    _write_table_artifacts(df, out_dir, stem, caption=caption)
    return df


def _load_agonist_map(loc_dir: Path, tau: float) -> Dict[str, float]:
    buckets_path = loc_dir / "neuron_buckets.json"
    if not buckets_path.exists():
        return {}
    buckets = _read_json(buckets_path)
    items = (buckets.get("non_catastrophic_agonists") or {}) if isinstance(buckets, dict) else {}
    out: Dict[str, float] = {}
    for key, info in items.items():
        if not isinstance(info, dict):
            continue
        rec = info.get("last_record") or {}
        effect = rec.get("max_effect", info.get("best_abs_gap", float("nan")))
        eff = abs(_safe_float(effect))
        if eff >= float(tau):
            out[str(key)] = eff
    return out


def _baseline_subsets_for_table13(scope: str, fallback_baseline_subset: str) -> List[str]:
    if scope == "both":
        return ["positive_baseline", "negative_baseline"]
    if scope == "positive":
        return ["positive_baseline"]
    if scope == "negative":
        return ["negative_baseline"]
    if scope == "selected":
        return [fallback_baseline_subset]
    raise ValueError(f"Unknown table13 baseline scope: {scope}")


def _load_agonist_union_map_for_baselines(
    model_root: Path,
    run_name: str,
    tau: float,
    baseline_subsets: Sequence[str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Load a neuron-identity union over the requested baseline regimes.

    Table 13 is a cross-task overlap table, not a same-circuit BF-vs-CHA
    comparison.  It should therefore include all localized singleton agonists
    from the requested baseline regimes and all circuits represented in each
    bucket file.  If the same neuron appears in more than one baseline file, we
    keep the largest effect for bin/threshold stability but count the neuron
    once in the set.
    """
    out: Dict[str, float] = {}
    present: List[str] = []
    missing: List[str] = []
    for baseline_subset in baseline_subsets:
        loc_dir = _resolve_localization_dir(model_root, run_name, baseline_subset=baseline_subset)
        buckets_path = loc_dir / "neuron_buckets.json"
        if not buckets_path.exists():
            missing.append(str(buckets_path))
            continue
        present.append(str(buckets_path))
        current = _load_agonist_map(loc_dir, tau=tau)
        for neuron, effect in current.items():
            if effect > out.get(neuron, -float("inf")):
                out[neuron] = effect
    return out, present, missing


def _iter_numeric_paths(obj: Any, prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], float]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_numeric_paths(v, prefix + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_numeric_paths(v, prefix + (str(i),))
    else:
        try:
            if obj is not None and not isinstance(obj, bool):
                val = float(obj)
                if math.isfinite(val):
                    yield prefix, val
        except Exception:
            return


def _extract_flip_share_from_global(stats: Dict[str, Any], direction: str) -> float:
    if not isinstance(stats, dict):
        return float("nan")

    direct_candidates = []
    if direction == "c2i":
        direct_candidates = [
            ("flip_c2i", "union_share_of_all"),
            ("c2i", "union_share_of_all"),
            ("correct_to_incorrect", "union_share_of_all"),
            ("1_to_0", "union_share_of_all"),
        ]
    else:
        direct_candidates = [
            ("flip_i2c", "union_share_of_all"),
            ("i2c", "union_share_of_all"),
            ("incorrect_to_correct", "union_share_of_all"),
            ("0_to_1", "union_share_of_all"),
        ]
    for a, b in direct_candidates:
        try:
            return _safe_float(stats[a][b])
        except Exception:
            pass

    wanted = {"c2i": ["c2i", "correct_to_incorrect", "1_to_0"], "i2c": ["i2c", "incorrect_to_correct", "0_to_1"]}[direction]
    best_score = -10**9
    best_val = float("nan")
    for path, val in _iter_numeric_paths(stats):
        joined = ".".join(path).lower()
        score = 0
        if any(tok in joined for tok in wanted):
            score += 100
        if "union_share_of_all" in joined:
            score += 60
        elif "share_of_all" in joined or "of_all" in joined:
            score += 40
        elif "union_share" in joined or "union" in joined:
            score += 25
        if "all" in joined:
            score += 5
        if "rate" in joined or "pct" in joined or "share" in joined:
            score += 2
        if score > best_score:
            best_score = score
            best_val = val
    return best_val


def _directional_union_rate_from_hq_coverage(
    hq: Dict[str, Any],
    direction: str,
    coverage_source: str,
) -> float:
    """Return Table 1's direction-specific union flip rate.

    The numerator is a unique union over prompts, so a prompt flipped by several
    neurons is counted once.  The denominator must be direction-specific:
    1->0 is normalized by baseline-positive/correct examples, and 0->1 is
    normalized by baseline-negative/incorrect examples.  This differs from the
    legacy `flip_stats_global.json` rates, which are normalized by all evaluated
    rows and therefore understate directional coverage when the baseline labels
    are imbalanced.

    With ``coverage_source="all"``, the numerator uses all localized
    rule-bearing neurons.  With ``coverage_source="hq"``, it uses only neurons
    that pass the high-quality rule threshold.
    """
    section_name = "flip_c2i" if direction == "c2i" else "flip_i2c"
    section = hq.get(section_name) if isinstance(hq, dict) else None
    if not isinstance(section, dict):
        return float("nan")

    rate_key = "eligible_union_rate_high" if coverage_source == "hq" else "eligible_union_rate_all"
    rate = _safe_float(section.get(rate_key))
    if rate == rate:
        return rate

    numerator_key = "union_high" if coverage_source == "hq" else "union_all"
    numerator = _safe_float(section.get(numerator_key))
    denominator = _safe_float(section.get("eligible_baseline_count"))
    if numerator == numerator and denominator == denominator and denominator > 0:
        return numerator / denominator

    return float("nan")



def _apply_min_dataset_coverage(df: pd.DataFrame, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE) -> pd.DataFrame:
    """Keep rule rows with sufficient held-out dataset_coverage.

    The support floor prevents ultra-sparse rules from entering HQ counts or
    strict threshold sweeps merely because their MCC is perfect on tiny support.
    Set min_dataset_coverage <= 0 to disable.  Older artifacts without coverage
    metadata are left untouched.
    """
    if df is None or df.empty:
        return df
    try:
        floor = float(min_dataset_coverage)
    except Exception:
        floor = DEFAULT_MIN_RULE_DATASET_COVERAGE
    if floor <= 0:
        return df.copy()
    cov_col = None
    for c in ("dataset_coverage", "coverage"):
        if c in df.columns:
            cov_col = c
            break
    if cov_col is None:
        return df.copy()
    out = df.copy()
    cov = pd.to_numeric(out[cov_col], errors="coerce")
    return out[cov >= floor].copy()

def _normalize_score_scope(scope: str) -> str:
    s = str(scope or "test").strip().lower().replace("_", "+").replace("-", "+").replace(" ", "")
    if s in ("eval", "heldout", "held+out", "test"):
        return "test"
    if s in ("all+fit", "allfit", "final+all+fit", "finalfit", "all"):
        return "all_fit"
    if s in ("train+test", "train+eval", "all+data", "traintest"):
        return "all_fit"
    if s.startswith("train"):
        return "train"
    return s.replace("+", "_")


def _score_scope_slug(scope: str) -> str:
    return _normalize_score_scope(scope).replace("+", "_").replace("-", "_")


def _score_scope_label(scope: str) -> str:
    scope = _normalize_score_scope(scope)
    if scope == "test":
        return "test"
    if scope == "all_fit":
        return "all-fit"
    return scope


def _filter_rule_rows_by_score_scope(df: pd.DataFrame, score_scope: str = "test") -> pd.DataFrame:
    """Keep rows for a requested scoring scope.

    TEST is the primary held-out metric. ALL-FIT is a descriptive score for a separate final-fit combo selected and scored on all rows. TRAIN rows are never included unless explicitly requested.
    """
    if df is None or df.empty:
        return df
    want = _normalize_score_scope(score_scope)
    out = df.copy()
    path = out.get("rule_combo_path", pd.Series([""] * len(out), index=out.index)).astype(str)
    fname = out.get("source_filename", pd.Series([""] * len(out), index=out.index)).astype(str)
    file_s = path + "/" + fname
    is_file_train_test = file_s.str.contains("rule_combo_train_test_", regex=False)
    if bool(is_file_train_test.any()):
        # Raw rule_combo_train_test_* files are obsolete.  TRAIN+TEST is derived
        # from raw TRAIN and TEST component rows, not read as a raw scope.
        out = out.loc[~is_file_train_test].copy()
        path = out.get("rule_combo_path", pd.Series([""] * len(out), index=out.index)).astype(str)
        fname = out.get("source_filename", pd.Series([""] * len(out), index=out.index)).astype(str)
        file_s = path + "/" + fname
        is_file_train_test = file_s.str.contains("rule_combo_train_test_", regex=False)
    is_file_all_fit = file_s.str.contains("rule_combo_all_fit_", regex=False)
    is_file_train = file_s.str.contains("rule_combo_train_", regex=False) & (~is_file_train_test) & (~is_file_all_fit)
    if "computed_on" in out.columns or "score_scope" in out.columns:
        if "computed_on" in out.columns:
            vals = out["computed_on"].map(_normalize_score_scope)
        else:
            vals = out["score_scope"].map(_normalize_score_scope)
        # Filename scope is authoritative for cached rule-combo artifacts.
        vals = vals.mask(is_file_train_test, "train+test").mask(is_file_all_fit, "all_fit").mask(is_file_train, "train")
        plain_test = file_s.str.contains("rule_combo_", regex=False) & (~is_file_train) & (~is_file_train_test) & (~is_file_all_fit)
        vals = vals.mask(plain_test & vals.isin(["train+test", "train", "all_fit"]), "test")
        out = out[vals == want].copy()
    else:
        # Legacy fallback based on filename/flip-target prefixes.
        ft = out.get("flip_target", pd.Series([""] * len(out), index=out.index)).astype(str)
        is_train = is_file_train | ft.str.startswith("train_")
        is_all_fit = is_file_all_fit | ft.str.startswith("all_fit_")
        is_train_test = is_file_train_test | ft.str.startswith("train_test_")
        if want == "test":
            out = out[(~is_train) & (~is_train_test)].copy()
        elif want == "all_fit":
            out = out[is_all_fit].copy()
        elif want == "train+test":
            out = out[is_train_test].copy()
        else:
            out = out[is_train & (~is_train_test)].copy()
    return out



def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return float(default)
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _row_get_num(row, names, default=np.nan):
    for name in names:
        if name in row:
            v = _safe_float(row.get(name), default=np.nan)
            if np.isfinite(v):
                return v
    return float(default)


def _metrics_from_confusion_counts(tp, fp, tn, fn, eps=1e-12):
    tp, fp, tn, fn = [max(0.0, _safe_float(x, 0.0)) for x in (tp, fp, tn, fn)]
    n = tp + fp + tn + fn
    P = tp + fn
    N = tn + fp
    pred = tp + fp
    prec = tp / (pred + eps) if pred > 0 else np.nan
    rec = tp / (P + eps) if P > 0 else np.nan
    f1 = (2.0 * prec * rec / (prec + rec + eps)) if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0) else np.nan
    acc = (tp + tn) / n if n > 0 else np.nan
    tnr = tn / (N + eps) if N > 0 else np.nan
    bal = 0.5 * (rec + tnr) if np.isfinite(rec) and np.isfinite(tnr) else np.nan
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / (np.sqrt(den) + eps) if den > 0 else 0.0
    base = P / n if n > 0 else np.nan
    return {
        "dataset_coverage": pred / n if n > 0 else np.nan,
        "P(target=1|fire)": prec,
        "Precision": prec,
        "R(fire|target=1)": rec,
        "F1": f1,
        "F1(target=1|fire)": f1,
        "Lift(target=1|fire)": prec / (base + eps) if np.isfinite(prec) and np.isfinite(base) else np.nan,
        "Acc": acc,
        "TPR": rec,
        "TNR": tnr,
        "BalancedAcc": bal,
        "MCC": mcc,
        "n_eval": n,
        "n_pos": P,
        "n_neg": N,
        "n_fire": pred,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "target_prevalence": base,
        "rule_fire_rate": pred / n if n > 0 else np.nan,
    }


def _row_rule_combo_path(row):
    if row is None:
        return None
    for key in ("rule_combo_path", "source_file"):
        val = row.get(key, "") if hasattr(row, "get") else ""
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        path_s = str(val).split("::", 1)[0].strip()
        if not path_s:
            continue
        path = Path(path_s)
        if path.exists():
            return path
    return None


def _semantic_remaining_for_metric_row(row, scope):
    path = _row_rule_combo_path(row)
    if path is None:
        return np.nan
    sem_path = path.parent / "semantic_filtering.json"
    if not sem_path.exists():
        return np.nan
    try:
        payload = _read_json(sem_path)
    except Exception:
        return np.nan
    records = payload.get("records", []) if isinstance(payload, dict) else []
    want = {"test": {"eval", "test"}, "train": {"train"}, "all_fit": {"train", "eval", "test"}}.get(_normalize_score_scope(scope), {str(scope)})
    vals = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("split", "")).lower() in want and rec.get("remaining", None) is not None:
            try:
                vals.append(int(rec.get("remaining")))
            except Exception:
                pass
    if not vals:
        return np.nan
    return int(sum(vals)) if _normalize_score_scope(scope) == "all_fit" else int(vals[0])


def _scores_n_eval_for_metric_row(scores_df, row, scope):
    m_sem = _semantic_remaining_for_metric_row(row, scope)
    if np.isfinite(m_sem) and m_sem > 0:
        return int(m_sem)
    if scores_df is None or scores_df.empty:
        return np.nan
    mask = pd.Series(True, index=scores_df.index)
    if "is_test" in scores_df.columns:
        is_test = scores_df["is_test"].fillna(False).astype(bool)
        scope_norm = _normalize_score_scope(scope)
        if scope_norm == "test":
            mask &= is_test
        elif scope_norm == "train":
            mask &= ~is_test
    if "_sampled" in scores_df.columns:
        try:
            mask &= scores_df["_sampled"].fillna(False).astype(bool)
        except Exception:
            pass
    target = str(row.get("flip_target", "")) if hasattr(row, "get") else ""
    if target in scores_df.columns:
        valid = scores_df[target].notna()
        if bool((~valid).any()):
            mask &= valid
    return int(mask.sum())


def _recover_confusion_from_metric_row(row, n_eval_fallback=np.nan, eps=1e-12):
    """Recover TP/FP/TN/FN from a metric row.

    Prefer explicit counts. For legacy summaries, recover from precision +
    dataset_coverage/rule_fire_rate + recall/TNR/Acc before falling back to the
    less stable TPR/TNR/F1 class-ratio formula.
    """
    def _finish(tp, fp, tn, fn, source):
        vals = [float(x) for x in (tp, fp, tn, fn)]
        if not all(np.isfinite(v) for v in vals):
            return None, "bad_counts"
        vals = [0.0 if abs(v) < 1e-7 else v for v in vals]
        if min(vals) < -1e-5:
            return None, "negative_counts"
        vals = [max(0.0, v) for v in vals]
        vals = [round(v) if abs(v - round(v)) < 1e-4 else v for v in vals]
        return tuple(vals), source

    tp = _row_get_num(row, ["tp", "TP"], np.nan)
    fp = _row_get_num(row, ["fp", "FP"], np.nan)
    tn = _row_get_num(row, ["tn", "TN"], np.nan)
    fn = _row_get_num(row, ["fn", "FN"], np.nan)
    if all(np.isfinite(x) for x in (tp, fp, tn, fn)):
        return _finish(tp, fp, tn, fn, "explicit_counts")

    M = _row_get_num(row, ["n_eval", "M", "n"], n_eval_fallback)
    if not np.isfinite(M) or M <= 0:
        return None, "missing_n_eval"

    prec = _row_get_num(row, ["P(target=1|fire)", "Precision", "precision", "P"], np.nan)
    rec = _row_get_num(row, ["R(fire|target=1)", "Recall", "recall", "TPR", "tpr"], np.nan)
    a = _row_get_num(row, ["TPR", "tpr"], rec)
    b = _row_get_num(row, ["TNR", "tnr"], np.nan)
    acc = _row_get_num(row, ["Acc", "acc", "accuracy"], np.nan)
    cov = _row_get_num(row, ["dataset_coverage", "rule_fire_rate", "coverage"], np.nan)
    n_fire = _row_get_num(row, ["n_fire", "pred_pos", "predicted_positive"], np.nan)
    if not np.isfinite(n_fire) and np.isfinite(cov):
        n_fire = cov * M

    if np.isfinite(n_fire) and n_fire >= 0 and np.isfinite(prec):
        TP = prec * n_fire
        FP = (1.0 - prec) * n_fire
        FN = np.nan
        TN = np.nan
        if np.isfinite(rec) and rec > eps:
            P_total = TP / rec
            FN = P_total - TP
        if not np.isfinite(TN):
            if np.isfinite(FN):
                TN = M - TP - FP - FN
            elif np.isfinite(b) and FP > eps and (1.0 - b) > eps:
                N_total = FP / (1.0 - b)
                TN = b * N_total
            elif np.isfinite(acc):
                TN = acc * M - TP
        if not np.isfinite(FN):
            FN = M - TP - FP - TN
        res, src = _finish(TP, FP, TN, FN, "recovered_from_precision_coverage_recall")
        if res is not None:
            return res, src

    f = _row_get_num(row, ["F1", "F1(target=1|fire)", "f1"], np.nan)
    if np.isfinite(a) and np.isfinite(b) and np.isfinite(f):
        denom = 2.0 * a - f * (1.0 + a)
        if abs(denom) > eps:
            p_ratio = f * (1.0 - b) / denom
            if np.isfinite(p_ratio) and p_ratio >= 0:
                P_total = M * p_ratio / (1.0 + p_ratio)
                N_total = M / (1.0 + p_ratio)
                res, src = _finish(a * P_total, (1.0 - b) * N_total, b * N_total, (1.0 - a) * P_total, "recovered_from_tpr_tnr_f1")
                if res is not None:
                    return res, src

    if np.isfinite(a) and np.isfinite(b) and np.isfinite(acc) and abs(acc - a) > eps:
        p_ratio = (b - acc) / (acc - a)
        if np.isfinite(p_ratio) and p_ratio >= 0:
            P_total = M * p_ratio / (1.0 + p_ratio)
            N_total = M / (1.0 + p_ratio)
            res, src = _finish(a * P_total, (1.0 - b) * N_total, b * N_total, (1.0 - a) * P_total, "recovered_from_tpr_tnr_acc")
            if res is not None:
                return res, src
    return None, "unrecoverable"


def _synthesize_train_test_scope_from_rows(df: pd.DataFrame, scores_df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None or df.empty or "computed_on" not in df.columns:
        return df
    key_cols = [c for c in ["rule_target_dir", "neuron_key", "flip_target"] if c in df.columns]
    if not key_cols:
        key_cols = [c for c in ["layer_key", "neuron_id", "flip_target"] if c in df.columns]
    if not key_cols:
        return df
    tmp = df.copy()
    tmp["_scope_norm"] = tmp["computed_on"].map(_normalize_score_scope)
    path = tmp.get("rule_combo_path", pd.Series([""] * len(tmp), index=tmp.index)).astype(str)
    fname = tmp.get("source_filename", pd.Series([""] * len(tmp), index=tmp.index)).astype(str)
    file_s = path + "/" + fname
    is_file_train_test = file_s.str.contains("rule_combo_train_test_", regex=False)
    if bool(is_file_train_test.any()):
        # Do not preserve obsolete raw TRAIN+TEST files in table generation.
        # Rebuild TRAIN+TEST strictly from TRAIN and TEST rows below.
        tmp = tmp.loc[~is_file_train_test].copy()
        path = tmp.get("rule_combo_path", pd.Series([""] * len(tmp), index=tmp.index)).astype(str)
        fname = tmp.get("source_filename", pd.Series([""] * len(tmp), index=tmp.index)).astype(str)
        file_s = path + "/" + fname
        is_file_train_test = file_s.str.contains("rule_combo_train_test_", regex=False)
    is_file_train = file_s.str.contains("rule_combo_train_", regex=False) & (~is_file_train_test)
    is_plain_combo = file_s.str.contains("rule_combo_", regex=False) & (~is_file_train) & (~is_file_train_test)
    tmp.loc[is_file_train_test, "_scope_norm"] = "train+test"
    tmp.loc[is_file_train, "_scope_norm"] = "train"
    tmp.loc[is_plain_combo & tmp["_scope_norm"].isin(["train", "train+test"]), "_scope_norm"] = "test"
    # Recompute train+test rows from train and test whenever possible, because
    # older generated train+test rows may have used a bad M or stale fallback.
    out_rows = [r.drop(labels=["_scope_norm"], errors="ignore").to_dict() for _, r in tmp[tmp["_scope_norm"] != "train+test"].iterrows()]
    for _, grp in tmp.groupby(key_cols, dropna=False):
        train_rows = grp[grp["_scope_norm"] == "train"]
        test_rows = grp[grp["_scope_norm"] == "test"]
        if train_rows.empty or test_rows.empty:
            # No safe reconstruction path.  Do not preserve cached train+test rows:
            # earlier versions could write copied or wrongly recovered all-scope rows.
            continue
        train_row = train_rows.iloc[0].drop(labels=["_scope_norm"], errors="ignore").to_dict()
        test_row = test_rows.iloc[0].drop(labels=["_scope_norm"], errors="ignore").to_dict()
        train_counts, train_source = _recover_confusion_from_metric_row(train_row, _scores_n_eval_for_metric_row(scores_df, train_row, "train"))
        test_counts, test_source = _recover_confusion_from_metric_row(test_row, _scores_n_eval_for_metric_row(scores_df, test_row, "test"))
        if train_counts is None or test_counts is None:
            continue
        tp = train_counts[0] + test_counts[0]
        fp = train_counts[1] + test_counts[1]
        tn = train_counts[2] + test_counts[2]
        fn = train_counts[3] + test_counts[3]
        merged = dict(test_row)
        merged.update(_metrics_from_confusion_counts(tp, fp, tn, fn))
        merged["computed_on"] = "train+test"
        merged["score_scope"] = "train+test"
        merged["chosen_from"] = "merged_cached_train_and_test_confusions"
        merged["confusion_merge_source"] = f"train:{train_source};test:{test_source}"
        out_rows.append(merged)
    return pd.DataFrame(out_rows)

def _scope_specific_metric_paths(stats_dir: Path, score_scope: str) -> Tuple[Path, Path]:
    slug = _score_scope_slug(score_scope)
    return (
        stats_dir / f"rule_combo_metrics_{slug}_all.csv",
        stats_dir / f"rule_combo_metrics_{slug}_best_per_neuron.csv",
    )


def _load_best_rule_metrics(stats_dir: Path, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE, score_scope: str = "test", scores_df: pd.DataFrame = None) -> pd.DataFrame:
    """Load one rule-combo row per localized neuron for the requested scope.

    ``score_scope='test'`` is the held-out score and remains the primary paper
    metric. ``score_scope='all_fit'`` is descriptive: a separate final-fit combo selected and scored on all available rows.
    """
    score_scope = _normalize_score_scope(score_scope)
    scoped_all_path, scoped_best_path = _scope_specific_metric_paths(stats_dir, score_scope)
    all_scopes_path = stats_dir / "rule_combo_metrics_all_scopes.csv"
    legacy_all_path = stats_dir / "rule_combo_metrics_all.csv"
    legacy_best_path = stats_dir / "rule_combo_metrics_best_per_neuron.csv"

    if score_scope == "train+test":
        # Reconstruct train+test from TRAIN and TEST rows only.  Never trust
        # cached train+test summaries as primary input: previous versions could
        # contain copied TEST rows or rows recovered with a wrong M.
        frames = []
        if all_scopes_path.exists():
            frames.append(pd.read_csv(all_scopes_path, low_memory=False))
        else:
            test_all, test_best = _scope_specific_metric_paths(stats_dir, "test")
            train_all, train_best = _scope_specific_metric_paths(stats_dir, "train")
            for path in (test_all, train_all, test_best, train_best):
                if path.exists():
                    frames.append(pd.read_csv(path, low_memory=False))
        if not frames:
            return pd.DataFrame()
        all_df = _synthesize_train_test_scope_from_rows(pd.concat(frames, ignore_index=True, sort=False), scores_df=scores_df)
        df = _filter_rule_rows_by_score_scope(all_df, "train+test")
        if df.empty:
            return pd.DataFrame()
    elif scoped_all_path.exists():
        df = pd.read_csv(scoped_all_path, low_memory=False)
    elif all_scopes_path.exists():
        all_df = pd.read_csv(all_scopes_path, low_memory=False)
        df = _filter_rule_rows_by_score_scope(all_df, score_scope)
    elif score_scope == "test" and legacy_all_path.exists():
        df = _filter_rule_rows_by_score_scope(pd.read_csv(legacy_all_path, low_memory=False), "test")
    elif scoped_best_path.exists():
        df = pd.read_csv(scoped_best_path, low_memory=False)
        return _apply_min_dataset_coverage(df, min_dataset_coverage)
    elif score_scope == "test" and legacy_best_path.exists():
        df = _filter_rule_rows_by_score_scope(pd.read_csv(legacy_best_path, low_memory=False), "test")
        return _apply_min_dataset_coverage(df, min_dataset_coverage)
    else:
        return pd.DataFrame()

    df = _filter_rule_rows_by_score_scope(df, score_scope)
    df = _apply_min_dataset_coverage(df, min_dataset_coverage)
    if df.empty:
        return df
    sort_cols = [c for c in ["MCC", "F1", "BalancedAcc"] if c in df.columns]
    for c in sort_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    key_cols = [c for c in ["rule_target_dir", "neuron_key"] if c in df.columns]
    if not key_cols:
        key_cols = [c for c in ["layer_key", "neuron_id"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(key_cols, keep="first")
    return df


# Backward-compatible alias used by older scripts/imports.
def _load_eval_best_rule_metrics(stats_dir: Path, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE) -> pd.DataFrame:
    return _load_best_rule_metrics(stats_dir, min_dataset_coverage=min_dataset_coverage, score_scope="test")

def _count_hq_for_scope(stats_dir: Path, score_scope: str, threshold: float, min_dataset_coverage: float, scores_df: pd.DataFrame = None) -> Any:
    df = _load_best_rule_metrics(stats_dir, min_dataset_coverage=min_dataset_coverage, score_scope=score_scope, scores_df=scores_df)
    if df.empty:
        # Empty can mean either unavailable or truly zero; for legacy test scope
        # this is a legitimate zero, while all-fit requires newly generated
        # scope-specific artifacts.
        if _normalize_score_scope(score_scope) == "train+test":
            scoped_all, scoped_best = _scope_specific_metric_paths(stats_dir, score_scope)
            if not scoped_all.exists() and not scoped_best.exists() and not (stats_dir / "rule_combo_metrics_all_scopes.csv").exists():
                return pd.NA
        return 0
    return int((pd.to_numeric(df.get("MCC"), errors="coerce") >= float(threshold)).sum())


def _table1(data_root: Path, out_dir: Path, run_name: str = MAIN_RUN, tasks: Optional[Sequence[str]] = DEFAULT_MAIN_TASKS, coverage_source: str = "all", min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE, score_scopes: Sequence[str] = ("test", "all_fit")) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    score_scopes = [_normalize_score_scope(s) for s in score_scopes]
    for model_root in _find_model_roots(data_root, tasks=tasks):
        stats_dir = _resolve_stats_dir(model_root, run_name)
        scores_df_scope = None
        sp = _feature_scores_path(model_root)
        if sp is not None:
            try:
                scores_df_scope = _load_scores_df(sp)
            except Exception:
                scores_df_scope = None
        hq_path = stats_dir / "high_quality_neuron_flip_coverage.json"
        global_path = stats_dir / "flip_stats_global.json"
        if not hq_path.exists() and not global_path.exists():
            continue
        hq = _read_json(hq_path) if hq_path.exists() else {}
        flip_global = _read_json(global_path) if global_path.exists() else {}
        task, org, model = _task_org_model(data_root, model_root)

        # Paper definition: direction-specific union flip coverage.  The
        # denominator is the number of eligible source examples for the
        # direction, not the total number of evaluated rows.  The HQ coverage
        # artifact stores the required eligible denominators for both the
        # all-neuron and HQ-only numerators.
        c2i_share = _directional_union_rate_from_hq_coverage(hq, "c2i", coverage_source)
        i2c_share = _directional_union_rate_from_hq_coverage(hq, "i2c", coverage_source)

        if c2i_share != c2i_share:
            c2i_share = _extract_flip_share_from_global(flip_global, "c2i")
            if c2i_share == c2i_share:
                print(
                    f"[table1] warning: {model_root} lacks direction-specific 1->0 "
                    "coverage metadata; using legacy all-row denominator fallback."
                )
        if i2c_share != i2c_share:
            i2c_share = _extract_flip_share_from_global(flip_global, "i2c")
            if i2c_share == i2c_share:
                print(
                    f"[table1] warning: {model_root} lacks direction-specific 0->1 "
                    "coverage metadata; using legacy all-row denominator fallback."
                )

        n_loc = _safe_int(hq.get("n_neurons_total"))
        if n_loc is None:
            n_loc = _safe_int(flip_global.get("n_neurons"))
        row = {
            "Task": _pretty_task(task),
            "Model": _pretty_model(org, model),
            "# Loc.": n_loc,
            "∪(1→0) (elig. %)": round(100.0 * c2i_share, 1) if c2i_share == c2i_share else float("nan"),
            "∪(0→1) (elig. %)": round(100.0 * i2c_share, 1) if i2c_share == i2c_share else float("nan"),
            "task_key": task,
            "model_key": model,
        }
        for scope in score_scopes:
            label = _score_scope_label(scope)
            col = "HQ test" if scope == "test" else f"HQ {label}"
            row[col] = _count_hq_for_scope(stats_dir, scope, 0.85, min_dataset_coverage, scores_df=scores_df_scope)
        # Backward-compatible column expected by older papers/scripts.
        if "HQ test" in row:
            row["HQ"] = row["HQ test"]
        rows.append(row)
    df = pd.DataFrame(rows)
    base_cols = ["Task", "Model", "# Loc."]
    hq_cols = ["HQ test", "HQ all-fit"]
    cov_cols = ["∪(1→0) (elig. %)", "∪(0→1) (elig. %)"]
    if df.empty:
        return _write_empty_table_artifacts(
            out_dir,
            "table1_main_high_quality_neurons",
            base_cols + hq_cols + cov_cols,
            caption="Table 1: End-to-end rule split + spectral coverage results."
        )
    order_task = {"Arithmetic": 0, "Jailbreaking": 1, "HANS NLI": 2}
    order_model = {"Qwen2-7B": 0, "GPT-J": 1, "Qwen2-1.5B": 2}
    df["_ot"] = df["Task"].map(order_task).fillna(99)
    df["_om"] = df["Model"].map(order_model).fillna(99)
    df = df.sort_values(["_ot", "_om", "task_key", "model_key"]).drop(columns=["_ot", "_om", "task_key", "model_key"])
    # Prefer a stable column order while preserving any non-default score scopes.
    ordered = [c for c in base_cols + hq_cols + ["HQ"] + cov_cols if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]
    caption = (
        "Table 1: End-to-end rule split + spectral coverage results. # Loc. counts all rule-bearing localized neurons; "
        f"HQ test counts held-out MCC >= 0.85 and dataset coverage >= {float(min_dataset_coverage):.3g}; "
        "HQ all-fit is descriptive for the same frozen train-selected combos scored on train plus held-out rows. "
        "Directional union flip coverage is normalized by eligible source examples and includes all rule-bearing localized neurons."
        if coverage_source != "hq"
        else "Table 1: High-quality neuron-anchored rules and HQ-only directional flip coverage, normalized by eligible source examples."
    )
    _write_table_artifacts(df, out_dir, "table1_main_high_quality_neurons", caption=caption)
    return df


def _table2(data_root: Path, out_dir: Path, thresholds: Sequence[float], run_names: Sequence[str] = TABLE2_RUNS, tasks: Optional[Sequence[str]] = DEFAULT_MAIN_TASKS, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE, score_scopes: Sequence[str] = ("test", "all_fit")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_run_rows: List[Dict[str, Any]] = []
    score_scopes = [_normalize_score_scope(s) for s in score_scopes]
    model_roots = _find_model_roots(data_root, tasks=tasks)
    for model_root in model_roots:
        task, org, model = _task_org_model(data_root, model_root)
        scores_df_scope = None
        sp = _feature_scores_path(model_root)
        if sp is not None:
            try:
                scores_df_scope = _load_scores_df(sp)
            except Exception:
                scores_df_scope = None
        for run_name in run_names:
            stats_dir = _resolve_stats_dir(model_root, run_name)
            for scope in score_scopes:
                df = _load_best_rule_metrics(stats_dir, min_dataset_coverage=min_dataset_coverage, score_scope=scope, scores_df=scores_df_scope)
                if not df.empty:
                    mcc = pd.to_numeric(df.get("MCC"), errors="coerce")
                    n_rule_bearing = int(mcc.notna().sum())
                    available = True
                else:
                    mcc = pd.Series(dtype=float)
                    n_rule_bearing = 0
                    scoped_all, scoped_best = _scope_specific_metric_paths(stats_dir, scope)
                    available = scope == "test" or scoped_all.exists() or scoped_best.exists() or (stats_dir / "rule_combo_metrics_all_scopes.csv").exists()
                for t in thresholds:
                    per_run_rows.append({
                        "task": task,
                        "org": org,
                        "model": model,
                        "run_name": run_name,
                        "Method": RUN_CONFIGS[run_name]["display"],
                        "score_scope": scope,
                        "score_label": _score_scope_label(scope),
                        "scope_available": bool(available),
                        "threshold": float(t),
                        "n_rule_bearing_neurons": n_rule_bearing if available else pd.NA,
                        "min_dataset_coverage": float(min_dataset_coverage),
                        "n_high_quality_neurons": int((mcc >= float(t)).sum()) if available and not mcc.empty else (0 if available else pd.NA),
                    })
    per_run_df = pd.DataFrame(per_run_rows)
    if per_run_df.empty:
        return per_run_df, per_run_df
    wide_rows = []
    for method in [RUN_CONFIGS[r]["display"] for r in run_names]:
        for scope in score_scopes:
            sub_scope = per_run_df[(per_run_df["Method"] == method) & (per_run_df["score_scope"] == scope)]
            if sub_scope.empty or not bool(sub_scope["scope_available"].any()):
                continue
            row = {"Method": method, "Score scope": _score_scope_label(scope)}
            for t in thresholds:
                sub = sub_scope[sub_scope["threshold"] == float(t)]
                vals = pd.to_numeric(sub["n_high_quality_neurons"], errors="coerce")
                row[f"t = {t:.2f}"] = int(vals.sum()) if vals.notna().any() else pd.NA
            wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)
    _write_table_artifacts(wide_df, out_dir, "table2_threshold_sweep_totals", caption=f"Table 2: Total number of MCC-qualified neurons (MCC ≥ t and dataset coverage ≥ {float(min_dataset_coverage):.3g}) summed over all available task/model runs. TEST is held-out; ALL-FIT is descriptive final-fit selected and scored on all rows.")

    # Also write one compact table per score scope for easy inclusion.
    for scope in score_scopes:
        sub_scope = wide_df[wide_df.get("Score scope", pd.Series(dtype=str)) == _score_scope_label(scope)].copy() if not wide_df.empty else pd.DataFrame()
        if not sub_scope.empty:
            _write_table_artifacts(sub_scope.drop(columns=["Score scope"]), out_dir, f"table2_threshold_sweep_totals_{_score_scope_slug(scope)}", caption=f"Table 2 ({_score_scope_label(scope)}): MCC-qualified neurons (MCC ≥ t and dataset coverage ≥ {float(min_dataset_coverage):.3g}).")

    # Also write threshold-independent yield table.
    yield_rows = []
    for method in [RUN_CONFIGS[r]["display"] for r in run_names]:
        for scope in score_scopes:
            sub = per_run_df[(per_run_df["Method"] == method) & (per_run_df["score_scope"] == scope) & (per_run_df["scope_available"])]
            if sub.empty:
                continue
            per_unit = sub.drop_duplicates(["task", "org", "model", "run_name", "score_scope"])
            yield_rows.append({"Method": method, "Score scope": _score_scope_label(scope), "# rule-bearing localized neurons": int(pd.to_numeric(per_unit["n_rule_bearing_neurons"], errors="coerce").fillna(0).sum())})
    yield_df = pd.DataFrame(yield_rows)
    _write_table_artifacts(yield_df, out_dir, "table2_rule_bearing_yield_totals", caption="Threshold-independent number of rule-bearing localized neurons summed over all available task/model runs, by score scope.")

    per_run_df.to_csv(out_dir / "table2_threshold_sweep_per_run_long.csv", index=False)
    return wide_df, per_run_df


def _table3_bin_specs(bins: Sequence[float]) -> List[Tuple[float, float, str, bool]]:
    specs: List[Tuple[float, float, str, bool]] = []
    if len(bins) < 2:
        raise ValueError("Need at least 2 bin edges")
    for i in range(len(bins) - 1):
        lo = float(bins[i])
        hi = float(bins[i + 1])
        inclusive = i == len(bins) - 2
        label = f"[{lo:.1f}, {hi:.1f}]" if inclusive else f"[{lo:.1f}, {hi:.1f})"
        specs.append((lo, hi, label, inclusive))
    return specs


def _load_effect_map_from_buckets(
    loc_dir: Path,
    min_effect: float = 0.0,
    circuit_ids: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """Load singleton agonist effects from neuron_buckets.json.

    When ``circuit_ids`` is provided, only entries whose recorded
    ``last_record.circuit_id`` belongs to that set are retained.  This is
    important for BF-vs-CHA comparisons: the slow/brute-force anchoring run
    is often executed for a single circuit, while the fast spectral-anchor
    run may contain agonists accumulated across several circuits.  Comparing
    BF against the unfiltered fast run can therefore count a neuron as
    recovered even if CHA found it in a different circuit.
    """
    buckets_path = loc_dir / "neuron_buckets.json"
    if not buckets_path.exists():
        return {}
    buckets = _read_json(buckets_path)
    items = (buckets.get("non_catastrophic_agonists") or {}) if isinstance(buckets, dict) else {}
    circuit_filter: Optional[set[int]] = None
    if circuit_ids is not None:
        circuit_filter = set()
        for cid in circuit_ids:
            try:
                circuit_filter.add(int(cid))
            except Exception:
                continue
    out: Dict[str, float] = {}
    for key, info in items.items():
        if not isinstance(info, dict):
            continue
        rec = info.get("last_record") or {}
        if circuit_filter is not None:
            try:
                cid = int(rec.get("circuit_id"))
            except Exception:
                continue
            if cid not in circuit_filter:
                continue
        effect = rec.get("max_effect", info.get("best_abs_gap", float("nan")))
        eff = abs(_safe_float(effect))
        if eff == eff and eff >= float(min_effect):
            out[str(key)] = eff
    return out


def _collect_circuit_ids_from_buckets(loc_dir: Path) -> List[int]:
    """Return circuit IDs represented in a localization bucket file.

    The value is inferred from non-catastrophic singleton agonist records.  If
    no such records exist, the caller receives an empty list and should avoid
    broadening the comparison to all circuits from the paired CHA run.
    """
    buckets_path = loc_dir / "neuron_buckets.json"
    if not buckets_path.exists():
        return []
    buckets = _read_json(buckets_path)
    items = (buckets.get("non_catastrophic_agonists") or {}) if isinstance(buckets, dict) else {}
    ids = set()
    for info in items.values():
        if not isinstance(info, dict):
            continue
        rec = info.get("last_record") or {}
        try:
            ids.add(int(rec.get("circuit_id")))
        except Exception:
            continue
    return sorted(ids)


def _count_bf_vs_cha_by_bins(bf: Dict[str, float], cha: Dict[str, float], bins: Sequence[float]) -> Dict[str, Any]:
    specs = _table3_bin_specs(bins)
    out: Dict[str, Any] = {
        "overall_bf": int(len(bf)),
        "overall_recovered": int(sum(1 for k in bf if k in cha)),
        "bin_counts": {},
    }
    for lo, hi, label, inclusive in specs:
        if inclusive:
            members = [k for k, v in bf.items() if v >= lo and v <= hi]
        else:
            members = [k for k, v in bf.items() if v >= lo and v < hi]
        recovered = [k for k in members if k in cha]
        out["bin_counts"][label] = {
            "bf": int(len(members)),
            "recovered": int(len(recovered)),
        }
    return out


def _compact_recall_cell(recovered: int, total: int) -> str:
    if total <= 0:
        return "0/0"
    return f"{int(recovered)}/{int(total)} ({100.0 * float(recovered) / float(total):.1f}%)"


def _parse_recall_cell(cell: Any) -> Tuple[int, int]:
    """Parse cells such as '120/160 (75.0%)' or '120/160'."""
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(cell))
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def _compact_budget_cell(numer: int, denom: int) -> str:
    if denom <= 0:
        return ""
    return f"{100.0 * float(numer) / float(denom):.2f}%"


def _parse_ablation_group_parts(value: Any) -> int:
    """Sum '0:2864,1:281' style group-count summaries."""
    total = 0
    for part in str(value).split(','):
        if ':' in part:
            part = part.split(':', 1)[1]
        n = _safe_int(str(part).strip(), default=0)
        total += n
    return int(total)


def _cha_ablation_count_from_rule_knockout(
    model_root: Path,
    run_name: str,
    baseline_subset: str,
) -> Tuple[int, List[str], str]:
    """Return total CHA group ablations for a model/regime."""
    loc_dir = _resolve_localization_dir(model_root, run_name, baseline_subset=baseline_subset)
    rk_path = loc_dir / "rule_knockout.json"
    if not rk_path.exists():
        return 0, [], str(rk_path)
    rk = _read_json(rk_path)
    if not isinstance(rk, list):
        return 0, [], str(rk_path)
    counts: List[int] = []
    parts: List[str] = []
    for i, rec in enumerate(rk):
        if not isinstance(rec, dict) or rec.get("status") != "ok":
            continue
        cid = _safe_int(rec.get("circuit_id"), i)
        n = _safe_int(rec.get("n_ablation_groups"), 0)
        counts.append(n)
        parts.append(f"{cid}:{n}")
    return int(sum(counts)), parts, str(rk_path)


def _architecture_fallback_for_model(model: str) -> Dict[str, int]:
    name = str(model).replace("-Instruct", "")
    candidates = [str(model), name]
    for cand in candidates:
        if cand in MODEL_ARCHITECTURE_FALLBACKS:
            return dict(MODEL_ARCHITECTURE_FALLBACKS[cand])
    for key, value in MODEL_ARCHITECTURE_FALLBACKS.items():
        if key in str(model) or str(model) in key:
            return dict(value)
    return {}


def _observed_hidden_width_from_bf_stats(model_root: Path, baseline_subset: str) -> Optional[int]:
    """Infer per-layer singleton width from BF statistics when present."""
    stats_dir = _resolve_stats_dir(model_root, BF_RUN)
    csv_candidates = [
        stats_dir / "agonist_metric_delta_by_unit.csv",
        stats_dir / "agonist_metric_summary.csv",
    ]
    wanted_source = baseline_subset.replace("_baseline", "")
    for path in csv_candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        cols = [c for c in df.columns if c.startswith("n_compared_activations")]
        if not cols:
            continue
        sub = df
        if "source_parent" in df.columns:
            mask = df["source_parent"].astype(str).str.contains(baseline_subset, regex=False)
            if mask.any():
                sub = df[mask]
            else:
                # Older summary paths contain only positive/negative in the path.
                mask = df["source_parent"].astype(str).str.contains(wanted_source, regex=False)
                if mask.any():
                    sub = df[mask]
        values = []
        for col in cols:
            values.extend(pd.to_numeric(sub[col], errors="coerce").dropna().astype(float).tolist())
        values = [int(v) for v in values if math.isfinite(v) and v > 0]
        if values:
            return int(max(values))
    return None


def _direct_bruteforce_candidate_count_from_artifacts(model_root: Path, baseline_subset: str) -> Tuple[int, str]:
    """Try to read the matched BF singleton candidate count N from artifacts."""
    loc_dir = _resolve_localization_dir(model_root, BF_RUN, baseline_subset=baseline_subset)
    stats_dir = _resolve_stats_dir(model_root, BF_RUN)
    candidate_paths = []
    for base in [loc_dir, stats_dir]:
        candidate_paths.extend([
            base / "ablation_budget.json",
            base / "budget.json",
            base / "run_summary.json",
            base / "metadata.json",
            base / "agonist_metric_stats.json",
        ])
    keys = [
        "n_bruteforce_candidates",
        "n_bruteforce_singleton_candidates",
        "n_singleton_candidates",
        "n_ablation_units",
        "n_candidate_units",
        "n_compared_units_total",
        "n_total_units",
    ]
    for path in _unique_preserve_order(candidate_paths):
        if not path.exists():
            continue
        try:
            obj = _read_json(path)
        except Exception:
            continue
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for key in keys:
                    if key in cur:
                        val = _safe_int(cur.get(key), 0)
                        if val > 0:
                            return val, f"artifact:{path}:{key}"
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
    return 0, ""


def _bruteforce_ablation_count(
    model_root: Path,
    baseline_subset: str,
    brute_force_candidate_cap: int = DEFAULT_BRUTEFORCE_CANDIDATE_CAP,
) -> Tuple[int, str]:
    """Return matched brute-force ablation denominator, 2*N - 1.

    N is the matched brute-force singleton candidate count.  The preferred
    source is an explicit artifact field for N.  When absent, N is inferred
    from n_layers * hidden_width, capped for larger sampled brute-force runs.
    The returned denominator is always 2*N - 1.
    """
    n_candidates, source = _direct_bruteforce_candidate_count_from_artifacts(model_root, baseline_subset)

    if n_candidates <= 0:
        try:
            _, _, model = _task_org_model(model_root.parent.parent.parent, model_root)
        except Exception:
            model = model_root.name
        arch = _architecture_fallback_for_model(model)
        width = _observed_hidden_width_from_bf_stats(model_root, baseline_subset) or arch.get("hidden_size")
        n_layers = arch.get("n_layers")
        if width and n_layers:
            raw = int(width) * int(n_layers)
            if brute_force_candidate_cap and raw > int(brute_force_candidate_cap):
                n_candidates = int(brute_force_candidate_cap)
                source = f"architecture_fallback_candidates:min({raw},{int(brute_force_candidate_cap)})"
            else:
                n_candidates = int(raw)
                source = f"architecture_fallback_candidates:{int(n_layers)}x{int(width)}"

    if n_candidates > 0:
        denom = 2 * int(n_candidates) - 1
        return int(denom), f"2N-1 from {source}; N={int(n_candidates)}"
    return 0, "missing"


def _find_model_root_by_pretty(data_root: Path, task_pretty: str, model_pretty: str) -> Optional[Path]:
    for model_root in _find_model_roots(data_root, tasks=None):
        task, org, model = _task_org_model(data_root, model_root)
        if _pretty_task(task) == str(task_pretty) and _pretty_model(org, model) == str(model_pretty):
            return model_root
    return None


def _find_precomputed_table_file(
    data_root: Path,
    out_dir: Path,
    filename: str,
    paper_tables_root: Optional[Path] = None,
) -> Optional[Path]:
    roots: List[Path] = []
    if paper_tables_root is not None:
        roots.append(Path(paper_tables_root))
    roots.extend([out_dir, data_root])
    roots.extend(list(data_root.resolve().parents)[:4])
    roots.append(Path.cwd())
    for root in _unique_preserve_order([r.resolve() for r in roots if r is not None]):
        direct = [root / filename, root / "paper_tables" / filename]
        for path in direct:
            if path.exists():
                return path
        try:
            hits = sorted(root.rglob(filename))
        except Exception:
            hits = []
        if hits:
            # Prefer non-metadata MacOS copies and paths that look like paper_tables.
            hits = [h for h in hits if "__MACOSX" not in h.parts]
            hits = sorted(hits, key=lambda h: ("paper_tables" not in h.parts, len(h.parts), str(h)))
            if hits:
                return hits[0]
    return None


def _precomputed_cha_count_lookup(table12_path: Optional[Path]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if table12_path is None or not table12_path.exists():
        return lookup
    df = pd.read_csv(table12_path)
    for _, row in df.iterrows():
        baseline = str(row.get("Baseline", "")).strip()
        baseline = {"pos": "positive", "neg": "negative"}.get(baseline, baseline)
        parts_col = "n ablation_groups (by circuit id)"
        total = 0
        if "Total" in df.columns:
            total = _safe_int(row.get("Total"), 0)
        if total <= 0 and parts_col in df.columns:
            total = _parse_ablation_group_parts(row.get(parts_col, ""))
        key = (str(row.get("Task", "")), str(row.get("Model", "")), baseline)
        lookup[key] = {
            "cha_ablation_count": int(total),
            "cha_ablation_parts": str(row.get(parts_col, "")),
            "cha_ablation_source": str(table12_path),
        }
    return lookup


def _collect_table3_records_from_precomputed(
    data_root: Path,
    out_dir: Path,
    bins: Sequence[float],
    paper_tables_root: Optional[Path] = None,
    brute_force_candidate_cap: int = DEFAULT_BRUTEFORCE_CANDIDATE_CAP,
) -> List[Dict[str, Any]]:
    table3_path = _find_precomputed_table_file(data_root, out_dir, "table3_compact_bf_vs_cha.csv", paper_tables_root)
    table12_path = _find_precomputed_table_file(data_root, out_dir, "table12_cha_ablation_budget.csv", paper_tables_root)
    if table3_path is None or not table3_path.exists():
        return []
    df = pd.read_csv(table3_path)
    cha_lookup = _precomputed_cha_count_lookup(table12_path)
    specs = _table3_bin_specs(bins)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        baseline = str(row.get("Baseline", ""))
        if baseline not in {"positive", "negative"}:
            continue
        task_pretty = str(row.get("Task", ""))
        model_pretty = str(row.get("Model", ""))
        if task_pretty == "All" or model_pretty == "All":
            continue
        recovered, total = _parse_recall_cell(row.get("Overall", ""))
        bin_counts = {}
        for _, _, label, _ in specs:
            r, t = _parse_recall_cell(row.get(label, ""))
            bin_counts[label] = {"recovered": int(r), "bf": int(t)}
        key = (task_pretty, model_pretty, baseline)
        cha_info = cha_lookup.get(key, {})
        baseline_subset = "positive_baseline" if baseline == "positive" else "negative_baseline"
        model_root = _find_model_root_by_pretty(data_root, task_pretty, model_pretty)
        bf_count = 0
        bf_source = ""
        task_key = task_pretty
        org = ""
        model_key = model_pretty
        if model_root is not None:
            task_key, org, model_key = _task_org_model(data_root, model_root)
            bf_count, bf_source = _bruteforce_ablation_count(model_root, baseline_subset, brute_force_candidate_cap=brute_force_candidate_cap)
        else:
            # arch = _architecture_fallback_for_model(model_pretty)
            # if arch.get("n_layers") and arch.get("hidden_size"):
            #     raw = int(arch["n_layers"]) * int(arch["hidden_size"])
            #     n_candidates = min(raw, int(brute_force_candidate_cap)) if brute_force_candidate_cap else raw
            #     bf_count = 2 * int(n_candidates) - 1
            #     bf_source = f"2N-1 from architecture_fallback_candidates:{raw}; N={int(n_candidates)}"
            raise ValueError("Missing bruteforce_ablation_count")
        records.append({
            "task": task_key,
            "org": org,
            "model": model_key,
            "Task": task_pretty,
            "Model": model_pretty,
            "Baseline": baseline,
            "overall_bf": int(total),
            "overall_recovered": int(recovered),
            "bin_counts": bin_counts,
            "bf_dir": "",
            "cha_dir": "",
            "comparison_circuit_ids": [],
            "cha_ablation_count": int(cha_info.get("cha_ablation_count", 0)),
            "cha_ablation_parts": cha_info.get("cha_ablation_parts", ""),
            "cha_ablation_source": cha_info.get("cha_ablation_source", ""),
            "bf_ablation_count": int(bf_count),
            "bf_ablation_source": bf_source,
            "precomputed_table3_source": str(table3_path),
            "precomputed_table12_source": "" if table12_path is None else str(table12_path),
        })
    return records


def _table3_collect_baseline_records(data_root: Path, bins: Sequence[float], tasks: Optional[Sequence[str]] = DEFAULT_MAIN_TASKS, model_spec: Optional[str] = None, baseline_subsets: Optional[Sequence[str]] = None, brute_force_candidate_cap: int = DEFAULT_BRUTEFORCE_CANDIDATE_CAP) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    min_effect = float(min(bins[:-1])) if len(bins) > 1 else float(min(bins))
    baseline_map_all = {
        "positive_baseline": "positive",
        "negative_baseline": "negative",
    }
    if baseline_subsets is None:
        baseline_map = baseline_map_all
    else:
        baseline_map = {k: baseline_map_all[k] for k in baseline_subsets if k in baseline_map_all}

    model_filter: Optional[Tuple[str, str]] = None
    if model_spec:
        if "/" not in model_spec:
            raise ValueError(f"Model spec must look like ORG/MODEL, got: {model_spec}")
        model_filter = tuple(model_spec.split("/", 1))  # type: ignore[assignment]

    for model_root in _find_model_roots(data_root, tasks=tasks):
        task, org, model = _task_org_model(data_root, model_root)
        if model_filter is not None and (org, model) != model_filter:
            continue
        pretty_task = _pretty_task(task)
        pretty_model = _pretty_model(org, model)
        for baseline_subset, baseline_label in baseline_map.items():
            bf_dir = _resolve_localization_dir(model_root, BF_RUN, baseline_subset=baseline_subset)
            cha_dir = _resolve_localization_dir(model_root, MAIN_RUN, baseline_subset=baseline_subset)
            if not ((bf_dir / "neuron_buckets.json").exists() and (cha_dir / "neuron_buckets.json").exists()):
                continue
            bf_circuit_ids = _collect_circuit_ids_from_buckets(bf_dir)
            bf = _load_effect_map_from_buckets(bf_dir, min_effect=min_effect)
            cha = _load_effect_map_from_buckets(cha_dir, min_effect=min_effect, circuit_ids=bf_circuit_ids)
            counts = _count_bf_vs_cha_by_bins(bf, cha, bins=bins)
            cha_count, cha_parts, cha_source = _cha_ablation_count_from_rule_knockout(
                model_root, MAIN_RUN, baseline_subset=baseline_subset
            )
            bf_count, bf_source = _bruteforce_ablation_count(
                model_root, baseline_subset=baseline_subset, brute_force_candidate_cap=brute_force_candidate_cap
            )
            records.append({
                "task": task,
                "org": org,
                "model": model,
                "Task": pretty_task,
                "Model": pretty_model,
                "Baseline": baseline_label,
                "overall_bf": counts["overall_bf"],
                "overall_recovered": counts["overall_recovered"],
                "bin_counts": counts["bin_counts"],
                "bf_dir": str(bf_dir),
                "cha_dir": str(cha_dir),
                "comparison_circuit_ids": bf_circuit_ids,
                "cha_ablation_count": int(cha_count),
                "cha_ablation_parts": ",".join(cha_parts),
                "cha_ablation_source": cha_source,
                "bf_ablation_count": int(bf_count),
                "bf_ablation_source": bf_source,
            })
    return records


def _table3_aggregate_records(records: Sequence[Dict[str, Any]], bins: Sequence[float], include_per_model: bool = True) -> pd.DataFrame:
    specs = _table3_bin_specs(bins)
    groups: List[Tuple[str, Any]] = []

    if include_per_model:
        groups.append(("per_model_baseline", ["Task", "Model", "Baseline"]))
        groups.append(("per_model_both", ["Task", "Model"]))

    groups.append(("per_task_both", ["Task"]))
    groups.append(("overall_both", []))

    agg_rows: List[Dict[str, Any]] = []

    for kind, group_cols in groups:
        grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for rec in records:
            key = tuple(rec.get(col) for col in group_cols) if group_cols else ("__all__",)
            bucket = grouped.setdefault(key, {
                "Task": rec["Task"] if "Task" in group_cols else ("All" if kind == "overall_both" else rec["Task"]),
                "Model": rec["Model"] if "Model" in group_cols else ("All" if kind in {"per_task_both", "overall_both"} else rec.get("Model", "All")),
                "Baseline": rec["Baseline"] if "Baseline" in group_cols else "both",
                "#completed experiments": 0,
                "overall_bf": 0,
                "overall_recovered": 0,
                "cha_ablation_count": 0,
                "bf_ablation_count": 0,
                "bin_counts": {label: {"bf": 0, "recovered": 0} for _, _, label, _ in specs},
            })
            bucket["#completed experiments"] += 1
            bucket["overall_bf"] += int(rec["overall_bf"])
            bucket["overall_recovered"] += int(rec["overall_recovered"])
            bucket["cha_ablation_count"] += int(rec.get("cha_ablation_count", 0))
            bucket["bf_ablation_count"] += int(rec.get("bf_ablation_count", 0))
            for _, _, label, _ in specs:
                bucket["bin_counts"][label]["bf"] += int((rec["bin_counts"].get(label) or {}).get("bf", 0))
                bucket["bin_counts"][label]["recovered"] += int((rec["bin_counts"].get(label) or {}).get("recovered", 0))

        for bucket in grouped.values():
            row = {
                "Task": bucket["Task"],
                "Model": bucket["Model"],
                "Baseline": bucket["Baseline"],
                "Budget": _compact_budget_cell(bucket["cha_ablation_count"], bucket["bf_ablation_count"]),
                "CHA ablations": int(bucket["cha_ablation_count"]),
                "BF ablations": int(bucket["bf_ablation_count"]),
                "Overall": _compact_recall_cell(bucket["overall_recovered"], bucket["overall_bf"]),
                "#completed experiments": int(bucket["#completed experiments"]),
            }
            for _, _, label, _ in specs:
                c = bucket["bin_counts"][label]
                row[label] = _compact_recall_cell(c["recovered"], c["bf"])
            agg_rows.append(row)

    df = pd.DataFrame(agg_rows)
    if df.empty:
        return df

    baseline_order = {"positive": 0, "negative": 1, "both": 2}
    task_order = {"Arithmetic": 0, "Jailbreaking": 1, "HANS NLI": 2, "All": 99}
    model_order = {"Qwen2": 0, "GPT-J": 1, "Qwen2-1.5B": 2, "All": 99}
    df["_ot"] = df["Task"].map(task_order).fillna(50)
    df["_om"] = df["Model"].map(model_order).fillna(50)
    df["_ob"] = df["Baseline"].map(baseline_order).fillna(50)
    df["_kind"] = df.apply(lambda r: 0 if r["Task"] != "All" and r["Model"] != "All" else (1 if r["Task"] != "All" and r["Model"] == "All" else 2), axis=1)
    df = df.sort_values(["_ot", "_kind", "_om", "_ob", "Task", "Model", "Baseline"]).drop(columns=["_ot", "_om", "_ob", "_kind"]).reset_index(drop=True)
    return df


def _table3_e2_from_records(records: Sequence[Dict[str, Any]], bins: Sequence[float]) -> pd.DataFrame:
    specs = _table3_bin_specs(bins)
    rows: List[Dict[str, Any]] = []
    sorted_records = sorted(
        records,
        key=lambda r: (
            {"Arithmetic": 0, "Jailbreaking": 1, "HANS NLI": 2}.get(str(r.get("Task")), 50),
            {"Qwen2-7B": 0, "GPT-J": 1, "Qwen2-1.5B": 2}.get(str(r.get("Model")), 50),
            {"positive": 0, "negative": 1}.get(str(r.get("Baseline")), 50),
            str(r.get("Task")),
            str(r.get("Model")),
        ),
    )
    total_recovered = total_bf = total_cha_ab = total_bf_ab = 0
    total_bins = {label: {"recovered": 0, "bf": 0} for _, _, label, _ in specs}
    for rec in sorted_records:
        b = str(rec.get("Baseline", ""))
        if b not in {"positive", "negative"}:
            continue
        cha_ab = int(rec.get("cha_ablation_count", 0))
        bf_ab = int(rec.get("bf_ablation_count", 0))
        total_cha_ab += cha_ab
        total_bf_ab += bf_ab
        total_recovered += int(rec.get("overall_recovered", 0))
        total_bf += int(rec.get("overall_bf", 0))
        row = {
            "Task/model": f"{str(rec.get('Task'))[:5] + '.' if str(rec.get('Task')) == 'Arithmetic' else 'Jail.'}/{rec.get('Model')}",
            "b": "pos." if b == "positive" else "neg.",
            "Budget": _compact_budget_cell(cha_ab, bf_ab),
            "Overall": _compact_recall_cell(int(rec.get("overall_recovered", 0)), int(rec.get("overall_bf", 0))),
        }
        for _, _, label, _ in specs:
            c = rec.get("bin_counts", {}).get(label, {"recovered": 0, "bf": 0})
            r = int(c.get("recovered", 0))
            t = int(c.get("bf", 0))
            total_bins[label]["recovered"] += r
            total_bins[label]["bf"] += t
            row[label] = _compact_recall_cell(r, t)
        rows.append(row)
    total_row = {
        "Task/model": "All completed",
        "b": "all",
        "Budget": _compact_budget_cell(total_cha_ab, total_bf_ab),
        "Overall": _compact_recall_cell(total_recovered, total_bf),
    }
    for _, _, label, _ in specs:
        total_row[label] = _compact_recall_cell(total_bins[label]["recovered"], total_bins[label]["bf"])
    if rows:
        rows.append(total_row)
    return pd.DataFrame(rows)


def _table3(
    data_root: Path,
    out_dir: Path,
    task: str,
    model_spec: str,
    tau: float,
    bins: Sequence[float],
    baseline_subset: str,
    scope: str = "all",
    tasks: Optional[Sequence[str]] = DEFAULT_MAIN_TASKS,
    paper_tables_root: Optional[Path] = None,
    brute_force_candidate_cap: int = DEFAULT_BRUTEFORCE_CANDIDATE_CAP,
) -> pd.DataFrame:
    if scope == "selected":
        records = _table3_collect_baseline_records(
            data_root,
            bins=bins,
            tasks=[task],
            model_spec=model_spec,
            baseline_subsets=[baseline_subset],
            brute_force_candidate_cap=brute_force_candidate_cap,
        )
    else:
        records = _table3_collect_baseline_records(data_root, bins=bins, tasks=tasks, brute_force_candidate_cap=brute_force_candidate_cap)

    if not records:
        records = _collect_table3_records_from_precomputed(
            data_root,
            out_dir,
            bins=bins,
            paper_tables_root=paper_tables_root,
            brute_force_candidate_cap=brute_force_candidate_cap,
        )
        if records:
            print("[table3] using precomputed Table 3/12 artifacts because raw neuron_buckets.json artifacts were not found")

    df = _table3_aggregate_records(records, bins=bins, include_per_model=True)
    if df.empty:
        df = pd.DataFrame(columns=["Task", "Model", "Baseline", "Budget", "CHA ablations", "BF ablations", "Overall", "#completed experiments"] + [label for _, _, label, _ in _table3_bin_specs(bins)])
    _write_table_artifacts(
        df,
        out_dir,
        "table3_compact_bf_vs_cha",
        caption=(
            "Table 3: Compact BF-vs-CHA comparison for rule split + spectral coverage vs. brute-force search. "
            "Budget is CHA group ablations divided by the matched brute-force budget, 2N-1."
        ),
    )
    e2_df = _table3_e2_from_records(records, bins=bins)
    if not e2_df.empty:
        _write_table_artifacts(
            e2_df,
            out_dir,
            "table3_e2_recall_budget",
            caption=(
                "E2: CHA recall and ablation budget versus brute force. "
                "Budget is CHA group ablations divided by the matched brute-force budget, 2N-1."
            ),
        )
    meta = {
        "compared_runs": {
            "cha": MAIN_RUN,
            "bruteforce": BF_RUN,
        },
        "counting_unit": "baseline-specific neuron_buckets.json entries from non_catastrophic_agonists",
        "cha_scope": "filtered to circuit_id values observed in the corresponding brute-force neuron_buckets.json",
        "effect_threshold_for_table3": float(bins[0]) if bins else float(tau),
        "note_on_tau": "Table 3 is binned by --table3_bins; --tau is used by Table 13 and only affects Table 3 when bins are changed accordingly.",
        "bins": list(map(float, bins)),
        "scope": scope,
        "tasks": None if tasks is None else list(tasks),
        "n_completed_baseline_records": int(len(records)),
        "records": [
            {
                "task": rec["task"],
                "org": rec["org"],
                "model": rec["model"],
                "baseline": rec["Baseline"],
                "bf_dir": rec["bf_dir"],
                "cha_dir": rec["cha_dir"],
                "comparison_circuit_ids": rec.get("comparison_circuit_ids", []),
                "overall_bf": rec["overall_bf"],
                "overall_recovered": rec["overall_recovered"],
                "cha_ablation_count": rec.get("cha_ablation_count", 0),
                "cha_ablation_parts": rec.get("cha_ablation_parts", ""),
                "cha_ablation_source": rec.get("cha_ablation_source", ""),
                "bf_ablation_count": rec.get("bf_ablation_count", 0),
                "bf_ablation_source": rec.get("bf_ablation_source", ""),
                "precomputed_table3_source": rec.get("precomputed_table3_source", ""),
                "precomputed_table12_source": rec.get("precomputed_table12_source", ""),
            }
            for rec in records
        ],
    }
    (out_dir / "table3_compact_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


def _table9(data_root: Path, out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_root in _find_model_or_feature_roots(data_root):
        task, org, model = _task_org_model(data_root, model_root)
        metric_value = _metric_from_dataset_stats(model_root, task)
        if metric_value is not None:
            metric, value = metric_value
        else:
            scores_path = _feature_scores_path(model_root)
            if scores_path is None:
                continue
            df = _load_scores_df(scores_path)
            metric = None
            value = float("nan")
            if "is_jailbroken" in df.columns:
                metric = "Jailbreak rate"
                value = pd.to_numeric(df["is_jailbroken"], errors="coerce").mean()
            elif "is_correct" in df.columns:
                metric = "Accuracy"
                value = pd.to_numeric(df["is_correct"], errors="coerce").mean()
            elif any(c.startswith("is_") for c in df.columns):
                cols = [c for c in df.columns if c.startswith("is_")]
                metric = cols[0]
                value = pd.to_numeric(df[cols[0]], errors="coerce").mean()
            if metric is None or value != value:
                continue
        rows.append({
            "Task": _pretty_task(task),
            "Model": _pretty_model(org, model),
            "Metric": metric,
            "Value": _fmt_pct_from_unit(value, nd=1),
            "task_key": task,
            "model_key": model,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return _write_empty_table_artifacts(
            out_dir,
            "table9_baseline_performance",
            ["Task", "Model", "Metric", "Value"],
            caption="Table 9: Baseline performance of each model on its task dataset (unablated)."
        )
    order_task = {"Arithmetic": 0, "Jailbreaking": 1, "HANS NLI": 2}
    order_model = {"Qwen2-7B": 0, "GPT-J": 1, "Qwen2-1.5B": 2}
    df["_ot"] = df["Task"].map(order_task).fillna(99)
    df["_om"] = df["Model"].map(order_model).fillna(99)
    df = df.sort_values(["_ot", "_om", "task_key", "model_key"]).drop(columns=["_ot", "_om", "task_key", "model_key"])
    _write_table_artifacts(df, out_dir, "table9_baseline_performance", caption="Table 9: Baseline performance of each model on its task dataset (unablated).")
    return df


def _aggregate_eapig_neurons(manifest_path: Path) -> pd.DataFrame:
    manifest = _read_json(manifest_path)
    best: Dict[str, float] = {}
    source_circuit: Dict[str, int] = {}
    for entry in manifest:
        cid = _safe_int(entry.get("circuit_id"), -1)
        nls = ((entry.get("metadata_topn") or {}).get("neuron_label_score") or {})
        if not isinstance(nls, dict):
            continue
        for k, v in nls.items():
            layer, neuron_id = _parse_manifest_neuron_label(k)
            if layer is None or neuron_id is None:
                continue
            if not str(layer).startswith("m"):
                continue
            score = abs(_safe_float(v))
            key = f"{layer}:{neuron_id}"
            if score > best.get(key, -float("inf")):
                best[key] = score
                source_circuit[key] = cid
    rows = []
    for key, score in best.items():
        layer, neuron_id = key.split(":", 1)
        rows.append({
            "neuron": key,
            "layer": layer,
            "neuron_id": int(neuron_id),
            "abs_ig": float(score),
            "source_circuit_id": source_circuit.get(key, -1),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_ig", "layer", "neuron_id"], ascending=[False, True, True]).reset_index(drop=True)
    return out


def _top_cha_neurons(stats_dir: Path, top_n: int = 50, rank_metric: str = "c2i") -> pd.DataFrame:
    flip_path = stats_dir / "flip_stats_by_neuron.csv"
    if not flip_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(flip_path, low_memory=False)
    for col in ["c2i_rate", "c2i_count", "i2c_rate", "i2c_count", "flip_any_rate", "flip_any_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "layer_label" not in df.columns and "neuron" in df.columns:
        df["layer_label"] = df["neuron"].astype(str).str.split(":").str[0]
    if "neuron" not in df.columns and {"layer_label", "neuron_id"}.issubset(df.columns):
        df["neuron"] = df["layer_label"].astype(str) + ":" + df["neuron_id"].astype(int).astype(str)
    rank_orders = {
        "c2i": ["c2i_rate", "c2i_count", "flip_any_rate", "flip_any_count"],
        "i2c": ["i2c_rate", "i2c_count", "flip_any_rate", "flip_any_count"],
        "flip_any": ["flip_any_rate", "flip_any_count", "c2i_rate", "c2i_count"],
    }
    sort_cols = [c for c in rank_orders.get(rank_metric, rank_orders["c2i"]) if c in df.columns]
    if not sort_cols:
        return pd.DataFrame()
    df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(int(top_n)).reset_index(drop=True)
    return df


def _table10(data_root: Path, out_dir: Path, task: str, model_spec: str, run_name: str, top_n: int, cha_rank_metric: str = "c2i") -> pd.DataFrame:
    model_root = _model_root_from_spec(data_root, task, model_spec)
    try:
        manifest_path = _discovery_manifest_path(model_root, run_name)
    except FileNotFoundError as e:
        print(f"[table10] skipped: {e}")
        return _write_empty_table_artifacts(
            out_dir,
            "table10_layer_concentration",
            ["Layer", f"EAP-IG top-{top_n}", f"CHA top-{top_n}"],
            caption=f"Table 10: skipped because the EAP-IG manifest was missing."
        )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    eap = _aggregate_eapig_neurons(manifest_path).head(int(top_n))
    cha = _top_cha_neurons(_resolve_stats_dir(model_root, run_name), top_n=top_n, rank_metric=cha_rank_metric)

    eap_counts = eap["layer"].value_counts().to_dict() if not eap.empty else {}
    cha_counts = cha["layer_label"].value_counts().to_dict() if not cha.empty else {}
    layers = sorted(set(eap_counts) | set(cha_counts), key=lambda layer: (-cha_counts.get(layer, 0), -eap_counts.get(layer, 0), _layer_num(layer), str(layer)))
    rows = []
    for layer in layers:
        ce = int(eap_counts.get(layer, 0))
        cc = int(cha_counts.get(layer, 0))
        rows.append({
            "Layer": layer,
            f"EAP-IG top-{top_n}": f"{ce} ({round(100.0 * ce / max(len(eap), 1)):.0f}%)",
            f"CHA top-{top_n}": f"{cc} ({round(100.0 * cc / max(len(cha), 1)):.0f}%)",
        })
    df = pd.DataFrame(rows)
    _write_table_artifacts(df, out_dir, "table10_layer_concentration", caption=f"Table 10: Layer concentration among the top-{top_n} neurons selected by EAP-IG vs. CHA ranked by {cha_rank_metric} flip rate.")
    eap.to_csv(out_dir / "table10_eapig_top_neurons.csv", index=False)
    cha.to_csv(out_dir / "table10_cha_top_neurons.csv", index=False)
    return df


def _table11(data_root: Path, out_dir: Path, task: str, model_spec: str, run_name: str, top_n: int, cha_rank_metric: str = "c2i") -> pd.DataFrame:
    model_root = _model_root_from_spec(data_root, task, model_spec)
    try:
        manifest_path = _discovery_manifest_path(model_root, run_name)
    except FileNotFoundError as e:
        print(f"[table11] skipped: {e}")
        return _write_empty_table_artifacts(
            out_dir,
            "table11_top10_eapig_vs_cha",
            ["Rank", "EAP-IG neuron", "|IG|", "CHA neuron"],
            caption="Table 11: skipped because the EAP-IG manifest was missing."
        )
    eap = _aggregate_eapig_neurons(manifest_path).head(int(top_n)).reset_index(drop=True)
    cha = _top_cha_neurons(_resolve_stats_dir(model_root, run_name), top_n=top_n, rank_metric=cha_rank_metric).reset_index(drop=True)
    rows = []
    for rank in range(int(top_n)):
        eap_row = eap.iloc[rank] if rank < len(eap) else None
        cha_row = cha.iloc[rank] if rank < len(cha) else None
        rows.append({
            "Rank": rank + 1,
            "EAP-IG neuron": "" if eap_row is None else str(eap_row["neuron"]),
            "|IG|": "" if eap_row is None else _fmt_num(eap_row["abs_ig"], nd=5),
            "CHA neuron": "" if cha_row is None else str(cha_row["neuron"]),
        })
    df = pd.DataFrame(rows)
    _write_table_artifacts(df, out_dir, "table11_top10_eapig_vs_cha", caption=f"Table 11: Top-ranked neurons under EAP-IG attribution vs. CHA {cha_rank_metric} flip ranking.")
    return df


def _table12(data_root: Path, out_dir: Path, run_name: str = MAIN_RUN, tasks: Optional[Sequence[str]] = DEFAULT_MAIN_TASKS) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_root in _find_model_roots(data_root, tasks=tasks):
        task, org, model = _task_org_model(data_root, model_root)
        for baseline_subset, baseline_label in [("positive_baseline", "pos"), ("negative_baseline", "neg")]:
            loc_dir = _resolve_localization_dir(model_root, run_name, baseline_subset=baseline_subset)
            rk_path = loc_dir / "rule_knockout.json"
            if not rk_path.exists():
                continue
            rk = _read_json(rk_path)
            if not isinstance(rk, list):
                continue
            ok = [r for r in rk if isinstance(r, dict) and r.get("status") == "ok"]
            if not ok:
                continue
            counts = []
            parts = []
            for rec in ok:
                cid = _safe_int(rec.get("circuit_id"), len(parts))
                n = _safe_int(rec.get("n_ablation_groups"), 0)
                counts.append(n)
                parts.append(f"{cid}:{n}")
            rows.append({
                "Task": _pretty_task(task),
                "Model": _pretty_model(org, model),
                "Baseline": baseline_label,
                "#rules": int(len(ok)),
                "n ablation_groups (by circuit id)": ",".join(parts),
                "Mean": round(float(np.mean(counts)), 1) if counts else float("nan"),
                "Median": int(np.median(counts)) if counts else float("nan"),
                "Min–Max": _fmt_range(min(counts), max(counts)) if counts else "",
                "task_key": task,
                "model_key": model,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return _write_empty_table_artifacts(
            out_dir,
            "table12_cha_ablation_budget",
            ["Task", "Model", "Baseline", "#rules", "n ablation_groups (by circuit id)", "Mean", "Median", "Min–Max"],
            caption="Table 12: Number of ablation groups evaluated by CHA."
        )
    order_task = {"Arithmetic": 0, "Jailbreaking": 1, "HANS NLI": 2}
    order_model = {"Qwen2-7B": 0, "GPT-J": 1, "Qwen2-1.5B": 2}
    order_base = {"pos": 0, "neg": 1}
    df["_ot"] = df["Task"].map(order_task).fillna(99)
    df["_om"] = df["Model"].map(order_model).fillna(99)
    df["_ob"] = df["Baseline"].map(order_base).fillna(99)
    df = df.sort_values(["_ot", "_om", "_ob", "task_key", "model_key"]).drop(columns=["_ot", "_om", "_ob", "task_key", "model_key"])
    _write_table_artifacts(df, out_dir, "table12_cha_ablation_budget", caption="Table 12: Number of ablation groups evaluated by CHA.")
    return df


def _table13(
    data_root: Path,
    out_dir: Path,
    task_a: str,
    task_b: str,
    model_spec: str,
    run_name: str,
    tau: float,
    baseline_subset: str,
    baseline_scope: str = "both",
) -> pd.DataFrame:
    root_a = _model_root_from_spec(data_root, task_a, model_spec)
    root_b = _model_root_from_spec(data_root, task_b, model_spec)
    baseline_subsets = _baseline_subsets_for_table13(baseline_scope, baseline_subset)

    map_a, present_a, missing_a = _load_agonist_union_map_for_baselines(root_a, run_name, tau=tau, baseline_subsets=baseline_subsets)
    map_b, present_b, missing_b = _load_agonist_union_map_for_baselines(root_b, run_name, tau=tau, baseline_subsets=baseline_subsets)

    if not present_a or not present_b:
        missing = []
        if not present_a:
            missing.extend(missing_a)
        if not present_b:
            missing.extend(missing_b)
        print("[table13] skipped: missing neuron_buckets.json artifacts for at least one task:\n" + "\n".join(missing))
        meta = {
            "task_a": task_a,
            "task_b": task_b,
            "model": model_spec,
            "tau": float(tau),
            "baseline_scope": baseline_scope,
            "baseline_subsets": baseline_subsets,
            "status": "skipped_missing_neuron_buckets",
            "present_a": present_a,
            "present_b": present_b,
            "missing_a": missing_a,
            "missing_b": missing_b,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "table13_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out_dir / "table13_intersection_neurons.txt").write_text("", encoding="utf-8")
        return _write_empty_table_artifacts(
            out_dir,
            "table13_cross_task_overlap",
            ["Setting", "|A|", "|J|", "|A ∩ J|", "Jaccard"],
            caption="Table 13: Overlap between localized singleton τ-agonists across tasks."
        )

    set_a = set(map_a.keys())
    set_b = set(map_b.keys())
    inter = set_a & set_b
    union = set_a | set_b
    j = float(len(inter) / len(union)) if union else float("nan")
    model_short = model_spec.split("/", 1)[1].replace("-Instruct", "")
    baseline_label = "both baselines" if baseline_scope == "both" else (baseline_subsets[0].replace("_baseline", "") + " baseline")
    row = {
        "Setting": f"{model_short}, spectral coverage, τ={tau}, {baseline_label}",
        "|A|": int(len(set_a)),
        "|J|": int(len(set_b)),
        "|A ∩ J|": int(len(inter)),
        "Jaccard": _fmt_num(j, nd=3),
    }
    df = pd.DataFrame([row])
    _write_table_artifacts(
        df,
        out_dir,
        "table13_cross_task_overlap",
        caption="Table 13: Overlap between localized singleton τ-agonists across tasks, using the union over the requested baseline regimes."
    )
    (out_dir / "table13_intersection_neurons.txt").write_text(", ".join(sorted(inter)) + "\n", encoding="utf-8")
    meta = {
        "task_a": task_a,
        "task_b": task_b,
        "model": model_spec,
        "tau": float(tau),
        "baseline_scope": baseline_scope,
        "baseline_subsets": baseline_subsets,
        "present_a": present_a,
        "present_b": present_b,
        "missing_a": missing_a,
        "missing_b": missing_b,
        "task_a_neurons": sorted(set_a),
        "task_b_neurons": sorted(set_b),
        "intersection_neurons": sorted(inter),
        "jaccard": j,
    }
    (out_dir / "table13_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate paper tables from MechaRule result directories.")
    ap.add_argument("--data_root", type=str, default="./data", help="Root containing task/org/model result trees.")
    ap.add_argument("--out_dir", type=str, default="./paper_tables", help="Output directory.")
    ap.add_argument("--which", type=str, default="all_supported", help="Comma-separated list among: table1,table2,table3,table9,table10,table11,table12,table13,all_supported. table3 writes a compact BF-vs-CHA summary from neuron_buckets.json.")
    ap.add_argument("--tau", type=float, default=0.2, help="τ threshold for agonists.")
    ap.add_argument("--table3_bins", type=str, default=",".join(map(str, DEFAULT_TABLE3_BINS)), help="Comma-separated Table 3 bin edges.")
    ap.add_argument("--thresholds", type=str, default=",".join(map(str, DEFAULT_THRESHOLDS)), help="Comma-separated thresholds for Table 2.")
    ap.add_argument("--table3_task", type=str, default="arithmetic")
    ap.add_argument("--table3_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--table10_task", type=str, default="arithmetic")
    ap.add_argument("--table10_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--table10_topn", type=int, default=50)
    ap.add_argument("--table11_task", type=str, default="arithmetic")
    ap.add_argument("--table11_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--table11_topn", type=int, default=10)
    ap.add_argument("--table13_task_a", type=str, default="arithmetic")
    ap.add_argument("--table13_task_b", type=str, default="bon_jailbreaking")
    ap.add_argument("--table13_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--table13_baseline_scope", type=str, default="both", choices=["both", "positive", "negative", "selected"], help="Baseline regimes used for Table 13 cross-task overlap. Default 'both' matches the paper's all-localized-neuron overlap; 'selected' uses --baseline_subset.")
    ap.add_argument("--baseline_subset", type=str, default="positive_baseline", choices=["positive_baseline", "negative_baseline"])
    ap.add_argument("--table1_coverage_source", type=str, default="all", choices=["all", "hq"], help="Table 1 flip coverage numerator: all rule-bearing localized neurons (paper default) or HQ-only neurons. Directional rates are normalized by eligible source examples.")
    ap.add_argument("--score_scopes", type=str, default="test,all_fit", help="Comma-separated score scopes for HQ reporting. Default reports held-out test and descriptive all-fit scores selected and scored on all rows.")
    ap.add_argument("--min_rule_dataset_coverage", type=float, default=DEFAULT_MIN_RULE_DATASET_COVERAGE, help="Minimum dataset_coverage required before a rule contributes to HQ counts or threshold sweeps. Use 0 to disable.")
    ap.add_argument("--include_hans_in_main_tables", action="store_true", help="Include HANS/NLI in Tables 1-3/12 when matching compatible artifacts exist. The paper main tables omit HANS.")
    ap.add_argument("--table3_scope", type=str, default="all", choices=["all", "selected"], help="Table 3 scope. 'all' reproduces the paper compact table; 'selected' honors --table3_task/--table3_model/--baseline_subset.")
    ap.add_argument("--paper_tables_root", type=str, default=None, help="Optional directory containing precomputed paper table CSVs. Used as a fallback when raw neuron_buckets/rule_knockout artifacts are absent.")
    ap.add_argument("--bruteforce_candidate_cap", type=int, default=DEFAULT_BRUTEFORCE_CANDIDATE_CAP, help="Cap used when inferring the brute-force candidate count N from model architecture before applying denominator 2*N-1.")
    ap.add_argument("--cha_rank_metric", type=str, default="c2i", choices=["c2i", "i2c", "flip_any"], help="Metric used to rank CHA neurons in Tables 10-11. The paper's Qwen2 arithmetic appendix uses c2i.")
    ap.add_argument("--notes_only", action="store_true", help="Only write a note about unsupported manual tables 4-8.")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    main_table_tasks = None if args.include_hans_in_main_tables else DEFAULT_MAIN_TASKS

    if args.notes_only:
        note = (
            "This utility generates all quantitative result tables supported directly by repository artifacts: "
            "Tables 1-3 and 9-13. Tables 4-8 in the paper are glossary or hand-picked qualitative/illustrative examples "
            "and are therefore not auto-generated from run outputs.\n"
        )
        (out_root / "unsupported_tables_note.txt").write_text(note, encoding="utf-8")
        print(note.strip())
        return

    which_raw = [w.strip() for w in args.which.split(",") if w.strip()]
    if "all_supported" in which_raw:
        which = ["table1", "table2", "table3", "table9", "table10", "table11", "table12", "table13"]
    else:
        which = which_raw

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    score_scopes = [_normalize_score_scope(x.strip()) for x in args.score_scopes.split(",") if x.strip()]
    if not score_scopes:
        score_scopes = ["test", "all_fit"]
    bins = _parse_bins(args.table3_bins)

    results: Dict[str, Any] = {}

    if "table1" in which:
        df = _table1(data_root, out_root, tasks=main_table_tasks, coverage_source=args.table1_coverage_source, min_dataset_coverage=args.min_rule_dataset_coverage, score_scopes=score_scopes)
        results["table1_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table1] no rows")

    if "table2" in which:
        df, long_df = _table2(data_root, out_root, thresholds=thresholds, tasks=main_table_tasks, min_dataset_coverage=args.min_rule_dataset_coverage, score_scopes=score_scopes)
        results["table2_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table2] no rows")

    if "table3" in which:
        df = _table3(data_root, out_root, task=args.table3_task, model_spec=args.table3_model, tau=args.tau, bins=bins, baseline_subset=args.baseline_subset, scope=args.table3_scope, tasks=main_table_tasks, paper_tables_root=Path(args.paper_tables_root).resolve() if args.paper_tables_root else None, brute_force_candidate_cap=args.bruteforce_candidate_cap)
        results["table3_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table3] no rows")

    if "table9" in which:
        df = _table9(data_root, out_root)
        results["table9_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table9] no rows")

    if "table10" in which:
        df = _table10(data_root, out_root, task=args.table10_task, model_spec=args.table10_model, run_name=MAIN_RUN, top_n=args.table10_topn, cha_rank_metric=args.cha_rank_metric)
        results["table10_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table10] no rows")

    if "table11" in which:
        df = _table11(data_root, out_root, task=args.table11_task, model_spec=args.table11_model, run_name=MAIN_RUN, top_n=args.table11_topn, cha_rank_metric=args.cha_rank_metric)
        results["table11_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table11] no rows")

    if "table12" in which:
        df = _table12(data_root, out_root, run_name=MAIN_RUN, tasks=main_table_tasks)
        results["table12_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table12] no rows")

    if "table13" in which:
        df = _table13(data_root, out_root, task_a=args.table13_task_a, task_b=args.table13_task_b, model_spec=args.table13_model, run_name=MAIN_RUN, tau=args.tau, baseline_subset=args.baseline_subset, baseline_scope=args.table13_baseline_scope)
        results["table13_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table13] no rows")

    note = (
        "Supported automatic tables: 1-3 and 9-13. "
        "Tables 4-8 are glossary/hand-picked qualitative examples and are not auto-generated from repository outputs."
    )
    (out_root / "unsupported_tables_note.txt").write_text(note + "\n", encoding="utf-8")
    (out_root / "paper_tables_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(note)


if __name__ == "__main__":
    main()
