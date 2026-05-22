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

DEFAULT_THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.99]
DEFAULT_TABLE3_BINS = [0.2, 0.3, 0.5, 1.0]

# Main paper tables intentionally cover arithmetic and jailbreaking.
# HANS/NLI is an appendix task and can be included with the CLI flag.
DEFAULT_MAIN_TASKS = ("arithmetic", "bon_jailbreaking", "jailbreaking")

SENSITIVITY_ARTIFACT_PATTERNS = (
    re.compile(r"-M\d+"),
    re.compile(r"-tau0\.\d"),
)


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
    tex = df.to_latex(index=index, escape=False)
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


def _table1(data_root: Path, out_dir: Path, run_name: str = MAIN_RUN, tasks: Optional[Sequence[str]] = DEFAULT_MAIN_TASKS, coverage_source: str = "all") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_root in _find_model_roots(data_root, tasks=tasks):
        stats_dir = _resolve_stats_dir(model_root, run_name)
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

        rows.append({
            "Task": _pretty_task(task),
            "Model": _pretty_model(org, model),
            "#HQ neurons": _safe_int(hq.get("n_neurons_high_quality")),
            "∪(1→0) (elig. %)": round(100.0 * c2i_share, 1) if c2i_share == c2i_share else float("nan"),
            "∪(0→1) (elig. %)": round(100.0 * i2c_share, 1) if i2c_share == i2c_share else float("nan"),
            "task_key": task,
            "model_key": model,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return _write_empty_table_artifacts(
            out_dir,
            "table1_main_high_quality_neurons",
            ["Task", "Model", "#HQ neurons", "∪(1→0) (elig. %)", "∪(0→1) (elig. %)"],
            caption="Table 1: End-to-end rule split + spectral coverage results."
        )
    order_task = {"Arithmetic": 0, "Jailbreaking": 1, "HANS NLI": 2}
    order_model = {"Qwen2-7B": 0, "GPT-J": 1, "Qwen2-1.5B": 2}
    df["_ot"] = df["Task"].map(order_task).fillna(99)
    df["_om"] = df["Model"].map(order_model).fillna(99)
    df = df.sort_values(["_ot", "_om", "task_key", "model_key"]).drop(columns=["_ot", "_om", "task_key", "model_key"])
    caption = (
        "Table 1: End-to-end rule split + spectral coverage results. HQ neurons have held-out MCC >= 0.85; "
        "directional union flip coverage is normalized by eligible source examples and includes all rule-bearing localized neurons."
        if coverage_source != "hq"
        else "Table 1: High-quality neuron-anchored rules and HQ-only directional flip coverage, normalized by eligible source examples."
    )
    _write_table_artifacts(df, out_dir, "table1_main_high_quality_neurons", caption=caption)
    return df


def _table2(data_root: Path, out_dir: Path, thresholds: Sequence[float], run_names: Sequence[str] = TABLE2_RUNS, tasks: Optional[Sequence[str]] = DEFAULT_MAIN_TASKS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_run_rows: List[Dict[str, Any]] = []
    model_roots = _find_model_roots(data_root, tasks=tasks)
    for model_root in model_roots:
        task, org, model = _task_org_model(data_root, model_root)
        for run_name in run_names:
            stats_dir = _resolve_stats_dir(model_root, run_name)
            csv_path = stats_dir / "rule_combo_metrics_best_per_neuron.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, low_memory=False)
            mcc = pd.to_numeric(df.get("MCC"), errors="coerce")
            for t in thresholds:
                per_run_rows.append({
                    "task": task,
                    "org": org,
                    "model": model,
                    "run_name": run_name,
                    "Method": RUN_CONFIGS[run_name]["display"],
                    "threshold": float(t),
                    "n_high_quality_neurons": int((mcc >= float(t)).sum()),
                })
    per_run_df = pd.DataFrame(per_run_rows)
    if per_run_df.empty:
        return per_run_df, per_run_df
    wide_rows = []
    for method in [RUN_CONFIGS[r]["display"] for r in run_names]:
        row = {"Method": method}
        for t in thresholds:
            sub = per_run_df[(per_run_df["Method"] == method) & (per_run_df["threshold"] == float(t))]
            row[f"t = {t:.2f}"] = int(sub["n_high_quality_neurons"].sum()) if not sub.empty else 0
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)
    _write_table_artifacts(wide_df, out_dir, "table2_threshold_sweep_totals", caption="Table 2: Total number of high-quality neurons (MCC ≥ t) summed over all available task/model runs.")
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


def _table3_collect_baseline_records(data_root: Path, bins: Sequence[float], tasks: Optional[Sequence[str]] = DEFAULT_MAIN_TASKS, model_spec: Optional[str] = None, baseline_subsets: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
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
                "bin_counts": {label: {"bf": 0, "recovered": 0} for _, _, label, _ in specs},
            })
            bucket["#completed experiments"] += 1
            bucket["overall_bf"] += int(rec["overall_bf"])
            bucket["overall_recovered"] += int(rec["overall_recovered"])
            for _, _, label, _ in specs:
                bucket["bin_counts"][label]["bf"] += int((rec["bin_counts"].get(label) or {}).get("bf", 0))
                bucket["bin_counts"][label]["recovered"] += int((rec["bin_counts"].get(label) or {}).get("recovered", 0))

        for bucket in grouped.values():
            row = {
                "Task": bucket["Task"],
                "Model": bucket["Model"],
                "Baseline": bucket["Baseline"],
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
) -> pd.DataFrame:
    if scope == "selected":
        records = _table3_collect_baseline_records(
            data_root,
            bins=bins,
            tasks=[task],
            model_spec=model_spec,
            baseline_subsets=[baseline_subset],
        )
    else:
        records = _table3_collect_baseline_records(data_root, bins=bins, tasks=tasks)

    df = _table3_aggregate_records(records, bins=bins, include_per_model=True)
    if df.empty:
        df = pd.DataFrame(columns=["Task", "Model", "Baseline", "Overall", "#completed experiments"] + [label for _, _, label, _ in _table3_bin_specs(bins)])
    _write_table_artifacts(
        df,
        out_dir,
        "table3_compact_bf_vs_cha",
        caption=(
            "Table 3: Compact BF-vs-CHA comparison for rule split + spectral coverage vs. brute-force search. "
            "Counts are baseline-specific and CHA recovery is restricted to the brute-force circuit(s)."
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
    ap.add_argument("--include_hans_in_main_tables", action="store_true", help="Include HANS/NLI in Tables 1-3/12 when matching compatible artifacts exist. The paper main tables omit HANS.")
    ap.add_argument("--table3_scope", type=str, default="all", choices=["all", "selected"], help="Table 3 scope. 'all' reproduces the paper compact table; 'selected' honors --table3_task/--table3_model/--baseline_subset.")
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
    bins = _parse_bins(args.table3_bins)

    results: Dict[str, Any] = {}

    if "table1" in which:
        df = _table1(data_root, out_root, tasks=main_table_tasks, coverage_source=args.table1_coverage_source)
        results["table1_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table1] no rows")

    if "table2" in which:
        df, long_df = _table2(data_root, out_root, thresholds=thresholds, tasks=main_table_tasks)
        results["table2_rows"] = int(len(df))
        print(df.to_string(index=False) if not df.empty else "[table2] no rows")

    if "table3" in which:
        df = _table3(data_root, out_root, task=args.table3_task, model_spec=args.table3_model, tau=args.tau, bins=bins, baseline_subset=args.baseline_subset, scope=args.table3_scope, tasks=main_table_tasks)
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
