#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_MIN_RULE_DATASET_COVERAGE = 0.005


def _apply_min_dataset_coverage(df: pd.DataFrame, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE) -> pd.DataFrame:
    """Filter rule rows by minimum held-out dataset_coverage.

    This support floor avoids sparse perfect-MCC artifacts driving HQ counts.
    Set min_dataset_coverage <= 0 to disable.  Older artifacts without a
    coverage column are left untouched.
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

def _drop_train_scored_rule_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only held-out/eval-scored rule-combo rows."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "flip_target" in out.columns:
        out = out[~out["flip_target"].astype(str).str.startswith("train_")].copy()
    if "rule_combo_path" in out.columns:
        out = out[~out["rule_combo_path"].astype(str).str.contains("rule_combo_train_", regex=False)].copy()
    return out

def _load_eval_best_rule_rows(run_dir: Path, min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE) -> pd.DataFrame:
    """Load eval-scored rows and keep one best row per localized neuron."""
    all_path = run_dir / "rule_combo_metrics_all.csv"
    best_path = run_dir / "rule_combo_metrics_best_per_neuron.csv"
    if all_path.exists():
        df = _drop_train_scored_rule_rows(safe_read_csv(all_path) if 'safe_read_csv' in globals() else pd.read_csv(all_path, low_memory=False))
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
    if best_path.exists():
        df = _drop_train_scored_rule_rows(safe_read_csv(best_path) if 'safe_read_csv' in globals() else pd.read_csv(best_path, low_memory=False))
        return _apply_min_dataset_coverage(df, min_dataset_coverage)
    return pd.DataFrame()


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN_RE_M = re.compile(r'(?:^|[-_/])M(?P<M>\d+)(?:$|[-_/])', re.IGNORECASE)
RUN_RE_TAU = re.compile(r'(?:^|[-_/])tau(?P<tau>\d+(?:\.\d+)?)(?:$|[-_/])', re.IGNORECASE)

DEFAULT_BASELINE_M = 43008
DEFAULT_BASELINE_TAU = 0.2
DEFAULT_BASELINE_RUN_NAME = 'rule_split-spectral_sample-decode_only-agonist_neurons-fast-spectral_anchor'
DEFAULT_QUALITY_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80]



def safe_load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def looks_like_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / fname).exists() for fname in [
        'rule_metrics_summary.json',
        'flip_stats_global.json',
    ])


def discover_run_dirs(root: Path) -> List[Path]:
    if looks_like_run_dir(root):
        return [root]
    direct = [p for p in sorted(root.iterdir()) if looks_like_run_dir(p)] if root.exists() else []
    if direct:
        return direct
    out = []
    for p in sorted(root.rglob('*')):
        if looks_like_run_dir(p):
            out.append(p)
    uniq = []
    seen = set()
    for p in out:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def parse_run_params(run_dir: Path, baseline_run_name: str, baseline_m: int, baseline_tau: float) -> Tuple[Optional[int], Optional[float]]:
    run_name = run_dir.name
    m = None
    tau = None

    m_match = RUN_RE_M.search(run_name)
    t_match = RUN_RE_TAU.search(run_name)

    if m_match:
        try:
            m = int(m_match.group('M'))
        except Exception:
            m = None
    if t_match:
        try:
            tau = float(t_match.group('tau'))
        except Exception:
            tau = None

    if run_name == baseline_run_name:
        m = baseline_m
        tau = baseline_tau
    elif m is None and tau is not None:
        m = baseline_m
    elif m is not None and tau is None:
        tau = baseline_tau

    return m, tau


def infer_quality_column(df: pd.DataFrame) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for candidate in ['mcc', 'quality', 'quality_metric', 'score']:
        if candidate in cols:
            return cols[candidate]
    for c in df.columns:
        if c.lower().startswith('mcc'):
            return c
    return None


def threshold_counts_from_csv(run_dir: Path, thresholds: List[float], min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE) -> Dict[str, float]:
    df = _load_eval_best_rule_rows(run_dir, min_dataset_coverage=min_dataset_coverage)
    if df.empty:
        candidates = [
            run_dir / 'rule_combo_metrics_top_50.csv',
        ]
        for path in candidates:
            if path.exists():
                df = _apply_min_dataset_coverage(_drop_train_scored_rule_rows(safe_read_csv(path)), min_dataset_coverage)
                if not df.empty:
                    break

    out: Dict[str, float] = {}
    if df.empty:
        return out

    qcol = infer_quality_column(df)
    if qcol is None:
        return out

    quality = pd.to_numeric(df[qcol], errors='coerce').dropna()
    if quality.empty:
        return out

    for thr in thresholds:
        key = f'n_quality_ge_{thr:.2f}'
        out[key] = float((quality >= thr).sum())
    out['n_quality_total_from_csv'] = float(len(quality))
    out['quality_max_from_csv'] = float(quality.max())
    out['quality_median_from_csv'] = float(quality.median())
    return out



def infer_model_root_from_stats_root(stats_root: Path) -> Optional[Path]:
    """Infer the model root from a stats directory or one of its run children."""
    try:
        root = stats_root.resolve()
    except Exception:
        root = stats_root

    candidates = [root] + list(root.parents)
    for candidate in candidates:
        if candidate.name == 'rule_extraction_results':
            return candidate.parent

    # Common layout:
    #   <model_root>/rule_extraction_results/neuron_flip_rules/stats[/<run_name>]
    for candidate in candidates:
        if candidate.name == 'stats' and candidate.parent.name == 'neuron_flip_rules':
            rdir = candidate.parent.parent
            if rdir.name == 'rule_extraction_results':
                return rdir.parent
    return None


def _coerce_int(value) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return out if out >= 0 else None


def _layer_counts_from_mapping(obj) -> Dict[str, int]:
    """Return {layer_label: count} from a circuit-details layer map."""
    out: Dict[str, int] = {}
    if not isinstance(obj, dict):
        return out
    for layer, values in obj.items():
        layer = str(layer)
        if isinstance(values, int):
            n = values
        elif isinstance(values, float) and float(values).is_integer():
            n = int(values)
        elif isinstance(values, (list, tuple, set)):
            n = len({int(v) for v in values if _coerce_int(v) is not None})
        elif isinstance(values, dict):
            # Accept either {unit_id: ...} or nested objects with explicit unit lists.
            unit_like = []
            for key in values.keys():
                ikey = _coerce_int(key)
                if ikey is not None:
                    unit_like.append(ikey)
            if unit_like:
                n = len(set(unit_like))
            else:
                nested = _extract_layer_counts_from_details(values)
                for k, v in nested.items():
                    out[k] = max(out.get(k, 0), int(v))
                continue
        else:
            continue
        if n > 0:
            out[layer] = max(out.get(layer, 0), int(n))
    return out


def _extract_layer_counts_from_details(detail) -> Dict[str, int]:
    """Extract per-layer candidate counts from circuit detail JSON content.

    The preferred source is the actual per-circuit detail file emitted by
    `6_analyze_bag_of_rules.py`. For CHA, each detail record stores the queried
    groups in `ablations`; the depth-0 record for each layer is the root of that
    layer's binary tree, so its `group_size` is N for the layer. If an older
    detail file lacks depths, fall back to the largest queried group per layer,
    then to the union of singleton/unit IDs seen for that layer.
    """
    counts: Dict[str, int] = {}
    if not isinstance(detail, dict):
        return counts

    # First infer from actual ablation records. This is the authoritative path
    # for current runs because it reflects exactly what CHA searched.
    by_depth0: Dict[str, int] = {}
    largest_group: Dict[str, int] = {}
    unit_union: Dict[str, set] = {}
    for rec in detail.get('ablations', []) or []:
        if not isinstance(rec, dict):
            continue
        layer_map = rec.get('layers_neurons_dict') or rec.get('layers_units_dict') or {}
        if not isinstance(layer_map, dict):
            continue
        for layer, units in layer_map.items():
            layer = str(layer)
            if isinstance(units, (list, tuple, set)):
                clean_units = {int(u) for u in units if _coerce_int(u) is not None}
                group_n = len(clean_units)
            elif isinstance(units, dict):
                clean_units = {int(k) for k in units.keys() if _coerce_int(k) is not None}
                group_n = len(clean_units)
            else:
                clean_units = set()
                group_n = _coerce_int(rec.get('group_size')) or 0
            if clean_units:
                unit_union.setdefault(layer, set()).update(clean_units)
            group_size = _coerce_int(rec.get('group_size')) or group_n
            if group_size > 0:
                largest_group[layer] = max(largest_group.get(layer, 0), int(group_size))
                if _coerce_int(rec.get('depth')) == 0:
                    by_depth0[layer] = max(by_depth0.get(layer, 0), int(group_size))

    # Depth-0 roots are best; largest groups are a safe fallback; singleton union
    # is a last resort for non-CHA/exhaustive detail files.
    for source in (by_depth0, largest_group):
        for layer, n in source.items():
            if n > 0:
                counts[layer] = max(counts.get(layer, 0), int(n))
    for layer, units in unit_union.items():
        if units:
            counts[layer] = max(counts.get(layer, 0), len(units))

    # If ablation records are unavailable, fall back to explicit maps in the
    # same circuit-detail file. Do not use model-width or EAP-IG path fallbacks.
    if not counts:
        explicit_keys = [
            'circuit_units', 'circuit_neurons', 'candidate_units', 'candidate_neurons',
            'layer_units', 'layer_neurons', 'neurons', 'mlp_neurons',
            'layer_neuron_counts', 'layer_unit_counts', 'candidate_counts_by_layer',
            'candidate_neuron_counts_by_layer', 'circuit_neuron_counts_by_layer',
        ]
        for key in explicit_keys:
            obj = detail.get(key)
            if isinstance(obj, dict):
                for layer, n in _layer_counts_from_mapping(obj).items():
                    counts[layer] = max(counts.get(layer, 0), int(n))
    return counts


def _detail_paths_for_record(loc_dir: Path, rec: dict) -> List[Path]:
    """Return likely per-circuit detail JSON paths for one compact summary row."""
    out: List[Path] = []
    cid = _coerce_int(rec.get('circuit_id'))
    if cid is None:
        return out
    analysis_mode = str(rec.get('analysis_mode') or 'rule')
    if analysis_mode == 'spectral_cluster':
        out.append(loc_dir / 'spectral_clusters' / f'cluster_{cid:04d}.json')
        out.extend(sorted(loc_dir.rglob(f'cluster_{cid:04d}.json')))
    else:
        target = rec.get('rule_target') or rec.get('target')
        if target is not None:
            out.append(loc_dir / str(target) / f'rule_{cid:04d}.json')
        out.extend(sorted(loc_dir.rglob(f'rule_{cid:04d}.json')))
    # Deduplicate while preserving order.
    seen = set()
    uniq = []
    for path in out:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    return uniq


def _load_matching_circuit_detail(loc_dir: Path, rec: dict) -> Optional[dict]:
    """Load the circuit detail file matching a compact rule_knockout entry."""
    for path in _detail_paths_for_record(loc_dir, rec):
        if not path.exists() or path.name in {'rule_knockout.json', 'neuron_buckets.json', 'neuron_bucket_stats.json'}:
            continue
        detail = safe_load_json(path)
        if not isinstance(detail, dict):
            continue
        cid = _coerce_int(rec.get('circuit_id'))
        did = _coerce_int(detail.get('circuit_id'))
        if cid is not None and did is not None and cid != did:
            continue
        # If labels are present in both files, require them to agree.
        rlabel = rec.get('circuit_label')
        dlabel = detail.get('circuit_label')
        if rlabel is not None and dlabel is not None and str(rlabel) != str(dlabel):
            continue
        return detail
    return None


def layerwise_tree_budget(layer_counts: Dict[str, int]) -> Optional[int]:
    """Full binary tree budget summed across independently searched layers."""
    vals = [int(n) for n in layer_counts.values() if int(n) > 0]
    if not vals:
        return None
    return int(sum(2 * n - 1 for n in vals))


def build_localization_index(stats_root: Path) -> Dict[str, List[Path]]:
    """Map stats run names to CHA localization baseline directories.

    The stats run name is built as ``<circuit_label>-<bag_label>`` in
    ``_run_pipeline.sh``. The corresponding localization outputs live under
    ``<model_root>/neural_circuit_discovery_results*/<circuit_label>/<bag_label>/``.
    """
    model_root = infer_model_root_from_stats_root(stats_root)
    if model_root is None:
        return {}

    index: Dict[str, List[Path]] = {}
    for discovery_root in sorted(model_root.glob('neural_circuit_discovery_results*')):
        if not discovery_root.is_dir():
            continue
        for rk_path in sorted(discovery_root.rglob('rule_knockout.json')):
            baseline_dir = rk_path.parent
            bag_dir = baseline_dir.parent
            circuit_dir = bag_dir.parent
            if not bag_dir.name or not circuit_dir.name:
                continue
            candidate_names = {
                f'{circuit_dir.name}-{bag_dir.name}',
                bag_dir.name,
                circuit_dir.name,
            }
            for run_name in candidate_names:
                index.setdefault(run_name, []).append(baseline_dir)
    return index


def _finite_numeric(values) -> np.ndarray:
    nums = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if np.isfinite(f):
            nums.append(f)
    return np.asarray(nums, dtype=float)


def cha_budget_metrics(run_name: str, localization_index: Dict[str, List[Path]]) -> Dict[str, float]:
    """Summarize CHA group-evaluation budget for one stats run.

    The raw scalar is `n_ablation_groups` from each compact `rule_knockout.json`
    entry. The percentage denominator is computed per circuit from the matching
    circuit-detail JSON file: sum_l (2N_l - 1), where N_l is the searched unit
    count for layer l inferred from the depth-0 CHA records in `ablations`.
    """
    loc_dirs = localization_index.get(run_name, [])
    rows = []
    seen_files = set()

    for loc_dir in loc_dirs:
        rk_path = loc_dir / 'rule_knockout.json'
        try:
            resolved = str(rk_path.resolve())
        except Exception:
            resolved = str(rk_path)
        if resolved in seen_files or not rk_path.exists():
            continue
        seen_files.add(resolved)

        records = safe_load_json(rk_path)
        if not isinstance(records, list):
            continue
        baseline = loc_dir.name
        if baseline.endswith('_baseline'):
            baseline = baseline[:-len('_baseline')]

        for rec in records:
            if not isinstance(rec, dict):
                continue
            if rec.get('status') != 'ok':
                continue
            arr = _finite_numeric([rec.get('n_ablation_groups')])
            if not arr.size:
                continue

            detail = _load_matching_circuit_detail(loc_dir, rec)
            layer_counts = _extract_layer_counts_from_details(detail) if detail else {}
            denom = layerwise_tree_budget(layer_counts)
            pct = (100.0 * float(arr[0]) / float(denom)) if denom and denom > 0 else np.nan

            rows.append({
                'baseline_subset': baseline,
                'n_ablation_groups': float(arr[0]),
                'tree_budget_denominator': float(denom) if denom is not None else np.nan,
                'pct_of_layerwise_tree_budget': float(pct),
                'n_layers_with_budget': float(len(layer_counts)) if layer_counts else np.nan,
                'layer_counts_json': json.dumps(layer_counts, sort_keys=True),
            })

    if not rows:
        return {}

    values = _finite_numeric(row['n_ablation_groups'] for row in rows)
    if values.size == 0:
        return {}

    pct_values = _finite_numeric(row['pct_of_layerwise_tree_budget'] for row in rows)
    denom_values = _finite_numeric(row['tree_budget_denominator'] for row in rows)
    layer_counts_values = _finite_numeric(row['n_layers_with_budget'] for row in rows)

    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    sem = float(std / np.sqrt(values.size)) if values.size > 1 else 0.0
    out: Dict[str, float] = {
        'cha_budget_n_circuits': float(values.size),
        'cha_budget_total_n_ablation_groups': float(values.sum()),
        'cha_budget_mean_n_ablation_groups': float(values.mean()),
        'cha_budget_std_n_ablation_groups': std,
        'cha_budget_sem_n_ablation_groups': sem,
        'cha_budget_ci95_n_ablation_groups': float(1.96 * sem),
        'cha_budget_q25_n_ablation_groups': float(np.percentile(values, 25)),
        'cha_budget_q75_n_ablation_groups': float(np.percentile(values, 75)),
        'cha_budget_min_n_ablation_groups': float(values.min()),
        'cha_budget_max_n_ablation_groups': float(values.max()),
        'cha_budget_n_circuits_with_tree_denominator': float(pct_values.size),
    }
    if denom_values.size:
        out.update({
            'cha_budget_mean_layerwise_tree_denominator': float(denom_values.mean()),
            'cha_budget_min_layerwise_tree_denominator': float(denom_values.min()),
            'cha_budget_max_layerwise_tree_denominator': float(denom_values.max()),
        })
    if layer_counts_values.size:
        out.update({
            'cha_budget_mean_n_layers_with_budget': float(layer_counts_values.mean()),
            'cha_budget_min_n_layers_with_budget': float(layer_counts_values.min()),
            'cha_budget_max_n_layers_with_budget': float(layer_counts_values.max()),
        })
    if pct_values.size:
        pstd = float(pct_values.std(ddof=1)) if pct_values.size > 1 else 0.0
        psem = float(pstd / np.sqrt(pct_values.size)) if pct_values.size > 1 else 0.0
        out.update({
            'cha_budget_mean_pct_of_layerwise_tree_budget': float(pct_values.mean()),
            'cha_budget_std_pct_of_layerwise_tree_budget': pstd,
            'cha_budget_sem_pct_of_layerwise_tree_budget': psem,
            'cha_budget_ci95_pct_of_layerwise_tree_budget': float(1.96 * psem),
            'cha_budget_q25_pct_of_layerwise_tree_budget': float(np.percentile(pct_values, 25)),
            'cha_budget_q75_pct_of_layerwise_tree_budget': float(np.percentile(pct_values, 75)),
            'cha_budget_min_pct_of_layerwise_tree_budget': float(pct_values.min()),
            'cha_budget_max_pct_of_layerwise_tree_budget': float(pct_values.max()),
        })

    for baseline in sorted({row['baseline_subset'] for row in rows}):
        bvals = _finite_numeric(row['n_ablation_groups'] for row in rows if row['baseline_subset'] == baseline)
        bpcts = _finite_numeric(row['pct_of_layerwise_tree_budget'] for row in rows if row['baseline_subset'] == baseline)
        bdenoms = _finite_numeric(row['tree_budget_denominator'] for row in rows if row['baseline_subset'] == baseline)
        if bvals.size:
            prefix = re.sub(r'[^0-9A-Za-z]+', '_', baseline).strip('_').lower()
            bstd = float(bvals.std(ddof=1)) if bvals.size > 1 else 0.0
            bsem = float(bstd / np.sqrt(bvals.size)) if bvals.size > 1 else 0.0
            out[f'cha_budget_{prefix}_n_circuits'] = float(bvals.size)
            out[f'cha_budget_{prefix}_mean_n_ablation_groups'] = float(bvals.mean())
            out[f'cha_budget_{prefix}_std_n_ablation_groups'] = bstd
            out[f'cha_budget_{prefix}_sem_n_ablation_groups'] = bsem
            out[f'cha_budget_{prefix}_ci95_n_ablation_groups'] = float(1.96 * bsem)
            if bdenoms.size:
                out[f'cha_budget_{prefix}_mean_layerwise_tree_denominator'] = float(bdenoms.mean())
            if bpcts.size:
                bpstd = float(bpcts.std(ddof=1)) if bpcts.size > 1 else 0.0
                bpsem = float(bpstd / np.sqrt(bpcts.size)) if bpcts.size > 1 else 0.0
                out[f'cha_budget_{prefix}_mean_pct_of_layerwise_tree_budget'] = float(bpcts.mean())
                out[f'cha_budget_{prefix}_std_pct_of_layerwise_tree_budget'] = bpstd
                out[f'cha_budget_{prefix}_sem_pct_of_layerwise_tree_budget'] = bpsem
                out[f'cha_budget_{prefix}_ci95_pct_of_layerwise_tree_budget'] = float(1.96 * bpsem)
    return out


def extract_metrics(
    run_dir: Path,
    thresholds: List[float],
    localization_index: Optional[Dict[str, List[Path]]] = None,
    min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE,
) -> Dict[str, float]:
    rule = safe_load_json(run_dir / 'rule_metrics_summary.json')
    flip = safe_load_json(run_dir / 'flip_stats_global.json')

    out: Dict[str, float] = {}

    def put(name: str, value) -> None:
        try:
            out[name] = np.nan if value is None else float(value)
        except Exception:
            out[name] = np.nan

    put('n_neurons_with_rules', rule.get('n_neurons_with_rules'))
    put('quality_threshold_from_json', rule.get('quality_threshold'))
    put('n_neurons_quality_ge_threshold_from_json', rule.get('n_neurons_quality_ge_threshold'))
    put('MCC_median', ((rule.get('MCC') or {}).get('median')))
    put('MCC_max', ((rule.get('MCC') or {}).get('max')))
    put('BalancedAcc_median', ((rule.get('BalancedAcc') or {}).get('median')))
    put('dataset_coverage_median', ((rule.get('dataset_coverage') or {}).get('median')))
    put('union_flip_any_unique_rate', flip.get('union_flip_any_unique_rate'))

    out.update(threshold_counts_from_csv(run_dir, thresholds, min_dataset_coverage=min_dataset_coverage))
    if localization_index:
        out.update(cha_budget_metrics(run_dir.name, localization_index))
    return out


def build_summary_df(stats_root: Path, baseline_run_name: str, baseline_m: int, baseline_tau: float, thresholds: List[float], min_dataset_coverage: float = DEFAULT_MIN_RULE_DATASET_COVERAGE) -> pd.DataFrame:
    rows = []
    localization_index = build_localization_index(stats_root)
    for run_dir in discover_run_dirs(stats_root):
        M, tau = parse_run_params(run_dir, baseline_run_name, baseline_m, baseline_tau)
        row = {'run_name': run_dir.name, 'run_dir': str(run_dir), 'M': M, 'tau': tau}
        row.update(extract_metrics(run_dir, thresholds, localization_index, min_dataset_coverage=min_dataset_coverage))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['M'] = pd.to_numeric(df['M'], errors='coerce').astype('Int64')
    df['tau'] = pd.to_numeric(df['tau'], errors='coerce')
    return df.sort_values(['M', 'tau', 'run_name'], kind='stable').reset_index(drop=True)


def choose_baseline(df: pd.DataFrame, baseline_run_name: str, baseline_m: int, baseline_tau: float) -> Optional[pd.Series]:
    exact_name = df[df['run_name'] == baseline_run_name]
    if not exact_name.empty:
        return exact_name.iloc[0]

    exact = df[
        df['M'].notna() &
        (df['M'].astype(float) == float(baseline_m)) &
        df['tau'].notna() &
        np.isclose(df['tau'].astype(float), float(baseline_tau), equal_nan=False)
    ]
    if not exact.empty:
        return exact.iloc[0]
    return None


def save_threshold_sweep_plots(df: pd.DataFrame, baseline_row: pd.Series, thresholds: List[float], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_m = int(baseline_row['M'])
    baseline_tau = float(baseline_row['tau'])

    tau_df = df[(df['M'] == baseline_m)].copy()
    tau_df = tau_df.dropna(subset=['tau']).sort_values('tau')

    m_df = df[df['tau'].notna() & np.isclose(df['tau'], baseline_tau, equal_nan=False)].copy()
    m_df = m_df.dropna(subset=['M']).sort_values('M')

    summary_rows = []

    for thr in thresholds:
        col = f'n_quality_ge_{thr:.2f}'
        if col not in df.columns:
            continue

        # vs tau
        d = tau_df[['tau', col]].copy()
        d['tau'] = pd.to_numeric(d['tau'], errors='coerce')
        d[col] = pd.to_numeric(d[col], errors='coerce')
        d = d.dropna().sort_values('tau')
        if not d.empty:
            d.to_csv(out_dir / f'count_ge_{thr:.2f}_vs_tau.csv', index=False)

        # vs M
        dm = m_df[['M', col]].copy()
        dm['M'] = pd.to_numeric(dm['M'], errors='coerce')
        dm[col] = pd.to_numeric(dm[col], errors='coerce')
        dm = dm.dropna().sort_values('M')
        if not dm.empty:
            dm.to_csv(out_dir / f'count_ge_{thr:.2f}_vs_M.csv', index=False)

        base_val = pd.to_numeric(pd.Series([baseline_row.get(col)]), errors='coerce').iloc[0]
        if np.isfinite(base_val):
            if not d.empty:
                summary_rows.append({
                    'factor': 'tau',
                    'threshold': thr,
                    'baseline_count': float(base_val),
                    'median_abs_change': float(np.median(np.abs(d[col].to_numpy(dtype=float) - base_val))),
                    'max_abs_change': float(np.max(np.abs(d[col].to_numpy(dtype=float) - base_val))),
                })
            if not dm.empty:
                summary_rows.append({
                    'factor': 'M',
                    'threshold': thr,
                    'baseline_count': float(base_val),
                    'median_abs_change': float(np.median(np.abs(dm[col].to_numpy(dtype=float) - base_val))),
                    'max_abs_change': float(np.max(np.abs(dm[col].to_numpy(dtype=float) - base_val))),
                })

    # Multi-threshold plot vs tau
    plt.figure(figsize=(7.2, 4.8))
    plotted = False
    for thr in thresholds:
        col = f'n_quality_ge_{thr:.2f}'
        if col not in tau_df.columns:
            continue
        d = tau_df[['tau', col]].copy()
        d['tau'] = pd.to_numeric(d['tau'], errors='coerce')
        d[col] = pd.to_numeric(d[col], errors='coerce')
        d = d.dropna().sort_values('tau')
        if d.empty:
            continue
        plt.plot(d['tau'].to_numpy(), d[col].to_numpy(), marker='o', label=f'MCC ≥ {thr:.2f}')
        plotted = True
    if plotted:
        plt.xlabel('tau')
        plt.ylabel('# neurons above threshold')
        plt.title(f'Sensitivity across quality thresholds vs tau (M={baseline_m})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'quality_threshold_sensitivity_vs_tau.pdf', bbox_inches='tight')
    plt.close()

    # Multi-threshold plot vs M
    plt.figure(figsize=(7.2, 4.8))
    plotted = False
    for thr in thresholds:
        col = f'n_quality_ge_{thr:.2f}'
        if col not in m_df.columns:
            continue
        d = m_df[['M', col]].copy()
        d['M'] = pd.to_numeric(d['M'], errors='coerce')
        d[col] = pd.to_numeric(d[col], errors='coerce')
        d = d.dropna().sort_values('M')
        if d.empty:
            continue
        plt.plot(d['M'].to_numpy(), d[col].to_numpy(), marker='o', label=f'MCC ≥ {thr:.2f}')
        plotted = True
    if plotted:
        plt.xlabel('M')
        plt.ylabel('# neurons above threshold')
        plt.title(f'Sensitivity across quality thresholds vs M (tau={baseline_tau})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'quality_threshold_sensitivity_vs_M.pdf', bbox_inches='tight')
    plt.close()

    # Heatmap over thresholds x tau
    tau_matrix = []
    tau_index = []
    tau_values = None
    for thr in thresholds:
        col = f'n_quality_ge_{thr:.2f}'
        if col not in tau_df.columns:
            continue
        d = tau_df[['tau', col]].copy()
        d['tau'] = pd.to_numeric(d['tau'], errors='coerce')
        d[col] = pd.to_numeric(d[col], errors='coerce')
        d = d.dropna().sort_values('tau')
        if d.empty:
            continue
        if tau_values is None:
            tau_values = d['tau'].to_numpy()
        if len(d) == len(tau_values):
            tau_matrix.append(d[col].to_numpy(dtype=float))
            tau_index.append(thr)
    if tau_matrix:
        arr = np.vstack(tau_matrix)
        plt.figure(figsize=(1.8 + 1.2 * arr.shape[1], 1.8 + 0.5 * arr.shape[0]))
        im = plt.imshow(arr, aspect='auto', origin='lower')
        plt.xticks(range(len(tau_values)), [f'{x:g}' for x in tau_values])
        plt.yticks(range(len(tau_index)), [f'{x:.2f}' for x in tau_index])
        plt.xlabel('tau')
        plt.ylabel('quality threshold')
        plt.title('Count of neurons above threshold')
        plt.colorbar(im).set_label('# neurons')
        plt.tight_layout()
        plt.savefig(out_dir / 'quality_threshold_sensitivity_heatmap_tau.pdf', bbox_inches='tight')
        plt.close()

    # Heatmap over thresholds x M
    m_matrix = []
    m_index = []
    m_values = None
    for thr in thresholds:
        col = f'n_quality_ge_{thr:.2f}'
        if col not in m_df.columns:
            continue
        d = m_df[['M', col]].copy()
        d['M'] = pd.to_numeric(d['M'], errors='coerce')
        d[col] = pd.to_numeric(d[col], errors='coerce')
        d = d.dropna().sort_values('M')
        if d.empty:
            continue
        if m_values is None:
            m_values = d['M'].to_numpy()
        if len(d) == len(m_values):
            m_matrix.append(d[col].to_numpy(dtype=float))
            m_index.append(thr)
    if m_matrix:
        arr = np.vstack(m_matrix)
        plt.figure(figsize=(1.8 + 1.2 * arr.shape[1], 1.8 + 0.5 * arr.shape[0]))
        im = plt.imshow(arr, aspect='auto', origin='lower')
        plt.xticks(range(len(m_values)), [str(int(x)) for x in m_values])
        plt.yticks(range(len(m_index)), [f'{x:.2f}' for x in m_index])
        plt.xlabel('M')
        plt.ylabel('quality threshold')
        plt.title('Count of neurons above threshold')
        plt.colorbar(im).set_label('# neurons')
        plt.tight_layout()
        plt.savefig(out_dir / 'quality_threshold_sensitivity_heatmap_M.pdf', bbox_inches='tight')
        plt.close()

    # CHA budget over tau. Budget is measured as CHA group evaluations per
    # circuit. The paper-facing plot shows the absolute mean group-evaluation
    # count as the y-value, with +/- one standard-error error bars across
    # localized circuits. Labels also show the mean percentage of each circuit's
    # own layerwise full binary tree budget, sum_l (2N_l - 1), where N_l is read
    # from the corresponding circuit-detail JSON.
    budget_required = ['cha_budget_mean_n_ablation_groups']
    budget_optional = [
        'cha_budget_sem_n_ablation_groups',
        'cha_budget_ci95_n_ablation_groups',
        'cha_budget_n_circuits',
        'cha_budget_n_circuits_with_tree_denominator',
        'cha_budget_total_n_ablation_groups',
        'cha_budget_mean_layerwise_tree_denominator',
        'cha_budget_min_layerwise_tree_denominator',
        'cha_budget_max_layerwise_tree_denominator',
        'cha_budget_mean_n_layers_with_budget',
        'cha_budget_mean_pct_of_layerwise_tree_budget',
        'cha_budget_sem_pct_of_layerwise_tree_budget',
        'cha_budget_ci95_pct_of_layerwise_tree_budget',
    ]
    budget_cols = budget_required + [c for c in budget_optional if c in tau_df.columns]
    if all(c in tau_df.columns for c in budget_required):
        budget_tau = tau_df[['tau'] + budget_cols].copy()
        budget_tau['tau'] = pd.to_numeric(budget_tau['tau'], errors='coerce')
        for col in budget_cols:
            budget_tau[col] = pd.to_numeric(budget_tau[col], errors='coerce')
        budget_tau = budget_tau.dropna(subset=['tau', 'cha_budget_mean_n_ablation_groups']).sort_values('tau')

        if not budget_tau.empty:
            budget_tau.to_csv(out_dir / 'cha_budget_vs_tau.csv', index=False)

            x = budget_tau['tau'].to_numpy(dtype=float)
            y = budget_tau['cha_budget_mean_n_ablation_groups'].to_numpy(dtype=float)
            if 'cha_budget_sem_n_ablation_groups' in budget_tau.columns:
                yerr = budget_tau['cha_budget_sem_n_ablation_groups'].to_numpy(dtype=float)
            else:
                yerr = np.zeros_like(y)
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
            if 'cha_budget_mean_pct_of_layerwise_tree_budget' in budget_tau.columns:
                pct = budget_tau['cha_budget_mean_pct_of_layerwise_tree_budget'].to_numpy(dtype=float)
            else:
                pct = np.full_like(y, np.nan, dtype=float)

            plt.figure(figsize=(5.9, 3.6))
            plt.errorbar(x, y, yerr=yerr, marker='o', capsize=3, linewidth=1.6)
            y_range = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
            offset = max(8.0, 0.02 * y_range)
            for xi, yi, pi in zip(x, y, pct):
                if np.isfinite(xi) and np.isfinite(yi):
                    label = f'{yi:.0f}'
                    if np.isfinite(pi):
                        label += f'\n({pi:.1f}%)'
                    plt.annotate(
                        label,
                        (xi, yi),
                        textcoords='offset points',
                        xytext=(0, offset),
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.78),
                    )
            plt.xlabel(r'CHA threshold $\tau$')
            plt.ylabel('Mean CHA group evaluations')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / 'cha_budget_vs_tau.pdf', bbox_inches='tight')
            plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / 'quality_threshold_sensitivity_compact.csv', index=False)
    return summary_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--baseline_M', type=int, default=DEFAULT_BASELINE_M)
    ap.add_argument('--baseline_tau', type=float, default=DEFAULT_BASELINE_TAU)
    ap.add_argument('--baseline_run_name', default=DEFAULT_BASELINE_RUN_NAME)
    ap.add_argument('--quality_thresholds', nargs='*', type=float, default=DEFAULT_QUALITY_THRESHOLDS)
    ap.add_argument('--min_rule_dataset_coverage', type=float, default=DEFAULT_MIN_RULE_DATASET_COVERAGE,
                    help='Minimum held-out dataset_coverage required before a rule contributes to threshold-sensitivity counts. Use 0 to disable.')
    ap.add_argument('--paper_fig_dir', default=None, help='Optional paper figure directory. If set, cha_budget_vs_tau.pdf is copied there.')
    args = ap.parse_args()

    thresholds = sorted(set(float(x) for x in args.quality_thresholds))
    stats_root = Path(args.stats_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_summary_df(stats_root, args.baseline_run_name, args.baseline_M, args.baseline_tau, thresholds, min_dataset_coverage=args.min_rule_dataset_coverage)
    if df.empty:
        raise SystemExit(f'No sensitivity-analysis runs found under: {stats_root}')

    df.to_csv(out_dir / 'sensitivity_summary.csv', index=False)

    baseline_row = choose_baseline(df, args.baseline_run_name, args.baseline_M, args.baseline_tau)
    if baseline_row is None:
        raise SystemExit('Could not determine a baseline run.\nParsed rows:\n' + df[['run_name', 'M', 'tau']].to_string(index=False))

    plot_dir = out_dir / 'plots'
    compact_df = save_threshold_sweep_plots(df, baseline_row, thresholds, plot_dir)

    if args.paper_fig_dir:
        paper_fig_dir = Path(args.paper_fig_dir)
        paper_fig_dir.mkdir(parents=True, exist_ok=True)
        for fname in ['cha_budget_vs_tau.pdf']:
            src = plot_dir / fname
            if src.exists():
                shutil.copy2(src, paper_fig_dir / fname)

    lines = [
        '# Sensitivity across quality thresholds',
        '',
        f'- baseline run: `{baseline_row["run_name"]}`',
        f'- baseline M: {int(baseline_row["M"])}',
        f'- baseline tau: {float(baseline_row["tau"])}',
        f'- thresholds: {", ".join(f"{x:.2f}" for x in thresholds)}',
        '',
        'Main outputs:',
        '- `plots/quality_threshold_sensitivity_vs_tau.pdf`',
        '- `plots/quality_threshold_sensitivity_vs_M.pdf`',
        '- `plots/quality_threshold_sensitivity_heatmap_tau.pdf`',
        '- `plots/quality_threshold_sensitivity_heatmap_M.pdf`',
        '- `plots/cha_budget_vs_tau.pdf` (mean CHA group-evaluation count with +/-1 SE error bars; labels also show % of per-circuit sum_l(2N_l-1) from circuit-detail JSON)',
        '- `plots/cha_budget_vs_tau.csv` (absolute budget metrics and % of per-circuit layerwise tree-budget values)',
        '- `plots/quality_threshold_sensitivity_compact.csv`',
    ]
    if not compact_df.empty:
        lines.append('')
        lines.append('Compact summary rows are saved in `plots/quality_threshold_sensitivity_compact.csv`.')
    (out_dir / 'README_quality_threshold_sensitivity.md').write_text('\n'.join(lines), encoding='utf-8')

    print(f'Baseline run: {baseline_row["run_name"]}')
    print('CHA budget percentages use per-circuit detail JSON denominators: sum_l(2N_l-1).')
    print(f'Wrote threshold sensitivity plots under: {plot_dir}')


if __name__ == '__main__':
    main()
