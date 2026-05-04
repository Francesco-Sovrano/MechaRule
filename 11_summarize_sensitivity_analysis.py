#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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


def threshold_counts_from_csv(run_dir: Path, thresholds: List[float]) -> Dict[str, float]:
    candidates = [
        run_dir / 'rule_combo_metrics_best_per_neuron.csv',
        run_dir / 'rule_combo_metrics_top_50.csv',
        run_dir / 'rule_combo_metrics_all.csv',
    ]
    df = pd.DataFrame()
    for path in candidates:
        if path.exists():
            df = safe_read_csv(path)
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


def extract_metrics(run_dir: Path, thresholds: List[float]) -> Dict[str, float]:
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

    out.update(threshold_counts_from_csv(run_dir, thresholds))
    return out


def build_summary_df(stats_root: Path, baseline_run_name: str, baseline_m: int, baseline_tau: float, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for run_dir in discover_run_dirs(stats_root):
        M, tau = parse_run_params(run_dir, baseline_run_name, baseline_m, baseline_tau)
        row = {'run_name': run_dir.name, 'run_dir': str(run_dir), 'M': M, 'tau': tau}
        row.update(extract_metrics(run_dir, thresholds))
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
    args = ap.parse_args()

    thresholds = sorted(set(float(x) for x in args.quality_thresholds))
    stats_root = Path(args.stats_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_summary_df(stats_root, args.baseline_run_name, args.baseline_M, args.baseline_tau, thresholds)
    if df.empty:
        raise SystemExit(f'No sensitivity-analysis runs found under: {stats_root}')

    df.to_csv(out_dir / 'sensitivity_summary.csv', index=False)

    baseline_row = choose_baseline(df, args.baseline_run_name, args.baseline_M, args.baseline_tau)
    if baseline_row is None:
        raise SystemExit('Could not determine a baseline run.\nParsed rows:\n' + df[['run_name', 'M', 'tau']].to_string(index=False))

    plot_dir = out_dir / 'plots'
    compact_df = save_threshold_sweep_plots(df, baseline_row, thresholds, plot_dir)

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
        '- `plots/quality_threshold_sensitivity_compact.csv`',
    ]
    if not compact_df.empty:
        lines.append('')
        lines.append('Compact summary rows are saved in `plots/quality_threshold_sensitivity_compact.csv`.')
    (out_dir / 'README_quality_threshold_sensitivity.md').write_text('\n'.join(lines), encoding='utf-8')

    print(f'Baseline run: {baseline_row["run_name"]}')
    print(f'Wrote threshold sensitivity plots under: {plot_dir}')


if __name__ == '__main__':
    main()
