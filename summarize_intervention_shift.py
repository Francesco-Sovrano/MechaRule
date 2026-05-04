#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def iter_rule_jsons(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob('*.json')):
        if path.name == 'manifest.json':
            continue
        yield path


def flatten_records(root: Path) -> Dict[Tuple[int, str, int], dict]:
    out = {}
    for path in iter_rule_jsons(root):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        circuit_id = payload.get('circuit_id')
        if circuit_id is None:
            continue
        for rec in payload.get('ablations', []):
            layer_map = rec.get('layers_neurons_dict') or {}
            if len(layer_map) != 1:
                continue
            layer_label, neuron_ids = next(iter(layer_map.items()))
            if len(neuron_ids) != 1:
                continue
            neuron_id = int(neuron_ids[0])
            key = (int(circuit_id), str(layer_label), neuron_id)
            out[key] = {
                'circuit_id': int(circuit_id),
                'layer_label': str(layer_label),
                'neuron_id': neuron_id,
                'max_effect': float(rec.get('max_effect', np.nan)),
                'accuracy_gap': float(rec.get('accuracy_gap', np.nan)),
            }
    return out


def rankdata_desc(xs: np.ndarray) -> np.ndarray:
    order = np.argsort(-xs, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(xs) + 1, dtype=float)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float('nan')
    rx = rankdata_desc(x)
    ry = rankdata_desc(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline_dir', required=True)
    ap.add_argument('--candidate_dir', required=True)
    ap.add_argument('--output_csv', required=True)
    args = ap.parse_args()

    a = flatten_records(Path(args.baseline_dir))
    b = flatten_records(Path(args.candidate_dir))
    keys = sorted(set(a) & set(b))
    rows = []
    for key in keys:
        ra = a[key]
        rb = b[key]
        rows.append({
            **{k: ra[k] for k in ('circuit_id', 'layer_label', 'neuron_id')},
            'baseline_max_effect': ra['max_effect'],
            'candidate_max_effect': rb['max_effect'],
            'baseline_accuracy_gap': ra['accuracy_gap'],
            'candidate_accuracy_gap': rb['accuracy_gap'],
            'delta_max_effect': rb['max_effect'] - ra['max_effect'],
            'delta_accuracy_gap': rb['accuracy_gap'] - ra['accuracy_gap'],
        })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else [
                'circuit_id','layer_label','neuron_id','baseline_max_effect','candidate_max_effect',
                'baseline_accuracy_gap','candidate_accuracy_gap','delta_max_effect','delta_accuracy_gap'
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print('No overlapping singleton ablation records found.')
        return

    base_eff = np.array([r['baseline_max_effect'] for r in rows], dtype=float)
    cand_eff = np.array([r['candidate_max_effect'] for r in rows], dtype=float)
    base_gap = np.array([r['baseline_accuracy_gap'] for r in rows], dtype=float)
    cand_gap = np.array([r['candidate_accuracy_gap'] for r in rows], dtype=float)

    print(f'n_overlap={len(rows)}')
    print(f'spearman_max_effect={spearman(base_eff, cand_eff):.4f}')
    print(f'spearman_accuracy_gap={spearman(base_gap, cand_gap):.4f}')
    print(f'mean_delta_max_effect={(cand_eff - base_eff).mean():.4f}')
    print(f'mean_delta_accuracy_gap={(cand_gap - base_gap).mean():.4f}')


if __name__ == '__main__':
    main()
