#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RATE_COLS = [
    "flip_any_rate",
    "c2i_rate",
    "i2c_rate",
    "semantic_wrong_rate",
    "flip_semantic_wrong_rate",
    "flip_empty_rate",
    "flip_unparseable_rate",
]
COUNT_COLS = [
    "flip_any_count",
    "c2i_count",
    "i2c_count",
    "semantic_wrong_count",
    "flip_semantic_wrong_count",
    "flip_empty_count",
    "flip_unparseable_count",
]
GLOBAL_KEYS = [
    "n_neurons",
    "n_evaluated_rows",
    "sum_flip_any_counts_over_neurons",
    "sum_c2i_counts_over_neurons",
    "sum_i2c_counts_over_neurons",
    "sum_semantic_wrong_counts_over_neurons",
    "sum_flip_semantic_wrong_counts_over_neurons",
    "union_flip_any_unique_count",
    "union_flip_any_unique_rate",
    "union_c2i_unique_count",
    "union_c2i_unique_rate",
    "union_i2c_unique_count",
    "union_i2c_unique_rate",
    "union_semantic_wrong_unique_count",
    "union_semantic_wrong_unique_rate",
    "union_flip_semantic_wrong_unique_count",
    "union_flip_semantic_wrong_unique_rate",
    "union_flip_empty_unique_count",
    "union_flip_empty_unique_rate",
    "union_flip_unparseable_unique_count",
    "union_flip_unparseable_unique_rate",
]
DISPLAY_METRIC_NAMES = {
    "flip_any": "Any flip",
    "flip_c2i": "Correct→incorrect",
    "flip_i2c": "Incorrect→correct",
    "flip_semantic_wrong": "Semantic-wrong flip",
}

INTERVENTION_DESCRIPTIONS = {
    "mean": (
        "Replaces the ablated neuron/unit activation with the pooled mean activation "
        "estimated from the sampled training-prompt mean pool. In script 7 this uses "
        "the same mean-style replacement precompute path as the other mean variants."
    ),
    "mean-donor": (
        "Uses the mean-style replacement machinery but with the donor-safe mean-donor "
        "variant. In script 7 this is passed through the same intervention switch and "
        "uses singleton donor-safe row-wise hooks for batched neuron evaluation."
    ),
    "zero": "Sets the ablated neuron/unit activation to zero.",
    "mean-positional": "Mean replacement with position-aware replacement values.",
    "mean-donor-positional": "Donor-safe mean replacement with position-aware replacement values.",
}



def _safe_layer_label(label: object) -> str:
    """Match the layer-key convention used by 7_refine_neuron_anchored_rules.py."""
    s = str(label)
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_") or "layer"


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] could not read {path}: {exc}")
        return {}


def _read_flip_stats(stats_dir: Path) -> pd.DataFrame:
    path = stats_dir / "flip_stats_by_neuron.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "layer_key" not in df.columns:
        if "layer_label" in df.columns:
            df["layer_key"] = df["layer_label"].map(_safe_layer_label)
        else:
            raise ValueError(f"{path} is missing both layer_key and layer_label")
    if "neuron_id" not in df.columns:
        raise ValueError(f"{path} is missing neuron_id")
    df["layer_key"] = df["layer_key"].astype(str)
    df["neuron_id"] = pd.to_numeric(df["neuron_id"], errors="coerce").astype("Int64")
    df = df.loc[df["neuron_id"].notna()].copy()
    df["neuron_id"] = df["neuron_id"].astype(int)
    if "layer_label" not in df.columns:
        df["layer_label"] = df["layer_key"]
    if "neuron" not in df.columns:
        df["neuron"] = df["layer_label"].astype(str) + ":" + df["neuron_id"].astype(str)
    for col in RATE_COLS + COUNT_COLS + ["n_eval"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep_cols = [
        "layer_key", "layer_label", "neuron_id", "neuron", "n_eval",
        *[c for c in COUNT_COLS if c in df.columns],
        *[c for c in RATE_COLS if c in df.columns],
    ]
    return df[keep_cols].drop_duplicates(subset=["layer_key", "neuron_id"], keep="first")


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = np.isfinite(xs.to_numpy(dtype=float)) & np.isfinite(ys.to_numpy(dtype=float))
    if int(mask.sum()) < 3:
        return None
    xv = xs[mask]
    yv = ys[mask]
    if xv.nunique(dropna=True) < 2 or yv.nunique(dropna=True) < 2:
        return None
    val = xv.corr(yv, method="pearson")
    return float(val) if val is not None and np.isfinite(float(val)) else None


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = np.isfinite(xs.to_numpy(dtype=float)) & np.isfinite(ys.to_numpy(dtype=float))
    if int(mask.sum()) < 3:
        return None
    xv = xs[mask]
    yv = ys[mask]
    if xv.nunique(dropna=True) < 2 or yv.nunique(dropna=True) < 2:
        return None
    val = xv.corr(yv, method="spearman")
    return float(val) if val is not None and np.isfinite(float(val)) else None


def _sign_test_p(delta: np.ndarray) -> Optional[float]:
    nz = delta[np.isfinite(delta) & (delta != 0)]
    n = int(len(nz))
    if n == 0:
        return None
    k = int((nz > 0).sum())
    lo = min(k, n - k)
    if n <= 200:
        prob = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
        return float(min(1.0, prob))
    z = (lo + 0.5 - n * 0.5) / math.sqrt(n * 0.25)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(min(1.0, 2.0 * cdf))


def _bootstrap_mean_ci(delta: np.ndarray, seed: int = 0, n_boot: int = 4000) -> Tuple[Optional[float], Optional[float]]:
    delta = delta[np.isfinite(delta)]
    n = int(len(delta))
    if n < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = delta[idx].mean(axis=1)
    lo, hi = np.nanpercentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _paired_metric_stats(df: pd.DataFrame, metric: str, use_zero_for_missing: bool) -> dict:
    bcol = f"baseline_{metric}"
    ccol = f"candidate_{metric}"
    if bcol not in df.columns or ccol not in df.columns:
        return {"metric": metric, "available": False}
    sub = df.copy()
    if use_zero_for_missing:
        x = pd.to_numeric(sub[bcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = pd.to_numeric(sub[ccol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        sub = sub.loc[sub["present_baseline"] & sub["present_candidate"]].copy()
        x = pd.to_numeric(sub[bcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[ccol], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    delta = y - x
    n = int(len(delta))
    if n == 0:
        return {"metric": metric, "available": False, "n": 0}
    ci_lo, ci_hi = _bootstrap_mean_ci(delta)
    return {
        "metric": metric,
        "available": True,
        "n": n,
        "baseline_mean": float(np.mean(x)),
        "candidate_mean": float(np.mean(y)),
        "mean_delta": float(np.mean(delta)),
        "mean_delta_ci95_low": ci_lo,
        "mean_delta_ci95_high": ci_hi,
        "median_delta": float(np.median(delta)),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "fraction_candidate_higher": float(np.mean(delta > 0)),
        "fraction_candidate_lower": float(np.mean(delta < 0)),
        "fraction_unchanged": float(np.mean(delta == 0)),
        "pearson": _pearson(x, y),
        "spearman": _spearman(x, y),
        "sign_test_p_two_sided": _sign_test_p(delta),
    }


def _merge_flip_stats(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    if baseline.empty and candidate.empty:
        return pd.DataFrame()

    def prefixed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = df.copy()
        rename = {c: f"{prefix}_{c}" for c in out.columns if c not in {"layer_key", "neuron_id"}}
        return out.rename(columns=rename)

    merged = prefixed(baseline, "baseline").merge(
        prefixed(candidate, "candidate"),
        on=["layer_key", "neuron_id"],
        how="outer",
        indicator=True,
    )
    merged["present_baseline"] = merged["_merge"].isin(["left_only", "both"])
    merged["present_candidate"] = merged["_merge"].isin(["right_only", "both"])
    merged["status"] = np.select(
        [
            merged["present_baseline"] & merged["present_candidate"],
            merged["present_candidate"] & ~merged["present_baseline"],
            merged["present_baseline"] & ~merged["present_candidate"],
        ],
        ["overlap", "gained", "lost"],
        default="unknown",
    )
    merged = merged.drop(columns=["_merge"])

    if "baseline_neuron" in merged.columns or "candidate_neuron" in merged.columns:
        merged["neuron"] = merged.get("baseline_neuron", pd.Series(index=merged.index, dtype=object)).combine_first(
            merged.get("candidate_neuron", pd.Series(index=merged.index, dtype=object))
        )
    else:
        merged["neuron"] = merged["layer_key"].astype(str) + ":" + merged["neuron_id"].astype(str)

    for metric in RATE_COLS + COUNT_COLS:
        bcol = f"baseline_{metric}"
        ccol = f"candidate_{metric}"
        if bcol in merged.columns or ccol in merged.columns:
            if bcol not in merged.columns:
                merged[bcol] = np.nan
            if ccol not in merged.columns:
                merged[ccol] = np.nan
            b = pd.to_numeric(merged[bcol], errors="coerce")
            c = pd.to_numeric(merged[ccol], errors="coerce")
            merged[f"delta_{metric}"] = c.fillna(0.0) - b.fillna(0.0)
            both = merged["present_baseline"] & merged["present_candidate"] & b.notna() & c.notna()
            merged[f"paired_delta_{metric}"] = np.where(both, c - b, np.nan)

    sort_cols = ["status"]
    if "delta_flip_any_rate" in merged.columns:
        merged["abs_delta_flip_any_rate"] = merged["delta_flip_any_rate"].abs()
        sort_cols.append("abs_delta_flip_any_rate")
        merged = merged.sort_values(sort_cols, ascending=[True, False]).reset_index(drop=True)
    return merged



def _intervention_description(label: str) -> str:
    return INTERVENTION_DESCRIPTIONS.get(str(label), "No built-in description for this intervention label.")


def _first_json_value(payload: object, keys: Sequence[str]) -> Optional[str]:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload and payload[key] not in (None, ""):
                return str(payload[key])
        for value in payload.values():
            found = _first_json_value(value, keys)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload[:25]:
            found = _first_json_value(value, keys)
            if found:
                return found
    return None


def _infer_context_from_nearby_files(
    baseline_dir: Path,
    candidate_dir: Path,
    task_name: Optional[str],
    model_name: Optional[str],
) -> dict:
    task = task_name or ""
    model = model_name or ""
    candidate_files = [
        "dataset_info.json",
        "run_config.json",
        "args.json",
        "manifest.json",
        "semantic_runtime_config.json",
    ]
    roots = []
    for d in [baseline_dir, candidate_dir]:
        d = Path(d).resolve()
        roots.extend([d, *list(d.parents)[:6]])

    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for name in candidate_files:
            fp = root / name
            if not fp.exists() or not fp.is_file():
                continue
            payload = _read_json(fp)
            if not task:
                task = _first_json_value(payload, ["task_name", "task", "task_module", "task_module_name"]) or ""
            if not model:
                model = _first_json_value(payload, ["ai_model", "model", "model_name", "hf_model", "base_model"]) or ""
            if task and model:
                break
        if task and model:
            break

    return {
        "task_name": task or "unknown task",
        "model_name": model or "unknown model",
    }


def _pct_num(num: object, den: object) -> float:
    n = _to_float(num)
    d = _to_float(den)
    if not math.isfinite(n) or not math.isfinite(d) or d == 0:
        return float("nan")
    return 100.0 * n / d


def _rate_pct(value: object) -> float:
    v = _to_float(value)
    return 100.0 * v if math.isfinite(v) else float("nan")


def _load_hq_summary(stats_dir: Path, *, fallback_n_identified: int) -> dict:
    """Load script-7 high-quality summary, with rule-metrics fallback if HQ coverage was not written."""
    hq_path = stats_dir / "high_quality_neuron_flip_coverage.json"
    rule_path = stats_dir / "rule_metrics_summary.json"
    hq = _read_json(hq_path)
    rule = _read_json(rule_path)

    quality_metric_label = hq.get("quality_metric_label") or rule.get("quality_metric_label")
    quality_threshold = hq.get("quality_threshold") if hq else rule.get("quality_threshold")

    n_identified = hq.get("n_neurons_total")
    if n_identified is None:
        n_identified = rule.get("n_neurons_with_rules")
    if n_identified is None:
        n_identified = fallback_n_identified

    n_hq = hq.get("n_neurons_high_quality")
    if n_hq is None:
        n_hq = rule.get("n_neurons_quality_ge_threshold")

    try:
        n_identified_i = int(n_identified)
    except Exception:
        n_identified_i = int(fallback_n_identified)
    try:
        n_hq_i = int(n_hq) if n_hq is not None else None
    except Exception:
        n_hq_i = None

    frac_hq = (float(n_hq_i) / float(n_identified_i)) if (n_hq_i is not None and n_identified_i) else float("nan")

    def flip_block(name: str) -> dict:
        block = hq.get(name, {}) if isinstance(hq, dict) else {}
        if not isinstance(block, dict):
            block = {}
        union_all = block.get("union_all")
        union_high = block.get("union_high")
        sum_all = block.get("sum_all")
        sum_high = block.get("sum_high")
        if name == "flip_any":
            den = block.get("n_rows_with_any_eval_all")
            all_pct = _pct_num(union_all, den)
            high_pct = _pct_num(union_high, den)
            den_label = "evaluated rows"
        elif name in {"flip_c2i", "flip_i2c"}:
            den = block.get("eligible_baseline_count")
            all_pct = _rate_pct(block.get("eligible_union_rate_all"))
            high_pct = _rate_pct(block.get("eligible_union_rate_high"))
            if not math.isfinite(all_pct):
                all_pct = _pct_num(union_all, den)
            if not math.isfinite(high_pct):
                high_pct = _pct_num(union_high, den)
            den_label = "eligible rows"
        else:
            den = None
            all_pct = float("nan")
            high_pct = float("nan")
            den_label = "rows"
        coverage = block.get("union_share_of_all")
        if coverage is None:
            coverage = _to_float(union_high) / _to_float(union_all) if _to_float(union_all) else float("nan")
        return {
            "union_all": _jsonable(union_all),
            "union_high": _jsonable(union_high),
            "sum_all": _jsonable(sum_all),
            "sum_high": _jsonable(sum_high),
            "denominator": _jsonable(den),
            "denominator_label": den_label,
            "all_union_pct": all_pct,
            "high_union_pct": high_pct,
            "hq_coverage_share_pct": _rate_pct(coverage),
        }

    return {
        "available": bool(hq) or bool(rule),
        "coverage_json": str(hq_path) if hq_path.exists() else None,
        "rule_metrics_json": str(rule_path) if rule_path.exists() else None,
        "quality_metric_label": quality_metric_label or "HQ metric",
        "quality_threshold": _jsonable(quality_threshold),
        "n_identified_neurons": n_identified_i,
        "n_high_quality_neurons": n_hq_i,
        "frac_high_quality": frac_hq,
        "flip_any": flip_block("flip_any"),
        "flip_c2i": flip_block("flip_c2i"),
        "flip_i2c": flip_block("flip_i2c"),
        "flip_semantic_wrong": flip_block("flip_semantic_wrong"),
    }


def _hq_rows_for_csv(label: str, hq_summary: Mapping[str, object]) -> List[dict]:
    rows = [{
        "intervention": label,
        "metric": "neurons",
        "identified_neurons": hq_summary.get("n_identified_neurons"),
        "high_quality_neurons": hq_summary.get("n_high_quality_neurons"),
        "high_quality_fraction": hq_summary.get("frac_high_quality"),
        "quality_metric_label": hq_summary.get("quality_metric_label"),
        "quality_threshold": hq_summary.get("quality_threshold"),
    }]
    for key in ["flip_any", "flip_c2i", "flip_i2c", "flip_semantic_wrong"]:
        block = hq_summary.get(key, {}) if isinstance(hq_summary, dict) else {}
        if not isinstance(block, dict):
            continue
        rows.append({
            "intervention": label,
            "metric": key,
            "identified_neurons": hq_summary.get("n_identified_neurons"),
            "high_quality_neurons": hq_summary.get("n_high_quality_neurons"),
            "high_quality_fraction": hq_summary.get("frac_high_quality"),
            "quality_metric_label": hq_summary.get("quality_metric_label"),
            "quality_threshold": hq_summary.get("quality_threshold"),
            "union_all": block.get("union_all"),
            "union_high": block.get("union_high"),
            "sum_all": block.get("sum_all"),
            "sum_high": block.get("sum_high"),
            "denominator": block.get("denominator"),
            "denominator_label": block.get("denominator_label"),
            "all_union_pct": block.get("all_union_pct"),
            "high_union_pct": block.get("high_union_pct"),
            "hq_coverage_share_pct": block.get("hq_coverage_share_pct"),
        })
    return rows


def _plot_title_context(context: Mapping[str, object]) -> str:
    task = str(context.get("task_name", "unknown task"))
    model = str(context.get("model_name", "unknown model"))
    return f"Task: {task} | Model: {model}"


def _save_hq_overview_plot(
    hq_comparison: Mapping[str, object],
    output_dir: Path,
    context: Mapping[str, object],
    baseline_label: str,
    candidate_label: str,
) -> Optional[Path]:
    records = hq_comparison.get("runs", {}) if isinstance(hq_comparison, dict) else {}
    if not isinstance(records, dict) or not records:
        return None
    labels = [baseline_label, candidate_label]
    labels = [lab for lab in labels if lab in records]
    if not labels:
        return None

    def arr(path: Sequence[str]) -> np.ndarray:
        vals = []
        for lab in labels:
            obj = records.get(lab, {})
            for key in path:
                obj = obj.get(key, {}) if isinstance(obj, dict) else {}
            vals.append(_to_float(obj))
        return np.asarray(vals, dtype=float)

    directions = ["flip_any", "flip_c2i", "flip_i2c"]
    direction_labels = ["Any flip", "Correct→incorrect", "Incorrect→correct"]

    with plt.rc_context({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=False)
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.88, wspace=0.28, hspace=0.38)
        # fig.suptitle(f"Mean vs mean-donor intervention shift\n{_plot_title_context(context)}", y=0.98, fontsize=12)
        ax_counts, ax_frac = axes[0]
        ax_prev, ax_cov = axes[1]

        x = np.arange(len(labels))
        w = 0.34
        identified = arr(["n_identified_neurons"])
        hq = arr(["n_high_quality_neurons"])
        b1 = ax_counts.bar(x - w / 2, identified, w, label="Identified")
        b2 = ax_counts.bar(x + w / 2, hq, w, label="High-quality")
        ax_counts.set_xticks(x)
        ax_counts.set_xticklabels(labels)
        ax_counts.set_ylabel("# neurons")
        ax_counts.set_title("Identified vs high-quality neurons")
        ax_counts.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
        ax_counts.legend(frameon=False)
        for bars in [b1, b2]:
            for b in bars:
                h = b.get_height()
                if h == h:
                    ax_counts.annotate(f"{int(round(h))}", (b.get_x() + b.get_width()/2, h), ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")

        frac = arr(["frac_high_quality"]) * 100.0
        bars = ax_frac.bar(x, frac, width=0.58)
        ax_frac.set_xticks(x)
        ax_frac.set_xticklabels(labels)
        ax_frac.set_ylabel("HQ / identified (%)")
        ax_frac.set_title("High-quality share")
        ax_frac.set_ylim(0, max(100.0, float(np.nanmax(frac)) * 1.15 if np.isfinite(frac).any() else 100.0))
        ax_frac.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
        for b in bars:
            h = b.get_height()
            if h == h:
                ax_frac.annotate(f"{h:.1f}%", (b.get_x() + b.get_width()/2, h), ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")

        xd = np.arange(len(directions))
        width = 0.8 / max(1, len(labels))
        prev_vals = []
        for i, lab in enumerate(labels):
            vals = np.asarray([_to_float(records[lab].get(d, {}).get("all_union_pct")) for d in directions], dtype=float)
            prev_vals.extend(vals[np.isfinite(vals)].tolist())
            bars = ax_prev.bar(
                xd + (i - (len(labels)-1)/2)*width,
                vals,
                width=width,
                label=f"{lab}: all",
            )
            for b in bars:
                h = b.get_height()
                if h == h:
                    ax_prev.annotate(
                        f"{h:.1f}%",
                        (b.get_x() + b.get_width() / 2, h),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        xytext=(0, 3),
                        textcoords="offset points",
                        clip_on=False,
                    )
        ax_prev.set_xticks(xd)
        ax_prev.set_xticklabels([s.replace("→", "→\n") for s in direction_labels])
        ax_prev.set_ylabel("Unique flips / eligible points (%)")
        ax_prev.set_title("Flip prevalence, all identified neurons", pad=22)
        if prev_vals:
            ax_prev.set_ylim(0, max(100.0, max(prev_vals) * 1.16))
        ax_prev.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
        ax_prev.legend(
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.10),
            borderaxespad=0.0,
        )

        for i, lab in enumerate(labels):
            vals = np.asarray([_to_float(records[lab].get(d, {}).get("high_union_pct")) for d in directions], dtype=float)
            ax_cov.plot(xd, vals, marker="o", linewidth=1.4, label=f"{lab}: HQ")
            cov_vals = np.asarray([_to_float(records[lab].get(d, {}).get("hq_coverage_share_pct")) for d in directions], dtype=float)
            for xx, yy, cov in zip(xd, vals, cov_vals):
                if yy == yy and cov == cov:
                    ax_cov.annotate(f"cov {cov:.0f}%", (xx, yy), ha="center", va="bottom", fontsize=6.5, xytext=(0, 4), textcoords="offset points")
        ax_cov.set_xticks(xd)
        ax_cov.set_xticklabels([s.replace("→", "→\n") for s in direction_labels])
        ax_cov.set_ylabel("Unique flips / eligible points (%)")
        ax_cov.set_title("Flip prevalence, high-quality neurons")
        ax_cov.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
        ax_cov.legend(frameon=False, ncol=2)

        for ax in axes.ravel():
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        out = output_dir / "intervention_shift_hq_overview.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out


def _load_layer_hq_summary(stats_dir: Path, label: str) -> pd.DataFrame:
    path = stats_dir / "high_quality_neuron_flip_coverage_by_layer.json"
    payload = _read_json(path)
    rows = payload.get("layers", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["intervention"] = label
    for col in [
        "n_neurons_total", "n_neurons_high_quality", "n_rows_with_any_eval_all",
        "flip_any_union_all", "flip_any_union_high", "flip_c2i_union_all", "flip_c2i_union_high",
        "flip_i2c_union_all", "flip_i2c_union_high", "layer_sort",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "layer_key" not in df.columns:
        return pd.DataFrame()
    return df


def _save_hq_by_layer_plot(layer_df: pd.DataFrame, output_dir: Path, context: Mapping[str, object]) -> Optional[Path]:
    if layer_df is None or layer_df.empty:
        return None
    df = layer_df.copy()
    df["layer_sort"] = pd.to_numeric(df.get("layer_sort"), errors="coerce")
    df = df.sort_values(["layer_sort", "layer_key", "intervention"], na_position="last")
    layers = list(dict.fromkeys(df["layer_key"].astype(str).tolist()))
    interventions = list(dict.fromkeys(df["intervention"].astype(str).tolist()))
    if not layers or not interventions:
        return None
    x = np.arange(len(layers))

    with plt.rc_context({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 1, figsize=(max(10.5, 0.62 * len(layers)), 7.0), constrained_layout=False)
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.90, hspace=0.28)
        fig.suptitle(f"Layer-wise HQ comparison\n{_plot_title_context(context)}", y=0.985, fontsize=11)
        ax1, ax2 = axes
        width = 0.8 / max(1, len(interventions) * 2)
        idx = 0
        for intervention in interventions:
            sub = df.loc[df["intervention"] == intervention].set_index("layer_key")
            total = np.asarray([_to_float(sub.loc[lk, "n_neurons_total"]) if lk in sub.index else np.nan for lk in layers])
            high = np.asarray([_to_float(sub.loc[lk, "n_neurons_high_quality"]) if lk in sub.index else np.nan for lk in layers])
            ax1.bar(x + (idx - (len(interventions)*2 - 1)/2)*width, total, width=width, label=f"{intervention}: identified")
            idx += 1
            ax1.bar(x + (idx - (len(interventions)*2 - 1)/2)*width, high, width=width, label=f"{intervention}: HQ")
            idx += 1
        ax1.set_ylabel("# neurons")
        ax1.set_title("Identified and HQ neurons by layer")
        ax1.set_xticks(x)
        ax1.set_xticklabels([lk.replace("_", ".") for lk in layers], rotation=45, ha="right")
        ax1.tick_params(axis="x", labelbottom=False)
        ax1.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
        ax1.legend(frameon=False, ncol=2)

        for intervention in interventions:
            sub = df.loc[df["intervention"] == intervention].set_index("layer_key")
            denom = np.asarray([_to_float(sub.loc[lk, "n_rows_with_any_eval_all"]) if lk in sub.index else np.nan for lk in layers])
            c2i = np.asarray([_pct_num(sub.loc[lk, "flip_c2i_union_all"], denom_i) if lk in sub.index else np.nan for lk, denom_i in zip(layers, denom)])
            i2c = np.asarray([_pct_num(sub.loc[lk, "flip_i2c_union_all"], denom_i) if lk in sub.index else np.nan for lk, denom_i in zip(layers, denom)])
            ax2.plot(x, c2i, marker="o", linewidth=1.2, label=f"{intervention}: C→I all")
            ax2.plot(x, i2c, marker="s", linewidth=1.2, label=f"{intervention}: I→C all")
        ax2.set_ylabel("Unique flips / evaluated rows (%)")
        ax2.set_title("Layer-wise directional flip prevalence")
        ax2.set_xlabel("Layer")
        ax2.set_xticks(x)
        ax2.set_xticklabels([lk.replace("_", ".") for lk in layers], rotation=45, ha="right")
        ax2.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
        ax2.legend(frameon=False, ncol=2)
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        out = output_dir / "intervention_shift_hq_by_layer.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

def _global_comparison(baseline_global: Mapping[str, object], candidate_global: Mapping[str, object]) -> dict:
    out: Dict[str, object] = {}
    for key in GLOBAL_KEYS:
        bv = baseline_global.get(key)
        cv = candidate_global.get(key)
        if bv is None and cv is None:
            continue
        row = {"baseline": _jsonable(bv), "candidate": _jsonable(cv)}
        bf = _to_float(bv)
        cf = _to_float(cv)
        if math.isfinite(bf) and math.isfinite(cf):
            row["delta"] = float(cf - bf)
            if bf != 0:
                row["relative_delta"] = float((cf - bf) / abs(bf))
        out[key] = row
    return out


def _format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _write_markdown_report(path: Path, summary: Mapping[str, object], top_changes: pd.DataFrame) -> None:
    overlap = summary.get("neuron_sets", {})
    paired = summary.get("paired_overlap_stats", {})
    union = summary.get("union_missing_as_zero_stats", {})
    global_cmp = summary.get("global_comparison", {})
    context = summary.get("context", {}) if isinstance(summary.get("context", {}), dict) else {}
    baseline_label = str(context.get("baseline_intervention", "baseline"))
    candidate_label = str(context.get("candidate_intervention", "candidate"))
    hq_cmp = summary.get("high_quality_comparison", {}) if isinstance(summary.get("high_quality_comparison", {}), dict) else {}

    def metric_line(stats_block: Mapping[str, object], metric: str) -> str:
        st = stats_block.get(metric, {}) if isinstance(stats_block, dict) else {}
        if not st or not st.get("available"):
            return f"- `{metric}`: unavailable"
        ci = ""
        if st.get("mean_delta_ci95_low") is not None and st.get("mean_delta_ci95_high") is not None:
            ci = f" [{_format_float(st.get('mean_delta_ci95_low'))}, {_format_float(st.get('mean_delta_ci95_high'))}]"
        return (
            f"- `{metric}`: n={st.get('n')}, baseline_mean={_format_float(st.get('baseline_mean'))}, "
            f"candidate_mean={_format_float(st.get('candidate_mean'))}, mean_delta={_format_float(st.get('mean_delta'))}{ci}, "
            f"median_delta={_format_float(st.get('median_delta'))}, spearman={_format_float(st.get('spearman'))}, "
            f"fraction_higher={_format_float(st.get('fraction_candidate_higher'))}, sign_test_p={_format_float(st.get('sign_test_p_two_sided'))}"
        )

    lines = [
        "# Intervention shift summary",
        "",
        "This compares the two script-7 interventions using `flip_stats_by_neuron.csv`, `flip_stats_global.json`, and, when available, `high_quality_neuron_flip_coverage.json`.",
        "",
        "## Run context",
        "",
        f"- task: {context.get('task_name', 'unknown task')}",
        f"- model: {context.get('model_name', 'unknown model')}",
        f"- baseline intervention: `{baseline_label}`",
        f"- candidate intervention: `{candidate_label}`",
        "",
        "## Intervention definitions",
        "",
        f"- `{baseline_label}`: {_intervention_description(baseline_label)}",
        f"- `{candidate_label}`: {_intervention_description(candidate_label)}",
        "",
        "## Identified-neuron comparison",
        "",
        f"- `{baseline_label}` identified neurons: {overlap.get('n_baseline_neurons', 0)}",
        f"- `{candidate_label}` identified neurons: {overlap.get('n_candidate_neurons', 0)}",
        f"- overlap: {overlap.get('n_overlap', 0)}",
        f"- gained only in `{candidate_label}`: {overlap.get('n_gained', 0)}",
        f"- lost from `{baseline_label}`: {overlap.get('n_lost', 0)}",
        "",
    ]
    runs = hq_cmp.get("runs", {}) if isinstance(hq_cmp, dict) else {}
    if isinstance(runs, dict) and runs:
        lines.extend(["## High-quality neurons and flip prevalence", ""])
        for label in [baseline_label, candidate_label]:
            hq = runs.get(label, {})
            if not isinstance(hq, dict) or not hq:
                continue
            n_id = hq.get("n_identified_neurons")
            n_hq = hq.get("n_high_quality_neurons")
            frac = hq.get("frac_high_quality")
            qlab = hq.get("quality_metric_label", "HQ metric")
            qthr = hq.get("quality_threshold")
            lines.append(f"### `{label}`")
            lines.append(
                f"- HQ neurons: {n_hq} / {n_id} ({_format_float(100.0 * float(frac), 1) if frac is not None and frac == frac else 'NA'}%) "
                f"using {qlab} ≥ {_format_float(qthr, 3)}"
            )
            for key in ["flip_any", "flip_c2i", "flip_i2c"]:
                block = hq.get(key, {}) if isinstance(hq.get(key, {}), dict) else {}
                lines.append(
                    f"- `{key}`: all={_format_float(block.get('all_union_pct'), 2)}%, "
                    f"HQ={_format_float(block.get('high_union_pct'), 2)}%, "
                    f"HQ coverage={_format_float(block.get('hq_coverage_share_pct'), 1)}%"
                )
            lines.append("")
    lines.extend([
        "## Paired overlap statistics",
        "",
    ])
    for metric in ["flip_any_rate", "c2i_rate", "i2c_rate", "flip_semantic_wrong_rate"]:
        lines.append(metric_line(paired, metric))
    lines.extend(["", "## Union statistics, treating missing neurons as zero", ""])
    for metric in ["flip_any_rate", "c2i_rate", "i2c_rate", "flip_semantic_wrong_rate"]:
        lines.append(metric_line(union, metric))
    lines.extend(["", "## Global unique-flip rates", ""])
    for key in ["union_flip_any_unique_rate", "union_c2i_unique_rate", "union_i2c_unique_rate", "union_flip_semantic_wrong_unique_rate"]:
        row = global_cmp.get(key, {}) if isinstance(global_cmp, dict) else {}
        lines.append(
            f"- `{key}`: baseline={_format_float(row.get('baseline'))}, candidate={_format_float(row.get('candidate'))}, delta={_format_float(row.get('delta'))}"
        )
    if top_changes is not None and not top_changes.empty:
        lines.extend(["", "## Largest per-neuron shifts by |delta flip_any_rate|", ""])
        cols = [c for c in ["status", "neuron", "baseline_flip_any_rate", "candidate_flip_any_rate", "delta_flip_any_rate", "baseline_flip_any_count", "candidate_flip_any_count"] if c in top_changes.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in top_changes.head(20).iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(_format_float(val))
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_scatter(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    stats: Mapping[str, object],
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    context: Optional[Mapping[str, object]] = None,
) -> Optional[Path]:
    bcol = f"baseline_{metric}"
    ccol = f"candidate_{metric}"
    if bcol not in df.columns or ccol not in df.columns:
        return None
    sub = df.loc[df["present_baseline"] & df["present_candidate"]].copy()
    x = pd.to_numeric(sub[bcol], errors="coerce")
    y = pd.to_numeric(sub[ccol], errors="coerce")
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if int(mask.sum()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    lo = float(np.nanmin([x.min(), y.min(), 0.0]))
    hi = float(np.nanmax([x.max(), y.max(), 1.0 if metric.endswith("rate") else x.max(), y.max()]))
    pad = (hi - lo) * 0.04 if hi > lo else 0.05
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.scatter(x, y, s=26, alpha=0.75)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linewidth=1, alpha=0.6)
    ax.set_xlabel(f"{baseline_label} {metric}")
    ax.set_ylabel(f"{candidate_label} {metric}")
    title_context = f"\n{_plot_title_context(context or {})}" if context else ""
    ax.set_title(f"{candidate_label} vs {baseline_label}: {metric}{title_context}")
    subtitle = f"n={int(mask.sum())}, Spearman={_format_float(stats.get('spearman'))}, mean delta={_format_float(stats.get('mean_delta'))}"
    ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, va="top", ha="left")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    out = output_dir / f"intervention_shift_scatter_{metric}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _save_delta_hist(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    context: Optional[Mapping[str, object]] = None,
) -> Optional[Path]:
    dcol = f"paired_delta_{metric}"
    if dcol not in df.columns:
        return None
    delta = pd.to_numeric(df[dcol], errors="coerce").dropna().to_numpy(dtype=float)
    if len(delta) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(delta, bins=min(30, max(8, int(np.sqrt(len(delta))))), alpha=0.85)
    ax.axvline(0.0, linewidth=1, alpha=0.65)
    ax.set_xlabel(f"{candidate_label} minus {baseline_label} {metric}")
    ax.set_ylabel("Number of neurons")
    title_context = f"\n{_plot_title_context(context or {})}" if context else ""
    ax.set_title(f"Distribution of per-neuron shifts: {metric}{title_context}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = output_dir / f"intervention_shift_delta_hist_{metric}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _save_top_delta_plot(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    topk: int,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    context: Optional[Mapping[str, object]] = None,
) -> Optional[Path]:
    dcol = f"delta_{metric}"
    if dcol not in df.columns:
        return None
    sub = df.copy()
    sub[dcol] = pd.to_numeric(sub[dcol], errors="coerce")
    sub = sub.loc[sub[dcol].notna()].copy()
    if sub.empty:
        return None
    sub["abs_delta"] = sub[dcol].abs()
    sub = sub.sort_values("abs_delta", ascending=False).head(int(topk)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.30 * len(sub) + 1.2)))
    labels = sub["neuron"].astype(str) + " (" + sub["status"].astype(str) + ")"
    ax.barh(np.arange(len(sub)), sub[dcol].to_numpy(dtype=float), alpha=0.85)
    ax.axvline(0.0, linewidth=1, alpha=0.65)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{candidate_label} minus {baseline_label} {metric}")
    title_context = f"\n{_plot_title_context(context or {})}" if context else ""
    ax.set_title(f"Largest intervention shifts: {metric}{title_context}")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out = output_dir / f"intervention_shift_top_delta_{metric}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _save_global_rate_plot(
    global_cmp: Mapping[str, object],
    output_dir: Path,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    context: Optional[Mapping[str, object]] = None,
) -> Optional[Path]:
    keys = [
        "union_flip_any_unique_rate",
        "union_c2i_unique_rate",
        "union_i2c_unique_rate",
        "union_flip_semantic_wrong_unique_rate",
    ]
    rows = []
    for key in keys:
        row = global_cmp.get(key, {}) if isinstance(global_cmp, dict) else {}
        b = _to_float(row.get("baseline")) if isinstance(row, dict) else float("nan")
        c = _to_float(row.get("candidate")) if isinstance(row, dict) else float("nan")
        if math.isfinite(b) or math.isfinite(c):
            rows.append((key, b, c))
    if not rows:
        return None
    labels = [r[0].replace("union_", "").replace("_unique_rate", "").replace("_", " ") for r in rows]
    baseline_vals = [r[1] if math.isfinite(r[1]) else np.nan for r in rows]
    candidate_vals = [r[2] if math.isfinite(r[2]) else np.nan for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.5, len(rows) * 1.9), 4.8))
    ax.bar(x - width / 2, baseline_vals, width, label=baseline_label, alpha=0.85)
    ax.bar(x + width / 2, candidate_vals, width, label=candidate_label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Unique-row rate")
    title_context = f"\n{_plot_title_context(context or {})}" if context else ""
    ax.set_title(f"Global unique flip rates{title_context}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "intervention_shift_global_unique_rates.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _run_stats_dir_mode(args: argparse.Namespace) -> bool:
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    baseline_flip_path = baseline_dir / "flip_stats_by_neuron.csv"
    candidate_flip_path = candidate_dir / "flip_stats_by_neuron.csv"
    if not baseline_flip_path.exists() and not candidate_flip_path.exists():
        return False

    baseline = _read_flip_stats(baseline_dir)
    candidate = _read_flip_stats(candidate_dir)
    merged = _merge_flip_stats(baseline, candidate)

    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir) if args.output_dir else output_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_csv, index=False)

    top_metric = "delta_flip_any_rate" if "delta_flip_any_rate" in merged.columns else None
    top_changes = pd.DataFrame()
    if top_metric is not None and not merged.empty:
        top_changes = merged.copy()
        top_changes["abs_delta_flip_any_rate"] = pd.to_numeric(top_changes[top_metric], errors="coerce").abs()
        top_changes = top_changes.sort_values("abs_delta_flip_any_rate", ascending=False).head(int(args.topk))
        top_changes.to_csv(output_dir / "intervention_shift_top_changes.csv", index=False)

    baseline_global = _read_json(baseline_dir / "flip_stats_global.json")
    candidate_global = _read_json(candidate_dir / "flip_stats_global.json")
    global_cmp = _global_comparison(baseline_global, candidate_global)

    paired = {
        metric: _paired_metric_stats(merged, metric, use_zero_for_missing=False)
        for metric in RATE_COLS + COUNT_COLS
        if f"baseline_{metric}" in merged.columns or f"candidate_{metric}" in merged.columns
    }
    union = {
        metric: _paired_metric_stats(merged, metric, use_zero_for_missing=True)
        for metric in RATE_COLS + COUNT_COLS
        if f"baseline_{metric}" in merged.columns or f"candidate_{metric}" in merged.columns
    }

    baseline_label = str(getattr(args, "baseline_label", None) or "mean")
    candidate_label = str(getattr(args, "candidate_label", None) or "mean-donor")
    inferred_context = _infer_context_from_nearby_files(
        baseline_dir,
        candidate_dir,
        getattr(args, "task_name", None),
        getattr(args, "model_name", None),
    )
    context = {
        **inferred_context,
        "baseline_intervention": baseline_label,
        "candidate_intervention": candidate_label,
    }

    baseline_hq = _load_hq_summary(baseline_dir, fallback_n_identified=int(len(baseline)))
    candidate_hq = _load_hq_summary(candidate_dir, fallback_n_identified=int(len(candidate)))
    hq_rows = _hq_rows_for_csv(baseline_label, baseline_hq) + _hq_rows_for_csv(candidate_label, candidate_hq)
    hq_csv_path = output_dir / "intervention_shift_hq_summary.csv"
    if hq_rows:
        pd.DataFrame(hq_rows).to_csv(hq_csv_path, index=False)

    layer_df = pd.concat([
        _load_layer_hq_summary(baseline_dir, baseline_label),
        _load_layer_hq_summary(candidate_dir, candidate_label),
    ], ignore_index=True, sort=False)
    layer_csv_path = output_dir / "intervention_shift_hq_by_layer.csv"
    if not layer_df.empty:
        layer_df.to_csv(layer_csv_path, index=False)

    summary = {
        "mode": "script7_stats_dir",
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "context": context,
        "intervention_definitions": {
            baseline_label: _intervention_description(baseline_label),
            candidate_label: _intervention_description(candidate_label),
        },
        "high_quality_comparison": {
            "source": "high_quality_neuron_flip_coverage.json, with rule_metrics_summary.json fallback",
            "runs": {
                baseline_label: baseline_hq,
                candidate_label: candidate_hq,
            },
        },
        "neuron_sets": {
            "n_baseline_neurons": int(len(baseline)),
            "n_candidate_neurons": int(len(candidate)),
            "n_union": int(len(merged)),
            "n_overlap": int((merged["present_baseline"] & merged["present_candidate"]).sum()) if not merged.empty else 0,
            "n_gained": int((~merged["present_baseline"] & merged["present_candidate"]).sum()) if not merged.empty else 0,
            "n_lost": int((merged["present_baseline"] & ~merged["present_candidate"]).sum()) if not merged.empty else 0,
        },
        "global_comparison": global_cmp,
        "paired_overlap_stats": paired,
        "union_missing_as_zero_stats": union,
        "notes": {
            "primary_interpretation": "Use paired_overlap_stats to isolate intervention changes for neurons present in both runs; use union_missing_as_zero_stats to include gained/lost neurons.",
            "delta_sign": "Positive deltas mean the candidate intervention produced a higher flip/effect statistic than the baseline.",
            "reason_current_old_script_was_empty": "run_resample_recheck.sh passes script-7 stats directories, while the old summarizer only parsed script-6 per-rule JSON files containing ablations.",
        },
    }
    summary_path = output_dir / "intervention_shift_report.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_report(output_dir / "intervention_shift_report.md", summary, top_changes)

    plot_paths: List[str] = []
    if not args.no_plots and not merged.empty:
        for metric in ["flip_any_rate", "c2i_rate", "i2c_rate", "flip_semantic_wrong_rate"]:
            st = paired.get(metric, {})
            for path in [
                _save_scatter(merged, metric, output_dir, st, baseline_label, candidate_label, context),
                _save_delta_hist(merged, metric, output_dir, baseline_label, candidate_label, context),
            ]:
                if path is not None:
                    plot_paths.append(str(path))
        path = _save_top_delta_plot(merged, "flip_any_rate", output_dir, topk=int(args.topk), baseline_label=baseline_label, candidate_label=candidate_label, context=context)
        if path is not None:
            plot_paths.append(str(path))
        path = _save_global_rate_plot(global_cmp, output_dir, baseline_label=baseline_label, candidate_label=candidate_label, context=context)
        if path is not None:
            plot_paths.append(str(path))
        path = _save_hq_overview_plot(summary["high_quality_comparison"], output_dir, context, baseline_label, candidate_label)
        if path is not None:
            plot_paths.append(str(path))
        path = _save_hq_by_layer_plot(layer_df, output_dir, context)
        if path is not None:
            plot_paths.append(str(path))
    summary["files"] = {
        "by_neuron_csv": str(output_csv),
        "top_changes_csv": str(output_dir / "intervention_shift_top_changes.csv") if not top_changes.empty else None,
        "hq_summary_csv": str(hq_csv_path) if hq_rows else None,
        "hq_by_layer_csv": str(layer_csv_path) if not layer_df.empty else None,
        "report_json": str(summary_path),
        "report_md": str(output_dir / "intervention_shift_report.md"),
        "plots_pdf": plot_paths,
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    ns = summary["neuron_sets"]
    print("[InterventionShift] script-7 stats comparison")
    print(f"  {baseline_label}_neurons={ns['n_baseline_neurons']} {candidate_label}_neurons={ns['n_candidate_neurons']} overlap={ns['n_overlap']} gained={ns['n_gained']} lost={ns['n_lost']}")
    for lab, hq in [(baseline_label, baseline_hq), (candidate_label, candidate_hq)]:
        frac_hq = _to_float(hq.get("frac_high_quality"))
        print(
            f"  {lab} HQ neurons={hq.get('n_high_quality_neurons')} / {hq.get('n_identified_neurons')} "
            f"({_format_float(100.0 * frac_hq, 1) if math.isfinite(frac_hq) else 'NA'}%)"
        )
    for metric in ["flip_any_rate", "c2i_rate", "i2c_rate", "flip_semantic_wrong_rate"]:
        st = paired.get(metric, {})
        if st.get("available"):
            print(
                f"  paired {metric}: n={st['n']} mean_delta={_format_float(st['mean_delta'])} "
                f"median_delta={_format_float(st['median_delta'])} spearman={_format_float(st['spearman'])} "
                f"sign_p={_format_float(st['sign_test_p_two_sided'])}"
            )
    g = global_cmp.get("union_flip_any_unique_rate", {})
    if isinstance(g, dict) and (g.get("baseline") is not None or g.get("candidate") is not None):
        print(
            "  global union_flip_any_unique_rate: "
            f"baseline={_format_float(g.get('baseline'))} candidate={_format_float(g.get('candidate'))} delta={_format_float(g.get('delta'))}"
        )
    print(f"[InterventionShift] wrote results in {output_dir}")
    print(f"[InterventionShift] wrote {output_csv}")
    print(f"[InterventionShift] wrote {summary_path}")
    if plot_paths:
        print(f"[InterventionShift] wrote {len(plot_paths)} PDF plots in {output_dir}")
    return True


# ------------------------- legacy script-6 JSON fallback -------------------------

def iter_rule_jsons(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        yield path


def flatten_ablation_records(root: Path) -> Dict[Tuple[int, str, int], dict]:
    out = {}
    for path in iter_rule_jsons(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        circuit_id = payload.get("circuit_id")
        if circuit_id is None:
            continue
        for rec in payload.get("ablations", []) or []:
            layer_map = rec.get("layers_neurons_dict") or {}
            if len(layer_map) != 1:
                continue
            layer_label, neuron_ids = next(iter(layer_map.items()))
            if len(neuron_ids) != 1:
                continue
            neuron_id = int(neuron_ids[0])
            key = (int(circuit_id), str(layer_label), neuron_id)
            out[key] = {
                "circuit_id": int(circuit_id),
                "layer_label": str(layer_label),
                "neuron_id": neuron_id,
                "max_effect": _to_float(rec.get("max_effect")),
                "accuracy_gap": _to_float(rec.get("accuracy_gap")),
            }
    return out


def _run_legacy_json_mode(args: argparse.Namespace) -> None:
    a = flatten_ablation_records(Path(args.baseline_dir))
    b = flatten_ablation_records(Path(args.candidate_dir))
    keys = sorted(set(a) & set(b))
    rows = []
    for key in keys:
        ra = a[key]
        rb = b[key]
        rows.append({
            **{k: ra[k] for k in ("circuit_id", "layer_label", "neuron_id")},
            "baseline_max_effect": ra["max_effect"],
            "candidate_max_effect": rb["max_effect"],
            "baseline_accuracy_gap": ra["accuracy_gap"],
            "candidate_accuracy_gap": rb["accuracy_gap"],
            "delta_max_effect": rb["max_effect"] - ra["max_effect"],
            "delta_accuracy_gap": rb["accuracy_gap"] - ra["accuracy_gap"],
        })

    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir) if args.output_dir else output_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else [
                "circuit_id", "layer_label", "neuron_id", "baseline_max_effect", "candidate_max_effect",
                "baseline_accuracy_gap", "candidate_accuracy_gap", "delta_max_effect", "delta_accuracy_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print("No overlapping singleton ablation records found. If you passed script-7 stats directories, make sure they contain flip_stats_by_neuron.csv.")
        return

    df = pd.DataFrame(rows)
    print("[InterventionShift] legacy script-6 ablation comparison")
    for metric in ["max_effect", "accuracy_gap"]:
        st = _paired_metric_stats(
            df.assign(present_baseline=True, present_candidate=True),
            metric,
            use_zero_for_missing=False,
        )
        print(
            f"  paired {metric}: n={st.get('n')} mean_delta={_format_float(st.get('mean_delta'))} "
            f"median_delta={_format_float(st.get('median_delta'))} spearman={_format_float(st.get('spearman'))}"
        )
    print(f"[InterventionShift] wrote results in {output_dir}")
    print(f"[InterventionShift] wrote {output_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare baseline vs candidate intervention outputs. Primarily expects the script-7 "
            "stats directories produced by 7_refine_neuron_anchored_rules.py, but can fall back "
            "to legacy script-6 per-rule JSON ablation directories."
        )
    )
    ap.add_argument("--baseline_dir", required=True, help="Baseline stats dir, usually .../neuron_flip_rules/stats/<stats_dirname>")
    ap.add_argument("--candidate_dir", required=True, help="Candidate stats dir, usually .../neuron_flip_rules_<intervention>/stats/<stats_dirname>")
    ap.add_argument("--output_csv", required=True, help="Merged per-neuron comparison CSV to write")
    ap.add_argument("--output_dir", default=None, help="Directory for JSON/Markdown reports and PDF plots. Defaults to output_csv parent.")
    ap.add_argument("--topk", type=int, default=30, help="Number of largest per-neuron shifts to export/plot")
    ap.add_argument("--baseline_label", default="mean", help="Human-readable baseline intervention label shown in reports/figures")
    ap.add_argument("--candidate_label", default="mean-donor", help="Human-readable candidate intervention label shown in reports/figures")
    ap.add_argument("--task_name", default=None, help="Task name/module to show in reports/figures")
    ap.add_argument("--model_name", default=None, help="Model name/id to show in reports/figures")
    ap.add_argument("--no_plots", action="store_true", help="Write only CSV/JSON/Markdown, no matplotlib figures")
    args = ap.parse_args()

    if _run_stats_dir_mode(args):
        return
    _run_legacy_json_mode(args)


if __name__ == "__main__":
    main()
