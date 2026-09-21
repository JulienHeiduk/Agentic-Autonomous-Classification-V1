"""Analyst: deterministic diagnostics on an experiment (README section 17.2, was Critic).

No LLM. Fold spread, calibration, the worst categorical slices, and how much the experiment
diverges from the current blend. The text goes into the Researcher's next prompt; the numbers
go to ``analysis.json`` next to the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aac.exec.sandbox import ExperimentResult
from aac.models.metrics import MetricSpec, score

MIN_SLICE_ROWS = 100
MAX_SLICES = 3
MAX_CATEGORIES_PER_COLUMN = 30


@dataclass
class Analysis:
    text: str
    data: dict[str, Any] = field(default_factory=dict)


def _positive(oof: np.ndarray) -> np.ndarray:
    return oof if oof.ndim == 1 else oof[:, 1] if oof.shape[1] == 2 else oof.max(axis=1)


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 5) -> dict[str, Any]:
    """Binary calibration: predicted mean vs observed rate overall and per quantile bin."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": i,
                "n": int(mask.sum()),
                "predicted": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
            }
        )
    return {
        "mean_predicted": float(p.mean()),
        "positive_rate": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "bins": rows,
    }


def worst_slices(
    y: np.ndarray, oof: np.ndarray, frame: pd.DataFrame | None, metric: MetricSpec
) -> list[dict[str, Any]]:
    """The categorical values on which the experiment scores worst (enough rows to matter)."""
    if frame is None or frame.empty:
        return []
    y = np.asarray(y)
    results: list[dict[str, Any]] = []
    for col in frame.columns:
        values = frame[col].astype("object").where(frame[col].notna(), "<missing>").astype(str)
        counts = values.value_counts()
        if len(counts) > MAX_CATEGORIES_PER_COLUMN:
            counts = counts.head(MAX_CATEGORIES_PER_COLUMN)
        for value, n in counts.items():
            if n < MIN_SLICE_ROWS:
                continue
            mask = (values == value).to_numpy()
            if len(np.unique(y[mask])) < 2:
                continue
            try:
                s = score(metric, y[mask], oof[mask])
            except ValueError:
                continue
            results.append(
                {"column": str(col), "value": str(value)[:40], "n": int(n), "score": float(s)}
            )
    results.sort(key=lambda r: r["score"], reverse=not metric.greater_is_better)
    return results[:MAX_SLICES]


def analyse(
    result: ExperimentResult,
    y: np.ndarray,
    folds: np.ndarray,
    metric: MetricSpec,
    *,
    slices: pd.DataFrame | None = None,
    blend_oof: np.ndarray | None = None,
    blend_score: float | None = None,
    leader_score: float | None = None,
) -> Analysis:
    if not result.ok or result.oof is None:
        return Analysis("Analyst: the experiment did not run, nothing to analyse.")
    y = np.asarray(y)
    oof = np.asarray(result.oof)
    fold_scores = list(result.fold_scores)
    data: dict[str, Any] = {
        "oof_score": result.oof_score,
        "fold_scores": fold_scores,
        "fold_std": float(np.std(fold_scores)) if fold_scores else None,
        "fold_seconds": result.fold_seconds,
    }
    lines = [
        f"Analyst notes on your last experiment (OOF {metric.display} {result.oof_score:.5f}):",
        f"- folds: {', '.join(f'{s:.5f}' for s in fold_scores)} (std {data['fold_std']:.5f}); "
        f"seconds per fold {', '.join(f'{t:.0f}' for t in result.fold_seconds)}",
    ]
    if oof.ndim == 1 or oof.shape[1] == 2:
        cal = calibration(y, _positive(oof))
        data["calibration"] = cal
        gap = cal["mean_predicted"] - cal["positive_rate"]
        worst_bin = max(
            cal["bins"], key=lambda b: abs(b["predicted"] - b["observed"]), default=None
        )
        note = ""
        if worst_bin is not None:
            note = (
                f"; worst bin predicted {worst_bin['predicted']:.3f} vs observed "
                f"{worst_bin['observed']:.3f}"
            )
        lines.append(
            f"- calibration: mean prediction {cal['mean_predicted']:.4f} vs positive rate "
            f"{cal['positive_rate']:.4f} (gap {gap:+.4f}), Brier {cal['brier']:.4f}{note}"
        )
    slices_out = worst_slices(y, oof, slices, metric)
    data["worst_slices"] = slices_out
    if slices_out:
        lines.append(
            "- weakest slices: "
            + "; ".join(
                f"{s['column']}={s['value']} (n={s['n']}) {s['score']:.4f}" for s in slices_out
            )
        )
    if blend_oof is not None and blend_oof.shape == oof.shape:
        a = _positive(oof) if oof.ndim > 1 else oof
        b = _positive(blend_oof) if blend_oof.ndim > 1 else blend_oof
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
        data["corr_with_blend"] = corr
        verdict = (
            "adds little diversity"
            if corr > 0.99
            else "adds diversity"
            if corr < 0.95
            else "moderately diverse"
        )
        blend_text = f" (blend OOF {blend_score:.5f})" if blend_score is not None else ""
        lines.append(f"- correlation with the current blend {corr:.4f}: {verdict}{blend_text}")
    if leader_score is not None:
        delta = metric.improvement(result.oof_score, leader_score)
        data["vs_leader"] = delta
        lines.append(
            f"- versus the best single model so far: {delta:+.5f} "
            f"({'ahead' if delta > 0 else 'behind'})"
        )
    return Analysis("\n".join(lines), data)
