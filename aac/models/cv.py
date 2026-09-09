"""Fold generation and cross-validation with OOF assembly (README section 10).

One fold assignment per run, generated once from the seed and saved to ``folds.npy``. Every
branch and every model family reuses it, so scores are comparable and OOF matrices can be
blended. Encoders that touch the target are fitted inside the fold loop only.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from aac.exec.artifacts import atomic_write_bytes
from aac.models.metrics import MetricSpec, score
from aac.models.registry import build_model, supports_early_stopping

ES_FRACTION = 0.1


def make_folds(y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Stratified fold id per row, int8. Deterministic for (y, n_folds, seed)."""
    folds = np.empty(len(y), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[valid_idx] = k
    return folds


def load_or_create_folds(path: Path, y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """The run's single fold assignment: created once, then always reloaded from disk."""
    if path.exists():
        folds = np.load(path)
        if len(folds) != len(y) or int(folds.max()) + 1 != n_folds:
            raise ValueError(
                f"{path} does not match this run's data ({len(y)} rows, {n_folds} folds)"
            )
        return folds
    folds = make_folds(y, n_folds, seed)
    buffer = io.BytesIO()
    np.save(buffer, folds)
    atomic_write_bytes(path, buffer.getvalue())
    return folds


@dataclass
class CVResult:
    family: str
    oof: np.ndarray  # (n,) positive-class probability for binary, (n, k) otherwise
    test_pred: np.ndarray  # same layout, averaged over folds
    fold_scores: list[float]
    oof_score: float
    duration: float
    importances: dict[str, float] | None = None
    best_iterations: list[int | None] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        return float(np.std(self.fold_scores))

    def summary(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "fold_scores": [float(s) for s in self.fold_scores],
            "cv_mean": self.mean,
            "cv_std": self.std,
            "oof_score": self.oof_score,
            "duration": self.duration,
            "best_iterations": self.best_iterations,
            "importances": self.importances,
        }


def _squeeze(proba: np.ndarray, n_classes: int) -> np.ndarray:
    return proba[:, 1] if n_classes == 2 else proba


def _target_encode(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    others: list[pd.DataFrame],
    columns: list[str],
    n_classes: int,
    seed: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Per-fold target encoding: fitted on the training part only, applied to everything else."""
    from sklearn.preprocessing import TargetEncoder

    if not columns:
        return X_tr, others
    encoder = TargetEncoder(
        target_type="binary" if n_classes == 2 else "multiclass",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    )
    raw_tr = X_tr[columns].astype(str)
    encoded_tr = encoder.fit_transform(raw_tr, y_tr)
    names = (
        [f"{c}__te" for c in columns]
        if n_classes == 2
        else [f"{c}__te{k}" for c in columns for k in range(n_classes)]
    )

    def attach(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        # A category column is replaced by its encoding; a numeric one (an integer treated as
        # a level set) keeps its order information alongside the encoding.
        replaced = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
        out = frame.drop(columns=replaced)
        for i, name in enumerate(names):
            out[name] = values[:, i]
        return out

    X_tr = attach(X_tr, encoded_tr)
    others = [attach(f, encoder.transform(f[columns].astype(str))) for f in others]
    return X_tr, others


def cross_validate(
    family: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    folds: np.ndarray,
    *,
    metric: MetricSpec,
    n_classes: int,
    categorical: list[str],
    seed: int,
    n_jobs: int,
    early_stopping: bool = True,
    target_encode: list[str] | None = None,
) -> CVResult:
    started = time.monotonic()
    n_folds = int(folds.max()) + 1
    oof = np.zeros((len(X), n_classes))
    test_sum = np.zeros((len(X_test), n_classes))
    fold_scores: list[float] = []
    best_iters: list[int | None] = []
    importance_sum: dict[str, float] | None = None
    te_cols = [c for c in (target_encode or []) if c in X.columns]
    use_es = early_stopping and supports_early_stopping(family)

    for k in range(n_folds):
        train_mask = folds != k
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[~train_mask], y[~train_mask]
        X_tr, (X_va_enc, X_te_enc) = _target_encode(
            X_tr, y_tr, [X_va, X_test], te_cols, n_classes, seed
        )
        cat_cols = [c for c in categorical if c in X_tr.columns]
        X_es = y_es = None
        if use_es:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_tr, y_tr, test_size=ES_FRACTION, random_state=seed + k, stratify=y_tr
            )
        model = build_model(
            family, params, n_classes=n_classes, seed=seed, n_jobs=n_jobs, categorical=cat_cols
        )
        model.fit(X_tr, y_tr, X_es, y_es)
        proba_va = model.predict_proba(X_va_enc)
        oof[~train_mask] = proba_va
        test_sum += model.predict_proba(X_te_enc)
        fold_scores.append(score(metric, y_va, _squeeze(proba_va, n_classes)))
        best_iters.append(model.best_iteration)
        imp = model.importances()
        if imp is not None:
            importance_sum = importance_sum or dict.fromkeys(imp, 0.0)
            for name, value in imp.items():
                importance_sum[name] = importance_sum.get(name, 0.0) + value / n_folds

    oof_final = _squeeze(oof, n_classes)
    return CVResult(
        family=family,
        oof=oof_final,
        test_pred=_squeeze(test_sum / n_folds, n_classes),
        fold_scores=fold_scores,
        oof_score=score(metric, y, oof_final),
        duration=time.monotonic() - started,
        importances=importance_sum,
        best_iterations=best_iters,
    )
