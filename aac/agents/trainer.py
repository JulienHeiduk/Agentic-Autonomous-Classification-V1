"""Trainer: deterministic, no LLM (README section 3).

Given a plan and the frames (already carrying any engineered features), runs the shared
stratified folds for every model family in the plan, writes OOF and test predictions plus
``metrics.json`` to the branch directory, and reports the best family by OOF score. It is
the only component that touches the target.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aac.agents.scout import Profile
from aac.exec.artifacts import atomic_write_bytes, atomic_write_json
from aac.models.cv import CVResult, cross_validate
from aac.models.metrics import MetricSpec, score
from aac.models.prepare import Matrix, add_flag, prepare_matrix
from aac.plan import Plan

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    plan_hash: str
    metric: str
    features: list[str]
    categorical: list[str]
    results: dict[str, CVResult]
    best_family: str
    blend_oof_score: float
    duration: float
    errors: dict[str, str] = field(default_factory=dict)
    n_extra: int = 0  # extra training rows appended to every fold

    @property
    def best(self) -> CVResult:
        return self.results[self.best_family]

    def metrics(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "metric": self.metric,
            "n_extra": self.n_extra,
            "n_features": len(self.features),
            "features": self.features,
            "categorical": self.categorical,
            "models": {f: r.summary() for f, r in self.results.items()},
            "errors": self.errors,
            "best_family": self.best_family,
            "best_oof_score": self.best.oof_score,
            "blend_mean_oof_score": self.blend_oof_score,
            "duration": self.duration,
        }


def select_features(
    plan: Plan, profile: Profile, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Feature and categorical column lists for this plan.

    Profile-usable columns minus the plan's drops, plus any column present in both frames
    that the profile has never seen (engineered features), typed by dtype.
    """
    drops = set(plan.drop_columns) | {profile.target.name, profile.id_col}
    known = {c.name: c for c in profile.columns}
    features: list[str] = []
    categorical: list[str] = []
    explicit_cats = set(plan.categorical_columns or [])
    for col in train.columns:
        if col in drops or col not in test.columns:
            continue
        info = known.get(col)
        if info is not None:
            if not info.usable and col not in explicit_cats:
                continue
            is_cat = col in explicit_cats or info.kind in ("categorical", "boolean")
        else:
            dtype = train[col].dtype
            is_cat = col in explicit_cats or not pd.api.types.is_numeric_dtype(dtype)
        features.append(col)
        if is_cat:
            categorical.append(col)
    if plan.categorical_columns is not None:
        categorical = [c for c in features if c in explicit_cats]
    if not features:
        raise ValueError("no usable feature columns after applying the plan")
    return features, categorical


def _save(path: Path, array: np.ndarray) -> None:
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array))
    atomic_write_bytes(path, buffer.getvalue())


def train_plan(
    plan: Plan,
    profile: Profile,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    metric: MetricSpec,
    seed: int,
    n_jobs: int,
    branch_dir: Path,
    max_trees: int | None = None,
    families: list[str] | None = None,
    extra: pd.DataFrame | None = None,
    extra_y: np.ndarray | None = None,
    extra_flag: str | None = None,
) -> TrainResult:
    """Train every family of the plan (or only ``families``) on the shared folds. ``extra``
    rows (with ``extra_y``) join the training part of every fold, never validation."""
    started = time.monotonic()
    n_classes = len(profile.target.classes)
    features, categorical = select_features(plan, profile, train, test)
    matrix: Matrix = prepare_matrix(train, test, features, categorical, extra=extra)
    matrix = add_flag(matrix, extra_flag)
    features = matrix.features
    extra_pair = None
    if matrix.X_extra is not None and extra_y is not None and len(matrix.X_extra):
        extra_pair = (matrix.X_extra, np.asarray(extra_y))
    models = [m for m in plan.models if families is None or m.family in families]
    if not models:
        raise ValueError(f"plan {plan.name!r} has none of the families {families}")
    log.info(
        "training plan %s (seed %d): %d features (%d categorical, %d target-encoded), "
        "%d rows + %d extra, %d families",
        plan.name,
        seed,
        len(features),
        len(categorical),
        len(plan.target_encode),
        len(train),
        matrix.n_extra,
        len(models),
    )
    results: dict[str, CVResult] = {}
    errors: dict[str, str] = {}
    for model in models:
        params = plan.resolved_params(model, max_trees)
        try:
            result = cross_validate(
                model.family,
                params,
                matrix.X_train,
                y,
                matrix.X_test,
                folds,
                metric=metric,
                n_classes=n_classes,
                categorical=matrix.categorical,
                seed=seed,
                n_jobs=n_jobs,
                early_stopping=model.early_stopping,
                target_encode=plan.target_encode,
                extra=extra_pair,
            )
        except Exception as exc:  # noqa: BLE001 - one family failing must not sink the branch
            log.exception("family %s failed", model.family)
            errors[model.family] = f"{exc.__class__.__name__}: {exc}"[:500]
            continue
        results[model.family] = result
        _save(branch_dir / f"oof_{model.family}.npy", result.oof)
        _save(branch_dir / f"test_{model.family}.npy", result.test_pred)
        log.info(
            "  %-10s %s %.5f (folds %.5f +/- %.5f) in %.0fs",
            model.family,
            metric.key,
            result.oof_score,
            result.mean,
            result.std,
            result.duration,
        )
    if not results:
        raise RuntimeError(f"every model family failed: {errors}")
    best_family = max(
        results, key=lambda f: results[f].oof_score * (1 if metric.greater_is_better else -1)
    )
    blend = np.mean([r.oof for r in results.values()], axis=0)
    blend_score = score(metric, y, blend)
    out = TrainResult(
        plan_hash=plan.hash(),
        metric=metric.key,
        features=features,
        categorical=categorical,
        results=results,
        best_family=best_family,
        blend_oof_score=blend_score,
        duration=time.monotonic() - started,
        errors=errors,
        n_extra=matrix.n_extra,
    )
    _save(branch_dir / "oof.npy", results[best_family].oof)
    _save(branch_dir / "test_pred.npy", results[best_family].test_pred)
    atomic_write_json(branch_dir / "metrics.json", out.metrics())
    return out
