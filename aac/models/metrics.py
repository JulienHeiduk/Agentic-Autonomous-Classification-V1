"""Competition metric resolution and scoring.

The metric name comes from the Kaggle API (README section 6). It is mapped here to one of the
implemented metrics, and anything unrecognised is refused rather than substituted. The only
override is the explicit ``competition.metric`` config key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class MetricSpec:
    key: str
    display: str
    greater_is_better: bool
    needs_proba: bool

    def better(self, a: float, b: float) -> bool:
        """True when score ``a`` beats score ``b``."""
        return a > b if self.greater_is_better else a < b

    def improvement(self, new: float, old: float) -> float:
        """Signed improvement of ``new`` over ``old``; positive is better."""
        return new - old if self.greater_is_better else old - new


METRICS: dict[str, MetricSpec] = {
    "auc": MetricSpec("auc", "ROC AUC", greater_is_better=True, needs_proba=True),
    "accuracy": MetricSpec("accuracy", "Accuracy", greater_is_better=True, needs_proba=False),
    "logloss": MetricSpec("logloss", "Log loss", greater_is_better=False, needs_proba=True),
    "f1": MetricSpec("f1", "F1", greater_is_better=True, needs_proba=False),
    "macro_f1": MetricSpec("macro_f1", "Macro F1", greater_is_better=True, needs_proba=False),
}

# Kaggle display names, normalised to lowercase alphanumerics. Only names whose meaning is
# unambiguous are listed; add to this table deliberately, never by guessing.
_KAGGLE_NAMES: dict[str, str] = {
    "rocaucscore": "auc",
    "rocauc": "auc",
    "auc": "auc",
    "areaunderreceiveroperatingcharacteristiccurve": "auc",
    "areaundercurve": "auc",
    "accuracy": "accuracy",
    "accuracyscore": "accuracy",
    "categorizationaccuracy": "accuracy",
    "logloss": "logloss",
    "logarithmicloss": "logloss",
    "binarylogloss": "logloss",
    "multiclassloss": "logloss",
    "f1score": "f1",
    "f1": "f1",
    "macrof1": "macro_f1",
    "macrof1score": "macro_f1",
    "f1macro": "macro_f1",
}


class UnsupportedMetricError(ValueError):
    """Kaggle reported a metric this framework does not implement."""


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_metric(kaggle_name: str | None, override: str | None = None) -> MetricSpec:
    """Map Kaggle's metric display name to a MetricSpec, or refuse.

    ``override`` is the explicit ``competition.metric`` config key and wins when present.
    """
    if override is not None:
        if override not in METRICS:
            raise UnsupportedMetricError(
                f"competition.metric={override!r} is not one of {sorted(METRICS)}"
            )
        return METRICS[override]
    if not kaggle_name:
        raise UnsupportedMetricError(
            "Kaggle reported no evaluation metric; set competition.metric explicitly"
        )
    key = _KAGGLE_NAMES.get(normalise(kaggle_name))
    if key is None:
        raise UnsupportedMetricError(
            f"Kaggle metric {kaggle_name!r} is not implemented (known: {sorted(METRICS)}). "
            "Refusing to guess; set competition.metric explicitly if one of them is right."
        )
    return METRICS[key]


def _labels_from(y_score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        return y_score.argmax(axis=1)
    return (y_score >= threshold).astype(int)


def score(spec: MetricSpec, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Score predictions. ``y_score`` is always probabilities: shape (n,) for binary, (n, k)
    for multiclass. Label metrics threshold at 0.5 or take the argmax."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if spec.key == "auc":
        if y_score.ndim == 2:
            return float(skm.roc_auc_score(y_true, y_score, multi_class="ovr"))
        return float(skm.roc_auc_score(y_true, y_score))
    if spec.key == "logloss":
        return float(skm.log_loss(y_true, y_score))
    labels = _labels_from(y_score)
    if spec.key == "accuracy":
        return float(skm.accuracy_score(y_true, labels))
    if spec.key == "f1":
        if len(np.unique(y_true)) > 2:
            raise UnsupportedMetricError("f1 is binary only; use macro_f1 for multiclass")
        return float(skm.f1_score(y_true, labels))
    if spec.key == "macro_f1":
        return float(skm.f1_score(y_true, labels, average="macro"))
    raise UnsupportedMetricError(spec.key)
