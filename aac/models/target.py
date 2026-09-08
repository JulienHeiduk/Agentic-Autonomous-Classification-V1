"""Target encoding for classification: class order, the positive label, and 0..k-1 codes.

Used by the baseline, the Scout, and the Trainer so every component agrees on which label
is "positive" for binary metrics. Binary: the positive class is the numeric maximum, or a
yes/true-like string, or failing that the alphabetically last label (scikit-learn's
LabelBinarizer convention). Multiclass: classes in sorted order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POSITIVE_HINTS = frozenset({"yes", "y", "true", "t", "1", "positive", "pos", "success"})
NEGATIVE_HINTS = frozenset({"no", "n", "false", "f", "0", "negative", "neg", "failure"})


class TargetError(ValueError):
    """The target column cannot be used for classification."""


@dataclass(frozen=True)
class TargetEncoding:
    classes: tuple[object, ...]  # index order; for binary, (negative, positive)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    @property
    def positive_label(self) -> object | None:
        return self.classes[1] if self.is_binary else None

    def encode(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """Labels to integer codes. Unknown labels or NaN are an error, never silently mapped."""
        series = pd.Series(np.asarray(y)) if not isinstance(y, pd.Series) else y
        lookup = {label: i for i, label in enumerate(self.classes)}
        codes = series.map(lookup)
        if codes.isna().any():
            bad = sorted({v for v in series[codes.isna()].unique()}, key=str)[:10]
            raise TargetError(f"labels not in {self.classes}: {bad}")
        return codes.to_numpy(dtype=np.int64)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes, dtype=object)[np.asarray(codes, dtype=np.int64)]


def _is_numeric_like(labels: list[object]) -> bool:
    return all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) for v in labels)


def choose_positive(labels: list[object]) -> object:
    if len(labels) != 2:
        raise TargetError(f"choose_positive needs exactly two labels, got {labels}")
    if _is_numeric_like(labels):
        return max(labels)
    lowered = {str(v).strip().lower(): v for v in labels}
    for hint in POSITIVE_HINTS:
        if hint in lowered:
            return lowered[hint]
    for hint in NEGATIVE_HINTS:
        if hint in lowered:
            return next(v for k, v in lowered.items() if k != hint)
    return sorted(labels, key=str)[-1]


def infer_target_encoding(y: pd.Series) -> TargetEncoding:
    if y.isna().any():
        raise TargetError(f"target has {int(y.isna().sum())} missing values")
    labels = list(pd.unique(y))
    if len(labels) < 2:
        raise TargetError(f"target has a single class: {labels}")
    if len(labels) == 2:
        positive = choose_positive(labels)
        negative = next(v for v in labels if v != positive)
        return TargetEncoding((negative, positive))
    return TargetEncoding(tuple(sorted(labels, key=str)))
