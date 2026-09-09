"""Feature matrix preparation: target-free, fold-agnostic, identical for train and test.

Numeric columns become float64. Categorical and boolean columns become pandas ``category`` with
a vocabulary taken from the union of train and test values (unsupervised, so not a leak), which
keeps the codes consistent across both frames and across model families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MISSING_CATEGORY = "__missing__"


@dataclass
class Matrix:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    features: list[str]
    categorical: list[str]
    X_extra: pd.DataFrame | None = None  # extra training rows, same columns and dtypes

    @property
    def numeric(self) -> list[str]:
        return [f for f in self.features if f not in self.categorical]

    @property
    def n_extra(self) -> int:
        return 0 if self.X_extra is None else len(self.X_extra)


def add_flag(matrix: Matrix, name: str | None) -> Matrix:
    """Append a feature that is 1 on the extra rows and 0 on train and test, so a model can
    treat the two sources differently. A no-op without extra rows or a name."""
    if not name or matrix.X_extra is None:
        return matrix
    if name in matrix.features:
        raise KeyError(f"flag column {name!r} collides with a feature")
    X_train = matrix.X_train.copy()
    X_test = matrix.X_test.copy()
    X_extra = matrix.X_extra.copy()
    X_train[name] = np.float64(0.0)
    X_test[name] = np.float64(0.0)
    X_extra[name] = np.float64(1.0)
    return Matrix(X_train, X_test, [*matrix.features, name], matrix.categorical, X_extra)


def _as_category_strings(s: pd.Series) -> pd.Series:
    out = s.astype("object")
    mask = out.isna()
    out = out.astype(str)
    out[mask] = MISSING_CATEGORY
    return out


def _numeric(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce").astype(np.float64)
    return values.replace([np.inf, -np.inf], np.nan)


def prepare_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    extra: pd.DataFrame | None = None,
) -> Matrix:
    """``extra`` rows (appended to training folds later) share the category vocabulary."""
    frames = [train, test] + ([extra] if extra is not None else [])
    missing = [f for f in features if any(f not in fr.columns for fr in frames)]
    if missing:
        raise KeyError(f"features absent from train, test, or extra rows: {missing}")
    cat_set = [c for c in categorical if c in features]
    outputs = [pd.DataFrame(index=fr.index) for fr in frames]
    for col in features:
        if col in cat_set:
            strings = [_as_category_strings(fr[col]) for fr in frames]
            vocab = sorted(set().union(*(set(s.unique()) for s in strings)))
            dtype = pd.CategoricalDtype(categories=vocab)
            for out, s in zip(outputs, strings, strict=True):
                out[col] = s.astype(dtype)
        else:
            # Engineered ratios divide by zero now and then; every family handles NaN, none
            # handles inf.
            for out, fr in zip(outputs, frames, strict=True):
                out[col] = _numeric(fr[col])
    X_extra = outputs[2] if extra is not None else None
    return Matrix(outputs[0], outputs[1], list(features), cat_set, X_extra)
