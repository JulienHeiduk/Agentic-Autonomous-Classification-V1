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

    @property
    def numeric(self) -> list[str]:
        return [f for f in self.features if f not in self.categorical]


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
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], categorical: list[str]
) -> Matrix:
    missing = [f for f in features if f not in train.columns or f not in test.columns]
    if missing:
        raise KeyError(f"features absent from train or test: {missing}")
    cat_set = [c for c in categorical if c in features]
    X_train = pd.DataFrame(index=train.index)
    X_test = pd.DataFrame(index=test.index)
    for col in features:
        if col in cat_set:
            a = _as_category_strings(train[col])
            b = _as_category_strings(test[col])
            vocab = sorted(set(a.unique()) | set(b.unique()))
            dtype = pd.CategoricalDtype(categories=vocab)
            X_train[col] = a.astype(dtype)
            X_test[col] = b.astype(dtype)
        else:
            # Engineered ratios divide by zero now and then; every family handles NaN, none
            # handles inf.
            X_train[col] = _numeric(train[col])
            X_test[col] = _numeric(test[col])
    return Matrix(X_train, X_test, list(features), cat_set)
