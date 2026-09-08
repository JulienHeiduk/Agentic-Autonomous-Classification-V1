"""Model family wrappers with deterministic settings (README sections 2, 4, 10).

Every wrapper exposes ``fit(X, y, X_es, y_es)`` and ``predict_proba(X) -> (n, k)``. Seeds and
thread counts are always explicit. Early stopping, when used, sees only the carve-out the
Trainer hands in, never the OOF fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from aac.models.prepare import MISSING_CATEGORY

EARLY_STOPPING_FAMILIES = frozenset({"lightgbm", "xgboost", "catboost"})


class Estimator(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_es: pd.DataFrame | None = None,
        y_es: np.ndarray | None = None,
    ) -> None: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def importances(self) -> dict[str, float] | None: ...

    @property
    def best_iteration(self) -> int | None: ...


def _full_proba(proba: np.ndarray, n_classes: int) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] != n_classes:
        raise ValueError(f"predict_proba gave {proba.shape[1]} columns, expected {n_classes}")
    if n_classes > 2:  # float32 outputs can sum to 1 +/- 1e-7; scorers want exact rows
        proba = proba / proba.sum(axis=1, keepdims=True)
    return proba


@dataclass
class _Base:
    params: dict[str, Any]
    n_classes: int
    seed: int
    n_jobs: int
    categorical: list[str]
    model: Any = field(default=None, init=False, repr=False)
    _features: list[str] = field(default_factory=list, init=False, repr=False)
    _best_iteration: int | None = field(default=None, init=False, repr=False)

    @property
    def binary(self) -> bool:
        return self.n_classes == 2

    @property
    def best_iteration(self) -> int | None:
        return self._best_iteration

    def _named(self, values: np.ndarray | None) -> dict[str, float] | None:
        if values is None:
            return None
        values = np.asarray(values, dtype=np.float64)
        return {f: float(v) for f, v in zip(self._features, values, strict=True)}


class LightGBM(_Base):
    def fit(self, X, y, X_es=None, y_es=None):
        import lightgbm as lgb

        self._features = list(X.columns)
        params = {
            **self.params,
            "objective": "binary" if self.binary else "multiclass",
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
        }
        if not self.binary:
            params["num_class"] = self.n_classes
        self.model = lgb.LGBMClassifier(**params)
        # "auto" picks up pandas category dtype; naming columns breaks with eval_X in 4.7.
        kwargs: dict[str, Any] = {"categorical_feature": "auto"}
        if X_es is not None:
            kwargs["eval_X"] = X_es  # one matrix, or a tuple of them; a list means raw rows
            kwargs["eval_y"] = y_es
            kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        self.model.fit(X, y, **kwargs)
        self._best_iteration = int(self.model.best_iteration_ or 0) or None

    def predict_proba(self, X):
        return _full_proba(self.model.predict_proba(X), self.n_classes)

    def importances(self):
        return self._named(self.model.booster_.feature_importance(importance_type="gain"))


class XGBoost(_Base):
    def fit(self, X, y, X_es=None, y_es=None):
        import xgboost as xgb

        self._features = list(X.columns)
        params = {
            **self.params,
            "objective": "binary:logistic" if self.binary else "multi:softprob",
            "eval_metric": "logloss" if self.binary else "mlogloss",
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
            "verbosity": 0,
        }
        if X_es is not None:
            params["early_stopping_rounds"] = 50
        self.model = xgb.XGBClassifier(**params)
        kwargs: dict[str, Any] = {}
        if X_es is not None:
            kwargs["eval_set"] = [(X_es, y_es)]
            kwargs["verbose"] = False
        self.model.fit(X, y, **kwargs)
        best = getattr(self.model, "best_iteration", None)
        self._best_iteration = int(best) + 1 if best is not None else None

    def predict_proba(self, X):
        return _full_proba(self.model.predict_proba(X), self.n_classes)

    def importances(self):
        return self._named(self.model.feature_importances_)


class CatBoost(_Base):
    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.categorical:
            out[c] = out[c].astype(str)
        return out

    def fit(self, X, y, X_es=None, y_es=None):
        from catboost import CatBoostClassifier

        self._features = list(X.columns)
        params = {
            **self.params,
            "loss_function": "Logloss" if self.binary else "MultiClass",
            "random_seed": self.seed,
            "thread_count": self.n_jobs,
            "verbose": 0,
            "allow_writing_files": False,
        }
        if X_es is not None:
            params["early_stopping_rounds"] = 50
        self.model = CatBoostClassifier(**params)
        cat_idx = [self._features.index(c) for c in self.categorical]
        eval_set = (self._frame(X_es), y_es) if X_es is not None else None
        self.model.fit(self._frame(X), y, cat_features=cat_idx, eval_set=eval_set)
        best = self.model.get_best_iteration()
        self._best_iteration = int(best) + 1 if best is not None else None

    def predict_proba(self, X):
        return _full_proba(self.model.predict_proba(self._frame(X)), self.n_classes)

    def importances(self):
        return self._named(self.model.get_feature_importance())


class HistGBDT(_Base):
    MAX_CATEGORIES = 255

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.categorical:
            if len(out[c].cat.categories) > self.MAX_CATEGORIES:
                out[c] = out[c].cat.codes.astype(np.float64)  # ordinal fallback
        return out

    def fit(self, X, y, X_es=None, y_es=None):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits

        self._features = list(X.columns)
        self.model = HistGradientBoostingClassifier(
            **self.params,
            categorical_features="from_dtype",
            random_state=self.seed,
            early_stopping=False,
        )
        # scikit-learn has no n_jobs here; OpenMP would take every core, and a varying thread
        # count is a determinism risk. Pin it the same way as the other families.
        with threadpool_limits(limits=self.n_jobs):
            self.model.fit(self._frame(X), y)
        self._best_iteration = int(self.model.n_iter_)

    def predict_proba(self, X):
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=self.n_jobs):
            proba = self.model.predict_proba(self._frame(X))
        return _full_proba(proba, self.n_classes)

    def importances(self):
        return None


class Logistic(_Base):
    def fit(self, X, y, X_es=None, y_es=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        self._features = list(X.columns)
        numeric = [c for c in self._features if c not in self.categorical]
        transformers = []
        if numeric:
            transformers.append(
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric)
            )
        if self.categorical:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True),
                    self.categorical,
                )
            )
        self.model = Pipeline(
            [
                ("prep", ColumnTransformer(transformers, sparse_threshold=0.3)),
                ("clf", LogisticRegression(**self.params, random_state=self.seed)),
            ]
        )
        self.model.fit(self._strings(X), y)

    def _strings(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.categorical:
            out[c] = out[c].astype(str).replace("nan", MISSING_CATEGORY)
        return out

    def predict_proba(self, X):
        return _full_proba(self.model.predict_proba(self._strings(X)), self.n_classes)

    def importances(self):
        return None


FAMILIES: dict[str, type[_Base]] = {
    "lightgbm": LightGBM,
    "xgboost": XGBoost,
    "catboost": CatBoost,
    "hist_gbdt": HistGBDT,
    "logistic": Logistic,
}


def build_model(
    family: str,
    params: dict[str, Any],
    *,
    n_classes: int,
    seed: int,
    n_jobs: int,
    categorical: list[str],
) -> Estimator:
    try:
        cls = FAMILIES[family]
    except KeyError:
        raise ValueError(f"unknown model family {family!r}; known: {sorted(FAMILIES)}") from None
    return cls(
        params=dict(params), n_classes=n_classes, seed=seed, n_jobs=n_jobs, categorical=categorical
    )


def supports_early_stopping(family: str) -> bool:
    return family in EARLY_STOPPING_FAMILIES
