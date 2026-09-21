"""Scout: deterministic data profile, no LLM (README section 3).

Infers the id and target columns, the target kind, and the submission shape; types every
column; measures missingness and train/test drift. Refuses anything that is not a
classification problem it can handle. The compact JSON profile is what every downstream
prompt sees instead of raw data.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from aac.config import CompetitionConfig, StrictModel
from aac.kaggle.submission import id_column, prediction_columns
from aac.models.metrics import MetricSpec
from aac.models.target import TargetEncoding, TargetError, infer_target_encoding

ColumnKind = Literal["numeric", "categorical", "boolean", "datetime", "text", "constant"]
SubmissionKind = Literal["proba", "label", "proba_per_class"]

TEXT_MIN_UNIQUE = 100
TEXT_UNIQUE_RATIO = 0.5
DRIFT_SAMPLE = 20_000
DRIFT_WARN = 0.1
TOP_VALUES = 8
BOOL_STRINGS = frozenset({"true", "false"})


class ScoutError(ValueError):
    """The data is not a classification problem this framework can run."""


class ColumnProfile(StrictModel):
    name: str
    dtype: str
    kind: ColumnKind
    n_unique: int
    missing_train: float
    missing_test: float | None
    in_test: bool
    usable: bool
    stats: dict[str, float] | None = None
    top_values: list[tuple[str, float]] | None = None
    drift: float | None = None


class TargetProfile(StrictModel):
    name: str
    kind: Literal["binary", "multiclass"]
    classes: list[Any]
    positive_label: Any | None
    counts: dict[str, int]
    rates: dict[str, float]
    imbalance_ratio: float  # majority / minority


class SubmissionProfile(StrictModel):
    kind: SubmissionKind
    id_col: str
    columns: list[str]


class Profile(StrictModel):
    slug: str
    metric: str
    n_train: int
    n_test: int
    n_columns: int
    memory_mb: float
    id_col: str
    target: TargetProfile
    submission: SubmissionProfile
    columns: list[ColumnProfile]
    warnings: list[str] = Field(default_factory=list)

    def feature_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.usable]

    def categorical_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.usable and c.kind in ("categorical", "boolean")]

    def unusable_columns(self) -> list[str]:
        return [c.name for c in self.columns if not c.usable]

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def encoding(self) -> TargetEncoding:
        return TargetEncoding(tuple(self.target.classes))


# -- inference ----------------------------------------------------------------------------


def infer_id_col(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, configured: str | None
) -> str:
    if configured:
        if configured not in train.columns or configured not in test.columns:
            raise ScoutError(f"competition.id_col={configured!r} is not in both train and test")
        return configured
    first = id_column(sample)
    if first in train.columns and first in test.columns:
        return first
    for col in test.columns:
        if (
            col in train.columns
            and "id" in col.lower()
            and test[col].is_unique
            and train[col].is_unique
        ):
            return col
    raise ScoutError(
        f"cannot infer the id column: sample starts with {first!r}, absent from train/test; "
        "set competition.id_col"
    )


def infer_target(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, configured: str | None
) -> str:
    if configured:
        if configured not in train.columns:
            raise ScoutError(f"competition.target={configured!r} is not in train")
        return configured
    pred_cols = prediction_columns(sample)
    if len(pred_cols) == 1 and pred_cols[0] in train.columns and pred_cols[0] not in test.columns:
        return pred_cols[0]
    only_in_train = [c for c in train.columns if c not in test.columns]
    if len(only_in_train) == 1:
        return only_in_train[0]
    raise ScoutError(
        f"cannot infer the target: sample predicts {pred_cols}, columns only in train are "
        f"{only_in_train}; set competition.target"
    )


def infer_submission_kind(
    sample: pd.DataFrame, enc: TargetEncoding, metric: MetricSpec
) -> SubmissionKind:
    pred_cols = prediction_columns(sample)
    if len(pred_cols) > 1:
        wanted = {str(c) for c in enc.classes}
        if set(pred_cols) != wanted:
            raise ScoutError(
                f"sample has {len(pred_cols)} prediction columns {pred_cols} but the target "
                f"classes are {sorted(wanted)}"
            )
        return "proba_per_class"
    if metric.needs_proba:
        if not enc.is_binary:
            raise ScoutError(
                f"metric {metric.key} needs probabilities for {enc.n_classes} classes but the "
                "sample has a single prediction column"
            )
        return "proba"
    return "label"


def target_profile(y: pd.Series, enc: TargetEncoding, max_classes: int) -> TargetProfile:
    if enc.n_classes > max_classes:
        raise ScoutError(
            f"target {y.name!r} has {enc.n_classes} distinct values (> run.max_classes="
            f"{max_classes}); this looks like regression, which V1 does not do"
        )
    if pd.api.types.is_float_dtype(y) and not np.all(np.mod(y.dropna().to_numpy(), 1) == 0):
        raise ScoutError(
            f"target {y.name!r} has non-integer float values; not a classification target"
        )
    counts = y.value_counts()
    ordered = {str(c): int(counts.get(c, 0)) for c in enc.classes}
    n = int(counts.sum())
    return TargetProfile(
        name=str(y.name),
        kind="binary" if enc.is_binary else "multiclass",
        classes=[_jsonable(c) for c in enc.classes],
        positive_label=_jsonable(enc.positive_label),
        counts=ordered,
        rates={k: v / n for k, v in ordered.items()},
        imbalance_ratio=float(max(ordered.values()) / max(1, min(ordered.values()))),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# -- column typing ------------------------------------------------------------------------


def column_kind(s: pd.Series) -> ColumnKind:
    non_null = s.dropna()
    n_unique = non_null.nunique()
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    lowered = {str(v).strip().lower() for v in non_null.unique()[:10]}
    if n_unique == 2 and lowered <= BOOL_STRINGS:
        return "boolean"
    if n_unique >= TEXT_MIN_UNIQUE and n_unique / max(1, len(non_null)) > TEXT_UNIQUE_RATIO:
        return "text"
    return "categorical"


def _numeric_stats(s: pd.Series) -> dict[str, float]:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {}
    return {
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=0)),
    }


def _top_values(s: pd.Series) -> list[tuple[str, float]]:
    counts = s.value_counts(normalize=True, dropna=True).head(TOP_VALUES)
    return [(str(k), float(v)) for k, v in counts.items()]


def _numeric_drift(a: pd.Series, b: pd.Series, seed: int) -> float | None:
    from scipy.stats import ks_2samp

    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(a) < 20 or len(b) < 20:
        return None
    rng = np.random.default_rng(seed)
    if len(a) > DRIFT_SAMPLE:
        a = rng.choice(a, DRIFT_SAMPLE, replace=False)
    if len(b) > DRIFT_SAMPLE:
        b = rng.choice(b, DRIFT_SAMPLE, replace=False)
    return float(ks_2samp(a, b).statistic)


def _categorical_drift(a: pd.Series, b: pd.Series) -> float | None:
    if a.dropna().empty or b.dropna().empty:
        return None
    pa = a.astype(str).value_counts(normalize=True)
    pb = b.astype(str).value_counts(normalize=True)
    keys = pa.index.union(pb.index)
    return float(
        0.5 * np.abs(pa.reindex(keys, fill_value=0) - pb.reindex(keys, fill_value=0)).sum()
    )


def profile_column(
    name: str, train: pd.DataFrame, test: pd.DataFrame, *, id_col: str, seed: int
) -> ColumnProfile:
    s = train[name]
    in_test = name in test.columns
    kind = column_kind(s)
    stats = top = None
    drift = None
    if kind == "numeric":
        stats = _numeric_stats(s)
        if in_test:
            drift = _numeric_drift(s, test[name], seed)
    elif kind in ("categorical", "boolean"):
        top = _top_values(s)
        if in_test:
            drift = _categorical_drift(s, test[name])
    usable = kind in ("numeric", "categorical", "boolean") and in_test and name != id_col
    return ColumnProfile(
        name=name,
        dtype=str(s.dtype),
        kind=kind,
        n_unique=int(s.dropna().nunique()),
        missing_train=float(s.isna().mean()),
        missing_test=float(test[name].isna().mean()) if in_test else None,
        in_test=in_test,
        usable=usable,
        stats=stats,
        top_values=top,
        drift=drift,
    )


# -- entry point --------------------------------------------------------------------------


def profile_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    slug: str,
    competition: CompetitionConfig,
    metric: MetricSpec,
    max_classes: int,
    seed: int,
) -> Profile:
    id_col = infer_id_col(train, test, sample, competition.id_col)
    target = infer_target(train, test, sample, competition.target)
    if target == id_col:
        raise ScoutError(f"target and id resolve to the same column {target!r}")
    y = train[target]
    try:
        enc = infer_target_encoding(y)
    except TargetError as exc:
        raise ScoutError(str(exc)) from exc
    tprofile = target_profile(y, enc, max_classes)
    kind = infer_submission_kind(sample, enc, metric)
    if len(sample) != len(test):
        raise ScoutError(f"sample submission has {len(sample)} rows, test has {len(test)}")

    columns = [
        profile_column(c, train, test, id_col=id_col, seed=seed)
        for c in train.columns
        if c != target
    ]
    warnings: list[str] = []
    for c in columns:
        if c.name == id_col:
            continue
        if not c.in_test:
            warnings.append(f"{c.name}: only in train, excluded")
        elif c.kind == "constant":
            warnings.append(f"{c.name}: constant, excluded")
        elif c.kind == "text":
            warnings.append(
                f"{c.name}: {c.n_unique} distinct strings, looks like free text or an identifier, "
                "excluded from the default plan"
            )
        elif c.kind == "datetime":
            warnings.append(f"{c.name}: datetime, excluded from the default plan (V1)")
        if c.drift is not None and c.drift > DRIFT_WARN:
            warnings.append(f"{c.name}: train/test drift {c.drift:.3f}")
    if tprofile.imbalance_ratio > 5:
        warnings.append(f"target imbalance {tprofile.imbalance_ratio:.1f}:1")
    extra_in_test = [c for c in test.columns if c not in train.columns]
    if extra_in_test:
        warnings.append(f"columns only in test, ignored: {extra_in_test}")

    return Profile(
        slug=slug,
        metric=metric.key,
        n_train=len(train),
        n_test=len(test),
        n_columns=train.shape[1],
        memory_mb=float(train.memory_usage(deep=True).sum() / 1e6),
        id_col=id_col,
        target=tprofile,
        submission=SubmissionProfile(
            kind=kind, id_col=id_column(sample), columns=prediction_columns(sample)
        ),
        columns=columns,
        warnings=warnings,
    )
