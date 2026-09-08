"""Submission construction, validation, and the daily budget (README sections 6 and 10).

The sample submission is the contract: same columns in the same order, same row count, the id
column equal and in order, no NaN, probabilities in [0, 1] or labels drawn from the training
label set. Nothing is uploaded until ``validate_submission`` passes.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from aac.exec.artifacts import atomic_write_bytes
from aac.kaggle.api import Submission

Kind = Literal["proba", "label"]


class SubmissionFormatError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def id_column(sample: pd.DataFrame) -> str:
    return str(sample.columns[0])


def prediction_columns(sample: pd.DataFrame) -> list[str]:
    return [str(c) for c in sample.columns[1:]]


def target_from_sample(sample: pd.DataFrame, train_columns: Iterable[str]) -> str:
    """Playground competitions name the sample's prediction column after the target."""
    train_columns = set(train_columns)
    candidates = [c for c in prediction_columns(sample) if c in train_columns]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"cannot infer the target from sample columns {prediction_columns(sample)}; "
        "set competition.target"
    )


def build_submission(sample: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    preds = np.asarray(predictions)
    if preds.ndim == 1:
        preds = preds[:, None]
    cols = prediction_columns(sample)
    if preds.shape != (len(sample), len(cols)):
        raise SubmissionFormatError(
            [f"predictions shape {preds.shape} does not match ({len(sample)}, {len(cols)})"]
        )
    out = sample[[id_column(sample)]].reset_index(drop=True).copy()
    for i, col in enumerate(cols):
        out[col] = preds[:, i]
    return out


def validate_submission(
    sub: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    kind: Kind = "proba",
    allowed_labels: Iterable[object] | None = None,
) -> None:
    problems: list[str] = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f"columns {list(sub.columns)} != sample {list(sample.columns)}")
    if len(sub) != len(sample):
        problems.append(f"{len(sub)} rows, sample has {len(sample)}")
    else:
        idc = id_column(sample)
        if idc in sub.columns:
            ours = sub[idc].astype(str).reset_index(drop=True)
            theirs = sample[idc].astype(str).reset_index(drop=True)
            if not ours.equals(theirs):
                problems.append(f"id column {idc!r} differs from sample in values or order")
    na_cols = [str(c) for c in sub.columns if sub[c].isna().any()]
    if na_cols:
        problems.append(f"NaN in {na_cols}")
    for col in prediction_columns(sample):
        if col not in sub.columns:
            continue
        if kind == "proba":
            values = pd.to_numeric(sub[col], errors="coerce")
            if values.isna().any() and not sub[col].isna().any():
                problems.append(f"{col!r}: non-numeric values")
            elif ((values < 0) | (values > 1)).any():
                lo, hi = values.min(), values.max()
                problems.append(f"{col!r}: values outside [0, 1] (min {lo:.4g}, max {hi:.4g})")
        elif allowed_labels is not None:
            allowed = set(allowed_labels)
            unknown = sorted({v for v in sub[col].unique() if v not in allowed}, key=str)
            if unknown:
                problems.append(f"{col!r}: labels not in training set: {unknown[:10]}")
    if problems:
        raise SubmissionFormatError(problems)


def write_submission(path: Path, sub: pd.DataFrame) -> Path:
    path = Path(path)
    atomic_write_bytes(path, sub.to_csv(index=False).encode("utf-8"))
    return path


def _submission_day(sub: Submission) -> datetime | None:
    if not sub.date:
        return None
    try:
        dt = datetime.fromisoformat(sub.date.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def submissions_today(subs: Iterable[Submission], now: datetime | None = None) -> int:
    """Kaggle's daily quota resets at UTC midnight; every upload counts, errored ones too."""
    today = (now or datetime.now(UTC)).astimezone(UTC).date()
    count = 0
    for sub in subs:
        day = _submission_day(sub)
        if day is not None and day.date() == today:
            count += 1
    return count


def submissions_remaining(
    subs: Iterable[Submission], max_per_day: int, now: datetime | None = None
) -> int:
    return max(0, max_per_day - submissions_today(subs, now))
