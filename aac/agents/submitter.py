"""Submitter: deterministic (README section 3).

Turns predictions into the sample submission's shape, validates against the sample, writes
the CSV, and uploads only inside the daily budget. Records every upload in the ledger and
polls for the public score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aac.agents.scout import Profile
from aac.context import BudgetExceeded, RunContext
from aac.kaggle.api import CompetitionInfo, KaggleClient, KaggleError, Submission
from aac.kaggle.submission import (
    build_submission,
    submissions_remaining,
    submissions_today,
    validate_submission,
    write_submission,
)
from aac.models.target import TargetEncoding

log = logging.getLogger(__name__)


def predictions_to_submission(
    pred: np.ndarray, profile: Profile, sample: pd.DataFrame, enc: TargetEncoding
) -> pd.DataFrame:
    """Map a probability vector/matrix onto the sample submission's columns."""
    pred = np.asarray(pred, dtype=np.float64)
    kind = profile.submission.kind
    columns = profile.submission.columns
    if kind == "proba":
        values = pred[:, 1] if pred.ndim == 2 else pred
        sub = build_submission(sample, values)
        validate_submission(sub, sample, kind="proba")
        return sub
    if kind == "proba_per_class":
        if pred.ndim != 2 or pred.shape[1] != enc.n_classes:
            raise ValueError(f"per-class submission needs (n, {enc.n_classes}) predictions")
        by_name = {str(c): i for i, c in enumerate(enc.classes)}
        matrix = np.column_stack([pred[:, by_name[col]] for col in columns])
        sub = build_submission(sample, matrix)
        validate_submission(sub, sample, kind="proba")
        return sub
    codes = pred.argmax(axis=1) if pred.ndim == 2 else (pred >= 0.5).astype(int)
    labels = enc.decode(codes)
    sample_dtype = sample[columns[0]].dtype
    try:
        labels = pd.Series(labels).astype(sample_dtype).to_numpy()
    except (TypeError, ValueError):
        labels = np.asarray(labels, dtype=object)
    sub = build_submission(sample, labels)
    validate_submission(sub, sample, kind="label", allowed_labels=enc.classes)
    return sub


def write_prediction_file(
    path: Path, pred: np.ndarray, profile: Profile, sample: pd.DataFrame, enc: TargetEncoding
) -> Path:
    return write_submission(path, predictions_to_submission(pred, profile, sample, enc))


def upload_submission(
    ctx: RunContext,
    kaggle: KaggleClient,
    info: CompetitionInfo,
    path: Path,
    *,
    description: str,
    oof_score: float | None,
    poll_timeout: float = 600.0,
    poll_interval: float = 15.0,
) -> Submission:
    """Enforce the daily budget, upload, record, and poll for the public score."""
    slug = ctx.config.competition.slug
    if info.submissions_disabled:
        raise KaggleError("submissions are disabled for this competition")
    if not info.user_has_entered:
        raise KaggleError("accept the competition rules on the website first")
    history = kaggle.list_submissions(slug)
    cap = ctx.config.kaggle.max_submissions_per_day
    if info.max_daily_submissions:
        cap = min(cap, info.max_daily_submissions)
    used = submissions_today(history)
    if submissions_remaining(history, cap) <= 0:
        raise BudgetExceeded(f"submission budget: {used} of {cap} used today")
    ref = kaggle.submit(slug, path, description)
    sid = ctx.ledger.record_submission(
        ctx.run_id, path.name, description=description, oof_score=oof_score, kaggle_ref=str(ref)
    )
    final = kaggle.wait_for_score(slug, ref, timeout=poll_timeout, interval=poll_interval)
    ctx.ledger.set_public_score(sid, final.public_score)
    log.info("submission %d: %s public=%s", ref, final.status, final.public_score)
    if final.failed:
        raise KaggleError(f"submission {ref} failed: {final.error_description}")
    return final
