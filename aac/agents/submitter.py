"""Submitter: deterministic (README section 3).

Turns predictions into the sample submission's shape, validates against the sample, writes
the CSV, and uploads only inside the daily budget. Records every upload in the ledger and
polls for the public score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
    final = kaggle.wait_for_score(
        slug, ref, timeout=poll_timeout, interval=poll_interval, raise_on_timeout=False
    )
    if final.failed:
        raise KaggleError(f"submission {ref} failed: {final.error_description}")
    if not final.settled:
        # The upload succeeded; Kaggle is still scoring. The row keeps a NULL score until the
        # next run, `aac submit`, or `aac scores` writes it back.
        log.warning(
            "submission %d still %s after %.0fs; the score will be written back later",
            ref,
            final.status,
            poll_timeout,
        )
        return final
    ctx.ledger.set_public_score(sid, final.public_score)
    log.info("submission %d: %s public=%s", ref, final.status, final.public_score)
    return final


PENDING_RUN_ERROR = "KaggleError: submission "


def backfill_public_scores(ledger, kaggle: KaggleClient, slug: str) -> list[dict[str, Any]]:
    """Write back the public scores of uploads that were still pending when their run ended,
    and mark a run that failed only on that poll as completed. Returns the rows updated."""
    pending = ledger.pending_submissions(slug)
    if not pending:
        return []
    by_ref = {str(s.ref): s for s in kaggle.list_submissions(slug)}
    updated: list[dict[str, Any]] = []
    for row in pending:
        sub = by_ref.get(str(row["kaggle_ref"]))
        if sub is None or not sub.complete or sub.public_score is None:
            continue
        ledger.set_public_score(int(row["id"]), sub.public_score)
        row = {**row, "public_score": sub.public_score}
        updated.append(row)
        run = ledger.get_run(str(row["run_id"]))
        if (
            run is not None
            and run.get("status") == "failed"
            and str(run.get("error") or "").startswith(
                f"{PENDING_RUN_ERROR}{row['kaggle_ref']} still"
            )
        ):
            ledger.set_run_status(str(row["run_id"]), "completed", None)
            log.info(
                "run %s: marked completed, its upload scored %.5f", row["run_id"], sub.public_score
            )
        log.info(
            "submission %s: public score %.5f written back", row["kaggle_ref"], sub.public_score
        )
    return updated
