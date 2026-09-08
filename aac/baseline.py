"""Constant-prediction baseline: the shortest end-to-end path through the Kaggle layer.

README section 14, milestone 2: download the data, write and validate a constant submission,
upload it, poll the public score, and record everything in the ledger. Also the permanent
smoke test for the submission handshake.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from aac.agents.submitter import upload_submission
from aac.config import Config
from aac.context import RunContext
from aac.kaggle.api import KaggleClient
from aac.kaggle.data import ensure_data
from aac.kaggle.submission import (
    build_submission,
    target_from_sample,
    validate_submission,
    write_submission,
)
from aac.models.metrics import MetricSpec, resolve_metric, score
from aac.models.target import TargetEncoding, infer_target_encoding

log = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    run_id: str
    submission_path: Path
    target: str
    constant: object
    train_score: float | None
    submitted: bool
    ref: int | None = None
    public_score: float | None = None
    status: str = ""
    message: str = ""


def constant_prediction(
    y: pd.Series, spec: MetricSpec
) -> tuple[object, str, np.ndarray, TargetEncoding]:
    """The constant to submit, its kind, the per-row score matrix for ``y``, and the encoding.

    Probability metrics get the positive rate (binary) or 1/k (multiclass). Label metrics get
    the majority class.
    """
    enc = infer_target_encoding(y)
    codes = enc.encode(y)
    n, k = len(y), enc.n_classes
    if spec.needs_proba:
        if enc.is_binary:
            rate = float(codes.mean())
            return rate, "proba", np.full(n, rate), enc
        return 1.0 / k, "proba", np.full((n, k), 1.0 / k), enc
    majority = int(np.bincount(codes).argmax())
    onehot = np.zeros((n, k))
    onehot[:, majority] = 1.0
    per_row = onehot[:, 1] if enc.is_binary else onehot
    return enc.classes[majority], "label", per_row, enc


def run_baseline(
    config: Config,
    *,
    submit: bool = True,
    runs_root: Path = Path("runs"),
    transport: httpx.BaseTransport | None = None,
    poll_timeout: float = 600.0,
    poll_interval: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
    console: Console | None = None,
) -> BaselineResult:
    console = console or Console()
    slug = config.competition.slug
    http = httpx.Client(transport=transport) if transport is not None else None
    ctx = RunContext.create(config, runs_root)
    try:
        with KaggleClient.from_env(client=http, sleep=sleep) as kaggle:
            info = kaggle.competition(slug)
            spec = resolve_metric(info.metric_name, override=config.competition.metric)
            files = ensure_data(kaggle, slug, ctx.paths.data_dir(slug))
            train = pd.read_parquet(files.train)
            sample = pd.read_parquet(files.sample_submission)
            target = config.competition.target or target_from_sample(sample, train.columns)
            if target not in train.columns:
                raise ValueError(f"target column {target!r} not in train")
            y = train[target]
            constant, kind, per_row, enc = constant_prediction(y, spec)
            train_score = score(spec, enc.encode(y), per_row)

            preds = np.full(len(sample), constant)
            sub = build_submission(sample, preds)
            validate_submission(
                sub, sample, kind=kind, allowed_labels=enc.classes if kind == "label" else None
            )
            path = write_submission(ctx.paths.run_dir / "submission.csv", sub)
            result = BaselineResult(ctx.run_id, path, target, constant, train_score, False)
            log.info(
                "baseline %s=%r, train %s %.5f, written to %s",
                target,
                constant,
                spec.key,
                train_score,
                path,
            )

            if submit and config.kaggle.submit:
                description = f"aac baseline: constant {target}={constant!r} (run {ctx.run_id})"
                result.submitted = True
                final = upload_submission(
                    ctx,
                    kaggle,
                    info,
                    path,
                    description=description,
                    oof_score=train_score,
                    poll_timeout=poll_timeout,
                    poll_interval=poll_interval,
                )
                result.ref = final.ref
                result.public_score = final.public_score
                result.status = final.status
                result.message = final.error_description
        ctx.finish("completed")
    except Exception as exc:
        ctx.finish("failed", error=f"{exc.__class__.__name__}: {exc}"[:500])
        raise
    finally:
        ctx.close()
        if http is not None:
            http.close()

    table = Table(title="aac baseline")
    table.add_column("field", style="bold")
    table.add_column("value")
    for name, value in (
        ("run", result.run_id),
        ("target", result.target),
        ("constant", repr(result.constant)),
        (
            f"train {spec.key}",
            f"{result.train_score:.5f}" if result.train_score is not None else "",
        ),
        ("file", str(result.submission_path)),
        ("submitted", "yes" if result.submitted else "no"),
        ("kaggle ref", str(result.ref or "")),
        ("status", result.status),
        ("public score", "" if result.public_score is None else f"{result.public_score:.5f}"),
    ):
        table.add_row(name, value)
    console.print(table)
    return result
