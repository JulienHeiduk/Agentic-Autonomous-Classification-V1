"""Assessor: interview each configured model and keep a track record (README 17.2, phase 4).

The interview is the real task in miniature: one experiment on a stratified subsample of the
competition, written from the Researcher prompt, run by the harness. The record (valid
module, ran, OOF, latency, tokens) lives in the ledger and drives job allocation: a model
that has never produced a working module after ``skip_after_failures`` interviews is left
out of the run.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aac.agents.researcher import degenerate_check, parse_reply, researcher_messages
from aac.agents.scout import Profile
from aac.config import AssessorConfig, ResearcherSpec
from aac.exec.sandbox import run_experiment
from aac.ledger import Ledger
from aac.llm.client import LLMError
from aac.llm.prompts import load_prompt
from aac.llm.router import Router
from aac.models.cv import make_folds
from aac.models.metrics import MetricSpec

log = logging.getLogger(__name__)


@dataclass
class Interview:
    backend: str
    model: str
    valid_module: bool
    ran_ok: bool
    oof_score: float | None
    latency: float
    tokens: int
    error: str | None = None


def subsample(
    train: pd.DataFrame, test: pd.DataFrame, target: str, rows: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified subsample of train and a matching slice of test."""
    if len(train) <= rows:
        return train, test.head(max(50, rows // 4))
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in train.groupby(target, sort=True):
        take = max(1, int(round(rows * len(group) / len(train))))
        idx = rng.choice(len(group), size=min(take, len(group)), replace=False)
        parts.append(group.iloc[np.sort(idx)])
    sub = pd.concat(parts).sort_index()
    return sub, test.head(max(50, rows // 4))


def interview(
    router: Router,
    ledger: Ledger,
    profile: Profile,
    metric: MetricSpec,
    *,
    config: AssessorConfig,
    candidates: list[tuple[str, str]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    workdir: Path,
    seed: int,
    n_threads: int,
    degenerate_margin: float = 0.002,
) -> list[Interview]:
    """One interview per (backend, model) that has no record for this task yet."""
    task = f"{profile.slug}@{config.rows}"
    todo = []
    for backend, model in candidates:
        records = [
            r for r in ledger.track_records(backend=backend, model=model) if r["task"] == task
        ]
        # interview once when it worked; a model with no working module gets another chance
        # each run until allocate() gives up on it
        if not records or not any(r["ran_ok"] for r in records):
            todo.append((backend, model))
    if not todo:
        return []
    target = profile.target.name
    sub_train, sub_test = subsample(train, test, target, config.rows, seed)
    enc = profile.encoding()
    y = enc.encode(sub_train[target])
    folds = make_folds(y, config.n_folds, seed)
    workdir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = workdir / "train.parquet", workdir / "test.parquet"
    sub_train.drop(columns=[target]).to_parquet(train_path, index=False)
    sub_test.to_parquet(test_path, index=False)
    sub_profile = profile.model_copy(update={"n_train": len(sub_train), "n_test": len(sub_test)})

    def one(pair: tuple[str, str]) -> Interview:
        backend, model = pair
        spec = ResearcherSpec(backend=backend, model=model, track="open", temperature=0.3)
        messages = researcher_messages(
            sub_profile,
            metric=metric,
            spec=spec,
            experiments=[],
            best=None,
            round_no=1,
            max_rounds=1,
            timeout=config.timeout_seconds,
            notes="Research notes: this is an interview on a subsample; write your strongest "
            "single experiment for this data.",
        )
        started = time.monotonic()
        try:
            completion = router.complete(
                messages,
                agent="assessor",
                backend=backend,
                model=model,
                temperature=0.3,
                max_tokens=8000,
                allow_failover=False,
            )
        except LLMError as exc:
            return Interview(
                backend, model, False, False, None, time.monotonic() - started, 0, str(exc)[:300]
            )
        latency = completion.latency_s
        tokens = completion.total_tokens
        _, code = parse_reply(completion.content)
        if code is None:
            retry = [
                *messages,
                {"role": "assistant", "content": completion.content},
                *load_prompt("code_repair").messages(),
            ]
            try:
                completion = router.complete(
                    retry,
                    agent="assessor:repair",
                    backend=backend,
                    model=model,
                    temperature=0.0,
                    max_tokens=8000,
                    allow_failover=False,
                )
                latency += completion.latency_s
                tokens += completion.total_tokens
                _, code = parse_reply(completion.content)
            except LLMError as exc:
                return Interview(
                    backend, model, False, False, None, latency, tokens, str(exc)[:300]
                )
        if code is None:
            return Interview(
                backend, model, False, False, None, latency, tokens, "no module in the reply"
            )
        safe = model.replace("/", "_").replace(":", "_")
        result = run_experiment(
            code,
            workdir=workdir / f"{backend}__{safe}",
            train_path=train_path,
            test_path=test_path,
            y=y,
            folds=folds,
            metric=metric,
            target=target,
            id_col=profile.id_col,
            n_classes=enc.n_classes,
            categorical=profile.categorical_columns(),
            drop_columns=profile.unusable_columns(),
            seed=seed,
            timeout=config.timeout_seconds,
            n_threads=n_threads,
            determinism_rows=min(2000, config.rows),
        )
        if result.ok:
            result = degenerate_check(result, y, metric, degenerate_margin)
        return Interview(
            backend,
            model,
            True,
            result.ok,
            result.oof_score,
            latency,
            tokens,
            None if result.ok else result.error[-300:],
        )

    with ThreadPoolExecutor(max_workers=max(1, min(4, len(todo)))) as pool:
        results = list(pool.map(one, todo))
    for r in results:
        ledger.record_track(
            backend=r.backend,
            model=r.model,
            task=task,
            valid_module=r.valid_module,
            ran_ok=r.ran_ok,
            oof_score=r.oof_score,
            latency=r.latency,
            tokens=r.tokens,
            error=r.error,
        )
        log.info(
            "interview %s/%s: module=%s ran=%s oof=%s latency=%.0fs tokens=%d",
            r.backend,
            r.model,
            r.valid_module,
            r.ran_ok,
            f"{r.oof_score:.5f}" if r.oof_score is not None else "-",
            r.latency,
            r.tokens,
        )
    return results


def allocate(
    specs: list[ResearcherSpec], ledger: Ledger, config: AssessorConfig
) -> tuple[list[ResearcherSpec], list[str]]:
    """Drop Researchers whose model never produced a working module in enough interviews."""
    kept, dropped = [], []
    for spec in specs:
        records = ledger.track_records(backend=spec.backend, model=spec.model)
        if len(records) >= config.skip_after_failures and not any(r["ran_ok"] for r in records):
            dropped.append(
                f"{spec.backend}/{spec.model}: 0 working modules in {len(records)} interviews"
            )
        else:
            kept.append(spec)
    return kept, dropped


def track_record_text(ledger: Ledger) -> str:
    rows = ledger.track_records()
    if not rows:
        return "(no interviews yet)"
    by_model: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_model.setdefault((r["backend"], r["model"]), []).append(r)
    lines = [
        "Model track record (backend/model: interviews, working modules, best OOF, mean latency):"
    ]
    for (backend, model), recs in sorted(by_model.items()):
        ok = sum(1 for r in recs if r["ran_ok"])
        best = max((r["oof_score"] for r in recs if r["oof_score"] is not None), default=None)
        latency = float(np.mean([r["latency"] or 0 for r in recs]))
        score_text = f"best OOF {best:.5f}" if best is not None else "no score"
        lines.append(
            f"- {backend}/{model}: {len(recs)} interviews, {ok} working, {score_text}, "
            f"{latency:.0f}s"
        )
    return "\n".join(lines)
