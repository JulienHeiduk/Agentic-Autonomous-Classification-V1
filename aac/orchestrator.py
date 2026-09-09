"""Orchestrator: the run loop (README sections 9 and 17).

Stages: data, profile, folds, the deterministic default branch (baseline and fallback),
interviews and allocation, priors and research packets, Researchers in parallel sharing a
live ensemble pool that uploads on improvement, then the final blend. Every failure is
recorded and the run continues; the budget stops new work but never the submission of what
was already earned. ``resume`` rebuilds the pool and each agent's history from the ledger
and the artifacts on disk, then continues the remaining rounds.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from aac.agents.assessor import allocate, interview, track_record_text
from aac.agents.ensembler import EnsembleResult, LivePool, Member, PoolUpdate
from aac.agents.historian import record_pitfalls, record_priors
from aac.agents.researcher import (
    Experiment,
    ResearcherOutcome,
    TeamState,
    load_history,
    render_notes,
    run_researcher,
)
from aac.agents.scholar import research, reuse_packets
from aac.agents.scout import Profile, profile_data
from aac.agents.submitter import (
    backfill_public_scores,
    upload_submission,
    write_prediction_file,
)
from aac.agents.trainer import TrainResult, train_plan
from aac.config import Config, ResearcherSpec, load_config
from aac.context import Budget, BudgetExceeded, RunContext
from aac.exec.artifacts import RunPaths, atomic_write_json
from aac.kaggle.api import CompetitionInfo, KaggleClient, KaggleError
from aac.kaggle.data import ensure_data
from aac.ledger import Ledger
from aac.llm.client import LLMError
from aac.llm.router import Router
from aac.models.cv import load_or_create_folds
from aac.models.metrics import MetricSpec, resolve_metric
from aac.plan import BAGGABLE_FAMILIES, Plan, default_plan, variant_plan

log = logging.getLogger(__name__)

DEFAULT_BRANCH_NAME = "b00-default"  # directory name under runs/{run_id}/branches/


class OrchestratorError(RuntimeError):
    """The run cannot proceed; the reason is written to the ledger."""


def branch_ledger_id(run_id: str, name: str) -> str:
    """Ledger branch ids are global (README: `aac replay --branch-id`), so they carry the run."""
    return f"{run_id}-{name}"


@dataclass
class BranchOutcome:
    """The deterministic default branch."""

    name: str
    branch_id: str
    status: str  # planned, promoted, failed
    plan: Plan | None = None
    training: TrainResult | None = None
    error: str | None = None
    duration: float = 0.0
    seed: int | None = None

    @property
    def oof_score(self) -> float | None:
        return self.training.best.oof_score if self.training else None


@dataclass
class Candidate:
    """One scored model whose test predictions could be submitted: a branch family or an
    experiment. The pool blends these; the best single one is kept for reporting."""

    name: str
    origin: str
    oof_score: float
    oof: np.ndarray
    test_pred: np.ndarray
    branch: BranchOutcome | None = None


@dataclass
class RunSummary:
    run_id: str
    run_dir: Path
    status: str
    profile: Profile | None = None
    branches: list[BranchOutcome] = field(default_factory=list)
    researchers: list[ResearcherOutcome] = field(default_factory=list)
    best: BranchOutcome | None = None
    best_candidate: Candidate | None = None
    ensemble: EnsembleResult | None = None
    uploads: int = 0
    submission_path: Path | None = None
    public_score: float | None = None
    resumed: bool = False

    @property
    def plan(self) -> Plan | None:
        return self.branches[0].plan if self.branches else None

    @property
    def training(self) -> TrainResult | None:
        return self.best.training if self.best else None


@dataclass
class _Seat:
    """One Researcher's place in the schedule: its history so far and whether it goes on."""

    index: int
    spec: ResearcherSpec
    history: list[Experiment] | None = None
    max_rounds: int = 1
    outcome: ResearcherOutcome | None = None
    duration: float = 0.0
    done: bool = False

    @property
    def next_round(self) -> int:
        return (self.history[-1].round + 1) if self.history else 1

    def wants(self, until_round: int | None) -> bool:
        if self.done or self.next_round > self.max_rounds:
            return False
        return until_round is None or self.next_round <= until_round

    def absorb(self, outcome: ResearcherOutcome | None) -> None:
        if outcome is None:  # the budget stopped it
            self.done = True
            return
        self.history = list(outcome.experiments)
        self.duration += outcome.duration
        outcome.duration = self.duration
        self.outcome = outcome
        if outcome.stopped_because:
            self.done = True


@dataclass
class RunData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    profile: Profile
    metric: MetricSpec
    y: np.ndarray
    folds: np.ndarray
    sandbox_train: Path
    sandbox_test: Path


def _stage(name: str, started: float) -> None:
    log.info("stage %-12s done in %.1fs", name, time.monotonic() - started)


def _prepare_sandbox_inputs(
    ctx: RunContext, train: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[Path, Path]:
    """Target-free parquet copies shared by every experiment's sandbox run."""
    train_path = ctx.paths.run_dir / "sandbox_train.parquet"
    test_path = ctx.paths.run_dir / "sandbox_test.parquet"
    if not train_path.exists():
        tmp = train_path.with_name(train_path.name + ".part")
        train.drop(columns=[target]).to_parquet(tmp, index=False)
        tmp.replace(train_path)
    if not test_path.exists():
        tmp = test_path.with_name(test_path.name + ".part")
        test.to_parquet(tmp, index=False)
        tmp.replace(test_path)
    return train_path, test_path


def run_plan_branch(
    ctx: RunContext,
    data: RunData,
    *,
    name: str,
    plan: Plan,
    dry_run: bool = False,
    seed: int | None = None,
    families: list[str] | None = None,
) -> BranchOutcome:
    """Train one deterministic plan as a branch; every family becomes a pool member."""
    config = ctx.config
    started = time.monotonic()
    seed = config.run.seed if seed is None else seed
    branch_id = branch_ledger_id(ctx.run_id, name)
    branch_dir = ctx.paths.branch_dir(name)
    branch_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(branch_dir / "plan.json", plan.model_dump(mode="json"))
    ctx.ledger.create_branch(
        branch_id,
        ctx.run_id,
        backend="none",
        model=f"{plan.name}-plan",
        plan=plan.model_dump(mode="json"),
        plan_hash=plan.hash(),
        status="training",
    )
    outcome = BranchOutcome(name, branch_id, "planned", plan=plan, seed=seed)
    if dry_run:
        return outcome
    try:
        result = train_plan(
            plan,
            data.profile,
            data.train,
            data.test,
            data.y,
            data.folds,
            metric=data.metric,
            seed=seed,
            n_jobs=config.run.n_jobs,
            branch_dir=branch_dir,
            max_trees=config.run.max_trees,
            families=families,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, the run continues
        log.exception("branch %s failed", name)
        outcome.status, outcome.error = "failed", f"{exc.__class__.__name__}: {exc}"[:500]
        ctx.ledger.update_branch(branch_id, status="failed", error=outcome.error)
        return outcome
    best = result.best
    ctx.ledger.update_branch(
        branch_id,
        status="promoted",
        cv_mean=best.mean,
        cv_std=best.std,
        fold_scores=best.fold_scores,
        duration=result.duration,
        error=None if not result.errors else str(result.errors)[:500],
    )
    outcome.training, outcome.status = result, "promoted"
    outcome.duration = time.monotonic() - started
    return outcome


def run_default_branch(ctx: RunContext, data: RunData, *, dry_run: bool) -> BranchOutcome:
    plan = default_plan(list(ctx.config.models.enabled), data.profile.unusable_columns())
    return run_plan_branch(ctx, data, name=DEFAULT_BRANCH_NAME, plan=plan, dry_run=dry_run)


def run_variant_branches(
    ctx: RunContext, data: RunData, *, start_index: int
) -> list[BranchOutcome]:
    """The deterministic plan variants (README 17.2): levels as categoricals, target encoding."""
    cfg = ctx.config.models
    families = [f for f in cfg.variant_families if f in cfg.enabled]
    outcomes: list[BranchOutcome] = []
    for variant in cfg.variants:
        plan = variant_plan(variant, data.profile, families, cfg.low_cardinality_max)
        if plan is None:
            log.info("variant %s: nothing to apply on this data; skipped", variant)
            continue
        name = f"b{start_index + len(outcomes):02d}-{plan.name}"
        outcomes.append(run_plan_branch(ctx, data, name=name, plan=plan))
    return outcomes


def run_seed_bags(
    ctx: RunContext, data: RunData, branches: list[BranchOutcome], *, start_index: int
) -> list[BranchOutcome]:
    """Retrain the best tree families with extra seeds: cheap, honest diversity for the pool."""
    cfg = ctx.config.models
    base_seed = ctx.config.run.seed
    if cfg.seed_bag == 0:
        return []
    direction = 1 if data.metric.greater_is_better else -1
    scored: list[tuple[float, BranchOutcome, str]] = []
    for b in branches:
        if b.training is None or b.plan is None or b.seed != base_seed:
            continue
        for family, r in b.training.results.items():
            if family in BAGGABLE_FAMILIES and r.duration <= cfg.seed_bag_max_seconds:
                scored.append((r.oof_score * direction, b, family))
    scored.sort(key=lambda t: t[0], reverse=True)
    by_branch: dict[str, tuple[BranchOutcome, list[str]]] = {}
    for _, b, family in scored[: cfg.seed_bag_top]:
        by_branch.setdefault(b.name, (b, []))[1].append(family)
    if not by_branch:
        log.info("seed bags: no family qualifies; skipped")
        return []
    outcomes: list[BranchOutcome] = []
    for extra in range(1, cfg.seed_bag + 1):
        seed = base_seed + extra
        for b, families in by_branch.values():
            assert b.plan is not None
            name = f"b{start_index + len(outcomes):02d}-{b.plan.name}-s{seed}"
            log.info("seed bag %s: %s of %s with seed %d", name, families, b.name, seed)
            outcomes.append(
                run_plan_branch(ctx, data, name=name, plan=b.plan, seed=seed, families=families)
            )
    return outcomes


def train_branches(ctx: RunContext, data: RunData, summary: RunSummary) -> None:
    """Default plan, its variants, then seed bags of the best tree families."""
    t = time.monotonic()
    summary.branches.append(run_default_branch(ctx, data, dry_run=False))
    _stage("default", t)
    t = time.monotonic()
    variants = run_variant_branches(ctx, data, start_index=len(summary.branches))
    summary.branches.extend(variants)
    if variants:
        _stage("variants", t)
    t = time.monotonic()
    bags = run_seed_bags(ctx, data, summary.branches, start_index=len(summary.branches))
    summary.branches.extend(bags)
    if bags:
        _stage("seed bags", t)


def load_branches(ctx: RunContext, metric: MetricSpec) -> list[Member]:
    """Members of every finished branch from its artifacts (resume)."""
    members: list[Member] = []
    if not ctx.paths.branches_dir.is_dir():
        return members
    for branch_dir in sorted(ctx.paths.branches_dir.iterdir()):
        metrics_path = branch_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        for family, summary in metrics.get("models", {}).items():
            oof_path = branch_dir / f"oof_{family}.npy"
            test_path = branch_dir / f"test_{family}.npy"
            if oof_path.exists() and test_path.exists():
                members.append(
                    Member(
                        f"{branch_dir.name}/{family}",
                        np.load(oof_path),
                        np.load(test_path),
                        float(summary["oof_score"]),
                        "branch",
                    )
                )
    return members


load_default_branch = load_branches


def candidates(
    branches: list[BranchOutcome], researchers: list[ResearcherOutcome]
) -> list[Candidate]:
    pool: list[Candidate] = []
    for b in branches:
        if b.training is None:
            continue
        for family, r in b.training.results.items():
            pool.append(
                Candidate(f"{b.name}/{family}", "branch", r.oof_score, r.oof, r.test_pred, b)
            )
    for out in researchers:
        for e in out.experiments:
            if e.ok:
                name = e.id.split("-", 2)[-1] if e.id.count("-") >= 2 else e.id
                pool.append(
                    Candidate(name, "experiment", e.oof_score, e.result.oof, e.result.test_pred)
                )
    return pool


def pick_best_candidate(pool: list[Candidate], metric: MetricSpec) -> Candidate | None:
    if not pool:
        return None
    return max(pool, key=lambda c: c.oof_score * (1 if metric.greater_is_better else -1))


def _load_data(
    ctx: RunContext, kaggle: KaggleClient, config: Config
) -> tuple[CompetitionInfo, RunData]:
    slug = config.competition.slug
    t = time.monotonic()
    info = kaggle.competition(slug)
    if info.raw.get("isKernelsSubmissionsOnly"):
        raise OrchestratorError(f"{slug} is a code competition; V1 handles file uploads only")
    metric = resolve_metric(info.metric_name, override=config.competition.metric)
    files = ensure_data(kaggle, slug, ctx.paths.data_dir(slug), config.competition.files)
    train, test, sample = files.frames()
    _stage("data", t)

    t = time.monotonic()
    profile = profile_data(
        train,
        test,
        sample,
        slug=slug,
        competition=config.competition,
        metric=metric,
        max_classes=config.run.max_classes,
        seed=config.run.seed,
    )
    atomic_write_json(ctx.paths.profile_json, profile.model_dump(mode="json"))
    for warning in profile.warnings:
        log.warning("scout: %s", warning)
    enc = profile.encoding()
    y = enc.encode(train[profile.target.name])
    folds = load_or_create_folds(ctx.paths.folds_path, y, config.run.n_folds, config.run.seed)
    sandbox_train, sandbox_test = _prepare_sandbox_inputs(ctx, train, test, profile.target.name)
    _stage("profile", t)
    return info, RunData(
        train, test, sample, profile, metric, y, folds, sandbox_train, sandbox_test
    )


def _backfill_scores(ledger: Ledger, kaggle: KaggleClient, slug: str) -> None:
    """Earlier uploads that were still scoring when their run ended get their score now."""
    try:
        updated = backfill_public_scores(ledger, kaggle, slug)
    except KaggleError as exc:
        log.warning("public score backfill skipped: %s", exc)
        return
    for row in updated:
        log.info(
            "backfilled run %s submission %s: public %.5f",
            row["run_id"],
            row["kaggle_ref"],
            row["public_score"],
        )


def _verify_backends(router: Router) -> None:
    t = time.monotonic()
    try:
        missing = router.verify_models()
    except LLMError as exc:
        raise OrchestratorError(f"LLM backend check failed: {exc}") from exc
    if missing:
        raise OrchestratorError(
            "configured models not served: "
            + "; ".join(f"{b}: {', '.join(m)}" for b, m in missing.items())
        )
    _stage("backends", t)


def _research_and_submit(
    ctx: RunContext,
    kaggle: KaggleClient,
    info: CompetitionInfo,
    data: RunData,
    router: Router,
    summary: RunSummary,
    *,
    submit: bool,
    poll_timeout: float,
    poll_interval: float,
    seed_members: list[Member],
    histories: dict[str, list[Experiment]] | None = None,
    last_uploaded: float | None = None,
) -> None:
    """Interviews, priors, packets, parallel Researchers on the live pool, final blend."""
    config = ctx.config
    metric, profile, y, folds = data.metric, data.profile, data.y, data.folds
    enc = profile.encoding()
    researchers = config.researcher_specs()
    pool = LivePool(
        y, folds, metric, seed=config.run.seed, min_improvement=config.run.min_improvement
    )
    for member in seed_members:
        pool.add(member)
    uploaded: dict[str, float | None] = {"score": last_uploaded}
    upload_lock = threading.Lock()
    team = TeamState(metric)
    slice_cols = profile.categorical_columns()
    slices = data.train[slice_cols] if slice_cols else None

    def maybe_upload(update: PoolUpdate) -> None:
        """Submit on improvement (README 17.2): the blend beat the last upload."""
        if not (submit and config.kaggle.submit) or pool.best is None:
            return
        if update.gain <= config.run.min_improvement:
            return
        with upload_lock:
            last = uploaded["score"]
            if (
                last is not None
                and metric.improvement(update.after, last) <= config.run.min_improvement
            ):
                return
            path = write_prediction_file(
                ctx.paths.run_dir / "submission.csv", pool.best.test_pred, profile, data.sample, enc
            )
            description = (
                f"aac run {ctx.run_id}: {pool.best.method} of {pool.best.n_members} after "
                f"{update.member}, OOF {metric.key} {update.after:.5f}"
            )
            try:
                final = upload_submission(
                    ctx,
                    kaggle,
                    info,
                    path,
                    description=description,
                    oof_score=update.after,
                    poll_timeout=poll_timeout,
                    poll_interval=poll_interval,
                )
            except BudgetExceeded as exc:
                log.warning("no upload: %s", exc)
                return
            uploaded["score"] = update.after
            summary.uploads += 1
            summary.public_score = final.public_score
            summary.submission_path = path

    if researchers and config.assessor.enabled:
        t = time.monotonic()
        try:
            ctx.budget.check()
            interview(
                router,
                ctx.ledger,
                profile,
                metric,
                config=config.assessor,
                candidates=sorted({(r.backend, r.model) for r in researchers}),
                train=data.train,
                test=data.test,
                workdir=ctx.paths.run_dir / "interview",
                seed=config.run.seed,
                n_threads=config.run.n_jobs,
                degenerate_margin=config.run.degenerate_margin,
            )
        except BudgetExceeded as exc:
            log.warning("interviews skipped: %s", exc)
        researchers, dropped = allocate(researchers, ctx.ledger, config.assessor)
        for reason in dropped:
            log.warning("assessor: skipping %s", reason)
        ctx.ledger.add_note(
            source="assessor",
            kind="track_record",
            run_id=ctx.run_id,
            text=track_record_text(ctx.ledger)
            + ("\nSkipped: " + "; ".join(dropped) if dropped else ""),
        )
        _stage("interviews", t)

    if researchers:
        t = time.monotonic()
        tracks = sorted({r.track for r in researchers})
        if not ctx.ledger.list_notes(ctx.run_id, kind="prior"):
            record_priors(
                ctx.ledger, ctx.paths.root, config.competition.slug, metric, ctx.run_id, tracks
            )
        if not ctx.ledger.list_notes(ctx.run_id, kind="pitfalls"):
            record_pitfalls(ctx.ledger, config.competition.slug, ctx.run_id, tracks)
        scholar = config.scholar_spec()
        if scholar is not None and not ctx.ledger.list_notes(ctx.run_id, kind="research"):
            reuse_packets(
                ctx.ledger, config.competition.slug, ctx.run_id, max_age=scholar.reuse_runs
            )
        if scholar is not None and not ctx.ledger.list_notes(ctx.run_id, kind="research"):
            try:
                ctx.budget.check()
                research(
                    router,
                    profile,
                    config=scholar,
                    metric=metric,
                    competition_text=f"{info.title}. {info.raw.get('description') or ''}",
                    leaderboard=pool.leaderboard(),
                    ledger=ctx.ledger,
                    run_id=ctx.run_id,
                )
            except BudgetExceeded as exc:
                log.warning("scholar skipped: %s", exc)
        _stage("scholar", t)

        t = time.monotonic()
        budget_hit: list[BudgetExceeded] = []

        def worker(seat: _Seat, until_round: int | None) -> ResearcherOutcome | None:
            spec = seat.spec
            try:
                return run_researcher(
                    ctx,
                    router,
                    spec=spec,
                    index=seat.index,
                    profile=profile,
                    metric=metric,
                    y=y,
                    folds=folds,
                    sandbox_train=data.sandbox_train,
                    sandbox_test=data.sandbox_test,
                    pool=pool,
                    after_experiment=maybe_upload,
                    notes=lambda: render_notes(
                        ctx.ledger.list_notes(ctx.run_id, track=spec.track), track=spec.track
                    ),
                    team=team,
                    slices=slices,
                    history=seat.history,
                    until_round=until_round,
                )
            except BudgetExceeded as exc:
                budget_hit.append(exc)
                return None

        seats = [
            _Seat(i, spec, (histories or {}).get(f"r{i:02d}-{spec.track}-{spec.backend}"))
            for i, spec in enumerate(researchers, start=1)
        ]
        for seat in seats:
            seat.max_rounds = seat.spec.rounds or config.run.max_rounds
        # Round-robin: every seat plays round n before any seat plays round n+1, so a slow or
        # failing seat cannot starve the others of the wall clock. Sequential: one pass.
        passes: list[int | None] = (
            [None]
            if config.run.schedule == "sequential"
            else list(range(1, max(s.max_rounds for s in seats) + 1))
        )
        workers = max(1, min(config.run.parallel_branches, len(researchers)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="researcher") as pool_exec:
            for until in passes:
                if budget_hit:
                    break
                active = [s for s in seats if s.wants(until)]
                if not active:
                    # nobody plays this round (resumed seats are past it); stop only when no
                    # seat can play any later round either
                    if all(s.done or s.next_round > s.max_rounds for s in seats):
                        break
                    continue
                futures = [(s, pool_exec.submit(worker, s, until)) for s in active]
                for seat, future in futures:
                    seat.absorb(future.result())
        summary.researchers.extend(s.outcome for s in seats if s.outcome is not None)
        if budget_hit:
            # The budget stops new work, never the submission of what was earned.
            log.warning("stopping new work: %s", budget_hit[0])
            ctx.ledger.add_note(
                source="orchestrator", kind="budget", text=str(budget_hit[0]), run_id=ctx.run_id
            )
        _stage(f"researchers x{len(researchers)}", t)

    if not pool.members:
        raise OrchestratorError("every branch and experiment failed; nothing to submit")
    # Best single over the whole pool (reloaded default families included on resume).
    top = max(pool.members, key=lambda m: m.oof_score * (1 if metric.greater_is_better else -1))
    best_candidate = Candidate(top.name, top.origin, top.oof_score, top.oof, top.test_pred)
    for c in candidates(summary.branches, summary.researchers):
        if c.name == top.name:
            best_candidate = c
            break
    summary.best_candidate = best_candidate
    summary.best = best_candidate.branch

    t = time.monotonic()
    result = pool.result
    assert result is not None
    summary.ensemble = result
    atomic_write_json(ctx.paths.run_dir / "ensemble.json", result.summary())
    ctx.ledger.add_note(
        source="ensembler",
        kind="ensemble",
        run_id=ctx.run_id,
        text=(
            f"{result.best.method} OOF {metric.key} {result.best.oof_score:.5f} over "
            f"{len(pool.members)} members; best single {best_candidate.oof_score:.5f}"
        ),
    )
    chosen_score = result.best.oof_score
    chosen_name = f"{result.best.method} of {result.best.n_members}"
    _stage("ensemble", t)

    t = time.monotonic()
    path = write_prediction_file(
        ctx.paths.run_dir / "submission.csv", result.best.test_pred, profile, data.sample, enc
    )
    summary.submission_path = path
    _stage("submission", t)

    last = uploaded["score"]
    already = (
        last is not None and metric.improvement(chosen_score, last) <= config.run.min_improvement
    )
    if submit and config.kaggle.submit and not already:
        description = (
            f"aac run {ctx.run_id}: {chosen_name} (best single {best_candidate.name}), "
            f"OOF {metric.key} {chosen_score:.5f}"
        )
        try:
            final = upload_submission(
                ctx,
                kaggle,
                info,
                path,
                description=description,
                oof_score=chosen_score,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
            )
        except BudgetExceeded as exc:
            # The file is on disk; `aac submit --run-id` uploads it once the quota resets.
            log.warning("final upload skipped: %s", exc)
            ctx.ledger.add_note(
                source="submitter",
                kind="budget",
                run_id=ctx.run_id,
                text=f"final upload skipped: {exc}; submission.csv kept for aac submit",
            )
        else:
            summary.uploads += 1
            summary.public_score = final.public_score
    elif already:
        log.info("final blend already uploaded (OOF %.5f); no further upload", last)


def run(
    config: Config,
    *,
    submit: bool = True,
    dry_run: bool = False,
    runs_root: Path = Path("runs"),
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_timeout: float = 600.0,
    poll_interval: float = 15.0,
    console: Console | None = None,
) -> RunSummary:
    console = console or Console()
    http = httpx.Client(transport=transport) if transport is not None else None
    ctx = RunContext.create(config, runs_root)
    summary = RunSummary(ctx.run_id, ctx.paths.run_dir, "running")
    metric_key = "score"
    try:
        with KaggleClient.from_env(client=http, sleep=sleep) as kaggle:
            router = Router(
                config,
                ledger=ctx.ledger,
                budget=ctx.budget,
                run_id=ctx.run_id,
                client=http,
                sleep=sleep,
            )
            info, data = _load_data(ctx, kaggle, config)
            _backfill_scores(ctx.ledger, kaggle, config.competition.slug)
            metric_key = data.metric.key
            summary.profile = data.profile
            ctx.ledger.add_note(
                source="kaggle",
                kind="competition",
                run_id=ctx.run_id,
                text=(
                    f"{info.title}. {info.raw.get('description') or ''} Metric: "
                    f"{info.metric_name}. Deadline: {info.deadline}."
                ).strip(),
            )
            if config.researcher_specs():
                _verify_backends(router)

            if dry_run:
                t = time.monotonic()
                summary.branches.append(run_default_branch(ctx, data, dry_run=True))
                _stage("default", t)
                ctx.finish("completed")
                summary.status = "completed"
                _print_summary(console, summary, metric_key)
                return summary
            train_branches(ctx, data, summary)

            seed_members = [
                Member(c.name, c.oof, c.test_pred, c.oof_score, c.origin)
                for c in candidates(summary.branches, [])
            ]
            _research_and_submit(
                ctx,
                kaggle,
                info,
                data,
                router,
                summary,
                submit=submit,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
                seed_members=seed_members,
            )
        ctx.finish("completed")
        summary.status = "completed"
    except Exception as exc:
        ctx.finish("failed", error=f"{exc.__class__.__name__}: {exc}"[:500])
        summary.status = "failed"
        raise
    finally:
        ctx.close()
        if http is not None:
            http.close()
    _print_summary(console, summary, metric_key)
    return summary


def resume(
    run_id: str,
    *,
    config: Config | None = None,
    submit: bool = True,
    runs_root: Path = Path("runs"),
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_timeout: float = 600.0,
    poll_interval: float = 15.0,
    console: Console | None = None,
) -> RunSummary:
    """Continue a crashed or stopped run: pool and histories from the ledger and disk."""
    console = console or Console()
    paths = RunPaths(Path(runs_root), run_id)
    ledger = Ledger(paths.ledger_path)
    row = ledger.get_run(run_id)
    if row is None:
        raise OrchestratorError(f"run {run_id} is not in the ledger")
    if config is None:
        if not row.get("config_path") or not Path(row["config_path"]).is_file():
            raise OrchestratorError(
                f"run {run_id}: config file {row.get('config_path')!r} is gone; pass --config"
            )
        config = load_config(row["config_path"])
    if config.fingerprint() != row["config_hash"]:
        log.warning(
            "resume %s: the config changed since the run started (hash %s vs %s)",
            run_id,
            config.fingerprint(),
            row["config_hash"],
        )
    if not paths.run_dir.is_dir():
        raise OrchestratorError(f"run directory {paths.run_dir} is gone")
    budget = Budget(
        max_tokens=config.run.max_tokens_total, max_seconds=config.run.max_wall_clock_minutes * 60
    )
    prompt_tokens, completion_tokens = ledger.token_totals(run_id)
    budget.charge(prompt_tokens, completion_tokens)
    ctx = RunContext(config, paths, ledger, budget)
    ledger.set_run_status(run_id, "running")
    summary = RunSummary(run_id, paths.run_dir, "running", resumed=True)
    http = httpx.Client(transport=transport) if transport is not None else None
    metric_key = "score"
    try:
        with KaggleClient.from_env(client=http, sleep=sleep) as kaggle:
            router = Router(
                config, ledger=ledger, budget=budget, run_id=run_id, client=http, sleep=sleep
            )
            info, data = _load_data(ctx, kaggle, config)
            _backfill_scores(ctx.ledger, kaggle, config.competition.slug)
            metric_key = data.metric.key
            summary.profile = data.profile
            if config.researcher_specs():
                _verify_backends(router)

            seed_members = load_branches(ctx, data.metric)
            if not seed_members:
                train_branches(ctx, data, summary)
                seed_members = [
                    Member(c.name, c.oof, c.test_pred, c.oof_score, c.origin)
                    for c in candidates(summary.branches, [])
                ]
            else:
                log.info("resume: %d branch members reloaded", len(seed_members))

            histories: dict[str, list[Experiment]] = {}
            for agent in {r["agent"] for r in ledger.list_experiments(run_id)}:
                histories[agent] = load_history(
                    paths.run_dir, ledger.list_experiments(run_id, agent), data.metric
                )
            for history in histories.values():
                for e in history:
                    if e.ok:
                        seed_members.append(
                            Member(
                                e.id.split("-", 2)[-1],
                                e.result.oof,
                                e.result.test_pred,
                                e.oof_score,
                                "experiment",
                            )
                        )
            uploads = ledger.list_submissions(run_id)
            last_uploaded = max(
                (u["oof_score"] for u in uploads if u["oof_score"] is not None), default=None
            )
            log.info(
                "resume: %d experiments across %d agents reloaded, last uploaded OOF %s",
                sum(len(h) for h in histories.values()),
                len(histories),
                last_uploaded,
            )
            _research_and_submit(
                ctx,
                kaggle,
                info,
                data,
                router,
                summary,
                submit=submit,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
                seed_members=seed_members,
                histories=histories,
                last_uploaded=last_uploaded,
            )
        ctx.finish("completed")
        summary.status = "completed"
    except Exception as exc:
        ctx.finish("failed", error=f"{exc.__class__.__name__}: {exc}"[:500])
        summary.status = "failed"
        raise
    finally:
        ctx.close()
        if http is not None:
            http.close()
    _print_summary(console, summary, metric_key)
    return summary


def submit_run(
    config: Config,
    run_id: str,
    *,
    runs_root: Path = Path("runs"),
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_timeout: float = 600.0,
    poll_interval: float = 15.0,
) -> float | None:
    """Upload a finished run's ``submission.csv`` (for example after the daily quota reset)."""
    paths = RunPaths(Path(runs_root), run_id)
    path = paths.run_dir / "submission.csv"
    if not path.is_file():
        raise OrchestratorError(f"{path} does not exist")
    oof_score = None
    ensemble_json = paths.run_dir / "ensemble.json"
    if ensemble_json.is_file():
        oof_score = json.loads(ensemble_json.read_text()).get("best", {}).get("oof_score")
    http = httpx.Client(transport=transport) if transport is not None else None
    ledger = Ledger(paths.ledger_path)
    if ledger.get_run(run_id) is None:
        raise OrchestratorError(f"run {run_id} is not in the ledger")
    budget = Budget(
        max_tokens=config.run.max_tokens_total, max_seconds=config.run.max_wall_clock_minutes * 60
    )
    ctx = RunContext(config, paths, ledger, budget)
    try:
        with KaggleClient.from_env(client=http, sleep=sleep) as kaggle:
            info = kaggle.competition(config.competition.slug)
            _backfill_scores(ledger, kaggle, config.competition.slug)
            score_text = f"{oof_score:.5f}" if oof_score is not None else "unknown"
            final = upload_submission(
                ctx,
                kaggle,
                info,
                path,
                description=f"aac submit {run_id}: stored blend, OOF {score_text}",
                oof_score=oof_score,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
            )
    finally:
        ledger.close()
        if http is not None:
            http.close()
    return final.public_score


def _print_summary(console: Console, s: RunSummary, metric: str) -> None:
    table = Table(title=f"aac {'resume' if s.resumed else 'run'} {s.run_id}")
    table.add_column("field", style="bold")
    table.add_column("value")
    rows: list[tuple[str, str]] = [("status", s.status), ("run dir", str(s.run_dir))]
    if s.profile:
        p = s.profile
        rows += [
            ("data", f"{p.n_train} train / {p.n_test} test rows, {p.n_columns} columns"),
            ("target", f"{p.target.name} ({p.target.kind}, positive={p.target.positive_label!r})"),
            ("submission", f"{p.submission.kind} in {p.submission.columns}"),
        ]
    for b in s.branches:
        if b.training:
            r = b.training.best
            rows.append(
                (
                    b.name,
                    f"{b.plan.name if b.plan else '?'} via deterministic: {r.family} OOF {metric} "
                    f"{r.oof_score:.5f} (folds {r.mean:.5f} +/- {r.std:.5f}); "
                    f"{len(b.training.features)} features",
                )
            )
            for family, err in b.training.errors.items():
                rows.append((f"  {family}", f"FAILED: {err[:80]}"))
        elif b.status == "planned":
            rows.append((b.name, f"{b.plan.name if b.plan else '?'}: planned (dry run)"))
        else:
            rows.append((b.name, f"FAILED: {(b.error or '')[:100]}"))
    for out in s.researchers:
        best = (
            f"best OOF {metric} {out.best.oof_score:.5f} (round {out.best.round})"
            if out.best
            else "no working experiment"
        )
        rows.append(
            (
                out.agent,
                f"{out.spec.model} on {out.spec.track}: {out.n_ok}/{len(out.experiments)} "
                f"experiments ok, {best}; {out.stopped_because}; {out.duration:.0f}s",
            )
        )
    if s.best_candidate:
        c = s.best_candidate
        rows.append(("best single", f"{c.name} ({c.origin}) OOF {metric} {c.oof_score:.5f}"))
    if s.ensemble:
        b = s.ensemble.best
        others = ", ".join(f"{x.method} {x.oof_score:.5f}" for x in s.ensemble.blends)
        rows.append(
            (
                "ensemble",
                f"{b.method} of {b.n_members} members OOF {metric} {b.oof_score:.5f} ({others})",
            )
        )
    if s.submission_path:
        rows.append(("submission file", str(s.submission_path)))
    if s.uploads:
        rows.append(("uploads", str(s.uploads)))
    if s.public_score is not None:
        rows.append(("public score", f"{s.public_score:.5f}"))
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)


def replay_experiment(
    experiment_id: str,
    *,
    config: Config | None = None,
    runs_root: Path = Path("runs"),
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Re-execute a stored experiment's module through the harness and compare to its OOF.

    README section 10: the same module on the same folds must reproduce to 1e-9.
    """
    from aac.agents.historian import experiment_module_path
    from aac.exec.sandbox import run_experiment

    runs_root = Path(runs_root)
    ledger = Ledger(RunPaths(runs_root, "_").ledger_path)
    try:
        rows = [r for r in ledger.list_experiments_by_id(experiment_id)]
    except AttributeError:
        rows = []
    if not rows:
        rows = [
            r
            for run in ledger.list_runs()
            for r in ledger.list_experiments(run["id"])
            if r["id"] == experiment_id
        ]
    if not rows:
        ledger.close()
        raise OrchestratorError(f"experiment {experiment_id} is not in the ledger")
    row = rows[0]
    run_id = row["run_id"]
    run_row = ledger.get_run(run_id)
    if config is None:
        if (
            not run_row
            or not run_row.get("config_path")
            or not Path(run_row["config_path"]).is_file()
        ):
            ledger.close()
            raise OrchestratorError("the run's config file is gone; pass --config")
        config = load_config(run_row["config_path"])
    module = experiment_module_path(runs_root, experiment_id, run_id)
    if module is None:
        ledger.close()
        raise OrchestratorError(f"module for {experiment_id} not found on disk")
    paths = RunPaths(runs_root, run_id)
    stored_oof = paths.run_dir / "experiments" / row["agent"] / f"r{row['round']}" / "oof.npy"
    budget = Budget(
        max_tokens=config.run.max_tokens_total, max_seconds=config.run.max_wall_clock_minutes * 60
    )
    ctx = RunContext(config, paths, ledger, budget)
    http = httpx.Client(transport=transport) if transport is not None else None
    try:
        with KaggleClient.from_env(client=http, sleep=sleep) as kaggle:
            _, data = _load_data(ctx, kaggle, config)
        result = run_experiment(
            module.read_text(),
            workdir=paths.run_dir / "replay" / experiment_id,
            train_path=data.sandbox_train,
            test_path=data.sandbox_test,
            y=data.y,
            folds=data.folds,
            metric=data.metric,
            target=data.profile.target.name,
            id_col=data.profile.id_col,
            n_classes=len(data.profile.target.classes),
            categorical=data.profile.categorical_columns(),
            drop_columns=data.profile.unusable_columns(),
            seed=config.run.seed,
            timeout=config.sandbox.experiment_timeout_seconds,
            memory_mb=config.sandbox.memory_mb,
            n_threads=config.run.n_jobs,
            determinism_rows=config.sandbox.determinism_rows,
        )
    finally:
        ledger.close()
        if http is not None:
            http.close()
    report: dict[str, object] = {
        "experiment_id": experiment_id,
        "ok": result.ok,
        "kind": result.kind,
        "oof_score": result.oof_score,
        "stored_oof_score": row.get("oof_score"),
    }
    if result.ok and stored_oof.exists():
        diff = float(np.abs(np.load(stored_oof) - result.oof).max())
        report["max_abs_diff"] = diff
        report["reproduced"] = diff <= 1e-9
    elif not result.ok:
        report["error"] = result.error[-500:]
    return report
