"""Researcher: LLM, one agent per (backend, model, track) (README section 17.2).

Each round: read the knowledge base (own history, the track leaderboard, notes), state a
hypothesis, write the experiment module, run it in the fold-owning harness, record the
result. Stops when its rounds are spent, after ``patience`` rounds without improvement, or
when the run budget is exhausted.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aac.agents.analyst import analyse
from aac.agents.ensembler import LivePool, Member, PoolUpdate
from aac.agents.scout import Profile
from aac.config import ResearcherSpec
from aac.context import BudgetExceeded, RunContext
from aac.exec.artifacts import atomic_write_json, atomic_write_text
from aac.exec.sandbox import EXPERIMENT_IMPORTS, ExperimentResult, run_experiment
from aac.llm.client import LLMError
from aac.llm.prompts import load_prompt
from aac.llm.router import Router
from aac.llm.schema import extract_code
from aac.models.metrics import MetricSpec, score

log = logging.getLogger(__name__)

LARGE_TABLE_ROWS = 300_000


def size_hint(profile: Profile) -> str:
    if profile.n_train >= LARGE_TABLE_ROWS:
        return (
            f"large table ({profile.n_train} rows): prefer lightgbm, xgboost, and hist_gbdt; "
            "catboost costs roughly 20x the time for the same score here; keep tree counts "
            "at or below 1500 with learning rates of 0.03 to 0.08."
        )
    return (
        f"small table ({profile.n_train} rows): any family is affordable; catboost is often "
        "the strongest single model; more trees with a lower learning rate is fine."
    )


def render_columns(profile: Profile) -> str:
    lines = []
    for c in profile.columns:
        if c.name == profile.id_col or not c.usable:
            continue
        if c.stats:
            detail = (
                f"range [{c.stats.get('min', 0):.4g}, {c.stats.get('max', 0):.4g}], "
                f"mean {c.stats.get('mean', 0):.4g}"
            )
        elif c.top_values:
            detail = "top: " + ", ".join(f"{v} ({s:.0%})" for v, s in c.top_values[:5])
        else:
            detail = ""
        drift = "" if c.drift is None else f"{c.drift:.3f}"
        lines.append(
            f"{c.name} | {c.kind} | {c.n_unique} | {100 * c.missing_train:.1f} | {drift} | {detail}"
        )
    return "\n".join(lines) if lines else "(none)"


TRACK_GUIDANCE = {
    "gbdt": "gradient boosting only: lightgbm, xgboost, catboost, or sklearn "
    "HistGradientBoosting; "
    "explore features, tree shape, learning rate with more trees, categorical handling, "
    "seed or fold-internal bagging.",
    "linear": "linear models only: logistic regression or SGD with careful preprocessing: "
    "imputation, "
    "scaling, one-hot or frequency encoding, interactions, spline or binning features.",
    "neural": "neural networks with torch: MLPs on standardised numerics plus embeddings for "
    "categoricals; seed with torch.manual_seed(meta['seed']); keep epochs small enough for the "
    "time limit; use an inner validation split of the training fold for early stopping.",
    "open": "anything: any family, feature engineering, or a blend of several models fitted inside "
    "fit_predict; pick what the evidence says is strongest.",
}
_HYPOTHESIS = re.compile(r"HYPOTHESIS:\s*(.+)")

BOOSTING = frozenset({"lightgbm", "xgboost", "catboost"})
TRACK_RULES: dict[str, tuple[frozenset[str], frozenset[str], str]] = {
    # track: (one of these must be imported, none of these may be imported, description)
    "gbdt": (BOOSTING | {"sklearn.ensemble"}, frozenset({"torch"}), "a gradient boosting library"),
    "linear": (
        frozenset({"sklearn.linear_model"}),
        BOOSTING | {"torch", "sklearn.ensemble"},
        "sklearn.linear_model only",
    ),
    "neural": (frozenset({"torch"}), BOOSTING, "torch"),
    "open": (frozenset(), frozenset(), "anything"),
}


def imported_modules(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.add(node.module.split(".")[0])
            for alias in node.names:
                names.add(f"{node.module}.{alias.name}")
    return names


def forbidden_imports(code: str, track: str) -> list[str]:
    """Libraries the module imports that its track forbids."""
    _, forbidden, _ = TRACK_RULES[track]
    imports = imported_modules(code)
    return sorted(r for r in forbidden if any(i == r or i.startswith(r + ".") for i in imports))


def check_track(code: str, track: str) -> list[str]:
    """Static track enforcement: the module must use its track's libraries and no others."""
    required, _, description = TRACK_RULES[track]
    imports = imported_modules(code)
    problems = []
    if required and not any(
        any(i == r or i.startswith(r + ".") for i in imports) for r in required
    ):
        problems.append(
            f"track {track!r} requires {description}; none of {sorted(required)} is imported"
        )
    bad = forbidden_imports(code, track)
    if bad:
        problems.append(f"track {track!r} forbids {bad}")
    return problems


# Words that mark an idea as belonging to a forbidden library, per forbidden import.
OFF_TRACK_WORDS: dict[str, tuple[str, ...]] = {
    "lightgbm": ("lightgbm", "lgbm", "lgb"),
    "xgboost": ("xgboost", "xgb"),
    "catboost": ("catboost",),
    "torch": ("torch", "pytorch", "mlp", "neural", "tabnet", "embeddings?", "realmlp"),
    "sklearn.ensemble": (
        r"histgradientboosting\w*",
        "hist_gbdt",
        "random ?forest",
        "extra ?trees",
        "gradient boosting",
        "gbdt",
        "boosted trees",
    ),
}
_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)


def strip_off_track(text: str, track: str) -> str:
    """Remove from a note what a Researcher on ``track`` must not copy: fenced modules that
    import a forbidden library, and bullet ideas that are about one (README 17.3)."""
    required, forbidden, _ = TRACK_RULES[track]
    if not required and not forbidden:
        return text
    words = [w for lib in forbidden for w in OFF_TRACK_WORDS.get(lib, (lib,))]
    pattern = re.compile(r"(?<![a-z0-9_])(?:" + "|".join(words) + r")(?![a-z0-9_])", re.I)
    omitted = 0

    def replace_block(match: re.Match[str]) -> str:
        nonlocal omitted
        problems = check_track(match.group(1), track)
        if not problems:
            return match.group(0)
        omitted += 1
        return f"(module omitted, outside your track: {'; '.join(problems)})"

    text = _FENCE.sub(replace_block, text)
    kept: list[str] = []
    dropped = 0
    for line in text.splitlines():
        if words and line.lstrip().startswith("- ") and pattern.search(line):
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        kept.append(f"({dropped} ideas about libraries outside the {track!r} track omitted)")
    return "\n".join(kept)


class TeamState:
    """Best experiment per agent, shared across parallel Researchers for tip sharing."""

    def __init__(self, metric: MetricSpec) -> None:
        self.metric = metric
        self._best: dict[str, Experiment] = {}
        self._lock = threading.Lock()

    def update(self, experiment: Experiment) -> None:
        with self._lock:
            current = self._best.get(experiment.agent)
            if current is None or self.metric.better(experiment.oof_score, current.oof_score):
                self._best[experiment.agent] = experiment

    def leader(self, exclude: str | None = None) -> Experiment | None:
        with self._lock:
            pool = [e for a, e in self._best.items() if a != exclude]
        if not pool:
            return None
        return max(pool, key=lambda e: e.oof_score * (1 if self.metric.greater_is_better else -1))

    def leaderboard(self) -> str:
        with self._lock:
            rows = sorted(
                self._best.values(),
                key=lambda e: e.oof_score * (1 if self.metric.greater_is_better else -1),
                reverse=True,
            )
        if not rows:
            return "Leaderboard of other Researchers: (no results yet)"
        lines = ["Leaderboard of Researchers (agent | best OOF | round | hypothesis):"]
        lines += [f"{e.agent} | {e.oof_score:.5f} | {e.round} | {e.hypothesis[:90]}" for e in rows]
        return "\n".join(lines)

    def tip_for(
        self,
        agent: str,
        own_best: Experiment | None,
        max_chars: int = 7000,
        track: str | None = None,
    ) -> str:
        """The leader's experiment for a trailing agent (README 17.2, tip sharing). A module
        outside the receiver's track is shared as its hypothesis only, never as code."""
        leader = self.leader(exclude=agent)
        if leader is None:
            return ""
        if own_best is not None and not self.metric.better(leader.oof_score, own_best.oof_score):
            return ""
        if track is not None and (problems := check_track(leader.code, track)):
            return (
                f"Tip from the leading Researcher {leader.agent} (OOF {leader.oof_score:.5f}, "
                f"hypothesis: {leader.hypothesis}). Its module is outside your track "
                f"({'; '.join(problems)}), so borrow the idea (features, preprocessing, "
                "validation), not the code."
            )
        code = (
            leader.code
            if len(leader.code) <= max_chars
            else leader.code[:max_chars] + "\n# ... truncated"
        )
        return (
            f"Tip from the leading Researcher {leader.agent} (OOF {leader.oof_score:.5f}, "
            f"hypothesis: {leader.hypothesis}). Study it, borrow what applies to your track, "
            f"and try to beat it:\n```python\n{code}\n```"
        )


API_NOTES = [
    "lightgbm: LGBMClassifier.fit(X, y, eval_X=X_es, eval_y=y_es, "
    "callbacks=[lgb.early_stopping(50, "
    "verbose=False), lgb.log_evaluation(0)]). fit() has NO early_stopping_rounds or verbose "
    "arguments; eval_set is deprecated. Constructor: verbose=-1, n_jobs=meta['n_threads'], "
    "random_state=meta['seed'], deterministic=True, force_row_wise=True. Pandas category columns "
    "are used natively.",
    "xgboost: XGBClassifier(tree_method='hist', enable_categorical=True, early_stopping_rounds=50, "
    "n_jobs=..., random_state=...) then fit(X, y, eval_set=[(X_es, y_es)], verbose=False); "
    "early_stopping_rounds belongs in the constructor, not in fit().",
    "catboost: CatBoostClassifier(random_seed=..., thread_count=..., verbose=0, "
    "allow_writing_files=False); fit(X, y, cat_features=[names], eval_set=(X_es, y_es), "
    "early_stopping_rounds=50). Cast category columns to str first; CatBoost rejects NaN in them.",
    "sklearn: HistGradientBoostingClassifier(categorical_features='from_dtype', "
    "random_state=...). LogisticRegression needs imputation, scaling, and one-hot or codes for "
    "categoricals.",
    "torch: torch.manual_seed(meta['seed']); torch.set_num_threads(meta['n_threads']); CPU is "
    "fine; no downloads.",
    "y_train is a numpy int64 array (no .values, no .to_numpy()); X_* are DataFrames "
    "(use .to_numpy() or column selection).",
    "pandas 3: string columns have dtype 'str'; use .astype(str) before .str methods; "
    "boolean-like object columns hold True/False/None: use .map({True: 1, False: 0}).",
]


def environment_card() -> str:
    """Installed versions plus the calling conventions models most often get wrong."""
    from importlib import metadata

    versions = []
    for name in (
        "python",
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
        "torch",
        "scipy",
    ):
        if name == "python":
            import sys

            versions.append(f"python {sys.version_info.major}.{sys.version_info.minor}")
            continue
        try:
            versions.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            continue
    return ", ".join(versions) + "\n" + "\n".join(f"- {note}" for note in API_NOTES)


@dataclass
class Experiment:
    id: str
    agent: str
    round: int
    hypothesis: str
    code: str
    backend: str
    model: str
    result: ExperimentResult | None = None
    raw: str = ""
    pool_gain: float | None = None
    analysis: str = ""

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok

    @property
    def oof_score(self) -> float | None:
        return self.result.oof_score if self.ok else None


@dataclass
class ResearcherOutcome:
    agent: str
    spec: ResearcherSpec
    experiments: list[Experiment] = field(default_factory=list)
    best: Experiment | None = None
    stopped_because: str = ""
    duration: float = 0.0

    @property
    def n_ok(self) -> int:
        return sum(1 for e in self.experiments if e.ok)


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


NOTE_KINDS = ("competition", "prior", "pitfalls", "research", "tip")


def render_notes(notes: list[dict[str, object]], limit: int = 12, track: str | None = None) -> str:
    """Ledger notes for the prompt: competition text, priors, pitfalls, research packets,
    tips (README 17.3). With ``track``, ideas and modules outside it are stripped."""
    keep = [n for n in notes if n.get("kind") in NOTE_KINDS]
    if not keep:
        return "Research notes: (none)"
    lines = ["Research notes:"]
    for n in keep[-limit:]:
        text = str(n["text"]).strip()
        if track is not None:
            text = strip_off_track(text, track)
        lines.append(f"- [{n['kind']} from {n['source']}] {text[:7000]}")
    return "\n".join(lines)


def render_history(experiments: list[Experiment]) -> str:
    if not experiments:
        return "(none yet)"
    lines = []
    for e in experiments:
        r = e.result
        if e.ok:
            gain = "" if e.pool_gain is None else f" [ensemble gain {e.pool_gain:+.5f}]"
            lines.append(
                f"{e.round} | ok | {r.oof_score:.5f} | {r.cv_std:.5f} | {r.duration:.0f} | "
                f"{e.hypothesis}{gain}"
            )
        else:
            tail = r.error.strip().splitlines()[-1][:160] if r and r.error else "no module"
            kind = r.kind if r else "no-code"
            lines.append(
                f"{e.round} | {kind} | - | - | {r.duration if r else 0:.0f} | "
                f"{e.hypothesis} => {tail}"
            )
    return "\n".join(lines)


def render_best_block(best: Experiment | None, last: Experiment | None) -> str:
    parts = []
    if best is not None:
        parts.append(
            f"Your best experiment so far (round {best.round}, OOF {best.oof_score:.5f}):\n"
            f"```python\n{best.code}\n```"
        )
    if last is not None and not last.ok and last.result is not None and last.code:
        parts.append(
            f"Your last experiment (round {last.round}) failed with {last.result.kind}:\n"
            f"{last.result.error[-2500:]}\n```python\n{last.code}\n```"
        )
    return "\n\n".join(parts) if parts else "No experiment has run yet."


def researcher_messages(
    profile: Profile,
    *,
    metric: MetricSpec,
    spec: ResearcherSpec,
    experiments: list[Experiment],
    best: Experiment | None,
    round_no: int,
    max_rounds: int,
    timeout: float,
    leaderboard: str = "",
    notes: str = "",
    ensemble_text: str = "",
    analysis: str = "",
) -> list[dict[str, str]]:
    t = profile.target
    last = experiments[-1] if experiments else None
    return load_prompt("researcher").messages(
        allowed_imports=", ".join(sorted(EXPERIMENT_IMPORTS)),
        timeout=int(timeout),
        n_train=profile.n_train,
        track_guidance=TRACK_GUIDANCE[spec.track],
        environment=environment_card(),
        slug=profile.slug,
        metric=metric.display,
        direction="higher is better" if metric.greater_is_better else "lower is better",
        target_name=t.name,
        target_kind=t.kind,
        positive_label=repr(t.positive_label),
        class_rates=", ".join(f"{k}: {v:.4f}" for k, v in t.rates.items()),
        n_test=profile.n_test,
        size_hint=size_hint(profile),
        columns=render_columns(profile),
        warnings="\n".join(f"- {w}" for w in profile.warnings) or "(none)",
        history=render_history(experiments),
        best_block=render_best_block(best, last),
        leaderboard=leaderboard or "Leaderboard of other Researchers: (not available yet)",
        ensemble=ensemble_text or "Ensemble so far: (empty)",
        analysis=analysis or "Analyst notes: (none yet)",
        notes=notes or "Research notes: (none)",
        round=round_no,
        max_rounds=max_rounds,
    )


def parse_reply(text: str) -> tuple[str, str | None]:
    match = _HYPOTHESIS.search(text or "")
    hypothesis = match.group(1).strip() if match else ""
    code = extract_code(text or "", required="fit_predict")
    if not hypothesis and code:
        first = code.strip().splitlines()[0]
        hypothesis = (
            first.lstrip("# ").strip() if first.startswith("#") else "(no hypothesis given)"
        )
    return hypothesis[:300], code


def propose(
    router: Router,
    messages: list[dict[str, str]],
    *,
    spec: ResearcherSpec,
    agent: str,
    branch_id: str,
) -> tuple[str, str | None, str, str, str]:
    """Returns (hypothesis, code, raw, backend, model). Code is None when no module came back."""
    try:
        completion = router.complete(
            messages,
            agent="researcher",
            backend=spec.backend,
            model=spec.model,
            branch_id=branch_id,
            temperature=spec.temperature,
            max_tokens=8000,
        )
    except LLMError as exc:
        return "", None, f"LLM error: {exc}", spec.backend, spec.model or ""
    hypothesis, code = parse_reply(completion.content)
    if code is None:
        retry = [
            *messages,
            {"role": "assistant", "content": completion.content},
            *load_prompt("code_repair").messages(),
        ]
        try:
            completion = router.complete(
                retry,
                agent="researcher:repair",
                backend=spec.backend,
                model=spec.model,
                branch_id=branch_id,
                temperature=0.0,
                max_tokens=8000,
            )
        except LLMError as exc:
            return hypothesis, None, f"LLM error on repair: {exc}", spec.backend, spec.model or ""
        hypothesis2, code = parse_reply(completion.content)
        hypothesis = hypothesis or hypothesis2
    return hypothesis, code, completion.content, completion.backend, completion.model


def run_researcher(
    ctx: RunContext,
    router: Router,
    *,
    spec: ResearcherSpec,
    index: int,
    profile: Profile,
    metric: MetricSpec,
    y,
    folds,
    sandbox_train: Path,
    sandbox_test: Path,
    leaderboard: Callable[[], str] | None = None,
    notes: Callable[[], str] | None = None,
    pool: LivePool | None = None,
    after_experiment: Callable[[PoolUpdate], None] | None = None,
    team: TeamState | None = None,
    slices: pd.DataFrame | None = None,
    history: list[Experiment] | None = None,
) -> ResearcherOutcome:
    """The loop for one Researcher. Failures are recorded; BudgetExceeded propagates."""
    config = ctx.config
    started = time.monotonic()
    agent = f"r{index:02d}-{spec.track}-{spec.backend}"
    outcome = ResearcherOutcome(agent, spec)
    max_rounds = spec.rounds or config.run.max_rounds
    since_improvement = 0
    consecutive_failures = 0
    first_round = 1
    if history:
        outcome.experiments = list(history)
        ok = [e for e in history if e.ok]
        if ok:
            outcome.best = max(
                ok, key=lambda e: e.oof_score * (1 if metric.greater_is_better else -1)
            )
            if team is not None:
                team.update(outcome.best)
        first_round = history[-1].round + 1
    n_classes = len(profile.target.classes)
    timeout = config.sandbox.experiment_timeout_seconds
    log.info(
        "%s: %s/%s on track %s for up to %d rounds",
        agent,
        spec.backend,
        spec.model,
        spec.track,
        max_rounds,
    )

    for round_no in range(first_round, max_rounds + 1):
        try:
            ctx.budget.check()
        except BudgetExceeded as exc:
            outcome.stopped_because = str(exc)
            raise
        exp_id = f"{ctx.run_id}-{agent}-r{round_no}"
        workdir = ctx.paths.run_dir / "experiments" / agent / f"r{round_no}"
        workdir.mkdir(parents=True, exist_ok=True)
        messages = researcher_messages(
            profile,
            metric=metric,
            spec=spec,
            experiments=outcome.experiments,
            best=outcome.best,
            round_no=round_no,
            max_rounds=max_rounds,
            timeout=timeout,
            leaderboard=_team_block(
                team, agent, outcome.best, round_no, config.run.share_every, spec.track
            )
            if team
            else (leaderboard() if leaderboard else ""),
            notes=notes() if notes else "",
            ensemble_text=pool.leaderboard() if pool else "",
            analysis=_last_analysis(outcome.experiments),
        )
        hypothesis, code, raw, backend, model = propose(
            router, messages, spec=spec, agent=agent, branch_id=exp_id
        )
        atomic_write_text(workdir / "reply.txt", raw)
        experiment = Experiment(
            exp_id, agent, round_no, hypothesis, code or "", backend, model, raw=raw
        )
        outcome.experiments.append(experiment)
        if code is None:
            log.warning("%s round %d: no module in the reply", agent, round_no)
            ctx.ledger.record_experiment(
                exp_id,
                ctx.run_id,
                agent=agent,
                round=round_no,
                status="failed",
                backend=backend,
                model=model,
                track=spec.track,
                hypothesis=hypothesis,
                kind="no-code",
                error=raw[-500:],
                parent=outcome.best.id if outcome.best else None,
            )
            since_improvement += 1
        elif track_problems := check_track(code, spec.track):
            error = "track violation:\n" + "\n".join(f"- {p}" for p in track_problems)
            experiment.result = ExperimentResult(False, "track", error)
            log.warning("%s round %d: %s", agent, round_no, error.replace("\n", " "))
            ctx.ledger.record_experiment(
                exp_id,
                ctx.run_id,
                agent=agent,
                round=round_no,
                status="failed",
                backend=backend,
                model=model,
                track=spec.track,
                code_hash=code_hash(code),
                hypothesis=hypothesis,
                kind="track",
                error=error[:500],
                parent=outcome.best.id if outcome.best else None,
            )
            since_improvement += 1
        else:
            result = run_experiment(
                code,
                workdir=workdir,
                train_path=sandbox_train,
                test_path=sandbox_test,
                y=y,
                folds=folds,
                metric=metric,
                target=profile.target.name,
                id_col=profile.id_col,
                n_classes=n_classes,
                categorical=profile.categorical_columns(),
                drop_columns=profile.unusable_columns(),
                seed=config.run.seed,
                timeout=timeout,
                memory_mb=config.sandbox.memory_mb,
                n_threads=config.run.n_jobs,
                determinism_rows=config.sandbox.determinism_rows,
            )
            if result.ok:
                result = degenerate_check(result, y, metric, config.run.degenerate_margin)
            if result.ok:
                result = leak_check(
                    result,
                    code,
                    y,
                    folds,
                    metric,
                    config=config,
                    profile=profile,
                    workdir=workdir,
                    sandbox_train=sandbox_train,
                    sandbox_test=sandbox_test,
                    reference=pool.score
                    if pool and pool.score is not None
                    else (outcome.best.oof_score if outcome.best else None),
                    agent=agent,
                    round_no=round_no,
                )
            experiment.result = result
            if result.ok:
                leader = team.leader() if team else None
                analysis = analyse(
                    result,
                    y,
                    folds,
                    metric,
                    slices=slices,
                    blend_oof=pool.best.oof if pool and pool.best else None,
                    blend_score=pool.score if pool else None,
                    leader_score=leader.oof_score if leader else None,
                )
                experiment.analysis = analysis.text
                atomic_write_json(workdir / "analysis.json", analysis.data)
            log.info("%s round %d: %s -> %s", agent, round_no, hypothesis[:80], result.summary())
            ctx.ledger.record_experiment(
                exp_id,
                ctx.run_id,
                agent=agent,
                round=round_no,
                status="ok" if result.ok else "failed",
                backend=backend,
                model=model,
                track=spec.track,
                code_hash=code_hash(code),
                hypothesis=hypothesis,
                kind=result.kind,
                cv_mean=result.cv_mean,
                cv_std=result.cv_std,
                oof_score=result.oof_score,
                fold_scores=result.fold_scores or None,
                duration=result.duration,
                error=None if result.ok else result.error[-500:],
                parent=outcome.best.id if outcome.best else None,
            )
            if result.ok:
                improved = (
                    outcome.best is None
                    or metric.improvement(result.oof_score, outcome.best.oof_score)
                    > config.run.min_improvement
                )
                if pool is not None:
                    update = pool.add(
                        Member(
                            exp_id.split("-", 2)[-1],
                            result.oof,
                            result.test_pred,
                            result.oof_score,
                            "experiment",
                        )
                    )
                    experiment.pool_gain = update.gain
                    log.info(
                        "%s round %d: ensemble %s -> %.5f (gain %+.5f, %s of %d)",
                        agent,
                        round_no,
                        "" if update.before is None else f"{update.before:.5f}",
                        update.after,
                        update.gain,
                        update.method,
                        update.n_members,
                    )
                    improved = improved or update.gain > config.run.min_improvement
                    if after_experiment is not None:
                        after_experiment(update)
                if team is not None:
                    team.update(experiment)
                if improved:
                    outcome.best = experiment
                    since_improvement = 0
                else:
                    since_improvement += 1
            else:
                since_improvement += 1
        last = outcome.experiments[-1]
        consecutive_failures = consecutive_failures + 1 if not last.ok else 0
        if consecutive_failures >= config.run.max_consecutive_failures:
            outcome.stopped_because = f"{consecutive_failures} failed experiments in a row"
            break
        if since_improvement >= config.run.patience and outcome.best is not None and last.ok:
            outcome.stopped_because = f"no improvement for {since_improvement} rounds"
            break
    else:
        outcome.stopped_because = "rounds exhausted"
    outcome.duration = time.monotonic() - started
    log.info(
        "%s done: %d/%d experiments ok, best %s, %s",
        agent,
        outcome.n_ok,
        len(outcome.experiments),
        f"{outcome.best.oof_score:.5f}" if outcome.best else "none",
        outcome.stopped_because,
    )
    return outcome


def _team_block(
    team: TeamState,
    agent: str,
    own_best: Experiment | None,
    round_no: int,
    share_every: int,
    track: str | None = None,
) -> str:
    text = team.leaderboard()
    if round_no > 1 and round_no % share_every == 0:
        tip = team.tip_for(agent, own_best, track=track)
        if tip:
            text += "\n\n" + tip
    return text


def _last_analysis(experiments: list[Experiment]) -> str:
    for e in reversed(experiments):
        if e.analysis:
            return e.analysis
    return ""


def load_history(
    run_dir: Path, rows: list[dict[str, object]], metric: MetricSpec
) -> list[Experiment]:
    """Rebuild an agent's experiments from ledger rows and the artifacts on disk (resume)."""
    import json

    import numpy as np

    history: list[Experiment] = []
    for row in rows:
        exp_id = str(row["id"])
        agent = str(row["agent"])
        round_no = int(row["round"])
        workdir = run_dir / "experiments" / agent / f"r{round_no}"
        code = (
            (workdir / "experiment.py").read_text() if (workdir / "experiment.py").exists() else ""
        )
        experiment = Experiment(
            exp_id,
            agent,
            round_no,
            str(row.get("hypothesis") or ""),
            code,
            str(row.get("backend") or ""),
            str(row.get("model") or ""),
        )
        metrics_path = workdir / "metrics.json"
        if row.get("status") == "ok" and (workdir / "oof.npy").exists() and metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            experiment.result = ExperimentResult(
                True,
                "ok",
                oof=np.load(workdir / "oof.npy"),
                test_pred=np.load(workdir / "test_pred.npy"),
                fold_scores=list(m.get("fold_scores") or []),
                oof_score=float(m["oof_score"]),
                features=list(m.get("features") or []),
                duration=float(m.get("duration") or 0),
            )
        else:
            experiment.result = ExperimentResult(
                False, str(row.get("kind") or "error"), str(row.get("error") or "")
            )
        history.append(experiment)
    return history


def chance_score(y: np.ndarray, metric: MetricSpec) -> float:
    """The metric's value for a constant prediction at the positive rate (or class rates)."""
    y = np.asarray(y)
    k = int(y.max()) + 1
    if k == 2:
        return score(metric, y, np.full(len(y), float(y.mean())))
    rates = np.bincount(y, minlength=k) / len(y)
    return score(metric, y, np.tile(rates, (len(y), 1)))


def degenerate_check(
    result: ExperimentResult, y: np.ndarray, metric: MetricSpec, margin: float
) -> ExperimentResult:
    """A module whose predictions are constant, or no better than a constant prediction by
    ``margin``, has not learned anything: fail it instead of counting it as a success."""
    if not result.ok or result.oof is None or result.oof_score is None:
        return result
    oof = np.asarray(result.oof, dtype=float)
    constant = oof.size > 0 and bool(np.all(np.nan_to_num(oof) == np.nan_to_num(oof.flat[0])))
    chance = chance_score(y, metric)
    gain = metric.improvement(result.oof_score, chance)
    if not constant and gain > margin:
        return result
    what = (
        f"every prediction is {oof.flat[0]:.4f}"
        if constant
        else f"OOF {result.oof_score:.5f} is within {margin} of a constant prediction"
    )
    error = (
        f"degenerate experiment: {what} (a constant prediction scores {chance:.5f}). "
        "The model learned nothing: check that fit_predict returns probabilities of the "
        "positive class from a fitted model, that the network trains (loss goes down, "
        "targets have shape (n, 1) to match the logits), and that features are not all NaN."
    )
    return ExperimentResult(
        False,
        "degenerate",
        error,
        fold_scores=result.fold_scores,
        oof_score=result.oof_score,
        duration=result.duration,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def leak_check(
    result: ExperimentResult,
    code: str,
    y: np.ndarray,
    folds: np.ndarray,
    metric: MetricSpec,
    *,
    config,
    profile: Profile,
    workdir: Path,
    sandbox_train: Path,
    sandbox_test: Path,
    reference: float | None,
    agent: str,
    round_no: int,
) -> ExperimentResult:
    """README section 10: a jump above ``leak_threshold`` over the best known score is re-run
    with shuffled targets; a score that still beats chance means the module leaks."""
    if (
        reference is None
        or metric.improvement(result.oof_score, reference) <= config.run.leak_threshold
    ):
        return result
    rng = np.random.default_rng(config.run.seed)
    y_shuffled = rng.permutation(np.asarray(y))
    log.warning(
        "%s round %d: OOF %.5f jumps %.4f over %.5f; re-running with shuffled targets",
        agent,
        round_no,
        result.oof_score,
        metric.improvement(result.oof_score, reference),
        reference,
    )
    shuffled = run_experiment(
        code,
        workdir=workdir / "shuffled",
        train_path=sandbox_train,
        test_path=sandbox_test,
        y=y_shuffled,
        folds=folds,
        metric=metric,
        target=profile.target.name,
        id_col=profile.id_col,
        n_classes=len(profile.target.classes),
        categorical=profile.categorical_columns(),
        drop_columns=profile.unusable_columns(),
        seed=config.run.seed,
        timeout=config.sandbox.experiment_timeout_seconds,
        memory_mb=config.sandbox.memory_mb,
        n_threads=config.run.n_jobs,
        determinism_rows=config.sandbox.determinism_rows,
    )
    if not shuffled.ok:
        log.warning(
            "%s round %d: shuffled re-run failed (%s); keeping the experiment",
            agent,
            round_no,
            shuffled.kind,
        )
        return result
    chance = chance_score(y_shuffled, metric)
    margin = metric.improvement(shuffled.oof_score, chance)
    if margin > config.run.leak_threshold / 2:
        error = (
            "leakage tripwire: with shuffled targets the module still scores "
            f"{shuffled.oof_score:.5f} "
            f"(chance {chance:.5f}, margin {margin:+.4f}); real score {result.oof_score:.5f}. "
            "The module must be using information it should not have."
        )
        log.error("%s round %d: %s", agent, round_no, error)
        return ExperimentResult(False, "leak", error, duration=result.duration + shuffled.duration)
    log.info(
        "%s round %d: shuffled targets score %.5f (chance %.5f); no leak",
        agent,
        round_no,
        shuffled.oof_score,
        chance,
    )
    return result
