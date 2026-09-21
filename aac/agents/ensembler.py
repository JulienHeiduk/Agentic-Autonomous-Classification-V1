"""Ensembler: mostly deterministic (README sections 3 and 17.2).

The pool holds every scored candidate's OOF and test predictions on the shared folds. Three
blends are tried, hill climbing with replacement (Caruana et al.), a rank average, and a
logistic stacker fitted fold-honestly on the OOF matrix, and whichever has the best honest
OOF score wins, including the best single model when nothing beats it.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import rankdata

from aac.models.metrics import MetricSpec, score

log = logging.getLogger(__name__)


@dataclass
class Member:
    name: str
    oof: np.ndarray  # (n,) binary or (n, k) multiclass, probabilities
    test_pred: np.ndarray
    oof_score: float
    origin: str = ""


@dataclass
class Blend:
    method: str
    weights: dict[str, float]
    oof: np.ndarray
    test_pred: np.ndarray
    oof_score: float

    @property
    def n_members(self) -> int:
        return sum(1 for w in self.weights.values() if w > 0)

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "oof_score": self.oof_score,
            "n_members": self.n_members,
            "weights": {k: v for k, v in self.weights.items() if v > 0},
        }


@dataclass
class EnsembleResult:
    best: Blend
    blends: list[Blend] = field(default_factory=list)
    members: list[Member] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "best": self.best.summary(),
            "blends": [b.summary() for b in self.blends],
            "members": [
                {"name": m.name, "origin": m.origin, "oof_score": m.oof_score} for m in self.members
            ],
        }


def _better(metric: MetricSpec, a: float, b: float, min_gain: float = 0.0) -> bool:
    return metric.improvement(a, b) > min_gain


def _weighted(members: list[Member], weights: np.ndarray, attr: str) -> np.ndarray:
    total = float(weights.sum())
    return sum(w * getattr(m, attr) for m, w in zip(members, weights, strict=True)) / total


def single_best(members: list[Member], metric: MetricSpec) -> Blend:
    best = max(members, key=lambda m: m.oof_score * (1 if metric.greater_is_better else -1))
    weights = {m.name: (1.0 if m is best else 0.0) for m in members}
    return Blend("single", weights, best.oof, best.test_pred, best.oof_score)


def hill_climb(
    members: list[Member],
    y: np.ndarray,
    metric: MetricSpec,
    *,
    max_steps: int = 60,
    min_gain: float = 1e-6,
    warm: Blend | None = None,
    warm_steps: int = 20,
) -> Blend:
    """Greedy forward selection with replacement.

    Starts from the best single member, or from ``warm``'s weights (quantised to
    ``warm_steps`` units) so a growing pool never climbs to a worse blend than before.
    """
    order = sorted(
        range(len(members)),
        key=lambda i: members[i].oof_score * (1 if metric.greater_is_better else -1),
        reverse=True,
    )
    counts = np.zeros(len(members))
    if warm is not None and any(warm.weights.get(m.name, 0.0) > 0 for m in members):
        for i, m in enumerate(members):
            counts[i] = round(warm.weights.get(m.name, 0.0) * warm_steps)
        if counts.sum() == 0:
            counts[order[0]] = 1
        current = _weighted(members, counts, "oof").astype(np.float64)
        current_score = score(metric, y, current)
    else:
        counts[order[0]] = 1
        current = members[order[0]].oof.astype(np.float64)
        current_score = members[order[0]].oof_score
    for _ in range(max_steps):
        best_i, best_score, best_oof = None, current_score, None
        total = counts.sum()
        for i, m in enumerate(members):
            candidate = (current * total + m.oof) / (total + 1)
            s = score(metric, y, candidate)
            if _better(metric, s, best_score, min_gain):
                best_i, best_score, best_oof = i, s, candidate
        if best_i is None:
            break
        counts[best_i] += 1
        current, current_score = best_oof, best_score
    weights = counts / counts.sum()
    test_pred = _weighted(members, weights, "test_pred")
    return Blend(
        "hill_climb",
        {m.name: float(w) for m, w in zip(members, weights, strict=True)},
        _weighted(members, weights, "oof"),
        test_pred,
        current_score,
    )


def _ranks(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return rankdata(a) / len(a)
    ranked = np.column_stack([rankdata(a[:, j]) / a.shape[0] for j in range(a.shape[1])])
    return ranked / ranked.sum(axis=1, keepdims=True)


def rank_average(members: list[Member], y: np.ndarray, metric: MetricSpec) -> Blend:
    """Equal-weight average of within-array ranks (scale free; only meaningful for AUC-like
    metrics, which is why it competes on the honest score rather than being assumed)."""
    n = len(members)
    oof = sum(_ranks(m.oof) for m in members) / n
    test = sum(_ranks(m.test_pred) for m in members) / n
    weights = {m.name: 1.0 / n for m in members}
    return Blend("rank_average", weights, oof, test, score(metric, y, oof))


def _stack_features(arrays: list[np.ndarray]) -> np.ndarray:
    cols = []
    for a in arrays:
        p = np.clip(a, 1e-6, 1 - 1e-6)
        cols.append(np.log(p / (1 - p)) if p.ndim == 1 else np.log(p))
    return np.column_stack(cols)


def stack_logistic(
    members: list[Member], y: np.ndarray, folds: np.ndarray, metric: MetricSpec, seed: int
) -> Blend:
    """Logistic regression on the members' logits, fitted per fold on the OOF matrix so the
    stacked OOF is as honest as the members' own."""
    from sklearn.linear_model import LogisticRegression

    X = _stack_features([m.oof for m in members])
    X_test = _stack_features([m.test_pred for m in members])
    n_classes = int(y.max()) + 1
    oof = np.zeros(len(y)) if n_classes == 2 else np.zeros((len(y), n_classes))
    for k in range(int(folds.max()) + 1):
        tr = folds != k
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[~tr])
        oof[~tr] = proba[:, 1] if n_classes == 2 else proba
    final = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(X, y)
    proba = final.predict_proba(X_test)
    test = proba[:, 1] if n_classes == 2 else proba
    coef = np.abs(final.coef_).sum(axis=0)
    per_member = []
    offset = 0
    for m in members:
        width = 1 if m.oof.ndim == 1 else m.oof.shape[1]
        per_member.append(float(coef[offset : offset + width].sum()))
        offset += width
    total = sum(per_member) or 1.0
    weights = {m.name: w / total for m, w in zip(members, per_member, strict=True)}
    return Blend("stack_logistic", weights, oof, test, score(metric, y, oof))


def ensemble(
    members: list[Member],
    y: np.ndarray,
    folds: np.ndarray,
    metric: MetricSpec,
    *,
    seed: int = 42,
    min_improvement: float = 0.0,
    previous: Blend | None = None,
) -> EnsembleResult:
    if not members:
        raise ValueError("the pool is empty")
    y = np.asarray(y)
    blends = [single_best(members, metric)]
    if previous is not None and any(previous.weights.get(m.name, 0.0) > 0 for m in members):
        weights = np.array([previous.weights.get(m.name, 0.0) for m in members])
        oof = _weighted(members, weights, "oof")
        blends.append(
            Blend(
                "previous",
                {m.name: float(w) for m, w in zip(members, weights, strict=True)},
                oof,
                _weighted(members, weights, "test_pred"),
                score(metric, y, oof),
            )
        )
    if len(members) > 1:
        blends.append(hill_climb(members, y, metric, warm=previous))
        blends.append(rank_average(members, y, metric))
        try:
            blends.append(stack_logistic(members, y, folds, metric, seed))
        except Exception as exc:  # noqa: BLE001 - a stacker failure is not a run failure
            log.warning("stacker failed: %s", exc)
    # The honest OOF decides between blends outright; min_improvement gates uploads, not this.
    best = blends[0]
    for b in blends[1:]:
        if _better(metric, b.oof_score, best.oof_score, 0.0):
            best = b
    log.info(
        "ensemble: %s wins with OOF %s %.5f over %d members (%s)",
        best.method,
        metric.key,
        best.oof_score,
        len(members),
        ", ".join(f"{b.method} {b.oof_score:.5f}" for b in blends),
    )
    return EnsembleResult(best, blends, members)


@dataclass
class PoolUpdate:
    member: str
    before: float | None
    after: float
    gain: float
    method: str
    n_members: int


class LivePool:
    """The ensemble pool during a run: re-blended after every addition so Researchers see
    each experiment's marginal gain and the Submitter can upload on improvement."""

    def __init__(
        self,
        y: np.ndarray,
        folds: np.ndarray,
        metric: MetricSpec,
        *,
        seed: int = 42,
        min_improvement: float = 0.0,
    ) -> None:
        self.y = np.asarray(y)
        self.folds = np.asarray(folds)
        self.metric = metric
        self.seed = seed
        self.min_improvement = min_improvement
        self.members: list[Member] = []
        self.result: EnsembleResult | None = None
        self.history: list[PoolUpdate] = []
        self._lock = threading.Lock()

    @property
    def score(self) -> float | None:
        return self.result.best.oof_score if self.result else None

    @property
    def best(self) -> Blend | None:
        return self.result.best if self.result else None

    def add(self, member: Member) -> PoolUpdate:
        with self._lock:
            before = self.score
            self.members.append(member)
            self.result = ensemble(
                self.members,
                self.y,
                self.folds,
                self.metric,
                seed=self.seed,
                min_improvement=self.min_improvement,
                previous=self.best,
            )
            after = self.result.best.oof_score
            gain = 0.0 if before is None else self.metric.improvement(after, before)
            update = PoolUpdate(
                member.name,
                before,
                after,
                gain,
                self.result.best.method,
                self.result.best.n_members,
            )
            self.history.append(update)
            return update

    def leaderboard(self, limit: int = 15) -> str:
        """Members by OOF with their weight in the current blend, for prompts."""
        if not self.members or self.result is None:
            return "Ensemble so far: (empty)"
        sign = 1 if self.metric.greater_is_better else -1
        weights = self.result.best.weights
        gains = {u.member: u.gain for u in self.history}
        rows = sorted(self.members, key=lambda m: sign * m.oof_score, reverse=True)[:limit]
        lines = [
            f"Ensemble so far: {self.result.best.method} of {self.result.best.n_members} members, "
            f"OOF {self.metric.display} {self.result.best.oof_score:.5f} "
            f"({'higher' if self.metric.greater_is_better else 'lower'} is better).",
            "Members (name | origin | OOF | weight in blend | gain to the ensemble when added):",
        ]
        for m in rows:
            lines.append(
                f"{m.name} | {m.origin} | {m.oof_score:.5f} | {weights.get(m.name, 0.0):.3f} | "
                f"{gains.get(m.name, 0.0):+.5f}"
            )
        return "\n".join(lines)
