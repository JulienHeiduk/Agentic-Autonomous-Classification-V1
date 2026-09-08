import numpy as np
import pytest

from aac.agents.ensembler import (
    LivePool,
    Member,
    ensemble,
    hill_climb,
    rank_average,
    stack_logistic,
)
from aac.models.cv import make_folds
from aac.models.metrics import METRICS, score


def make_members(n=1200, k_members=4, seed=0, noise=(0.6, 0.7, 0.8, 1.2)):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal = y * 1.5 - 0.75
    members = []
    for i in range(k_members):
        logit = signal + rng.normal(scale=noise[i % len(noise)], size=n)
        oof = 1 / (1 + np.exp(-logit))
        test = 1 / (1 + np.exp(-(signal[:300] + rng.normal(scale=noise[i % len(noise)], size=300))))
        members.append(Member(f"m{i}", oof, test, score(METRICS["auc"], y, oof), "test"))
    return y, members


def test_blends_beat_every_single_member():
    y, members = make_members()
    folds = make_folds(y, 5, 1)
    result = ensemble(members, y, folds, METRICS["auc"], seed=1)
    best_single = max(m.oof_score for m in members)
    assert result.best.oof_score > best_single + 0.01
    assert result.best.method in ("hill_climb", "rank_average", "stack_logistic")
    methods = [b.method for b in result.blends]
    assert methods == ["single", "hill_climb", "rank_average", "stack_logistic"]
    for b in result.blends:
        assert b.oof.shape == (1200,) and b.test_pred.shape == (300,)
        assert abs(sum(b.weights.values()) - 1) < 1e-9
    summary = result.summary()
    assert summary["best"]["n_members"] >= 2 and len(summary["members"]) == 4


def test_hill_climb_prefers_good_members_and_is_deterministic():
    y, members = make_members()
    a = hill_climb(members, y, METRICS["auc"])
    b = hill_climb(members, y, METRICS["auc"])
    assert a.weights == b.weights and np.array_equal(a.oof, b.oof)
    assert a.weights["m3"] <= a.weights["m0"], "the noisiest member gets the least weight"
    assert a.oof_score >= max(m.oof_score for m in members)


def test_rank_average_is_scale_free():
    y, members = make_members(k_members=2)
    scaled = [Member("s0", members[0].oof * 0.5, members[0].test_pred * 0.5, 0.0), members[1]]
    assert np.allclose(
        rank_average(scaled, y, METRICS["auc"]).oof, rank_average(members, y, METRICS["auc"]).oof
    )


def test_stacker_is_fold_honest_and_handles_logloss_direction():
    y, members = make_members()
    folds = make_folds(y, 4, 3)
    blend = stack_logistic(members, y, folds, METRICS["logloss"], seed=0)
    assert blend.method == "stack_logistic" and 0 < blend.oof_score < 0.7
    result = ensemble(members, y, folds, METRICS["logloss"], seed=0)
    assert result.best.oof_score <= min(score(METRICS["logloss"], y, m.oof) for m in members)


def test_single_member_pool_returns_single():
    y, members = make_members(k_members=1)
    result = ensemble(members[:1], y, make_folds(y, 3, 0), METRICS["auc"])
    assert result.best.method == "single" and len(result.blends) == 1
    with pytest.raises(ValueError):
        ensemble([], y, make_folds(y, 3, 0), METRICS["auc"])


def test_multiclass_pool():
    rng = np.random.default_rng(0)
    n, k = 600, 3
    y = rng.integers(0, k, n)
    members = []
    for i in range(3):
        logits = np.eye(k)[y] * 2 + rng.normal(scale=1.0 + 0.3 * i, size=(n, k))
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        t = np.exp(logits[:100]) / np.exp(logits[:100]).sum(axis=1, keepdims=True)
        members.append(Member(f"m{i}", p, t, score(METRICS["logloss"], y, p)))
    result = ensemble(members, y, make_folds(y, 4, 0), METRICS["logloss"])
    assert result.best.oof.shape == (n, k) and result.best.test_pred.shape == (100, k)
    np.testing.assert_allclose(result.best.oof.sum(axis=1), 1.0, atol=1e-6)
    assert result.best.oof_score <= min(m.oof_score for m in members)


def test_live_pool_tracks_gains_and_leaderboard():
    y, members = make_members()
    pool = LivePool(y, make_folds(y, 4, 0), METRICS["auc"], seed=0, min_improvement=0.0005)
    assert pool.score is None and "empty" in pool.leaderboard()
    first = pool.add(members[0])
    assert first.before is None and first.gain == 0.0 and first.method == "single"
    second = pool.add(members[1])
    assert second.before == first.after and second.gain > 0 and second.n_members >= 2
    assert pool.score == second.after and len(pool.history) == 2
    text = pool.leaderboard()
    assert "Ensemble so far" in text and "m0 |" in text and "m1 |" in text and "+0." in text
    worse = Member(
        "noise", np.random.default_rng(1).random(len(y)), np.zeros(300), 0.5, "experiment"
    )
    third = pool.add(worse)
    assert third.gain <= 1e-6, "pure noise must not improve the blend"
    assert third.after >= second.after - 1e-12, "the pool never gets worse when a member joins"


def test_pool_is_monotone_under_accuracy():
    y, members = make_members(n=800, k_members=5, noise=(0.9, 1.0, 1.1, 1.2, 1.3))
    pool = LivePool(y, make_folds(y, 4, 0), METRICS["accuracy"], seed=0)
    scores = [pool.add(m).after for m in members]
    assert all(b >= a - 1e-12 for a, b in zip(scores, scores[1:], strict=False)), scores
    assert (
        any(b.method == "previous" for b in pool.result.blends)
        or pool.result.best.method != "single"
    )


def test_a_blend_wins_on_any_positive_margin():
    """min_improvement gates uploads, not the choice among blends."""
    y, members = make_members(n=1500, k_members=3, noise=(1.0, 1.05, 1.1))
    folds = make_folds(y, 4, 0)
    strict = ensemble(members, y, folds, METRICS["auc"], min_improvement=0.5)
    loose = ensemble(members, y, folds, METRICS["auc"], min_improvement=0.0)
    assert strict.best.method == loose.best.method and strict.best.oof_score == loose.best.oof_score
    assert strict.best.method != "single"
