import httpx
import numpy as np
import pandas as pd
import pytest

from aac.agents.ensembler import LivePool, PoolUpdate
from aac.agents.researcher import parse_reply, render_history, run_researcher
from aac.agents.scout import profile_data
from aac.config import CompetitionConfig, ResearcherSpec, load_config
from aac.context import RunContext
from aac.llm.router import Router
from aac.models.cv import make_folds
from aac.models.metrics import METRICS
from tests.llm_fake import FakeLLM
from tests.synth import make_frames
from tests.test_harness import LIGHTGBM_WITH_FEATURES, LOGISTIC

WEAK = (
    "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\n"
    "def fit_predict(a, b, c, d, e):\n"
    "    m = LogisticRegression().fit(a[['x3']].fillna(0), b)\n"
    "    pv = m.predict_proba(c[['x3']].fillna(0))[:, 1]\n"
    "    return pv, m.predict_proba(d[['x3']].fillna(0))[:, 1]\n"
)
BAD = (
    "import numpy as np\ndef fit_predict(a, b, c, d, e):\n"
    "    return a['nope'].values, np.zeros(len(d))\n"
)


def reply(hypothesis, code):
    return f"HYPOTHESIS: {hypothesis}\n```python\n{code}\n```"


def test_parse_reply_forms():
    h, c = parse_reply(reply("try logistic", LOGISTIC))
    assert h == "try logistic" and c is not None and "def fit_predict" in c
    h, c = parse_reply(f"```python\n# first idea\n{LOGISTIC}\n```")
    assert h == "first idea" and c is not None
    h, c = parse_reply("HYPOTHESIS: nothing\nno code at all")
    assert h == "nothing" and c is None
    h, c = parse_reply("```python\ndef build_features(a, b):\n    return a, b, []\n```")
    assert c is None, "a module without fit_predict is not an experiment"


@pytest.fixture
def setup(tmp_path, write_config, minimal_config, monkeypatch):
    monkeypatch.setenv("NV_KEY", "k")
    minimal_config["run"] = {
        "n_folds": 4,
        "n_jobs": 2,
        "patience": 2,
        "max_rounds": 5,
    }
    minimal_config["sandbox"] = {"experiment_timeout_seconds": 180}
    config = load_config(write_config(minimal_config))
    ctx = RunContext.create(config, tmp_path / "runs")
    train, test, sample = make_frames(300, 90)
    profile = profile_data(
        train,
        test,
        sample,
        slug="s",
        competition=CompetitionConfig(slug="s"),
        metric=METRICS["auc"],
        max_classes=50,
        seed=0,
    )
    enc = profile.encoding()
    y = enc.encode(train["target"])
    folds = make_folds(y, 4, 0)
    train_path, test_path = ctx.paths.run_dir / "st.parquet", ctx.paths.run_dir / "se.parquet"
    train.drop(columns=["target"]).to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    return ctx, profile, y, folds, train_path, test_path


def run(setup, fake, track="open", rounds=None, pool=None, after_experiment=None, team=None):
    ctx, profile, y, folds, train_path, test_path = setup
    router = Router(
        ctx.config,
        ledger=ctx.ledger,
        budget=ctx.budget,
        run_id=ctx.run_id,
        client=httpx.Client(transport=fake.transport),
        sleep=lambda s: None,
    )
    spec = ResearcherSpec(backend="nvidia", model="m-nv", track=track, rounds=rounds)
    return run_researcher(
        ctx,
        router,
        spec=spec,
        index=1,
        profile=profile,
        metric=METRICS["auc"],
        y=y,
        folds=folds,
        sandbox_train=train_path,
        sandbox_test=test_path,
        pool=pool,
        after_experiment=after_experiment,
        team=team,
    )


def test_loop_records_improves_feeds_back_errors_and_stops_on_patience(setup):
    ctx = setup[0]
    fake = FakeLLM(
        {
            "nv.test": [
                reply("logistic baseline", LOGISTIC),
                reply("broken idea", BAD),
                reply("lightgbm with an interaction", LIGHTGBM_WITH_FEATURES),
                reply("logistic again", LOGISTIC),
                reply("should not be asked", LOGISTIC),
            ]
        }
    )
    out = run(setup, fake)
    statuses = [(e.round, e.ok) for e in out.experiments]
    # logistic beats lightgbm on this synthetic data, so round 3 is no improvement: patience hits
    assert statuses == [(1, True), (2, False), (3, True)]
    assert out.best.round == 1 and out.experiments[2].oof_score < out.best.oof_score
    assert out.stopped_because.startswith("no improvement for 2 rounds")
    prompts = fake.calls("nv.test")
    round3_user = prompts[2]["messages"][-1]["content"]
    assert "KeyError" in round3_user and "broken idea" in round3_user, "failure fed back"
    assert "logistic baseline" in round3_user and "Your best experiment so far" in round3_user
    assert (
        "Round 3 of" in round3_user
        and "anything: any family" in prompts[0]["messages"][0]["content"]
    )
    rows = ctx.ledger.list_experiments(ctx.run_id, out.agent)
    assert [(r["round"], r["status"], r["kind"]) for r in rows] == [
        (1, "ok", "ok"),
        (2, "failed", "error"),
        (3, "ok", "ok"),
    ]
    assert rows[2]["parent"] == rows[0]["id"] and rows[0]["oof_score"] == out.best.oof_score
    assert (ctx.paths.run_dir / "experiments" / out.agent / "r2" / "reply.txt").exists()
    assert (ctx.paths.run_dir / "experiments" / out.agent / "r3" / "oof.npy").exists()
    calls = ctx.ledger.list_llm_calls(ctx.run_id)
    assert all(
        c["agent"] == "researcher" and c["branch_id"].endswith(f"-r{i + 1}")
        for i, c in enumerate(calls)
    )


def test_no_code_then_repair_then_gives_up(setup):
    fake = FakeLLM({"nv.test": ["I have no idea", "still nothing", "nope", "nope", "nope", "nope"]})
    out = run(setup, fake, rounds=3)
    assert out.best is None and all(not e.ok for e in out.experiments)
    assert (
        out.stopped_because.endswith("without a working experiment")
        or out.stopped_because == "rounds exhausted"
    )
    agents = [c["agent"] for c in setup[0].ledger.list_llm_calls(setup[0].run_id)]
    assert agents[:2] == ["researcher", "researcher:repair"]


def test_render_history_handles_failures():
    assert render_history([]) == "(none yet)"


def test_environment_card_and_failure_allowance(setup):
    from aac.agents.researcher import environment_card

    card = environment_card()
    assert "lightgbm" in card and "early_stopping_rounds" in card and "pandas 3" in card
    replies = [reply(f"bad {i}", BAD) for i in range(4)] + [reply("fifth", LOGISTIC)]
    fake = FakeLLM({"nv.test": replies})
    out = run(setup, fake, rounds=6)
    assert [e.ok for e in out.experiments] == [False, False, False, False]
    assert out.stopped_because == "4 failed experiments in a row"
    system = fake.calls("nv.test")[0]["messages"][0]["content"]
    assert "Installed environment" in system and "eval_X" in system


def test_pool_gain_feeds_prompt_patience_and_callback(setup):
    ctx, profile, y, folds, *_ = setup
    pool = LivePool(y, folds, METRICS["auc"], seed=0, min_improvement=0.0005)
    updates: list[PoolUpdate] = []
    fake = FakeLLM(
        {
            "nv.test": [
                reply("logistic", LOGISTIC),
                reply("lightgbm, diverse", LIGHTGBM_WITH_FEATURES),
                reply("logistic again", LOGISTIC),
                reply("logistic yet again", LOGISTIC),
            ]
        }
    )
    out = run(setup, fake, rounds=4, pool=pool, after_experiment=updates.append)
    assert [e.ok for e in out.experiments[:2]] == [True, True]
    assert out.experiments[0].pool_gain == 0.0 and out.experiments[1].pool_gain is not None
    assert len(updates) == len([e for e in out.experiments if e.ok]) and updates[0].n_members == 1
    second_prompt = fake.calls("nv.test")[1]["messages"][-1]["content"]
    assert "Ensemble so far" in second_prompt and "r01-open-nvidia-r1" in second_prompt
    third_prompt = fake.calls("nv.test")[2]["messages"][-1]["content"]
    assert "[ensemble gain" in third_prompt
    assert "team ensemble" in fake.calls("nv.test")[0]["messages"][0]["content"]


def test_render_notes_filters_kinds():
    from aac.agents.researcher import render_notes

    assert render_notes([]) == "Research notes: (none)"
    notes = [
        {"kind": "competition", "source": "kaggle", "text": "Predict EV purchases. Metric: AUC."},
        {"kind": "budget", "source": "orchestrator", "text": "ignored"},
        {"kind": "tip", "source": "r01", "text": "cabin deck helps"},
    ]
    text = render_notes(notes)
    assert "[competition from kaggle] Predict EV" in text and "cabin deck" in text
    assert "ignored" not in text


def test_track_enforcement_and_tip_sharing(setup):
    from aac.agents.researcher import TeamState, check_track

    assert check_track(LOGISTIC, "linear") == [] and check_track(LOGISTIC, "open") == []
    assert any("requires" in p for p in check_track(LOGISTIC, "gbdt"))
    assert any("forbids" in p for p in check_track(LIGHTGBM_WITH_FEATURES, "linear"))
    assert check_track(LIGHTGBM_WITH_FEATURES, "gbdt") == []
    torch_code = "import torch\ndef fit_predict(a, b, c, d, e):\n    return None\n"
    assert check_track(torch_code, "neural") == [] and check_track(torch_code, "gbdt")

    ctx = setup[0]
    team = TeamState(METRICS["auc"])
    fake = FakeLLM(
        {
            "nv.test": [
                reply("logistic on gbdt track", LOGISTIC),
                reply("lightgbm", LIGHTGBM_WITH_FEATURES),
            ]
        }
    )
    out = run(setup, fake, track="gbdt", rounds=2, team=team)
    assert [e.ok for e in out.experiments] == [False, True]
    assert out.experiments[0].result.kind == "track"
    rows = ctx.ledger.list_experiments(ctx.run_id, out.agent)
    assert rows[0]["kind"] == "track" and "requires" in rows[0]["error"]
    second_prompt = fake.calls("nv.test")[1]["messages"][-1]["content"]
    assert "track violation" in second_prompt
    assert team.leader() is out.experiments[1]

    # a trailing agent on round 2 receives the leader's code as a tip
    other = FakeLLM({"nv.test": [reply("weak", WEAK), reply("try again", WEAK)]})
    import httpx

    from aac.config import ResearcherSpec
    from aac.llm.router import Router

    ctx2 = setup[0]
    router = Router(
        ctx2.config,
        ledger=ctx2.ledger,
        budget=ctx2.budget,
        run_id=ctx2.run_id,
        client=httpx.Client(transport=other.transport),
        sleep=lambda s: None,
    )
    profile, y, folds, train_path, test_path = setup[1:]
    run_researcher(
        ctx2,
        router,
        spec=ResearcherSpec(backend="nvidia", model="m-nv", track="linear", rounds=2),
        index=2,
        profile=profile,
        metric=METRICS["auc"],
        y=y,
        folds=folds,
        sandbox_train=train_path,
        sandbox_test=test_path,
        team=team,
    )
    prompts = other.calls("nv.test")
    assert "Leaderboard of Researchers" in prompts[0]["messages"][-1]["content"]
    tip = prompts[1]["messages"][-1]["content"].split("Tip from")[1]
    assert tip.startswith(" the leading Researcher r01-gbdt-nvidia")
    # the leader's lightgbm module never reaches a linear seat as code: idea only
    assert "def fit_predict" not in tip and "outside your track" in tip
    assert "hypothesis: lightgbm" in tip
    assert "def fit_predict" in team.tip_for("r02-linear-nvidia", None)
    assert "def fit_predict" in team.tip_for("r02-linear-nvidia", None, track="gbdt")
    assert "def fit_predict" in team.tip_for("r02-linear-nvidia", None, track="open")
    neural_tip = team.tip_for("r02-linear-nvidia", None, track="neural")
    assert "def fit_predict" not in neural_tip and "requires torch" in neural_tip


CONSTANT = (
    "import numpy as np\ndef fit_predict(a, b, c, d, e):\n"
    "    return np.zeros(len(c)), np.zeros(len(d))\n"
)


def test_strip_off_track_keeps_only_what_the_track_may_copy():
    from aac.agents.researcher import render_notes, strip_off_track

    packet = (
        "Research packet:\n"
        "- (model) XGBoost tuned with class weight: learning_rate 0.02, depth 6.\n"
        "- (feature) Income per car: Annual_Income_USD / Number_of_Cars_Owned.\n"
        "- (model) A small MLP with embeddings for the categoricals.\n"
        "- (model) HistGradientBoosting with categorical_features='from_dtype'.\n"
        "- (feature) Target encoding of City_Type inside the fold.\n"
    )
    linear = strip_off_track(packet, "linear")
    assert "Income per car" in linear and "Target encoding" in linear
    assert "XGBoost" not in linear and "MLP" not in linear and "HistGradient" not in linear
    assert "(3 ideas about libraries outside the 'linear' track omitted)" in linear
    gbdt = strip_off_track(packet, "gbdt")
    assert "XGBoost" in gbdt and "HistGradient" in gbdt and "MLP" not in gbdt
    neural = strip_off_track(packet, "neural")
    assert "MLP" in neural and "XGBoost" not in neural and "HistGradient" in neural
    assert strip_off_track(packet, "open") == packet

    prior = f"Best so far:\n```python\n{LIGHTGBM_WITH_FEATURES}\n```\nend"
    stripped = strip_off_track(prior, "linear")
    assert "import lightgbm" not in stripped and "module omitted" in stripped
    assert "forbids ['lightgbm']" in stripped and stripped.endswith("end")
    assert strip_off_track(prior, "gbdt") == prior
    logistic_prior = f"```python\n{LOGISTIC}\n```"
    assert strip_off_track(logistic_prior, "linear") == logistic_prior
    # a module that lacks the track's required library would be rejected too: omitted
    assert "requires a gradient boosting" in strip_off_track(logistic_prior, "gbdt")
    assert "def fit_predict" not in strip_off_track(logistic_prior, "neural")

    notes = [
        {"kind": "research", "source": "scholar", "text": packet},
        {"kind": "pitfalls", "source": "historian", "text": "Pitfalls:\n- [error] n_jobs"},
    ]
    text = render_notes(notes, track="linear")
    assert "[pitfalls from historian]" in text and "n_jobs" in text
    assert "Income per car" in text and "XGBoost" not in text
    assert "XGBoost" in render_notes(notes)


def test_constant_predictions_fail_as_degenerate(setup):
    import numpy as np

    from aac.agents.researcher import degenerate_check
    from aac.exec.sandbox import ExperimentResult

    ctx, _, y, *_ = setup
    fake = FakeLLM({"nv.test": [reply("zeros", CONSTANT), reply("logistic", LOGISTIC)]})
    out = run(setup, fake, rounds=2)
    first, second = out.experiments
    assert not first.ok and first.result.kind == "degenerate"
    assert "every prediction is 0.0000" in first.result.error
    assert second.ok and out.n_ok == 1 and out.best is second
    rows = ctx.ledger.list_experiments(ctx.run_id, out.agent)
    assert rows[0]["status"] == "failed" and rows[0]["kind"] == "degenerate"
    assert "degenerate" in rows[0]["error"] and rows[0]["oof_score"] == 0.5
    best = ctx.ledger.best_experiments(ctx.config.competition.slug)
    assert [r["kind"] for r in best] == ["ok"], "a degenerate score never ranks as best"
    second_prompt = fake.calls("nv.test")[1]["messages"][-1]["content"]
    assert "degenerate" in second_prompt and "learned nothing" in second_prompt

    rng = np.random.default_rng(0)
    y = np.asarray(y)
    noise = rng.random(len(y))
    noisy = ExperimentResult(True, "ok", oof=noise, oof_score=0.5, fold_scores=[0.5])
    assert degenerate_check(noisy, y, METRICS["auc"], 0.002).kind == "degenerate"
    signal = np.clip(y + rng.normal(0, 0.5, len(y)), 0, 1)
    from sklearn.metrics import roc_auc_score

    good = ExperimentResult(True, "ok", oof=signal, oof_score=roc_auc_score(y, signal))
    assert degenerate_check(good, y, METRICS["auc"], 0.002).ok
    assert degenerate_check(ExperimentResult(False, "error", "x"), y, METRICS["auc"], 0.1).kind == (
        "error"
    )


def test_analysis_reaches_the_next_prompt(setup):
    ctx, profile, y, folds, train_path, test_path = setup
    fake = FakeLLM({"nv.test": [reply("logistic", LOGISTIC), reply("logistic again", LOGISTIC)]})
    frame = pd.read_parquet(train_path)[["cat", "flag"]]
    out = run_with_slices(setup, fake, frame)
    assert out.experiments[0].analysis.startswith("Analyst notes")
    assert (ctx.paths.run_dir / "experiments" / out.agent / "r1" / "analysis.json").exists()
    second = fake.calls("nv.test")[1]["messages"][-1]["content"]
    assert "Analyst notes on your last experiment" in second and "calibration" in second


def run_with_slices(setup, fake, frame):
    import httpx

    from aac.config import ResearcherSpec
    from aac.llm.router import Router

    ctx, profile, y, folds, train_path, test_path = setup
    router = Router(
        ctx.config,
        ledger=ctx.ledger,
        budget=ctx.budget,
        run_id=ctx.run_id,
        client=httpx.Client(transport=fake.transport),
        sleep=lambda s: None,
    )
    return run_researcher(
        ctx,
        router,
        spec=ResearcherSpec(backend="nvidia", model="m-nv", track="open", rounds=2),
        index=1,
        profile=profile,
        metric=METRICS["auc"],
        y=y,
        folds=folds,
        sandbox_train=train_path,
        sandbox_test=test_path,
        slices=frame,
    )


def test_leak_tripwire_reruns_with_shuffled_targets(setup, monkeypatch):
    import aac.agents.researcher as mod
    from aac.exec.sandbox import ExperimentResult

    ctx, profile, y, folds, *_ = setup
    real = ExperimentResult(
        True, "ok", oof=np.zeros(len(y)), test_pred=np.zeros(10), fold_scores=[0.99], oof_score=0.99
    )
    calls = []

    def fake_run(code, **kw):
        calls.append(kw["y"])
        return ExperimentResult(
            True,
            "ok",
            oof=np.zeros(len(y)),
            test_pred=np.zeros(10),
            fold_scores=[0.9],
            oof_score=0.9,
        )

    monkeypatch.setattr(mod, "run_experiment", fake_run)
    checked = mod.leak_check(
        real,
        "code",
        y,
        folds,
        METRICS["auc"],
        config=ctx.config,
        profile=profile,
        workdir=ctx.paths.run_dir / "x",
        sandbox_train=ctx.paths.run_dir,
        sandbox_test=ctx.paths.run_dir,
        reference=0.80,
        agent="a",
        round_no=1,
    )
    assert not checked.ok and checked.kind == "leak" and "shuffled targets" in checked.error
    assert len(calls) == 1 and not np.array_equal(calls[0], y) and sorted(calls[0]) == sorted(y)

    # below the threshold: no re-run at all
    calls.clear()
    kept = mod.leak_check(
        real,
        "code",
        y,
        folds,
        METRICS["auc"],
        config=ctx.config,
        profile=profile,
        workdir=ctx.paths.run_dir / "x",
        sandbox_train=ctx.paths.run_dir,
        sandbox_test=ctx.paths.run_dir,
        reference=0.97,
        agent="a",
        round_no=1,
    )
    assert kept is real and not calls

    # a jump that does not survive shuffling is legitimate
    def honest_run(code, **kw):
        return ExperimentResult(
            True,
            "ok",
            oof=np.zeros(len(y)),
            test_pred=np.zeros(10),
            fold_scores=[0.5],
            oof_score=0.51,
        )

    monkeypatch.setattr(mod, "run_experiment", honest_run)
    assert mod.leak_check(
        real,
        "code",
        y,
        folds,
        METRICS["auc"],
        config=ctx.config,
        profile=profile,
        workdir=ctx.paths.run_dir / "x",
        sandbox_train=ctx.paths.run_dir,
        sandbox_test=ctx.paths.run_dir,
        reference=0.80,
        agent="a",
        round_no=1,
    ).ok
    assert abs(mod.chance_score(y, METRICS["auc"]) - 0.5) < 1e-9
