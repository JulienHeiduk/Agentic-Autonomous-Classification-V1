import io
import json

import httpx
import pytest
from rich.console import Console

from aac.config import load_config
from aac.kaggle.api import KaggleError
from aac.ledger import Ledger
from aac.orchestrator import DEFAULT_BRANCH_NAME, OrchestratorError, branch_ledger_id, run
from tests.kaggle_fake import FakeKaggle
from tests.llm_fake import FakeLLM, combined_transport
from tests.synth import bundle_from_frames, make_frames
from tests.test_harness import LOGISTIC


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("NV_KEY", "k")
    monkeypatch.setenv("KAGGLE_ACCESS_TOKEN", "KGAT_t")


def fast_config(write_config, minimal_config, **overrides):
    minimal_config["models"] = {"enabled": ["logistic", "lightgbm"], "tuning": "none"}
    minimal_config["run"] = {"n_folds": 4, "n_jobs": 2}
    minimal_config.update(overrides)
    return load_config(write_config(minimal_config))


def go(fake, tmp_path, config, **kw):
    return run(
        config,
        runs_root=tmp_path / "runs",
        transport=fake.transport,
        sleep=lambda s: None,
        poll_interval=0,
        console=Console(file=io.StringIO(), width=160),
        **kw,
    )


def test_end_to_end_label_competition(env, tmp_path, write_config, minimal_config):
    train, test, sample = make_frames(400, 150, labels=(False, True), sample_kind="label")
    fake = FakeKaggle(
        bundle=bundle_from_frames(train, test, sample),
        metric="Categorization Accuracy",
        pending_polls=1,
        public_score=0.83,
    )
    config = fast_config(write_config, minimal_config)
    summary = go(fake, tmp_path, config, submit=True)
    assert summary.status == "completed" and summary.public_score == 0.83
    assert summary.profile.submission.kind == "label" and summary.training.best.oof_score > 0.8
    run_dir = summary.run_dir
    assert (run_dir / "profile.json").exists() and (run_dir / "folds.npy").exists()
    branch = run_dir / "branches" / DEFAULT_BRANCH_NAME
    assert (branch / "plan.json").exists() and (branch / "metrics.json").exists()
    lines = (run_dir / "submission.csv").read_text().splitlines()
    assert lines[0] == "id,target" and set(line.split(",")[1] for line in lines[1:]) <= {
        "True",
        "False",
    }
    assert fake.uploads["tok-1"] == (run_dir / "submission.csv").read_bytes()
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        assert ledger.get_run(summary.run_id)["status"] == "completed"
        branches = ledger.list_branches(summary.run_id)
        b = branches[0]
        assert b["id"] == branch_ledger_id(summary.run_id, DEFAULT_BRANCH_NAME)
        assert b["status"] == "promoted" and b["cv_mean"] > 0.8 and len(b["fold_scores"]) == 4
        assert b["plan_json"]["name"] == "default" and b["plan_hash"] == summary.plan.hash()
        # deterministic variants (lightgbm only: xgboost is not enabled here) and seed bags
        names = [x.name for x in summary.branches]
        assert names[:3] == [DEFAULT_BRANCH_NAME, "b01-categorical", "b02-encoded"]
        assert [n.rsplit("-", 1)[-1] for n in names[3:]] == ["s43", "s43", "s44", "s44"]
        assert all(x.status == "promoted" for x in summary.branches)
        assert summary.branches[1].plan.categorical_columns == ["cat", "flag", "x3"]
        assert summary.branches[2].plan.target_encode == ["cat", "flag", "x3"]
        assert {x.seed for x in summary.branches[3:]} == {43, 44}
        assert all(set(x.training.results) == {"lightgbm"} for x in summary.branches[3:])
        assert len(branches) == 7 and all(r["status"] == "promoted" for r in branches)
        assert len(summary.ensemble.members) == 2 + 2 + 4
        [sub] = ledger.list_submissions(summary.run_id)
        assert sub["public_score"] == 0.83 and sub["oof_score"] == summary.ensemble.best.oof_score


def test_proba_competition_no_submit_and_dry_run(env, tmp_path, write_config, minimal_config):
    train, test, sample = make_frames(300, 100)
    fake = FakeKaggle(bundle=bundle_from_frames(train, test, sample), metric="Roc Auc Score")
    config = fast_config(write_config, minimal_config)
    summary = go(fake, tmp_path, config, submit=False)
    assert summary.status == "completed" and summary.public_score is None
    assert not any("Submission" in c for c in fake.calls)
    values = [
        float(line.split(",")[1])
        for line in (summary.run_dir / "submission.csv").read_text().splitlines()[1:]
    ]
    assert all(0 <= v <= 1 for v in values)

    dry = go(fake, tmp_path, config, dry_run=True)
    assert dry.status == "completed" and dry.training is None and dry.submission_path is None
    assert (dry.run_dir / "branches" / DEFAULT_BRANCH_NAME / "plan.json").exists()
    assert not (dry.run_dir / "branches" / DEFAULT_BRANCH_NAME / "metrics.json").exists()
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        [b] = ledger.list_branches(dry.run_id)
        assert b["status"] == "training" and b["cv_mean"] is None


def test_same_run_twice_is_identical(env, tmp_path, write_config, minimal_config):
    train, test, sample = make_frames(300, 100)
    fake = FakeKaggle(bundle=bundle_from_frames(train, test, sample), metric="Roc Auc Score")
    config = fast_config(write_config, minimal_config)
    a = go(fake, tmp_path, config, submit=False)
    b = go(fake, tmp_path, config, submit=False)
    assert a.training.best.fold_scores == b.training.best.fold_scores
    assert (a.run_dir / "submission.csv").read_bytes() == (
        b.run_dir / "submission.csv"
    ).read_bytes()
    assert json.loads((a.run_dir / "profile.json").read_text()) == json.loads(
        (b.run_dir / "profile.json").read_text()
    )


def test_code_competition_and_bad_metric_are_refused(env, tmp_path, write_config, minimal_config):
    train, test, sample = make_frames(100, 40)
    bundle = bundle_from_frames(train, test, sample)
    fake = FakeKaggle(bundle=bundle, metric="Roc Auc Score", code_only=True)
    config = fast_config(write_config, minimal_config)
    with pytest.raises(OrchestratorError, match="code competition"):
        go(fake, tmp_path, config)
    fake = FakeKaggle(bundle=bundle, metric="Quadratic Weighted Kappa")
    with pytest.raises(Exception, match="not implemented"):
        go(fake, tmp_path, config)
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        assert {r["status"] for r in ledger.list_runs()} == {"failed"}
        assert all(r["error"] for r in ledger.list_runs())


def test_not_entered_fails_at_upload_after_training(env, tmp_path, write_config, minimal_config):
    train, test, sample = make_frames(200, 60)
    fake = FakeKaggle(
        bundle=bundle_from_frames(train, test, sample),
        metric="Roc Auc Score",
        user_has_entered=False,
    )
    config = fast_config(write_config, minimal_config)
    with pytest.raises(KaggleError, match="accept the competition rules"):
        go(fake, tmp_path, config, submit=True)
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        [r] = ledger.list_runs()
        branches = ledger.list_branches(r["id"])
        assert r["status"] == "failed" and len(branches) == 7, "default, 2 variants, 4 seed bags"
        assert all(b["status"] == "promoted" for b in branches), "training survived, upload failed"


def llm_config(write_config, minimal_config, **run_overrides):
    minimal_config["models"] = {"enabled": ["logistic", "lightgbm"], "tuning": "none"}
    minimal_config["run"] = {"n_folds": 4, "n_jobs": 2, **run_overrides}
    minimal_config["researchers"] = [{"backend": "nvidia", "track": "open", "temperature": 0.3}]
    minimal_config["sandbox"] = {"timeout_seconds": 60}
    return load_config(write_config(minimal_config))


def go_llm(kaggle, llm, tmp_path, config, **kw):
    return run(
        config,
        runs_root=tmp_path / "runs",
        transport=combined_transport(kaggle, llm),
        sleep=lambda s: None,
        poll_interval=0,
        console=Console(file=io.StringIO(), width=160),
        **kw,
    )


def test_researcher_end_to_end_wins_submission(env, tmp_path, write_config, minimal_config):
    train, test, sample = make_frames(300, 100)
    kaggle = FakeKaggle(
        bundle=bundle_from_frames(train, test, sample),
        metric="Roc Auc Score",
        pending_polls=0,
        public_score=0.9,
    )
    llm = FakeLLM({"nv.test": [f"HYPOTHESIS: logistic\n```python\n{LOGISTIC}\n```"] * 3})
    minimal_config["models"] = {"enabled": ["logistic"], "tuning": "none"}
    minimal_config["run"] = {
        "n_folds": 4,
        "n_jobs": 2,
        "max_rounds": 2,
        "patience": 1,
    }
    minimal_config["researchers"] = [{"backend": "nvidia", "track": "linear"}]
    config = load_config(write_config(minimal_config))
    summary = go_llm(kaggle, llm, tmp_path, config, submit=True)
    assert summary.status == "completed" and summary.public_score == 0.9
    [out] = summary.researchers
    assert out.agent == "r01-linear-nvidia" and out.best is not None and out.n_ok >= 1
    assert summary.best_candidate is not None
    assert summary.best_candidate.origin in ("experiment", "branch")
    assert summary.ensemble is not None and (summary.run_dir / "ensemble.json").exists()
    first_prompt = llm.calls("nv.test")[0]["messages"][-1]["content"]
    assert "[competition from kaggle] Predicting Electric Vehicle Purchases" in first_prompt
    assert summary.ensemble.best.oof_score >= summary.best_candidate.oof_score - 1e-12
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        rows = ledger.list_experiments(summary.run_id)
        assert rows and rows[0]["agent"] == "r01-linear-nvidia" and rows[0]["track"] == "linear"
        [sub] = ledger.list_submissions(summary.run_id)
        assert sub["oof_score"] == summary.ensemble.best.oof_score
        [note] = ledger.list_notes(summary.run_id, kind="ensemble")
        assert "members" in note["text"]
    assert (summary.run_dir / "experiments" / "r01-linear-nvidia" / "r1" / "experiment.py").exists()


def test_budget_exhaustion_stops_new_work_but_still_submits(
    env, tmp_path, write_config, minimal_config
):
    train, test, sample = make_frames(200, 60)
    kaggle = FakeKaggle(
        bundle=bundle_from_frames(train, test, sample),
        metric="Roc Auc Score",
        pending_polls=0,
        public_score=0.7,
    )
    llm = FakeLLM({"nv.test": [f"HYPOTHESIS: x\n```python\n{LOGISTIC}\n```"] * 4})
    minimal_config["models"] = {"enabled": ["logistic"], "tuning": "none"}
    minimal_config["run"] = {
        "n_folds": 4,
        "n_jobs": 2,
        "max_rounds": 2,
        "max_wall_clock_minutes": 0.0001,  # expires before the first Researcher round
    }
    minimal_config["researchers"] = [{"backend": "nvidia", "track": "linear"}]
    config = load_config(write_config(minimal_config))
    summary = go_llm(kaggle, llm, tmp_path, config, submit=True)
    assert summary.status == "completed" and summary.public_score == 0.7
    assert summary.best_candidate.origin == "branch"
    assert not llm.calls("nv.test"), "no experiment was started once the clock had expired"
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        [note] = ledger.list_notes(summary.run_id, kind="budget")
        assert "wall clock" in note["text"]


def test_missing_researcher_model_fails_loudly_at_startup(
    env, tmp_path, write_config, minimal_config
):
    train, test, sample = make_frames(100, 40)
    kaggle = FakeKaggle(bundle=bundle_from_frames(train, test, sample), metric="Roc Auc Score")
    llm = FakeLLM(models={"local.test": ["m-local"], "nv.test": ["other"]})
    config = llm_config(write_config, minimal_config)
    with pytest.raises(OrchestratorError, match="not served"):
        go_llm(kaggle, llm, tmp_path, config, submit=False)


def test_dry_run_skips_researchers(env, tmp_path, write_config, minimal_config):
    train, test, sample = make_frames(100, 40)
    kaggle = FakeKaggle(bundle=bundle_from_frames(train, test, sample), metric="Roc Auc Score")
    llm = FakeLLM()
    config = llm_config(write_config, minimal_config)
    summary = go_llm(kaggle, llm, tmp_path, config, dry_run=True)
    assert (
        summary.status == "completed"
        and summary.researchers == []
        and summary.best_candidate is None
    )
    assert not llm.calls("nv.test")


def test_uploads_during_the_run_when_the_blend_improves(
    env, tmp_path, write_config, minimal_config
):
    from tests.test_harness import LIGHTGBM_WITH_FEATURES

    train, test, sample = make_frames(300, 100)
    kaggle = FakeKaggle(
        bundle=bundle_from_frames(train, test, sample),
        metric="Roc Auc Score",
        pending_polls=0,
        public_score=0.9,
    )
    llm = FakeLLM(
        {
            "nv.test": [
                f"HYPOTHESIS: lgb\n```python\n{LIGHTGBM_WITH_FEATURES}\n```",
                f"HYPOTHESIS: logistic\n```python\n{LOGISTIC}\n```",
            ]
        }
    )
    minimal_config["models"] = {"enabled": ["logistic"], "tuning": "none"}
    minimal_config["run"] = {"n_folds": 4, "n_jobs": 2, "max_rounds": 2, "patience": 3}
    minimal_config["researchers"] = [{"backend": "nvidia", "track": "open"}]
    config = load_config(write_config(minimal_config))
    summary = go_llm(kaggle, llm, tmp_path, config, submit=True)
    assert summary.status == "completed"
    assert summary.uploads >= 1 and summary.public_score == 0.9
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        subs = ledger.list_submissions(summary.run_id)
        assert len(subs) == summary.uploads
        scores = [s["oof_score"] for s in reversed(subs)]
        assert scores == sorted(scores), "each upload beat the previous one"
        history = summary.ensemble
        assert history is not None
        assert subs[0]["oof_score"] == summary.ensemble.best.oof_score
    assert len(kaggle.uploads) == summary.uploads


def test_scholar_packets_and_priors_reach_researchers(env, tmp_path, write_config, minimal_config):
    from tests.test_scholar import PACKET

    train, test, sample = make_frames(200, 60)
    kaggle = FakeKaggle(bundle=bundle_from_frames(train, test, sample), metric="Roc Auc Score")
    packet = json.dumps(PACKET)
    logistic = f"HYPOTHESIS: logistic\n```python\n{LOGISTIC}\n```"
    llm = FakeLLM({"nv.test": [packet, packet, packet, logistic, logistic]})
    minimal_config["models"] = {"enabled": ["logistic"], "tuning": "none"}
    minimal_config["run"] = {"n_folds": 4, "n_jobs": 2, "max_rounds": 1}
    minimal_config["researchers"] = [{"backend": "nvidia", "track": "linear"}]
    minimal_config["scholar"] = {"backend": "nvidia", "max_ideas": 2}
    config = load_config(write_config(minimal_config))
    first = go_llm(kaggle, llm, tmp_path, config, submit=False)
    researcher_calls = [
        c for c in llm.calls("nv.test") if "Round 1 of" in c["messages"][-1]["content"]
    ]
    assert researcher_calls, "the researcher was prompted"
    prompt = researcher_calls[0]["messages"][-1]["content"]
    assert "[research from scholar] Research packet" in prompt and "x1 times x2" in prompt
    assert "[prior from historian]" not in prompt, "no earlier run yet"
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        assert len(ledger.list_notes(first.run_id, kind="research")) == 3

    # a second run on the same competition sees the first run's best experiment as a prior,
    # and reuses its research packets instead of asking the Scholar again
    llm2 = FakeLLM({"nv.test": [logistic]})
    second = go_llm(kaggle, llm2, tmp_path, config, submit=False)
    assert all("Round 1 of" in c["messages"][-1]["content"] for c in llm2.calls("nv.test"))
    prompt2 = llm2.calls("nv.test")[0]["messages"][-1]["content"]
    assert "[research from scholar] Research packet" in prompt2 and "x1 times x2" in prompt2
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        reused = ledger.list_notes(second.run_id, kind="research")
        assert len(reused) == 3 and {n["origin_run"] for n in reused} == {first.run_id}
    assert "[prior from historian] Prior knowledge from earlier runs" in prompt2
    assert "for the 'linear' track" in prompt2, "priors are addressed per track"
    assert "def fit_predict" in prompt2 and first.run_id in prompt2
    assert "[pitfalls from historian]" not in prompt2, "the first run had no failure"
    assert second.status == "completed"
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        [prior] = ledger.list_notes(second.run_id, kind="prior")
        assert prior["track"] == "linear"
        assert ledger.list_notes(second.run_id, kind="pitfalls") == []


def test_assessor_interviews_and_allocation(env, tmp_path, write_config, minimal_config):
    train, test, sample = make_frames(600, 100)
    kaggle = FakeKaggle(bundle=bundle_from_frames(train, test, sample), metric="Roc Auc Score")
    logistic = f"HYPOTHESIS: logistic\n```python\n{LOGISTIC}\n```"
    # nvidia answers the interview with a working module; local never produces code
    llm = FakeLLM({"nv.test": [logistic, logistic], "local.test": ["no code", "still none"] * 4})
    minimal_config["models"] = {"enabled": ["logistic"], "tuning": "none"}
    minimal_config["run"] = {"n_folds": 4, "n_jobs": 2, "max_rounds": 1}
    minimal_config["researchers"] = [
        {"backend": "nvidia", "track": "linear"},
        {"backend": "local", "track": "open"},
    ]
    minimal_config["assessor"] = {
        "enabled": True,
        "rows": 500,
        "n_folds": 3,
        "skip_after_failures": 2,
    }
    config = load_config(write_config(minimal_config))
    first = go_llm(kaggle, llm, tmp_path, config, submit=False)
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        records = {(r["backend"], r["model"]): r for r in ledger.track_records()}
        assert (
            records[("nvidia", "m-nv")]["ran_ok"] == 1
            and records[("nvidia", "m-nv")]["oof_score"] > 0.7
        )
        assert records[("local", "m-local")]["valid_module"] == 0
        [note] = ledger.list_notes(first.run_id, kind="track_record")
        assert "nvidia/m-nv: 1 interviews, 1 working" in note["text"]
    assert {o.agent for o in first.researchers} == {"r01-linear-nvidia", "r02-open-local"}
    assert (first.run_dir / "interview" / "train.parquet").exists()

    # second run: local now has 2 failed interviews -> skipped; nvidia is not re-interviewed
    llm2 = FakeLLM({"nv.test": [logistic], "local.test": ["no code", "still none"] * 4})
    second = go_llm(kaggle, llm2, tmp_path, config, submit=False)
    assert [o.agent for o in second.researchers] == ["r01-linear-nvidia"]
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        assert len(ledger.track_records(backend="nvidia")) == 1
        assert len(ledger.track_records(backend="local")) == 2
        [note] = ledger.list_notes(second.run_id, kind="track_record")
        assert "Skipped: local/m-local" in note["text"]
    assert not [c for c in llm2.calls("nv.test") if "interview" in c["messages"][-1]["content"]]


def test_final_upload_respects_the_daily_quota_and_submit_run_uploads_later(
    env, tmp_path, write_config, minimal_config
):
    from datetime import UTC, datetime

    from aac.orchestrator import submit_run

    train, test, sample = make_frames(200, 60)
    today = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    used = [{"ref": i, "date": today, "status": "COMPLETE"} for i in range(5)]
    kaggle = FakeKaggle(
        bundle=bundle_from_frames(train, test, sample),
        metric="Roc Auc Score",
        submissions=used,
        pending_polls=0,
        public_score=0.88,
    )
    config = fast_config(write_config, minimal_config, kaggle={"max_submissions_per_day": 5})
    summary = go(kaggle, tmp_path, config, submit=True)
    assert summary.status == "completed" and summary.uploads == 0 and summary.public_score is None
    assert summary.submission_path.exists()
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        notes = ledger.list_notes(summary.run_id, kind="budget")
        assert any("final upload skipped" in n["text"] for n in notes)

    kaggle.submissions = []  # the quota reset
    score = submit_run(
        config,
        summary.run_id,
        runs_root=tmp_path / "runs",
        transport=kaggle.transport,
        sleep=lambda s: None,
        poll_interval=0,
    )
    assert score == 0.88
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        [sub] = ledger.list_submissions(summary.run_id)
        assert sub["oof_score"] == summary.ensemble.best.oof_score and sub["public_score"] == 0.88


def test_resume_continues_rounds_with_history_and_pool(env, tmp_path, write_config, minimal_config):
    from aac.orchestrator import resume

    train, test, sample = make_frames(300, 100)
    kaggle = FakeKaggle(
        bundle=bundle_from_frames(train, test, sample),
        metric="Roc Auc Score",
        pending_polls=0,
        public_score=0.91,
    )
    logistic = f"HYPOTHESIS: logistic first\n```python\n{LOGISTIC}\n```"
    llm = FakeLLM({"nv.test": [logistic]})
    minimal_config["models"] = {"enabled": ["logistic"], "tuning": "none"}
    minimal_config["run"] = {"n_folds": 4, "n_jobs": 2, "max_rounds": 1, "patience": 3}
    minimal_config["researchers"] = [{"backend": "nvidia", "track": "linear"}]
    config_path = write_config(minimal_config)
    config = load_config(config_path)
    first = go_llm(kaggle, llm, tmp_path, config, submit=True)
    assert first.status == "completed" and len(first.researchers[0].experiments) == 1
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        ledger.set_run_status(first.run_id, "stopped", "simulated crash")
        uploads_before = len(ledger.list_submissions(first.run_id))

    minimal_config["run"]["max_rounds"] = 3
    config2 = load_config(write_config(minimal_config, "c2.yaml"))
    from tests.test_harness import LIGHTGBM_WITH_FEATURES

    llm2 = FakeLLM(
        {
            "nv.test": [
                f"HYPOTHESIS: lightgbm on resume\n```python\n{LIGHTGBM_WITH_FEATURES}\n```",
                f"HYPOTHESIS: logistic again\n```python\n{LOGISTIC}\n```",
            ]
        }
    )
    summary = resume(
        first.run_id,
        config=config2,
        runs_root=tmp_path / "runs",
        transport=combined_transport(kaggle, llm2),
        sleep=lambda s: None,
        poll_interval=0,
        console=Console(file=io.StringIO(), width=160),
    )
    assert summary.resumed and summary.status == "completed" and summary.run_id == first.run_id
    [out] = summary.researchers
    assert [e.round for e in out.experiments] == [1, 2, 3], "history reloaded, rounds continued"
    assert out.experiments[0].hypothesis == "logistic first" and out.experiments[0].ok
    prompt = llm2.calls("nv.test")[0]["messages"][-1]["content"]
    assert "Round 2 of 3" in prompt and "logistic first" in prompt, (
        "the resumed agent sees its history"
    )
    assert len(summary.ensemble.members) >= 3, "default families + reloaded + new experiments"
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        assert ledger.get_run(first.run_id)["status"] == "completed"
        rows = ledger.list_experiments(first.run_id)
        assert [r["round"] for r in rows] == [1, 2, 3]
        assert len(ledger.list_submissions(first.run_id)) >= uploads_before
    with pytest.raises(OrchestratorError, match="not in the ledger"):
        resume(
            "nope",
            config=config2,
            runs_root=tmp_path / "runs",
            transport=combined_transport(kaggle, llm2),
        )


def test_replay_reproduces_a_stored_experiment(env, tmp_path, write_config, minimal_config):
    from aac.orchestrator import replay_experiment

    train, test, sample = make_frames(200, 60)
    kaggle = FakeKaggle(bundle=bundle_from_frames(train, test, sample), metric="Roc Auc Score")
    llm = FakeLLM({"nv.test": [f"HYPOTHESIS: logistic\n```python\n{LOGISTIC}\n```"]})
    minimal_config["models"] = {"enabled": ["logistic"], "tuning": "none"}
    minimal_config["run"] = {"n_folds": 4, "n_jobs": 2, "max_rounds": 1}
    minimal_config["researchers"] = [{"backend": "nvidia", "track": "linear"}]
    config = load_config(write_config(minimal_config))
    summary = go_llm(kaggle, llm, tmp_path, config, submit=False)
    [experiment] = summary.researchers[0].experiments
    report = replay_experiment(
        experiment.id, config=config, runs_root=tmp_path / "runs", transport=kaggle.transport
    )
    assert report["ok"] and report["reproduced"] and report["max_abs_diff"] <= 1e-9
    assert report["oof_score"] == report["stored_oof_score"]
    with pytest.raises(OrchestratorError, match="not in the ledger"):
        replay_experiment(
            "nope", config=config, runs_root=tmp_path / "runs", transport=kaggle.transport
        )


def test_seats_take_turns_round_robin(env, tmp_path, write_config, minimal_config):
    train, test, sample = make_frames(300, 100)
    kaggle = FakeKaggle(bundle=bundle_from_frames(train, test, sample), metric="Roc Auc Score")
    logistic = f"HYPOTHESIS: logistic\n```python\n{LOGISTIC}\n```"
    minimal_config["models"] = {"enabled": ["logistic"], "tuning": "none"}
    minimal_config["run"] = {
        "n_folds": 4,
        "n_jobs": 2,
        "max_rounds": 2,
        "patience": 5,
        "parallel_branches": 1,
    }
    minimal_config["researchers"] = [
        {"backend": "nvidia", "track": "linear"},
        {"backend": "nvidia", "track": "open", "rounds": 1},
    ]
    config = load_config(write_config(minimal_config))
    llm = FakeLLM({"nv.test": [logistic] * 3})
    summary = go_llm(kaggle, llm, tmp_path, config, submit=False)
    rounds = [
        int(c["messages"][-1]["content"].rsplit("Round ", 1)[1].split(" ")[0])
        for c in llm.calls("nv.test")
    ]
    assert rounds == [1, 1, 2], "both seats play round 1 before the first seat plays round 2"
    assert [o.agent for o in summary.researchers] == ["r01-linear-nvidia", "r02-open-nvidia"]
    linear, opened = summary.researchers
    assert [e.round for e in linear.experiments] == [1, 2] and linear.n_ok == 2
    assert [e.round for e in opened.experiments] == [1] and opened.n_ok == 1
    assert linear.stopped_because == "rounds exhausted"
    assert opened.stopped_because == "rounds exhausted"
    assert "minutes of wall clock remain" in llm.calls("nv.test")[0]["messages"][-1]["content"]

    # the sequential schedule runs one seat to the end before the next starts
    minimal_config["run"]["schedule"] = "sequential"
    config = load_config(write_config(minimal_config, "seq.yaml"))
    llm = FakeLLM({"nv.test": [logistic] * 3})
    go_llm(kaggle, llm, tmp_path, config, submit=False)
    rounds = [
        int(c["messages"][-1]["content"].rsplit("Round ", 1)[1].split(" ")[0])
        for c in llm.calls("nv.test")
    ]
    assert rounds == [1, 2, 1]


def test_pending_score_does_not_fail_the_run_and_is_backfilled_later(
    env, tmp_path, write_config, minimal_config
):
    from aac.agents.submitter import backfill_public_scores
    from aac.kaggle.api import KaggleClient

    train, test, sample = make_frames(200, 60)
    # Kaggle keeps the upload PENDING for longer than the poll window
    kaggle = FakeKaggle(
        bundle=bundle_from_frames(train, test, sample),
        metric="Roc Auc Score",
        pending_polls=10**6,
        public_score=0.87,
    )
    config = fast_config(write_config, minimal_config)
    summary = go(kaggle, tmp_path, config, submit=True, poll_timeout=0.0)
    assert summary.status == "completed" and summary.uploads == 1
    assert summary.public_score is None, "not scored yet"
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        [sub] = ledger.list_submissions(summary.run_id)
        assert sub["public_score"] is None and sub["kaggle_ref"]
        assert ledger.get_run(summary.run_id)["status"] == "completed"
        [pending] = ledger.pending_submissions("playground-series-s6e9")
        assert pending["id"] == sub["id"]
        # the old behaviour left such a run failed; the backfill repairs that too
        ledger.set_run_status(
            summary.run_id, "failed", f"KaggleError: submission {sub['kaggle_ref']} still PENDING"
        )
        with KaggleClient.from_env(client=httpx.Client(transport=kaggle.transport)) as kc:
            assert backfill_public_scores(ledger, kc, "playground-series-s6e9") == []
            # Kaggle finishes scoring
            for row in kaggle.submissions:
                row["status"], row["publicScore"] = "COMPLETE", "0.87"
            [updated] = backfill_public_scores(ledger, kc, "playground-series-s6e9")
        assert updated["public_score"] == 0.87
        assert ledger.list_submissions(summary.run_id)[0]["public_score"] == 0.87
        assert ledger.pending_submissions("playground-series-s6e9") == []
        run_row = ledger.get_run(summary.run_id)
        assert run_row["status"] == "completed" and run_row["error"] is None

    # the next run on the competition writes the score back by itself
    kaggle2 = FakeKaggle(
        bundle=bundle_from_frames(train, test, sample),
        metric="Roc Auc Score",
        submissions=[
            {"ref": 77, "date": "2026-01-01T00:00:00Z", "status": "COMPLETE", "publicScore": "0.9"}
        ],
        pending_polls=0,
        public_score=0.91,
    )
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        [sub] = ledger.list_submissions(summary.run_id)
        ledger.set_public_score(sub["id"], None)
        ledger._conn.execute("UPDATE submissions SET kaggle_ref = '77' WHERE id = ?", (sub["id"],))
        ledger._conn.commit()
    second = go(kaggle2, tmp_path, config, submit=False)
    assert second.status == "completed"
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        assert ledger.list_submissions(summary.run_id)[0]["public_score"] == 0.9
