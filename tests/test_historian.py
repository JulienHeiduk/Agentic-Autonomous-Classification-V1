from aac.agents.historian import (
    error_tail,
    experiment_module_path,
    pitfalls,
    priors,
    record_pitfalls,
    record_priors,
)
from aac.ledger import Ledger
from aac.models.metrics import METRICS
from tests.test_harness import LIGHTGBM_WITH_FEATURES, LOGISTIC


def _module(runs, run_id, agent, round_no, code):
    path = runs / run_id / "experiments" / agent / f"r{round_no}" / "experiment.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return path


def test_priors_from_an_earlier_run(tmp_path):
    runs = tmp_path / "runs"
    ledger = Ledger(runs / "ledger.db")
    ledger.create_run("old", "slug", "h", None)
    ledger.create_run("new", "slug", "h", None)
    ledger.record_experiment(
        "old-r01-gbdt-nvidia-r2",
        "old",
        agent="r01-gbdt-nvidia",
        round=2,
        status="ok",
        model="m",
        track="gbdt",
        hypothesis="cabin deck",
        oof_score=0.81,
    )
    ledger.record_experiment(
        "old-r02-open-nvidia-r1",
        "old",
        agent="r02-open-nvidia",
        round=1,
        status="ok",
        model="m2",
        track="open",
        hypothesis="plain",
        oof_score=0.79,
    )
    ledger.record_experiment(
        "new-r01-gbdt-nvidia-r1",
        "new",
        agent="r01-gbdt-nvidia",
        round=1,
        status="ok",
        oof_score=0.99,
    )
    module = _module(
        runs, "old", "r01-gbdt-nvidia", 2, "def fit_predict(a, b, c, d, e):\n    return c, d\n"
    )
    assert experiment_module_path(runs, "old-r01-gbdt-nvidia-r2", "old") == module
    assert experiment_module_path(runs, "old-r09-x-r9", "old") is None

    text = priors(ledger, runs, "slug", METRICS["auc"], exclude_run="new")
    assert text.startswith("Prior knowledge from earlier runs on slug")
    assert "0.81000 by m on track gbdt (old): cabin deck" in text and "0.99" not in text
    assert "def fit_predict" in text and text.index("0.81000") < text.index("0.79000")
    assert record_priors(ledger, runs, "slug", METRICS["auc"], "new")
    [note] = ledger.list_notes("new", kind="prior")
    assert "cabin deck" in note["text"] and note["track"] is None
    assert priors(ledger, runs, "other", METRICS["auc"]) is None
    assert not record_priors(ledger, runs, "other", METRICS["auc"], "new")


def test_priors_per_track_never_show_a_module_outside_the_track(tmp_path):
    runs = tmp_path / "runs"
    ledger = Ledger(runs / "ledger.db")
    ledger.create_run("old", "slug", "h", None)
    ledger.create_run("new", "slug", "h", None)
    # the best experiment is lightgbm on the open track; a logistic one trails it
    ledger.record_experiment(
        "old-r03-open-nvidia-r1",
        "old",
        agent="r03-open-nvidia",
        round=1,
        status="ok",
        model="ultra",
        track="open",
        hypothesis="tuned lightgbm with ratios",
        oof_score=0.9416,
    )
    _module(runs, "old", "r03-open-nvidia", 1, LIGHTGBM_WITH_FEATURES)
    ledger.record_experiment(
        "old-r04-linear-local-r2",
        "old",
        agent="r04-linear-local",
        round=2,
        status="ok",
        model="qwen",
        track="linear",
        hypothesis="logistic with polynomial features",
        oof_score=0.9381,
    )
    _module(runs, "old", "r04-linear-local", 2, LOGISTIC)
    # a track-labelled experiment whose module is gone falls back to its label
    ledger.record_experiment(
        "old-r02-neural-nvidia-r3",
        "old",
        agent="r02-neural-nvidia",
        round=3,
        status="ok",
        model="super",
        track="neural",
        hypothesis="mlp with embeddings",
        oof_score=0.9383,
    )
    # a degenerate failure with a score never counts as prior knowledge
    ledger.record_experiment(
        "old-r02-neural-nvidia-r4",
        "old",
        agent="r02-neural-nvidia",
        round=4,
        status="failed",
        kind="degenerate",
        track="neural",
        oof_score=0.5,
    )

    linear = priors(ledger, runs, "slug", METRICS["auc"], exclude_run="new", track="linear")
    assert "for the 'linear' track" in linear
    assert "logistic with polynomial" in linear and "LogisticRegression" in linear
    assert "lightgbm" not in linear and "mlp" not in linear
    gbdt = priors(ledger, runs, "slug", METRICS["auc"], exclude_run="new", track="gbdt")
    assert "tuned lightgbm" in gbdt and "import lightgbm" in gbdt
    assert "logistic with polynomial" not in gbdt, "no boosting library: rejected on gbdt"
    assert "mlp" not in gbdt
    neural = priors(ledger, runs, "slug", METRICS["auc"], exclude_run="new", track="neural")
    assert "mlp with embeddings" in neural and "lightgbm" not in neural
    assert "logistic with polynomial" not in neural and "0.5" not in neural.split("\n")[1]
    assert "def fit_predict" not in neural, "the mlp module is gone from disk: no code block"
    opened = priors(ledger, runs, "slug", METRICS["auc"], exclude_run="new", track="open")
    assert "tuned lightgbm" in opened and "mlp" in opened and "for the" not in opened

    assert record_priors(
        ledger, runs, "slug", METRICS["auc"], "new", tracks=["gbdt", "linear", "neural"]
    )
    notes = ledger.list_notes("new", kind="prior")
    assert sorted(n["track"] for n in notes) == ["gbdt", "linear", "neural"]
    seen_by_linear = ledger.list_notes("new", track="linear")
    assert [n["track"] for n in seen_by_linear] == ["linear"]
    assert "import lightgbm" not in seen_by_linear[0]["text"]


def test_pitfalls_list_distinct_failures_with_counts(tmp_path):
    runs = tmp_path / "runs"
    ledger = Ledger(runs / "ledger.db")
    ledger.create_run("old", "slug", "h", None)
    ledger.create_run("older", "slug", "h", None)
    ledger.create_run("new", "slug", "h", None)
    n_jobs = (
        'Traceback (most recent call last):\n  File "experiment.py", line 9, in fit_predict\n'
        "    gb = HistGradientBoostingClassifier(**p)\n         ^^^^^^^^^^^^^^^^^^^^^^^^^\n"
        "TypeError: HistGradientBoostingClassifier.__init__() got an unexpected keyword "
        "argument 'n_jobs'\n"
    )
    for run_id, agent in (("older", "r03-open-nvidia"), ("old", "r01-gbdt-nvidia")):
        ledger.record_experiment(
            f"{run_id}-{agent}-r4",
            run_id,
            agent=agent,
            round=4,
            status="failed",
            kind="error",
            track=agent.split("-")[1],
            hypothesis="hist gradient boosting",
            error=n_jobs,
        )
    ledger.record_experiment(
        "old-r01-gbdt-nvidia-r3",
        "old",
        agent="r01-gbdt-nvidia",
        round=3,
        status="failed",
        kind="timeout",
        track="gbdt",
        hypothesis="CatBoost with 3-seed averaging on all folds",
        error="the experiment did not finish within 1800s and was killed",
    )
    for round_no in (1, 2):
        ledger.record_experiment(
            f"old-r04-linear-local-r{round_no}",
            "old",
            agent="r04-linear-local",
            round=round_no,
            status="failed",
            kind="track",
            track="linear",
            error="track violation:\n- track 'linear' forbids ['lightgbm', 'torch']",
        )
    ledger.record_experiment(
        "old-r02-neural-nvidia-r1",
        "old",
        agent="r02-neural-nvidia",
        round=1,
        status="failed",
        kind="no-code",
        track="neural",
        error="I would suggest...",
    )
    ledger.record_experiment(
        "new-r01-gbdt-nvidia-r1",
        "new",
        agent="r01-gbdt-nvidia",
        round=1,
        status="failed",
        kind="error",
        track="gbdt",
        error="ValueError: from the current run, excluded",
    )

    text = pitfalls(ledger, "slug", exclude_run="new")
    assert text.startswith("Pitfalls from earlier runs on slug")
    lines = text.splitlines()[1:]
    assert len(lines) == 2, text
    [error_line] = [ln for ln in lines if ln.startswith("- [error, 2x, last old on gbdt] ")]
    assert "TypeError: HistGradient" in error_line and "n_jobs'" in error_line
    assert "^^^" not in error_line and "File " not in error_line
    [timeout_line] = [ln for ln in lines if ln.startswith("- [timeout, 1x, last old on gbdt] ")]
    assert "did not finish within 1800s" in timeout_line
    assert "(hypothesis: CatBoost with 3-seed" in timeout_line
    assert "I would suggest" not in text and "excluded" not in text and "forbids" not in text

    linear = pitfalls(ledger, "slug", exclude_run="new", track="linear")
    assert linear.startswith("Track pitfalls for 'linear' on slug: 2 earlier modules")
    assert (
        "[track, 2x, last old on linear] - track 'linear' forbids ['lightgbm', 'torch']" in linear
    )
    assert pitfalls(ledger, "slug", exclude_run="new", track="gbdt") is None
    assert pitfalls(ledger, "other") is None

    assert record_pitfalls(ledger, "slug", "new", tracks=["gbdt", "linear"])
    notes = ledger.list_notes("new", kind="pitfalls")
    assert [n["track"] for n in notes] == [None, "linear"]
    assert [n["track"] for n in ledger.list_notes("new", kind="pitfalls", track="gbdt")] == [None]
    assert not record_pitfalls(ledger, "other", "new")

    assert error_tail("") == "(no error text)"
    assert error_tail('a\n    ^^^^\n  File "x.py", line 1\n') == "a"
    assert error_tail("x" * 500).endswith("x") and len(error_tail("x" * 500)) == 220
