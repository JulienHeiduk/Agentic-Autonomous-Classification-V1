import io

import pytest
from rich.console import Console

from aac.baseline import run_baseline
from aac.config import load_config
from aac.context import BudgetExceeded
from aac.kaggle.api import KaggleError
from aac.ledger import Ledger
from tests.kaggle_fake import FakeKaggle, make_bundle


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("NV_KEY", "k")
    monkeypatch.setenv("KAGGLE_ACCESS_TOKEN", "KGAT_t")


def run(fake, tmp_path, config, **kw):
    kw.setdefault("submit", True)
    return run_baseline(
        config,
        runs_root=tmp_path / "runs",
        transport=fake.transport,
        sleep=lambda s: None,
        poll_interval=0,
        console=Console(file=io.StringIO(), width=160),
        **kw,
    )


def test_baseline_end_to_end(env, tmp_path, write_config, minimal_config):
    fake = FakeKaggle(pending_polls=2, public_score=0.5)
    config = load_config(write_config(minimal_config))
    result = run(fake, tmp_path, config)
    assert result.submitted and result.ref == 1001 and result.public_score == 0.5
    assert result.target == "purchased" and 0 < result.constant < 1
    assert result.train_score == pytest.approx(0.5)  # constant prediction -> AUC 0.5
    assert result.submission_path.read_text().startswith("id,purchased\n60,")
    assert fake.uploads["tok-1"] == result.submission_path.read_bytes()
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        run_row = ledger.get_run(result.run_id)
        assert run_row["status"] == "completed"
        [sub] = ledger.list_submissions(result.run_id)
        assert sub["public_score"] == 0.5 and sub["kaggle_ref"] == "1001"
        assert "constant purchased=" in sub["description"]
    assert (tmp_path / "runs" / "_data" / config.competition.slug / "train.parquet").exists()


def test_baseline_no_submit_touches_nothing(env, tmp_path, write_config, minimal_config):
    fake = FakeKaggle()
    result = run(fake, tmp_path, load_config(write_config(minimal_config)), submit=False)
    assert not result.submitted and result.ref is None
    assert not any("Submission" in c for c in fake.calls)
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        assert ledger.get_run(result.run_id)["status"] == "completed"
        assert ledger.list_submissions(result.run_id) == []


def test_baseline_respects_daily_budget(env, tmp_path, write_config, minimal_config):
    from datetime import UTC, datetime

    today = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    used = [{"ref": i, "date": today, "status": "COMPLETE"} for i in range(5)]
    fake = FakeKaggle(submissions=used)
    minimal_config["kaggle"] = {"max_submissions_per_day": 5}
    config = load_config(write_config(minimal_config))
    with pytest.raises(BudgetExceeded, match="5 of 5"):
        run(fake, tmp_path, config)
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        [row] = ledger.list_runs()
        assert row["status"] == "failed" and "BudgetExceeded" in row["error"]
    assert not fake.uploads


def test_baseline_refuses_when_not_entered(env, tmp_path, write_config, minimal_config):
    fake = FakeKaggle(user_has_entered=False)
    with pytest.raises(KaggleError, match="accept the competition rules"):
        run(fake, tmp_path, load_config(write_config(minimal_config)))


def test_baseline_reports_kaggle_side_failure(env, tmp_path, write_config, minimal_config):
    fake = FakeKaggle(fail_submission="Evaluation exception", pending_polls=0)
    with pytest.raises(KaggleError, match="Evaluation exception"):
        run(fake, tmp_path, load_config(write_config(minimal_config)))
    with Ledger(tmp_path / "runs" / "ledger.db") as ledger:
        [row] = ledger.list_runs()
        assert row["status"] == "failed"
        [sub] = ledger.list_submissions(row["id"])
        assert sub["public_score"] is None


def test_baseline_with_string_labels_submits_positive_rate(
    env, tmp_path, write_config, minimal_config
):
    fake = FakeKaggle(bundle=make_bundle(labels=("No", "Yes")), pending_polls=0, public_score=0.5)
    result = run(fake, tmp_path, load_config(write_config(minimal_config)))
    assert result.submitted and 0 < result.constant < 1
    assert result.train_score == pytest.approx(0.5)
    body = result.submission_path.read_text().splitlines()
    assert body[0] == "id,purchased" and float(body[1].split(",")[1]) == pytest.approx(
        result.constant
    )


def test_baseline_label_metric_submits_majority_class(env, tmp_path, write_config, minimal_config):
    fake = FakeKaggle(bundle=make_bundle(labels=("No", "Yes")), metric="Accuracy", pending_polls=0)
    result = run(fake, tmp_path, load_config(write_config(minimal_config)))
    assert result.constant in ("No", "Yes")
    assert 0.4 < result.train_score <= 1.0
    assert result.submission_path.read_text().splitlines()[1].endswith(f",{result.constant}")
