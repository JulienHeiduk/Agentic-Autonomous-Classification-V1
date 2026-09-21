import pytest

from aac.config import load_config
from aac.context import Budget, BudgetExceeded, RunContext
from aac.ledger import Ledger


def test_budget_tokens_and_clock():
    b = Budget(max_tokens=10, max_seconds=1000)
    assert b.exhausted() is None and b.remaining_tokens == 10
    assert b.charge(4, 4) == 8
    b.check()
    b.charge(1, 1)
    assert "token" in (b.exhausted() or "")
    with pytest.raises(BudgetExceeded, match="token"):
        b.check()
    clock = Budget(max_tokens=1, max_seconds=0.0)
    assert "wall clock" in (clock.exhausted() or "")
    assert clock.remaining_seconds == 0.0


def test_run_context_create_snapshots_config_and_registers_run(write_config, minimal_config):
    config = load_config(write_config(minimal_config), env={"NV_KEY": "sekrit"})
    runs_root = write_config.__self__ if hasattr(write_config, "__self__") else None
    root = config.source.parent / "runs"
    ctx = RunContext.create(config, runs_root=root)
    try:
        assert ctx.paths.run_dir.is_dir()
        snapshot = ctx.paths.config_snapshot.read_text()
        assert "sekrit" not in snapshot and "m-nv" in snapshot
        row = ctx.ledger.get_run(ctx.run_id)
        assert row["slug"] == "playground-series-s6e9"
        assert row["config_hash"] == config.fingerprint()
        assert row["config_path"] == str(config.source)
        assert ctx.budget.max_tokens == config.run.max_tokens_total
        assert ctx.budget.max_seconds == config.run.max_wall_clock_minutes * 60
        ctx.finish("completed")
    finally:
        ctx.close()
    with Ledger(root / "ledger.db") as ledger:
        assert ledger.get_run(ctx.run_id)["status"] == "completed"
    del runs_root
