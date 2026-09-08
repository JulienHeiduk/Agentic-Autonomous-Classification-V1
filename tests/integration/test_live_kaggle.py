"""Live Kaggle integration (README section 6): submit a constant baseline, get a score back.

Skipped unless AAC_INTEGRATION=1. Uses one of the day's submissions on the configured
competition, so run it deliberately, not in CI.
"""

import io
import os
from pathlib import Path

import pytest
from rich.console import Console

from aac.baseline import run_baseline
from aac.config import load_config
from aac.env import load_secrets

pytestmark = pytest.mark.skipif(
    os.environ.get("AAC_INTEGRATION") != "1", reason="set AAC_INTEGRATION=1 to hit Kaggle"
)

REPO = Path(__file__).resolve().parents[2]


def test_constant_baseline_round_trip(tmp_path):
    load_secrets(REPO / "secrets.yml")
    config = load_config(REPO / "configs" / "s6e9.yaml")
    result = run_baseline(
        config,
        runs_root=tmp_path / "runs",
        poll_timeout=900,
        console=Console(file=io.StringIO()),
    )
    assert result.submitted and result.ref
    assert result.status == "COMPLETE"
    assert result.public_score is not None
    assert result.public_score == pytest.approx(0.5, abs=1e-6), "constant prediction scores 0.5 AUC"
