import numpy as np
import pandas as pd

from aac.agents.analyst import analyse, calibration, worst_slices
from aac.exec.sandbox import ExperimentResult
from aac.models.metrics import METRICS


def make(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(scale=0.15, size=n), 0.001, 0.999)
    cat = rng.choice(["a", "b", "c"], n)
    p[cat == "c"] = rng.random((cat == "c").sum())  # the model is noise on slice c
    frame = pd.DataFrame({"cat": cat, "flag": rng.choice([True, False, None], n)})
    folds = rng.integers(0, 4, n)
    result = ExperimentResult(
        True,
        "ok",
        oof=p,
        test_pred=p[:100],
        fold_scores=[0.91, 0.92, 0.9, 0.93],
        oof_score=0.915,
        fold_seconds=[1.0, 1.2, 1.1, 1.0],
    )
    return y, p, frame, folds, result


def test_calibration_shape():
    y, p, *_ = make()
    cal = calibration(y, p)
    assert abs(cal["positive_rate"] - y.mean()) < 1e-12 and 0 < cal["brier"] < 0.25
    assert len(cal["bins"]) == 5 and sum(b["n"] for b in cal["bins"]) == len(y)


def test_worst_slices_find_the_noisy_category():
    y, p, frame, *_ = make()
    slices = worst_slices(y, p, frame, METRICS["auc"])
    assert slices and slices[0]["column"] == "cat" and slices[0]["value"] == "c"
    assert slices[0]["score"] < 0.6 and all(s["n"] >= 100 for s in slices)
    assert worst_slices(y, p, None, METRICS["auc"]) == []


def test_analyse_text_and_data():
    y, p, frame, folds, result = make()
    blend = np.clip(p + 0.01, 0, 1)
    a = analyse(
        result,
        y,
        folds,
        METRICS["auc"],
        slices=frame,
        blend_oof=blend,
        blend_score=0.92,
        leader_score=0.93,
    )
    assert a.text.startswith("Analyst notes on your last experiment (OOF ROC AUC 0.91500)")
    assert "- folds:" in a.text and "calibration" in a.text and "weakest slices: cat=c" in a.text
    assert "adds little diversity" in a.text and "behind" in a.text
    assert a.data["corr_with_blend"] > 0.99 and a.data["vs_leader"] < 0
    failed = analyse(ExperimentResult(False, "error", "boom"), y, folds, METRICS["auc"])
    assert "nothing to analyse" in failed.text
