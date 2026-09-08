import numpy as np
import pytest

from aac.models.metrics import METRICS, UnsupportedMetricError, resolve_metric, score


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Roc Auc Score", "auc"),
        ("AUC", "auc"),
        ("Area Under Receiver Operating Characteristic Curve", "auc"),
        ("Accuracy", "accuracy"),
        ("LogLoss", "logloss"),
        ("Log Loss", "logloss"),
        ("F1 Score", "f1"),
        ("Macro F1", "macro_f1"),
        ("MacroF1Score", "macro_f1"),
    ],
)
def test_known_kaggle_names_resolve(name, key):
    assert resolve_metric(name).key == key


@pytest.mark.parametrize("name", ["Quadratic Weighted Kappa", "RMSE", "MAP@3", "", None])
def test_unknown_or_missing_names_are_refused(name):
    with pytest.raises(UnsupportedMetricError):
        resolve_metric(name)


def test_override_wins_and_is_validated():
    assert resolve_metric("Quadratic Weighted Kappa", override="auc").key == "auc"
    with pytest.raises(UnsupportedMetricError, match="competition.metric"):
        resolve_metric("Roc Auc Score", override="rmse")


def test_binary_scores_and_directions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert score(METRICS["auc"], y, p) == 1.0
    assert score(METRICS["accuracy"], y, p) == 1.0
    assert score(METRICS["f1"], y, p) == 1.0
    assert score(METRICS["macro_f1"], y, p) == 1.0
    assert score(METRICS["logloss"], y, p) < 0.3
    assert METRICS["auc"].better(0.9, 0.8) and not METRICS["logloss"].better(0.9, 0.8)
    assert METRICS["logloss"].improvement(0.4, 0.5) == pytest.approx(0.1)
    assert METRICS["auc"].improvement(0.4, 0.5) == pytest.approx(-0.1)


def test_multiclass_shapes():
    y = np.array([0, 1, 2, 1])
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.2, 0.7, 0.1]])
    assert score(METRICS["accuracy"], y, p) == 1.0
    assert score(METRICS["macro_f1"], y, p) == 1.0
    assert score(METRICS["auc"], y, p) == 1.0
    assert score(METRICS["logloss"], y, p) < 0.5
    with pytest.raises(UnsupportedMetricError, match="binary"):
        score(METRICS["f1"], y, p)
