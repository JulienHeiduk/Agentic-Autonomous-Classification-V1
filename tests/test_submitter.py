import numpy as np
import pandas as pd
import pytest

from aac.agents.scout import profile_data
from aac.agents.submitter import predictions_to_submission
from aac.config import CompetitionConfig
from aac.kaggle.submission import SubmissionFormatError
from aac.models.metrics import METRICS
from tests.synth import make_frames


def profile_for(train, test, sample, metric):
    return profile_data(
        train,
        test,
        sample,
        slug="s",
        competition=CompetitionConfig(slug="s"),
        metric=METRICS[metric],
        max_classes=50,
        seed=0,
    )


def test_proba_submission():
    train, test, sample = make_frames(100, 40)
    p = profile_for(train, test, sample, "auc")
    pred = np.linspace(0, 1, 40)
    sub = predictions_to_submission(pred, p, sample, p.encoding())
    assert list(sub.columns) == ["id", "target"] and np.allclose(sub["target"], pred)
    two_col = np.column_stack([1 - pred, pred])
    assert np.allclose(predictions_to_submission(two_col, p, sample, p.encoding())["target"], pred)
    with pytest.raises(SubmissionFormatError):
        predictions_to_submission(pred * 2, p, sample, p.encoding())


def test_label_submission_keeps_sample_dtype():
    train, test, sample = make_frames(100, 40, labels=(False, True), sample_kind="label")
    p = profile_for(train, test, sample, "accuracy")
    pred = np.array([0.2, 0.8] * 20)
    sub = predictions_to_submission(pred, p, sample, p.encoding())
    assert sub["target"].dtype == bool and sub["target"].tolist() == [False, True] * 20
    train, test, sample = make_frames(100, 40, labels=("No", "Yes"), sample_kind="label")
    p = profile_for(train, test, sample, "accuracy")
    sub = predictions_to_submission(pred, p, sample, p.encoding())
    assert sub["target"].tolist() == ["No", "Yes"] * 20


def test_per_class_submission_orders_by_sample_columns():
    train, test, sample = make_frames(150, 30, kind="multiclass")
    p = profile_for(train, test, sample, "logloss")
    enc = p.encoding()
    assert enc.classes == ("high", "low", "mid")
    pred = np.tile([0.2, 0.3, 0.5], (30, 1))
    sub = predictions_to_submission(pred, p, sample, enc)
    assert list(sub.columns) == ["id", "high", "low", "mid"]
    assert sub.iloc[0][["high", "low", "mid"]].tolist() == [0.2, 0.3, 0.5]
    with pytest.raises(ValueError, match="per-class"):
        predictions_to_submission(pred[:, :2], p, sample, enc)


def test_label_submission_multiclass_argmax():
    train, test, sample = make_frames(150, 30, kind="multiclass", sample_kind="label")
    p = profile_for(train, test, sample, "accuracy")
    pred = np.tile([0.2, 0.3, 0.5], (30, 1))
    sub = predictions_to_submission(pred, p, sample, p.encoding())
    assert set(sub["target"]) == {"mid"}
    assert pd.api.types.is_string_dtype(sub["target"]) or sub["target"].dtype == object
