import numpy as np
import pandas as pd
import pytest

from aac.agents.scout import Profile, ScoutError, column_kind, infer_target, profile_data
from aac.config import CompetitionConfig
from aac.models.metrics import METRICS
from tests.synth import make_frames


def scout(train, test, sample, metric="auc", **cfg):
    return profile_data(
        train,
        test,
        sample,
        slug="synthetic",
        competition=CompetitionConfig(slug="s", **cfg),
        metric=METRICS[metric],
        max_classes=50,
        seed=42,
    )


def test_profile_binary_proba():
    train, test, sample = make_frames(300, 100)
    p = scout(train, test, sample)
    assert p.id_col == "id" and p.target.name == "target" and p.target.kind == "binary"
    assert p.target.positive_label == 1 and p.target.classes == [0, 1]
    assert p.submission.kind == "proba" and p.submission.columns == ["target"]
    kinds = {c.name: c.kind for c in p.columns}
    assert kinds == {
        "id": "numeric",
        "x1": "numeric",
        "x2": "numeric",
        "x3": "numeric",
        "cat": "categorical",
        "flag": "boolean",
        "note": "text",
        "const": "constant",
    }
    assert p.feature_columns() == ["x1", "x2", "x3", "cat", "flag"]
    assert p.categorical_columns() == ["cat", "flag"]
    assert set(p.unusable_columns()) == {"id", "note", "const"}
    x2 = p.column("x2")
    assert x2.missing_train > 0 and x2.drift is not None and x2.stats and "mean" in x2.stats
    assert p.column("cat").top_values and p.column("cat").drift is not None
    assert any("note" in w for w in p.warnings) and any("const" in w for w in p.warnings)
    assert p.n_train == 300 and p.n_test == 100 and p.memory_mb > 0
    assert abs(sum(p.target.rates.values()) - 1) < 1e-9


def test_profile_roundtrips_through_json():
    train, test, sample = make_frames(200, 80, labels=("No", "Yes"))
    p = scout(train, test, sample)
    again = Profile.model_validate(p.model_dump(mode="json"))
    assert again == p
    assert again.encoding().positive_label == "Yes"


def test_label_submission_and_bool_target():
    train, test, sample = make_frames(200, 80, labels=(False, True), sample_kind="label")
    p = scout(train, test, sample, metric="accuracy")
    assert p.submission.kind == "label" and p.target.positive_label is True
    assert p.target.counts == {"False": 100, "True": 100}


def test_multiclass_per_class_columns():
    train, test, sample = make_frames(300, 100, kind="multiclass")
    p = scout(train, test, sample, metric="logloss")
    assert p.target.kind == "multiclass" and p.target.classes == ["high", "low", "mid"]
    assert p.submission.kind == "proba_per_class" and p.submission.columns == ["high", "low", "mid"]
    bad = sample.rename(columns={"mid": "medium"})
    with pytest.raises(ScoutError, match="prediction columns"):
        scout(train, test, bad, metric="logloss")
    single = pd.DataFrame({"id": test["id"], "target": 0.3})
    with pytest.raises(ScoutError, match="single prediction column"):
        scout(train, test, single, metric="logloss")


def test_target_inference_fallbacks_and_overrides():
    train, test, sample = make_frames(200, 80)
    renamed = sample.rename(columns={"target": "prediction"})
    assert infer_target(train, test, renamed, None) == "target", "only-in-train fallback"
    p = scout(train, test, renamed, target="target", id_col="id")
    assert p.target.name == "target"
    with pytest.raises(ScoutError, match="competition.target"):
        scout(train, test, renamed, target="nope")
    with pytest.raises(ScoutError, match="competition.id_col"):
        scout(train, test, sample, id_col="nope")
    train2 = train.copy()
    train2["extra"] = 1.0
    with pytest.raises(ScoutError, match="cannot infer the target"):
        scout(train2, test, renamed)


def test_refuses_regression_and_row_mismatch():
    train, test, sample = make_frames(200, 80)
    train["target"] = np.random.default_rng(0).normal(size=200)
    with pytest.raises(ScoutError, match="regression|non-integer"):
        scout(train, test, sample)
    train, test, sample = make_frames(200, 80)
    with pytest.raises(ScoutError, match="rows"):
        scout(train, test, sample.iloc[:-1])


def test_column_kind_edge_cases():
    assert column_kind(pd.Series(["True", "false", None])) == "boolean"
    assert column_kind(pd.Series([True, False, True])) == "boolean"
    assert column_kind(pd.Series([1, 1, 1])) == "constant"
    assert column_kind(pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01"]))) == "datetime"
    assert column_kind(pd.Series([f"u{i}" for i in range(500)])) == "text"
    assert column_kind(pd.Series(["a", "b"] * 250)) == "categorical"
    assert column_kind(pd.Series([1.5, 2.5, 3.5])) == "numeric"
