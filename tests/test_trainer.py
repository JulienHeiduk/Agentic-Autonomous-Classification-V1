import json

import numpy as np
import pytest

from aac.agents.scout import profile_data
from aac.agents.trainer import select_features, train_plan
from aac.config import CompetitionConfig
from aac.models.cv import make_folds
from aac.models.metrics import METRICS
from aac.plan import ModelConfig, Plan
from tests.synth import make_frames


@pytest.fixture
def setup():
    train, test, sample = make_frames(400, 120)
    profile = profile_data(
        train,
        test,
        sample,
        slug="s",
        competition=CompetitionConfig(slug="s"),
        metric=METRICS["auc"],
        max_classes=50,
        seed=1,
    )
    enc = profile.encoding()
    y = enc.encode(train["target"])
    folds = make_folds(y, 4, 1)
    return train, test, sample, profile, y, folds


def test_select_features_respects_plan_and_profile(setup):
    train, test, _, profile, _, _ = setup
    plan = Plan(name="p", drop_columns=["x3"], models=[ModelConfig(family="logistic")])
    feats, cats = select_features(plan, profile, train, test)
    assert feats == ["x1", "x2", "cat", "flag"] and cats == ["cat", "flag"]
    train2, test2 = train.copy(), test.copy()
    train2["eng_num"] = train2["x1"] ** 2
    test2["eng_num"] = test2["x1"] ** 2
    train2["eng_cat"] = train2["cat"].fillna("z") + "!"
    test2["eng_cat"] = test2["cat"].fillna("z") + "!"
    feats, cats = select_features(plan, profile, train2, test2)
    assert feats[-2:] == ["eng_num", "eng_cat"] and cats == ["cat", "flag", "eng_cat"]
    explicit = Plan(name="p", categorical_columns=["x3"], models=[ModelConfig(family="logistic")])
    feats, cats = select_features(explicit, profile, train, test)
    assert "x3" in feats and cats == ["x3"]
    with pytest.raises(ValueError, match="no usable"):
        select_features(
            Plan(name="p", drop_columns=feats, models=[ModelConfig(family="logistic")]),
            profile,
            train,
            test,
        )


def test_train_plan_writes_artifacts_and_picks_best(setup, tmp_path):
    train, test, _, profile, y, folds = setup
    plan = Plan(
        name="t",
        models=[
            ModelConfig(family="logistic"),
            ModelConfig(family="lightgbm", params={"n_estimators": 60, "num_leaves": 15}),
        ],
    )
    result = train_plan(
        plan,
        profile,
        train,
        test,
        y,
        folds,
        metric=METRICS["auc"],
        seed=1,
        n_jobs=2,
        branch_dir=tmp_path,
    )
    assert set(result.results) == {"logistic", "lightgbm"} and not result.errors
    assert result.best_family in result.results
    assert result.best.oof_score == max(r.oof_score for r in result.results.values())
    assert result.blend_oof_score > 0.8
    for name in (
        "oof_logistic.npy",
        "test_lightgbm.npy",
        "oof.npy",
        "test_pred.npy",
        "metrics.json",
    ):
        assert (tmp_path / name).exists()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["best_family"] == result.best_family and metrics["plan_hash"] == plan.hash()
    assert metrics["models"]["lightgbm"]["importances"]
    assert np.load(tmp_path / "oof.npy").shape == (400,)


def test_one_failing_family_does_not_sink_the_branch(setup, tmp_path):
    train, test, _, profile, y, folds = setup
    plan = Plan(
        name="t",
        models=[
            ModelConfig(family="lightgbm", params={"n_estimators": -5}),
            ModelConfig(family="logistic"),
        ],
    )
    result = train_plan(
        plan,
        profile,
        train,
        test,
        y,
        folds,
        metric=METRICS["auc"],
        seed=1,
        n_jobs=2,
        branch_dir=tmp_path,
    )
    assert "lightgbm" in result.errors and result.best_family == "logistic"
    only_bad = Plan(name="t", models=[ModelConfig(family="lightgbm", params={"n_estimators": -5})])
    with pytest.raises(RuntimeError, match="every model family failed"):
        train_plan(
            only_bad,
            profile,
            train,
            test,
            y,
            folds,
            metric=METRICS["auc"],
            seed=1,
            n_jobs=2,
            branch_dir=tmp_path,
        )


def test_variant_plans_from_the_profile(setup, tmp_path):
    from aac.plan import low_cardinality_numeric, variant_plan

    train, test, _, profile, y, folds = setup
    assert low_cardinality_numeric(profile, 50) == ["x3"], "x3 holds five integer levels"
    cat = variant_plan("categorical", profile, ["lightgbm"], 50)
    assert cat.name == "categorical" and cat.categorical_columns == ["cat", "flag", "x3"]
    assert cat.target_encode == [] and [m.family for m in cat.models] == ["lightgbm"]
    enc = variant_plan("encoded", profile, ["lightgbm", "xgboost"], 50)
    assert enc.name == "encoded" and enc.target_encode == ["cat", "flag", "x3"]
    assert enc.categorical_columns is None
    assert variant_plan("categorical", profile, ["lightgbm"], 3) is None, "x3 has 5 values"
    assert variant_plan("encoded", profile, [], 50) is None
    with pytest.raises(ValueError, match="unknown plan variant"):
        variant_plan("ghost", profile, ["lightgbm"], 50)

    features, categorical = select_features(cat, profile, train, test)
    assert "x3" in categorical and "cat" in categorical
    result = train_plan(
        enc,
        profile,
        train,
        test,
        y,
        folds,
        metric=METRICS["auc"],
        seed=1,
        n_jobs=2,
        branch_dir=tmp_path / "enc",
        families=["lightgbm"],
    )
    assert set(result.results) == {"lightgbm"}, "the family filter drops xgboost"
    assert result.results["lightgbm"].oof_score > 0.8
    imp = result.results["lightgbm"].importances
    assert "x3" in imp and "x3__te" in imp, "an integer keeps its column beside its encoding"
    assert "cat" not in imp and "cat__te" in imp, "a category is replaced by its encoding"
    with pytest.raises(ValueError, match="none of the families"):
        train_plan(
            enc,
            profile,
            train,
            test,
            y,
            folds,
            metric=METRICS["auc"],
            seed=1,
            n_jobs=2,
            branch_dir=tmp_path / "none",
            families=["catboost"],
        )
