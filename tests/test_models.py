import numpy as np
import pandas as pd
import pytest

from aac.models.cv import cross_validate, load_or_create_folds, make_folds
from aac.models.metrics import METRICS
from aac.models.prepare import MISSING_CATEGORY, prepare_matrix
from aac.models.registry import FAMILIES, supports_early_stopping
from aac.models.target import infer_target_encoding
from tests.synth import make_frames

FAST = {
    "lightgbm": {"n_estimators": 80, "num_leaves": 15},
    "xgboost": {"n_estimators": 80, "max_depth": 4},
    "catboost": {"iterations": 100, "depth": 4},
    "hist_gbdt": {"max_iter": 40, "max_leaf_nodes": 15},
    "logistic": {},
}
FEATURES = ["x1", "x2", "x3", "cat", "flag"]
CATS = ["cat", "flag"]


@pytest.fixture(scope="module")
def binary():
    train, test, _ = make_frames(600, 200, kind="binary")
    enc = infer_target_encoding(train["target"])
    y = enc.encode(train["target"])
    folds = make_folds(y, 5, 42)
    matrix = prepare_matrix(train, test, FEATURES, CATS)
    return matrix, y, folds


@pytest.fixture(scope="module")
def multiclass():
    train, test, _ = make_frames(600, 200, kind="multiclass")
    enc = infer_target_encoding(train["target"])
    y = enc.encode(train["target"])
    folds = make_folds(y, 4, 42)
    matrix = prepare_matrix(train, test, FEATURES, CATS)
    return matrix, y, folds


def test_make_folds_stratified_and_deterministic():
    y = np.array([0] * 90 + [1] * 10)
    a = make_folds(y, 5, 42)
    b = make_folds(y, 5, 42)
    assert a.dtype == np.int8 and (a == b).all()
    assert sorted(set(a)) == [0, 1, 2, 3, 4]
    for k in range(5):
        assert y[a == k].sum() == 2, "each fold holds 2 positives"
    assert not (make_folds(y, 5, 7) == a).all()


def test_load_or_create_folds_roundtrip(tmp_path):
    y = np.array([0, 1] * 50)
    path = tmp_path / "folds.npy"
    a = load_or_create_folds(path, y, 5, 42)
    assert path.exists() and not list(tmp_path.glob("*.tmp"))
    b = load_or_create_folds(path, y, 5, 99)  # seed ignored: the file wins
    assert (a == b).all()
    with pytest.raises(ValueError, match="does not match"):
        load_or_create_folds(path, y[:-1], 5, 42)
    with pytest.raises(ValueError, match="does not match"):
        load_or_create_folds(path, y, 4, 42)


def test_prepare_matrix_shared_vocabulary():
    train = pd.DataFrame({"n": [1, 2, None], "c": ["a", None, "b"]})
    test = pd.DataFrame({"n": ["4", "x", 6], "c": ["z", "a", None]})
    m = prepare_matrix(train, test, ["n", "c"], ["c"])
    assert m.X_train["n"].dtype == np.float64 and np.isnan(m.X_test["n"].iloc[1])
    assert list(m.X_train["c"].cat.categories) == [MISSING_CATEGORY, "a", "b", "z"]
    assert list(m.X_train["c"].cat.categories) == list(m.X_test["c"].cat.categories)
    assert m.X_test["c"].iloc[2] == MISSING_CATEGORY
    assert m.numeric == ["n"] and m.categorical == ["c"]
    with pytest.raises(KeyError, match="absent"):
        prepare_matrix(train, test, ["n", "missing"], [])


def run_cv(family, matrix, y, folds, **kw):
    kw.setdefault("metric", METRICS["auc"] if len(set(y)) == 2 else METRICS["logloss"])
    return cross_validate(
        family,
        FAST[family],
        matrix.X_train,
        y,
        matrix.X_test,
        folds,
        n_classes=len(set(y)),
        categorical=matrix.categorical,
        seed=42,
        n_jobs=2,
        **kw,
    )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_binary_cv(family, binary):
    matrix, y, folds = binary
    r = run_cv(family, matrix, y, folds)
    assert r.oof.shape == (600,) and r.test_pred.shape == (200,)
    assert r.oof.min() >= 0 and r.oof.max() <= 1
    assert len(r.fold_scores) == 5 and r.oof_score > 0.8, f"{family} has no signal: {r.oof_score}"
    assert r.duration > 0
    if supports_early_stopping(family):
        assert all(b is not None and b > 0 for b in r.best_iterations)
    if family in ("lightgbm", "xgboost", "catboost"):
        assert r.importances and set(r.importances) == set(FEATURES)


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost", "hist_gbdt", "logistic"])
def test_family_multiclass_shapes(family, multiclass):
    matrix, y, folds = multiclass
    r = run_cv(family, matrix, y, folds)
    assert r.oof.shape == (600, 3) and r.test_pred.shape == (200, 3)
    np.testing.assert_allclose(r.oof.sum(axis=1), 1.0, atol=1e-6)
    assert r.oof_score < 1.0  # logloss well below chance (ln 3 = 1.0986)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_determinism_to_1e9(family, binary):
    """README section 10: the same plan run twice must produce identical CV to 1e-9."""
    matrix, y, folds = binary
    a = run_cv(family, matrix, y, folds)
    b = run_cv(family, matrix, y, folds)
    np.testing.assert_allclose(a.oof, b.oof, rtol=0, atol=1e-9)
    np.testing.assert_allclose(a.test_pred, b.test_pred, rtol=0, atol=1e-9)
    assert a.fold_scores == b.fold_scores


def test_per_fold_target_encoding(binary):
    matrix, y, folds = binary
    r = run_cv("logistic", matrix, y, folds, target_encode=["cat"])
    assert r.oof.shape == (600,) and r.oof_score > 0.8


def test_early_stopping_can_be_disabled(binary):
    matrix, y, folds = binary
    r = run_cv("lightgbm", matrix, y, folds, early_stopping=False)
    assert r.best_iterations == [None] * 5


def test_prepare_matrix_maps_infinities_to_nan():
    train = pd.DataFrame({"r": [1.0, np.inf, -np.inf, 2.0]})
    test = pd.DataFrame({"r": [np.inf, 3.0]})
    m = prepare_matrix(train, test, ["r"], [])
    assert m.X_train["r"].isna().tolist() == [False, True, True, False]
    assert m.X_test["r"].isna().tolist() == [True, False]
