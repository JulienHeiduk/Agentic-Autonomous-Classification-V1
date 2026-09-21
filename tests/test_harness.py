"""The experiment harness: fit_predict per fold, owned by the runner (README 17.1)."""

import textwrap

import numpy as np
import pandas as pd
import pytest

from aac.exec.sandbox import EXPERIMENT_IMPORTS, check_source, run_experiment
from aac.models.cv import make_folds
from aac.models.metrics import METRICS
from aac.models.target import infer_target_encoding
from tests.synth import make_frames

LOGISTIC = textwrap.dedent(
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    def _numeric(X):
        return X.select_dtypes(include=["number"]).copy()

    def fit_predict(X_train, y_train, X_valid, X_test, meta):
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500)
        )
        model.fit(_numeric(X_train), y_train)
        p_valid = model.predict_proba(_numeric(X_valid))[:, 1]
        p_test = model.predict_proba(_numeric(X_test))[:, 1]
        return p_valid, p_test
    """
)

LIGHTGBM_WITH_FEATURES = textwrap.dedent(
    """
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    def build_features(train_df, test_df):
        for df in (train_df, test_df):
            df["x1_x2"] = df["x1"].fillna(0) * df["x2"].fillna(0)
        return train_df, test_df, ["x1_x2"]

    def fit_predict(X_train, y_train, X_valid, X_test, meta):
        model = lgb.LGBMClassifier(
            n_estimators=60, num_leaves=15, random_state=meta["seed"], n_jobs=meta["n_threads"],
            deterministic=True, force_row_wise=True, verbose=-1,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_valid)[:, 1], model.predict_proba(X_test)[:, 1]
    """
)

TORCH_MLP = textwrap.dedent(
    """
    import numpy as np
    import torch

    def fit_predict(X_train, y_train, X_valid, X_test, meta):
        torch.manual_seed(meta["seed"])
        torch.set_num_threads(meta["n_threads"])
        num = [c for c in X_train.columns if c not in meta["categorical"]]
        def prep(X, mu=None, sd=None):
            A = X[num].to_numpy(dtype="float64")
            A = np.nan_to_num(A, nan=0.0)
            mu = A.mean(axis=0) if mu is None else mu
            sd = A.std(axis=0) + 1e-6 if sd is None else sd
            return torch.tensor((A - mu) / sd, dtype=torch.float32), mu, sd
        Xt, mu, sd = prep(X_train)
        yt = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        net = torch.nn.Sequential(
            torch.nn.Linear(Xt.shape[1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
        )
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(60):
            opt.zero_grad()
            loss = loss_fn(net(Xt).squeeze(1), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pv = torch.sigmoid(net(prep(X_valid, mu, sd)[0]).squeeze(1)).numpy()
            pt = torch.sigmoid(net(prep(X_test, mu, sd)[0]).squeeze(1)).numpy()
        return pv, pt
    """
)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("harness")
    train, test, _ = make_frames(400, 120)
    enc = infer_target_encoding(train["target"])
    y = enc.encode(train["target"])
    folds = make_folds(y, 4, 7)
    train_path, test_path = tmp / "train.parquet", tmp / "test.parquet"
    train.drop(columns=["target"]).to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    return tmp, train_path, test_path, y, folds


def run(code, data, tmp_path, **kw):
    _, train_path, test_path, y, folds = data
    kw.setdefault("timeout", 180.0)
    return run_experiment(
        code,
        workdir=tmp_path / "exp",
        train_path=train_path,
        test_path=test_path,
        y=y,
        folds=folds,
        metric=METRICS["auc"],
        target="target",
        id_col="id",
        n_classes=2,
        categorical=["cat", "flag"],
        drop_columns=["note", "const"],
        seed=3,
        n_threads=2,
        **kw,
    )


def test_static_check_experiment_mode():
    assert check_source(LOGISTIC, target="target", mode="experiment") == []
    assert "lightgbm" in EXPERIMENT_IMPORTS and "torch" in EXPERIMENT_IMPORTS
    assert any(
        "no top-level def fit_predict" in p
        for p in check_source(
            "def build_features(a, b):\n    return a, b, []\n", target="t", mode="experiment"
        )
    )
    assert any(
        "five positional" in p
        for p in check_source(
            "def fit_predict(a, b):\n    return a, b\n", target="t", mode="experiment"
        )
    )
    assert any(
        "import of 'lightgbm'" in p
        for p in check_source(
            "import lightgbm\ndef build_features(a, b):\n    return a, b, []\n", target="t"
        )
    )
    bad_feat = (
        "def build_features(a):\n    return a\ndef fit_predict(a, b, c, d, e):\n    return None\n"
    )
    assert any("two positional" in p for p in check_source(bad_feat, target="t", mode="experiment"))


def test_logistic_experiment_scores_and_writes_artifacts(data, tmp_path):
    result = run(LOGISTIC, data, tmp_path)
    assert result.ok, result.error
    assert result.oof.shape == (400,) and result.test_pred.shape == (120,)
    assert len(result.fold_scores) == 4 and result.oof_score > 0.8
    assert result.features == ["x1", "x2", "x3", "cat", "flag"] and result.categorical == [
        "cat",
        "flag",
    ]
    assert len(result.fold_seconds) == 4
    for name in (
        "experiment.py",
        "oof.npy",
        "test_pred.npy",
        "metrics.json",
        "job.json",
        "stdout.log",
    ):
        assert (tmp_path / "exp" / name).exists(), name


def test_lightgbm_with_build_features(data, tmp_path):
    result = run(LIGHTGBM_WITH_FEATURES, data, tmp_path)
    assert result.ok, result.error
    assert result.new_columns == ["x1_x2"] and "x1_x2" in result.features
    assert result.oof_score > 0.85 and result.feature_seconds >= 0


def test_torch_mlp_runs_deterministically(data, tmp_path):
    result = run(TORCH_MLP, data, tmp_path)
    assert result.ok, result.error
    assert result.oof_score > 0.75
    again = run(TORCH_MLP, data, tmp_path / "again")
    np.testing.assert_allclose(result.oof, again.oof, rtol=0, atol=1e-9)


def test_fit_predict_contract_violations(data, tmp_path):
    wrong_shape = (
        "def fit_predict(a, b, c, d, e):\n    import numpy as np\n"
        "    return np.zeros(3), np.zeros(len(d))\n"
    )
    r = run(wrong_shape, data, tmp_path)
    assert not r.ok and r.kind == "error" and "shape" in r.error and "fold 0" in r.error
    out_of_range = (
        "def fit_predict(a, b, c, d, e):\n    import numpy as np\n"
        "    return np.full(len(c), 2.0), np.full(len(d), 0.5)\n"
    )
    r = run(out_of_range, data, tmp_path)
    assert not r.ok and "[0, 1]" in r.error
    nan = (
        "def fit_predict(a, b, c, d, e):\n    import numpy as np\n"
        "    return np.full(len(c), np.nan), np.full(len(d), 0.5)\n"
    )
    r = run(nan, data, tmp_path)
    assert not r.ok and "NaN" in r.error


def test_non_deterministic_fit_predict_is_refused(data, tmp_path):
    code = (
        "import numpy as np\ndef fit_predict(a, b, c, d, e):\n"
        "    return np.random.rand(len(c)), np.random.rand(len(d))\n"
    )
    r = run(code, data, tmp_path)
    assert not r.ok and r.kind == "violation" and "not deterministic" in r.error


def test_target_leak_via_name_is_rejected_statically(data, tmp_path):
    code = "def fit_predict(a, b, c, d, e):\n    return c['target'].values, d['x1'].values\n"
    r = run(code, data, tmp_path)
    assert not r.ok and r.kind == "rejected" and "target column" in r.error


EXTRA_AWARE = textwrap.dedent(
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    def fit_predict(X_train, y_train, X_valid, X_test, meta):
        n_extra = meta["n_extra"]
        assert n_extra == 30, meta
        assert len(X_train) == len(y_train)
        assert X_train["is_original"].sum() == n_extra, "extras carry the flag"
        assert X_train["is_original"].iloc[-n_extra:].eq(1.0).all(), "extras come last"
        assert X_valid["is_original"].sum() == 0 and X_test["is_original"].sum() == 0
        assert "is_original" in meta["features"]
        cols = ["x1", "x2", "x3"]
        m = LogisticRegression(max_iter=200).fit(X_train[cols].fillna(0), y_train)
        return m.predict_proba(X_valid[cols].fillna(0))[:, 1], m.predict_proba(
            X_test[cols].fillna(0)
        )[:, 1]
    """
)


def test_extra_rows_reach_fit_predict_in_every_fold(data, tmp_path):
    tmp, train_path, _, y, _ = data
    train = pd.read_parquet(train_path)
    extra = train.tail(30).copy()
    extra["x1"] = extra["x1"] + 50.0
    extra["id"] = -np.arange(1, 31)
    extra_path = tmp_path / "extra.parquet"
    extra.to_parquet(extra_path, index=False)
    y_extra = y[-30:]
    result = run(
        EXTRA_AWARE,
        data,
        tmp_path,
        extra_train=extra_path,
        extra_y=y_extra,
        extra_flag="is_original",
    )
    assert result.ok, result.error
    assert result.features[-1] == "is_original" and result.oof.shape == (400,)
    # without extras the module's own assertions fail, so the harness reports the error
    plain = run(EXTRA_AWARE, data, tmp_path / "plain")
    assert not plain.ok and "AssertionError" in plain.error
