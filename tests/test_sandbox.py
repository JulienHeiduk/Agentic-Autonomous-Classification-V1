import textwrap

import pandas as pd
import pytest

from aac.exec.sandbox import check_source, run_feature_module
from tests.synth import make_frames

GOOD = textwrap.dedent(
    """
    import numpy as np
    import pandas as pd

    def build_features(train_df, test_df):
        both = pd.concat([train_df, test_df])
        freq = both["cat"].astype(str).value_counts(normalize=True)
        for df in (train_df, test_df):
            df["x1_x2"] = df["x1"].fillna(0) * df["x2"].fillna(0)
            df["cat_freq"] = df["cat"].astype(str).map(freq).astype(float)
        return train_df, test_df, ["x1_x2", "cat_freq"]
    """
)


@pytest.mark.parametrize(
    ("snippet", "problem"),
    [
        ("import os\ndef build_features(a, b):\n    return a, b, []\n", "import of 'os'"),
        (
            "from subprocess import run\ndef build_features(a, b):\n    return a, b, []\n",
            "import from 'subprocess'",
        ),
        (
            "import sklearn.preprocessing\ndef build_features(a, b):\n"
            "    open('x', 'w')\n    return a, b, []\n",
            "open()",
        ),
        ("def build_features(a, b):\n    eval('1')\n    return a, b, []\n", "eval()"),
        (
            "def build_features(a, b):\n    x = ().__class__.__bases__\n    return a, b, []\n",
            "__class__",
        ),
        ("def build_features(a, b):\n    a['target'] = 1\n    return a, b, []\n", "target column"),
        ("def build_features(a, b):\n    y = a.Target\n    return a, b, []\n", "target column"),
        ("def make(a, b):\n    return a, b, []\n", "no top-level def build_features"),
        ("def build_features(a):\n    return a, a, []\n", "exactly two positional"),
        ("def build_features(a, b:\n", "syntax error"),
        ("from . import helper\ndef build_features(a, b):\n    return a, b, []\n", "not allowed"),
    ],
)
def test_static_check_rejects(snippet, problem):
    problems = check_source(snippet, target="target")
    assert any(problem in p for p in problems), problems


def test_static_check_accepts_good_module():
    assert check_source(GOOD, target="target") == []
    assert check_source(GOOD, target="Will_Buy_EV") == []


@pytest.fixture
def inputs(tmp_path):
    train, test, _ = make_frames(120, 40)
    train_path, test_path = tmp_path / "train.parquet", tmp_path / "test.parquet"
    train.drop(columns=["target"]).to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    return train_path, test_path


def run(code, inputs, tmp_path, **kw):
    train_path, test_path = inputs
    kw.setdefault("timeout", 60.0)
    return run_feature_module(
        code,
        workdir=tmp_path / "sandbox",
        train_path=train_path,
        test_path=test_path,
        target="target",
        id_col="id",
        **kw,
    )


def test_run_good_module(inputs, tmp_path):
    result = run(GOOD, inputs, tmp_path)
    assert result.ok, result.error
    assert result.kind == "ok" and result.new_columns == ["x1_x2", "cat_freq"]
    train_out = pd.read_parquet(result.train_out)
    assert "cat_freq" in train_out.columns and len(train_out) == 120
    assert (tmp_path / "sandbox" / "features.py").read_text() == GOOD
    assert (tmp_path / "sandbox" / "stdout.log").exists()


def test_static_rejection_does_not_execute(inputs, tmp_path):
    result = run("import os\ndef build_features(a, b):\n    return a, b, []\n", inputs, tmp_path)
    assert not result.ok and result.kind == "rejected" and "import of 'os'" in result.error
    assert not (tmp_path / "sandbox" / "job.json").exists()


def test_runtime_error_returns_traceback(inputs, tmp_path):
    code = (
        "def build_features(train_df, test_df):\n"
        "    train_df['z'] = train_df['missing_col'] * 2\n"
        "    return train_df, test_df, ['z']\n"
    )
    result = run(code, inputs, tmp_path)
    assert not result.ok and result.kind == "error"
    assert (
        "KeyError" in result.error
        and "missing_col" in result.error
        and "features.py" in result.error
    )


VIOLATIONS = [
    ("train_df = train_df.iloc[:-1]", [], "rows"),
    ("train_df['id'] = 0", [], "id column"),
    ("train_df['TARGET'] = 1.0; test_df['TARGET'] = 1.0", [], "target column"),
    ("pass", ["new"], "new_columns not present"),
    (
        "train_df['z'] = [[1]] * len(train_df); test_df['z'] = [[1]] * len(test_df)",
        ["z"],
        "non-scalar",
    ),
    ("train_df['z'] = 1.0", ["z"], "train and test columns differ"),
]


@pytest.mark.parametrize(("body", "cols", "message"), VIOLATIONS)
def test_contract_violations(inputs, tmp_path, body, cols, message):
    code = (
        f"def build_features(train_df, test_df):\n    {body}\n"
        f"    return train_df, test_df, {cols!r}\n"
    )
    result = run(code, inputs, tmp_path)
    assert not result.ok and message in result.error, result.error


def test_non_deterministic_module_is_refused(inputs, tmp_path):
    code = (
        "import numpy as np\ndef build_features(train_df, test_df):\n"
        "    train_df['r'] = np.random.rand(len(train_df))\n"
        "    test_df['r'] = np.random.rand(len(test_df))\n"
        "    return train_df, test_df, ['r']\n"
    )
    result = run(code, inputs, tmp_path)
    assert not result.ok and result.kind == "violation" and "not deterministic" in result.error


def test_network_and_file_writes_are_blocked(inputs, tmp_path):
    code = (
        "import pandas as pd\ndef build_features(train_df, test_df):\n"
        "    pd.read_csv('https://example.com/x.csv')\n    return train_df, test_df, []\n"
    )
    result = run(code, inputs, tmp_path)
    assert not result.ok and "network access is disabled" in result.error
    code = (
        "import pandas as pd\ndef build_features(train_df, test_df):\n"
        "    train_df.to_csv('/tmp/aac_escape.csv')\n    return train_df, test_df, []\n"
    )
    result = run(code, inputs, tmp_path)
    assert not result.ok and "outside the sandbox" in result.error


def test_timeout_kills_the_process(inputs, tmp_path):
    code = (
        "import itertools\ndef build_features(train_df, test_df):\n"
        "    for _ in itertools.count():\n        pass\n    return train_df, test_df, []\n"
    )
    result = run(code, inputs, tmp_path, timeout=3.0)
    assert not result.ok and result.kind == "timeout" and "3s" in result.error
