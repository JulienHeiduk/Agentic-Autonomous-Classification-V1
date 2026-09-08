"""Synthetic tabular data with signal, for model and pipeline tests."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd


def make_frames(
    n_train: int = 800,
    n_test: int = 300,
    seed: int = 0,
    kind: str = "binary",
    labels: tuple[object, object] = (0, 1),
    target: str = "target",
    sample_kind: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Numeric, categorical, boolean-with-missing, free-text and constant columns.

    ``kind``: "binary" (labels from ``labels``), "multiclass" (low/mid/high).
    ``sample_kind``: "proba", "label", or "proba_per_class"; defaults by kind.
    """
    rng = np.random.default_rng(seed)

    def block(n: int, offset: int) -> pd.DataFrame:
        flag = rng.choice([True, False, None], n, p=[0.45, 0.45, 0.10])
        df = pd.DataFrame(
            {
                "id": np.arange(offset, offset + n),
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "x3": rng.integers(0, 5, n).astype(float),
                "cat": rng.choice(["a", "b", "c"], n),
                "flag": pd.Series(flag, dtype="object"),
                "note": [f"note-{i}" for i in range(offset, offset + n)],
                "const": 7,
            }
        )
        df.loc[rng.random(n) < 0.05, "x2"] = np.nan
        df.loc[rng.random(n) < 0.03, "cat"] = None
        return df

    train = block(n_train, 0)
    test = block(n_test, n_train)

    def signal(df: pd.DataFrame) -> np.ndarray:
        return (
            df["x1"].to_numpy()
            + 0.5 * df["x2"].fillna(0).to_numpy()
            + 0.8 * (df["cat"] == "a").to_numpy()
            + 0.3 * (df["flag"] == True).to_numpy()  # noqa: E712 - object column
            + rng.normal(scale=0.4, size=len(df))
        )

    s = signal(train)
    if kind == "binary":
        train[target] = np.where(s > np.median(s), labels[1], labels[0])
        classes = [labels[0], labels[1]]
    elif kind == "multiclass":
        train[target] = pd.qcut(s, 3, labels=["low", "mid", "high"]).astype(str)
        classes = ["high", "low", "mid"]
    else:
        raise ValueError(kind)

    sample_kind = sample_kind or ("proba" if kind == "binary" else "proba_per_class")
    if sample_kind == "proba":
        sample = pd.DataFrame({"id": test["id"], target: 0.5})
    elif sample_kind == "label":
        sample = pd.DataFrame({"id": test["id"], target: classes[0]})
    elif sample_kind == "proba_per_class":
        sample = pd.DataFrame({"id": test["id"], **{str(c): 1.0 / len(classes) for c in classes}})
    else:
        raise ValueError(sample_kind)
    return train, test, sample


def bundle_from_frames(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("train.csv", train.to_csv(index=False))
        zf.writestr("test.csv", test.to_csv(index=False))
        zf.writestr("sample_submission.csv", sample.to_csv(index=False))
    return buf.getvalue()
