"""Sandbox runner (README sections 7 and 17.1). Executed as ``python -I _runner.py <workdir>``.

Two job kinds, chosen by ``job.json``:

- ``features``: run ``build_features`` on target-free frames, check the contract, check
  purity by calling it twice, write the engineered frames.
- ``experiment``: optionally run ``build_features``, prepare the matrix, then own the folds:
  call ``fit_predict(X_train, y_train, X_valid, X_test, meta)`` once per fold with that fold's
  training rows and target only, assemble OOF and averaged test predictions, check
  determinism on a subsample, write the arrays. Scoring happens in the parent.

Shims: no sockets, no writes outside the work directory, a memory cap where the OS honours
it. This file imports ``aac.models.prepare`` for the matrix (installed in the same venv) but
never anything that touches the network or the ledger.
"""

from __future__ import annotations

import builtins
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

WORKDIR = Path(sys.argv[1]).resolve()
os.chdir(WORKDIR)
JOB = json.loads((WORKDIR / "job.json").read_text(encoding="utf-8"))


def _limit_memory(megabytes: int) -> None:
    try:
        import resource

        limit = int(megabytes) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):
        pass  # macOS does not honour RLIMIT_AS; the wall clock still applies


def _blocked(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("network access is disabled in the sandbox")


class _BlockedSocket(socket.socket):
    """Still a class (ssl subclasses socket.socket at import time) but cannot be instantiated."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        _blocked()


socket.socket = _BlockedSocket  # type: ignore[misc]
socket.SocketType = _BlockedSocket  # type: ignore[misc]
socket.create_connection = _blocked  # type: ignore[assignment]
socket.getaddrinfo = _blocked  # type: ignore[assignment]

_real_open = builtins.open


def _guarded_open(file, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
    if any(ch in str(mode) for ch in "wax+"):
        try:
            target = Path(os.fspath(file)).resolve()
        except TypeError:
            target = None
        if target is None or (target != WORKDIR and WORKDIR not in target.parents):
            raise PermissionError(f"writing outside the sandbox directory is not allowed: {file!r}")
    return _real_open(file, mode, *args, **kwargs)


builtins.open = _guarded_open  # type: ignore[assignment]


def _fail(error: str, kind: str = "error") -> None:
    (WORKDIR / "result.json").write_text(json.dumps({"ok": False, "kind": kind, "error": error}))
    sys.exit(1)


def _write_result(payload: dict) -> None:
    (WORKDIR / "result.json").write_text(json.dumps(payload))


# -- build_features contract ------------------------------------------------------------


def _validate_features(result, train_in, test_in, id_col: str, target: str):  # type: ignore[no-untyped-def]
    import pandas as pd

    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("build_features must return a tuple (train_df, test_df, new_columns)")
    train_out, test_out, new_columns = result
    for name, frame, original in (("train", train_out, train_in), ("test", test_out, test_in)):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{name}_df returned is {type(frame).__name__}, not a DataFrame")
        if len(frame) != len(original):
            raise ValueError(f"{name}_df has {len(frame)} rows, input had {len(original)}")
        if id_col not in frame.columns:
            raise ValueError(f"{name}_df lost the id column {id_col!r}")
        if not frame[id_col].reset_index(drop=True).equals(original[id_col].reset_index(drop=True)):
            raise ValueError(f"{name}_df changed the id column values or their order")
        dup = frame.columns[frame.columns.duplicated()].tolist()
        if dup:
            raise ValueError(f"{name}_df has duplicate columns {dup}")
        for col in frame.columns:
            if str(col).lower() == target.lower():
                raise ValueError(f"{name}_df contains the target column {col!r}")
    if list(train_out.columns) != list(test_out.columns):
        only_train = [c for c in train_out.columns if c not in test_out.columns]
        only_test = [c for c in test_out.columns if c not in train_out.columns]
        raise ValueError(
            f"train and test columns differ: only in train {only_train}, only in test {only_test}"
        )
    if not isinstance(new_columns, list) or not all(isinstance(c, str) for c in new_columns):
        raise ValueError("new_columns must be a list of column name strings")
    missing = [c for c in new_columns if c not in train_out.columns]
    if missing:
        raise ValueError(f"new_columns not present in the returned frames: {missing}")
    for col in new_columns:
        if getattr(train_out[col].dtype, "kind", "O") == "O":
            sample = train_out[col].dropna().head(20)
            if not all(isinstance(v, str) for v in sample):
                raise ValueError(
                    f"new column {col!r} holds non-scalar objects; return numbers or strings"
                )
    return train_out, test_out, new_columns


def _run_features(fn, train_in, test_in, id_col: str, target: str):  # type: ignore[no-untyped-def]
    """Call build_features twice on fresh copies; refuse any difference."""
    import pandas as pd

    started = time.monotonic()
    try:
        first = fn(train_in.copy(), test_in.copy())
        train_out, test_out, new_columns = _validate_features(
            first, train_in, test_in, id_col, target
        )
        second = fn(train_in.copy(), test_in.copy())
        train_2, test_2, columns_2 = _validate_features(second, train_in, test_in, id_col, target)
    except Exception:
        _fail(traceback.format_exc(), "error")
    duration = time.monotonic() - started
    try:
        pd.testing.assert_frame_equal(train_out, train_2, check_exact=True)
        pd.testing.assert_frame_equal(test_out, test_2, check_exact=True)
        if new_columns != columns_2:
            raise AssertionError(f"new_columns differ between runs: {new_columns} vs {columns_2}")
    except AssertionError as exc:
        detail = str(exc)[:400]
        _fail(
            "build_features is not deterministic: two calls on the same input gave different "
            f"outputs. Fix the source of randomness (seed it or remove it). Detail: {detail}",
            "violation",
        )
    return train_out, test_out, new_columns, duration


# -- fit_predict contract -----------------------------------------------------------------


def _as_proba(value, n_rows: int, n_classes: int, what: str):  # type: ignore[no-untyped-def]
    import numpy as np

    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not numeric: {exc}") from None
    if arr.ndim == 2 and n_classes == 2 and arr.shape[1] == 2:
        arr = arr[:, 1]
    if n_classes == 2:
        if arr.shape != (n_rows,):
            raise ValueError(f"{what} must have shape ({n_rows},) for binary, got {arr.shape}")
    elif arr.shape != (n_rows, n_classes):
        raise ValueError(f"{what} must have shape ({n_rows}, {n_classes}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{what} contains NaN or inf")
    if arr.min() < -1e-6 or arr.max() > 1 + 1e-6:
        raise ValueError(
            f"{what} must be probabilities in [0, 1], got min {arr.min():.4g} max {arr.max():.4g}"
        )
    arr = np.clip(arr, 0.0, 1.0)
    if n_classes > 2:
        sums = arr.sum(axis=1)
        if np.abs(sums - 1).max() > 1e-3:
            deviation = float(np.abs(sums - 1).max())
            raise ValueError(
                f"{what} rows must sum to 1 for multiclass (max deviation {deviation:.4g})"
            )
        arr = arr / sums[:, None]
    return arr


def _call_fit_predict(fn, X_tr, y_tr, X_va, X_te, meta):  # type: ignore[no-untyped-def]
    result = fn(X_tr, y_tr, X_va, X_te, meta)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("fit_predict must return a tuple (p_valid, p_test)")
    p_va = _as_proba(result[0], len(X_va), meta["n_classes"], "p_valid")
    p_te = _as_proba(result[1], len(X_te), meta["n_classes"], "p_test")
    return p_va, p_te


def _run_experiment(namespace) -> None:  # type: ignore[no-untyped-def]
    import numpy as np
    import pandas as pd

    from aac.models.prepare import prepare_matrix

    fit_predict = namespace.get("fit_predict")
    if not callable(fit_predict):
        _fail(
            "the module does not define fit_predict(X_train, y_train, X_valid, X_test, meta)",
            "violation",
        )
    build_features = namespace.get("build_features")
    meta = dict(JOB["meta"])
    id_col, target = JOB["id_col"], JOB["target"]

    train = pd.read_parquet(JOB["train"])
    test = pd.read_parquet(JOB["test"])
    n_train = len(train)
    extra_y = None
    if JOB.get("extra_train") and JOB.get("extra_y"):
        # Extra rows ride along with train through build_features and the matrix (shared
        # vocabulary), then split off; they join the training part of every fold below.
        extra = pd.read_parquet(JOB["extra_train"])[list(train.columns)]
        extra_y = np.load(JOB["extra_y"])
        train = pd.concat([train, extra], ignore_index=True)
    new_columns: list[str] = []
    feature_seconds = 0.0
    if callable(build_features):
        train, test, new_columns, feature_seconds = _run_features(
            build_features, train, test, id_col, target
        )

    drop = set(meta.get("drop_columns", [])) | {id_col}
    features = [c for c in train.columns if c not in drop and c in test.columns]
    known_cats = set(meta.get("categorical", []))
    categorical = [
        c
        for c in features
        if c in known_cats or (c in new_columns and not pd.api.types.is_numeric_dtype(train[c]))
    ]
    if not features:
        _fail("no feature columns left after dropping id and unusable columns", "violation")
    matrix = prepare_matrix(train, test, features, categorical)
    X, X_test = matrix.X_train, matrix.X_test
    X_extra = None
    if extra_y is not None:
        X_extra = X.iloc[n_train:].reset_index(drop=True)
        X = X.iloc[:n_train].reset_index(drop=True)
        flag = JOB.get("extra_flag")
        if flag and flag not in features:
            X[flag], X_test[flag], X_extra[flag] = 0.0, 0.0, 1.0
            features = [*features, flag]
    y = np.load(JOB["y"])
    folds = np.load(JOB["folds"])
    n_classes = int(meta["n_classes"])
    n_folds = int(folds.max()) + 1
    meta.update(
        {
            "features": features,
            "categorical": matrix.categorical,
            "n_folds": n_folds,
            "n_extra": 0 if X_extra is None else len(X_extra),
            "extra_flag": JOB.get("extra_flag") if X_extra is not None else None,
        }
    )

    oof = np.zeros(len(X)) if n_classes == 2 else np.zeros((len(X), n_classes))
    test_sum = np.zeros(len(X_test)) if n_classes == 2 else np.zeros((len(X_test), n_classes))
    fold_seconds: list[float] = []
    for k in range(n_folds):
        tr = folds != k
        started = time.monotonic()
        X_tr, y_tr = X[tr], y[tr]
        if X_extra is not None:
            X_tr = pd.concat([X_tr, X_extra], ignore_index=True)
            y_tr = np.concatenate([y_tr, extra_y])
        try:
            p_va, p_te = _call_fit_predict(
                fit_predict, X_tr, y_tr, X[~tr], X_test, {**meta, "fold": k}
            )
        except Exception:
            _fail(f"fold {k}: " + traceback.format_exc(), "error")
        fold_seconds.append(time.monotonic() - started)
        oof[~tr] = p_va
        test_sum += p_te

    # Determinism on a subsample of fold 0: same inputs, same outputs, to 1e-9.
    rows = int(JOB.get("determinism_rows", 5000))
    tr0 = np.flatnonzero(folds != 0)[:rows]
    va0 = np.flatnonzero(folds == 0)[: max(50, rows // 5)]
    te0 = np.arange(min(len(X_test), max(50, rows // 5)))
    X_d, y_d = X.iloc[tr0], y[tr0]
    if X_extra is not None:  # the contract holds here too: extras are always in X_train
        X_d = pd.concat([X_d, X_extra], ignore_index=True)
        y_d = np.concatenate([y_d, extra_y])
    if len(tr0) >= 50 and len(va0) >= 20 and len(np.unique(y[tr0])) == n_classes:
        try:
            a = _call_fit_predict(
                fit_predict,
                X_d,
                y_d,
                X.iloc[va0],
                X_test.iloc[te0],
                {**meta, "fold": 0, "determinism_check": True},
            )
            b = _call_fit_predict(
                fit_predict,
                X_d,
                y_d,
                X.iloc[va0],
                X_test.iloc[te0],
                {**meta, "fold": 0, "determinism_check": True},
            )
        except Exception:
            _fail("determinism check: " + traceback.format_exc(), "error")
        delta = max(float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1] - b[1]).max()))
        if delta > 1e-9:
            _fail(
                "fit_predict is not deterministic: two calls on the same fold gave predictions "
                f"differing by up to {delta:.3g}. Seed every source of randomness (random_state, "
                "torch.manual_seed, fixed thread counts) or remove it.",
                "violation",
            )

    np.save(JOB["oof_out"], oof)
    np.save(JOB["test_pred_out"], test_sum / n_folds)
    _write_result(
        {
            "ok": True,
            "kind": "experiment",
            "n_classes": n_classes,
            "n_folds": n_folds,
            "features": features,
            "categorical": matrix.categorical,
            "new_columns": new_columns,
            "feature_seconds": feature_seconds,
            "fold_seconds": fold_seconds,
        }
    )


def _run_features_job(namespace) -> None:  # type: ignore[no-untyped-def]
    import pandas as pd

    fn = namespace.get("build_features")
    if not callable(fn):
        _fail("the module does not define build_features(train_df, test_df)", "violation")
    train_in = pd.read_parquet(JOB["train"])
    test_in = pd.read_parquet(JOB["test"])
    train_out, test_out, new_columns, duration = _run_features(
        fn, train_in, test_in, JOB["id_col"], JOB["target"]
    )
    train_out.to_parquet(JOB["train_out"], index=False)
    test_out.to_parquet(JOB["test_out"], index=False)
    _write_result(
        {
            "ok": True,
            "kind": "features",
            "new_columns": new_columns,
            "columns": [str(c) for c in train_out.columns],
            "dtypes": {str(c): str(train_out[c].dtype) for c in new_columns},
            "duration": duration,
            "n_train": len(train_out),
            "n_test": len(test_out),
        }
    )


def main() -> None:
    _limit_memory(int(JOB.get("memory_mb", 8192)))
    import runpy

    try:
        namespace = runpy.run_path(JOB["module"], run_name="feature_module")
    except SystemExit:
        _fail("the module called exit()", "violation")
    except Exception:
        _fail(traceback.format_exc(), "error")
    if JOB.get("kind", "features") == "experiment":
        _run_experiment(namespace)
    else:
        _run_features_job(namespace)


if __name__ == "__main__":
    main()
