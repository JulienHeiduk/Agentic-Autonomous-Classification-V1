"""Execution sandbox for LLM-authored feature modules (README section 7).

Two layers. ``check_source`` is a static AST pass: import allowlist, banned calls and dunder
escape hatches, the required function signature, and the leakage tripwire (any mention of the
target column). ``run_feature_module`` then executes the module in a separate ``python -I``
process with a wall-clock timeout, a memory cap where the OS honours it, a socket shim, and
write access limited to the work directory. The process also runs the function twice and
refuses non-deterministic output.

This is best-effort isolation on macOS (no network namespaces), which is why the static
pass and the runner shims both exist. Never run untrusted code from other people this way.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aac.exec.artifacts import atomic_write_json, atomic_write_text
from aac.models.metrics import MetricSpec, score

log = logging.getLogger(__name__)

RUNNER = Path(__file__).with_name("_runner.py")
FUNCTION_NAME = "build_features"

# README section 7 list plus a few harmless stdlib modules feature code routinely needs.
ALLOWED_IMPORTS = frozenset(
    {
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "itertools",
        "math",
        "re",
        "collections",
        "warnings",
        "typing",
        "functools",
        "string",
        "datetime",
        "dataclasses",
    }
)
EXPERIMENT_IMPORTS = ALLOWED_IMPORTS | frozenset(
    {"lightgbm", "xgboost", "catboost", "torch", "optuna"}
)
FIT_PREDICT = "fit_predict"

BANNED_CALLS = frozenset(
    {
        "open",
        "eval",
        "exec",
        "__import__",
        "compile",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "help",
    }
)
BANNED_NAMES = frozenset({"__builtins__", "__import__", "__loader__", "__spec__"})
BANNED_ATTRS = frozenset(
    {
        "__subclasses__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__loader__",
        "__spec__",
        "__import__",
        "__dict__",
        "__class__",
        "__bases__",
        "__mro__",
        "__getattribute__",
        "system",
        "popen",
        "spawn",
        "fork",
        "execv",
        "execve",
    }
)


def check_source(
    code: str,
    *,
    target: str,
    allowed_imports: frozenset[str] | None = None,
    mode: str = "features",
) -> list[str]:
    """Problems that make the module unrunnable. Empty means the static pass is clean.

    ``mode="features"`` requires ``build_features(train_df, test_df)``; ``mode="experiment"``
    requires ``fit_predict(X_train, y_train, X_valid, X_test, meta)``, allows an optional
    ``build_features``, and uses the wider model-library allowlist.
    """
    if allowed_imports is None:
        allowed_imports = EXPERIMENT_IMPORTS if mode == "experiment" else ALLOWED_IMPORTS
    problems: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]
    target_lower = target.lower()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_imports:
                    problems.append(f"line {node.lineno}: import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in allowed_imports:
                problems.append(f"line {node.lineno}: import from {node.module!r} is not allowed")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in BANNED_CALLS
        ):
            problems.append(f"line {node.lineno}: call to {node.func.id}() is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"line {node.lineno}: use of {node.id} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                problems.append(f"line {node.lineno}: attribute .{node.attr} is not allowed")
            if node.attr.lower() == target_lower:
                problems.append(f"line {node.lineno}: references the target column {target!r}")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.strip().lower() == target_lower
        ):
            problems.append(f"line {node.lineno}: references the target column {target!r}")
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def arity(fn: ast.FunctionDef) -> int | None:
        args = fn.args
        if args.vararg or args.kwonlyargs:
            return None
        return len(args.posonlyargs) + len(args.args)

    if mode == "experiment":
        fit = defs.get(FIT_PREDICT)
        if fit is None:
            problems.append(
                f"no top-level def {FIT_PREDICT}(X_train, y_train, X_valid, X_test, meta)"
            )
        elif arity(fit) != 5:
            problems.append(f"{FIT_PREDICT} must take exactly five positional arguments")
        feat = defs.get(FUNCTION_NAME)
        if feat is not None and arity(feat) != 2:
            problems.append(f"{FUNCTION_NAME} must take exactly two positional arguments")
    else:
        feat = defs.get(FUNCTION_NAME)
        if feat is None:
            problems.append(f"no top-level def {FUNCTION_NAME}(train_df, test_df)")
        elif arity(feat) != 2:
            problems.append(f"{FUNCTION_NAME} must take exactly two positional arguments")
    return sorted(
        set(problems),
        key=lambda p: (int(p.split(":")[0].split()[-1]) if p.startswith("line") else 0, p),
    )


@dataclass
class SandboxResult:
    ok: bool
    kind: str  # ok, rejected, error, violation, timeout; researcher adds track, leak, degenerate
    error: str = ""
    new_columns: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    train_out: Path | None = None
    test_out: Path | None = None
    duration: float = 0.0
    stdout: str = ""
    stderr: str = ""

    def summary(self) -> str:
        if self.ok:
            return f"ok: {len(self.new_columns)} new columns in {self.duration:.1f}s"
        return f"{self.kind}: {self.error.strip().splitlines()[-1][:200] if self.error else ''}"


def _tail(text: str, lines: int = 40, chars: int = 4000) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])[-chars:]


def run_feature_module(
    code: str,
    *,
    workdir: Path,
    train_path: Path,
    test_path: Path,
    target: str,
    id_col: str,
    timeout: float = 900.0,
    memory_mb: int = 8192,
    n_threads: int = 4,
) -> SandboxResult:
    """Static check, then execute in a subprocess. Inputs must already be target-free."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    module_path = workdir / "features.py"
    atomic_write_text(module_path, code)
    problems = check_source(code, target=target)
    if problems:
        error = "static check rejected the module:\n" + "\n".join(f"- {p}" for p in problems)
        atomic_write_json(
            workdir / "result.json", {"ok": False, "kind": "rejected", "error": error}
        )
        return SandboxResult(False, "rejected", error)

    train_out = workdir / "train_features.parquet"
    test_out = workdir / "test_features.parquet"
    for stale in (train_out, test_out, workdir / "result.json"):
        stale.unlink(missing_ok=True)
    atomic_write_json(
        workdir / "job.json",
        {
            "module": str(module_path),
            "train": str(Path(train_path).resolve()),
            "test": str(Path(test_path).resolve()),
            "train_out": str(train_out),
            "test_out": str(test_out),
            "target": target,
            "id_col": id_col,
            "memory_mb": memory_mb,
        },
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": str(n_threads),
        "OPENBLAS_NUM_THREADS": str(n_threads),
        "MKL_NUM_THREADS": str(n_threads),
        "no_proxy": "*",
    }
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(RUNNER), str(workdir)],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            (exc.stdout or b"").decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            (exc.stderr or b"").decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        error = f"build_features did not finish within {timeout:.0f}s and was killed"
        atomic_write_text(workdir / "stdout.log", stdout)
        atomic_write_text(workdir / "stderr.log", stderr)
        atomic_write_json(workdir / "result.json", {"ok": False, "kind": "timeout", "error": error})
        return SandboxResult(
            False, "timeout", error, duration=timeout, stdout=stdout, stderr=stderr
        )
    duration = time.monotonic() - started
    atomic_write_text(workdir / "stdout.log", proc.stdout)
    atomic_write_text(workdir / "stderr.log", proc.stderr)
    result_path = workdir / "result.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text())
        except ValueError:
            payload = {}
    else:
        payload = {}
    if payload.get("ok"):
        return SandboxResult(
            True,
            "ok",
            new_columns=list(payload.get("new_columns", [])),
            columns=list(payload.get("columns", [])),
            train_out=train_out,
            test_out=test_out,
            duration=float(payload.get("duration", duration)),
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    kind = str(payload.get("kind") or ("error" if proc.returncode else "violation"))
    error = str(payload.get("error") or "")
    if not error:
        error = f"runner exited with code {proc.returncode}\n{_tail(proc.stderr)}"
    return SandboxResult(
        False, kind, error, duration=duration, stdout=proc.stdout, stderr=proc.stderr
    )


@dataclass
class ExperimentResult:
    ok: bool
    kind: str  # ok, rejected, error, violation, timeout; researcher adds track, leak, degenerate
    error: str = ""
    oof: np.ndarray | None = None
    test_pred: np.ndarray | None = None
    fold_scores: list[float] = field(default_factory=list)
    oof_score: float | None = None
    features: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    new_columns: list[str] = field(default_factory=list)
    fold_seconds: list[float] = field(default_factory=list)
    feature_seconds: float = 0.0
    duration: float = 0.0
    stdout: str = ""
    stderr: str = ""

    @property
    def cv_mean(self) -> float | None:
        return float(np.mean(self.fold_scores)) if self.fold_scores else None

    @property
    def cv_std(self) -> float | None:
        return float(np.std(self.fold_scores)) if self.fold_scores else None

    def summary(self) -> str:
        if self.ok:
            return (
                f"ok: OOF {self.oof_score:.5f} (folds {self.cv_mean:.5f} +/- {self.cv_std:.5f}), "
                f"{len(self.features)} features, {self.duration:.0f}s"
            )
        last = self.error.strip().splitlines()[-1][:200] if self.error else ""
        return f"{self.kind}: {last}"

    def metrics(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "error": self.error[-2000:] if self.error else "",
            "oof_score": self.oof_score,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "fold_scores": self.fold_scores,
            "fold_seconds": self.fold_seconds,
            "feature_seconds": self.feature_seconds,
            "duration": self.duration,
            "n_features": len(self.features),
            "features": self.features,
            "categorical": self.categorical,
            "new_columns": self.new_columns,
        }


def _run_subprocess(
    workdir: Path, *, timeout: float, n_threads: int
) -> tuple[subprocess.CompletedProcess[str] | None, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": str(n_threads),
        "OPENBLAS_NUM_THREADS": str(n_threads),
        "MKL_NUM_THREADS": str(n_threads),
        "no_proxy": "*",
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(RUNNER), str(workdir)],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out = (
            exc.stdout
            if isinstance(exc.stdout, str)
            else (exc.stdout or b"").decode(errors="replace")
        )
        err = (
            exc.stderr
            if isinstance(exc.stderr, str)
            else (exc.stderr or b"").decode(errors="replace")
        )
        atomic_write_text(workdir / "stdout.log", out)
        atomic_write_text(workdir / "stderr.log", err)
        return None, f"did not finish within {timeout:.0f}s and was killed"
    atomic_write_text(workdir / "stdout.log", proc.stdout)
    atomic_write_text(workdir / "stderr.log", proc.stderr)
    return proc, ""


def run_experiment(
    code: str,
    *,
    workdir: Path,
    train_path: Path,
    test_path: Path,
    y: np.ndarray,
    folds: np.ndarray,
    metric: MetricSpec,
    target: str,
    id_col: str,
    n_classes: int,
    categorical: list[str],
    drop_columns: list[str],
    seed: int,
    timeout: float = 1800.0,
    memory_mb: int = 8192,
    n_threads: int = 4,
    determinism_rows: int = 5000,
    extra_train: Path | None = None,
    extra_y: np.ndarray | None = None,
    extra_flag: str | None = None,
) -> ExperimentResult:
    """Static check, then run the module's fit_predict per fold in the subprocess and score.

    ``extra_train`` (a target-free frame with the train columns) and ``extra_y`` are rows the
    harness appends to every fold's training part, never to validation."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    module_path = workdir / "experiment.py"
    atomic_write_text(module_path, code)
    problems = check_source(code, target=target, mode="experiment")
    if problems:
        error = "static check rejected the module:\n" + "\n".join(f"- {p}" for p in problems)
        atomic_write_json(
            workdir / "result.json", {"ok": False, "kind": "rejected", "error": error}
        )
        return ExperimentResult(False, "rejected", error)

    oof_path, pred_path = workdir / "oof.npy", workdir / "test_pred.npy"
    y_path, folds_path = workdir / "y.npy", workdir / "folds.npy"
    for stale in (oof_path, pred_path, workdir / "result.json"):
        stale.unlink(missing_ok=True)
    np.save(y_path, np.asarray(y))
    np.save(folds_path, np.asarray(folds))
    extra_y_path = None
    if extra_train is not None and extra_y is not None and len(extra_y):
        extra_y_path = workdir / "extra_y.npy"
        np.save(extra_y_path, np.asarray(extra_y))
    atomic_write_json(
        workdir / "job.json",
        {
            "kind": "experiment",
            "module": str(module_path),
            "train": str(Path(train_path).resolve()),
            "test": str(Path(test_path).resolve()),
            "extra_train": str(Path(extra_train).resolve()) if extra_y_path else None,
            "extra_y": str(extra_y_path) if extra_y_path else None,
            "extra_flag": extra_flag if extra_y_path else None,
            "y": str(y_path),
            "folds": str(folds_path),
            "oof_out": str(oof_path),
            "test_pred_out": str(pred_path),
            "target": target,
            "id_col": id_col,
            "memory_mb": memory_mb,
            "determinism_rows": determinism_rows,
            "meta": {
                "n_classes": n_classes,
                "categorical": list(categorical),
                "drop_columns": list(drop_columns),
                "seed": seed,
                "n_threads": n_threads,
                "metric": metric.key,
                "greater_is_better": metric.greater_is_better,
            },
        },
    )
    started = time.monotonic()
    proc, timeout_error = _run_subprocess(workdir, timeout=timeout, n_threads=n_threads)
    duration = time.monotonic() - started
    if proc is None:
        atomic_write_json(
            workdir / "result.json", {"ok": False, "kind": "timeout", "error": timeout_error}
        )
        return ExperimentResult(
            False, "timeout", f"the experiment {timeout_error}", duration=duration
        )
    payload: dict[str, Any] = {}
    if (workdir / "result.json").exists():
        try:
            payload = json.loads((workdir / "result.json").read_text())
        except ValueError:
            payload = {}
    if not payload.get("ok"):
        kind = str(payload.get("kind") or ("error" if proc.returncode else "violation"))
        error = str(payload.get("error") or "")
        if not error:
            error = f"runner exited with code {proc.returncode}\n{_tail(proc.stderr)}"
        return ExperimentResult(
            False, kind, error, duration=duration, stdout=proc.stdout, stderr=proc.stderr
        )

    oof = np.load(oof_path)
    test_pred = np.load(pred_path)
    y_arr = np.asarray(y)
    fold_scores = [
        score(metric, y_arr[folds == k], oof[folds == k]) for k in range(int(folds.max()) + 1)
    ]
    result = ExperimentResult(
        True,
        "ok",
        oof=oof,
        test_pred=test_pred,
        fold_scores=fold_scores,
        oof_score=score(metric, y_arr, oof),
        features=list(payload.get("features", [])),
        categorical=list(payload.get("categorical", [])),
        new_columns=list(payload.get("new_columns", [])),
        fold_seconds=[float(t) for t in payload.get("fold_seconds", [])],
        feature_seconds=float(payload.get("feature_seconds", 0.0)),
        duration=duration,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
    atomic_write_json(workdir / "metrics.json", result.metrics())
    return result
