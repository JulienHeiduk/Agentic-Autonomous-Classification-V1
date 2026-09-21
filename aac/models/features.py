"""Deterministic feature recipes for plan variants (README 17.2).

Synthetic Playground tables leak target information through generator artefacts that a tree
cannot see in a raw number: the low-order digits of a value, how often the exact value
occurs, and the exact value itself (target-encoded inside each fold by ``cross_validate``).
Every recipe here is target-free and fold-agnostic: it looks only at the feature values of
train, test and any extra rows, so it can run once before the folds are cut.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

RECIPES = ("digits", "value_frequency")
MIN_DISTINCT_FOR_DIGITS = 100  # a column with fewer exact values has no digit structure
MAX_DECIMALS = 3


def integer_scale(values: pd.Series) -> int | None:
    """The power of ten that turns the column's values into integers (1 for whole numbers,
    10 for one decimal, ...), or None when it takes more than ``MAX_DECIMALS`` decimals."""
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if v.size == 0:
        return None
    for decimals in range(MAX_DECIMALS + 1):
        scale = 10**decimals
        scaled = v * scale
        if np.allclose(scaled, np.round(scaled), atol=1e-6):
            return scale
    return None


def digit_columns(profile: Any, min_distinct: int = MIN_DISTINCT_FOR_DIGITS) -> list[str]:
    """Numeric columns with enough distinct values for digits and moduli to mean anything."""
    return [
        c.name
        for c in profile.columns
        if c.usable
        and c.kind == "numeric"
        and c.name != profile.id_col
        and c.n_unique >= min_distinct
    ]


def _digits(frame: pd.DataFrame, col: str, scale: int) -> dict[str, np.ndarray]:
    x = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
    missing = np.isnan(x)
    v = np.round(np.nan_to_num(x) * scale).astype(np.int64)
    out = {
        f"{col}__d1": (v % 10).astype(np.float64),
        f"{col}__d2": (v // 10 % 10).astype(np.float64),
        f"{col}__d3": (v // 100 % 10).astype(np.float64),
        f"{col}__mod100": (v % 100).astype(np.float64),
        f"{col}__mod1000": (v % 1000).astype(np.float64),
    }
    for arr in out.values():
        arr[missing] = np.nan
    return out


def apply_recipes(
    recipes: list[str],
    profile: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    extra: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Return copies of the frames with the recipes' columns appended, and the new names."""
    if not recipes:
        return train, test, extra, []
    unknown = [r for r in recipes if r not in RECIPES]
    if unknown:
        raise ValueError(f"unknown feature recipes {unknown}; known: {list(RECIPES)}")
    frames = [train.copy(), test.copy()] + ([extra.copy()] if extra is not None else [])
    new: list[str] = []
    feature_cols = [c.name for c in profile.columns if c.usable and c.name != profile.id_col]
    if "digits" in recipes:
        for col in digit_columns(profile):
            scale = integer_scale(pd.concat([f[col] for f in frames], ignore_index=True))
            if scale is None:
                continue
            for frame in frames:
                for name, arr in _digits(frame, col, scale).items():
                    frame[name] = arr
            new.extend(f"{col}__{s}" for s in ("d1", "d2", "d3", "mod100", "mod1000"))
    if "value_frequency" in recipes:
        for col in feature_cols:
            key = [
                f[col].astype("object").where(f[col].notna(), "__missing__").astype(str)
                for f in frames
            ]
            counts = pd.concat(key, ignore_index=True).value_counts()
            for frame, k in zip(frames, key, strict=True):
                frame[f"{col}__freq"] = k.map(counts).astype(np.float64).to_numpy()
            new.append(f"{col}__freq")
    train_out, test_out = frames[0], frames[1]
    extra_out = frames[2] if extra is not None else None
    return train_out, test_out, extra_out, new
