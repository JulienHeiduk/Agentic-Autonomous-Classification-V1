"""The Plan: what the Architect emits and the Trainer executes (README section 3).

A plan is data. It names the columns to drop, the categorical columns, feature ideas for the
Engineer, per-fold target encoding, and the model families with their starting parameters.
``default_plan`` is the hardcoded plan that makes the deterministic core a working autopilot
without any LLM, and the fallback when every LLM branch fails.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, get_args

from pydantic import Field, ValidationInfo, model_validator

from aac.config import ModelFamily, StrictModel

TREE_COUNT_KEYS = ("n_estimators", "iterations", "max_iter")
MAX_MODELS = len(get_args(ModelFamily))  # a plan may use every family once

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    "lightgbm_focal": {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "alpha": 0.25,  # focal loss: positive-class weight
        "gamma": 2.0,  # focal loss: how hard confident rows are down-weighted
    },
    "xgboost": {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    "catboost": {
        "iterations": 800,
        "learning_rate": 0.06,
        "depth": 6,
        "l2_leaf_reg": 3.0,
    },
    "hist_gbdt": {
        "max_iter": 400,
        "learning_rate": 0.06,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
    },
    "logistic": {"C": 1.0, "max_iter": 2000},
}


class ModelConfig(StrictModel):
    family: ModelFamily
    params: dict[str, Any] = Field(default_factory=dict)
    early_stopping: bool = True  # on a seeded 10% carve-out of the training fold, never on OOF
    why: str = ""


class FeatureIdea(StrictModel):
    name: str
    description: str
    columns: list[str] = Field(default_factory=list)


class Plan(StrictModel):
    """What the Architect emits and the Trainer executes.

    Validate with ``context={"columns": [...], "families": [...]}`` to reject references to
    columns that do not exist or families that are not enabled; the JSON repair loop then
    shows the model exactly what was wrong.
    """

    name: str = Field(min_length=1, max_length=60)
    rationale: str = ""
    preprocessing: list[str] = Field(default_factory=list)  # prose steps for the Engineer
    drop_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] | None = None  # None: the Scout's inference
    features: list[FeatureIdea] = Field(default_factory=list, max_length=25)
    target_encode: list[str] = Field(default_factory=list)  # fitted inside each fold
    recipes: list[str] = Field(default_factory=list)  # deterministic feature recipes, in order
    models: list[ModelConfig] = Field(min_length=1, max_length=MAX_MODELS)

    @model_validator(mode="after")
    def _references(self, info: ValidationInfo) -> Plan:
        context = info.context or {}
        columns = context.get("columns")
        families = context.get("families")
        problems: list[str] = []
        if columns is not None:
            known = set(columns)
            refs = {
                "drop_columns": self.drop_columns,
                "categorical_columns": self.categorical_columns or [],
                "target_encode": self.target_encode,
            }
            for where, names in refs.items():
                unknown = [n for n in names if n not in known]
                if unknown:
                    problems.append(f"{where} refers to unknown columns {unknown}")
            defined: set[str] = set()  # a feature may build on features listed before it
            for i, idea in enumerate(self.features):
                unknown = [n for n in idea.columns if n not in known and n not in defined]
                if unknown:
                    problems.append(
                        f"features[{i}].columns refers to unknown columns {unknown} "
                        "(only input columns or features listed earlier in the plan)"
                    )
                defined.add(idea.name)
        if families is not None:
            allowed = set(families)
            bad = sorted({m.family for m in self.models if m.family not in allowed})
            if bad:
                problems.append(f"models use disabled families {bad}; allowed: {sorted(allowed)}")
        seen = [m.family for m in self.models]
        if len(seen) != len(set(seen)):
            problems.append(f"models list a family twice: {seen}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def resolved_params(self, model: ModelConfig, max_trees: int | None = None) -> dict[str, Any]:
        params = {**DEFAULT_PARAMS.get(model.family, {}), **model.params}
        if max_trees is not None:
            for key in TREE_COUNT_KEYS:
                if (
                    key in params
                    and isinstance(params[key], int | float)
                    and params[key] > max_trees
                ):
                    params[key] = max_trees
        return params

    def canonical(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]


BAGGABLE_FAMILIES = frozenset({"lightgbm", "lightgbm_focal", "xgboost", "catboost", "hist_gbdt"})


def usable_families(families: list[str], n_classes: int) -> list[str]:
    """The families that can fit this target: binary-only ones drop out for multiclass."""
    from aac.models.registry import supports_multiclass

    return [f for f in families if n_classes == 2 or supports_multiclass(f)]


def low_cardinality_numeric(profile: Any, max_unique: int) -> list[str]:
    """Usable numeric columns with few distinct values: integers that are really levels."""
    return [
        c.name
        for c in profile.columns
        if c.usable
        and c.kind == "numeric"
        and c.name != profile.id_col
        and 2 <= c.n_unique <= max_unique
    ]


def variant_plan(
    variant: str, profile: Any, families: list[str], low_cardinality_max: int
) -> Plan | None:
    """A deterministic plan variant, or None when the data gives it nothing to work on.

    ``categorical``: the low-cardinality integers join the categoricals for native handling.
    ``encoded``: categoricals and those integers are target-encoded inside each fold; the
    integers also stay as numbers.
    """
    if not families:
        return None
    levels = low_cardinality_numeric(profile, low_cardinality_max)
    cats = [c.name for c in profile.columns if c.usable and c.kind in ("categorical", "boolean")]
    drops = sorted(profile.unusable_columns())
    models = [ModelConfig(family=f) for f in families]
    if variant == "categorical":
        if not levels:
            return None
        return Plan(
            name="categorical",
            rationale=(
                f"Low-cardinality integer columns {levels} handled as categorical levels "
                "next to the true categoricals; otherwise the default plan."
            ),
            drop_columns=drops,
            categorical_columns=cats + levels,
            models=models,
        )
    if variant == "encoded":
        columns = cats + levels
        if not columns:
            return None
        return Plan(
            name="encoded",
            rationale=(
                f"Fold-internal target encoding of {columns}; integers keep their numeric "
                "column as well; otherwise the default plan."
            ),
            drop_columns=drops,
            target_encode=columns,
            models=models,
        )
    if variant == "digits":
        from aac.models.features import digit_columns

        fine = digit_columns(profile)
        if not fine:
            return None
        numeric = [
            c.name
            for c in profile.columns
            if c.usable and c.kind == "numeric" and c.name != profile.id_col
        ]
        return Plan(
            name="digits",
            rationale=(
                f"Generator artefacts: low-order digits and moduli of {fine}, how often each "
                "exact value occurs (train, test and extra rows), and fold-internal exact-value "
                "target encoding of every raw column. Public S6E9 notebooks reach 0.946 with this."
            ),
            drop_columns=drops,
            target_encode=cats + numeric,
            recipes=["digits", "value_frequency"],
            models=models,
        )
    raise ValueError(f"unknown plan variant {variant!r}")


def default_plan(families: list[str], drop_columns: list[str]) -> Plan:
    return Plan(
        name="default",
        rationale=(
            "Deterministic baseline: every usable column as-is, native categorical handling, "
            "library defaults tuned for tabular playground data, no feature engineering."
        ),
        drop_columns=sorted(drop_columns),
        models=[ModelConfig(family=f) for f in families],
    )
