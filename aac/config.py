"""Configuration loading and validation (README section 11).

Config files are YAML. String values may reference environment variables as ``${VAR}`` or
``${VAR:-default}``. Expansion happens after parsing, on string leaves only, so a secret
containing YAML-significant characters can never break the file. A value that is exactly one
unset reference with no default becomes ``None``; the name is collected so ``aac doctor`` can
report it and ``aac run`` can refuse to start.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)


class ConfigError(ValueError):
    """The config file is missing, malformed, or fails validation."""


_ENV_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def expand_env(value: Any, missing: list[str], env: Mapping[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` in string leaves.

    Unset variables without a default are appended to ``missing``. A string that is exactly one
    such reference becomes ``None``; inside a longer string the reference becomes ``""``.
    """
    env = os.environ if env is None else env
    if isinstance(value, str):
        whole = _ENV_RE.fullmatch(value)
        if whole and whole.group("default") is None and whole.group("name") not in env:
            missing.append(whole.group("name"))
            return None

        def sub(match: re.Match[str]) -> str:
            name, default = match.group("name"), match.group("default")
            if name in env:
                return env[name]
            if default is not None:
                return default
            missing.append(name)
            return ""

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v, missing, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, missing, env) for v in value]
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


MetricKey = Literal["auc", "accuracy", "logloss", "f1", "macro_f1"]
ModelFamily = Literal["lightgbm", "lightgbm_focal", "xgboost", "catboost", "hist_gbdt", "logistic"]
Tier = Literal["cheap", "code", "reason"]
TIERS: tuple[Tier, ...] = get_args(Tier)


class FilesConfig(StrictModel):
    """Override the filename heuristic: paths or globs relative to the extracted bundle."""

    train: str | None = None
    test: str | None = None
    sample_submission: str | None = None


class ExtraData(StrictModel):
    """A Kaggle dataset appended to the training side of every fold (README 17.2): the
    original data a Playground competition was generated from, never validation or test."""

    dataset: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")  # owner/slug
    file: str | None = None  # a file inside the dataset; None: its only tabular file
    rename: dict[str, str] = Field(default_factory=dict)  # dataset column -> train column
    dedupe: bool = True  # drop rows identical to a synthetic training row


class CompetitionConfig(StrictModel):
    slug: str = Field(min_length=1)
    target: str | None = None  # None: the Scout infers it
    id_col: str | None = None
    files: FilesConfig = FilesConfig()
    extra_train: list[ExtraData] = Field(default_factory=list)
    # Feature added when extra rows are present: 1 on them, 0 on the synthetic rows and test.
    # None: no flag column.
    extra_flag: str | None = "is_original"
    # What the owner knows about this competition (a leak, a trick, a trap): each entry
    # becomes a note every Researcher reads in every round.
    notes: list[str] = Field(default_factory=list)
    # Explicit metric override, only for when Kaggle's display name is not in the mapping
    # table. The framework never guesses a metric on its own.
    metric: MetricKey | None = None


class RunConfig(StrictModel):
    seed: int = 42
    n_folds: int = Field(5, ge=2)
    max_iters_per_branch: int = Field(3, ge=1)
    patience: int = Field(2, ge=1)
    min_improvement: float = Field(0.0005, ge=0)
    max_wall_clock_minutes: float = Field(180, gt=0)
    max_tokens_total: int = Field(2_000_000, gt=0)
    parallel_branches: int = Field(2, ge=1)
    leak_threshold: float = Field(0.05, gt=0)
    degenerate_margin: float = Field(0.002, ge=0)  # min OOF gain over a constant prediction
    # round_robin: every seat gets round n before any seat gets round n+1, so the budget is
    # spread across tracks; sequential: one seat runs all its rounds, then the next.
    schedule: Literal["round_robin", "sequential"] = "round_robin"
    n_jobs: int = Field(4, ge=1)  # fixed thread count: determinism needs it, never -1
    max_trees: int = Field(3000, ge=1)  # clamp on n_estimators / iterations / max_iter from plans
    max_rounds: int = Field(4, ge=1)  # experiments per Researcher unless the spec overrides it
    max_consecutive_failures: int = Field(
        4, ge=1
    )  # failed rounds in a row before a Researcher stops
    share_every: int = Field(2, ge=1)  # rounds between leader-to-trailer sharing (M8)
    max_classes: int = Field(50, ge=2)  # more distinct target values than this is refused


PlanVariant = Literal["categorical", "encoded", "digits"]


class ModelsConfig(StrictModel):
    enabled: list[ModelFamily] = Field(default_factory=lambda: list(get_args(ModelFamily)))
    tuning: Literal["none", "optuna"] = "optuna"
    tuning_trials: int = Field(30, ge=1)
    # Deterministic plan variants trained after the default plan (README 17.2): "categorical"
    # treats low-cardinality integer columns as categorical; "encoded" target-encodes the
    # categoricals and those integers inside each fold. Only variant_families run on them.
    variants: list[PlanVariant] = Field(default_factory=lambda: ["digits", "encoded"])
    # None: every enabled family whose default-plan training took at most
    # variant_max_seconds; a list pins the families regardless of time.
    variant_families: list[ModelFamily] | None = None
    variant_max_seconds: float = Field(120, gt=0)
    low_cardinality_max: int = Field(50, ge=2)  # numeric columns with at most this many values
    # Seed bagging: the best seed_bag_top tree families are retrained with seed_bag extra seeds
    # and every replica joins the pool. 0 disables. Families slower than seed_bag_max_seconds
    # on the base seed are skipped.
    seed_bag: int = Field(2, ge=0)
    seed_bag_top: int = Field(2, ge=1)
    seed_bag_max_seconds: float = Field(120, gt=0)

    @field_validator("variants")
    @classmethod
    def _unique_variants(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("models.variants has duplicates")
        return v

    @field_validator("enabled")
    @classmethod
    def _non_empty_unique(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("models.enabled must list at least one family")
        if len(set(v)) != len(v):
            raise ValueError("models.enabled has duplicates")
        return v


class KaggleConfig(StrictModel):
    submit: bool = True
    max_submissions_per_day: int = Field(5, ge=1)


class SandboxConfig(StrictModel):
    timeout_seconds: float = Field(900, gt=0)  # build_features runs
    experiment_timeout_seconds: float = Field(1800, gt=0)  # a whole fit_predict CV run
    memory_mb: int = Field(8192, gt=0)
    determinism_rows: int = Field(5000, ge=100)


class BackendConfig(StrictModel):
    base_url: str
    model: str = Field(min_length=1)
    api_key: SecretStr | None = None
    cost_per_1k: float = Field(0.0, ge=0)  # USD per 1k tokens, blended; 0 for local
    timeout: float = Field(120.0, gt=0)
    supports_json_mode: bool = True
    max_concurrency: int = Field(1, ge=1)
    # Where a call goes after this backend fails its retries. None: the other backend in
    # config order with its default model. The same backend with another model is allowed.
    fallback_backend: str | None = None
    fallback_model: str | None = None
    # Circuit breaker: after trip_after consecutive failed calls (each already retried) the
    # backend is skipped in favour of its fallback for cooldown_seconds.
    trip_after: int = Field(2, ge=1)
    cooldown_seconds: float = Field(600, ge=0)

    @field_validator("base_url")
    @classmethod
    def _url(cls, v: str) -> str:
        v = v.rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {v!r}")
        return v

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, v: Any) -> Any:
        return None if v in ("", None) else v


class RouterConfig(StrictModel):
    cheap: str = "local"
    code: str = "local"
    reason: str = "nvidia"


Track = Literal["gbdt", "linear", "neural", "open"]


class ResearcherSpec(StrictModel):
    backend: str
    model: str | None = None  # None: the backend's default model
    track: Track = "open"
    temperature: float = Field(0.4, ge=0, le=2)
    rounds: int | None = Field(None, ge=1)  # None: run.max_rounds


class ScholarConfig(StrictModel):
    backend: str
    model: str | None = None
    temperature: float = Field(0.7, ge=0, le=2)
    angles: list[str] | None = None  # None: the three default angles
    max_ideas: int = Field(8, ge=1, le=20)
    # Packets are competition knowledge, not run knowledge: reuse the latest ones for up to
    # this many later runs on the slug before asking again. 0: ask every run.
    reuse_runs: int = Field(3, ge=0)


class AssessorConfig(StrictModel):
    enabled: bool = True
    rows: int = Field(4000, ge=500)  # subsample of the competition's train rows
    n_folds: int = Field(3, ge=2)
    timeout_seconds: float = Field(300, gt=0)
    skip_after_failures: int = Field(2, ge=1)  # interviews with no working module before skipping


class Config(StrictModel):
    competition: CompetitionConfig
    run: RunConfig = RunConfig()
    models: ModelsConfig = ModelsConfig()
    kaggle: KaggleConfig = KaggleConfig()
    sandbox: SandboxConfig = SandboxConfig()
    backends: dict[str, BackendConfig] = Field(min_length=1)
    router: RouterConfig = RouterConfig()
    researchers: list[ResearcherSpec] = Field(default_factory=list)
    scholar: ScholarConfig | None = None
    assessor: AssessorConfig = AssessorConfig()

    _missing_env: list[str] = PrivateAttr(default_factory=list)
    _source: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _cross_references(self) -> Config:
        for tier in TIERS:
            name = getattr(self.router, tier)
            if name not in self.backends:
                raise ValueError(f"router.{tier} refers to unknown backend {name!r}")
        if self.scholar is not None and self.scholar.backend not in self.backends:
            raise ValueError(f"scholar refers to unknown backend {self.scholar.backend!r}")
        for i, spec in enumerate(self.researchers):
            if spec.backend not in self.backends:
                raise ValueError(f"researchers[{i}] refers to unknown backend {spec.backend!r}")
        for name, backend in self.backends.items():
            if (
                backend.fallback_backend is not None
                and backend.fallback_backend not in self.backends
            ):
                raise ValueError(
                    f"backends.{name}.fallback_backend refers to unknown backend "
                    f"{backend.fallback_backend!r}"
                )
            if backend.fallback_model is not None and backend.fallback_backend is None:
                raise ValueError(f"backends.{name}.fallback_model needs fallback_backend")
        return self

    @property
    def missing_env(self) -> list[str]:
        return list(self._missing_env)

    @property
    def source(self) -> Path | None:
        return self._source

    def scholar_spec(self) -> ScholarConfig | None:
        if self.scholar is None:
            return None
        return self.scholar.model_copy(
            update={"model": self.scholar.model or self.backends[self.scholar.backend].model}
        )

    def researcher_specs(self) -> list[ResearcherSpec]:
        return [
            spec.model_copy(update={"model": spec.model or self.backends[spec.backend].model})
            for spec in self.researchers
        ]

    def redacted(self) -> dict[str, Any]:
        """JSON-safe dump with secrets masked; safe to write into the run directory."""
        return self.model_dump(mode="json")  # SecretStr serialises as '**********'

    def fingerprint(self) -> str:
        """Stable hash of everything that affects a run's behaviour. Secrets are excluded."""
        canonical = json.dumps(self.redacted(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_config(
    path: str | os.PathLike[str],
    *,
    strict: bool = True,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Read, expand, and validate a config file.

    With ``strict`` (the default) any unset ``${VAR}`` without a default is an error. Doctor
    loads with ``strict=False`` and reports the names instead.
    """
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"config file not found: {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{file}: not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{file}: top level must be a mapping")
    missing: list[str] = []
    expanded = expand_env(raw, missing, env)
    if strict and missing:
        raise ConfigError(f"{file}: unset environment variables: {', '.join(sorted(set(missing)))}")
    try:
        config = Config.model_validate(expanded)
    except ValueError as exc:
        raise ConfigError(f"{file}: {exc}") from exc
    config._missing_env = sorted(set(missing))
    config._source = file
    return config


def resolved_yaml(config: Config) -> str:
    """The effective config with secrets masked, for the run directory snapshot."""
    return yaml.safe_dump(config.redacted(), sort_keys=False)
