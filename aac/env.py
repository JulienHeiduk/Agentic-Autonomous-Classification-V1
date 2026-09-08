"""Secrets and credential resolution.

Secrets live in a gitignored ``secrets.yml`` (flat ``NAME: value`` pairs) and are exported
into ``os.environ`` at startup. Variables already present in the environment always win, so
a shell or CI can override the file without editing it. The Kaggle access token is also
picked up from ``~/.kaggle/access_token``, the location the official CLI uses.

Nothing in this module ever logs or returns a secret value. ``credential_report`` reports
presence and source only, which is what ``aac doctor`` prints.
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

SECRETS_FILENAME = "secrets.yml"
SECRETS_FILE_ENV = "AAC_SECRETS_FILE"
KAGGLE_TOKEN_FILE = Path.home() / ".kaggle" / "access_token"

# Every credential or endpoint variable the framework reads, with its purpose.
KNOWN_VARS: dict[str, str] = {
    "NVIDIA_API_KEY": "NVIDIA inference hub bearer token",
    "NVIDIA_BASE_URL": "NVIDIA hub base URL (optional override)",
    "LOCAL_LLM_BASE_URL": "local OpenAI-compatible server base URL (optional override)",
    "KAGGLE_ACCESS_TOKEN": "Kaggle access token, sent as a bearer token",
    "KAGGLE_USERNAME": "Kaggle username (legacy basic auth fallback)",
    "KAGGLE_KEY": "Kaggle API key (legacy basic auth fallback)",
}

# Where each variable that we exported came from. Populated by load_secrets().
_SOURCES: dict[str, str] = {}


class SecretsError(RuntimeError):
    """The secrets file exists but cannot be used."""


def secrets_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the secrets file: explicit argument, then $AAC_SECRETS_FILE, then ./secrets.yml."""
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get(SECRETS_FILE_ENV)
    if from_env:
        return Path(from_env)
    return Path.cwd() / SECRETS_FILENAME


def load_secrets(
    path: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
    kaggle_token_file: Path | None = None,
) -> list[str]:
    """Export the secrets file (and the Kaggle token file) into os.environ.

    Returns the names that were set. A missing secrets file is not an error: the environment
    may already carry everything. Existing environment variables are kept unless ``override``.
    """
    loaded: list[str] = []
    file = secrets_path(path)
    if file.is_file():
        _warn_if_permissive(file)
        try:
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SecretsError(f"{file}: not valid YAML: {exc}") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            kind = type(data).__name__
            raise SecretsError(f"{file}: expected a mapping of NAME: value, got {kind}")
        for key, value in data.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise SecretsError(f"{file}: {key!r} is not a valid environment variable name")
            if value is None:
                continue
            if isinstance(value, dict | list):
                raise SecretsError(f"{file}: {key} must be a scalar, nested values are not allowed")
            if _export(key, str(value), source=file.name, override=override):
                loaded.append(key)

    token_file = KAGGLE_TOKEN_FILE if kaggle_token_file is None else kaggle_token_file
    if token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
        if token and _export(
            "KAGGLE_ACCESS_TOKEN", token, source="~/.kaggle/access_token", override=False
        ):
            loaded.append("KAGGLE_ACCESS_TOKEN")
    return loaded


def _export(key: str, value: str, *, source: str, override: bool) -> bool:
    if key in os.environ and not override:
        _SOURCES.setdefault(key, "environment")
        return False
    os.environ[key] = value
    _SOURCES[key] = source
    return True


def _warn_if_permissive(file: Path) -> None:
    mode = file.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("%s is readable by other users; run: chmod 600 %s", file, file)


@dataclass(frozen=True)
class CredentialStatus:
    name: str
    purpose: str
    present: bool
    source: str  # "environment", "secrets.yml", "~/.kaggle/access_token", or "missing"


def credential_report() -> list[CredentialStatus]:
    """Presence and source of every known variable. Never includes values."""
    report = []
    for name, purpose in KNOWN_VARS.items():
        present = bool(os.environ.get(name))
        source = _SOURCES.get(name, "environment") if present else "missing"
        report.append(CredentialStatus(name, purpose, present, source))
    return report
