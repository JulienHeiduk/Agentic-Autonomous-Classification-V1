import os

import pytest

from aac import env
from aac.env import SecretsError, credential_report, load_secrets, secrets_path

NO_TOKEN = object()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Empty environment for every known var, cwd in tmp, no ~/.kaggle token."""
    for name in env.KNOWN_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(env.SECRETS_FILE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env, "_SOURCES", {})
    return tmp_path


def _load(path=None, **kw):
    kw.setdefault("kaggle_token_file", env.Path("/nonexistent/access_token"))
    return load_secrets(path, **kw)


def test_missing_file_is_not_an_error(isolated):
    assert _load() == []


def test_flat_mapping_is_exported(isolated):
    (isolated / "secrets.yml").write_text("NVIDIA_API_KEY: abc\nKAGGLE_ACCESS_TOKEN: KGAT_x\n")
    assert sorted(_load()) == ["KAGGLE_ACCESS_TOKEN", "NVIDIA_API_KEY"]
    assert os.environ["NVIDIA_API_KEY"] == "abc"


def test_environment_wins_over_file(isolated, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "from-env")
    (isolated / "secrets.yml").write_text("NVIDIA_API_KEY: from-file\n")
    assert _load() == []
    assert os.environ["NVIDIA_API_KEY"] == "from-env"
    assert _load(override=True) == ["NVIDIA_API_KEY"]
    assert os.environ["NVIDIA_API_KEY"] == "from-file"


def test_explicit_path_and_env_var_resolution(isolated, monkeypatch):
    assert secrets_path() == isolated / "secrets.yml"
    monkeypatch.setenv(env.SECRETS_FILE_ENV, "/elsewhere/s.yml")
    assert str(secrets_path()) == "/elsewhere/s.yml"
    assert str(secrets_path("/explicit.yml")) == "/explicit.yml"


def test_null_values_are_skipped_and_non_strings_stringified(isolated):
    (isolated / "secrets.yml").write_text("KAGGLE_KEY: null\nKAGGLE_USERNAME: 12345\n")
    assert _load() == ["KAGGLE_USERNAME"]
    assert os.environ["KAGGLE_USERNAME"] == "12345"
    assert "KAGGLE_KEY" not in os.environ


@pytest.mark.parametrize(
    "content",
    ["- a\n- b\n", "NVIDIA_API_KEY:\n  nested: 1\n", "not a name: x\n", "KEY: [1, 2]\n"],
)
def test_malformed_files_raise(isolated, content):
    (isolated / "secrets.yml").write_text(content)
    with pytest.raises(SecretsError):
        _load()


def test_invalid_yaml_raises(isolated):
    (isolated / "secrets.yml").write_text("KEY: [unclosed\n")
    with pytest.raises(SecretsError):
        _load()


def test_kaggle_token_file_fallback(isolated):
    token_file = isolated / "access_token"
    token_file.write_text("KGAT_fromfile\n")
    assert load_secrets(kaggle_token_file=token_file) == ["KAGGLE_ACCESS_TOKEN"]
    assert os.environ["KAGGLE_ACCESS_TOKEN"] == "KGAT_fromfile"


def test_secrets_file_beats_kaggle_token_file(isolated):
    (isolated / "secrets.yml").write_text("KAGGLE_ACCESS_TOKEN: KGAT_yaml\n")
    token_file = isolated / "access_token"
    token_file.write_text("KGAT_fromfile\n")
    load_secrets(kaggle_token_file=token_file)
    assert os.environ["KAGGLE_ACCESS_TOKEN"] == "KGAT_yaml"


def test_credential_report_gives_sources_not_values(isolated, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "env-secret")
    (isolated / "secrets.yml").write_text("KAGGLE_ACCESS_TOKEN: file-secret\n")
    token_file = isolated / "access_token"
    token_file.write_text("unused")
    load_secrets(kaggle_token_file=token_file)
    by_name = {c.name: c for c in credential_report()}
    assert by_name["NVIDIA_API_KEY"].source == "environment"
    assert by_name["KAGGLE_ACCESS_TOKEN"].source == "secrets.yml"
    assert by_name["KAGGLE_KEY"].source == "missing" and not by_name["KAGGLE_KEY"].present
    assert set(by_name) == set(env.KNOWN_VARS)
    dumped = repr(list(by_name.values()))
    assert "env-secret" not in dumped and "file-secret" not in dumped
