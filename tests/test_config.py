import pytest

from aac.config import ConfigError, expand_env, load_config, resolved_yaml
from tests.conftest import REPO


def test_expand_env_forms():
    missing: list[str] = []
    env = {"A": "1"}
    assert expand_env("${A}", missing, env) == "1"
    assert expand_env("${B:-dflt}", missing, env) == "dflt"
    assert expand_env("${B:-}", missing, env) == ""
    assert expand_env("pre-${A}-post", missing, env) == "pre-1-post"
    assert missing == []
    assert expand_env("${B}", missing, env) is None
    assert expand_env("x-${C}-y", missing, env) == "x--y"
    assert missing == ["B", "C"]
    assert expand_env({"k": ["${A}", 3, None, True]}, missing, env) == {"k": ["1", 3, None, True]}


def test_strict_load_fails_on_unset_variable(write_config, minimal_config):
    with pytest.raises(ConfigError, match="NV_KEY"):
        load_config(write_config(minimal_config), env={})


def test_non_strict_records_missing(write_config, minimal_config):
    c = load_config(write_config(minimal_config), strict=False, env={})
    assert c.missing_env == ["NV_KEY"]
    assert c.backends["nvidia"].api_key is None


def test_secret_never_appears_in_dumps(write_config, minimal_config):
    c = load_config(write_config(minimal_config), env={"NV_KEY": "sekrit-value"})
    assert c.backends["nvidia"].api_key is not None
    assert c.backends["nvidia"].api_key.get_secret_value() == "sekrit-value"
    for text in (resolved_yaml(c), repr(c), str(c.redacted())):
        assert "sekrit-value" not in text


def test_fingerprint_ignores_secret_but_tracks_settings(write_config, minimal_config):
    p = write_config(minimal_config)
    a = load_config(p, env={"NV_KEY": "one"})
    b = load_config(p, env={"NV_KEY": "two"})
    assert a.fingerprint() == b.fingerprint()
    minimal_config["run"] = {"seed": 7}
    c = load_config(write_config(minimal_config, "d.yaml"), env={"NV_KEY": "one"})
    assert c.fingerprint() != a.fingerprint()


def test_base_url_normalised_and_validated(write_config, minimal_config):
    c = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    assert c.backends["nvidia"].base_url == "https://nv.test/v1"
    minimal_config["backends"]["local"]["base_url"] = "local.test/v1"
    with pytest.raises(ConfigError, match="http"):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})


def test_router_must_reference_known_backend(write_config, minimal_config):
    minimal_config["router"] = {"reason": "nope"}
    with pytest.raises(ConfigError, match="router.reason"):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})


def test_unknown_key_rejected(write_config, minimal_config):
    minimal_config["run"] = {"typo": 1}
    with pytest.raises(ConfigError, match="typo"):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})


def test_models_enabled_validation(write_config, minimal_config):
    minimal_config["models"] = {"enabled": ["lightgbm", "lightgbm"]}
    with pytest.raises(ConfigError, match="duplicates"):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    minimal_config["models"] = {"enabled": ["neuralnet"]}
    with pytest.raises(ConfigError):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_shipped_s6e9_config_loads():
    c = load_config(REPO / "configs" / "s6e9.yaml", strict=False, env={})
    assert c.competition.slug == "playground-series-s6e9"
    assert c.missing_env == ["NVIDIA_API_KEY"]
    assert set(c.backends) == {"local", "nvidia"}
    researchers = c.researcher_specs()
    assert len(researchers) >= 3
    assert len({r.track for r in researchers}) >= 3
    # 2026-09-08: the local seat was retired (1 working round in 20); the second backend
    # stays configured and verified, and hub failures fall over to another hub model.
    nvidia = c.backends["nvidia"]
    assert nvidia.fallback_backend == "nvidia" and nvidia.fallback_model
    assert nvidia.trip_after == 2 and c.sandbox.experiment_timeout_seconds == 900
    assert c.models.variants == ["digits", "encoded"] and c.models.seed_bag == 2
    assert c.models.variant_families is None and c.models.variant_max_seconds == 120
    assert "lightgbm_focal" in c.models.enabled
    assert c.scholar is not None and c.scholar.reuse_runs == 3
    assert c.run.schedule == "round_robin"
    assert c.router.reason == "nvidia"
    assert c.competition.metric is None, "metric must come from the Kaggle API by default"


def test_files_override_and_new_run_keys(write_config, minimal_config):
    minimal_config["competition"]["files"] = {
        "train": "data/tr*.csv",
        "sample_submission": "sub.parquet",
    }
    minimal_config["run"] = {"n_jobs": 3, "max_classes": 10}
    c = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    assert c.competition.files.train == "data/tr*.csv" and c.competition.files.test is None
    assert c.run.n_jobs == 3 and c.run.max_classes == 10
    minimal_config["run"] = {"n_jobs": -1}
    with pytest.raises(ConfigError):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})


def test_researchers_config(write_config, minimal_config):
    minimal_config["researchers"] = [
        {"backend": "nvidia", "track": "gbdt", "temperature": 0.3},
        {"backend": "local", "model": "other", "track": "neural", "rounds": 2},
    ]
    c = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    specs = c.researcher_specs()
    assert [(s.backend, s.model, s.track, s.rounds) for s in specs] == [
        ("nvidia", "m-nv", "gbdt", None),
        ("local", "other", "neural", 2),
    ]
    minimal_config["researchers"] = [{"backend": "ghost"}]
    with pytest.raises(ConfigError, match="researchers\\[0\\]"):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})


def test_fallback_pair_validation(write_config, minimal_config):
    minimal_config["backends"]["nvidia"]["fallback_backend"] = "ghost"
    with pytest.raises(ConfigError, match="fallback_backend"):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    minimal_config["backends"]["nvidia"].pop("fallback_backend")
    minimal_config["backends"]["nvidia"]["fallback_model"] = "m-nv2"
    with pytest.raises(ConfigError, match="needs fallback_backend"):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    minimal_config["backends"]["nvidia"]["fallback_backend"] = "nvidia"
    c = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    assert c.backends["nvidia"].fallback_model == "m-nv2"
    assert c.backends["nvidia"].trip_after == 2 and c.backends["nvidia"].cooldown_seconds == 600
    minimal_config["models"] = {"variants": ["encoded", "encoded"]}
    with pytest.raises(ConfigError, match="duplicates"):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})


def test_extra_train_config(write_config, minimal_config):
    minimal_config["competition"]["extra_train"] = [
        {"dataset": "owner/ev-data", "rename": {"Buyer_ID": "id"}, "dedupe": False}
    ]
    minimal_config["competition"]["extra_flag"] = None
    c = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    [spec] = c.competition.extra_train
    assert spec.dataset == "owner/ev-data" and spec.file is None and not spec.dedupe
    assert spec.rename == {"Buyer_ID": "id"} and c.competition.extra_flag is None
    minimal_config["competition"]["extra_train"] = [{"dataset": "no-slash"}]
    with pytest.raises(ConfigError, match="extra_train"):
        load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    c = load_config(REPO / "configs" / "s6e9.yaml", strict=False, env={})
    assert [e.dataset for e in c.competition.extra_train] == [
        "itzzomkar/ev-adoption-behavior-and-range-anxiety"
    ]
    assert c.competition.extra_flag == "is_original"


def test_owner_notes_and_digits_variant_config(write_config, minimal_config):
    minimal_config["competition"]["notes"] = ["income digits leak the target"]
    minimal_config["models"] = {"variants": ["digits"]}
    c = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    assert c.competition.notes == ["income digits leak the target"]
    assert c.models.variants == ["digits"]
    shipped = load_config(REPO / "configs" / "s6e9.yaml", strict=False, env={})
    assert shipped.models.variants == ["digits", "encoded"]
    assert any("Annual_Income_USD" in n for n in shipped.competition.notes)
