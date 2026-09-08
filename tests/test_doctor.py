import io
import json

import httpx
import pytest
from rich.console import Console

from aac.doctor import run_doctor
from tests.kaggle_fake import FakeKaggle


def make_transport(*, served_nv=("m-nv", "m-nv2"), dead_models=(), metric="Roc Auc Score"):
    kaggle = FakeKaggle(metric=metric)

    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host in ("api.kaggle.com", "storage.googleapis.com"):
            return kaggle.handler(request)
        if path.endswith("/models"):
            ids = ["m-local"] if host == "local.test" else list(served_nv)
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})
        if path.endswith("/chat/completions"):
            payload = json.loads(request.content)
            model = payload["model"]
            if model in dead_models:
                return httpx.Response(404, json={"detail": "Function not found for account"})
            content = (
                f'{{"ok": true, "model": "{model}"}}' if "response_format" in payload else "ok"
            )
            return httpx.Response(
                200,
                json={
                    "model": model,
                    "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            )
        return httpx.Response(404, text=f"unexpected {request.url}")

    return httpx.MockTransport(handler)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("NV_KEY", "k")
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("KAGGLE_ACCESS_TOKEN", "KGAT_t")


def run(config_path, tmp_path, transport):
    buf = io.StringIO()
    code = run_doctor(
        config_path,
        runs_root=tmp_path / "runs",
        transport=transport,
        console=Console(file=buf, width=200),
    )
    return code, buf.getvalue()


def test_doctor_all_green(env, tmp_path, write_config, minimal_config):
    minimal_config["researchers"] = [{"backend": "nvidia", "model": "m-nv2"}, {"backend": "local"}]
    code, out = run(write_config(minimal_config), tmp_path, make_transport())
    assert code == 0, out
    assert "All checks passed" in out
    assert "backend nvidia m-nv2" in out and "backend local m-local" in out
    assert "backend nvidia json" in out and "backend local json" in out
    assert "'Roc Auc Score' -> auc" in out
    assert (tmp_path / "runs" / "ledger.db").exists()


def test_doctor_flags_unserved_model_dead_model_and_bad_metric(
    env, tmp_path, write_config, minimal_config
):
    minimal_config["researchers"] = [
        {"backend": "nvidia", "model": "ghost"},
        {"backend": "nvidia", "model": "m-nv2"},
    ]
    transport = make_transport(dead_models=("m-nv2",), metric="Quadratic Weighted Kappa")
    code, out = run(write_config(minimal_config), tmp_path, transport)
    assert code == 1
    assert "NOT served: ghost" in out
    assert "backend nvidia m-nv2" in out and "404" in out
    assert "not implemented" in out
    assert "check(s) failed" in out


def test_doctor_reports_missing_env_and_bad_config(
    tmp_path, monkeypatch, write_config, minimal_config
):
    monkeypatch.delenv("NV_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("KAGGLE_ACCESS_TOKEN", "KGAT_t")
    code, out = run(write_config(minimal_config), tmp_path, make_transport())
    assert code == 1 and "unset variables NV_KEY" in out
    code, out = run(tmp_path / "missing.yaml", tmp_path, make_transport())
    assert code == 1 and "not found" in out


def test_resources_check_warns_on_oversubscription(write_config, minimal_config, monkeypatch):
    from aac.config import load_config
    from aac.doctor import check_resources

    minimal_config["run"] = {"parallel_branches": 4, "n_jobs": 8}
    config = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    monkeypatch.setattr("aac.doctor.machine_memory_gib", lambda: 24.0)
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    check = check_resources(config)
    assert check.ok and check.warn
    assert "one researcher at a time" in check.detail and "exceed 8 cores" in check.detail
    minimal_config["run"] = {"parallel_branches": 1, "n_jobs": 4}
    config = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    assert not check_resources(config).warn
