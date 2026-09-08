import json

import httpx
import pytest
from pydantic import SecretStr

from aac.config import BackendConfig
from aac.llm.client import Backend, Completion, LLMError, complete, complete_once, list_models
from tests.llm_fake import FakeLLM

NV = Backend(name="nv", base_url="https://nv.test/v1", model="m", api_key="k-secret")
LOCAL = Backend(name="local", base_url="http://local.test/v1", model="q", supports_json_mode=False)


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_backend_from_config_and_repr():
    cfg = BackendConfig(base_url="https://x/v1/", model="m", api_key=SecretStr("shh"), timeout=9)
    b = Backend.from_config("nv", cfg)
    assert b.api_key == "shh" and b.timeout == 9 and b.base_url == "https://x/v1"
    assert "shh" not in repr(b)
    assert b.headers()["Authorization"] == "Bearer shh"
    assert "Authorization" not in LOCAL.headers()


def test_list_models_parses_ids():
    def handler(request):
        assert request.url == "https://nv.test/v1/models"
        assert request.headers["authorization"] == "Bearer k-secret"
        return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}, {"bad": 1}]})

    assert list_models(NV, client=make_client(handler)) == ["a", "b"]


def test_complete_payload_and_parsing():
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "model": "m-served",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"x": 1}',
                            "reasoning_content": "thinking",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    c = complete(
        [{"role": "user", "content": "hi"}],
        backend=NV,
        json_mode=True,
        temperature=0.7,
        max_tokens=99,
        client=make_client(handler),
    )
    assert isinstance(c, Completion)
    assert c.content == '{"x": 1}' and c.reasoning == "thinking" and c.model == "m-served"
    assert (c.prompt_tokens, c.completion_tokens, c.total_tokens) == (12, 3, 15)
    assert c.finish_reason == "stop" and c.latency_s >= 0
    p = seen["payload"]
    assert p["model"] == "m" and p["temperature"] == 0.7 and p["max_tokens"] == 99
    assert p["response_format"] == {"type": "json_object"} and p["stream"] is False
    assert seen["auth"] == "Bearer k-secret"


def test_json_mode_omitted_when_unsupported_and_model_override():
    def handler(request):
        p = json.loads(request.content)
        assert "response_format" not in p and p["model"] == "other"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    c = complete([], backend=LOCAL, model="other", json_mode=True, client=make_client(handler))
    assert c.content == "" and c.prompt_tokens == 0 and c.model == "other"


def test_http_error_raises_llm_error_with_status():
    def handler(request):
        return httpx.Response(503, text="worker limit reached")

    with pytest.raises(LLMError, match="503") as exc:
        complete([], backend=NV, client=make_client(handler))
    assert exc.value.status == 503 and "worker" in exc.value.body


def test_transport_error_raises_llm_error():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMError, match="ReadTimeout"):
        complete([], backend=NV, client=make_client(handler))
    with pytest.raises(LLMError):
        list_models(NV, client=make_client(handler))


def test_malformed_body_raises():
    def handler(request):
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(LLMError, match="malformed"):
        complete([], backend=NV, client=make_client(handler))


def retrying(fake, backend=NV, **kw):
    sleeps: list[float] = []
    kw.setdefault("rng", lambda: 0.0)
    result = complete(
        [{"role": "user", "content": "x"}],
        backend=backend,
        client=httpx.Client(transport=fake.transport),
        sleep=sleeps.append,
        **kw,
    )
    return result, sleeps


def test_retries_on_503_and_429_then_succeeds():
    fake = FakeLLM({"nv.test": [503, 429, "done"]})
    result, sleeps = retrying(fake)
    assert result.content == "done" and result.attempts == 3 and result.backend == "nv"
    assert sleeps == [1.0, 2.0], "exponential backoff from one second"
    assert len(fake.calls("nv.test")) == 3


def test_retries_on_transport_error():
    fake = FakeLLM({"nv.test": [httpx.ReadTimeout("slow"), httpx.ConnectError("down"), "ok"]})
    result, sleeps = retrying(fake)
    assert result.attempts == 3 and len(sleeps) == 2


def test_no_retry_on_client_errors_or_malformed():
    fake = FakeLLM({"nv.test": [404, "never"]})
    with pytest.raises(LLMError, match="404") as exc:
        retrying(fake)
    assert exc.value.attempts == 1 and not exc.value.retryable

    def malformed(request):
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(LLMError, match="malformed") as exc:
        complete(
            [],
            backend=NV,
            client=httpx.Client(transport=httpx.MockTransport(malformed)),
            sleep=lambda s: None,
        )
    assert exc.value.kind == "malformed" and not exc.value.retryable


def test_gives_up_after_four_attempts():
    fake = FakeLLM({"nv.test": [503, 503, 503, 503, "too late"]})
    with pytest.raises(LLMError, match="503") as exc:
        retrying(fake)
    assert exc.value.attempts == 4 and len(fake.calls("nv.test")) == 4


def test_jitter_and_cap():
    fake = FakeLLM({"nv.test": [503] * 5 + ["ok"]})
    result, sleeps = retrying(fake, attempts=6, rng=lambda: 1.0)
    assert result.attempts == 6
    assert sleeps == [1.5, 3.0, 6.0, 12.0, 24.0]


def test_complete_once_is_single_attempt():
    fake = FakeLLM({"nv.test": [503, "ok"]})
    with pytest.raises(LLMError):
        complete_once([], backend=NV, client=httpx.Client(transport=fake.transport))
