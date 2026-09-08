"""Scripted OpenAI-compatible server for httpx.MockTransport.

``scripts`` maps a host to the replies it gives, in order: a string is the assistant content,
an int is an HTTP status to return, an Exception is raised as a transport error. When a
script runs dry the host answers ``"ok"``.
"""

from __future__ import annotations

import json

import httpx

DEFAULT_MODELS = {"local.test": ["m-local"], "nv.test": ["m-nv", "m-nv2"]}


class FakeLLM:
    def __init__(
        self,
        scripts: dict[str, list[object]] | None = None,
        models: dict[str, list[str]] | None = None,
    ) -> None:
        self.scripts = {host: list(items) for host, items in (scripts or {}).items()}
        self.models = models or DEFAULT_MODELS
        self.requests: list[tuple[str, dict]] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def calls(self, host: str) -> list[dict]:
        return [payload for h, payload in self.requests if h == host]

    def handler(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": m} for m in self.models.get(host, [])]}
            )
        if not path.endswith("/chat/completions"):
            return httpx.Response(404, text=f"unexpected {request.url}")
        payload = json.loads(request.content)
        self.requests.append((host, payload))
        script = self.scripts.get(host, [])
        item: object = script.pop(0) if script else "ok"
        if isinstance(item, Exception):
            raise item
        if isinstance(item, int):
            return httpx.Response(item, text=f"scripted {item}")
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "choices": [{"message": {"content": item}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )


def combined_transport(kaggle, llm: FakeLLM) -> httpx.MockTransport:
    """Route Kaggle and storage hosts to a FakeKaggle, everything else to a FakeLLM."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host in ("api.kaggle.com", "storage.googleapis.com"):
            return kaggle.handler(request)
        return llm.handler(request)

    return httpx.MockTransport(handler)
