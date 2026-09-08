"""Raw httpx client for OpenAI-compatible ``/v1/chat/completions`` (README section 5.1).

``complete`` retries on 429, 5xx, and transport errors with exponential backoff and jitter,
capped at four attempts. Logging to the ledger and budget accounting live one level up, in
``aac.llm.router``, so this module stays a pure HTTP function.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from aac.config import BackendConfig

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 4
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
BACKOFF_BASE = 1.0
BACKOFF_CAP = 30.0


class LLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        kind: str = "http",  # "http", "transport", or "malformed"
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.kind = kind
        self.attempts = 1

    @property
    def retryable(self) -> bool:
        return self.kind == "transport" or (self.kind == "http" and self.status in RETRY_STATUSES)


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    cost_per_1k: float = 0.0
    timeout: float = 120.0
    supports_json_mode: bool = True

    @classmethod
    def from_config(cls, name: str, cfg: BackendConfig) -> Backend:
        return cls(
            name=name,
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key.get_secret_value() if cfg.api_key else None,
            cost_per_1k=cfg.cost_per_1k,
            timeout=cfg.timeout,
            supports_json_mode=cfg.supports_json_mode,
        )

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def __repr__(self) -> str:  # never leak the key through logging
        return f"Backend(name={self.name!r}, base_url={self.base_url!r}, model={self.model!r})"


@dataclass
class Completion:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str | None = None
    reasoning: str = ""  # reasoning_content from thinking models; never parsed for output
    attempts: int = 1
    backend: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _http(client: httpx.Client | None, timeout: float) -> tuple[httpx.Client, bool]:
    if client is not None:
        return client, False
    return httpx.Client(timeout=timeout), True


def _raise_for_status(resp: httpx.Response, what: str) -> None:
    if resp.status_code >= 400:
        body = resp.text[:500]
        raise LLMError(
            f"{what}: HTTP {resp.status_code}: {body}", status=resp.status_code, body=body
        )


def list_models(
    backend: Backend, *, client: httpx.Client | None = None, timeout: float = 20.0
) -> list[str]:
    """Model identifiers the backend serves, via ``GET {base_url}/models``."""
    http, owned = _http(client, timeout)
    try:
        try:
            resp = http.get(
                f"{backend.base_url}/models", headers=backend.headers(), timeout=timeout
            )
        except httpx.HTTPError as exc:
            raise LLMError(
                f"{backend.name}: {exc.__class__.__name__}: {exc}", kind="transport"
            ) from exc
        _raise_for_status(resp, f"{backend.name} GET /models")
        data = resp.json()
        return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
    finally:
        if owned:
            http.close()


def complete_once(
    messages: list[dict[str, str]],
    *,
    backend: Backend,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float | None = None,
    client: httpx.Client | None = None,
) -> Completion:
    """One chat completion, single attempt. Raises LLMError on transport or HTTP failure."""
    timeout = backend.timeout if timeout is None else timeout
    payload: dict[str, Any] = {
        "model": model or backend.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode and backend.supports_json_mode:
        payload["response_format"] = {"type": "json_object"}
    http, owned = _http(client, timeout)
    started = time.monotonic()
    try:
        try:
            resp = http.post(
                f"{backend.base_url}/chat/completions",
                headers=backend.headers(),
                json=payload,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(
                f"{backend.name}: {exc.__class__.__name__}: {exc}", kind="transport"
            ) from exc
        latency = time.monotonic() - started
        _raise_for_status(resp, f"{backend.name} POST /chat/completions")
        try:
            data = resp.json()
            choice = data["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"{backend.name}: malformed completion response: {resp.text[:300]}",
                kind="malformed",
            ) from exc
        usage = data.get("usage") or {}
        message = choice.get("message") or {}
        return Completion(
            content=message.get("content") or "",
            reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
            model=str(data.get("model") or payload["model"]),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=latency,
            finish_reason=choice.get("finish_reason"),
            backend=backend.name,
            raw=data,
        )
    finally:
        if owned:
            http.close()


def complete(
    messages: list[dict[str, str]],
    *,
    backend: Backend,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float | None = None,
    client: httpx.Client | None = None,
    attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> Completion:
    """A chat completion with retries on 429 / 5xx / transport errors (README section 5.1).

    Exponential backoff from one second, doubled per attempt, capped, with up to 50% jitter.
    Non-retryable failures (4xx other than 429, malformed bodies) raise immediately.
    """
    delay = BACKOFF_BASE
    for attempt in range(1, attempts + 1):
        try:
            result = complete_once(
                messages,
                backend=backend,
                model=model,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                client=client,
            )
        except LLMError as exc:
            exc.attempts = attempt
            if not exc.retryable or attempt == attempts:
                raise
            wait = delay + rng() * delay * 0.5
            log.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                backend.name,
                attempt,
                attempts,
                str(exc)[:120],
                wait,
            )
            sleep(wait)
            delay = min(delay * 2, BACKOFF_CAP)
            continue
        result.attempts = attempt
        return result
    raise AssertionError("unreachable")
