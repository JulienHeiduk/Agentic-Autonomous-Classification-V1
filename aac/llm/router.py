"""Task-tier routing, failover, concurrency, and call accounting (README section 5.3).

Route by tier, not by agent name: ``cheap`` and ``code`` default to the local server,
``reason`` to the NVIDIA hub. A backend that fails after its retries falls through to the
other backend once, and the escalation is recorded. Every call, successful or not, is a row
in ``llm_calls`` and is charged to the run's token budget.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from aac.config import TIERS, Config
from aac.context import Budget
from aac.ledger import Ledger
from aac.llm.client import Backend, Completion, LLMError, complete, list_models

log = logging.getLogger(__name__)


class Router:
    def __init__(
        self,
        config: Config,
        *,
        ledger: Ledger | None = None,
        budget: Budget | None = None,
        run_id: str | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.backends: dict[str, Backend] = {
            name: Backend.from_config(name, cfg) for name, cfg in config.backends.items()
        }
        self.tiers: dict[str, str] = {tier: getattr(config.router, tier) for tier in TIERS}
        self._semaphores = {
            name: threading.BoundedSemaphore(cfg.max_concurrency)
            for name, cfg in config.backends.items()
        }
        self.ledger = ledger
        self.budget = budget
        self.run_id = run_id
        self._client = client or httpx.Client()
        self._sleep = sleep
        self._clock = clock
        # Circuit breaker state per backend: consecutive failed calls, and until when it is
        # skipped in favour of its fallback.
        self._failures: dict[str, int] = dict.fromkeys(self.backends, 0)
        self._down_until: dict[str, float] = dict.fromkeys(self.backends, 0.0)
        self._breaker_lock = threading.Lock()

    # -- selection --------------------------------------------------------------------------

    def backend(self, name: str) -> Backend:
        try:
            return self.backends[name]
        except KeyError:
            raise LLMError(
                f"unknown backend {name!r}; configured: {sorted(self.backends)}"
            ) from None

    def for_tier(self, tier: str | None) -> Backend:
        if tier is None:
            raise LLMError("either a tier or a backend must be given")
        if tier not in self.tiers:
            raise LLMError(f"unknown tier {tier!r}; known: {sorted(self.tiers)}")
        return self.backend(self.tiers[tier])

    def fallback(self, name: str) -> Backend | None:
        """The other backend, in config order. None when only one backend is configured."""
        for other in self.backends:
            if other != name:
                return self.backends[other]
        return None

    def fallback_target(self, backend: Backend, model: str) -> tuple[Backend, str] | None:
        """Where a failed call goes: the configured fallback backend and model, else the other
        backend with its default model. None when that would be the same pair again."""
        cfg = self.config.backends[backend.name]
        if cfg.fallback_backend is None:
            other = self.fallback(backend.name)
            return (other, other.model) if other is not None else None
        target = self.backend(cfg.fallback_backend)
        target_model = cfg.fallback_model or target.model
        if target.name == backend.name and target_model == model:
            return None
        return target, target_model

    def is_down(self, name: str) -> bool:
        with self._breaker_lock:
            return self._clock() < self._down_until[name]

    def _note_failure(self, backend: Backend, exc: LLMError) -> None:
        if not exc.retryable:  # a 404 or a malformed body is not overload
            return
        cfg = self.config.backends[backend.name]
        with self._breaker_lock:
            self._failures[backend.name] += 1
            if self._failures[backend.name] < cfg.trip_after or cfg.cooldown_seconds <= 0:
                return
            self._failures[backend.name] = 0
            self._down_until[backend.name] = self._clock() + cfg.cooldown_seconds
        log.warning(
            "%s: %d consecutive failed calls; skipping it for %.0fs",
            backend.name,
            cfg.trip_after,
            cfg.cooldown_seconds,
        )

    def _note_success(self, name: str) -> None:
        with self._breaker_lock:
            self._failures[name] = 0

    def configured_models(self) -> dict[str, list[str]]:
        wanted: dict[str, list[str]] = {name: [b.model] for name, b in self.backends.items()}
        scholar = self.config.scholar_spec()
        specs = list(self.config.researcher_specs())
        if scholar is not None:
            specs.append(scholar)
        for spec in specs:
            if spec.model and spec.model not in wanted[spec.backend]:
                wanted[spec.backend].append(spec.model)
        for cfg in self.config.backends.values():
            fb, fm = cfg.fallback_backend, cfg.fallback_model
            if fb and fm and fm not in wanted[fb]:
                wanted[fb].append(fm)
        return wanted

    def verify_models(self) -> dict[str, list[str]]:
        """Configured models missing from each backend's ``GET /models``. Empty means all good."""
        missing: dict[str, list[str]] = {}
        for name, models in self.configured_models().items():
            served = set(list_models(self.backends[name], client=self._client))
            absent = [m for m in models if m not in served]
            if absent:
                missing[name] = absent
        return missing

    # -- calls ------------------------------------------------------------------------------

    def _record(
        self,
        *,
        agent: str,
        tier: str | None,
        backend: Backend,
        model: str,
        branch_id: str | None,
        ok: bool,
        completion: Completion | None = None,
        error: str | None = None,
        attempts: int = 1,
        escalated: bool = False,
    ) -> None:
        prompt_tokens = completion.prompt_tokens if completion else 0
        completion_tokens = completion.completion_tokens if completion else 0
        cost = (prompt_tokens + completion_tokens) / 1000 * backend.cost_per_1k
        if self.budget is not None and completion is not None:
            self.budget.charge(prompt_tokens, completion_tokens)
        if self.ledger is not None:
            self.ledger.record_llm_call(
                run_id=self.run_id,
                branch_id=branch_id,
                agent=agent,
                tier=tier,
                backend=backend.name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=completion.latency_s if completion else None,
                cost=cost,
                ok=ok,
                error=None if ok else (error or "")[:500],
                attempts=attempts,
                escalated=escalated,
            )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        agent: str,
        tier: str | None = None,
        backend: str | None = None,
        model: str | None = None,
        branch_id: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        allow_failover: bool = True,
        escalated: bool = False,
        **client_kwargs: Any,
    ) -> Completion:
        """Complete on the tier's backend (or an explicit one); fall over once on failure."""
        primary = self.backend(backend) if backend else self.for_tier(tier)
        primary_model = model or primary.model
        chain: list[tuple[Backend, str, bool]] = [(primary, primary_model, escalated)]
        fallback = self.fallback_target(primary, primary_model) if allow_failover else None
        if fallback is not None:
            chain.append((fallback[0], fallback[1], True))

        last_error: LLMError | None = None
        for hop, (target, resolved_model, hop_escalated) in enumerate(chain):
            if self.budget is not None:
                self.budget.check()
            if hop < len(chain) - 1 and self.is_down(target.name):
                last_error = LLMError(
                    f"{target.name} is cooling down after repeated failures", kind="transport"
                )
                log.warning("%s: %s; trying the fallback first", agent, last_error)
                continue
            if hop > 0:
                log.warning(
                    "%s: %s failed (%s); falling over to %s/%s",
                    agent,
                    primary.name,
                    str(last_error)[:120],
                    target.name,
                    resolved_model,
                )
            with self._semaphores[target.name]:
                try:
                    result = complete(
                        messages,
                        backend=target,
                        model=resolved_model,
                        json_mode=json_mode,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        client=self._client,
                        sleep=self._sleep,
                        **client_kwargs,
                    )
                except LLMError as exc:
                    last_error = exc
                    self._note_failure(target, exc)
                    self._record(
                        agent=agent,
                        tier=tier,
                        backend=target,
                        model=resolved_model,
                        branch_id=branch_id,
                        ok=False,
                        error=str(exc),
                        attempts=exc.attempts,
                        escalated=hop_escalated,
                    )
                    continue
            self._note_success(target.name)
            self._record(
                agent=agent,
                tier=tier,
                backend=target,
                model=resolved_model,
                branch_id=branch_id,
                ok=True,
                completion=result,
                attempts=result.attempts,
                escalated=hop_escalated,
            )
            return result
        assert last_error is not None
        raise last_error
