"""Structured output without an SDK (README section 5.4).

1. The JSON schema and one filled example go into the prompt (``describe_schema``).
2. ``response_format: json_object`` is sent when the backend supports it.
3. Extraction tries the whole body, then the first fenced block, then the first balanced
   ``{...}`` span.
4. Validation is a pydantic model.
5. On failure, one repair turn at temperature 0 carrying the raw output and the error.
6. Two unusable replies from one backend fall through to the other backend once.
7. Still failing: ``value`` is ``None`` and the caller decides. Nothing is guessed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from aac.llm.client import Completion, LLMError
from aac.llm.prompts import load_prompt
from aac.llm.router import Router

log = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.S)
_PY_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def _balanced_span(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find(open_ch, start + 1)
    return None


def extract_json(text: str) -> Any | None:
    """Whole body, then first fenced block, then first balanced object or array."""
    if not text:
        return None
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        span = _balanced_span(text, open_ch, close_ch)
        if span:
            candidates.append(span)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None


def parse_as[T: BaseModel](
    model: type[T], text: str, context: dict[str, Any] | None = None
) -> tuple[T | None, str | None]:
    """Extract and validate. Returns (value, None) or (None, reason)."""
    data = extract_json(text)
    if data is None:
        return None, "no JSON object could be extracted from the reply"
    try:
        return model.model_validate(data, context=context), None
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()[:8]
        )
        return None, f"schema validation failed: {problems}"


def describe_schema(model: type[BaseModel], example: BaseModel | dict[str, Any]) -> str:
    """Schema plus one filled example, for the prompt."""
    schema = json.dumps(model.model_json_schema(), indent=2)
    sample = example.model_dump(mode="json") if isinstance(example, BaseModel) else example
    return f"JSON schema:\n{schema}\n\nExample of a valid reply:\n{json.dumps(sample, indent=2)}"


@dataclass
class StructuredResult[T: BaseModel]:
    value: T | None
    raw: str = ""
    error: str | None = None
    backend: str = ""
    model: str = ""
    repaired: bool = False
    escalated: bool = False
    completions: list[Completion] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None


def structured_completion[T: BaseModel](
    router: Router,
    messages: list[dict[str, str]],
    schema: type[T],
    *,
    agent: str,
    tier: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    branch_id: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    allow_failover: bool = True,
    context: dict[str, Any] | None = None,
) -> StructuredResult[T]:
    primary = router.backend(backend) if backend else router.for_tier(tier)
    chain: list[tuple[str, str | None]] = [(primary.name, model)]
    fallback = router.fallback(primary.name) if allow_failover else None
    if fallback is not None:
        chain.append((fallback.name, None))

    result: StructuredResult[T] = StructuredResult(None)
    for hop, (backend_name, model_name) in enumerate(chain):
        escalated = hop > 0
        if escalated:
            log.warning("%s: escalating from %s to %s", agent, chain[0][0], backend_name)
        try:
            first = router.complete(
                messages,
                agent=agent,
                tier=tier,
                backend=backend_name,
                model=model_name,
                branch_id=branch_id,
                json_mode=True,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_failover=False,
                escalated=escalated,
            )
        except LLMError as exc:
            result.error = str(exc)
            continue
        result.completions.append(first)
        result.raw, result.backend, result.model = first.content, first.backend, first.model
        value, error = parse_as(schema, first.content, context)
        if value is not None:
            result.value, result.error, result.escalated = value, None, escalated
            return result

        repair = load_prompt("repair").messages(
            error=error, schema=json.dumps(schema.model_json_schema(), indent=2)
        )
        repair_messages = [*messages, {"role": "assistant", "content": first.content}, *repair]
        try:
            second = router.complete(
                repair_messages,
                agent=f"{agent}:repair",
                tier=tier,
                backend=backend_name,
                model=model_name,
                branch_id=branch_id,
                json_mode=True,
                temperature=0.0,
                max_tokens=max_tokens,
                allow_failover=False,
                escalated=escalated,
            )
        except LLMError as exc:
            result.error = f"{error}; repair call failed: {exc}"
            continue
        result.completions.append(second)
        result.raw = second.content
        value, error2 = parse_as(schema, second.content, context)
        if value is not None:
            result.value, result.error, result.repaired, result.escalated = (
                value,
                None,
                True,
                escalated,
            )
            return result
        result.error = f"{error}; after repair: {error2}"
        log.warning("%s on %s: unusable JSON twice (%s)", agent, backend_name, result.error[:200])
    return result


def extract_code(text: str, required: str = "build_features") -> str | None:
    """A fenced python block defining ``required``, else JSON {"code": ...}, else a bare module."""
    if not text:
        return None
    blocks = [b.strip() for b in _PY_FENCE.findall(text) if f"def {required}" in b]
    if blocks:
        return max(blocks, key=len)
    data = extract_json(text)
    code = data.get("code") if isinstance(data, dict) else None
    if isinstance(code, str) and f"def {required}" in code:
        return code.strip()
    if f"def {required}" in text and "```" not in text:
        return text.strip()
    return None
