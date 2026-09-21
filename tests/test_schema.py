import json

import httpx
import pytest
from pydantic import BaseModel

from aac.config import load_config
from aac.context import Budget, BudgetExceeded
from aac.ledger import Ledger
from aac.llm.router import Router
from aac.llm.schema import describe_schema, extract_json, parse_as, structured_completion
from tests.llm_fake import FakeLLM


class Verdict(BaseModel):
    decision: str
    score: float


GOOD = '{"decision": "promote", "score": 0.91}'


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (GOOD, {"decision": "promote", "score": 0.91}),
        (
            f"Sure, here you go:\n```json\n{GOOD}\n```\nHope this helps.",
            {"decision": "promote", "score": 0.91},
        ),
        (f"```\n{GOOD}\n```", {"decision": "promote", "score": 0.91}),
        (f"The answer is {GOOD} and that is final.", {"decision": "promote", "score": 0.91}),
        ('{"a": {"b": "}"}, "c": "\\"}"}', {"a": {"b": "}"}, "c": '"}'}),
        ('[1, 2, {"x": 3}]', [1, 2, {"x": 3}]),
        ("no json here", None),
        ("{unbalanced", None),
        ("", None),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_parse_as_reports_problems():
    value, err = parse_as(Verdict, GOOD)
    assert value == Verdict(decision="promote", score=0.91) and err is None
    value, err = parse_as(Verdict, '{"decision": "promote"}')
    assert value is None and "score" in err and "validation" in err
    value, err = parse_as(Verdict, "nothing")
    assert value is None and "no JSON" in err


def test_describe_schema_has_schema_and_example():
    text = describe_schema(Verdict, Verdict(decision="refine", score=0.5))
    assert '"properties"' in text and '"decision"' in text and '"refine"' in text
    assert "Example of a valid reply" in text


@pytest.fixture
def make_router(tmp_path, write_config, minimal_config):
    def _make(fake, budget=None):
        config = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
        ledger = Ledger(tmp_path / "ledger.db")
        ledger.create_run("r1", "s", "h", None)
        return Router(
            config,
            ledger=ledger,
            budget=budget,
            run_id="r1",
            client=httpx.Client(transport=fake.transport),
            sleep=lambda s: None,
        )

    return _make


MSGS = [{"role": "user", "content": "judge"}]


def test_first_try_success(make_router):
    fake = FakeLLM({"nv.test": [GOOD]})
    router = make_router(fake)
    result = structured_completion(router, MSGS, Verdict, agent="critic", tier="reason")
    assert result.ok and result.value.decision == "promote"
    assert not result.repaired and not result.escalated and result.backend == "nvidia"
    [payload] = fake.calls("nv.test")
    assert payload["response_format"] == {"type": "json_object"}
    [row] = router.ledger.list_llm_calls("r1")
    assert row["agent"] == "critic" and row["ok"] == 1 and row["escalated"] == 0


def test_repair_turn_at_temperature_zero(make_router):
    fake = FakeLLM({"nv.test": ["I think promote, score high", GOOD]})
    router = make_router(fake)
    result = structured_completion(
        router, MSGS, Verdict, agent="critic", tier="reason", temperature=0.7
    )
    assert result.ok and result.repaired and not result.escalated
    first, repair = fake.calls("nv.test")
    assert first["temperature"] == 0.7 and repair["temperature"] == 0.0
    assert repair["messages"][-2] == {"role": "assistant", "content": "I think promote, score high"}
    assert "could not be used" in repair["messages"][-1]["content"]
    assert "no JSON" in repair["messages"][-1]["content"]
    assert '"decision"' in repair["messages"][-1]["content"], "schema is in the repair prompt"
    agents = [r["agent"] for r in router.ledger.list_llm_calls("r1")]
    assert agents == ["critic", "critic:repair"]


def test_two_bad_replies_fall_over_to_the_other_backend(make_router):
    fake = FakeLLM({"nv.test": ["bad", '{"decision": 1}'], "local.test": [GOOD]})
    router = make_router(fake)
    result = structured_completion(router, MSGS, Verdict, agent="critic", tier="reason")
    assert result.ok and result.escalated and not result.repaired
    assert result.backend == "local" and result.model == "m-local"
    rows = router.ledger.list_llm_calls("r1")
    assert [(r["backend"], r["escalated"]) for r in rows] == [
        ("nvidia", 0),
        ("nvidia", 0),
        ("local", 1),
    ]


def test_backend_error_falls_over_and_total_failure_returns_none(make_router):
    fake = FakeLLM({"nv.test": [500, 500, 500, 500], "local.test": ["nope", "still nope"]})
    router = make_router(fake)
    result = structured_completion(router, MSGS, Verdict, agent="critic", tier="reason")
    assert not result.ok and result.value is None
    assert "after repair" in result.error
    rows = router.ledger.list_llm_calls("r1")
    assert rows[0]["ok"] == 0 and rows[0]["attempts"] == 4 and rows[0]["backend"] == "nvidia"
    assert [r["backend"] for r in rows[1:]] == ["local", "local"]


def test_no_failover_when_disabled(make_router):
    fake = FakeLLM({"nv.test": ["bad", "bad"], "local.test": [GOOD]})
    router = make_router(fake)
    result = structured_completion(
        router, MSGS, Verdict, agent="c", tier="reason", allow_failover=False
    )
    assert not result.ok and not fake.calls("local.test")


def test_budget_is_charged_and_enforced(make_router):
    fake = FakeLLM({"nv.test": [GOOD, GOOD]})
    budget = Budget(max_tokens=20, max_seconds=1000)
    router = make_router(fake, budget=budget)
    assert structured_completion(router, MSGS, Verdict, agent="c", tier="reason").ok
    assert budget.tokens_used == 15
    result = structured_completion(router, MSGS, Verdict, agent="c", tier="reason")
    assert result.ok and budget.tokens_used == 30
    with pytest.raises(BudgetExceeded):
        structured_completion(router, MSGS, Verdict, agent="c", tier="reason")


def test_explicit_backend_and_model(make_router):
    fake = FakeLLM({"nv.test": [GOOD]})
    router = make_router(fake)
    result = structured_completion(
        router, MSGS, Verdict, agent="a", backend="nvidia", model="m-nv2"
    )
    assert result.ok and result.model == "m-nv2"
    assert fake.calls("nv.test")[0]["model"] == "m-nv2"
    assert json.loads(json.dumps(result.value.model_dump()))["score"] == 0.91
