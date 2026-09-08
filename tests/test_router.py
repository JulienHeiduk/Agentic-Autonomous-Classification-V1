import threading

import httpx
import pytest

from aac.config import load_config
from aac.context import Budget, BudgetExceeded
from aac.ledger import Ledger
from aac.llm.client import LLMError
from aac.llm.router import Router
from tests.llm_fake import FakeLLM

MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture
def config(write_config, minimal_config):
    minimal_config["researchers"] = [{"backend": "nvidia", "model": "m-nv2"}, {"backend": "local"}]
    return load_config(write_config(minimal_config), env={"NV_KEY": "k"})


def make(config, fake, tmp_path, budget=None):
    ledger = Ledger(tmp_path / "ledger.db")
    if ledger.get_run("r1") is None:
        ledger.create_run("r1", "s", "h", None)
    return Router(
        config,
        ledger=ledger,
        budget=budget,
        run_id="r1",
        client=httpx.Client(transport=fake.transport),
        sleep=lambda s: None,
    )


def test_tiers_and_fallbacks(config, tmp_path):
    router = make(config, FakeLLM(), tmp_path)
    assert router.for_tier("cheap").name == "local"
    assert router.for_tier("code").name == "local"
    assert router.for_tier("reason").name == "nvidia"
    assert router.fallback("local").name == "nvidia" and router.fallback("nvidia").name == "local"
    with pytest.raises(LLMError, match="unknown tier"):
        router.for_tier("premium")
    with pytest.raises(LLMError, match="unknown backend"):
        router.backend("ghost")
    with pytest.raises(LLMError, match="either a tier"):
        router.complete(MSGS, agent="x")


def test_verify_models(config, tmp_path):
    router = make(config, FakeLLM(), tmp_path)
    assert router.configured_models() == {"local": ["m-local"], "nvidia": ["m-nv", "m-nv2"]}
    assert router.verify_models() == {}
    fake = FakeLLM(models={"local.test": ["m-local"], "nv.test": ["m-nv"]})
    assert make(config, fake, tmp_path).verify_models() == {"nvidia": ["m-nv2"]}


def test_complete_routes_records_and_charges(config, tmp_path):
    fake = FakeLLM({"local.test": ["cheap reply"]})
    budget = Budget(max_tokens=1000, max_seconds=100)
    router = make(config, fake, tmp_path, budget)
    c = router.complete(MSGS, agent="namer", tier="cheap", temperature=0.9, max_tokens=33)
    assert c.content == "cheap reply" and c.backend == "local" and c.model == "m-local"
    [payload] = fake.calls("local.test")
    assert payload["temperature"] == 0.9 and payload["max_tokens"] == 33
    assert budget.tokens_used == 15
    [row] = router.ledger.list_llm_calls("r1")
    assert row["tier"] == "cheap" and row["backend"] == "local" and row["prompt_tokens"] == 10
    assert row["cost"] == 0.0 and row["ok"] == 1 and row["attempts"] == 1


def test_failover_after_retries_is_recorded(config, tmp_path):
    fake = FakeLLM({"nv.test": [503, 503, 503, 503], "local.test": ["fallback reply"]})
    router = make(config, fake, tmp_path)
    c = router.complete(MSGS, agent="architect", tier="reason", branch_id="b1")
    assert c.content == "fallback reply" and c.backend == "local"
    rows = router.ledger.list_llm_calls("r1")
    assert [(r["backend"], r["ok"], r["attempts"], r["escalated"]) for r in rows] == [
        ("nvidia", 0, 4, 0),
        ("local", 1, 1, 1),
    ]
    assert rows[0]["branch_id"] == "b1" and "503" in rows[0]["error"]


def test_no_failover_raises_the_original_error(config, tmp_path):
    fake = FakeLLM({"nv.test": [404], "local.test": ["never"]})
    router = make(config, fake, tmp_path)
    with pytest.raises(LLMError, match="404"):
        router.complete(MSGS, agent="a", tier="reason", allow_failover=False)
    assert not fake.calls("local.test")


def test_both_backends_failing_raises_last_error(config, tmp_path):
    fake = FakeLLM({"nv.test": [404], "local.test": [httpx.ConnectError("down")] * 4})
    router = make(config, fake, tmp_path)
    with pytest.raises(LLMError, match="ConnectError"):
        router.complete(MSGS, agent="a", tier="reason")
    rows = router.ledger.list_llm_calls("r1")
    assert [r["ok"] for r in rows] == [0, 0] and rows[1]["attempts"] == 4


def test_budget_checked_before_every_call(config, tmp_path):
    router = make(config, FakeLLM(), tmp_path, Budget(max_tokens=1, max_seconds=100))
    router.complete(MSGS, agent="a", tier="cheap")  # first call allowed, then over budget
    with pytest.raises(BudgetExceeded):
        router.complete(MSGS, agent="a", tier="cheap")


def test_per_backend_concurrency_limit(config, tmp_path):
    """local has max_concurrency 1: two threads must serialise on it."""
    active = {"local": 0, "peak": 0}
    lock = threading.Lock()
    gate = threading.Event()

    def handler(request):
        if request.url.host == "local.test":
            with lock:
                active["local"] += 1
                active["peak"] = max(active["peak"], active["local"])
            gate.wait(0.2)
            with lock:
                active["local"] -= 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    ledger = Ledger(tmp_path / "ledger.db")
    router = Router(
        config, ledger=ledger, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    threads = [
        threading.Thread(
            target=router.complete, args=(MSGS,), kwargs={"agent": "a", "tier": "cheap"}
        )
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert active["peak"] == 1
