import json

import httpx

from aac.agents.scholar import DEFAULT_ANGLES, Packet, packet_to_note, research
from aac.agents.scout import profile_data
from aac.config import CompetitionConfig, ScholarConfig, load_config
from aac.ledger import Ledger
from aac.llm.router import Router
from aac.models.metrics import METRICS
from tests.llm_fake import FakeLLM
from tests.synth import make_frames

PACKET = {
    "ideas": [
        {
            "title": "x1 times x2",
            "kind": "feature",
            "description": "multiply x1 and x2",
            "why": "interaction",
            "columns": ["x1", "x2"],
        },
        {
            "title": "bagged lightgbm",
            "kind": "model",
            "description": "3 seeds averaged",
            "why": "variance",
        },
    ]
}


def test_packet_to_note():
    text = packet_to_note("features", Packet.model_validate(PACKET))
    assert text.startswith("Research packet (features):")
    assert "- (feature) x1 times x2: multiply x1 and x2 [columns: x1, x2] Why: interaction" in text


def test_research_runs_every_angle_and_records_notes(tmp_path, write_config, minimal_config):
    config = load_config(write_config(minimal_config), env={"NV_KEY": "k"})
    train, test, sample = make_frames(120, 40)
    profile = profile_data(
        train,
        test,
        sample,
        slug="s",
        competition=CompetitionConfig(slug="s"),
        metric=METRICS["auc"],
        max_classes=50,
        seed=0,
    )
    bad = dict(PACKET, ideas=[dict(PACKET["ideas"][0], columns=["ghost"])])
    fake = FakeLLM(
        {"nv.test": [json.dumps(PACKET), json.dumps(bad), json.dumps(PACKET), "garbage", "garbage"]}
    )
    ledger = Ledger(tmp_path / "ledger.db")
    ledger.create_run("r1", "s", "h", None)
    router = Router(
        config,
        ledger=ledger,
        run_id="r1",
        client=httpx.Client(transport=fake.transport),
        sleep=lambda s: None,
    )
    spec = ScholarConfig(backend="nvidia", model="m-nv", max_ideas=1)
    notes = research(
        router,
        profile,
        config=spec,
        metric=METRICS["auc"],
        competition_text="Synthetic",
        ledger=ledger,
        run_id="r1",
    )
    assert 1 <= len(notes) <= 3
    assert all(n.count("- (") == 1 for n in notes), "max_ideas trims each packet to one idea"
    stored = ledger.list_notes("r1", kind="research")
    assert len(stored) == len(notes) and stored[0]["source"] == "scholar"
    calls = fake.calls("nv.test")
    assert len(calls) >= 3
    angles_seen = {json.dumps(c["messages"][-1]["content"][:80]) for c in calls}
    assert len(angles_seen) >= 2
    assert any(a[:30] in calls[0]["messages"][-1]["content"] for a in DEFAULT_ANGLES)
