import sqlite3
import threading

import pytest

from aac.ledger import SCHEMA_VERSION, Ledger


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "runs" / "ledger.db") as lg:
        yield lg


def test_schema_created_and_reopenable(tmp_path):
    path = tmp_path / "ledger.db"
    with Ledger(path) as a:
        assert a.schema_version == SCHEMA_VERSION
        a.create_run("r1", "slug", "hash", None)
    with Ledger(path) as b:
        assert b.get_run("r1")["status"] == "running"


def test_run_lifecycle(ledger):
    ledger.create_run("r1", "slug-a", "h1", "configs/x.yaml")
    ledger.create_run("r2", "slug-b", "h2", None)
    run = ledger.get_run("r1")
    assert run["status"] == "running" and run["finished_at"] is None
    ledger.set_run_status("r1", "failed", error="boom")
    run = ledger.get_run("r1")
    assert run["status"] == "failed" and run["error"] == "boom" and run["finished_at"]
    with pytest.raises(ValueError):
        ledger.set_run_status("r1", "exploded")
    assert [r["id"] for r in ledger.list_runs("slug-a")] == ["r1"]
    assert len(ledger.list_runs()) == 2
    assert ledger.get_run("nope") is None


def test_branch_json_roundtrip_and_updates(ledger):
    ledger.create_run("r1", "slug", "h", None)
    plan = {"features": ["a", "b"], "models": [{"family": "lightgbm"}]}
    ledger.create_branch("b1", "r1", backend="nvidia", model="m", plan=plan, plan_hash="ph")
    b = ledger.get_branch("b1")
    assert b["plan_json"] == plan and b["status"] == "planned" and b["iteration"] == 0
    ledger.update_branch(
        "b1", status="promoted", cv_mean=0.91, cv_std=0.01, fold_scores=[0.9, 0.92], iteration=2
    )
    b = ledger.get_branch("b1")
    assert b["fold_scores"] == [0.9, 0.92] and b["status"] == "promoted" and b["iteration"] == 2
    assert b["updated_at"] >= b["created_at"]
    with pytest.raises(ValueError, match="not updatable"):
        ledger.update_branch("b1", run_id="r2")
    with pytest.raises(ValueError, match="status"):
        ledger.update_branch("b1", status="meh")
    assert [x["id"] for x in ledger.list_branches("r1")] == ["b1"]


def test_best_branches_across_runs_respects_direction(ledger):
    ledger.create_run("r1", "slug", "h", None)
    ledger.create_run("r2", "slug", "h", None)
    ledger.create_run("r3", "other", "h", None)
    for bid, run, cv in [
        ("b1", "r1", 0.80),
        ("b2", "r2", 0.95),
        ("b3", "r2", None),
        ("b4", "r3", 0.99),
    ]:
        ledger.create_branch(bid, run)
        if cv is not None:
            ledger.update_branch(bid, cv_mean=cv)
    assert [b["id"] for b in ledger.best_branches("slug")] == ["b2", "b1"]
    assert [b["id"] for b in ledger.best_branches("slug", greater_is_better=False)] == ["b1", "b2"]
    assert [b["id"] for b in ledger.best_branches("slug", limit=1)] == ["b2"]


def test_llm_calls_and_token_totals(ledger):
    ledger.create_run("r1", "slug", "h", None)
    ledger.record_llm_call(
        run_id="r1",
        agent="architect",
        tier="reason",
        backend="nvidia",
        model="m",
        prompt_tokens=100,
        completion_tokens=50,
        latency=1.5,
        cost=0.0,
        ok=True,
    )
    ledger.record_llm_call(
        run_id="r1", agent="critic", backend="local", model="q", ok=False, error="timeout"
    )
    ledger.record_llm_call(agent="doctor", backend="local", model="q", ok=True, prompt_tokens=5)
    assert ledger.token_totals("r1") == (100, 50)
    assert ledger.token_totals("none") == (0, 0)


def test_submissions(ledger):
    ledger.create_run("r1", "slug", "h", None)
    sid = ledger.record_submission("r1", "submission.csv", description="blend", oof_score=0.9)
    assert ledger.list_submissions("r1")[0]["public_score"] is None
    ledger.set_public_score(sid, 0.905)
    assert ledger.list_submissions("r1")[0]["public_score"] == 0.905
    assert len(ledger.list_submissions()) == 1


def test_concurrent_writes(ledger):
    ledger.create_run("r1", "slug", "h", None)

    def work():
        for _ in range(50):
            ledger.record_llm_call(
                run_id="r1",
                agent="t",
                backend="b",
                model="m",
                ok=True,
                prompt_tokens=1,
                completion_tokens=1,
            )

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ledger.token_totals("r1") == (400, 400)


def test_migrates_a_v1_ledger(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (1);
        CREATE TABLE runs (id TEXT PRIMARY KEY, slug TEXT NOT NULL, started_at TEXT NOT NULL,
            finished_at TEXT, config_hash TEXT NOT NULL, config_path TEXT, status TEXT NOT NULL,
            error TEXT);
        CREATE TABLE llm_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, branch_id TEXT,
            agent TEXT NOT NULL, tier TEXT, backend TEXT NOT NULL, model TEXT NOT NULL,
            prompt_tokens INTEGER, completion_tokens INTEGER, latency REAL, cost REAL,
            ok INTEGER NOT NULL, error TEXT, created_at TEXT NOT NULL);
        INSERT INTO llm_calls (agent, backend, model, ok, created_at)
            VALUES ('old', 'b', 'm', 1, '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()
    with Ledger(path) as ledger:
        assert ledger.schema_version == SCHEMA_VERSION
        [old] = ledger.list_llm_calls()
        assert old["attempts"] == 1 and old["escalated"] == 0
        ledger.record_llm_call(
            agent="new", backend="b", model="m", ok=False, attempts=4, escalated=True, error="boom"
        )
        new = ledger.list_llm_calls()[-1]
        assert new["attempts"] == 4 and new["escalated"] == 1 and new["error"] == "boom"


def test_refuses_a_newer_ledger(tmp_path):
    path = tmp_path / "future.db"
    with Ledger(path) as ledger:
        ledger._conn.execute("UPDATE schema_version SET version = 999")
        ledger._conn.commit()
    with pytest.raises(RuntimeError, match="newer"):
        Ledger(path)


def test_experiments_and_notes(ledger):
    ledger.create_run("r1", "slug", "h", None)
    ledger.record_experiment(
        "r1-a-r1",
        "r1",
        agent="a",
        round=1,
        status="ok",
        track="gbdt",
        oof_score=0.9,
        cv_mean=0.9,
        cv_std=0.01,
        fold_scores=[0.89, 0.91],
    )
    ledger.record_experiment(
        "r1-a-r2",
        "r1",
        agent="a",
        round=2,
        status="failed",
        kind="error",
        error="boom",
        parent="r1-a-r1",
    )
    ledger.record_experiment("r1-b-r1", "r1", agent="b", round=1, status="ok", oof_score=0.95)
    rows = ledger.list_experiments("r1")
    assert len(rows) == 3 and rows[0]["fold_scores"] == [0.89, 0.91]
    assert [r["round"] for r in ledger.list_experiments("r1", "a")] == [1, 2]
    assert [e["id"] for e in ledger.best_experiments("slug")] == ["r1-b-r1", "r1-a-r1"]
    with pytest.raises(ValueError):
        ledger.record_experiment("x", "r1", agent="a", round=3, status="meh")
    ledger.add_note(source="scholar", kind="research", text="try ratios", run_id="r1")
    ledger.add_note(source="assessor", kind="track_record", text="m-nv: good")
    assert [n["text"] for n in ledger.list_notes("r1")] == ["try ratios"]
    assert len(ledger.list_notes(kind="track_record")) == 1
    assert ledger.schema_version == SCHEMA_VERSION >= 5
    # notes addressed to one track reach that track and the shared readers only
    for track in ("gbdt", "linear"):
        ledger.add_note(
            source="historian", kind="prior", text=f"{track} code", run_id="r1", track=track
        )
    gbdt_view = [n["text"] for n in ledger.list_notes("r1", track="gbdt")]
    assert gbdt_view == ["try ratios", "gbdt code"]
    assert [n["text"] for n in ledger.list_notes("r1", track="open")] == ["try ratios"]
    assert [n["track"] for n in ledger.list_notes("r1")] == [None, "gbdt", "linear"]
    # failures come back most recent first; scored failures never rank as best
    ledger.record_experiment(
        "r1-b-r2", "r1", agent="b", round=2, status="failed", kind="degenerate", oof_score=0.5
    )
    ledger.create_run("r2", "slug", "h", None)
    ledger.record_experiment(
        "r2-a-r1", "r2", agent="a", round=1, status="failed", kind="timeout", error="killed"
    )
    failed = [e["id"] for e in ledger.failed_experiments("slug")]
    assert failed == ["r2-a-r1", "r1-b-r2", "r1-a-r2"]
    assert [e["id"] for e in ledger.failed_experiments("slug", exclude_run="r2")] == [
        "r1-b-r2",
        "r1-a-r2",
    ]
    assert [e["id"] for e in ledger.best_experiments("slug")] == ["r1-b-r1", "r1-a-r1"]
    assert [e["id"] for e in ledger.best_experiments("slug", track="gbdt")] == ["r1-a-r1"]
    # the latest notes of a kind on the slug, and how many runs came after their origin
    ledger._conn.execute("UPDATE runs SET started_at = '2026-01-01T00:00:00+00:00' WHERE id='r1'")
    ledger._conn.execute("UPDATE runs SET started_at = '2026-01-02T00:00:00+00:00' WHERE id='r2'")
    ledger._conn.commit()
    assert [n["text"] for n in ledger.latest_notes("slug", "research")] == ["try ratios"]
    assert ledger.latest_notes("slug", "research", exclude_run="r1") == []
    ledger.add_note(source="scholar", kind="research", text="copy", run_id="r2", origin_run="r1")
    [copy] = ledger.latest_notes("slug", "research")
    assert copy["origin_run"] == "r1" and copy["run_id"] == "r2"
    assert ledger.runs_since("slug", "r1") == 1 and ledger.runs_since("slug", "r2") == 0
    assert ledger.runs_since("slug", "r1", exclude_run="r2") == 0
    assert ledger.runs_since("slug", "gone") >= 10**6


def test_migrates_a_v4_ledger_notes_table(tmp_path):
    path = tmp_path / "v4.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (4);
        CREATE TABLE notes (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
            source TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
            created_at TEXT NOT NULL);
        INSERT INTO notes (run_id, source, kind, text, created_at)
            VALUES ('r0', 'scholar', 'research', 'old packet', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()
    with Ledger(path) as ledger:
        assert ledger.schema_version == SCHEMA_VERSION
        [old] = ledger.list_notes("r0")
        assert old["track"] is None and old["text"] == "old packet"
        ledger.add_note(source="historian", kind="prior", text="p", run_id="r0", track="linear")
        assert [n["track"] for n in ledger.list_notes("r0", track="linear")] == [None, "linear"]
        assert old["origin_run"] is None
        ledger.add_note(source="scholar", kind="research", text="c", run_id="r0", origin_run="x")
        assert ledger.list_notes("r0", kind="research")[-1]["origin_run"] == "x"
