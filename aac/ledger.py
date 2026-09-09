"""SQLite run ledger (README section 8).

One file at ``runs/ledger.db`` shared by every run. WAL mode, a single connection per process,
and a lock around writes so branch threads can record concurrently. Rows come back as plain
dicts; JSON columns are decoded on the way out.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 6

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    slug         TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    config_hash  TEXT NOT NULL,
    config_path  TEXT,
    status       TEXT NOT NULL,
    error        TEXT
);
CREATE TABLE IF NOT EXISTS branches (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(id),
    backend      TEXT,
    model        TEXT,
    plan_json    TEXT,
    plan_hash    TEXT,
    status       TEXT NOT NULL,
    iteration    INTEGER NOT NULL DEFAULT 0,
    cv_mean      REAL,
    cv_std       REAL,
    fold_scores  TEXT,
    duration     REAL,
    error        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS branches_run ON branches(run_id);
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    branch_id         TEXT,
    agent             TEXT NOT NULL,
    tier              TEXT,
    backend           TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency           REAL,
    cost              REAL,
    ok                INTEGER NOT NULL,
    error             TEXT,
    created_at        TEXT NOT NULL,
    attempts          INTEGER NOT NULL DEFAULT 1,
    escalated         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS llm_calls_run ON llm_calls(run_id);
CREATE TABLE IF NOT EXISTS experiments (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(id),
    agent        TEXT NOT NULL,
    backend      TEXT,
    model        TEXT,
    track        TEXT,
    round        INTEGER NOT NULL,
    code_hash    TEXT,
    hypothesis   TEXT,
    status       TEXT NOT NULL,
    kind         TEXT,
    cv_mean      REAL,
    cv_std       REAL,
    oof_score    REAL,
    fold_scores  TEXT,
    duration     REAL,
    error        TEXT,
    parent       TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS experiments_run ON experiments(run_id);
CREATE TABLE IF NOT EXISTS notes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT,
    source       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    text         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    track        TEXT,
    origin_run   TEXT
);
CREATE TABLE IF NOT EXISTS model_track_record (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    backend      TEXT NOT NULL,
    model        TEXT NOT NULL,
    task         TEXT NOT NULL,
    valid_module INTEGER NOT NULL,
    ran_ok       INTEGER NOT NULL,
    oof_score    REAL,
    latency      REAL,
    tokens       INTEGER,
    error        TEXT,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS track_record_model ON model_track_record(backend, model);
CREATE TABLE IF NOT EXISTS submissions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(id),
    filename      TEXT NOT NULL,
    description   TEXT,
    oof_score     REAL,
    public_score  REAL,
    kaggle_ref    TEXT,
    submitted_at  TEXT NOT NULL
);
"""

# Applied in order to bring an older ledger up to SCHEMA_VERSION. Never edit a shipped step.
_MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE llm_calls ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE llm_calls ADD COLUMN escalated INTEGER NOT NULL DEFAULT 0",
    ],
    3: [],  # experiments and notes are CREATE TABLE IF NOT EXISTS in the base schema
    4: [],  # model_track_record likewise
    5: ["ALTER TABLE notes ADD COLUMN track TEXT"],  # notes addressed to one track
    6: ["ALTER TABLE notes ADD COLUMN origin_run TEXT"],  # the run a reused note came from
}

RUN_STATUSES = ("running", "completed", "failed", "stopped")
BRANCH_STATUSES = (
    "planned",
    "engineering",
    "training",
    "judging",
    "promoted",
    "abandoned",
    "failed",
)
_BRANCH_UPDATABLE = frozenset(
    {
        "plan_json",
        "plan_hash",
        "status",
        "iteration",
        "cv_mean",
        "cv_std",
        "fold_scores",
        "duration",
        "error",
        "backend",
        "model",
    }
)
EXPERIMENT_STATUSES = ("ok", "failed")
_JSON_COLUMNS = frozenset({"fold_scores", "plan_json"})


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._ensure_version()

    def _ensure_version(self) -> None:
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self._conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
            self._conn.commit()
            return
        version = int(row["version"])
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path}: ledger schema v{version} is newer than this code (v{SCHEMA_VERSION})"
            )
        for target in range(version + 1, SCHEMA_VERSION + 1):
            for statement in _MIGRATIONS.get(target, []):
                try:
                    self._conn.execute(statement)
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc):
                        raise
            self._conn.execute("UPDATE schema_version SET version = ?", (target,))
            self._conn.commit()

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("SELECT version FROM schema_version").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- helpers --------------------------------------------------------------------------

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._decode(dict(r)) for r in rows]

    def _one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self._rows(sql, params)
        return rows[0] if rows else None

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        for col in _JSON_COLUMNS:
            if isinstance(row.get(col), str):
                row[col] = json.loads(row[col])
        return row

    @staticmethod
    def _encode(col: str, value: Any) -> Any:
        if col in _JSON_COLUMNS and value is not None and not isinstance(value, str):
            return json.dumps(value)
        return value

    # -- runs -----------------------------------------------------------------------------

    def create_run(self, run_id: str, slug: str, config_hash: str, config_path: str | None) -> None:
        self._write(
            "INSERT INTO runs (id, slug, started_at, config_hash, config_path, status) "
            "VALUES (?, ?, ?, ?, ?, 'running')",
            (run_id, slug, utcnow(), config_hash, config_path),
        )

    def set_run_status(self, run_id: str, status: str, error: str | None = None) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"unknown run status {status!r}")
        finished = utcnow() if status != "running" else None
        self._write(
            "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            (status, error, finished, run_id),
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def list_runs(self, slug: str | None = None) -> list[dict[str, Any]]:
        if slug is None:
            return self._rows("SELECT * FROM runs ORDER BY started_at DESC")
        return self._rows("SELECT * FROM runs WHERE slug = ? ORDER BY started_at DESC", (slug,))

    # -- branches -------------------------------------------------------------------------

    def create_branch(
        self,
        branch_id: str,
        run_id: str,
        *,
        backend: str | None = None,
        model: str | None = None,
        plan: dict[str, Any] | None = None,
        plan_hash: str | None = None,
        status: str = "planned",
    ) -> None:
        now = utcnow()
        self._write(
            "INSERT INTO branches (id, run_id, backend, model, plan_json, plan_hash, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                branch_id,
                run_id,
                backend,
                model,
                self._encode("plan_json", plan),
                plan_hash,
                status,
                now,
                now,
            ),
        )

    def update_branch(self, branch_id: str, **fields: Any) -> None:
        unknown = set(fields) - _BRANCH_UPDATABLE
        if unknown:
            raise ValueError(f"branch columns not updatable: {sorted(unknown)}")
        if "status" in fields and fields["status"] not in BRANCH_STATUSES:
            raise ValueError(f"unknown branch status {fields['status']!r}")
        cols = list(fields)
        values = [self._encode(c, fields[c]) for c in cols]
        assignments = ", ".join(f"{c} = ?" for c in cols) + ", updated_at = ?"
        self._write(
            f"UPDATE branches SET {assignments} WHERE id = ?", (*values, utcnow(), branch_id)
        )

    def get_branch(self, branch_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM branches WHERE id = ?", (branch_id,))

    def list_branches(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM branches WHERE run_id = ? ORDER BY created_at", (run_id,))

    def best_branches(
        self, slug: str, limit: int = 10, greater_is_better: bool = True
    ) -> list[dict[str, Any]]:
        """Top scored branches across every run of a competition, for the Historian."""
        order = "DESC" if greater_is_better else "ASC"
        return self._rows(
            "SELECT b.*, r.slug FROM branches b JOIN runs r ON r.id = b.run_id "
            f"WHERE r.slug = ? AND b.cv_mean IS NOT NULL ORDER BY b.cv_mean {order} LIMIT ?",
            (slug, limit),
        )

    # -- llm calls ------------------------------------------------------------------------

    def record_llm_call(
        self,
        *,
        agent: str,
        backend: str,
        model: str,
        ok: bool,
        run_id: str | None = None,
        branch_id: str | None = None,
        tier: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency: float | None = None,
        cost: float | None = None,
        error: str | None = None,
        attempts: int = 1,
        escalated: bool = False,
    ) -> int:
        return self._write(
            "INSERT INTO llm_calls (run_id, branch_id, agent, tier, backend, model, prompt_tokens, "
            "completion_tokens, latency, cost, ok, error, created_at, attempts, escalated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                branch_id,
                agent,
                tier,
                backend,
                model,
                prompt_tokens,
                completion_tokens,
                latency,
                cost,
                int(ok),
                error,
                utcnow(),
                attempts,
                int(escalated),
            ),
        )

    def list_llm_calls(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None:
            return self._rows("SELECT * FROM llm_calls ORDER BY id")
        return self._rows("SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id", (run_id,))

    def token_totals(self, run_id: str) -> tuple[int, int]:
        row = self._one(
            "SELECT COALESCE(SUM(prompt_tokens), 0) AS p, COALESCE(SUM(completion_tokens), 0) AS c "
            "FROM llm_calls WHERE run_id = ?",
            (run_id,),
        )
        assert row is not None
        return int(row["p"]), int(row["c"])

    # -- experiments ----------------------------------------------------------------------

    def record_experiment(
        self,
        experiment_id: str,
        run_id: str,
        *,
        agent: str,
        round: int,
        status: str,
        backend: str | None = None,
        model: str | None = None,
        track: str | None = None,
        code_hash: str | None = None,
        hypothesis: str | None = None,
        kind: str | None = None,
        cv_mean: float | None = None,
        cv_std: float | None = None,
        oof_score: float | None = None,
        fold_scores: list[float] | None = None,
        duration: float | None = None,
        error: str | None = None,
        parent: str | None = None,
    ) -> None:
        if status not in EXPERIMENT_STATUSES:
            raise ValueError(f"unknown experiment status {status!r}")
        self._write(
            "INSERT INTO experiments (id, run_id, agent, backend, model, track, round, code_hash, "
            "hypothesis, status, kind, cv_mean, cv_std, oof_score, fold_scores, duration, error, "
            "parent, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                experiment_id,
                run_id,
                agent,
                backend,
                model,
                track,
                round,
                code_hash,
                hypothesis,
                status,
                kind,
                cv_mean,
                cv_std,
                oof_score,
                self._encode("fold_scores", fold_scores),
                duration,
                error,
                parent,
                utcnow(),
            ),
        )

    def list_experiments(self, run_id: str, agent: str | None = None) -> list[dict[str, Any]]:
        if agent is None:
            return self._rows(
                "SELECT * FROM experiments WHERE run_id = ? ORDER BY created_at", (run_id,)
            )
        return self._rows(
            "SELECT * FROM experiments WHERE run_id = ? AND agent = ? ORDER BY round",
            (run_id, agent),
        )

    def best_experiments(
        self,
        slug: str,
        limit: int = 10,
        greater_is_better: bool = True,
        track: str | None = None,
    ) -> list[dict[str, Any]]:
        order = "DESC" if greater_is_better else "ASC"
        track_clause = "" if track is None else " AND e.track = ?"
        params: tuple[Any, ...] = (slug, track, limit) if track is not None else (slug, limit)
        return self._rows(
            "SELECT e.*, r.slug FROM experiments e JOIN runs r ON r.id = e.run_id "
            f"WHERE r.slug = ? AND e.oof_score IS NOT NULL AND e.status = 'ok'{track_clause} "
            f"ORDER BY e.oof_score {order} LIMIT ?",
            params,
        )

    def failed_experiments(
        self, slug: str, *, exclude_run: str | None = None, limit: int = 300
    ) -> list[dict[str, Any]]:
        """Failed experiments on a competition, most recent first (the Historian's pitfalls)."""
        clause = "" if exclude_run is None else " AND e.run_id != ?"
        params: tuple[Any, ...] = (slug, exclude_run, limit) if exclude_run else (slug, limit)
        return self._rows(
            "SELECT e.*, r.slug FROM experiments e JOIN runs r ON r.id = e.run_id "
            f"WHERE r.slug = ? AND e.status = 'failed'{clause} "
            "ORDER BY e.created_at DESC, e.rowid DESC LIMIT ?",
            params,
        )

    def add_note(
        self,
        *,
        source: str,
        kind: str,
        text: str,
        run_id: str | None = None,
        track: str | None = None,
        origin_run: str | None = None,
    ) -> int:
        """A note; ``track`` addresses it to one Researcher track, None to every reader;
        ``origin_run`` records where a reused note was first produced."""
        return self._write(
            "INSERT INTO notes (run_id, source, kind, text, created_at, track, origin_run) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, source, kind, text, utcnow(), track, origin_run),
        )

    def latest_notes(
        self, slug: str, kind: str, *, exclude_run: str | None = None
    ) -> list[dict[str, Any]]:
        """Notes of ``kind`` from the most recent run on the slug that has any."""
        clause = "" if exclude_run is None else " AND n.run_id != ?"
        params: tuple[Any, ...] = (slug, kind, exclude_run) if exclude_run else (slug, kind)
        row = self._one(
            "SELECT n.run_id FROM notes n JOIN runs r ON r.id = n.run_id "
            f"WHERE r.slug = ? AND n.kind = ?{clause} ORDER BY r.started_at DESC, n.id DESC "
            "LIMIT 1",
            params,
        )
        return [] if row is None else self.list_notes(row["run_id"], kind=kind)

    def runs_since(self, slug: str, run_id: str, *, exclude_run: str | None = None) -> int:
        """How many runs on the slug started after ``run_id``; a missing run counts as ancient."""
        origin = self.get_run(run_id)
        if origin is None:
            return 10**6
        clause = "" if exclude_run is None else " AND id != ?"
        params: tuple[Any, ...] = (
            (slug, origin["started_at"], exclude_run)
            if exclude_run
            else (slug, origin["started_at"])
        )
        row = self._one(
            f"SELECT COUNT(*) AS n FROM runs WHERE slug = ? AND started_at > ?{clause}", params
        )
        return int(row["n"]) if row else 0

    def list_notes(
        self, run_id: str | None = None, kind: str | None = None, track: str | None = None
    ) -> list[dict[str, Any]]:
        """Notes, oldest first. With ``track``: the shared notes plus that track's own."""
        clauses, params = [], []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if track is not None:
            clauses.append("(track IS NULL OR track = ?)")
            params.append(track)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._rows(f"SELECT * FROM notes {where} ORDER BY id", tuple(params))

    # -- model track record ---------------------------------------------------------------

    def record_track(
        self,
        *,
        backend: str,
        model: str,
        task: str,
        valid_module: bool,
        ran_ok: bool,
        oof_score: float | None = None,
        latency: float | None = None,
        tokens: int | None = None,
        error: str | None = None,
    ) -> int:
        return self._write(
            "INSERT INTO model_track_record (backend, model, task, valid_module, ran_ok, "
            "oof_score, "
            "latency, tokens, error, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                backend,
                model,
                task,
                int(valid_module),
                int(ran_ok),
                oof_score,
                latency,
                tokens,
                error,
                utcnow(),
            ),
        )

    def track_records(
        self, backend: str | None = None, model: str | None = None
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if backend is not None:
            clauses.append("backend = ?")
            params.append(backend)
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._rows(f"SELECT * FROM model_track_record {where} ORDER BY id", tuple(params))

    # -- submissions ----------------------------------------------------------------------

    def record_submission(
        self,
        run_id: str,
        filename: str,
        *,
        description: str | None = None,
        oof_score: float | None = None,
        kaggle_ref: str | None = None,
    ) -> int:
        return self._write(
            "INSERT INTO submissions (run_id, filename, description, oof_score, kaggle_ref, "
            "submitted_at) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, filename, description, oof_score, kaggle_ref, utcnow()),
        )

    def set_public_score(self, submission_id: int, public_score: float | None) -> None:
        self._write(
            "UPDATE submissions SET public_score = ? WHERE id = ?", (public_score, submission_id)
        )

    def pending_submissions(self, slug: str) -> list[dict[str, Any]]:
        """Uploads on a competition that never received a public score."""
        return self._rows(
            "SELECT s.* FROM submissions s JOIN runs r ON r.id = s.run_id "
            "WHERE r.slug = ? AND s.public_score IS NULL AND s.kaggle_ref IS NOT NULL "
            "ORDER BY s.submitted_at",
            (slug,),
        )

    def list_submissions(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is None:
            return self._rows("SELECT * FROM submissions ORDER BY submitted_at DESC")
        return self._rows(
            "SELECT * FROM submissions WHERE run_id = ? ORDER BY submitted_at DESC", (run_id,)
        )
