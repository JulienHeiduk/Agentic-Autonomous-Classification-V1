"""RunContext: everything an agent needs, handed in by the orchestrator.

Budget counters live here from the start (README section 2, budget-capped). They are
enforced by ``Budget.check()`` at every LLM call and stage boundary, never by convention.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from aac.config import Config, resolved_yaml
from aac.exec.artifacts import RunPaths, allocate_run_dir, atomic_write_text
from aac.ledger import Ledger

log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """A hard limit was hit; the run must stop and record why."""


@dataclass
class Budget:
    max_tokens: int
    max_seconds: float
    started: float = field(default_factory=time.monotonic)
    tokens_used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    def charge(self, prompt_tokens: int, completion_tokens: int) -> int:
        with self._lock:
            self.tokens_used += int(prompt_tokens) + int(completion_tokens)
            return self.tokens_used

    def exhausted(self) -> str | None:
        """Reason the budget is spent, or None while there is room left."""
        if self.tokens_used >= self.max_tokens:
            return f"token budget exhausted ({self.tokens_used} >= {self.max_tokens})"
        if self.elapsed >= self.max_seconds:
            return f"wall clock budget exhausted ({self.elapsed:.0f}s >= {self.max_seconds:.0f}s)"
        return None

    def check(self) -> None:
        reason = self.exhausted()
        if reason:
            raise BudgetExceeded(reason)


class RunContext:
    def __init__(self, config: Config, paths: RunPaths, ledger: Ledger, budget: Budget) -> None:
        self.config = config
        self.paths = paths
        self.ledger = ledger
        self.budget = budget

    @property
    def run_id(self) -> str:
        return self.paths.run_id

    @classmethod
    def create(
        cls, config: Config, runs_root: str | Path = "runs", now: datetime | None = None
    ) -> RunContext:
        """Start a new run: allocate the directory, open the ledger, snapshot the config."""
        paths = allocate_run_dir(runs_root, now)
        atomic_write_text(paths.config_snapshot, resolved_yaml(config))
        ledger = Ledger(paths.ledger_path)
        ledger.create_run(
            paths.run_id,
            config.competition.slug,
            config.fingerprint(),
            str(config.source) if config.source else None,
        )
        budget = Budget(
            max_tokens=config.run.max_tokens_total,
            max_seconds=config.run.max_wall_clock_minutes * 60,
        )
        log.info("run %s started in %s", paths.run_id, paths.run_dir)
        return cls(config, paths, ledger, budget)

    def finish(self, status: str, error: str | None = None) -> None:
        self.ledger.set_run_status(self.run_id, status, error)

    def close(self) -> None:
        self.ledger.close()
