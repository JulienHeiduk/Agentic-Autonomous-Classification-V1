"""Run directory layout and atomic writes (README section 8).

runs/
  ledger.db                      shared across runs
  _data/{slug}/                  cached competition downloads
  {run_id}/
    config.resolved.yaml         effective config, secrets masked
    profile.json                 Scout output
    folds.npy                    the one fold assignment for the whole run
    run.log
    branches/{branch_id}/        plan.json features.py oof.npy test_pred.npy metrics.json stdout.log
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_ID_FORMAT = "%Y%m%d-%H%M"


def new_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime(RUN_ID_FORMAT)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: str

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.db"

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def branches_dir(self) -> Path:
        return self.run_dir / "branches"

    @property
    def config_snapshot(self) -> Path:
        return self.run_dir / "config.resolved.yaml"

    @property
    def profile_json(self) -> Path:
        return self.run_dir / "profile.json"

    @property
    def folds_path(self) -> Path:
        return self.run_dir / "folds.npy"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"

    def branch_dir(self, branch_id: str) -> Path:
        return self.branches_dir / branch_id

    def data_dir(self, slug: str) -> Path:
        return self.root / "_data" / slug

    def create(self) -> RunPaths:
        self.branches_dir.mkdir(parents=True, exist_ok=False)
        return self


def allocate_run_dir(root: str | Path, now: datetime | None = None) -> RunPaths:
    """Create a fresh run directory. Two runs in the same minute get a ``-2``, ``-3`` suffix."""
    root = Path(root)
    base = new_run_id(now)
    for n in range(1, 1000):
        run_id = base if n == 1 else f"{base}-{n}"
        paths = RunPaths(root, run_id)
        if not paths.run_dir.exists():
            return paths.create()
    raise RuntimeError(f"could not allocate a run directory under {root} for {base}")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write to a temp file in the same directory, then rename: readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
