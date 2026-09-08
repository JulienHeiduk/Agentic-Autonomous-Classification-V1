"""Historian: prior knowledge from earlier runs on the same competition (README 17.2, 17.3).

Two kinds of note, written once per run before the Researchers start:

- ``prior``: the best earlier experiments and the top module, one note per track, holding
  only experiments a Researcher on that track may build on (a module that imports a library
  the track forbids is never shown to it; it would be copied and rejected).
- ``pitfalls``: the distinct failures seen on the competition, with counts, so the same API
  slip or timeout is not repeated run after run. Shared note for runtime failures; one note
  per track for that track's own import violations.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from aac.agents.researcher import check_track
from aac.ledger import Ledger
from aac.models.metrics import MetricSpec

log = logging.getLogger(__name__)

MAX_CODE_CHARS = 6000
PITFALL_KINDS = ("error", "timeout", "leak", "degenerate", "rejected", "violation")
_TRACEBACK_NOISE = re.compile(r"^\s*(\^+|~+[\^~]*|File \".*|Traceback .*)\s*$")


def experiment_module_path(runs_root: Path, experiment_id: str, run_id: str) -> Path | None:
    """runs/{run_id}/experiments/{agent}/r{round}/experiment.py from the ledger id."""
    tail = experiment_id[len(run_id) + 1 :] if experiment_id.startswith(run_id + "-") else None
    if not tail or "-r" not in tail:
        return None
    agent, _, round_part = tail.rpartition("-r")
    path = runs_root / run_id / "experiments" / agent / f"r{round_part}" / "experiment.py"
    return path if path.is_file() else None


def _module_code(runs_root: Path, row: dict[str, object]) -> str | None:
    path = experiment_module_path(runs_root, str(row["id"]), str(row["run_id"]))
    return path.read_text(encoding="utf-8") if path is not None else None


def compatible(runs_root: Path, row: dict[str, object], track: str | None) -> bool:
    """May a Researcher on ``track`` build on this experiment? Judged by the track rules on
    the module when it is on disk, on the recorded track otherwise. ``open`` and None take
    everything."""
    if track is None or track == "open":
        return True
    code = _module_code(runs_root, row)
    if code is not None:
        return not check_track(code, track)
    return row.get("track") == track


def priors(
    ledger: Ledger,
    runs_root: Path,
    slug: str,
    metric: MetricSpec,
    *,
    exclude_run: str | None = None,
    limit: int = 5,
    track: str | None = None,
) -> str | None:
    rows = [
        r
        for r in ledger.best_experiments(
            slug, limit=limit * 4 + 5, greater_is_better=metric.greater_is_better
        )
        if r["run_id"] != exclude_run and compatible(runs_root, r, track)
    ][:limit]
    if not rows:
        return None
    scope = f" for the {track!r} track" if track and track != "open" else ""
    lines = [f"Prior knowledge from earlier runs on {slug}{scope} (best experiments, honest OOF):"]
    for r in rows:
        lines.append(
            f"- {r['oof_score']:.5f} by {r['model'] or '?'} on track {r['track'] or '?'} "
            f"({r['run_id']}): {r['hypothesis'] or ''}"
        )
    code = _module_code(runs_root, rows[0])
    if code is not None:
        if len(code) > MAX_CODE_CHARS:
            code = code[:MAX_CODE_CHARS] + "\n# ... truncated"
        lines.append(f"The best of them, to build on rather than repeat:\n```python\n{code}\n```")
    return "\n".join(lines)


def record_priors(
    ledger: Ledger,
    runs_root: Path,
    slug: str,
    metric: MetricSpec,
    run_id: str,
    tracks: list[str] | None = None,
) -> bool:
    """One shared prior note (``tracks`` None), or one note per track addressed to it."""
    written = False
    for track in tracks if tracks is not None else [None]:
        text = priors(ledger, runs_root, slug, metric, exclude_run=run_id, track=track)
        if text is None:
            continue
        ledger.add_note(source="historian", kind="prior", text=text, run_id=run_id, track=track)
        written = True
    if written:
        log.info("historian: priors from earlier runs added")
    return written


def error_tail(error: str | None, width: int = 220) -> str:
    """The last informative line of an error: the exception line, not the caret art."""
    lines = [ln.rstrip() for ln in (error or "").splitlines()]
    lines = [ln for ln in lines if ln.strip() and not _TRACEBACK_NOISE.match(ln)]
    if not lines:
        return "(no error text)"
    tail = re.sub(r"\s+", " ", lines[-1].strip())
    return tail[:width]


def pitfalls(
    ledger: Ledger,
    slug: str,
    *,
    exclude_run: str | None = None,
    track: str | None = None,
    limit: int = 10,
) -> str | None:
    """Distinct failures from earlier runs, most recent first, with how often each was seen.

    ``track`` None: runtime failures every Researcher can hit (errors, timeouts, leaks,
    degenerate output). A track: that track's own import violations."""
    rows = ledger.failed_experiments(slug, exclude_run=exclude_run)
    if track is None:
        rows = [r for r in rows if r.get("kind") in PITFALL_KINDS]
    else:
        rows = [r for r in rows if r.get("kind") == "track" and r.get("track") == track]
    if not rows:
        return None
    order: list[str] = []
    seen: dict[str, dict[str, object]] = {}
    counts: Counter[str] = Counter()
    for r in rows:
        tail = error_tail(r.get("error"))  # type: ignore[arg-type]
        counts[tail] += 1
        if tail not in seen:
            seen[tail] = r
            order.append(tail)
    if track is None:
        lines = [
            f"Pitfalls from earlier runs on {slug} (distinct failures, most recent first; "
            "do not repeat them):"
        ]
    else:
        lines = [
            f"Track pitfalls for {track!r} on {slug}: {len(rows)} earlier modules were "
            "rejected before running because of their imports. Import only what your track "
            "allows, whatever the notes and tips below use:"
        ]
    for tail in order[:limit]:
        r = seen[tail]
        n = counts[tail]
        hyp = str(r.get("hypothesis") or "").strip()
        hint = f" (hypothesis: {hyp[:80]})" if hyp and r.get("kind") == "timeout" else ""
        lines.append(
            f"- [{r.get('kind')}, {n}x, last {r.get('run_id')} on {r.get('track') or '?'}] "
            f"{tail}{hint}"
        )
    return "\n".join(lines)


def record_pitfalls(
    ledger: Ledger, slug: str, run_id: str, tracks: list[str] | None = None
) -> bool:
    """The shared pitfalls note plus one per track with violations on record."""
    written = False
    for track in [None, *(tracks or [])]:
        text = pitfalls(ledger, slug, exclude_run=run_id, track=track)
        if text is None:
            continue
        ledger.add_note(source="historian", kind="pitfalls", text=text, run_id=run_id, track=track)
        written = True
    if written:
        log.info("historian: pitfalls from earlier runs added")
    return written
