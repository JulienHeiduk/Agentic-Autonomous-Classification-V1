"""Scholar: LLM research packets, ideas only (README section 17.2, the article's phase 3).

The strongest slow hub model is asked, from several angles in parallel, for testable ideas.
Each packet becomes a ledger note of kind ``research`` that every Researcher reads.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import Field

from aac.agents.researcher import render_columns
from aac.agents.scout import Profile
from aac.config import ScholarConfig, StrictModel
from aac.ledger import Ledger
from aac.llm.prompts import load_prompt
from aac.llm.router import Router
from aac.llm.schema import describe_schema, structured_completion
from aac.models.metrics import MetricSpec

log = logging.getLogger(__name__)

DEFAULT_ANGLES = [
    "feature construction from the raw columns: ratios, interactions, parsing of structured "
    "strings, counts and frequencies, missingness patterns, binning",
    "model families and their configuration: what to tune, regularisation, class imbalance "
    "handling, categorical treatment, seeds and bagging for ensemble diversity",
    "what has won recent Kaggle playground tabular competitions of this size and metric, and "
    "which validation traps to avoid",
]


class Idea(StrictModel):
    title: str = Field(min_length=1, max_length=80)
    kind: Literal["feature", "model", "validation", "ensemble"]
    description: str = Field(min_length=1)
    why: str = ""
    columns: list[str] = Field(default_factory=list)


class Packet(StrictModel):
    ideas: list[Idea] = Field(min_length=1)


EXAMPLE = Packet(
    ideas=[
        Idea(
            title="spend per item",
            kind="feature",
            description="amount_col divided by (count_col + 1), NaN when amount_col is missing",
            why="ratios expose per-unit behaviour trees cannot form from two raw columns",
            columns=["amount_col", "count_col"],
        ),
        Idea(
            title="lower learning rate, more trees, bagged seeds",
            kind="model",
            description="lightgbm learning_rate 0.02 with 2000 trees and early stopping inside "
            "fit_predict, averaged over 3 seeds",
            why="reduces variance and adds a diverse member to the blend",
        ),
    ]
)


def packet_to_note(angle: str, packet: Packet) -> str:
    lines = [f"Research packet ({angle[:60]}):"]
    for idea in packet.ideas:
        cols = f" [columns: {', '.join(idea.columns)}]" if idea.columns else ""
        why = f" Why: {idea.why}" if idea.why else ""
        lines.append(f"- ({idea.kind}) {idea.title}: {idea.description}{cols}{why}")
    return "\n".join(lines)


def research(
    router: Router,
    profile: Profile,
    *,
    config: ScholarConfig,
    metric: MetricSpec,
    competition_text: str,
    leaderboard: str = "",
    ledger: Ledger | None = None,
    run_id: str | None = None,
) -> list[str]:
    """Run one prompt per angle in parallel; store and return the packets as note texts."""
    angles = config.angles or DEFAULT_ANGLES
    t = profile.target

    def one(angle: str) -> str | None:
        messages = load_prompt("scholar").messages(
            max_ideas=config.max_ideas,
            angle=angle,
            competition=competition_text,
            metric=metric.display,
            direction="higher is better" if metric.greater_is_better else "lower is better",
            target=t.name,
            target_kind=t.kind,
            positive_label=repr(t.positive_label),
            class_rates=", ".join(f"{k}: {v:.4f}" for k, v in t.rates.items()),
            n_train=profile.n_train,
            n_test=profile.n_test,
            columns=render_columns(profile),
            warnings="\n".join(f"- {w}" for w in profile.warnings) or "(none)",
            leaderboard=leaderboard or "Team leaderboard: (no results yet)",
            schema=describe_schema(Packet, EXAMPLE),
        )
        result = structured_completion(
            router,
            messages,
            Packet,
            agent="scholar",
            backend=config.backend,
            model=config.model,
            temperature=config.temperature,
            max_tokens=4096,
            context={"columns": [c.name for c in profile.columns]},
        )
        if not result.ok:
            log.warning("scholar packet failed (%s): %s", angle[:40], result.error)
            return None
        packet = result.value
        packet = Packet(ideas=packet.ideas[: config.max_ideas])
        return packet_to_note(angle, packet)

    with ThreadPoolExecutor(max_workers=max(1, min(len(angles), 3))) as pool:
        notes = [n for n in pool.map(one, angles) if n]
    for note in notes:
        if ledger is not None:
            ledger.add_note(source="scholar", kind="research", text=note, run_id=run_id)
    log.info("scholar: %d/%d packets", len(notes), len(angles))
    return notes
