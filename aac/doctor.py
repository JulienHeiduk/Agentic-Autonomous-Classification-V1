"""``aac doctor``: is the environment worth debugging anything else in? (README section 12)

Checks, in order: credentials by source, config expansion, ledger, each LLM backend (model
list, configured models present, one-token completion), and Kaggle competition metadata with
the metric resolved. Prints a table and returns 0 only when every check passes.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from aac.config import Config, ConfigError, load_config
from aac.env import credential_report
from aac.exec.artifacts import RunPaths
from aac.kaggle.api import KaggleClient, KaggleError
from aac.ledger import Ledger
from aac.llm.client import Backend, LLMError, complete, list_models
from aac.llm.prompts import load_prompt
from aac.llm.router import Router
from aac.llm.schema import structured_completion
from aac.models.metrics import UnsupportedMetricError, resolve_metric

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/s6e9.yaml")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    warn: bool = False  # ok, but worth reading


SLOW_SECONDS = 60.0  # a configured model slower than this gets a WARN, not a FAIL
PING = [{"role": "user", "content": "Reply with the single word: ok"}]


class Ping(BaseModel):
    ok: bool
    model: str = ""


def check_json_contract(name: str, config: Config, client: httpx.Client | None = None) -> Check:
    """README 5.4 end to end on the backend's default model: json_mode, extraction, validation,
    one repair turn. No failover, so the verdict is about this backend alone."""
    router = Router(config, client=client)
    label = f"backend {name} json"
    result = structured_completion(
        router,
        load_prompt("ping").messages(),
        Ping,
        agent="doctor",
        backend=name,
        max_tokens=256,
        allow_failover=False,
    )
    if not result.ok:
        return Check(label, False, f"no valid JSON after repair: {result.error}")
    detail = f"{result.model} returned valid JSON"
    if result.repaired:
        detail += " (after one repair turn)"
    if not result.value.ok:
        detail += "; but ok=false"
    return Check(label, True, detail, warn=result.repaired)


def check_credentials() -> list[Check]:
    checks = []
    for cred in credential_report():
        optional = "optional" in cred.purpose or "legacy" in cred.purpose
        if cred.present:
            detail = f"{cred.source}; {cred.purpose}"
        elif optional:
            detail = f"not set (optional); {cred.purpose}"
        else:
            detail = f"missing; {cred.purpose}"
        checks.append(Check(f"env {cred.name}", cred.present or optional, detail))
    return checks


def check_config(path: Path) -> tuple[Config | None, Check]:
    try:
        config = load_config(path, strict=False)
    except ConfigError as exc:
        return None, Check("config", False, str(exc))
    if config.missing_env:
        return config, Check(
            "config", False, f"{path}: unset variables {', '.join(config.missing_env)}"
        )
    return config, Check("config", True, f"{path} (fingerprint {config.fingerprint()})")


def machine_memory_gib() -> float | None:
    try:
        import os
        import subprocess

        if os.uname().sysname == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            return int(out.stdout.strip()) / 2**30
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / 2**30
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def check_resources(config: Config) -> Check:
    """Parallel large-table experiments plus a local model exhaust small machines (seen live)."""
    import os

    memory = machine_memory_gib()
    cores = os.cpu_count() or 1
    parallel = config.run.parallel_branches
    threads = config.run.n_jobs
    detail = (
        f"{cores} cores, {memory:.0f} GiB RAM; {parallel} parallel x {threads} threads"
        if memory
        else f"{cores} cores; {parallel} parallel x {threads} threads"
    )
    warn = False
    if memory is not None and parallel > 1 and memory < 32:
        detail += (
            "; under 32 GiB run one researcher at a time on large tables (run.parallel_branches: 1)"
        )
        warn = True
    if parallel * threads > cores:
        detail += f"; {parallel * threads} threads exceed {cores} cores"
        warn = True
    return Check("resources", True, detail, warn=warn)


def check_ledger(runs_root: Path) -> Check:
    path = RunPaths(runs_root, "_doctor").ledger_path
    try:
        with Ledger(path) as ledger:
            version = ledger.schema_version
            n_runs = len(ledger.list_runs())
    except Exception as exc:  # noqa: BLE001 - any failure is the finding
        return Check("ledger", False, f"{path}: {exc}")
    return Check("ledger", True, f"{path} schema v{version}, {n_runs} runs recorded")


def check_backend(name: str, config: Config, client: httpx.Client | None = None) -> list[Check]:
    """List the backend's models, then actually complete on every configured model.

    The list alone is not enough: the NVIDIA hub advertises models that return 404 for a
    given account, and queued models can take minutes. Only a real completion proves a
    (backend, model) pair is usable.
    """
    cfg = config.backends[name]
    backend = Backend.from_config(name, cfg)
    wanted = Router(config, client=client).configured_models()[name]
    try:
        available = list_models(backend, client=client)
    except LLMError as exc:
        return [Check(f"backend {name} models", False, str(exc))]
    missing = [m for m in wanted if m not in available]
    if missing:
        checks = [
            Check(
                f"backend {name} models",
                False,
                f"{len(available)} served; configured but NOT served: {', '.join(missing)}",
            )
        ]
    else:
        checks = [
            Check(
                f"backend {name} models",
                True,
                f"{len(available)} served; all {len(wanted)} configured present",
            )
        ]

    def ping(model: str) -> Check:
        label = f"backend {name} {model}"
        try:
            result = complete(
                PING, backend=backend, model=model, max_tokens=64, temperature=0.0, client=client
            )
        except LLMError as exc:
            return Check(label, False, str(exc))
        if not (result.content.strip() or result.reasoning.strip()):
            return Check(
                label,
                False,
                f"empty response after {result.latency_s:.1f}s (finish={result.finish_reason})",
            )
        kind = "reasoning model, " if result.reasoning else ""
        detail = (
            f"answered in {result.latency_s:.1f}s ({kind}"
            f"{result.prompt_tokens}+{result.completion_tokens} tokens)"
        )
        if result.latency_s > SLOW_SECONDS:
            detail += f"; slower than {SLOW_SECONDS:.0f}s, expect long iterations"
        return Check(label, True, detail, warn=result.latency_s > SLOW_SECONDS)

    with ThreadPoolExecutor(max_workers=max(1, min(cfg.max_concurrency, len(wanted)))) as pool:
        checks.extend(pool.map(ping, wanted))
    checks.append(check_json_contract(name, config, client))
    return checks


def check_kaggle(config: Config, client: httpx.Client | None = None) -> list[Check]:
    slug = config.competition.slug
    try:
        with KaggleClient.from_env(client=client) as kaggle:
            info = kaggle.competition(slug)
            auth_mode = kaggle.auth_mode
    except KaggleError as exc:
        return [Check("kaggle competition", False, str(exc))]
    checks = [
        Check(
            "kaggle competition",
            True,
            f"{info.title!r} via {auth_mode} auth; deadline {info.deadline}; "
            f"{info.max_daily_submissions} submissions/day",
        )
    ]
    try:
        spec = resolve_metric(info.metric_name, override=config.competition.metric)
    except UnsupportedMetricError as exc:
        checks.append(Check("kaggle metric", False, str(exc)))
    else:
        direction = "higher is better" if spec.greater_is_better else "lower is better"
        checks.append(
            Check("kaggle metric", True, f"{info.metric_name!r} -> {spec.key} ({direction})")
        )
    if not info.user_has_entered:
        checks.append(
            Check(
                "kaggle entry",
                True,
                "you have not accepted the rules on the website; uploads will be rejected",
                warn=True,
            )
        )
    if info.submissions_disabled:
        checks.append(Check("kaggle submissions", False, "submissions are disabled"))
    return checks


def run_doctor(
    config_path: Path | None = None,
    *,
    runs_root: Path = Path("runs"),
    transport: httpx.BaseTransport | None = None,
    console: Console | None = None,
) -> int:
    console = console or Console()
    client = httpx.Client(transport=transport) if transport is not None else None
    checks: list[Check] = check_credentials()
    path = config_path or DEFAULT_CONFIG
    config, config_check = check_config(path)
    checks.append(config_check)
    checks.append(check_ledger(runs_root))
    if config is not None:
        checks.append(check_resources(config))
    if config is not None:
        names = list(config.backends)
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            for result in pool.map(lambda n: check_backend(n, config, client), names):
                checks.extend(result)
        checks.extend(check_kaggle(config, client))
    if client is not None:
        client.close()

    table = Table(title="aac doctor", show_lines=False)
    table.add_column("status", width=6)
    table.add_column("check", style="bold")
    table.add_column("detail", overflow="fold")
    for c in checks:
        if not c.ok:
            status = "[red]FAIL[/red]"
        elif c.warn:
            status = "[yellow]WARN[/yellow]"
        else:
            status = "[green]OK[/green]"
        table.add_row(status, c.name, c.detail)
    console.print(table)
    failed = [c for c in checks if not c.ok]
    if failed:
        console.print(f"[red]{len(failed)} check(s) failed.[/red]")
        return 1
    console.print("[green]All checks passed.[/green]")
    return 0
