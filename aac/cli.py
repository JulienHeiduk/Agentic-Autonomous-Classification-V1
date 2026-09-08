"""Command-line entry point.

Subcommands mirror README section 12. Each one is a thin function that resolves secrets,
builds whatever context it needs, and delegates. Keep this file free of business logic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from rich.console import Console
from rich.logging import RichHandler

from aac import __version__
from aac.env import SecretsError, load_secrets

console = Console()
err_console = Console(stderr=True)

Handler = Callable[[argparse.Namespace], int]

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def _not_implemented(name: str) -> int:
    err_console.print(f"[yellow]aac {name}[/yellow] is not implemented yet.")
    return EXIT_FAILURE


def _load_config_or_exit(path: str):
    from pathlib import Path

    from aac.config import ConfigError, load_config

    try:
        return load_config(Path(path))
    except ConfigError as exc:
        err_console.print(f"[red]config:[/red] {exc}")
        return None


def cmd_run(args: argparse.Namespace) -> int:
    from aac.orchestrator import run

    config = _load_config_or_exit(args.config)
    if config is None:
        return EXIT_USAGE
    try:
        summary = run(
            config,
            submit=not args.no_submit,
            dry_run=args.dry_run,
            poll_timeout=args.poll_timeout,
            console=console,
        )
    except Exception as exc:  # noqa: BLE001 - the CLI boundary reports and exits
        err_console.print(f"[red]run failed:[/red] {exc.__class__.__name__}: {exc}")
        return EXIT_FAILURE
    return EXIT_OK if summary.status == "completed" else EXIT_FAILURE


def cmd_resume(args: argparse.Namespace) -> int:
    from aac.orchestrator import resume

    config = _load_config_or_exit(args.config) if args.config else None
    if args.config and config is None:
        return EXIT_USAGE
    try:
        summary = resume(
            args.run_id,
            config=config,
            submit=not args.no_submit,
            poll_timeout=args.poll_timeout,
            console=console,
        )
    except Exception as exc:  # noqa: BLE001 - the CLI boundary reports and exits
        err_console.print(f"[red]resume failed:[/red] {exc.__class__.__name__}: {exc}")
        return EXIT_FAILURE
    return EXIT_OK if summary.status == "completed" else EXIT_FAILURE


def cmd_profile(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from rich.table import Table

    from aac.agents.scout import profile_data
    from aac.exec.artifacts import RunPaths
    from aac.kaggle.api import KaggleClient
    from aac.kaggle.data import ensure_data
    from aac.models.metrics import resolve_metric

    config = _load_config_or_exit(args.config)
    if config is None:
        return EXIT_USAGE
    slug = config.competition.slug
    try:
        with KaggleClient.from_env() as kaggle:
            info = kaggle.competition(slug)
            metric = resolve_metric(info.metric_name, override=config.competition.metric)
            files = ensure_data(
                kaggle, slug, RunPaths(Path("runs"), "_").data_dir(slug), config.competition.files
            )
        train, test, sample = files.frames()
        profile = profile_data(
            train,
            test,
            sample,
            slug=slug,
            competition=config.competition,
            metric=metric,
            max_classes=config.run.max_classes,
            seed=config.run.seed,
        )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]profile failed:[/red] {exc.__class__.__name__}: {exc}")
        return EXIT_FAILURE
    if args.json:
        print(json.dumps(profile.model_dump(mode="json"), indent=2, default=str))
        return EXIT_OK
    console.print(
        f"[bold]{slug}[/bold]: {profile.n_train} train / {profile.n_test} test rows, "
        f"metric {profile.metric}, target [bold]{profile.target.name}[/bold] "
        f"({profile.target.kind}, positive={profile.target.positive_label!r}), "
        f"id {profile.id_col}, submission {profile.submission.kind}"
    )
    console.print(
        "class rates: " + ", ".join(f"{k}={v:.4f}" for k, v in profile.target.rates.items())
    )
    table = Table(title="columns")
    for col in ("name", "kind", "dtype", "unique", "miss%", "drift", "usable", "detail"):
        table.add_column(col)
    for c in profile.columns:
        detail = ""
        if c.stats:
            lo, hi, mean = (c.stats.get(k, 0) for k in ("min", "max", "mean"))
            detail = f"[{lo:.4g}, {hi:.4g}] mean {mean:.4g}"
        elif c.top_values:
            detail = ", ".join(f"{v}:{s:.2f}" for v, s in c.top_values[:4])
        table.add_row(
            c.name,
            c.kind,
            c.dtype,
            str(c.n_unique),
            f"{100 * c.missing_train:.1f}",
            "" if c.drift is None else f"{c.drift:.3f}",
            "yes" if c.usable else "no",
            detail,
        )
    console.print(table)
    for w in profile.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    return EXIT_OK


def cmd_replay(args: argparse.Namespace) -> int:
    from aac.orchestrator import replay_experiment

    config = _load_config_or_exit(args.config) if args.config else None
    if args.config and config is None:
        return EXIT_USAGE
    try:
        report = replay_experiment(args.experiment_id, config=config)
    except Exception as exc:  # noqa: BLE001 - the CLI boundary reports and exits
        err_console.print(f"[red]replay failed:[/red] {exc.__class__.__name__}: {exc}")
        return EXIT_FAILURE
    for key, value in report.items():
        console.print(f"{key}: {value}")
    return EXIT_OK if report.get("reproduced") else EXIT_FAILURE


def cmd_ledger(args: argparse.Namespace) -> int:
    from pathlib import Path

    from rich.table import Table

    from aac.exec.artifacts import RunPaths
    from aac.ledger import Ledger

    path = RunPaths(Path("runs"), "_").ledger_path
    if not path.exists():
        err_console.print(f"no ledger at {path}")
        return EXIT_FAILURE
    with Ledger(path) as ledger:
        runs = ledger.list_runs(args.slug)
        branches = ledger.best_branches(args.slug, limit=args.limit)
        experiments = ledger.best_experiments(args.slug, limit=args.limit)
        notes = [n for r in runs[:3] for n in ledger.list_notes(r["id"])]
    table = Table(title=f"runs for {args.slug}")
    for col in ("id", "status", "started_at", "finished_at", "config_hash"):
        table.add_column(col)
    for r in runs:
        table.add_row(
            *(str(r[c] or "") for c in ("id", "status", "started_at", "finished_at", "config_hash"))
        )
    console.print(table)
    table = Table(title="best branches by CV")
    for col in ("id", "run_id", "backend", "model", "status", "cv_mean", "cv_std"):
        table.add_column(col)
    for b in branches:
        table.add_row(
            *(
                str(b[c] if b[c] is not None else "")
                for c in ("id", "run_id", "backend", "model", "status", "cv_mean", "cv_std")
            )
        )
    console.print(table)
    table = Table(title="best experiments by OOF")
    cols = ("id", "agent", "model", "round", "oof_score", "cv_std", "duration", "hypothesis")
    for col in cols:
        table.add_column(col)
    for e in experiments:
        table.add_row(*(str(e[c] if e[c] is not None else "")[:60] for c in cols))
    console.print(table)
    table = Table(title="notes (latest 3 runs)")
    for col in ("run_id", "kind", "source", "text"):
        table.add_column(col)
    for n in notes:
        table.add_row(str(n["run_id"]), str(n["kind"]), str(n["source"]), str(n["text"])[:100])
    console.print(table)
    return EXIT_OK


def cmd_baseline(args: argparse.Namespace) -> int:
    from pathlib import Path

    from aac.baseline import run_baseline
    from aac.config import ConfigError, load_config

    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        err_console.print(f"[red]config:[/red] {exc}")
        return EXIT_USAGE
    try:
        run_baseline(
            config,
            submit=not args.no_submit,
            poll_timeout=args.poll_timeout,
            console=console,
        )
    except Exception as exc:  # noqa: BLE001 - the CLI boundary reports and exits
        err_console.print(f"[red]baseline failed:[/red] {exc.__class__.__name__}: {exc}")
        return EXIT_FAILURE
    return EXIT_OK


def cmd_submit(args: argparse.Namespace) -> int:
    from aac.orchestrator import submit_run

    config = _load_config_or_exit(args.config)
    if config is None:
        return EXIT_USAGE
    try:
        score = submit_run(config, args.run_id, poll_timeout=args.poll_timeout)
    except Exception as exc:  # noqa: BLE001 - the CLI boundary reports and exits
        err_console.print(f"[red]submit failed:[/red] {exc.__class__.__name__}: {exc}")
        return EXIT_FAILURE
    console.print(f"run {args.run_id} submitted; public score {score}")
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    from pathlib import Path

    from aac.doctor import run_doctor

    return run_doctor(Path(args.config) if args.config else None, console=console)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aac",
        description=(
            "Agentic Autonomous Classification: unattended tabular classification for Kaggle."
        ),
    )
    parser.add_argument("--version", action="version", version=f"aac {__version__}")
    parser.add_argument(
        "--secrets",
        metavar="PATH",
        help="secrets file to load (default: $AAC_SECRETS_FILE or ./secrets.yml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("run", help="start a new run from a config file")
    p.add_argument("--config", required=True, metavar="YAML")
    p.add_argument("--dry-run", action="store_true", help="plan and profile only, no training")
    p.add_argument("--no-submit", action="store_true", help="never upload to Kaggle")
    p.add_argument("--poll-timeout", type=float, default=600.0, metavar="SECONDS")
    p.set_defaults(handler=cmd_run)

    p = sub.add_parser("resume", help="continue a crashed or stopped run from the ledger")
    p.add_argument("--run-id", required=True)
    p.add_argument("--config", metavar="YAML", help="override the config recorded in the ledger")
    p.add_argument("--no-submit", action="store_true", help="never upload to Kaggle")
    p.add_argument("--poll-timeout", type=float, default=600.0, metavar="SECONDS")
    p.set_defaults(handler=cmd_resume)

    p = sub.add_parser("profile", help="run the Scout only and print the data profile")
    p.add_argument("--config", required=True, metavar="YAML")
    p.add_argument("--json", action="store_true", help="print the full profile as JSON")
    p.set_defaults(handler=cmd_profile)

    p = sub.add_parser("replay", help="re-execute a stored experiment and check it reproduces")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--config", metavar="YAML", help="override the config recorded in the ledger")
    p.set_defaults(handler=cmd_replay)

    p = sub.add_parser("ledger", help="show what has been tried for a competition, sorted by CV")
    p.add_argument("--slug", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(handler=cmd_ledger)

    p = sub.add_parser(
        "baseline", help="download data, submit a constant-prediction baseline, record the score"
    )
    p.add_argument("--config", required=True, metavar="YAML")
    p.add_argument("--no-submit", action="store_true", help="write and validate only")
    p.add_argument("--poll-timeout", type=float, default=600.0, metavar="SECONDS")
    p.set_defaults(handler=cmd_baseline)

    p = sub.add_parser(
        "submit", help="upload a finished run's submission.csv (after a quota reset)"
    )
    p.add_argument("--config", required=True, metavar="YAML")
    p.add_argument("--run-id", required=True)
    p.add_argument("--poll-timeout", type=float, default=600.0, metavar="SECONDS")
    p.set_defaults(handler=cmd_submit)

    p = sub.add_parser("doctor", help="check env vars, LLM backends, and Kaggle auth")
    p.add_argument("--config", metavar="YAML", help="config to check (default: configs/s6e9.yaml)")
    p.set_defaults(handler=cmd_doctor)

    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, show_path=verbose, rich_tracebacks=True)],
        force=True,
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE
    _configure_logging(args.verbose)
    try:
        load_secrets(args.secrets)
    except SecretsError as exc:
        err_console.print(f"[red]secrets:[/red] {exc}")
        return EXIT_USAGE
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
