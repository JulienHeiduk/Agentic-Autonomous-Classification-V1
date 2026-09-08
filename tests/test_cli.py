import pytest

from aac.cli import EXIT_FAILURE, EXIT_USAGE, build_parser, main

SUBCOMMANDS = {
    "run": ["--config", "configs/s6e9.yaml"],
    "resume": ["--run-id", "20260905-1432"],
    "profile": ["--config", "configs/s6e9.yaml"],
    "replay": ["--experiment-id", "20260906-1200-r01-open-nvidia-r1"],
    "ledger": ["--slug", "playground-series-s6e9"],
    "doctor": [],
    "submit": ["--config", "configs/s6e9.yaml", "--run-id", "20260906-1200"],
}


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "aac 0." in capsys.readouterr().out


def test_no_command_prints_help_and_returns_usage(capsys):
    assert main([]) == EXIT_USAGE
    assert "COMMAND" in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(SUBCOMMANDS))
def test_every_readme_subcommand_parses(name):
    args = build_parser().parse_args([name, *SUBCOMMANDS[name]])
    assert args.command == name
    assert callable(args.handler)


@pytest.mark.parametrize("name", ["run", "resume", "profile", "replay", "ledger", "submit"])
def test_required_arguments_are_enforced(name):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([name])
    assert exc.value.code == 2


def test_run_flags():
    args = build_parser().parse_args(["run", "--config", "c.yaml", "--dry-run", "--no-submit"])
    assert args.dry_run and args.no_submit


def test_failing_command_reports_loudly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no secrets.yml, no ledger here; must not crash
    assert main(["replay", "--experiment-id", "x"]) == EXIT_FAILURE
    assert "replay failed" in capsys.readouterr().err


def test_bad_secrets_file_is_a_usage_error(tmp_path, capsys):
    bad = tmp_path / "secrets.yml"
    bad.write_text("- just\n- a list\n")
    assert main(["--secrets", str(bad), "replay", "--experiment-id", "x"]) == EXIT_USAGE
    assert "secrets" in capsys.readouterr().err
