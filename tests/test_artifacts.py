import json
from datetime import datetime

from aac.exec.artifacts import (
    RunPaths,
    allocate_run_dir,
    atomic_write_json,
    atomic_write_text,
    new_run_id,
)


def test_run_id_format():
    assert new_run_id(datetime(2026, 9, 5, 14, 32)) == "20260905-1432"


def test_allocate_run_dir_suffixes_on_collision(tmp_path):
    now = datetime(2026, 9, 5, 14, 32)
    a = allocate_run_dir(tmp_path, now)
    b = allocate_run_dir(tmp_path, now)
    c = allocate_run_dir(tmp_path, now)
    assert [a.run_id, b.run_id, c.run_id] == ["20260905-1432", "20260905-1432-2", "20260905-1432-3"]
    assert a.branches_dir.is_dir() and a.run_dir.parent == tmp_path


def test_layout():
    p = RunPaths(root=__import__("pathlib").Path("runs"), run_id="r")
    assert str(p.ledger_path) == "runs/ledger.db"
    assert str(p.branch_dir("b1")) == "runs/r/branches/b1"
    assert str(p.data_dir("slug")) == "runs/_data/slug"
    assert p.config_snapshot.name == "config.resolved.yaml"
    assert p.folds_path.name == "folds.npy" and p.profile_json.name == "profile.json"


def test_atomic_writes_leave_no_temp_files(tmp_path):
    target = tmp_path / "deep" / "metrics.json"
    atomic_write_json(target, {"b": 1, "a": [1, 2]})
    assert json.loads(target.read_text()) == {"a": [1, 2], "b": 1}
    atomic_write_text(target, "replaced")
    assert target.read_text() == "replaced"
    assert [f.name for f in target.parent.iterdir()] == ["metrics.json"]
