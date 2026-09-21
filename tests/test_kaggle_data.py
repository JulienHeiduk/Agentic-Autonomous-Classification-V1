import zipfile
from pathlib import Path

import httpx
import pandas as pd
import pytest

from aac.config import FilesConfig
from aac.kaggle.api import KaggleClient
from aac.kaggle.data import (
    DataLayoutError,
    assign_roles,
    ensure_data,
    ensure_dataset,
    load_extra_train,
    resolve_files,
)
from tests.kaggle_fake import SLUG, FakeKaggle


def test_ensure_data_downloads_once_and_caches_parquet(tmp_path):
    fake = FakeKaggle()
    kc = KaggleClient(access_token="t", client=httpx.Client(transport=fake.transport))
    files = ensure_data(kc, SLUG, tmp_path / "_data" / SLUG)
    assert files.archive.exists() and (files.raw_dir / "train.csv").exists()
    train, test, sample = files.frames()
    assert list(train.columns) == ["id", "age", "income", "region", "purchased"]
    assert len(train) == 60 and len(test) == 25 and len(sample) == 25
    assert list(sample.columns) == ["id", "purchased"]
    n_calls = len(fake.calls)
    again = ensure_data(kc, SLUG, tmp_path / "_data" / SLUG)
    assert again == files and len(fake.calls) == n_calls, "second call must not touch the network"


def test_extraction_is_recreated_when_raw_missing(tmp_path):
    fake = FakeKaggle()
    kc = KaggleClient(access_token="t", client=httpx.Client(transport=fake.transport))
    data_dir = tmp_path / SLUG
    files = ensure_data(kc, SLUG, data_dir)
    for p in files.raw_dir.iterdir():
        p.unlink()
    files.raw_dir.rmdir()
    n_calls = len(fake.calls)
    ensure_data(kc, SLUG, data_dir)
    assert (files.raw_dir / "test.csv").exists() and len(fake.calls) == n_calls


def test_assign_roles_errors():
    with pytest.raises(DataLayoutError, match="no sample_submission"):
        assign_roles([Path("train.csv"), Path("test.csv")])
    with pytest.raises(DataLayoutError, match="several train"):
        assign_roles(
            [
                Path("train.csv"),
                Path("train_extra.csv"),
                Path("test.csv"),
                Path("sample_submission.csv"),
            ]
        )
    roles = assign_roles([Path("x/Test.csv"), Path("x/Train.csv"), Path("x/SAMPLE_SUBMISSION.csv")])
    assert {r: p.name for r, p in roles.items()} == {
        "train": "Train.csv",
        "test": "Test.csv",
        "sample_submission": "SAMPLE_SUBMISSION.csv",
    }


def test_unsafe_archive_member_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.csv", "a\n1\n")
    fake = FakeKaggle(bundle=bad.read_bytes())
    kc = KaggleClient(access_token="t", client=httpx.Client(transport=fake.transport))
    with pytest.raises(DataLayoutError, match="unsafe"):
        ensure_data(kc, SLUG, tmp_path / "d")


def test_resolve_files_override_and_parquet_inputs(tmp_path):
    raw = tmp_path / "raw"
    (raw / "data").mkdir(parents=True)
    pd.DataFrame({"id": [1], "y": [0]}).to_parquet(raw / "data" / "training_set.parquet")
    (raw / "data" / "holdout.csv").write_text("id\n2\n")
    (raw / "sample_submission.csv").write_text("id,y\n2,0\n")
    with pytest.raises(DataLayoutError, match="could not resolve"):
        resolve_files(raw, None)
    files = FilesConfig(train="data/training_set.parquet", test="data/hold*.csv")
    roles = resolve_files(raw, files)
    assert roles["train"].name == "training_set.parquet" and roles["test"].name == "holdout.csv"
    assert roles["sample_submission"].name == "sample_submission.csv"
    with pytest.raises(DataLayoutError, match="matched 0"):
        resolve_files(raw, FilesConfig(train="nope*.csv"))


def test_ensure_data_with_override_reads_parquet(tmp_path):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        tr = io.BytesIO()
        pd.DataFrame({"id": [1, 2], "y": [0, 1]}).to_parquet(tr)
        zf.writestr("inputs/training_set.parquet", tr.getvalue())
        zf.writestr("inputs/holdout.csv", "id\n3\n")
        zf.writestr("sample_submission.csv", "id,y\n3,0\n")
    fake = FakeKaggle(bundle=buf.getvalue())
    kc = KaggleClient(access_token="t", client=httpx.Client(transport=fake.transport))
    files = ensure_data(
        kc,
        SLUG,
        tmp_path / SLUG,
        FilesConfig(train="inputs/training_set.parquet", test="inputs/holdout.csv"),
    )
    train, test, sample = files.frames()
    assert list(train.columns) == ["id", "y"] and len(test) == 1 and len(sample) == 1


def _original(train: pd.DataFrame) -> pd.DataFrame:
    """Ten rows copied from train and fifteen fresh ones, with the dataset's own id column."""
    fresh = train.tail(15).copy()
    fresh["x1"] = fresh["x1"] + 100.0
    orig = pd.concat([train.head(10), fresh], ignore_index=True).drop(columns=["id"])
    orig = orig.rename(columns={"x3": "level"})  # the dataset's own name for x3
    orig.insert(0, "Buyer_ID", [f"EV{i:05d}" for i in range(len(orig))])
    orig["junk"] = "ignored"
    return orig


def test_ensure_dataset_caches_and_load_extra_train_aligns(tmp_path):
    from tests.synth import make_frames

    train, _, _ = make_frames(60, 20)
    orig = _original(train)
    fake = FakeKaggle(dataset=orig.to_csv(index=False).encode(), dataset_files=["EV.csv"])
    kc = KaggleClient(client=httpx.Client(transport=fake.transport), access_token="KGAT_x")
    parquet = ensure_dataset(kc, "owner/ev-data", tmp_path / "datasets", None)
    assert parquet == tmp_path / "datasets" / "owner__ev-data" / "EV.parquet"
    assert (tmp_path / "datasets" / "owner__ev-data" / "raw" / "EV.csv").exists()
    again = ensure_dataset(kc, "owner/ev-data", tmp_path / "datasets", None)
    assert again == parquet and fake.dataset_downloads == 1, "cached, not downloaded twice"

    rename = {"level": "x3"}
    with pytest.raises(DataLayoutError, match="lacks the training columns \\['x3'\\]"):
        load_extra_train(parquet, train, target="target", id_col="id")
    extra = load_extra_train(parquet, train, target="target", id_col="id", rename=rename)
    assert list(extra.columns) == list(train.columns), "same columns in the same order"
    assert len(extra) == 15, "the ten copies of synthetic rows were dropped"
    assert extra["id"].tolist() == list(range(-1, -16, -1)), "fresh negative ids"
    assert extra["x3"].dtype == train["x3"].dtype and extra["cat"].dtype == train["cat"].dtype
    assert set(extra["target"]) <= set(train["target"])
    kept = load_extra_train(
        parquet, train, target="target", id_col="id", rename=rename, dedupe=False
    )
    assert len(kept) == 25
    no_id = load_extra_train(
        parquet, train.drop(columns=["id"]), target="target", id_col=None, rename=rename
    )
    assert list(no_id.columns) == [c for c in train.columns if c != "id"]
    later = load_extra_train(
        parquet, train, target="target", id_col="id", rename=rename, first_id=-100
    )
    assert later["id"].iloc[0] == -100
    with pytest.raises(DataLayoutError, match="duplicate columns"):
        load_extra_train(
            parquet, train, target="target", id_col="id", rename={**rename, "junk": "x1"}
        )
    with pytest.raises(DataLayoutError, match="lacks the training columns"):
        load_extra_train(
            parquet, train.assign(other=1.0), target="target", id_col="id", rename=rename
        )


def test_ensure_dataset_with_several_files_needs_a_choice(tmp_path):
    buf = __import__("io").BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", "x\n1\n")
        zf.writestr("b.csv", "x\n2\n")
    fake = FakeKaggle(dataset=buf.getvalue(), dataset_files=["a.csv", "b.csv"])
    kc = KaggleClient(client=httpx.Client(transport=fake.transport), access_token="KGAT_x")
    with pytest.raises(DataLayoutError, match="several tabular files"):
        ensure_dataset(kc, "o/two", tmp_path / "datasets", None)
    parquet = ensure_dataset(kc, "o/two", tmp_path / "datasets", "b.csv")
    assert pd.read_parquet(parquet)["x"].tolist() == [2]
    with pytest.raises(DataLayoutError, match="no tabular file"):
        ensure_dataset(kc, "o/two", tmp_path / "datasets", "c.csv")
