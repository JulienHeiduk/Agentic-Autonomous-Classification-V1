"""Competition data cache under ``runs/_data/{slug}/`` (README section 6).

    archive.zip       the bundle as downloaded
    raw/              extracted CSVs
    train.parquet, test.parquet, sample_submission.parquet

Downloaded once, never inside a loop. Parquet copies make every later load fast and preserve
dtypes across runs.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aac.config import FilesConfig
from aac.kaggle.api import KaggleClient

log = logging.getLogger(__name__)

ROLES = ("train", "test", "sample_submission")
MISSING_KEY = "__missing__"


class DataLayoutError(RuntimeError):
    """The bundle does not contain a recognisable train/test/sample_submission trio."""


@dataclass(frozen=True)
class DataFiles:
    slug: str
    data_dir: Path
    archive: Path
    raw_dir: Path
    train: Path
    test: Path
    sample_submission: Path

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            pd.read_parquet(self.train),
            pd.read_parquet(self.test),
            pd.read_parquet(self.sample_submission),
        )


INPUT_SUFFIXES = (".csv", ".parquet")


def resolve_files(raw_dir: Path, files: FilesConfig | None) -> dict[str, Path]:
    """Config overrides first (path or glob under raw_dir), the filename heuristic otherwise."""
    inputs = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in INPUT_SUFFIXES)
    resolved = assign_roles(inputs, strict=False)
    for role in ROLES:
        pattern = getattr(files, role, None) if files else None
        if not pattern:
            continue
        matches = sorted(raw_dir.glob(pattern))
        if len(matches) != 1:
            raise DataLayoutError(
                f"competition.files.{role}={pattern!r} matched {len(matches)} files under {raw_dir}"
            )
        resolved[role] = matches[0]
    missing = [r for r in ROLES if r not in resolved]
    if missing:
        raise DataLayoutError(
            f"could not resolve {missing} among {[p.name for p in inputs]}; set competition.files"
        )
    return resolved


def assign_roles(csvs: list[Path], *, strict: bool = True) -> dict[str, Path]:
    roles: dict[str, list[Path]] = {r: [] for r in ROLES}
    for path in csvs:
        name = path.name.lower()
        if "sample" in name and "submission" in name:
            roles["sample_submission"].append(path)
        elif name.startswith("train"):
            roles["train"].append(path)
        elif name.startswith("test"):
            roles["test"].append(path)
    problems = []
    for role, found in roles.items():
        if not found:
            problems.append(f"no {role} file")
        elif len(found) > 1:
            problems.append(f"several {role} files: {[p.name for p in found]}")
    if problems and strict:
        raise DataLayoutError("; ".join(problems) + f" among {[p.name for p in csvs]}")
    return {role: found[0] for role, found in roles.items() if len(found) == 1}


def _safe_extract(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = member.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise DataLayoutError(f"refusing unsafe archive member {name!r}")
        zf.extractall(target)


def _to_parquet(source: Path, parquet: Path) -> None:
    df = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    tmp = parquet.with_name(parquet.name + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, parquet)
    log.info("cached %s -> %s (%d rows, %d cols)", source.name, parquet.name, len(df), df.shape[1])


def ensure_dataset(client: KaggleClient, ref: str, datasets_dir: Path, file: str | None) -> Path:
    """Download and cache one tabular file of a Kaggle dataset as parquet.

    Layout: ``{datasets_dir}/{owner}__{slug}/`` with the download (``archive.zip`` or the raw
    file), an extracted ``raw/`` directory, and ``{file stem}.parquet``.
    """
    owner, _, name = ref.partition("/")
    data_dir = Path(datasets_dir) / f"{owner}__{name}"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    if not raw_dir.is_dir() or not any(raw_dir.iterdir()):
        download = data_dir / "download.bin"
        if not download.exists():
            log.info("downloading dataset %s", ref)
            client.download_dataset(ref, download)
        staging = Path(tempfile.mkdtemp(prefix=".raw.", dir=data_dir))
        try:
            if zipfile.is_zipfile(download):
                _safe_extract(download, staging)
            else:
                names = [f.name for f in client.list_dataset_files(ref)]
                single = names[0] if len(names) == 1 else (file or "data.csv")
                shutil.copyfile(download, staging / single)
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            os.replace(staging, raw_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    candidates = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES
    )
    if file is not None:
        candidates = [
            p for p in candidates if p.name == file or str(p.relative_to(raw_dir)) == file
        ]
    if not candidates:
        raise DataLayoutError(f"dataset {ref}: no tabular file{' ' + file if file else ''} found")
    if len(candidates) > 1:
        raise DataLayoutError(
            f"dataset {ref} has several tabular files {[c.name for c in candidates]}; "
            "set competition.extra_train[].file"
        )
    source = candidates[0]
    target = data_dir / f"{source.stem}.parquet"
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        _to_parquet(source, target)
    return target


def _dedupe_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            # rounded: a CSV round trip keeps 16 significant digits, not the exact float
            out[col] = pd.to_numeric(s, errors="coerce").astype("float64").round(8)
        else:
            out[col] = s.astype("object").where(s.notna(), MISSING_KEY).astype(str)
    return out


def load_extra_train(
    path: Path,
    train: pd.DataFrame,
    *,
    target: str,
    id_col: str | None,
    rename: dict[str, str] | None = None,
    dedupe: bool = True,
    first_id: int = -1,
) -> pd.DataFrame:
    """Align a dataset's rows with the competition's train frame: same columns in the same
    order and dtypes, fresh negative ids, rows identical to a synthetic row dropped."""
    extra = pd.read_parquet(path)
    if rename:
        extra = extra.rename(columns=rename)
        dup = extra.columns[extra.columns.duplicated()].tolist()
        if dup:
            raise DataLayoutError(f"{path.name}: rename {rename} produces duplicate columns {dup}")
    wanted = [c for c in train.columns if c != id_col]
    missing = [c for c in wanted if c not in extra.columns]
    if missing:
        raise DataLayoutError(f"{path.name} lacks the training columns {missing}")
    dropped = [c for c in extra.columns if c not in wanted]
    if dropped:
        log.info("%s: ignoring columns %s", path.name, dropped)
    extra = extra[wanted].copy()
    for col in wanted:
        kind = train[col].dtype
        if pd.api.types.is_numeric_dtype(kind):
            extra[col] = pd.to_numeric(extra[col], errors="coerce").astype(kind, errors="ignore")
        else:
            extra[col] = extra[col].astype(train[col].dtype, errors="ignore")
    before = len(extra)
    if dedupe:
        # Compare on canonical keys: floats as floats (NaN matches NaN in a merge), everything
        # else as strings with one missing marker, so None and NaN read as the same value.
        synthetic = _dedupe_keys(train[wanted]).drop_duplicates()
        merged = _dedupe_keys(extra).merge(synthetic.assign(_seen=1), on=wanted, how="left")
        extra = extra.loc[merged["_seen"].isna().to_numpy()].reset_index(drop=True)
    if id_col is not None:
        extra.insert(0, id_col, np.arange(first_id, first_id - len(extra), -1))
    extra = extra[list(train.columns)]
    log.info(
        "extra training rows from %s: %d kept of %d (%d duplicates of synthetic rows)",
        path.name,
        len(extra),
        before,
        before - len(extra),
    )
    return extra


def ensure_data(
    client: KaggleClient, slug: str, data_dir: Path, files: FilesConfig | None = None
) -> DataFiles:
    """Download, extract, and parquet-cache the bundle if any step is missing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "archive.zip"
    raw_dir = data_dir / "raw"

    if not archive.exists():
        log.info("downloading %s bundle", slug)
        client.download_all(slug, archive)

    if not raw_dir.is_dir() or not any(raw_dir.iterdir()):
        staging = Path(tempfile.mkdtemp(prefix=".raw.", dir=data_dir))
        try:
            _safe_extract(archive, staging)
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            os.replace(staging, raw_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    roles = resolve_files(raw_dir, files)
    parquet: dict[str, Path] = {}
    for role, source in roles.items():
        target = data_dir / f"{role}.parquet"
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            _to_parquet(source, target)
        parquet[role] = target

    return DataFiles(
        slug=slug,
        data_dir=data_dir,
        archive=archive,
        raw_dir=raw_dir,
        train=parquet["train"],
        test=parquet["test"],
        sample_submission=parquet["sample_submission"],
    )
