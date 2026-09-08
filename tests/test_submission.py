from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from aac.kaggle.api import Submission
from aac.kaggle.submission import (
    SubmissionFormatError,
    build_submission,
    submissions_remaining,
    submissions_today,
    target_from_sample,
    validate_submission,
    write_submission,
)

SAMPLE = pd.DataFrame({"id": [10, 11, 12], "purchased": [0.5, 0.5, 0.5]})


def test_build_and_validate_proba():
    sub = build_submission(SAMPLE, np.array([0.1, 0.2, 0.3]))
    assert list(sub.columns) == ["id", "purchased"] and sub["purchased"].tolist() == [0.1, 0.2, 0.3]
    validate_submission(sub, SAMPLE)
    with pytest.raises(SubmissionFormatError, match="shape"):
        build_submission(SAMPLE, np.array([0.1, 0.2]))


def problems(sub, **kw):
    with pytest.raises(SubmissionFormatError) as exc:
        validate_submission(sub, SAMPLE, **kw)
    return exc.value.problems


def test_validation_catches_each_defect():
    assert any("columns" in p for p in problems(SAMPLE.rename(columns={"purchased": "target"})))
    assert any("rows" in p for p in problems(SAMPLE.iloc[:2]))
    shuffled = SAMPLE.iloc[[2, 0, 1]].reset_index(drop=True)
    assert any("order" in p for p in problems(shuffled))
    nan = SAMPLE.copy()
    nan.loc[0, "purchased"] = np.nan
    assert any("NaN" in p for p in problems(nan))
    big = SAMPLE.copy()
    big.loc[1, "purchased"] = 1.5
    assert any("outside [0, 1]" in p for p in problems(big))
    text = SAMPLE.copy()
    text["purchased"] = ["a", "b", "c"]
    assert any("non-numeric" in p for p in problems(text))
    labels = SAMPLE.copy()
    labels["purchased"] = ["yes", "no", "maybe"]
    assert any("maybe" in p for p in problems(labels, kind="label", allowed_labels={"yes", "no"}))
    validate_submission(labels, SAMPLE, kind="label", allowed_labels={"yes", "no", "maybe"})


def test_id_comparison_ignores_dtype():
    sub = SAMPLE.copy()
    sub["id"] = sub["id"].astype(str)
    validate_submission(sub, SAMPLE)


def test_write_submission(tmp_path):
    path = write_submission(tmp_path / "sub" / "submission.csv", SAMPLE)
    assert path.read_text() == "id,purchased\n10,0.5\n11,0.5\n12,0.5\n"


def test_target_from_sample():
    assert target_from_sample(SAMPLE, ["id", "age", "purchased"]) == "purchased"
    with pytest.raises(ValueError, match="competition.target"):
        target_from_sample(SAMPLE, ["id", "age"])


def sub(date):
    return Submission(1, date, "", "f", "COMPLETE", None, None, "")


def test_submissions_today_uses_utc_day():
    now = datetime(2026, 9, 5, 23, 30, tzinfo=UTC)
    subs = [
        sub("2026-09-05T00:00:01Z"),
        sub("2026-09-05T22:00:00.123Z"),
        sub("2026-09-06T00:00:00Z"),
        sub("2026-09-04T23:59:59Z"),
        sub("2026-09-05T23:00:00+02:00"),  # 21:00 UTC, counts
        sub(None),
        sub("garbage"),
    ]
    assert submissions_today(subs, now) == 3
    assert submissions_remaining(subs, 5, now) == 2
    assert submissions_remaining(subs, 2, now) == 0
