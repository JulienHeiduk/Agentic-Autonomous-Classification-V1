import numpy as np
import pandas as pd
import pytest

from aac.models.target import TargetError, choose_positive, infer_target_encoding


@pytest.mark.parametrize(
    ("values", "positive"),
    [
        (["No", "Yes", "No"], "Yes"),
        (["yes", "no"], "yes"),
        (["TRUE", "FALSE"], "TRUE"),
        ([0, 1, 1], 1),
        ([True, False], True),
        ([2, 5], 5),
        (["cat", "dog"], "dog"),
        (["neg", "other"], "other"),
    ],
)
def test_binary_positive_label(values, positive):
    enc = infer_target_encoding(pd.Series(values))
    assert enc.is_binary and enc.positive_label == positive
    codes = enc.encode(pd.Series(values))
    assert set(codes) <= {0, 1}
    assert all((c == 1) == (v == positive) for c, v in zip(codes, values, strict=True))
    assert list(enc.decode(codes)) == values


def test_multiclass_sorted_and_roundtrip():
    y = pd.Series(["b", "a", "c", "a"])
    enc = infer_target_encoding(y)
    assert enc.classes == ("a", "b", "c") and not enc.is_binary and enc.positive_label is None
    assert enc.encode(y).tolist() == [1, 0, 2, 0]
    assert list(enc.decode(np.array([2, 0]))) == ["c", "a"]


def test_errors():
    with pytest.raises(TargetError, match="single class"):
        infer_target_encoding(pd.Series([1, 1, 1]))
    with pytest.raises(TargetError, match="missing"):
        infer_target_encoding(pd.Series([1, None, 0]))
    enc = infer_target_encoding(pd.Series(["No", "Yes"]))
    with pytest.raises(TargetError, match="not in"):
        enc.encode(pd.Series(["No", "Maybe"]))
    with pytest.raises(TargetError):
        choose_positive(["a", "b", "c"])
