import numpy as np
import pandas as pd
import pytest

import preprocess


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    X = rng.integers(0, 2, size=(n, 12)).astype(float)
    X[5, 3] = np.nan
    labels = np.where(rng.random(n) < 0.35, "S", "B")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["class"] = labels
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)
    return path


def test_splits_disjoint_and_stratified(csv_path):
    s, _ = preprocess.prepare(csv_path)
    a, b, c = set(s.X_train.index), set(s.X_val.index), set(s.X_test.index)
    assert not (a & b) and not (a & c) and not (b & c)
    ratios = [s.y_train.mean(), s.y_val.mean(), s.y_test.mean()]
    assert max(ratios) - min(ratios) < 0.05


def test_no_nans_and_binary_mask(csv_path):
    s, art = preprocess.prepare(csv_path)
    assert not s.X_train.isna().any().any()
    assert not s.X_test.isna().any().any()
    assert art["binary_mask"].all()


def test_artifact_roundtrip(csv_path, tmp_path):
    s, art = preprocess.prepare(csv_path)
    p = tmp_path / "pre.joblib"
    preprocess.save_artifact(art, p)
    loaded = preprocess.load_artifact(p)
    raw = pd.read_csv(csv_path).drop(columns=["class"]).head(10)
    a = preprocess.transform_inputs(raw, art)
    b = preprocess.transform_inputs(raw, loaded)
    pd.testing.assert_frame_equal(a, b)


def test_missing_columns_rejected(csv_path):
    _, art = preprocess.prepare(csv_path)
    bad = pd.DataFrame({"f0": [1, 0]})
    with pytest.raises(ValueError):
        preprocess.transform_inputs(bad, art)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        preprocess.load_dataset("nope.csv")


def test_unknown_label(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"f0": [0, 1], "class": ["B", "weird"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        preprocess.prepare(p)


def test_duplicate_rows_removed(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    X = rng.integers(0, 2, size=(n, 10)).astype(float)
    labels = np.where(rng.random(n) < 0.4, "S", "B")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["class"] = labels
    df_with_dups = pd.concat([df, df.head(20)], ignore_index=True)
    p = tmp_path / "dups.csv"
    df_with_dups.to_csv(p, index=False)
    s, _ = preprocess.prepare(p)
    total = len(s.X_train) + len(s.X_val) + len(s.X_test)
    assert total <= n
