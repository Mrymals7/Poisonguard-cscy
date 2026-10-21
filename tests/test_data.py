import numpy as np
import pandas as pd
from src.data import clean, split, scale, ID_COLUMNS


def make_raw_table():
    """A tiny fake raw table with the problems clean() must fix."""
    rows = [
        # Flow Duration, Flow Bytes/s, Label
        [10, 100.0, "BENIGN"],
        [20, 200.0, "Port_scan"],
        [30, np.inf, "DDoS"],        # infinite value -> should be removed
        [40, 400.0, "BENIGN"],
        [40, 400.0, "BENIGN"],       # duplicate of the row above -> should be removed
    ]
    df = pd.DataFrame(rows, columns=["Flow Duration", "Flow Bytes/s", "Label"])
    for col in ID_COLUMNS:
        df[col] = "x"                # fake identifier columns
    return df


def test_clean():
    df = clean(make_raw_table())
    assert not any(col in df.columns for col in ID_COLUMNS)
    assert "Label" not in df.columns
    assert "Port_scan" not in df["attack_type"].values
    assert "PortScan" in df["attack_type"].values
    assert set(df["label"]) <= {0, 1}
    assert not np.isinf(df["Flow Bytes/s"]).any()
    assert len(df) == 3              # 5 rows - 1 infinite - 1 duplicate


def make_clean_table(n=200):
    """A tiny fake clean table: 2 features, half benign, half attack."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "f1": rng.random(n) * 1000,
        "f2": rng.random(n) * 5,
        "label": [0, 1] * (n // 2),
        "attack_type": ["BENIGN", "DDoS"] * (n // 2),
    })


def test_split():
    train, val, test = split(make_clean_table())
    assert (len(train), len(val), len(test)) == (140, 30, 30)
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)


def test_scale():
    train, val, test = split(make_clean_table())
    train_s, val_s, test_s, scaler = scale(train, val, test)
    for part in (train_s, val_s, test_s):
        assert part[["f1", "f2"]].min().min() >= 0.0
        assert part[["f1", "f2"]].max().max() <= 1.0
    assert scaler.data_max_[0] == train["f1"].max()   # learned from train only