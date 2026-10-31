
import numpy as np
import pandas as pd
import pytest

from src.attacks.backdoor import apply_backdoor


def sample_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.random((20, 3)),
        columns=["feature_1", "feature_2", "feature_3"]
    )
    y = np.array([0, 1] * 10)
    return X, y


def test_backdoor_poisoning():
    X, y = sample_data()
    original_X = X.copy()
    original_y = y.copy()

    X_p, y_p, indices = apply_backdoor(
        X, y, 0.2,
        ["feature_1", "feature_2"],
        [0.5, 0.8],
        seed=42
    )

    assert len(indices) == 4
    assert np.all(y[indices] == 1)
    assert np.all(y_p[indices] == 0)
    assert np.sum(y != y_p) == 4

    assert np.all(X_p.iloc[indices, 0] == 0.5)
    assert np.all(X_p.iloc[indices, 1] == 0.8)

    assert X.equals(original_X)
    assert np.array_equal(y, original_y)


def test_unchanged_features():
    X, y = sample_data()

    X_p, y_p, indices = apply_backdoor(
        X, y, 0.2, ["feature_1"], [0.5]
    )

    assert X_p["feature_2"].equals(X["feature_2"])
    assert X_p["feature_3"].equals(X["feature_3"])

    other = np.setdiff1d(np.arange(len(y)), indices)
    assert np.array_equal(y_p[other], y[other])


def test_reproducibility():
    X, y = sample_data()

    X1, y1, i1 = apply_backdoor(
        X, y, 0.2, ["feature_1"], [0.5], seed=42
    )

    X2, y2, i2 = apply_backdoor(
        X, y, 0.2, ["feature_1"], [0.5], seed=42
    )

    assert X1.equals(X2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(i1, i2)


def test_numpy_input():
    X, y = sample_data()

    X_p, y_p, indices = apply_backdoor(
        X.to_numpy(), y, 0.2, [0], [0.5]
    )

    assert isinstance(X_p, np.ndarray)
    assert len(indices) == 4
    assert np.all(X_p[indices, 0] == 0.5)


def test_zero_poisoning():
    X, y = sample_data()

    X_p, y_p, indices = apply_backdoor(
        X, y, 0, ["feature_1"], [0.5]
    )

    assert len(indices) == 0
    assert X_p.equals(X)
    assert np.array_equal(y_p, y)


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_invalid_rates(rate):
    X, y = sample_data()

    with pytest.raises(ValueError):
        apply_backdoor(
            X, y, rate, ["feature_1"], [0.5]
        )


def test_insufficient_malicious_samples():
    X, y = sample_data()

    with pytest.raises(ValueError):
        apply_backdoor(
            X, y, 0.6, ["feature_1"], [0.5]
        )


def test_invalid_trigger_feature():
    X, y = sample_data()

    with pytest.raises(ValueError):
        apply_backdoor(
            X, y, 0.2, ["unknown_feature"], [0.5]
        )


def test_invalid_trigger_value():
    X, y = sample_data()

    with pytest.raises(ValueError):
        apply_backdoor(
            X, y, 0.2, ["feature_1"], [1.5]
        )
