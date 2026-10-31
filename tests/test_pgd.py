import numpy as np
import pandas as pd

from src.models.mlp import train_mlp
from src.attacks.pgd import pgd_attack


def test_pgd_attack():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.random((20, 80)))
    y = pd.Series([0, 1] * 10)

    model = train_mlp(X, y, epochs=1)
    X_adv = pgd_attack(model, X, y)

    assert X_adv.shape == X.shape
    assert np.isfinite(X_adv).all()
    assert X_adv.min() >= 0
    assert X_adv.max() <= 1
    assert np.max(np.abs(X_adv - X.to_numpy())) <= 0.030001


def test_pgd_allowed_features():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.random((20, 80)))
    y = pd.Series([0, 1] * 10)

    model = train_mlp(X, y, epochs=1)

    X_adv = pgd_attack(
        model, X, y,
        epsilon=0.03,
        allowed_features=[0, 1]
    )

    assert np.array_equal(
        X_adv[:, 2:],
        X.to_numpy(dtype=np.float32)[:, 2:]
    )