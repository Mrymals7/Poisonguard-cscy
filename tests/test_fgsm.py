import numpy as np
import pandas as pd

from src.models.mlp import train_mlp
from src.attacks.fgsm import fgsm_attack


def test_fgsm_with_trained_model():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.random((40, 80)))
    y = pd.Series([0, 1] * 20)

    model = train_mlp(X, y, epochs=1)
    X_adv = fgsm_attack(model, X, y)

    assert X_adv.shape == X.shape
    assert np.isfinite(X_adv).all()
    assert X_adv.min() >= 0
    assert X_adv.max() <= 1
    assert np.max(np.abs(X_adv - X.to_numpy())) <= 0.010001 
def test_fgsm_allowed_features():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.random((20, 80)))
    y = pd.Series([0, 1] * 10)

    model = train_mlp(X, y, epochs=1)

    X_adv = fgsm_attack(
        model, X, y,
        epsilon=0.01,
        allowed_features=[0, 1]
    )

    assert np.array_equal(
        X_adv[:, 2:],
        X.to_numpy(dtype=np.float32)[:, 2:]
    )