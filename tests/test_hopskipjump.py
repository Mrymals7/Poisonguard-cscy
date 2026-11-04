import numpy as np
import pandas as pd

from src.models.random_forest import train_random_forest
from src.attacks.hopskipjump import hopskipjump_attack


def test_hopskipjump():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.random((40, 80)))
    y = pd.Series([0, 1] * 20)

    model = train_random_forest(X, y)

    X_attack = X.head(1)
    X_adv = hopskipjump_attack(model, X_attack)

    assert X_adv.shape == X_attack.shape
    assert np.isfinite(X_adv).all()
    assert X_adv.min() >= 0
    assert X_adv.max() <= 1