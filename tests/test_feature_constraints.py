import numpy as np
import pandas as pd

from src.models.mlp import train_mlp
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.hopskipjump import hopskipjump_attack
from src.models.random_forest import train_random_forest


def test_fgsm_preserves_fixed_features():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.random((20, 80)),
        columns=[f"feature_{i}" for i in range(80)]
    )
    y = pd.Series([0, 1] * 10)

    allowed = ["feature_5", "feature_6"]

    model = train_mlp(X, y, epochs=1)

    X_adv = fgsm_attack(
        model, X, y,
        epsilon=0.01,
        allowed_features=allowed
    )

    fixed = [c for c in X.columns if c not in allowed]

    np.testing.assert_array_equal(
        X_adv[:, [X.columns.get_loc(c) for c in fixed]],
        X[fixed].to_numpy(dtype=np.float32)
    )



def test_pgd_preserves_fixed_features():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.random((20, 80)),
        columns=[f"feature_{i}" for i in range(80)]
    )
    y = pd.Series([0, 1] * 10)

    allowed = ["feature_5", "feature_6"]

    model = train_mlp(X, y, epochs=1)

    X_adv = pgd_attack(
        model, X, y,
        epsilon=0.01,
        allowed_features=allowed
    )

    fixed = [c for c in X.columns if c not in allowed]

    np.testing.assert_array_equal(
        X_adv[:, [X.columns.get_loc(c) for c in fixed]],
        X[fixed].to_numpy(dtype=np.float32)
    )

def test_hopskipjump_preserves_fixed_features():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.random((20, 80)),
        columns=[f"feature_{i}" for i in range(80)]
    )
    y = pd.Series([0, 1] * 10)

    model = train_random_forest(X, y)

    allowed = ["feature_5", "feature_6"]

    X_adv = hopskipjump_attack(
        model,
        X.iloc[:2],
        allowed_features=allowed
    )

    fixed = [c for c in X.columns if c not in allowed]

    np.testing.assert_array_equal(
        X_adv[:, [X.columns.get_loc(c) for c in fixed]],
        X.iloc[:2][fixed].to_numpy(dtype=np.float32)
    )
