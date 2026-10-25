import numpy as np
import pandas as pd

from src.defences.adversarial_training import adversarial_train_mlp
from src.models.mlp import predict_mlp


def test_adversarial_training():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.random((40, 80)))
    y = pd.Series([0, 1] * 20)

    model = adversarial_train_mlp(
        X, y, epsilon=0.01, n_samples=10
    )

    predictions = predict_mlp(model, X)

    assert len(predictions) == len(y)
    assert np.isin(predictions, [0, 1]).all()
