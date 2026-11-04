import numpy as np
import pandas as pd

from src.defences.adversarial_training import adversarial_train_mlp, adversarial_train_rf
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


def test_adversarial_training_with_pgd():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.random((20, 80)))
    y = pd.Series([0, 1] * 10)

    model = adversarial_train_mlp(
        X, y,
        n_samples=10,
        attack_type="pgd"
    )

    predictions = predict_mlp(model, X)

    assert len(predictions) == len(y)
    assert np.isin(predictions, [0, 1]).all()


def test_adversarial_training_rf():
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.random((40, 80)),
        columns=[f"feature_{i}" for i in range(80)]
    )
    y = pd.Series([0, 1] * 20)

    hopskipjump_examples = X.iloc[[1, 3]].to_numpy()
    transferred_examples = X.iloc[[5, 7]].to_numpy()

    model = adversarial_train_rf(
        X,
        y,
        hopskipjump_examples,
        transferred_examples
    )

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert np.isin(predictions, [0, 1]).all()
