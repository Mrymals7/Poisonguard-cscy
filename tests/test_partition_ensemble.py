import numpy as np
import pandas as pd
from src.defences.partition_ensemble import (
    train_partition_ensemble,
    predict_partition_ensemble,
)


def test_creates_requested_number_of_models():
    X = pd.DataFrame({"feature": np.arange(60, dtype=float)})
    y = pd.Series([0, 1] * 30)

    models = train_partition_ensemble(
        X, y, n_partitions=3, seed=42, model_type="rf"
    )

    assert len(models) == 3


def test_predictions_have_correct_shape_and_binary_labels():
    X = pd.DataFrame({"feature": np.arange(60, dtype=float)})
    y = pd.Series([0, 1] * 30)

    models = train_partition_ensemble(
        X, y, n_partitions=3, seed=42, model_type="rf"
    )
    predictions = predict_partition_ensemble(
        models, X, model_type="rf"
    )

    assert predictions.shape == (len(X),)
    assert np.isin(predictions, [0, 1]).all()

def test_same_seed_gives_same_predictions():
    X = pd.DataFrame({"feature": np.arange(60, dtype=float)})
    y = pd.Series([0, 1] * 30)

    models_a = train_partition_ensemble(
        X, y, n_partitions=3, seed=42, model_type="rf"
    )
    models_b = train_partition_ensemble(
        X, y, n_partitions=3, seed=42, model_type="rf"
    )

    predictions_a = predict_partition_ensemble(models_a, X)
    predictions_b = predict_partition_ensemble(models_b, X)

    np.testing.assert_array_equal(predictions_a, predictions_b)
