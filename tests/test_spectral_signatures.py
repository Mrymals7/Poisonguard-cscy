import numpy as np
import pytest
from src.defences.spectral_signatures import (
    extract_hidden_activations,
    spectral_scores,
    spectral_keep_mask,
)


def test_removes_obvious_outlier():
    activations = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [-0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
    ])
    y = np.zeros(5, dtype=int)

    scores = spectral_scores(activations, y)
    keep = spectral_keep_mask(
        activations, y, removal_fraction=0.2
    )

    assert np.argmax(scores) == 4
    np.testing.assert_array_equal(
        keep, [True, True, True, True, False]
    )


def test_scores_are_centred_within_each_class():
    activations = np.array([
        [0.0],
        [1.0],
        [2.0],
        [100.0],
        [101.0],
        [102.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    scores = spectral_scores(activations, y)

    np.testing.assert_allclose(
        scores, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    )


def test_zero_removal_keeps_all_samples():
    activations = np.array([[0.0], [1.0], [10.0]])
    y = np.array([0, 0, 0])

    keep = spectral_keep_mask(
        activations, y, removal_fraction=0.0
    )

    assert keep.dtype == bool
    np.testing.assert_array_equal(keep, [True, True, True])


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5, np.nan])
def test_rejects_invalid_removal_fraction(fraction):
    activations = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 0])

    with pytest.raises(ValueError):
        spectral_keep_mask(
            activations, y, removal_fraction=fraction
        )


def test_does_not_change_inputs():
    activations = np.array([[0.0], [1.0], [10.0]])
    y = np.array([0, 0, 0])
    original_activations = activations.copy()
    original_y = y.copy()

    spectral_keep_mask(
        activations, y, removal_fraction=0.2
    )

    np.testing.assert_array_equal(
        activations, original_activations
    )
    np.testing.assert_array_equal(y, original_y)


def test_extracts_hidden_activations_in_batches():
    import torch
    from src.models.mlp import MLP

    torch.manual_seed(42)
    model = MLP(n_features=2)
    X = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
    ], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        expected = model.hidden(torch.tensor(X)).numpy()

    model.train()
    actual = extract_hidden_activations(model, X, batch_size=2)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert model.training
