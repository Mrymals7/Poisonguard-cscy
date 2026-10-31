import numpy as np
import pytest
from src.defences.knn_sanitisation import sanitise_labels


def test_corrects_flipped_label():
    X = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y = np.array([0, 0, 1, 0, 0])

    cleaned = sanitise_labels(X, y, k=3)

    np.testing.assert_array_equal(cleaned, [0, 0, 0, 0, 0])

def test_does_not_change_inputs():
    X = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y = np.array([0, 0, 1, 0, 0])
    original_X = X.copy()
    original_y = y.copy()

    sanitise_labels(X, y, k=3)

    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)

@pytest.mark.parametrize("k", [0, -1, 5, 6, 1.5])
def test_rejects_invalid_k(k):
    X = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y = np.array([0, 0, 1, 0, 0])

    with pytest.raises(ValueError):
        sanitise_labels(X, y, k=k)
