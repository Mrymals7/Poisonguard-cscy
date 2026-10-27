import numpy as np
import pytest

from src.attacks.label_flipping import (
    random_label_flipping,
    targeted_label_flipping,
)
def test_random_flipping():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    original = y.copy()

    poisoned, indices = random_label_flipping(y, 0.2)

    assert len(indices) == 2
    assert np.sum(y != poisoned) == 2
    assert np.array_equal(y, original)

# Purpose: Verify that targeted label-flipping changes 
# only malicious labels (1) to benign (0) 
# and preserves the original data.
def test_targeted_flipping():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    original = y.copy()

    poisoned, indices = targeted_label_flipping(y, 0.2)

    assert len(indices) == 2
    assert np.all(y[indices] == 1)
    assert np.all(poisoned[indices] == 0)
    assert np.sum(y != poisoned) == 2
    assert np.array_equal(y, original)


#Purpose: Ensure both functions reject invalid poisoning rates.
@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_invalid_poisoning_rates(rate):
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError):
        random_label_flipping(y, rate)

    with pytest.raises(ValueError):
        targeted_label_flipping(y, rate)

#Purpose: Verify that using the same 
#random seed produces identical poisoning results for both attacks.
@pytest.mark.parametrize("attack", [
    random_label_flipping,
    targeted_label_flipping,
])
def test_reproducibility(attack):
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    p1, i1 = attack(y, 0.2, seed=42)
    p2, i2 = attack(y, 0.2, seed=42)

    assert np.array_equal(p1, p2)
    assert np.array_equal(i1, i2)