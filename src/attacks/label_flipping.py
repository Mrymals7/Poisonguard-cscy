import numpy as np


def random_label_flipping(y, poisoning_rate, seed=42):
    """
    Here we will randomly flip a percentage of training labels.

    y: Original binary labels (0 or 1).
    poisoning_rate: Fraction to change (0.10 means 10%).
    seed: Makes the random selection reproducible.
    """

    # Make a copy of the original labels.
    if not 0 <= poisoning_rate <= 1:
        raise ValueError("Poisoning rate must be between 0 and 1")
    y_poisoned = np.array(y, copy=True)


    # Calculate how many labels to flip.
    n_flip = int(len(y_poisoned) * poisoning_rate)

    # Randomly select the samples.
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(y_poisoned), size=n_flip, replace=False)

    # Flip selected labels (0 to 1, 1 to 0).
    y_poisoned[selected_indices] = 1 - y_poisoned[selected_indices]

    # Return the modified labels and selected indices.
    return y_poisoned, selected_indices

def targeted_label_flipping(y, poisoning_rate, seed=42):
    """
    wo will change the selected malicious labels (1) to benign (0)
    Poisoning rate is based on total training samples.
    """

    # Copy the original labels.
    y_poisoned = np.array(y, copy=True)

    # Identify malicious samples (label = 1).
    malicious_indices = np.where(y_poisoned == 1)[0]

    # Calculate how many samples to poison
    if not 0 <= poisoning_rate <= 1:
        raise ValueError("Poisoning rate must be between 0 and 1")

    n_poison = int(len(y_poisoned) * poisoning_rate)

    if n_poison > len(malicious_indices):
        raise ValueError("Not enough malicious samples")
    

    # Randomly select malicious samples.
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(malicious_indices, size=n_poison, replace=False)

    # Change selected labels from 1 to 0
    y_poisoned[selected_indices] = 0

    # Return the modified labels and indices.
    return y_poisoned, selected_indices