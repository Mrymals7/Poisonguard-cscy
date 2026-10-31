
import numpy as np
import pandas as pd


def apply_backdoor(
    X, y, poisoning_rate,
    trigger_features, trigger_values,
    seed=42
):
    """
    Here we will pply a backdoor poisoning attack to training data.

    We will Select malicious samples (label 1), then insert a fixed
    trigger into their features, and relabel them benign (0).

    also the returns will be:
        X_poisoned, y_poisoned, selected_indices
    """

    # vaalidate the poisoning rate
    if not 0 <= poisoning_rate <= 1:
        raise ValueError(
            "Poisoning rate must be between 0 and 1"
        )

    # Coping the original data
    X_poisoned = X.copy()
    y_poisoned = np.array(y, copy=True)

    if X_poisoned.ndim != 2 or y_poisoned.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")

    if len(X_poisoned) != len(y_poisoned):
        raise ValueError("X and y must have the same length")

    if not np.all(np.isin(y_poisoned, [0, 1])):
        raise ValueError("Labels must be binary: 0 or 1")

    # Identify the malicious samples
    malicious_indices = np.where(y_poisoned == 1)[0]

    # here we Calculate poisoning amount
    n_poison = int(len(y_poisoned) * poisoning_rate)

    if n_poison > len(malicious_indices):
        raise ValueError("Not enough malicious samples")

    # then we Validate trigger
    features = list(trigger_features)
    values = np.asarray(trigger_values, dtype=float)

    if len(features) == 0 or len(features) != len(values):
        raise ValueError("Invalid trigger features or values")

    if not np.all(np.isfinite(values)):
        raise ValueError("Trigger values must be finite")

    if np.any((values < 0) | (values > 1)):
        raise ValueError("Trigger values must be between 0 and 1")

    if isinstance(X_poisoned, pd.DataFrame):
        if not X_poisoned.columns.is_unique:
            raise ValueError("Feature columns must be unique")

        if any(f not in X_poisoned.columns for f in features):
            raise ValueError("Unknown trigger feature")

        feature_indices = [
            X_poisoned.columns.get_loc(f) for f in features
        ]
    else:
        feature_indices = features

        if not all(
            isinstance(i, (int, np.integer)) and
            not isinstance(i, (bool, np.bool_)) and
            0 <= i < X_poisoned.shape[1]
            for i in feature_indices
        ):
            raise ValueError("Invalid feature indices")

    if len(set(feature_indices)) != len(feature_indices):
        raise ValueError("Duplicate trigger features")

    # here we also Randomly select malicious training samples
    rng = np.random.default_rng(seed)

    selected_indices = rng.choice(
        malicious_indices,
        size=n_poison,
        replace=False
    )

    # then Apply fixed trigger
    if isinstance(X_poisoned, pd.DataFrame):
        X_poisoned.iloc[
            selected_indices, feature_indices
        ] = values
    else:
        X_poisoned[
            np.ix_(selected_indices, feature_indices)
        ] = values

    # finally we Change malicious labels to benign
    y_poisoned[selected_indices] = 0

    return X_poisoned, y_poisoned, selected_indices
