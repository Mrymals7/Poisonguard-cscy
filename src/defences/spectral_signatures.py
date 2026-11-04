import numpy as np
import torch


def extract_hidden_activations(model, X, batch_size=256):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("X must be a non-empty 2D array.")

    device = next(model.parameters()).device
    was_training = model.training
    activations = []
    model.eval()

    try:
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(
                    values[start:start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                hidden = model.hidden(batch)
                activations.append(hidden.cpu().numpy())
    finally:
        model.train(was_training)

    return np.concatenate(activations, axis=0)


def spectral_scores(activations, y):
    activations = np.asarray(activations, dtype=np.float64)
    y = np.asarray(y)

    if activations.ndim != 2 or y.ndim != 1:
        raise ValueError("Activations must be 2D and labels 1D.")
    if len(activations) != len(y):
        raise ValueError("Each activation must have one label.")
    if activations.shape[1] == 0:
        raise ValueError("Activations must have at least one feature.")
    if not np.isfinite(activations).all():
        raise ValueError("Activations must contain finite values.")

    scores = np.zeros(len(y), dtype=float)
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        if len(indices) < 2:
            continue

        group = activations[indices]
        centred = group - group.mean(axis=0)
        _, _, directions = np.linalg.svd(
            centred, full_matrices=False
        )
        scores[indices] = (centred @ directions[0]) ** 2

    return scores


def spectral_keep_mask(activations, y, removal_fraction=0.15):
    if not 0 <= removal_fraction < 1:
        raise ValueError("removal_fraction must be in [0, 1).")

    y = np.asarray(y)
    scores = spectral_scores(activations, y)
    keep = np.ones(len(y), dtype=bool)

    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        n_remove = min(
            int(np.ceil(removal_fraction * len(indices))),
            len(indices) - 1,
        )
        if n_remove <= 0:
            continue

        order = np.argsort(-scores[indices], kind="stable")
        keep[indices[order[:n_remove]]] = False

    return keep
