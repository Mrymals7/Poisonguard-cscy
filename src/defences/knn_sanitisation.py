import numpy as np
from sklearn.neighbors import NearestNeighbors

def sanitise_labels(X, y, k=5):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if len(X) != len(y):
        raise ValueError("Each sample must have one label.")
    if not isinstance(k, int) or not 1 <= k < len(y):
        raise ValueError("k must be an integer between 1 and n_samples - 1.")

    neighbours = NearestNeighbors(n_neighbors=k)
    neighbours.fit(X)
    neighbour_indices = neighbours.kneighbors(return_distance=False)
    neighbour_labels = y[neighbour_indices]
    cleaned_y = y.copy()
    for i, labels in enumerate(neighbour_labels):
        values, counts = np.unique(labels, return_counts=True)
        winner = np.argmax(counts)
        if counts[winner] > k / 2:
            cleaned_y[i] = values[winner]

    return cleaned_y

