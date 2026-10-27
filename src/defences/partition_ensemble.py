import numpy as np
from src.models.random_forest import train_random_forest
def train_partition_ensemble(X_train, y_train, n_partitions=3, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_train))
    partitions = np.array_split(indices, n_partitions)
    models = []
    for group in partitions:

        X_group = X_train.iloc[group]
        y_group = y_train.iloc[group]
        model = train_random_forest(X_group, y_group, seed)
        models.append(model)
    return models

def predict_partition_ensemble(models, X):
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    votes = np.array(predictions)
    counts = votes.sum(axis=0)
    return (counts > len(models) / 2).astype(int)




