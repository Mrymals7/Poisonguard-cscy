import numpy as np
from src.models.random_forest import train_random_forest
from src.models.mlp import train_mlp, predict_mlp
def train_partition_ensemble(X_train, y_train, n_partitions=3, seed=42, model_type="rf"):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_train))
    partitions = np.array_split(indices, n_partitions)
    models = []
    for group in partitions:

        X_group = X_train.iloc[group]
        y_group = y_train.iloc[group]
        if model_type == "rf":
            model = train_random_forest(X_group, y_group, seed)
        elif model_type == "mlp":
            model = train_mlp(X_group, y_group, seed)
        else:
            raise ValueError("model_type must be 'rf' or 'mlp'")
        models.append(model)
    return models

def predict_partition_ensemble(models, X, model_type="rf"):
    predictions = []
    for model in models:
        if model_type == "rf":
            pred = model.predict(X)
        elif model_type == "mlp":
            pred = predict_mlp(model, X)
        else:
            raise ValueError("model_type must be 'rf' or 'mlp'")
        predictions.append(pred)
    votes = np.array(predictions)
    counts = votes.sum(axis=0)
    return (counts > len(models) / 2).astype(int)




