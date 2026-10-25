import pandas as pd

from src.models.mlp import train_mlp
from src.attacks.fgsm import fgsm_attack


def adversarial_train_mlp(X_train, y_train, epsilon=0.01, n_samples=1000,
                          allowed_features=None):
    """Train an MLP using clean and adversarial examples."""

    model = train_mlp(X_train, y_train)

    X_sample = X_train.head(n_samples)
    y_sample = y_train.loc[X_sample.index]

    X_adv = fgsm_attack(
        model, X_sample, y_sample,
        epsilon=epsilon,
        allowed_features=allowed_features
    )

    X_adv = pd.DataFrame(
        X_adv,
        columns=X_train.columns
    )

    X_combined = pd.concat([X_train, X_adv], ignore_index=True)
    y_combined = pd.concat([y_train, y_sample], ignore_index=True)

    defended_model = train_mlp(X_combined, y_combined)

    return defended_model
