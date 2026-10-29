import pandas as pd
import numpy as np

from src.models.random_forest import train_random_forest
from src.attacks.hopskipjump import hopskipjump_attack
from src.models.mlp import train_mlp
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack


def adversarial_train_mlp(
    X_train, y_train, epsilon=0.01, n_samples=10000,
    allowed_features=None, attack_type="fgsm"
):
    """Train an MLP using clean and adversarial examples."""

    model = train_mlp(X_train, y_train)

    X_sample = X_train.sample(
        n=min(n_samples, len(X_train)),
        random_state=42
    )
    y_sample = y_train.loc[X_sample.index]

    if attack_type == "pgd":
        X_adv = pgd_attack(
            model, X_sample, y_sample,
            epsilon=epsilon,
            allowed_features=allowed_features
        )
    elif attack_type == "fgsm":
        X_adv = fgsm_attack(
            model, X_sample, y_sample,
            epsilon=epsilon,
            allowed_features=allowed_features
        )
    else:
        raise ValueError("attack_type must be 'fgsm' or 'pgd'")

    X_adv = pd.DataFrame(
        X_adv,
        columns=X_train.columns
    )

    X_combined = pd.concat(
        [X_train, X_adv], ignore_index=True
    )
    y_combined = pd.concat(
        [y_train, y_sample], ignore_index=True
    )

    defended_model = train_mlp(X_combined, y_combined)

    return defended_model



def adversarial_train_rf(
    X_train, y_train,
    hopskipjump_examples,
    transferred_examples,
    seed=42
):
    """Train Random Forest on clean and adversarial attack examples."""

    def prepare_examples(examples):
        if isinstance(examples, pd.DataFrame):
            result = examples.copy()
        else:
            result = pd.DataFrame(
                np.asarray(examples),
                columns=X_train.columns
            )

        result = result[X_train.columns]

        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("Adversarial examples contain invalid values")

        return result

    hs_examples = prepare_examples(hopskipjump_examples)
    transfer_examples = prepare_examples(transferred_examples)

    X_combined = pd.concat(
        [X_train, hs_examples, transfer_examples],
        ignore_index=True
    )

    y_adversarial = pd.Series(
        np.ones(
            len(hs_examples) + len(transfer_examples),
            dtype=int
        )
    )

    y_combined = pd.concat(
        [y_train.reset_index(drop=True), y_adversarial],
        ignore_index=True
    )

    defended_rf = train_random_forest(
        X_combined,
        y_combined,
        seed=seed
    )

    return defended_rf
