import numpy as np

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier

def hopskipjump_attack(model, X, allowed_features=None):
    """Generate HopSkipJump examples with optional feature constraints."""

    original = X.to_numpy(dtype=np.float32, copy=True)

    classifier = SklearnClassifier(
        model=model,
        clip_values=(0.0, 1.0)
    )

    attack = HopSkipJump(
        classifier=classifier,
        targeted=False,
        max_iter=10,
        max_eval=100,
        init_eval=10
    )

    mask = None

    if allowed_features is not None:
        mask = np.zeros(X.shape[1], dtype=np.float32)

        for feature in allowed_features:
            index = X.columns.get_loc(feature)
            mask[index] = 1.0

    if mask is None:
        X_adv = attack.generate(x=original)
    else:
        X_adv = attack.generate(x=original, mask=mask)

        # Ensure immutable features remain unchanged.
        X_adv[:, mask == 0] = original[:, mask == 0]

    return X_adv
