import numpy as np
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier


def hopskipjump_attack(model, X):
    """Prepare a HopSkipJump attack for Random Forest."""
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

    return attack.generate(x=X.to_numpy(dtype=np.float32))