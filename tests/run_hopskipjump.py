import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.random_forest import train_random_forest
from src.attacks.hopskipjump import hopskipjump_attack


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

model = train_random_forest(X_train, y_train)

X_attack = X_val[y_val == 1].head(3)
y_attack = y_val.loc[X_attack.index]

before = model.predict(X_attack)

X_adv = hopskipjump_attack(model, X_attack)
X_adv = pd.DataFrame(
    X_adv,
    columns=X_attack.columns,
    index=X_attack.index
)

after = model.predict(X_adv)

print("Before HopSkipJump:", recall_score(y_attack, before))
print("After HopSkipJump:", recall_score(y_attack, after))
print("Changed values:", np.count_nonzero(
    X_adv.to_numpy() - X_attack.to_numpy()
))
