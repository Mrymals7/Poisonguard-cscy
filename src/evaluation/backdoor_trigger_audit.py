import numpy as np
from pathlib import Path
from src.data import load_processed

X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

features = ["Bwd IAT Std", "Fwd Packet Length Std"]

benign = X_train.loc[
    np.asarray(y_train) == 0, features
]

trigger = benign.quantile(0.75)

print("BACKDOOR TRIGGER AUDIT")

for feature in features:
    value = trigger[feature]
    minimum = benign[feature].min()
    maximum = benign[feature].max()

    print(f"\nFeature: {feature}")
    print(f"Trigger value: {value:.6f}")
    print(f"Benign range: {minimum:.6f} to {maximum:.6f}")
    print(f"Within benign range: {minimum <= value <= maximum}")

# Check approximate joint occurrence
nearby = benign.sub(trigger, axis="columns").abs()
joint_count = int((nearby <= 0.01).all(axis=1).sum())

print("\nBenign samples near both trigger values:", joint_count)
print("Total benign training samples:", len(benign))

# Audit validation modifications
malicious = X_val.loc[
    np.asarray(y_val) == 1
].head(500).copy()

triggered = malicious.copy()
triggered.loc[:, features] = trigger.to_numpy()

other_columns = malicious.columns.difference(features)

unchanged = triggered[other_columns].equals(
    malicious[other_columns]
)

changed = int(
    np.any(
        triggered[features].to_numpy()
        != malicious[features].to_numpy(),
        axis=1
    ).sum()
)

print("\nValidation samples with trigger changes:", changed)
print("Other features unchanged:", unchanged)

# Write audit documentation
Path("reports").mkdir(exist_ok=True)

report = f"""# Backdoor Trigger Consistency Audit

## Trigger
Features: {features}
Values: {trigger.to_dict()}

## Audit Results
- Values within benign ranges: {
    all(benign[f].min() <= trigger[f] <= benign[f].max()
        for f in features)
}
- Benign records near both values (tolerance 0.01): {joint_count}
- Validation samples altered: {changed}
- Other features unchanged: {unchanged}

## Limitations
The experiment changes two normalized flow features.

Being inside benign feature ranges does not establish
that their combination represents a physically valid
network flow.

Related packet and timing measurements are not
recomputed from raw packet data.

The joint-neighborhood count is exploratory and
does not prove packet-level consistency.

Therefore, these results demonstrate a synthetic
feature-space backdoor, not a verified realistic
network-level attack.

## Experimental Findings
Random Forest mean triggered benign rate across
seeds 7, 42, and 123: 92.2%.

The MLP showed a much smaller backdoor effect.
"""

Path("reports/backdoor_trigger_audit.md").write_text(
    report
)

print("\nAudit saved to reports/backdoor_trigger_audit.md")
