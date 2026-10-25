# Attack Constraints and Limitations

## 1. Purpose

This document describes the feature constraints and limitations of the adversarial evasion experiments in PoisonGuard.

The goal is to distinguish successful changes in model predictions from realistic network traffic modifications.

## 2. Dataset Preprocessing

PoisonGuard uses the CICIDS2017 dataset with 80 input features.

MinMaxScaler is fitted on training data only. The same scaler is applied to training, validation, and test data, with scaled values clipped to the range [0, 1].

The label and attack_type columns are excluded from model inputs.

## 3. Feature Constraints

Network flow features have different constraints:

- Protocol identifiers are categorical and should not be modified as continuous values.
- Packet counts and flag counts must remain non-negative integers in their original units.
- Packet lengths, totals, averages, and rates have mathematical dependencies.
- Flow duration and inter-arrival-time statistics must remain consistent.
- Active and idle time statistics must satisfy valid minimum, maximum, and average relationships.

Changing individual scaled values does not guarantee valid network traffic.

## 4. Implemented Attacks

The project implements:

- FGSM against the MLP model.
- PGD against the MLP model.
- HopSkipJump against Random Forest.
- Transfer attacks from MLP to Random Forest.

FGSM and PGD support an allowed_features parameter that restricts which input columns can be modified.

Automated tests verify that non-allowed features remain unchanged in synthetic test cases.

HopSkipJump does not currently enforce these feature-level constraints.

## 5. Experimental Limitations

The current attack evaluations are preliminary.

The experiments measure model behavior under numerical feature perturbations rather than demonstrating executable network traffic attacks.

Successful evasion does not automatically imply that the modified network flows are physically or semantically valid.

The current experiments do not fully enforce relationships between dependent CICIDS2017 features.

## 6. Future Improvements

Before claiming realistic evasion performance, the project should:

1. Define a validated threat model and attacker capabilities.
2. Identify editable features for each attack scenario.
3. Preserve immutable fields and valid feature dependencies.
4. Recalculate dependent flow statistics where necessary.
5. Evaluate attacks and defenses on held-out test data.
6. Report attack success, detection metrics, and constraint violations separately.

## 7. Conclusion

PoisonGuard demonstrates preliminary adversarial evasion and defense experiments on machine-learning intrusion detection models.

The results support analysis of model vulnerability under numerical perturbations, but should not be interpreted as proof of realistic network-level attacks without further constraint validation.
