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


FGSM, PGD, and HopSkipJump support the `allowed_features` parameter to restrict which input columns can be modified.


In the current experiments, the allowed features are selected using column names that start with `Fwd`. This results in 20 forward-direction features:

- Fwd Packet Length Max
- Fwd Packet Length Min
- Fwd Packet Length Mean
- Fwd Packet Length Std
- Fwd IAT Total
- Fwd IAT Mean
- Fwd IAT Std
- Fwd IAT Max
- Fwd IAT Min
- Fwd PSH Flags
- Fwd URG Flags
- Fwd RST Flags
- Fwd Header Length
- Fwd Packets/s
- Fwd Segment Size Avg
- Fwd Bytes/Bulk Avg
- Fwd Packet/Bulk Avg
- Fwd Bulk Rate Avg
- Fwd Act Data Pkts
- Fwd Seg Size Min

These features are selected as a preliminary experimental approximation of forward-direction properties that may be influenced by an attacker. Their selection does not establish that all 20 features can be independently modified in realistic network traffic.

All other features are treated as fixed during the constrained attack experiments. FGSM, PGD, and HopSkipJump use feature restrictions to limit modifications to the allowed columns. HopSkipJump also restores restricted feature values after generating adversarial examples.

Automated tests verify that FGSM, PGD, and HopSkipJump preserve non-allowed features in synthetic test cases.

However, allowing changes to these 20 Fwd features does not guarantee realistic network traffic. Packet statistics, timing measurements, flags, and other derived values may have mathematical or semantic dependencies. These relationships are not fully enforced in the current experiments.


## 5. Experimental Limitations

The current attack evaluations are preliminary.

The experiments measure model behavior under numerical feature perturbations rather than demonstrating executable network traffic attacks.

Successful evasion does not automatically imply that the modified network flows are physically or semantically valid.

The current experiments do not fully enforce relationships between dependent CICIDS2017 features.

## 6. Adversarial Training Evaluation

For the experiments with adversarial training, the MLP was tested using FGSM and PGD training with the epsilon set to 0.01, 0.03 and 0.05.

A fixed random seed (42) was employed in each experiment, 10,000 samples were randomly selected for use as training samples, and 100 samples were randomly selected for use as malicious validation samples. The evaluation was conducted using FGSM, PGD attacks and clean data.

Adversarial training did not uniformly outperform the baseline MLP, as demonstrated by the results.

The baseline obtained recall of 0.67 when using FGSM and 0.66 when using PGD on an evaluation epsilon of 0.05. The best defended results at the same evaluation epsilon were 0.27 and 0.25, respectively.

The performance of the defended models against stronger attacks has been enhanced by increasing the training epsilon, but their recall has not reached the base level.

The results suggest that the existing adversarial training setup is not robust enough against more powerful perturbations under the considered set-up.

The results are preliminary as only one random seed and a small validation sample were used in the experiments. More testing with additional seeds and validation sets is required to make generalizations.
## 7. Future Improvements

Before claiming realistic evasion performance, the project should:

1. Define a validated threat model and attacker capabilities.
2. Identify editable features for each attack scenario.
3. Preserve immutable fields and valid feature dependencies.
4. Recalculate dependent flow statistics where necessary.
5. Evaluate attacks and defenses on held-out test data.
6. Report attack success, detection metrics, and constraint violations separately.

## 8. Conclusion

PoisonGuard demonstrates preliminary adversarial evasion and defense experiments on machine-learning intrusion detection models.

The results support analysis of model vulnerability under numerical perturbations, but should not be interpreted as proof of realistic network-level attacks without further constraint validation.
