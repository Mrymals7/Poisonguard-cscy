# Spectral filtering trial - seed 42

## Settings
- Script: scripts/spectral_trial.py
- Training: 15,000 stratified samples; seed 42.
- Poisoning: 0.02 (300 malicious samples triggered and relabelled benign).
- Normalized trigger: Bwd IAT Std = 0.02220528; Fwd Packet Length Std = 0.01483571.
- MLP training: 10 epochs for each model.
- Spectral removal fraction: 0.03 per observed class, rounded up.
- Evaluation: untouched validation data for classification metrics; all 6,098 malicious validation samples with the trigger for attack success rate (ASR).
- ASR: proportion of triggered malicious samples classified as benign.

## Results
| Training | Precision | Recall | F1 | FPR | ASR |
|---|---:|---:|---:|---:|---:|
| Clean | 0.9328 | 0.9910 | 0.9610 | 0.0182 | 0.0369 |
| Poisoned | 0.9427 | 0.9903 | 0.9659 | 0.0154 | 0.0413 |
| Spectral filtering | 0.9797 | 0.9643 | 0.9719 | 0.0051 | 0.0707 |

Filtering removed 451 samples: 276 poisoned and 175 unpoisoned.
This removed 92% of poisoned samples, leaving 24.
Known poisoned indices were used only to measure removal, not to select samples for filtering.

## Interpretation and limitations
Filtering improved validation F1 and reduced false positives, but recall decreased and ASR increased. This run does not demonstrate successful backdoor mitigation.

The poisoned model's ASR was only slightly above the clean model's ASR. The attack showed a weak effect on this MLP under these settings.

This trial uses Maryam's trigger settings, but evaluates 6,098 triggered malicious samples rather than the 500 in her pilot. It is not an exact replication of backdoor_pilot.csv.

The repeated seed-42 run matched the first run. This checks repeatability, not performance across different seeds. Further seeds and poisoning rates are needed.
