# Partition ensemble trials — seed 42

## Settings
- Script: scripts/ensemble_trial.py
- Training: 5,000 stratified samples; seed 42.
- Attack: random label flipping at 10% (500 labels).
- Evaluation: untouched validation data.
- Disjoint random partitions with majority voting.
- MLP sweep: 100 epochs per model.

## Random Forest results

| Partitions | Clean F1 | Poisoned F1 |
|---|---:|---:|
| 11 | 0.9753 | 0.9773 |
| 25 | 0.9538 | 0.9622 |
| 51 | 0.9143 | 0.9133 |

## MLP results — 100 epochs

| Partitions | Clean F1 | Poisoned F1 |
|---|---:|---:|
| 3 | 0.9708 | 0.9392 |
| 11 | 0.9503 | 0.9345 |
| 25 | 0.8717 | 0.8736 |
| 51 | 0.8374 | 0.8379 |

## Interpretation and limitations
- RF with 11 partitions achieved the highest F1 among the RF partition counts shown.
- MLP with 3 partitions achieved the highest F1 among the MLP partition counts tested.
- Poisoned 3-partition MLP F1 increased from 0.7576 at 10 epochs to 0.9392 at 100 epochs.
- Larger partition counts reduced F1 in this experiment.
- Equal epoch counts do not give equal numbers of optimizer updates when partition sizes differ.
- These are exploratory validation results from one seed and one poisoning rate.
- Single MLP baselines at 100 epochs are now included for comparison.
- These results do not establish robustness to targeted label flipping or backdoor attacks.

## Raw logs
- results/ensemble_trial_seed42.txt
- results/ensemble_trial_mlp100_seed42.txt

## Single MLP baselines — 100 epochs

| Training | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| Clean | 0.9771 | 0.9884 | 0.9827 | 0.0059 |
| Poisoned | 0.9280 | 0.9869 | 0.9565 | 0.0195 |

The single MLP achieved higher validation F1 than all tested MLP
partition ensembles under both clean and poisoned training.
This pilot therefore does not demonstrate an MLP ensemble benefit.
All comparisons use seed 42; equal epochs do not equalize optimizer updates.

Raw log: results/ensemble_trial_with_single100_seed42.txt
