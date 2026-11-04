# Label-flipping defence validation: seeds 1, 2 and 3

## Settings and checks

- Source: `results/defences_val_seeds123.csv`. All 162 expected model/attack/rate/defence/seed combinations are present exactly once. All precision, recall, F1 and FPR values are finite and within [0, 1].
- Training: one fixed stratified subset of 5,000 training samples, selected with subset seed 42. Evaluation uses the validation split.
- Repeated seeds: 1, 2 and 3 for poisoning, model training and partition assignment. These runs do not vary the selected training subset or validation split.
- Attacks: random binary label flipping and targeted malicious-to-benign label flipping at 1%, 5%, 10% and 20% of the total training subset (50, 250, 500 and 1,000 labels). The same poisoned labels are used across models and defences for each attack/rate/seed.
- Controls: clean training with no defence, k-NN and partition ensembles.
- k-NN: fixed k=7 throughout. This sweep does not retune k per attack or seed. The earlier targeted seed-42 pilot selected k=3 and is a separate experiment.
- Partition ensemble: three disjoint random partitions with majority voting. MLP single models and ensemble members each use 100 epochs. Equal epochs do not equalize optimizer updates or total computation.
- Reported values are the arithmetic mean ± sample standard deviation across three seeds. F1 is averaged per run, not recomputed from averaged precision and recall.

## Findings

Random Forest partition ensembles improve mean F1 under random flipping at 5% and 10%. At 10%, mean F1 rises from 0.9248 without defence to 0.9786 with partition voting. At 20%, k-NN has the highest mean Random Forest F1 among the tested configurations (0.9343 versus 0.8104 without defence and 0.9116 with partitions). At 1%, both defences reduce mean F1 compared with the undefended Random Forest.

For the MLP, k-NN improves mean F1 under random flipping at 10% and 20% (0.9521 versus 0.9440, and 0.9120 versus 0.8861). Partition voting does not improve mean MLP F1 at 1%, 5% or 10% random flipping. Its 20% result is only slightly higher than the undefended mean (0.8868 versus 0.8861), with substantial run variation.

Targeted flipping is particularly damaging at 20%. k-NN gives F1=0 and recall=0 for both models in all three runs. Partition voting also performs worse than the undefended model at this rate. The MLP mean F1 is 0.8198 without defence, 0.3688 with partitions and 0 with k-NN. The Random Forest mean F1 is 0.0108 without defence, 0.0012 with partitions and 0 with k-NN. These outcomes show a failure of the tested defences under this condition.

Clean-training controls also show costs: mean F1 falls from 0.9923 to 0.9695 with k-NN and 0.9886 with partitions for Random Forest, and from 0.9839 to 0.9599 and 0.9702 for the MLP. Defence performance must therefore be discussed alongside its clean-data cost.

## Full results

### Random Forest

| Attack | Rate | Defence | Precision | Recall | F1 | FPR |
|---|---:|---|---:|---:|---:|---:|
| none | 0% | None | 0.9964 ± 0.0003 | 0.9883 ± 0.0010 | 0.9923 ± 0.0006 | 0.0009 ± 0.0001 |
| none | 0% | k-NN (k=7) | 0.9890 ± 0.0004 | 0.9508 ± 0.0004 | 0.9695 ± 0.0002 | 0.0027 ± 0.0001 |
| none | 0% | Partition (3 models) | 0.9957 ± 0.0012 | 0.9817 ± 0.0006 | 0.9886 ± 0.0003 | 0.0011 ± 0.0003 |
| random_flip | 1% | None | 0.9901 ± 0.0012 | 0.9887 ± 0.0009 | 0.9894 ± 0.0008 | 0.0025 ± 0.0003 |
| random_flip | 1% | k-NN (k=7) | 0.9900 ± 0.0018 | 0.9494 ± 0.0070 | 0.9693 ± 0.0043 | 0.0025 ± 0.0004 |
| random_flip | 1% | Partition (3 models) | 0.9948 ± 0.0014 | 0.9807 ± 0.0034 | 0.9877 ± 0.0016 | 0.0013 ± 0.0004 |
| random_flip | 5% | None | 0.9377 ± 0.0076 | 0.9880 ± 0.0017 | 0.9622 ± 0.0036 | 0.0168 ± 0.0022 |
| random_flip | 5% | k-NN (k=7) | 0.9855 ± 0.0033 | 0.9475 ± 0.0029 | 0.9661 ± 0.0026 | 0.0036 ± 0.0008 |
| random_flip | 5% | Partition (3 models) | 0.9887 ± 0.0025 | 0.9826 ± 0.0015 | 0.9856 ± 0.0019 | 0.0029 ± 0.0006 |
| random_flip | 10% | None | 0.8707 ± 0.0064 | 0.9861 ± 0.0037 | 0.9248 ± 0.0038 | 0.0374 ± 0.0021 |
| random_flip | 10% | k-NN (k=7) | 0.9858 ± 0.0023 | 0.9526 ± 0.0026 | 0.9689 ± 0.0005 | 0.0035 ± 0.0006 |
| random_flip | 10% | Partition (3 models) | 0.9750 ± 0.0033 | 0.9822 ± 0.0012 | 0.9786 ± 0.0021 | 0.0064 ± 0.0009 |
| random_flip | 20% | None | 0.6963 ± 0.0109 | 0.9693 ± 0.0074 | 0.8104 ± 0.0094 | 0.1079 ± 0.0050 |
| random_flip | 20% | k-NN (k=7) | 0.9162 ± 0.0014 | 0.9531 ± 0.0101 | 0.9343 ± 0.0043 | 0.0222 ± 0.0006 |
| random_flip | 20% | Partition (3 models) | 0.8544 ± 0.0010 | 0.9772 ± 0.0065 | 0.9116 ± 0.0033 | 0.0425 ± 0.0002 |
| targeted_flip | 1% | None | 0.9968 ± 0.0003 | 0.9864 ± 0.0013 | 0.9916 ± 0.0008 | 0.0008 ± 0.0001 |
| targeted_flip | 1% | k-NN (k=7) | 0.9891 ± 0.0005 | 0.9474 ± 0.0049 | 0.9678 ± 0.0028 | 0.0027 ± 0.0001 |
| targeted_flip | 1% | Partition (3 models) | 0.9969 ± 0.0007 | 0.9785 ± 0.0021 | 0.9876 ± 0.0014 | 0.0008 ± 0.0002 |
| targeted_flip | 5% | None | 0.9985 ± 0.0007 | 0.9639 ± 0.0077 | 0.9809 ± 0.0043 | 0.0004 ± 0.0002 |
| targeted_flip | 5% | k-NN (k=7) | 0.9945 ± 0.0015 | 0.9365 ± 0.0071 | 0.9646 ± 0.0031 | 0.0013 ± 0.0004 |
| targeted_flip | 5% | Partition (3 models) | 0.9986 ± 0.0007 | 0.9661 ± 0.0101 | 0.9821 ± 0.0053 | 0.0003 ± 0.0002 |
| targeted_flip | 10% | None | 0.9993 ± 0.0002 | 0.8202 ± 0.0011 | 0.9010 ± 0.0007 | 0.0001 ± 0.0000 |
| targeted_flip | 10% | k-NN (k=7) | 0.9964 ± 0.0020 | 0.6624 ± 0.0216 | 0.7956 ± 0.0151 | 0.0006 ± 0.0004 |
| targeted_flip | 10% | Partition (3 models) | 0.9989 ± 0.0012 | 0.8553 ± 0.0211 | 0.9214 ± 0.0118 | 0.0002 ± 0.0003 |
| targeted_flip | 20% | None | 1.0000 ± 0.0000 | 0.0054 ± 0.0016 | 0.0108 ± 0.0031 | 0.0000 ± 0.0000 |
| targeted_flip | 20% | k-NN (k=7) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| targeted_flip | 20% | Partition (3 models) | 0.6667 ± 0.5774 | 0.0006 ± 0.0009 | 0.0012 ± 0.0018 | 0.0000 ± 0.0000 |

### MLP

| Attack | Rate | Defence | Precision | Recall | F1 | FPR |
|---|---:|---|---:|---:|---:|---:|
| none | 0% | None | 0.9814 ± 0.0023 | 0.9865 ± 0.0025 | 0.9839 ± 0.0003 | 0.0048 ± 0.0006 |
| none | 0% | k-NN (k=7) | 0.9687 ± 0.0039 | 0.9512 ± 0.0009 | 0.9599 ± 0.0019 | 0.0079 ± 0.0010 |
| none | 0% | Partition (3 models) | 0.9612 ± 0.0058 | 0.9793 ± 0.0024 | 0.9702 ± 0.0022 | 0.0101 ± 0.0016 |
| random_flip | 1% | None | 0.9607 ± 0.0162 | 0.9870 ± 0.0035 | 0.9737 ± 0.0077 | 0.0103 ± 0.0044 |
| random_flip | 1% | k-NN (k=7) | 0.9677 ± 0.0062 | 0.9501 ± 0.0094 | 0.9588 ± 0.0078 | 0.0081 ± 0.0015 |
| random_flip | 1% | Partition (3 models) | 0.9652 ± 0.0031 | 0.9763 ± 0.0046 | 0.9707 ± 0.0009 | 0.0090 ± 0.0009 |
| random_flip | 5% | None | 0.9318 ± 0.0210 | 0.9887 ± 0.0010 | 0.9593 ± 0.0116 | 0.0185 ± 0.0061 |
| random_flip | 5% | k-NN (k=7) | 0.9646 ± 0.0146 | 0.9474 ± 0.0012 | 0.9559 ± 0.0072 | 0.0089 ± 0.0038 |
| random_flip | 5% | Partition (3 models) | 0.9404 ± 0.0044 | 0.9653 ± 0.0078 | 0.9527 ± 0.0033 | 0.0156 ± 0.0013 |
| random_flip | 10% | None | 0.9061 ± 0.0275 | 0.9856 ± 0.0024 | 0.9440 ± 0.0147 | 0.0262 ± 0.0083 |
| random_flip | 10% | k-NN (k=7) | 0.9515 ± 0.0030 | 0.9526 ± 0.0027 | 0.9521 ± 0.0028 | 0.0124 ± 0.0008 |
| random_flip | 10% | Partition (3 models) | 0.9153 ± 0.0338 | 0.9647 ± 0.0156 | 0.9392 ± 0.0222 | 0.0230 ± 0.0097 |
| random_flip | 20% | None | 0.8052 ± 0.0550 | 0.9866 ± 0.0056 | 0.8861 ± 0.0357 | 0.0618 ± 0.0216 |
| random_flip | 20% | k-NN (k=7) | 0.8699 ± 0.0467 | 0.9593 ± 0.0066 | 0.9120 ± 0.0272 | 0.0371 ± 0.0153 |
| random_flip | 20% | Partition (3 models) | 0.8200 ± 0.0271 | 0.9661 ± 0.0152 | 0.8868 ± 0.0150 | 0.0543 ± 0.0102 |
| targeted_flip | 1% | None | 0.9794 ± 0.0029 | 0.9857 ± 0.0033 | 0.9825 ± 0.0002 | 0.0053 ± 0.0008 |
| targeted_flip | 1% | k-NN (k=7) | 0.9685 ± 0.0036 | 0.9481 ± 0.0058 | 0.9581 ± 0.0042 | 0.0079 ± 0.0009 |
| targeted_flip | 1% | Partition (3 models) | 0.9616 ± 0.0067 | 0.9770 ± 0.0043 | 0.9692 ± 0.0023 | 0.0100 ± 0.0018 |
| targeted_flip | 5% | None | 0.9764 ± 0.0139 | 0.9814 ± 0.0059 | 0.9788 ± 0.0042 | 0.0061 ± 0.0037 |
| targeted_flip | 5% | k-NN (k=7) | 0.9756 ± 0.0030 | 0.9415 ± 0.0056 | 0.9582 ± 0.0016 | 0.0060 ± 0.0008 |
| targeted_flip | 5% | Partition (3 models) | 0.9641 ± 0.0019 | 0.9629 ± 0.0112 | 0.9635 ± 0.0047 | 0.0091 ± 0.0006 |
| targeted_flip | 10% | None | 0.9853 ± 0.0047 | 0.9712 ± 0.0069 | 0.9782 ± 0.0017 | 0.0037 ± 0.0012 |
| targeted_flip | 10% | k-NN (k=7) | 0.9795 ± 0.0110 | 0.9086 ± 0.0292 | 0.9424 ± 0.0109 | 0.0049 ± 0.0028 |
| targeted_flip | 10% | Partition (3 models) | 0.9641 ± 0.0067 | 0.9366 ± 0.0094 | 0.9501 ± 0.0031 | 0.0089 ± 0.0018 |
| targeted_flip | 20% | None | 0.9746 ± 0.0308 | 0.7137 ± 0.1051 | 0.8198 ± 0.0558 | 0.0055 ± 0.0072 |
| targeted_flip | 20% | k-NN (k=7) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| targeted_flip | 20% | Partition (3 models) | 0.9938 ± 0.0068 | 0.2320 ± 0.0997 | 0.3688 ± 0.1337 | 0.0004 ± 0.0006 |

## Limitations and implementation note

These are validation results on a fixed 5,000-sample subset, not final held-out test results. The three seeds estimate variation from poisoning, initialization and partition assignment on that subset. They do not establish statistical significance, generalization to other subsets or certified robustness. No defence consistently improves every model, attack and rate. Spectral backdoor filtering is evaluated separately and is not included in these 162 label-flipping rows.

The original run stopped after 148 completed rows because the MLP class-weight calculation returned a one-element tensor for a single-class training set. The fix uses two finite weights, preserving the previous calculation when both classes are present. Both single-class cases and the existing 44 tests passed. The remaining 14 runs were completed with the fix and appended without repeating completed configurations. The interrupted CSV was backed up. Single-class training after sanitisation is a meaningful defence failure and must not be hidden or repaired by restoring ground-truth labels.

The CSV does not record the class counts after sanitisation. Therefore it demonstrates the zero-recall outcome but does not independently establish those class counts. The `asr_clean_attacks` column measures malicious validation samples classified as benign without an inference-time trigger. It must not be described as backdoor-trigger ASR.

## Reproduce the complete sweep

```bash
python -u -m src.run_experiments \
  --defences none knn partition \
  --models random_forest mlp \
  --attacks random_flip targeted_flip \
  --seeds 1 2 3 \
  --rates 0.01 0.05 0.10 0.20 \
  --split val --train-size 5000 --subset-seed 42 \
  --epochs 100 --k 7 --partitions 3 \
  --output results/defences_val_seeds123.csv
```

This command writes to the specified output file. Existing completed results should be preserved before deliberately reproducing the run.
