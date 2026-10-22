
# III. Methodology

## A. Approach and Threat Model

This project follows an **experimental, quantitative approach**. Controlled experiments measure detection performance (precision, recall, F1 score, and false-positive rate) while one factor at a time (the attack, the poisoning rate, or the defence) is varied and the dataset, models, and metrics are held constant.

This suits the aim of the project because it isolates the effect of each attack and defence and allows direct comparison with the clean and undefended baselines.

**PoisonGuard** evaluates how data poisoning and evasion attacks affect ML-based network intrusion detectors and how well defences restore reliable detection. To address the gaps identified in Section II-K, every attack and defence is tested on the same data, the same two models, and the same metrics.

### Experimental Pipeline

The pipeline has five stages:

1. **Baseline Training:** Train a clean Random Forest and a clean MLP to classify flows as benign or malicious.
2. **Poisoning:** Corrupt the training data with label-flipping and backdoor attacks, then retrain both models.
3. **Poisoning Defences:** Apply k-NN label sanitisation, a partition-based ensemble, and spectral signatures, then retrain and re-evaluate.
4. **Evasion:** Attack the trained models at inference time with constrained adversarial flows.
5. **Evasion Defence:** Apply adversarial training and re-measure the detection of evasive flows.

### Threat Model

Following the NIST taxonomy [2], the attacker's goal is **integrity**: having malicious traffic classified as benign.

- **Poisoning attacker:** Can modify part of the training data but cannot access the test data.
- **Evasion attacker (MLP):** Has white-box access, including model gradients.
- **Evasion attacker (Random Forest):** Has black-box access to output labels only.

---

## B. Tools and Datasets

**Table II** lists the tools and data used and the reason for each choice.

### Table II. Tools and Datasets

| Item | Choice | Justification |
|---|---|---|
| Language | Python | Standard for ML research; all required libraries are available. |
| Classical ML | Scikit-learn [18] | Random Forest, k-NN, preprocessing, and metrics. |
| Deep Learning | PyTorch [19] | MLP with gradient access, needed for FGSM and PGD. |
| Attack and Defence Library | IBM Adversarial Robustness Toolbox (ART) [20] | Tested implementations of poisoning, evasion, and defence methods, supporting comparable, reproducible results. |
| Dataset | CICIDS2017 [6], corrected version [7], stratified subset | Widely used benchmark with benign and attack flows. The corrected version removes known labelling errors that could be confused with the effect of label flipping. A stratified subset keeps experiments manageable. |
| Version Control and Testing | GitHub, pytest | Required by the brief; provides commit evidence and automated tests. |

---

## C. Experiment Design

### 1. Data Preparation

- Remove identifiers such as IP addresses.
- Handle missing and infinite values.
- Scale features.
- Create one stratified train/validation/test split so that every set keeps the same class proportions.
- Keep the test set unpoisoned and exclude it from training, so all models are judged on the same clean data.
- Report class imbalance through per-class metrics.

### 2. Label Flipping

Two label-flipping strategies are evaluated:

- **Random flipping:** Randomly selected training labels are changed.
- **Targeted flipping:** Selected malicious flows are relabelled as benign.

Poisoning rates range from **1% to 20%**, including **10%**, the rate used for the backdoor target.

**Models:** Random Forest and MLP.

### 3. Backdoor Attack

- Select a fixed trigger on a few low-importance features.
- Choose these features using feature importances from a model trained on clean data [9].
- Use malicious training flows that carry the trigger and are labelled benign.
- Keep trigger values within realistic feature ranges.

### 4. Evasion Attacks

Evasion attacks are applied at inference time.

| Target Model | Attack Method | Reference |
|---|---|---|
| MLP | Fast Gradient Sign Method (FGSM) | [5] |
| MLP | Projected Gradient Descent (PGD) | [15] |
| Random Forest | HopSkipJump | [16] |
| Random Forest | Adversarial examples transferred from the MLP | — |

- FGSM and PGD use the MLP's gradients.
- HopSkipJump and transferred adversarial examples are used against the Random Forest, which does not provide gradients.
- Perturbation sizes are kept small and tested at several values.
- Modified flows remain within realistic feature ranges so that the traffic remains valid [1].

### 5. Defence Methods

Four defences are evaluated.

#### a. k-NN Label Sanitisation [11]

A training point is relabelled if most of its nearest neighbours disagree with its label.

- The number of neighbours is tuned on the validation set.
- **Applied to:** Random Forest and MLP.

#### b. Partition-Based Ensemble [13], [14]

Several models are trained on disjoint partitions of the training data and combined by majority vote.

- A poisoned sample can influence only one model.
- The number of partitions is chosen to leave each model enough training data.
- This approach differs from the bootstrap sampling inside a Random Forest.
- **Applied to:** Random Forest and MLP.

#### c. Spectral Signatures [12]

Suspicious samples are identified and removed using the MLP's hidden-layer representations. The model is then retrained on the filtered data.

- **Applied to:** MLP only.

#### d. Adversarial Training [15]

The model is retrained on training data augmented with adversarial examples.

- **Applied to:** Random Forest and MLP.
- For the Random Forest, adversarial training means retraining on the augmented dataset.

### 6. Repetition

Each experiment is repeated with multiple random seeds.

Results are reported using the **mean and spread** across repeated runs.

---

## D. Evaluation Plan

Results are compared against two baselines:

1. **Clean baseline:** A model trained on unpoisoned data.
2. **Undefended attacked baseline:** A model evaluated under attack without the relevant defence.

### Evaluation Targets

The project aims to achieve:

- **Precision ≥ 0.80** on clean data.
- **Recall ≥ 0.80** on clean data.
- **Backdoor attack success rate below 10%** at a 10% poisoning rate.
- **Clear recovery in detection of evasive flows** after adversarial training.

### Table III. Evaluation Measures

| Measure | Purpose |
|---|---|
| Precision, Recall, F1 Score | Detection quality on the clean test set. |
| False-Positive Rate | Operational usability, since false alarms are costly. |
| Backdoor Attack Success Rate | Share of triggered malicious flows classified as benign. |
| Detection Rate on Evasive Flows | Robustness against inference-time attacks. |
| Run Time | Cost of each attack and defence. |

### Threats to Validity

Results come from a single benchmark dataset (**CICIDS2017**), so they may not generalise to other networks or current traffic.

To avoid data leakage, the test set is never used for training, poisoning, or tuning.

Cross-dataset and temporal evaluation are left as future work.

---

## E. Ethical and Legal Considerations

All attacks target only models trained by the project team, using public benchmark datasets in an isolated lab environment.

- No live, third-party, or production systems are tested.
- IP addresses and other identifiers are removed.
- No personal data is used.
- All datasets, tools, and models are cited and used within their licences.
- Attack code is presented as research for evaluating defences.

---

## F. Reproducibility

The project repository is available at:

**[PoisonGuard — GitHub Repository](https://github.com/Mrymals7/Poisonguard-cscy)**

The repository will include:

- A `README.md` with setup instructions.
- A `requirements.txt` file.
- A one-command setup script.
- Fixed random seeds for reproducible experiments.
- Automated tests covering core functions.

Only public or anonymised data is used.
