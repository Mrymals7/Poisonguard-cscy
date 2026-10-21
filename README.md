# PoisonGuard: Attacking and Defending ML-Based Intrusion Detectors Against Data Poisoning and Evasion

CSCY413 Special Topics in Cybersecurity, Group Project
Track: Adversarial ML & Data Poisoning Defenses
Module coordinators: Dr. Sapna Sadhwani / Dr. Suleiman Yerima

## About the project
Machine-learning intrusion detectors can be attacked during training (data poisoning) and at inference time (evasion). PoisonGuard evaluates how label-flipping, backdoor and evasion attacks affect a Random Forest and an MLP trained on the corrected CICIDS2017 dataset, and compares defences that restore reliable detection: k-NN label sanitisation, partition-based ensembles, spectral signatures and adversarial training.

## Team and responsibilities
| Member | Main responsibility |
| --- | --- |
| Rahaf Alkasadi | Project lead, data preparation, baseline models and evaluation |
| Maryam Yousef | Poisoning attacks (label-flipping and backdoor) |
| Mariam Hussein | Poisoning defences (sanitisation, spectral signatures and ensembles) |
| Shahad Saleh | Evasion attacks, adversarial training and GitHub testing |

## Repository structure
- `data/` : dataset instructions and processed samples (no raw personal data)
- `src/models/` : baseline Random Forest and MLP
- `src/attacks/` : poisoning and evasion attacks
- `src/defences/` : poisoning and evasion defences
- `src/evaluation/` : metrics and result scripts
- `tests/` : automated tests (pytest)
- `docs/` : proposal and project notes
- `reports/` : Deliverable 1 and Deliverable 2 reports
- `notebooks/` : exploratory notebooks

## Setup

### 1. Install the requirements
```bash
pip install -r requirements.txt
```

### 2. Get the data
The dataset is **not** stored in this repository (it is large, and the repository is public).
We use the corrected CICIDS2017 dataset from Lanvin et al. (CRiSIS 2022), configuration
**Exp_RP**: packets re-ordered, duplicate packets removed and port-scan labels corrected.

Run this once from the folder that contains this repository (in Codespaces: `/workspaces`):
```bash
git clone https://gitlab.inria.fr/mlanvin/crisis2022.git
mkdir -p Poisonguard-cscy/data/raw/lanvin_exp_rp
cp crisis2022/Labels/Exp_RP/*.csv Poisonguard-cscy/data/raw/lanvin_exp_rp/
rm -rf crisis2022
```

### 3. Prepare the data (one command)
From the repository folder:
```bash
python -m src.data
```
This cleans the data, takes a stratified 200,000-row sample, splits it 70/15/15
(train / validation / test, seed 42), scales the features to 0–1 using the training
set only, and saves the result to `data/processed/`.

### 4. Train the baselines
```bash
python -m src.models.random_forest
python -m src.models.mlp
```
Each trains on the training set and prints precision, recall, F1 score and
false-positive rate on the validation set.

### 5. Run the tests
```bash
pytest
```

## Project status
Deliverable 1 (introduction, literature review, methodology and team plan): completed.
Deliverable 2 (implementation, evaluation and presentation): in progress. Data pipeline, baseline models and evaluation are complete; attacks and defences are under development.

## Ethics
All attacks target only models we train ourselves on public benchmark datasets in an isolated lab environment. No live or third-party systems are tested. Identifiers such as IP addresses are removed, and all datasets and tools are cited.

## Tools
Python, scikit-learn, PyTorch, IBM Adversarial Robustness Toolbox, pytest.
