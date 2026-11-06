# PoisonGuard: Attacking and Defending ML-Based Intrusion Detectors Against Data Poisoning and Evasion

CSCY413 Special Topics in Cybersecurity, Group Project
Track: Adversarial ML & Data Poisoning Defenses
Module coordinators: Dr. Sapna Sadhwani / Dr. Suleiman Yerima

## About the project
Machine-learning intrusion detectors can be attacked during training (data poisoning) and at inference time (evasion). PoisonGuard evaluates how label-flipping, backdoor and evasion attacks affect a Random Forest and an MLP trained on the corrected CICIDS2017 dataset, and compares defences that restore reliable detection: k-NN label sanitisation, partition-based ensembles, spectral signatures and adversarial training.

## Team and responsibilities
| Member | Main responsibility |
| --- | --- |
| Rahaf Alkasadi | Project lead, data preparation, baseline models, experiment runner, evaluation and charts |
| Maryam Yousef | Poisoning attacks (label flipping and backdoor) |
| Mariam Hussein | Poisoning defences (k-NN sanitisation, partition ensemble and spectral signatures) |
| Shahad Saleh | Evasion attacks, adversarial training and GitHub testing (CI) |

## Repository structure
- `src/data.py` : data cleaning, sampling, 70/15/15 split and scaling
- `src/models/` : baseline Random Forest and MLP
- `src/attacks/` : label flipping, backdoor, FGSM, PGD and HopSkipJump
- `src/defences/` : k-NN sanitisation, partition ensemble, spectral signatures and adversarial training
- `src/evaluation/` : metrics, backdoor trigger selection and backdoor pilot scripts
- `src/run_experiments.py` : runs poisoning attacks and defences over several seeds and saves the results
- `src/make_charts.py` : makes the result charts and summary tables
- `scripts/` : evasion and defence trial scripts
- `tests/` : automated tests (pytest), also run by GitHub Actions on every pull request
- `results/` : result tables (CSV), notes and charts (`results/figures/`)
- `reports/` : Deliverable 1 report sections, methodology and backdoor audit
- `docs/` : evasion attack constraints and other project notes

## Quick start (one command)
From the repository folder:
```bash
bash setup.sh
```
This installs the requirements, downloads the corrected CICIDS2017 dataset, and prepares
the training, validation and test splits. Then check that everything works:
```bash
pytest
```

## Setup step by step
If you prefer to run the steps yourself, `setup.sh` does exactly the following.

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

### 3. Prepare the data
From the repository folder:
```bash
python -m src.data
```
This cleans the data, takes a stratified 200,000-row sample, splits it 70/15/15
(train / validation / test, seed 42), scales the features to 0–1 using the training
set only, and saves the result to `data/processed/`.

## Running the experiments

### Baselines
```bash
python -m src.models.random_forest
python -m src.models.mlp
```
Each trains on the training set and prints precision, recall, F1 score and false-positive rate on the validation set.

### Poisoning attacks (clean, label flipping and backdoor)
```bash
python -m src.run_experiments
```
Runs random flipping, targeted flipping and the backdoor at 1, 5, 10 and 20% on both models with seeds 1, 2 and 3,
and saves the results to `results/results.csv` (about 1 hour).

### Defences against the backdoor
```bash
python -m src.run_experiments --attacks backdoor --rates 0.01 0.10 --defences none knn partition spectral --output results/backdoor_defences.csv
```
Compares no defence, k-NN sanitisation, the partition ensemble and spectral signatures (MLP only) (about 2 hours).

### Defences against label flipping
```bash
python -u -m src.run_experiments --defences none knn partition --models random_forest mlp --attacks random_flip targeted_flip --seeds 1 2 3 --rates 0.01 0.05 0.10 0.20 --split val --train-size 5000 --subset-seed 42 --epochs 100 --k 7 --partitions 3 --output results/defences_val_seeds123.csv
```
Run `python -m src.run_experiments --help` to see all options.

### Evasion attacks and adversarial training
```bash
python -m scripts.final_evaluation
python -m scripts.compare_defence
python -m scripts.compare_rf_defence
```
See `docs/attack_constraints.md` for the feature constraints and limitations.

### Charts
```bash
python -m src.make_charts
```
Saves the charts to `results/figures/` and the summary tables to `results/`.

## Main results
All results are the mean over seeds 1, 2 and 3 on the held-out test set unless noted.
- **Baselines:** Random Forest F1 0.999, MLP F1 0.989, both above the 0.80 precision and recall target.
- **Targeted label flipping** is silent: Random Forest recall falls to 0.88 at 10% and 0.01 at 20%, while false positives stay at 0. The MLP stays robust (F1 0.96 at 20%).
- **Random label flipping** mainly causes false alarms in the Random Forest (false-positive rate 11.8% at 20%).
- **Backdoor:** triggered attacks are classified benign 97–100% of the time (Random Forest) and 74–96% (MLP), even at 1% poisoning, while normal accuracy barely changes.
- **Backdoor defences:** at 1% poisoning, k-NN sanitisation (8%) and the partition ensemble (5%) bring the MLP below the 10% target. At 10% poisoning no defence meets the target; the best is k-NN on the MLP (96% to 60%). Spectral signatures did not reduce backdoor success at full scale.
- **Adversarial training** did not improve robustness to FGSM or PGD in our setting (see `docs/attack_constraints.md`).

Charts: `results/figures/`. Full tables: `results/results.csv`, `results/backdoor_defences.csv`, `results/defences_val_seeds123.csv`.

## Project status
Deliverable 1 (introduction, literature review, methodology and team plan): completed.
Deliverable 2 (implementation, evaluation and presentation): implementation and experiments completed; final report and live demonstration in progress.

## Ethics
All attacks target only models we train ourselves on public benchmark datasets in an isolated lab environment. No live or third-party systems are tested. Identifiers such as IP addresses are removed, and all datasets and tools are cited.

## Tools
Python, scikit-learn, PyTorch, IBM Adversarial Robustness Toolbox, pytest, GitHub Actions.