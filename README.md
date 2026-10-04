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
```
pip install -r requirements.txt
pytest
```
Full run instructions will be added as the implementation is completed (Deliverable 2).

## Project status
Deliverable 1 (introduction, literature review, methodology and team plan): completed.
Deliverable 2 (implementation, evaluation and presentation): in planning, see the Issues board.

## Ethics
All attacks target only models we train ourselves on public benchmark datasets in an isolated lab environment. No live or third-party systems are tested. Identifiers such as IP addresses are removed, and all datasets and tools are cited.

## Tools
Python, scikit-learn, PyTorch, IBM Adversarial Robustness Toolbox, pytest.
