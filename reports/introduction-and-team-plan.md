# PoisonGuard: Attacking and Defending ML-Based Intrusion Detectors Against Data Poisoning and Evasion

**Deliverable 1: Problem Definition, Literature Review, Methodology and Team Plan**

This file contains the Abstract, Section I (Introduction) and Section IV (Team Plan) of the Deliverable 1 report.

## Abstract

Machine learning (ML) is widely used in network intrusion detection systems (NIDS) to separate benign from malicious traffic, which makes these detectors a target for adversarial attacks. This project, PoisonGuard, studies two such threats: training-time data poisoning (label-flipping and backdoor attacks) and inference-time evasion attacks. We review the attack and defence literature, identify gaps in how defences have been evaluated on tabular, imbalanced security data, and propose an experimental methodology based on the corrected CICIDS2017 dataset. A Random Forest and a multilayer perceptron (MLP) are attacked at poisoning rates of 1–20% and with constrained FGSM, PGD, HopSkipJump and transfer attacks, then defended with k-NN label sanitisation, a partition-based ensemble, spectral signatures and adversarial training. Defences are judged on precision, recall, F1 score, false-positive rate, attack success rate and run time. The report also sets out the team plan for delivering a reproducible open-source framework and a live demonstration.

**Index Terms:** Adversarial machine learning, backdoor attacks, CICIDS2017, data poisoning, evasion attacks, network intrusion detection.

---

# I. Introduction

## A. Background and Motivation

Machine learning is widely used in network intrusion detection systems (NIDS) to distinguish benign from malicious flows [1]. This reliance also makes the models an attractive target. The NIST taxonomy of adversarial machine learning groups attacks on predictive systems into evasion, poisoning and privacy attacks [2], and MITRE ATLAS lists poisoning (AML.T0020, Poison Training Data) and evasion (AML.T0015, Evade ML Model) as separate attacker techniques [3].

This project investigates both attacks. In data poisoning, the attacker manipulates a small fraction of the training data, either by flipping labels to degrade the model or by adding a hidden backdoor trigger so that the attacker's traffic is classified as benign [4]. In evasion attacks, small, carefully engineered changes to malicious traffic cause an already trained model to classify that traffic as benign [5].

Both attacks can go unnoticed. A backdoored model still passes regular accuracy tests (Section II-C), and evasive traffic looks like ordinary traffic. A detector that silently misses these attacks is worse than having none, because it gives defenders a false sense of security. Most existing defences were designed and evaluated on image data, and there is little evidence of how they perform on tabular security data, where classes are imbalanced, features must remain realistic and false positives are costly (Sections II-J and II-K).

## B. Problem Statement and Aim

**Problem statement:** Data poisoning and evasion attacks can compromise ML-based intrusion detectors without being detected, and it is not yet clear which defences can restore reliable detection on realistic network data without generating an excessive number of false alarms.

**Aim:** To evaluate the impact of training-time and inference-time adversarial attacks on ML-based intrusion detectors, and to develop and test defences that make these detectors more reliable.

## C. Objectives

1. Implement two baseline classifiers, a Random Forest and a multilayer perceptron (MLP), each achieving precision and recall of at least 0.80 on clean data.
2. Implement poisoning attacks (label flipping and a backdoor trigger) at poisoning rates of 1–20% against both models.
3. Implement evasion attacks that keep modified flows realistic: FGSM and PGD against the MLP, and HopSkipJump and transferred adversarial examples against the Random Forest.
4. Implement and evaluate four defences: k-NN label sanitisation, a partition-based ensemble and spectral signatures against poisoning, and adversarial training against evasion.
5. Deliver a tested, reproducible open-source framework and a live demonstration.

## D. Scope

This work focuses on ML-based intrusion detection using tabular network data [6], [7]. It compares two model families, a tree-based model and a neural network, under the attacks in Objectives 2 and 3 and the defences in Objective 4. Each attack and defence is applied only to the models it can technically be applied to.

Model inversion, membership inference and federated learning attacks are out of scope, and malware datasets such as EMBER are left for future work. All experiments use publicly available datasets and models built by the team in an isolated lab, with no live or third-party systems involved (Section III-E).

## E. Expected Outcomes and Real-World Relevance

The project will produce a practical framework, PoisonGuard; a comparison of poisoning and evasion attacks and their defences on security data; and recommendations for security operations and ML teams. The targets are to keep precision and recall above 0.80 on clean data, to reduce the backdoor attack success rate below 10% at a 10% poisoning rate, and to show a clear improvement in the detection of evasive flows after adversarial training.

The results are framed in the language of security teams. Attacks are mapped to MITRE ATLAS techniques and the NIST taxonomy, and defences are evaluated not only on detection accuracy but also on false-positive rate and run time, which determine whether a defence is usable in a real security operations centre.

The rest of this report is organised as follows. Section II reviews the literature, Section III describes the methodology and Section IV presents the team plan.

---

# IV. Team Plan

The project is delivered by four members over Weeks 4–8 after this submission. Each member owns a workstream (Table IV), the work is tracked as issues on the team's GitHub repository, and the schedule from the Deliverable 1 submission to the final demonstration is shown in Table V.

## A. Roles and Responsibilities

**Table IV. Team Roles and Responsibilities**

| Member | Project Responsibility | Contribution to Deliverable 1 |
|---|---|---|
| Rahaf Alkasadi | Project lead, data preparation, baseline models and evaluation | Literature review, Part B: defences, standards, dataset, synthesis and gaps |
| Maryam Yousef | Poisoning attacks (label flipping and backdoor) | Methodology and GitHub repository setup |
| Mariam Hussain | Poisoning defences (sanitisation, spectral signatures and ensembles) | Literature review, Part A: data poisoning attacks |
| Shahad Saleh | Evasion attacks, adversarial training and GitHub testing | Introduction, team plan, final merge and proofreading |

Evaluation and testing are shared. Rahaf leads the evaluation, Maryam and Mariam write the tests for their own attack and defence modules, and Shahad sets up repository-wide automated testing. Every member contributes code to the repository and is prepared to explain any part of the project.

## B. Timeline, Weeks 4–8

Table V shows the weekly schedule, the lead for each stage and the expected output.

**Table V. Project Timeline**

| Week | Main Tasks | Lead | Output |
|---|---|---|---|
| 4 | Submit Deliverable 1. Finalise the repository, README and issue board. Download and clean the corrected CICIDS2017 data. | All; Shahad (repository); Rahaf (data) | Report submitted; repository ready; cleaned dataset |
| 5 | Create the stratified split and scale features. Train and evaluate the Random Forest and MLP baselines. Start the label-flipping and backdoor attack code. | Rahaf; Maryam | Baselines evaluated against the 0.80 precision and recall target; first attack code |
| 6 | Run poisoning attacks at 1–20%. Implement k-NN sanitisation, the partition-based ensemble and spectral signatures. Implement FGSM, PGD, HopSkipJump and transfer attacks. | Maryam; Mariam; Shahad | Attack and defence modules; first results |
| 7 | Implement adversarial training. Run the full experiments with multiple random seeds. Produce results tables and charts. Complete automated tests and one-command setup. Draft the final report. | All; Rahaf (evaluation); Shahad (testing) | Complete results; passing tests; report draft |
| 8 | Finish the final report. Prepare the presentation and rehearse the live demo. Final check of the README and repository. Submit. | All | Final report, live demo and presentation |

## C. Collaboration and Risk Management

The team meets weekly to review progress against the timeline. Every task is a GitHub issue assigned to one member, each member commits from their own account, and changes are reviewed through pull requests before they are merged. Table VI lists the main risks and how the team will manage them.

**Table VI. Risks and Mitigation**

| Risk | Mitigation |
|---|---|
| The scope is too large for five weeks | Work in priority order: baselines, label-flipping and backdoor attacks, FGSM and PGD, then the remaining defences; add extra experiments only if time allows |
| Long training and attack run times, especially for the query-heavy HopSkipJump attack | Use a stratified subset of the data and run HopSkipJump on a sample of test flows |
| Errors already in the dataset are confused with attack effects | Use the corrected CICIDS2017 and compare every attacked model with a clean baseline trained on the same data |
| Uneven contribution or an unavailable member | Assign every task as an issue, check progress weekly, and keep all code in the shared repository so any member can continue the work |

---

## References

*Numbering matches the full Deliverable 1 report.*

[1] K. He, D. D. Kim, and M. R. Asghar, "Adversarial machine learning for network intrusion detection systems: A comprehensive survey," *IEEE Commun. Surveys Tuts.*, vol. 25, no. 1, pp. 538–566, 2023, doi: 10.1109/COMST.2022.3233793.

[2] A. Vassilev, A. Oprea, A. Fordyce, H. Anderson, X. Davies, and M. Hamin, "Adversarial machine learning: A taxonomy and terminology of attacks and mitigations," Nat. Inst. Standards Technol., Gaithersburg, MD, USA, Rep. NIST AI 100-2e2025, 2025, doi: 10.6028/NIST.AI.100-2e2025.

[3] MITRE, "ATLAS: Adversarial Threat Landscape for Artificial-Intelligence Systems." Accessed: Oct. 1, 2026. [Online]. Available: <https://atlas.mitre.org>

[4] A. E. Cinà *et al.*, "Wild patterns reloaded: A survey of machine learning security against training data poisoning," *ACM Comput. Surv.*, vol. 55, no. 13s, pp. 1–39, 2023, doi: 10.1145/3585385.

[5] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2015. [Online]. Available: <https://arxiv.org/abs/1412.6572>

[6] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in *Proc. 4th Int. Conf. Inf. Syst. Secur. Privacy (ICISSP)*, 2018, pp. 108–116, doi: 10.5220/0006639801080116.

[7] M. Lanvin, P.-F. Gimenez, Y. Han, F. Majorczyk, L. Mé, and É. Totel, "Errors in the CICIDS2017 dataset and the significant differences in detection performances it makes," in *Risks and Security of Internet and Systems (CRiSIS 2022)*, S. Kallel *et al.*, Eds., Lecture Notes in Computer Science, vol. 13857. Cham, Switzerland: Springer, 2023, pp. 18–33, doi: 10.1007/978-3-031-31108-6_2.
