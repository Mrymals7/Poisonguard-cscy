# II. Literature Review — Sections F to K

## F. Defences Against Data Poisoning

The previous subsections showed how label-flipping and backdoor attacks can damage an intrusion detector without anyone noticing. This subsection considers what defenders can do about it. Most poisoning defences fall into one of three categories: some clean the data before training, some use the trained model to detect poisoned samples, and some change the training process so that a small number of poisoned samples cannot cause much damage.

### 1) Data Sanitisation

Paudice *et al.* [11] proposed a way to undo label flipping, the attack described in Section II-B. Their idea is to examine each training point's nearest neighbours: if most neighbours have a different label, the point probably has the wrong one and is relabelled. We chose this method because it is easy to build and works with any model, so it can be applied to both the Random Forest and the MLP. It has a weakness, however: it only works if poisoned points stand out, and an attacker who places many poisoned points close together can make them look normal to each other. Cinà *et al.* [4] make a similar point: if attackers make poisoned samples resemble normal data, defences like this can miss them. This matters for our project, because our backdoor trigger uses realistic values on low-importance features (Section II-C). This differs from Jang *et al.* [9], who found that triggers on high-importance features were far more successful but also more likely to be spotted as outliers. We chose low-importance features deliberately to test the quieter side of this trade-off, which means our poisoned flows may not look like outliers either.

### 2) Detecting Backdoors Through the Model

Backdoor samples can look completely ordinary, so checking the data alone is often not enough. Some defences therefore inspect the trained model instead. Tran *et al.* [12] observed that backdoor samples leave a "spectral signature" in a neural network's hidden-layer representations: the poisoned samples stand out along certain directions, which can be found using singular value decomposition. These samples are then removed and the model is retrained. Because this method needs access to hidden layers, it can be used only on the MLP and not on the Random Forest. It was also designed and tested on image classifiers. We found very few detection methods designed for intrusion detection data; the topological data analysis approach of Monkam *et al.* [10], discussed in Section II-D, is one of them.

### 3) Robust Training

A third strategy is a partition-based ensemble, an idea introduced as Deep Partition Aggregation by Levine and Feizi [13]. Instead of training a single model, the training data is split into many non-overlapping parts, one model is trained on each part, and the models vote on the final prediction. Because each poisoned sample can affect only one part, it can change only one vote. Rezaei *et al.* [14] show that this kind of defence is provably robust: the final prediction cannot change unless the number of poisoned samples exceeds a certain limit. They also improved the vote by adding a second "run-off" round between the top two classes. We chose a partition-based ensemble because it works with any base model, allowing a fair comparison between the Random Forest and the MLP. The drawbacks are that training many models takes more time, and these methods have only been tested on image datasets (MNIST, CIFAR-10 and GTSRB).

## G. Evasion Attacks and Adversarial Training

Poisoning is only one part of the problem. In an evasion attack, the model is already trained, and the attacker modifies the input while the model is in use. One of the best-known methods is the Fast Gradient Sign Method (FGSM) of Goodfellow *et al.* [5], which changes an input by a small amount in the direction that most increases the model's error. The same paper showed that training on such examples, known as adversarial training, makes a model more robust. Madry *et al.* [15] extended this with Projected Gradient Descent (PGD), which takes many small steps instead of one and is therefore a much stronger attack. They also formalised adversarial training as a min–max problem in which the model is trained on PGD examples so that it learns to resist them. This has become the most common defence against evasion, although it slows down training and can reduce accuracy on clean data.

FGSM and PGD both rely on the model's gradients, which a Random Forest does not have, so they can be used only against our MLP. The Random Forest requires a different approach. HopSkipJump [16] uses only the labels the model outputs and estimates the decision boundary from them. This is also more realistic, since a real attacker usually cannot see inside the model. In our project, the MLP will therefore face FGSM and PGD, while the Random Forest will be attacked with HopSkipJump and with adversarial examples transferred from the MLP. For the same reason, adversarial training differs between the models: the MLP is trained on its own PGD examples, while the Random Forest is retrained on HopSkipJump and transferred examples.

As with poisoning (Section II-D), modified traffic must remain realistic. He *et al.* [1], who reviewed adversarial attacks on intrusion detectors, explain that attacks designed for images cannot simply be applied to network traffic: the modified traffic must still be valid and must still work as an attack. We therefore keep all changes within realistic feature ranges. We also recognise that this alone does not guarantee valid traffic, because many flow features depend on each other (for example, packet counts and total bytes), so we modify only features that can be changed without breaking these relationships.

## H. Standards, Frameworks and Industry Guidance

Section II-A used the NIST definition of data poisoning [2], but the NIST report offers more than a definition. It classifies attacks by the stage at which they occur, the attacker's goal (availability, integrity or privacy), and the attacker's knowledge and capabilities. For predictive systems, it identifies three main attack types: evasion, poisoning and privacy attacks. We based our threat model on this classification. Because the report is general guidance, however, it does not recommend specific defences for network security.

MITRE ATLAS [3] takes a more practical approach. Organised like MITRE ATT&CK, it describes the tactics and techniques attackers use against AI systems, supported by real case studies. It treats poisoning (AML.T0020, Poison Training Data) and evasion (AML.T0015, Evade ML Model) as separate techniques, matching the two attack types in our project. The OWASP Machine Learning Security Top 10 [17] lists input manipulation (ML01) and data poisoning (ML02) as its first two risks and recommends validating training data and using anomaly detection to reduce them. All three sources treat these attacks as serious threats, but none shows how a security team can test whether a defence actually works on its own data.

## I. Dataset

We selected CICIDS2017 [6], which is widely used in intrusion detection research. It contains five days of labelled network traffic, mostly normal activity, together with common attacks such as brute force, denial of service, web attacks, botnets and port scans, and each flow has more than 80 features. It is also more recent than NSL-KDD, the dataset used by Jang *et al.* [9].

CICIDS2017 does have problems. Lanvin *et al.* [7] found several errors in it: some records are mislabelled, some packets are out of order or duplicated, and some attacks were carried out but never labelled. They also showed that correcting these errors changes evaluation results. This matters greatly for our project. Label flipping is entirely about changing labels, so if some labels are wrong from the start, we could not tell which errors were caused by our attack and which were already present. We will therefore use the corrected version of the dataset [7] and always compare each poisoned model with a clean baseline trained on the same data. Malware datasets such as EMBER are left for future work.

## J. Critical Synthesis

Comparing the attack papers with the defence papers shows that attack research on intrusion detection has advanced much faster. For example, we know that the damage caused by label flipping depends on the target class, the data distribution and the attack settings [8], and that backdoor triggers on the most important features are far more successful than triggers on less important ones [9]. Defence research in this area has not yet caught up. A few studies, such as [10], were designed for intrusion detection, but most of the defences we reviewed were built and tested on images. In addition, each defence handles one type of attack well but has a known weakness against others. Table I summarises the main studies and methods reviewed, the data each was tested on, and how we use them. The dataset column makes the pattern clear: only the three intrusion detection studies used network data.

**Table I. Methods Reviewed and Their Role in This Project**

| Study / Method | Tested On | Strength | Limitation for Our Project | Use in PoisonGuard |
|---|---|---|---|---|
| Label flipping in federated IDS [8] | Intrusion detection data | Shows damage depends on target class, data and settings | Federated setting, not central training | Background; supports testing several rates and targets |
| Feature-importance backdoor [9] | NSL-KDD | Backdoor built for tabular NIDS data | Older dataset; low-importance triggers worked poorly | Background; we test the low-importance side of the trade-off |
| TDA poisoning detection [10] | Network intrusion data | One of the few defences built for NIDS | Covers poisoning only, not evasion | Reviewed only |
| Label sanitisation [11] | BreastCancer, MNIST, Spambase | Simple; works with any model | Struggles with clustered poisons | Defence, both models |
| Spectral signatures [12] | CIFAR-10 | Strong on deep models | Needs hidden layers; images only | Defence, MLP only |
| Partition-based ensemble [13], [14] | MNIST, CIFAR-10, GTSRB | Provable robustness; any base model | Extra training cost; images only | Defence, both models |

Several other issues emerged from the review. Attacks and defences are rarely tested together under the same conditions, so it is hard to tell which defence works best against which attack. Most papers focus on accuracy and say little about what matters to a security team, such as how many extra false alarms a defence causes. Benchmark datasets sometimes contain their own errors [7]. Finally, some studies, such as [8], examine poisoning in federated or collaborative intrusion detection, where several parties train one model together. We do not cover that setting; instead, we focus on detectors trained in one place.

## K. Research Gaps Addressed by This Project

Based on this review, the project addresses five gaps:

- ***Domain gap:*** Most poisoning and backdoor defences have been tested only on images, and hardly any were designed for intrusion detection. We apply well-known defences to tabular network data from CICIDS2017.
- ***Comparison gap:*** Most studies examine one attack or one defence in isolation. We test label-flipping, backdoor and evasion attacks against the defences relevant to each attack type, using the same data, models and metrics throughout.
- ***Model gap:*** We compare a tree-based model with a neural network, and we apply each attack and defence only to the models it can actually be applied to.
- ***Realism and practicality gap:*** We constrain our attacks to produce realistic traffic [1]. In addition to accuracy, we report the false-positive rate and execution time, as these matter to security teams.
- ***Data quality gap:*** Instead of older datasets such as NSL-KDD, we use the corrected version of CICIDS2017 [7] to reduce the effect of known labelling errors on our results.

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

[8] L. Lavaur, Y. Busnel, and F. Autrel, "Systematic analysis of label-flipping attacks against federated learning in collaborative intrusion detection systems," in *Proc. 19th Int. Conf. Availability, Rel. Secur. (ARES)*, 2024, pp. 1–12, doi: 10.1145/3664476.3670434.

[9] J. Jang, Y. An, D. Kim, and D. Choi, "Feature importance-based backdoor attack in NSL-KDD," *Electronics*, vol. 12, no. 24, Art. no. 4953, 2023, doi: 10.3390/electronics12244953.

[10] G. F. Monkam, M. J. De Lucia, and N. D. Bastian, "A topological data analysis approach for detecting data poisoning attacks against machine learning based network intrusion detection systems," *Comput. Secur.*, vol. 144, Art. no. 103929, 2024, doi: 10.1016/j.cose.2024.103929.

[11] A. Paudice, L. Muñoz-González, and E. C. Lupu, "Label sanitization against label flipping poisoning attacks," in *ECML PKDD 2018 Workshops*, Lecture Notes in Computer Science, vol. 11329. Cham, Switzerland: Springer, 2019, pp. 5–15, doi: 10.1007/978-3-030-13453-2_1.

[12] B. Tran, J. Li, and A. Madry, "Spectral signatures in backdoor attacks," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 31, 2018. [Online]. Available: <https://arxiv.org/abs/1811.00636>

[13] A. Levine and S. Feizi, "Deep partition aggregation: Provable defenses against general poisoning attacks," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2021. [Online]. Available: <https://arxiv.org/abs/2006.14768>

[14] K. Rezaei, K. Banihashem, A. Chegini, and S. Feizi, "Run-off election: Improved provable defense against data poisoning attacks," in *Proc. 40th Int. Conf. Mach. Learn. (ICML)*, PMLR, vol. 202, 2023, pp. 29030–29050. [Online]. Available: <https://proceedings.mlr.press/v202/rezaei23a.html>

[15] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards deep learning models resistant to adversarial attacks," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2018. [Online]. Available: <https://arxiv.org/abs/1706.06083>

[16] J. Chen, M. I. Jordan, and M. J. Wainwright, "HopSkipJumpAttack: A query-efficient decision-based attack," in *Proc. IEEE Symp. Secur. Privacy (SP)*, 2020, pp. 1277–1294, doi: 10.1109/SP40000.2020.00045.

[17] OWASP Foundation, "OWASP Machine Learning Security Top Ten (Draft v0.3)," 2023. [Online]. Available: <https://owasp.org/www-project-machine-learning-security-top-10/>
