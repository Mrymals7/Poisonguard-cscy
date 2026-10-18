# II. Literature Review — Sections A to E

## A. Adversarial Machine Learning and Data Poisoning

Machine learning is widely used in cybersecurity, particularly in NIDS, which examine patterns in network traffic to decide whether a flow is normal or malicious. However, ML models are highly sensitive to the quality of their training data. If an attacker can manipulate this data, the model learns incorrect patterns and makes wrong predictions.

According to the National Institute of Standards and Technology (NIST), adversarial machine learning attacks can target any stage of the ML lifecycle. Vassilev *et al.* [2] define data poisoning as the insertion or modification of malicious samples in the training data in order to influence the final model. However, the NIST report provides a broad taxonomy and guidance rather than experimental results, and it does not directly evaluate the impact of different poisoning attacks on intrusion detection models such as Random Forest and MLP classifiers.

Cinà *et al.* [4] show that poisoning attacks can have several goals. Indiscriminate attacks aim to degrade the model's overall performance, while targeted attacks focus on specific classes or samples. A third type is the backdoor attack, in which the attacker hides behaviour that is activated only when a specific trigger appears. Their study, however, is a broad survey of poisoning attacks rather than one focused on intrusion detection, which is the emphasis of PoisonGuard. These attacks are highly relevant to intrusion detection, because a poisoned model may misclassify malicious traffic as benign while still handling benign traffic normally.

## B. Label-Flipping Attacks

Label flipping is one of the main forms of poisoning. In supervised learning, each training example has a label indicating its class; for example, a network flow is labelled either benign or malicious. In a label-flipping attack, the attacker changes some of these labels, for example from malicious to benign. The features of the flow stay the same, but the model receives the wrong information during training. If enough labels are flipped, the model can learn that certain malicious behaviour is normal.

Label flipping can be random or targeted. Random flipping changes labels without focusing on any specific attack class, whereas targeted flipping relabels selected malicious flows as benign.

Lavaur *et al.* [8] studied label-flipping attacks in collaborative intrusion detection systems and showed that the effect of the attack depends on several factors, including the targeted class, the data distribution and the attack parameters. Their work, however, focuses on collaborative and federated training, whereas PoisonGuard uses centralised model training, so their results may not transfer fully to our setting.

Their findings also show that the fraction of poisoned samples alone is not enough to describe the effect of a poisoning attack. PoisonGuard therefore tests both random and targeted label flipping across a range of poisoning rates (1% to 20%) rather than at a single rate.

## C. Backdoor Poisoning Attacks

PoisonGuard also studies backdoor attacks. Rather than only changing labels, the attacker embeds a distinctive pattern, called a trigger, into some training samples. During training, the model learns to associate the trigger with a chosen target class. For example, an attacker can add a trigger to malicious traffic labelled as benign, so that the model learns that the presence of the trigger means benign traffic. The danger is that the model can still perform well when the trigger is absent, so standard accuracy tests may not reveal that it has been compromised.

Jang *et al.* [9] studied backdoor attacks on network intrusion detection using the NSL-KDD dataset. Instead of choosing trigger features at random, they selected them based on feature importance, and showed that triggers on high-importance features were far more successful than triggers on randomly selected features; in their main model, the random trigger succeeded in only 10% of attacks. Their study is limited by its reliance on the older NSL-KDD dataset and its focus on high-importance trigger features. PoisonGuard instead uses CICIDS2017 and applies a fixed trigger to a few low-importance features. The project will therefore explore whether this subtler trigger can still produce an effective backdoor while keeping feature values valid.

## D. Data Poisoning in Network Intrusion Detection

Poisoning network intrusion data is generally harder than poisoning image data, because network traffic is tabular. Each feature has a specific meaning and a valid range of values, and changing a value without respecting these limits can produce unrealistic traffic that is easier to detect.

Monkam *et al.* [10] studied data poisoning against ML-based NIDS. Their work highlights that standard ML pipelines usually assume a trustworthy training environment, an assumption that is dangerous in cybersecurity, where attackers may deliberately poison training samples. Their focus, however, is on detecting poisoned samples rather than directly evaluating the impact of different poisoning attacks on Random Forest and MLP models. Their study also used intrusion detection datasets such as CICIDS2017, which makes it particularly relevant to this project.

The literature also indicates that ML algorithms are not equally affected by poisoning. Jang *et al.* [9] compared the effectiveness of backdoor attacks across several machine learning and deep learning models, showing that comparing model types gives a better understanding of the threat. PoisonGuard therefore compares a tree-based model (Random Forest) with an MLP neural network rather than testing a single classifier.

## E. Summary of Poisoning Attacks

Recent studies show that data poisoning is a significant threat to ML-based intrusion detection. Label-flipping attacks can teach a model the wrong relationship between malicious and benign traffic, while backdoor attacks can implant hidden behaviour that activates only when a trigger is present. Research also shows that attack success depends on the poisoning rate, the chosen features, the target class, the dataset and the model. These findings support PoisonGuard's strategy of testing random and targeted label flipping and backdoor poisoning on CICIDS2017 and comparing their effects on Random Forest and MLP models. The following subsections discuss the defensive strategies used to detect or mitigate these attacks.

---

## References

*Numbering matches the full Deliverable 1 report.*

[2] A. Vassilev, A. Oprea, A. Fordyce, H. Anderson, X. Davies, and M. Hamin, "Adversarial machine learning: A taxonomy and terminology of attacks and mitigations," Nat. Inst. Standards Technol., Gaithersburg, MD, USA, Rep. NIST AI 100-2e2025, 2025, doi: 10.6028/NIST.AI.100-2e2025.

[4] A. E. Cinà *et al.*, "Wild patterns reloaded: A survey of machine learning security against training data poisoning," *ACM Comput. Surv.*, vol. 55, no. 13s, pp. 1–39, 2023, doi: 10.1145/3585385.

[8] L. Lavaur, Y. Busnel, and F. Autrel, "Systematic analysis of label-flipping attacks against federated learning in collaborative intrusion detection systems," in *Proc. 19th Int. Conf. Availability, Rel. Secur. (ARES)*, 2024, pp. 1–12, doi: 10.1145/3664476.3670434.

[9] J. Jang, Y. An, D. Kim, and D. Choi, "Feature importance-based backdoor attack in NSL-KDD," *Electronics*, vol. 12, no. 24, Art. no. 4953, 2023, doi: 10.3390/electronics12244953.

[10] G. F. Monkam, M. J. De Lucia, and N. D. Bastian, "A topological data analysis approach for detecting data poisoning attacks against machine learning based network intrusion detection systems," *Comput. Secur.*, vol. 144, Art. no. 103929, 2024, doi: 10.1016/j.cose.2024.103929.
