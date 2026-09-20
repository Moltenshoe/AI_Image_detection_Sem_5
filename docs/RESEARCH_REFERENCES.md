# Research References

This document records literature that informs the design. The cited papers provide motivation or provenance; they do **not** automatically validate the project's exact feature definitions.

## 1. Frequency-domain evidence

Karageorgiou et al., **Any-Resolution AI-Generated Image Detection by Spectral Learning**, CVPR 2025.

The work describes spectral artifacts in generated images and investigates spectral learning for AI-generated image detection, including robustness to online perturbations.

https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html

---

## 2. DCT traces

Pontorno, Guarnera, Battiato, **On the Exploitation of DCT-Traces in the Generative-AI Domain**, 2024.

The paper studies DCT coefficient distributions in generated images and examines discriminative combinations and JPEG robustness.

https://arxiv.org/abs/2402.02209

The project uses this as motivation for E1, but E1 is a compact aggregate candidate set and is not a reproduction of the paper's detector.

---

## 3. Dataset/compression bias

Grommelt et al., **Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets**, 2024.

The paper demonstrates that dataset-level JPEG and image-size biases can affect detector behavior and cross-generator performance.

https://arxiv.org/abs/2403.17608

This supports the project's strict separation of image evidence from container/metadata shortcuts and its explicit compression experiments.

---

## 4. Compression and phase

Li et al., **Detecting Compressed AI-Generated Images via Phase Spectrum Robustness**, CVPR 2026.

The paper studies phase-spectrum behavior under compression and develops a compression-robust detector.

https://openaccess.thecvf.com/content/CVPR2026/html/Li_Detecting_Compressed_AI-Generated_Images_via_Phase_Spectrum_Robustness_CVPR_2026_paper.html

The project uses this as motivation for E3. E3 is not a reproduction of the CVPR 2026 architecture.

---

## 5. Controlled JPEG response

Mandala, **Format-Controlled Multi-Scale JPEG Compression Response Analysis for Image-Level Forgery Screening**, 2026.

The work studies multi-quality JPEG response/ELA-style features and emphasizes format-controlled evaluation.

https://arxiv.org/abs/2607.06615

The project uses this as motivation for E2, but E2 is designed for AI-image detection and is not a claim to reproduce that work.

---

## 6. Data efficiency / unseen generators

Wu et al., **Few-Shot Learner Generalizes Across AI-Generated Image Detection**, ICML 2025.

The paper explicitly addresses performance decline on unseen generators and the cost of collecting large amounts of training data.

https://proceedings.mlr.press/v267/wu25r.html

This supports making training-data budget and generator-disjoint evaluation explicit experimental dimensions.

---

## 7. Generalization

He et al., **Leveraging Arbitrary Data Sources for AI-Generated Image Detection Without Sacrificing Generalization**, CVPR 2026 Findings.

The work emphasizes the difficulty of generalizing to unseen generative models and explores compact attribution representations.

https://openaccess.thecvf.com/content/CVPR2026F/html/He_Leveraging_Arbitrary_Data_Sources_for_AI-Generated_Image_Detection_Without_Sacrificing_CVPRF_2026_paper.html

---

## 8. Feature relevance and redundancy

Gong et al., **A new filter feature selection algorithm for classification task by ensembling Pearson correlation coefficient and mutual information**, Engineering Applications of Artificial Intelligence, 2024.

The paper combines linear correlation and nonlinear mutual information while considering feature redundancy.

https://www.sciencedirect.com/science/article/pii/S095219762400023X

This supports the project's plan to use both correlation and nonlinear dependency analysis rather than relying on one statistic.

---

## 9. Feature redundancy

Feature-selection literature commonly distinguishes:

- relevance to the target;
- redundancy among selected features.

A maximum-relevance/minimum-redundancy formulation is therefore appropriate as a conceptual basis for the selection stage.

The project does not need to claim a novel feature-selection algorithm unless one is actually developed.

---

# Research-use rule

Literature provides:

- motivation;
- prior observations;
- methodological context;
- terminology;
- possible controls.

Literature does not prove that the project's exact 111-feature candidate set works.

The project's exact feature effectiveness remains a Block 4 empirical question.
