# PROJECT.md

# AI-Generated Image Detection — Research Project

## 1. Project Status

**Current phase:** Project initialization and dataset preparation

This document describes the current project state.

It is a living document and must be updated when a finalized project state changes.

---

# 2. Research Objective

Investigate whether carefully selected low-level forensic evidence can provide effective AI-generated image detection while reducing:

- training data requirements
- generator exposure requirements
- feature dimensionality
- model complexity
- inference cost

while retaining useful performance on:

- unseen generators
- compressed images
- relevant image transformations

The project should prioritize experimentally demonstrated generalization rather than performance caused by dataset-specific shortcuts.

---

# 3. Research Question

Can carefully selected low-level forensic evidence provide robust AI-image detection with reduced training data, generator exposure, feature dimensionality, model complexity, and inference cost, while remaining useful on unseen generators and compressed images?

---

# 4. Core Experimental Principles

## 4.1 Image-Based Detection

The detector should operate on the image itself.

The following must not be used as detector features unless a future decision explicitly changes this:

- captions
- generator identity
- dataset-specific metadata
- filenames
- file paths
- directory names
- source identifiers
- other information that directly reveals the class or source

Metadata may be retained for dataset auditing and evaluation.

---

## 4.2 Leakage Prevention

The project must actively investigate and control:

- train/test contamination
- duplicate images
- content leakage
- metadata leakage
- generator leakage
- resolution leakage
- format leakage
- compression leakage
- preprocessing leakage
- target leakage

A detector must not obtain its performance primarily from an unintended dataset shortcut.

---

## 4.3 Experimental Separation

Every experiment must clearly distinguish:

- training data
- validation data
- test data
- generator identity
- target label
- auxiliary metadata

Information used to construct or analyze an experiment must not automatically become model input.

---

# 5. Dataset

## Primary Dataset

Defactify Image Dataset

Source:

`Rajarshi-Roy-research/Defactify_Image_Dataset`

Current dataset state established by the project's dataset audit:

- Total: 96,000 images
- Train: 42,000
- Validation: 9,000
- Test: 45,000
- Real: 16,000
- AI-generated: 80,000
- Five AI generator categories
- 16,000 images per generator category

The raw dataset is stored under:

```text
data/defactify/
```

---

# 6. Dataset Labels

The dataset contains:

### Label_A

```text
0 = real
1 = AI-generated
```

This is the binary detection target.

### Label_B

The generator/source category label.

This may be used for:

- auditing
- generator-specific analysis
- evaluation
- constructing generator-disjoint experiments

It must not be supplied to the detector as an input feature.

### Caption

Caption/text information may be retained for:

- dataset analysis
- content-leakage analysis
- split analysis

It must not be supplied to the detector as an input feature.

---

# 7. Raw Data Policy

```text
data/defactify/
```

is treated as immutable raw data.

Do not modify files in this directory.

All preprocessing and derived artifacts must be stored separately.

---

# 8. Current Dataset Audit State

An initial dataset integrity audit has been completed.

The audit established the expected split totals, class/source counts, absence of null values in the audited fields, successful decoding of sampled images, and a bounded exact-duplicate check.

The audit also identified experimental confounders that require further investigation.

## Known issue: native resolution differences

Native image dimensions vary substantially between dataset sources/generator categories.

Therefore, native resolution must not become an unintended detector shortcut.

A common preprocessing strategy must be established before the main forensic experiments.

## Known issue: caption overlap

Captions are extensively reused across the dataset, including across real and generated images.

Captions are therefore not detector inputs.

Caption overlap between dataset splits must still be considered when evaluating content leakage.

## Known issue: class imbalance

The binary dataset contains substantially more AI-generated images than real images.

Therefore accuracy alone is not sufficient as a primary metric.

---

# 9. Preprocessing

A common preprocessing pipeline must be applied consistently to real and AI-generated images.

The exact spatial preprocessing strategy is currently **not finalized**.

Potential strategies must be evaluated for their effect on:

- forensic evidence
- resolution normalization
- aspect ratio
- image content
- artificial preprocessing artifacts

No candidate preprocessing method should be treated as finalized until verified.

---

# 10. Forensic Feature Families

The planned feature families include:

## Frequency-domain features

- FFT magnitude
- FFT phase
- radial frequency statistics
- frequency bands
- periodic spectral statistics

## Wavelet-domain features

- Haar DWT
- wavelet statistics

## Local texture features

- LBP
- GLCM

## Residual/error features

- high-pass residual statistics
- Laplacian statistics
- cross-difference statistics
- ELA-style statistics

## Transform/JPEG-related features

- 8×8 DCT statistics
- JPEG-related statistics

The final feature set is not yet finalized.

---

# 11. Feature Dimensionality

The initial planned forensic feature implementation is approximately 269 scalar features.

Feature reduction will be experimentally investigated.

Candidate feature counts include:

```text
269
128
64
32
16
```

These are experimental candidates, not yet finalized requirements.

---

# 12. Models

## Primary forensic model

LightGBM

## Secondary lightweight model

Tiny MLP

## RGB baseline

A lightweight RGB image classifier such as MobileNetV3-Small or ShuffleNetV2 is planned for comparison.

The final RGB baseline architecture is not yet finalized.

---

# 13. Evaluation Metrics

The project should report metrics appropriate for binary detection and class imbalance.

Planned metrics include:

- ROC-AUC
- PR-AUC
- F1
- TPR at fixed FPR
- per-generator AUC
- mean generator performance
- worst-generator performance

Efficiency reporting should include, where applicable:

- feature count
- model size
- parameter count
- FLOPs
- inference runtime
- memory/RAM usage

---

# 14. Generator Generalization

The project requires evaluation on unseen generators.

A planned strategy is generator-disjoint evaluation, including leave-one-generator-out experiments.

Conceptually:

```text
Training:
real + four AI generators

Testing:
real + one held-out AI generator
```

The exact split construction must be finalized and verified before the final experiments.

---

# 15. Compression Robustness

A planned robustness grid is:

```text
Clean
JPEG Q95
JPEG Q80
JPEG Q60
JPEG Q40
JPEG Q20
```

Compression must be applied symmetrically to real and AI-generated images.

The final implementation must ensure that the transformation itself does not introduce a class-specific shortcut.

---

# 16. Data Efficiency

The project intends to investigate performance as training data is reduced.

Candidate training sizes include:

```text
1K
5K
10K
20K
```

Generator diversity is another experimental variable.

The exact sampling and balancing protocol remains to be finalized.

---

# 17. Planned Research Pipeline

```text
Dataset
    ↓
Dataset Integrity Audit
    ↓
Leakage / Confound Audit
    ↓
Standardized Preprocessing
    ↓
Forensic Feature Extraction
    ↓
Feature Validation
    ↓
Feature Reduction
    ↓
Controlled Forensic Baseline
    ↓
Generator-Disjoint Evaluation
    ↓
Compression Robustness
    ↓
Data-Efficiency Experiments
    ↓
RGB Baseline
    ↓
Optional Fusion
    ↓
Efficiency Analysis
    ↓
Final Evaluation
```

---

# 18. Current Completed Work

- Primary dataset selected (`Defactify_Image_Dataset`).
- Raw dataset downloaded and stored under `data/defactify/`.
- Initial dataset integrity audit completed.
- Full 96,000-image dataset confound and leakage audit completed (`src/analysis/defactify_confound_audit.py`).
- Aspect-ratio confound discovered (100% AI square vs 97.46% real non-square) and documented.
- Deterministic AI generator resolutions and identical JPEG quantization tables (IJG Q75) discovered.
- Exact-duplicate and caption-overlap structures audited.
- Standardized spatial preprocessing finalized (DEC-008: largest centered square crop $\to$ 256×256 area-based resizing (`cv2.INTER_AREA`), implemented in `src/data/preprocessing.py`).
- Asymmetric Block 1 architecture finalized (DEC-009: materialized training data + on-demand evaluation).
- Canonical processed training dataset established at `data/processed/train/` with `cv2.INTER_AREA` (DEC-010).
- Lazy on-demand raw Parquet dataset loader implemented and verified (`src/data/loader.py`).
- Training dataset materializer implemented and verified (`src/data/materializer.py`).
- Materialized training dataset loader implemented and verified (`src/data/processed_loader.py`).
- Comprehensive 23-test suite implemented and verified (23/23 PASS, `src/data/tests/run_tests.py`).
- Architectural module map registry created (`docs/MODULE_MAP.md`).

---

# 19. Current Pending Work

1. Implement and verify Block 2 forensic feature extraction (FFT, DWT, LBP, GLCM, residuals, DCT).
2. Validate feature behavior and distribution shifts.
3. Establish feature-reduction methodology (selection to 128, 64, 32, 16).
4. Train the first controlled forensic baseline (LightGBM on materialized train features).
5. Implement generator-disjoint evaluation (leave-one-generator-out).
6. Evaluate compression robustness grid (clean, Q95, Q80, Q60, Q40, Q20).
7. Investigate data efficiency (1k, 5k, 10k, 20k sample regimes).
8. Establish lightweight RGB baseline (MobileNetV3-Small / ShuffleNetV2).
9. Evaluate optional RGB-forensic fusion architecture.
10. Measure computational efficiency and inference cost.
11. Conduct final evaluation and generate consolidated research findings.

---

# 20. Documentation Rule

This document describes the current verified project state.

Do not mark planned work as completed.

Do not convert hypotheses into facts.

Do not remove failed experiments or discovered problems from project history merely because they were inconvenient.

For historical changes, see `CHANGELOG.md`.

For finalized methodological decisions, see `DECISIONS.md`.
