# PROJECT.md

# AI-Generated Image Detection — Research Project

## 1. Project Status

**Current phase:** Block 2 — Image Analysis (Forensic Feature Extraction, Feature Analysis & Selection, and Persistent Forensic Dataset Materialization Completed; RGB Pipeline Pending).

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

The project prioritizes experimentally demonstrated generalization rather than performance caused by dataset-specific shortcuts.

---

# 3. Research Question

Can carefully selected low-level forensic evidence provide robust AI-image detection with reduced training data, generator exposure, feature dimensionality, model complexity, and inference cost, while remaining useful on unseen generators and compressed images?

---

# 4. Core Experimental Principles

## 4.1 Image-Based Detection

The detector must operate on the image itself.

The following must not be used as detector features:

- captions
- generator identity
- dataset-specific metadata
- filenames
- file paths
- directory names
- source identifiers
- other information that directly reveals the class or source

Metadata may be retained for dataset auditing, split construction, and evaluation.

---

## 4.2 Leakage Prevention

The project actively investigates and controls:

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

Source: `Rajarshi-Roy-research/Defactify_Image_Dataset`

Complete dataset structure established by the project's dataset audit:

- **Total Defactify Dataset:** 96,000 images
  - Train: 42,000 images (7 raw Parquet files × 6,000 rows each)
  - Validation: 9,000 images
  - Test: 45,000 images
- Real: 16,000 images
- AI-generated: 80,000 images
- Five AI generator categories (16,000 images per generator category across all splits):
  - 0: Real
  - 1: Stable Diffusion 2.1
  - 2: Stable Diffusion XL
  - 3: Stable Diffusion 3
  - 4: DALL-E 3
  - 5: Midjourney v6

The raw dataset is stored under:

```text
data/defactify/
```

### Current Processed Status

- **Block 1 Processed Dataset:** Currently materialized for the **train split only** (42,000 images at `data/processed/train/`).
- **Block 2 Forensic Dataset:** Materialized for the **train split only** (42,000 rows × 111 canonical forensic features at `data/forensic_dataset/features.parquet`).
- **Validation & Test Splits:** Will be deterministically processed and materialized from the raw Defactify validation (9,000) and test (45,000) splits later, after Block 2 is fully finalized.

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

The generator/source category label (0=Real, 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL-E3, 5=Midjourney).

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

An initial dataset integrity audit and a comprehensive 96,000-image confound audit (`src/analysis/defactify_confound_audit.py`) have been completed.

The audit established the expected split totals, class/source counts, absence of null values in audited fields, successful decoding of sampled images, and a bounded exact-duplicate check.

Key confounders identified and controlled:

## Known issue: native resolution differences
Native image dimensions vary substantially between dataset sources/generator categories (100% of AI images are square; 97.46% of real images are non-square). Controlled via canonical largest centered square crop followed by area-based resizing (`cv2.INTER_AREA` to 256×256) per DEC-008.

## Known issue: caption overlap
Captions are extensively reused across the dataset, including across real and generated images. Captions are strictly excluded from detector inputs.

## Known issue: class imbalance
The binary dataset contains substantially more AI-generated images than real images (5:1 ratio). Accuracy alone is not sufficient as a primary metric; ROC-AUC, PR-AUC, F1, and TPR@FPR must be reported.

---

# 9. Preprocessing

The canonical spatial preprocessing pipeline for all images in Block 1 is finalized (DEC-008, DEC-010):

```text
ORIGINAL IMAGE
    ↓
LARGEST CENTERED SQUARE CROP
    ↓
RESIZE TO 256 × 256 (cv2.INTER_AREA)
    ↓
STANDARDIZED PROCESSED IMAGE (RGB, uint8)
```

- Implemented in `src/data/preprocessing.py`.
- Materialized training data stored under `data/processed/train/` (42,000 samples, 42 part files, `manifest.json`).
- Verified by 23/23 passing tests in `src/data/tests/run_tests.py`.

---

# 10. Forensic Feature Families

The canonical forensic feature pool contains exactly **111 features** across five evidence branches:

| Branch | Subbranch | Domain / Description | Canonical Count |
|---|---|---|---:|
| **A** | `A1_FFT` (4) + `A2_SYNTH` (30) | Global spectral energy ratios & per-channel cross-difference periodicity | 34 |
| **B** | `B_HAAR` | 3-level separable 2D Haar DWT subband statistics (energy, absmean, entropy) | 30 |
| **C_LBP** | `C_LBP` | Rotation-invariant uniform LBP histogram & summary statistics | 16 |
| **D_MFR** | `D_MFR` | 3×3 median filter residual summary statistics | 5 |
| **E** | `E1_DCT` (10) + `E2_RESP` (8) + `E3_PHASE` (4) + `E4_GRID` (4) | 8×8 block DCT, controlled JPEG response (Q95..Q60), phase stability, 8×8 grid | 26 |
| **Total** | | **Canonical Forensic Feature Bank** | **111** |

### Alternative / Ablation Branches (Not in Canonical Pool):
- `C_GLCM`: 24 Haralick texture features.
- `C_LBP_EDGE`: 16 Canny edge-guided LBP features.
- `D_HIGHPASS`: 5 Gaussian high-pass residual features.
- `D_LAPLACIAN`: 5 2nd-order Laplacian residual features.

These alternatives are reserved for later ablation studies and are not part of the production 111-feature dataset.

---

# 11. Feature Analysis & Selection (Branch F)

Branch F implements a train-only feature analysis and selection pipeline (`src/forensics/branch_f/`):

- **Statistical Analysis:** Validity stats, univariate ROC-AUC, mutual information, 111×111 Pearson and Spearman redundancy matrices, branch complementarity summaries, generator stability diagnostics, and symmetric JPEG compression degradation (Q95..Q20).
- **Selection Algorithm:** Train-only Maximum Relevance Minimum Redundancy (mRMR) selector producing deterministic ranked feature lists.
- **Budget Views:** Subsets of 111, 64, 32, 16, and 8 features.
- **Persistent Analysis Artifacts:** 19 verified artifacts stored under `analysis/forensic_feature_analysis/`.

---

# 12. Persistent Forensic Dataset (Block 2 Output)

The persistent canonical forensic dataset is materialized under:

```text
data/forensic_dataset/
├── features.parquet                  (42,000 rows × 111 float32 features + 11 metadata columns, 25.88 MB)
├── dataset_manifest.csv              (42,000 rows identity manifest, 2.04 MB)
├── dataset_metadata.json             (Dataset summary and file index)
├── feature_registry.csv              (Machine-readable feature provenance)
├── feature_registry.json             (Full feature metadata schema)
└── selected/
    ├── features_8.parquet            (42,000 rows × 8 features + 11 metadata cols, 2.87 MB)
    ├── features_16.parquet           (42,000 rows × 16 features + 11 metadata cols, 4.52 MB)
    ├── features_32.parquet           (42,000 rows × 32 features + 11 metadata cols, 7.75 MB)
    ├── features_64.parquet           (42,000 rows × 64 features + 11 metadata cols, 14.69 MB)
    ├── features_111.parquet          (42,000 rows × 111 features + 11 metadata cols, 25.80 MB)
    ├── selected_features_8.json
    ├── selected_features_16.json
    ├── selected_features_32.json
    ├── selected_features_64.json
    └── selected_features_111.json
```

- Total directory size: **84 MB**.
- All 42,000 samples verified: 0 NaN, 0 Inf, strict metadata isolation, deterministic sample indices.
- Reader interface: `ForensicDataset` in `src/forensics/dataset.py` allows downstream consumers to load budgets, branch slices, Pandas DataFrames, or PyTorch Datasets with zero dependency on feature extraction code.

---

# 13. Models (Block 3 — Planned, Not Started)

Block 3 is planned to consume the persistent outputs of Block 2:

## Forensic Models
- **Primary:** LightGBM (tabular gradient-boosted decision trees).
- **Secondary:** Tiny MLP (compact neural model for nonlinear interaction comparison).

## RGB Models
- Candidate lightweight CNNs: MobileNetV3-Small, ShuffleNetV2.

## Fusion
- Optional feature-level or decision-level fusion will be evaluated only after independent baselines are established.

---

# 14. Evaluation Metrics (Block 4 — Planned, Not Started)

Planned evaluation metrics:
- ROC-AUC, PR-AUC, F1, TPR at fixed FPR (e.g. FPR=0.01, 0.05).
- Per-generator AUC, mean generator AUC, worst-generator AUC.
- Efficiency: feature count, model size, parameter count, FLOPs, inference runtime, RAM usage.

---

# 15. Generator Generalization Protocol (Planned)

Generator-disjoint evaluation (leave-one-generator-out) across the five Defactify AI generators:
- Train: real + four AI generators.
- Test: real + one held-out AI generator.
- Rotated across all five generators to measure generalization to unseen architectures.

---

# 16. Compression Robustness Protocol (Planned)

Symmetric JPEG evaluation grid:
```text
Clean, Q95, Q80, Q60, Q40, Q20
```
Applied symmetrically to real and AI images to evaluate degradation of forensic vs. RGB representations.

---

# 17. Planned Research Pipeline & Future Extension Path

The intended project progression is:

```text
1. Complete and freeze the RGB pipeline (Block 2).
2. Finalize Block 2 closeout.
3. Generate deterministic Block 1 validation (9k) and test (45k) processed splits.
4. Generate corresponding Block 2 validation and test forensic datasets.
5. Train and compare planned forensic (LightGBM, Tiny MLP) and RGB models (Block 3).
6. Perform Block 4 experiments (data budgets, feature budgets 8..111, model complexity, generator-disjoint generalization, compression robustness).
7. Future research extensions (larger feature banks, additional evidence families, dynamic/image-conditioned routing).
```

---

# 18. Current Completed Work

- Primary dataset selected (`Defactify_Image_Dataset`, 96k total images).
- Raw dataset stored under `data/defactify/`.
- Dataset integrity and 96k-image confound audit completed.
- Canonical spatial preprocessing finalized (DEC-008: largest centered square crop $\to$ 256×256 area-based resizing `cv2.INTER_AREA`).
- Block 1 training dataset materialized at `data/processed/train/` (42,000 images, `manifest.json`).
- Block 1 test suite verified: **23/23 PASS** (`src/data/tests/run_tests.py`).
- Forensic Branch A implemented and verified (34 features: A1 FFT=4, A2 Synthbuster=30).
- Forensic Branch B implemented and verified (30 features: 3-level Haar DWT).
- Forensic Branch C implemented and verified (16 canonical LBP features; GLCM and edge LBP alternatives).
- Forensic Branch D implemented and verified (5 canonical MFR features; Highpass and Laplacian alternatives).
- Forensic Branch E implemented and verified (26 features: E1 DCT=10, E2 Response=8, E3 Phase=4, E4 Grid=4).
- Branch F Feature Analysis & Selection implemented and verified (19 artifacts, mRMR selector, LightGBM validator).
- Block 2 Persistent Forensic Dataset materialized at `data/forensic_dataset/` (42,000 rows × 111 features, 11 metadata columns, 5 persistent selected budget views, registries, manifest).
- `ForensicDataset` reader interface implemented and verified (`src/forensics/dataset.py`).
- Block 2 test suite verified: **89/89 PASS** (`src/forensics/tests/run_tests.py`).
- Total test suite: **112/112 PASS**.

---

# 19. Current Pending Work

1. Complete and freeze the RGB representation pipeline in Block 2.
2. Finalize Block 2.
3. Generate deterministic Block 1 validation (9,000) and test (45,000) processed datasets.
4. Generate corresponding Block 2 validation and test forensic datasets.
5. Implement Block 3 forensic models (LightGBM, Tiny MLP) and RGB baseline models.
6. Conduct Block 4 experiments (training data budgets, feature budgets, generator-disjoint evaluation, compression robustness, efficiency).
7. Future Research Extensions (larger feature banks, additional evidence families, dynamic feature routing).

---

# 20. Documentation Rule

This document describes the current verified project state.

Do not mark planned work as completed.

Do not convert hypotheses into facts.

Do not remove failed experiments or discovered problems from project history merely because they were inconvenient.

For historical changes, see `CHANGELOG.md`.

For finalized methodological decisions, see `DECISIONS.md`.

