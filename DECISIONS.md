# DECISIONS.md

# Finalized Project Decisions

This file contains methodological decisions that have been explicitly finalized.

An item must not be added here merely because it is being considered.

If a finalized decision needs to change, the change must be explicitly recorded with a new decision entry and an explanation.

---

## DEC-001 — Primary Dataset

**Status:** FINAL

The Defactify Image Dataset is the primary dataset for this project.

---

## DEC-002 — Raw Dataset Is Immutable

**Status:** FINAL

The raw dataset under:

```text
data/defactify/
```

must not be modified in place.

All preprocessing and derived artifacts must be stored separately.

---

## DEC-003 — Detector Uses Image Information

**Status:** FINAL

The detector must use the image itself as its input.

The following must not be detector input:

- Caption
- Label_B
- generator identity
- filename
- filepath
- dataset metadata
- source-identifying information

`Label_A` is the supervised target label.

`Label_B` may be retained as evaluation metadata.

Caption information may be used for dataset/content-leakage analysis but must not be supplied to the detector.

---

## DEC-004 — Metadata Is Not Detector Evidence

**Status:** FINAL

Metadata may be used for:

- dataset auditing
- constructing experimental splits
- identifying generators during evaluation
- leakage analysis
- reporting per-generator results

Metadata must not be used as classifier input.

---

## DEC-005 — Generator-Disjoint Generalization

**Status:** FINAL

The project must include evaluation on an AI generator that was not used during training.

The purpose is to measure generalization beyond the generators exposed during training.

---

## DEC-006 — Symmetric Transformations

**Status:** FINAL

When evaluating image transformations such as compression, the same transformation must be applied to both real and AI-generated images.

A transformation must not be applied differently based on class.

---

## DEC-007 — Accuracy Alone Is Not Sufficient

**Status:** FINAL

Accuracy must not be the sole evaluation metric.

The project must report appropriate detection metrics, including:

- ROC-AUC
- PR-AUC
- F1
- TPR at fixed FPR
- per-generator performance

Efficiency metrics should also be reported for the lightweight forensic detector.

---

## DEC-008 — Canonical Spatial Preprocessing

**Status:** FINAL

The canonical preprocessing pipeline for all images in Block 1 is:

```
ORIGINAL IMAGE
    ↓
LARGEST CENTERED SQUARE CROP
    ↓
RESIZE TO 256 × 256 (area-based downsampling: cv2.INTER_AREA)
    ↓
STANDARDIZED PROCESSED IMAGE (RGB, uint8)
```

**Crop formula (exact):**
- `side = min(W, H)`
- `left = (W - side) // 2`
- `top  = (H - side) // 2`
- `box  = (left, top, left + side, top + side)`

**Rationale (from Defactify confound audit):**
- AI images are 100% square; real images are 97.46% non-square.
- A naive `resize(256, 256)` would introduce anisotropic geometric distortion that differs structurally between classes — a resolution/aspect-ratio confound.
- Centered square crop removes border context symmetrically; the operation is an identity for already-square images so AI images are untouched by the crop step.
- Canonical preprocessing uses largest centered square crop followed by 256×256 area-based resizing (`cv2.INTER_AREA`). Area-based downsampling is deterministic and applied identically across all samples.

**Constraints:**
- No resize before crop.
- No padding.
- No random operations.
- No ImageNet normalization at this stage.
- Only `Image["bytes"]` consumed; `Image["path"]` discarded.
- `Label_A` is the only target; `Label_B` and `Caption` go to audit metadata only.

**Implemented in:** `src/data/preprocessing.py`  
**Verified by:** `src/data/tests/test_preprocessing.py` (7/7 tests PASS)

---

## DEC-009 — Materialized Training Data & On-Demand Evaluation

**Status:** FINAL

The dataset ingestion and preprocessing architecture operates with asymmetric materialization:

1. **Training Data:**
   - Raw training Parquet files are preprocessed once and **materialized** into versioned Parquet files (`data/processed/train_v{version}/part-*.parquet`) alongside a JSON manifest (`manifest.json`).
   - Stored columns: `image_rgb: binary` ($256 \times 256 \times 3$ flat uint8 bytes, row-major) and `label_a: int32`.
   - All non-essential and leak-prone metadata (`path`, `caption`, `generator`, `source`, `filename`, `label_b`) is stripped from the image Parquet files.
   - `label_b` generator distributions are captured in `manifest.json` for auditing only.
   - Eliminates redundant JPEG decompression across multiple training epochs.

2. **Validation / Test Data:**
   - Processed **on demand** directly from raw Parquet files via `DefactifyDataset`.
   - Not permanently materialized by default to preserve disk storage and ensure evaluation remains dynamically coupled to raw data.

3. **Methodological Invariance:**
   - Both paths share the exact same `preprocess()` function from `src/data/preprocessing.py` (DEC-008). Zero train/eval preprocessing divergence is strictly verified.

4. **Derivation & Reproducibility:**
   - Raw Defactify data (`data/defactify/`) remains immutable.
   - Materialized training datasets are disposable, versioned derived artifacts that can be regenerated at any time.

**Implemented in:** `src/data/materializer.py`, `src/data/processed_loader.py`, `src/data/loader.py`  
**Verified by:** `src/data/tests/test_materializer.py` (Tests C–K PASS)

---

## DEC-010 — Canonical Processed Training Dataset Is data/processed/train/

**Status:** FINAL

The canonical processed training dataset is:

```text
data/processed/train/
```

It was built using the current preprocessing pipeline:

- Largest centered square crop (unchanged)
- `cv2.INTER_AREA` resize to 256×256 (DEC-008 — area-based downsampling)
- RGB uint8 output

**Versioning:** `PREPROCESSING_VERSION = "v2"` recorded in `manifest.json`.

**Statistics (RESULT):**
- Total samples: 42,000
- real (label_a=0): 7,000
- AI (label_a=1): 35,000
- Decode errors: 0

**Archival Dataset:**
`data/processed/train_old/` (formerly `train_v1`, produced with PIL bilinear resizing) is retained temporarily for audit/history only and is unreachable by default loaders.

**Implemented by:** `src/data/materializer.py` (`materialize_training_dataset()`)  
**Verified:** 2026-09-19 — all verification steps PASS.

---

## DEC-011 — Canonical 111-Feature Forensic Bank across Branches A–E

**Status:** FINAL

The canonical forensic feature bank consists of exactly 111 scalar float32 features across five low-level evidence branches:

1. **Branch A (Frequency & Periodicity):** 34 features
   - Subbranch A1 (FFT Spectral Ratios & Centroid): 4 features
   - Subbranch A2 (Synthbuster Cross-Difference Periodicity): 30 features (10 per RGB channel)
2. **Branch B (Haar Wavelet):** 30 features
   - Subbranch B (3-Level 2D Haar DWT): 2 LL3 statistics + 27 detail subband statistics + 1 detail energy ratio
3. **Branch C_LBP (Local Texture):** 16 features
   - Subbranch C_LBP: 10-bin rotation-invariant uniform LBP histogram + 6 summary statistics
4. **Branch D_MFR (Residual / Noise):** 5 features
   - Subbranch D_MFR: 5 summary statistics from 3×3 median filter residual
5. **Branch E (JPEG / Compression-Aware):** 26 features
   - Subbranch E1 (8×8 Block DCT Fingerprint): 10 features
   - Subbranch E2 (Multi-Quality Recompression Response Q95..Q60): 8 features
   - Subbranch E3 (Fourier Phase Stability under JPEG): 4 features
   - Subbranch E4 (Canonical 8×8 Grid Step Discontinuities): 4 features

**Ablation Alternatives (Not in Canonical Pool):**
`C_GLCM` (24), `C_LBP_EDGE` (16), `D_HIGHPASS` (5), `D_LAPLACIAN` (5) are isolated alternative candidates for ablation experiments and are not concatenated into the production 111-feature pool.

**Implemented in:** `src/forensics/pipeline.py`, `src/forensics/branch_a_frequency/`, `src/forensics/branch_b_wavelet/`, `src/forensics/branch_c_texture/`, `src/forensics/branch_d_residual/`, `src/forensics/branch_e/`  
**Verified by:** `src/forensics/tests/run_tests.py` (89/89 tests PASS)

---

## DEC-012 — Branch F Train-Only Feature Analysis & Selection Contract

**Status:** FINAL

Feature analysis and selection must follow strict scientific controls:

1. **Train-Only Isolation:** Univariate relevance (effective ROC-AUC, Mutual Information), redundancy estimation (111×111 Pearson and Spearman correlation matrices), and Maximum Relevance Minimum Redundancy (mRMR) feature selection must be computed strictly on training data (`split == "train"`). Validation and test data are strictly excluded from selection.
2. **Effective ROC-AUC:** Univariate discrimination is quantified as $\text{AUC}_{\text{eff}} = \max(\text{AUC}, 1 - \text{AUC}) \in [0.5, 1.0]$ to capture both direct and inverse discriminators symmetrically.
3. **Selected Feature Budgets:** Canonical evaluated budgets are 111, 64, 32, 16, and 8 features.
4. **Metadata Protection:** Generator identity (`label_b`), captions, file paths, and dataset split labels are strictly excluded from feature extraction and selection.

**Implemented in:** `src/forensics/branch_f/`  
**Verified by:** `src/forensics/tests/test_feature_analysis.py` (16/16 tests PASS)

---

## DEC-013 — Block 2 Persistent Forensic Dataset Materialization Architecture

**Status:** FINAL

The persistent Block 2 forensic dataset is materialized under `data/forensic_dataset/` as the immutable handoff artifact for downstream Block 3 models:

1. **Format & Separation:**
   - `features.parquet`: Full training split (42,000 rows) containing 11 metadata columns strictly separated from 111 float32 feature columns. Snappy compression.
   - 11 metadata columns: `image_id`, `sample_idx`, `split`, `source_path`, `source_file`, `source_row_group`, `source_row`, `label_a` (target), `label_b` (audit), `generator_name` (audit), `caption` (audit).
2. **Provenance & Manifests:**
   - `feature_registry.csv` and `feature_registry.json` map every feature to its subbranch, branch, domain, data type, and sensitivity.
   - `dataset_manifest.csv` records row-level image identity.
   - `dataset_metadata.json` records complete dataset statistics and file inventory.
3. **Selected Budget Views:**
   - `selected/features_{8,16,32,64,111}.parquet` materialize the mRMR-selected feature subsets for all 42,000 images, accompanied by `selected_features_{8,16,32,64,111}.json`.
4. **Decoupled Downstream Reader:**
   - `ForensicDataset` (`src/forensics/dataset.py`) provides the sole consumption interface for Block 3 and Block 4, with zero runtime dependency on feature extraction libraries or operators.

**Implemented in:** `src/forensics/dataset_materializer.py`, `src/forensics/dataset.py`  
**Verified by:** `src/forensics/tests/test_forensic_dataset.py` (12/12 tests PASS)

---

# Pending / Not Yet Finalized

The following remain open decisions:

- RGB pipeline representation extraction, analysis, and selection
- exact RGB baseline architecture (MobileNetV3-Small vs. ShuffleNetV2)
- final evaluation split construction for validation (9,000) and test (45,000) forensic/RGB datasets
- exact generator-disjoint split protocol (leave-one-generator-out rotation)
- exact data-efficiency sampling protocol (1k, 5k, 10k, 20k)
- optional fusion architecture (feature-level vs. decision-level)
- final compression robustness evaluation grid implementation details

