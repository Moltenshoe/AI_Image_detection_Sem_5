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

# Pending / Not Yet Finalized

The following remain open decisions:

- exact train/validation/test construction for custom experiments
- exact generator-disjoint split protocol
- final forensic feature set
- final feature-selection method
- final feature counts
- exact data-efficiency sampling protocol
- final RGB baseline architecture
- optional fusion architecture
- final compression experiment implementation details
