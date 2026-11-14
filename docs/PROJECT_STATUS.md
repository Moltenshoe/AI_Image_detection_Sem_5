# Project Status

Last documented status: 2026-11-14

## Completed

### Block 1 — Data Ingestion & Preprocessing
- Raw-data loading and deterministic verification (`src/data/loader.py`).
- Canonical preprocessing finalized (DEC-008, DEC-010: largest centered square crop $\to$ 256×256 area-based downsampling `cv2.INTER_AREA` $\to$ RGB uint8 in `src/data/preprocessing.py`).
- Block 1 training dataset materialized at `data/processed/train/` (42,000 images, `manifest.json`).
- Test suite: **23/23 tests PASS** (`src/data/tests/run_tests.py`).

### Block 2 — Forensic Feature Extraction & Pipeline
- **Branch A (Frequency & Periodicity):** 34 features (A1 FFT=4, A2 Synthbuster=30) — 11/11 tests PASS.
- **Branch B (Haar Wavelet):** 30 features (3-level 2D Haar DWT subband statistics) — 11/11 tests PASS.
- **Branch C (Local Texture):** 16 canonical LBP features (plus GLCM=24 and edge LBP=16 alternatives) — 15/15 tests PASS.
- **Branch D (Residual / Noise):** 5 canonical MFR features (plus Highpass=5 and Laplacian=5 alternatives) — 12/12 tests PASS.
- **Branch E (JPEG / Compression-Aware):** 26 features (E1 DCT=10, E2 Response=8, E3 Phase=4, E4 Grid=4) — 12/12 tests PASS.
- **Unified Forensic Pipeline:** `src/forensics/pipeline.py` orchestrating canonical 111-feature pool.

Canonical Forensic Candidate Pool:
```text
Branch A (Frequency):         34
Branch B (Haar Wavelet):      30
Branch C_LBP (Texture):       16
Branch D_MFR (Residual):       5
Branch E (Compression-aware): 26
────────────────────────────────
Total Canonical Bank:        111
```

### Block 2 — Feature Analysis & Selection (Branch F)
- Train-only statistical validity, univariate relevance (effective ROC-AUC & MI), 111×111 redundancy matrices (Pearson & Spearman), branch complementarity summaries, generator stability diagnostics, and symmetric JPEG degradation analysis.
- Maximum Relevance Minimum Redundancy (mRMR) selector producing deterministic ranked feature budgets (111, 64, 32, 16, 8).
- LightGBM downstream tabular validation.
- 19 persistent artifacts verified under `analysis/forensic_feature_analysis/`.
- Test suite: **16/16 tests PASS** (`src/forensics/tests/test_feature_analysis.py`).

### Block 2 — Persistent Forensic Dataset Materialization
- `data/forensic_dataset/` materialized over all 42,000 Block 1 training images (Snappy Parquet, 84 MB total):
  - `features.parquet` (42,000 rows × 111 float32 features + 11 metadata columns, 25.88 MB).
  - `feature_registry.csv` and `feature_registry.json` (explicit provenance: feature $\to$ subbranch $\to$ branch).
  - `dataset_manifest.csv` (sample identity manifest, 42k rows).
  - `dataset_metadata.json` (summary statistics and inventory).
  - `selected/features_{8,16,32,64,111}.parquet` (5 persistent budget views, 42,000 rows each).
  - `selected/selected_features_{8,16,32,64,111}.json` (selection manifests).
- `ForensicDataset` reader interface (`src/forensics/dataset.py`) for decoupled downstream consumption.
- Test suite: **12/12 tests PASS** (`src/forensics/tests/test_forensic_dataset.py`).

---

## Pending

### Block 2 — RGB Pipeline

#### Implementation and screening — completed
- RGB Branch A: Lab local-correlation maps.
- RGB Branch B: PiD BT.601 YCbCr residuals using the selected scale `s=0.2`.
- RGB Branch C: validation of the Branch B representation; it is not a separate feature-generation branch.
- Branch B screening experiments (A, B, B1, B2, C, D) and Stage 2 comparative analysis are complete. The accepted screening decisions are reflected in the active pipeline.
- NQP implementations and screening outputs are preserved as research-only; NQP is excluded from active RGB execution.
- The RGB implementation task reported **22/22 RGB tests PASS**. The combined pre-review-correction test result was **134/134 PASS** (RGB 22 + forensic 89 + Block 1 23).

#### Final review corrections — pending
1. In `src/rgb/branch_b_nqp/pid.py`, clip the rounded reconstructed RGB image to `[0, 255]` before calculating the residual, matching the documented PiD operation.
2. Add the missing `Any` import in `src/rgb/pipeline.py`.
3. Add the missing `Sequence` import in `src/rgb/materializer.py`.
4. Clean up stale NQP wording in relevant RGB materializer, dataset, and validation docstrings.
5. Clarify BT.601 YCbCr terminology while preserving compatibility-sensitive identifiers.
6. Document the input-scale heuristic limitation for unusually dark float images in `[0, 255]`; current production materialization paths use unambiguous input conventions.

The review reported that active Branch B execution uses PiD, NQP is not invoked by the active pipeline, compatibility aliases are retained, and no unrelated changes were identified. The local CPU benchmark was reported at approximately 1.36 ms/image; treat this as a machine-specific measurement, not a general performance guarantee.

#### Materialization — pending
- Materialize the active RGB representations across the required dataset splits.
- Verify sample identity, labels, split coverage, output shapes/dtypes, finite values, and reproducibility.
- Confirm that the persistent RGB dataset can be loaded by Block 3.

MobileNetV3-Small and ShuffleNetV2 are Block 3 RGB model candidates, not RGB representation-extraction backbones.

### Block 2 Closeout & Eval Split Preparation
- Apply the three important RGB code corrections and documentation clean-up listed above; rerun the RGB tests and relevant repository suites.
- Generate deterministic Block 1 validation (9,000) and test (45,000) processed datasets from the raw Defactify validation and test splits, if they have not already been materialized.
- Generate corresponding Block 2 validation and test forensic datasets.
- Generate the corresponding RGB representations for train, validation, and test splits.
- Verify both persistent Block 2 handoff datasets, then freeze Block 2.

### Block 3 — Models (Not Started)
- Forensic models: LightGBM baseline, Tiny MLP.
- RGB models: MobileNetV3-Small, ShuffleNetV2.
- Model comparison under identical data and feature budgets.
- Optional controlled fusion.

### Block 4 — Evaluation & Experiments (Not Started)
- Data-budget experiments (1k, 5k, 10k, 20k).
- Feature-budget experiments (111, 64, 32, 16, 8).
- Model complexity comparisons.
- Generator-disjoint evaluation (leave-one-generator-out across the 5 Defactify AI generators).
- Compression robustness evaluation grid (Clean, Q95, Q80, Q60, Q40, Q20).
- Computational efficiency and inference profiling.

### Future Research Extensions (Post-Block 4)
- Larger forensic feature banks (incorporating alternative branches: C_GLCM, C_LBP_EDGE, D_HIGHPASS, D_LAPLACIAN).
- Additional forensic evidence families.
- Image-conditioned / dynamic feature routing.

---

# Block 2 Definition of Done

```text
Canonical 111-feature forensic extraction (Branches A–E) [DONE]
                     +
Forensic feature analysis & mRMR selection (Branch F)   [DONE]
                     +
Persistent forensic training dataset (42k rows)         [DONE]
                     +
RGB representation implementation and screening         [DONE]
                     +
RGB final review corrections and regression tests       [PENDING]
                     +
RGB dataset materialization across required splits       [PENDING]
                     +
Validation/test split preparation and forensic outputs  [PENDING]
                     ↓
Two reproducible, independent Block 2 handoff datasets
```

The RGB screening and representation decision are complete; the remaining work is limited to the listed code corrections, regression tests, and persistent dataset materialization. Do not repeat the completed Branch B screening experiments unless a concrete defect is discovered.

---

# Test Summary

Last reported results from the RGB implementation task, **before the final review corrections**:

- Block 1 (`src/data/tests/run_tests.py`): **23 / 23 PASS**
- Forensic pipeline (`src/forensics/tests/run_tests.py`): **89 / 89 PASS**
- RGB pipeline (`src/rgb/tests/run_tests.py`): **22 / 22 PASS**
- **Combined reported total: 134 / 134 PASS**

The three important RGB corrections listed above have not yet been applied and retested. Therefore, 134/134 is the last reported pre-correction result, not a post-fix verification result. Re-run the relevant suites after applying the corrections.

