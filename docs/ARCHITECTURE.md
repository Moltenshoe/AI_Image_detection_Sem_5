# System Architecture

## 1. Purpose

This document defines the authoritative high-level architecture of the AI-generated image detection project.

The system is divided into four sequential blocks:

```text
Block 1
Data Loading + Preprocessing
        ↓
Stored Processed Data
        ↓
Block 2
Image Analysis
   ├── RGB
   └── Forensic
        ↓
Two Stored Block-2 Datasets
        ↓
Block 3
Models
        ↓
Predictions
        ↓
Block 4
Evaluation + Experiments
```

The architecture is intentionally modular so that later experiments can operate on persistent Block 1 and Block 2 artifacts rather than repeatedly recomputing earlier stages.

---

## 2. Block boundaries

### Block 1

Input:

- raw Defactify dataset

Output:

- canonical processed image representation

### Block 2

Input:

- Block 1 processed images (`data/processed/train/`)

Outputs:

- **Forensic Dataset:** Materialized at `data/forensic_dataset/features.parquet` (42,000 rows × 111 float32 features + 11 metadata columns) with 5 selected budget views (`selected/features_{8,16,32,64,111}.parquet`) and explicit registry manifests. [COMPLETED & VERIFIED]
- **RGB Dataset:** Learned representation dataset. [PENDING]

### Block 3 (Planned, Not Started)

Input:

- Block 2 persistent datasets (`data/forensic_dataset/`, RGB dataset)

Output:

- trained models (LightGBM, Tiny MLP, MobileNetV3-Small, ShuffleNetV2)
- predictions

### Block 4 (Planned, Not Started)

Input:

- Block 3 predictions and experiment logs

Output:

- performance analysis
- robustness analysis (compression grid)
- generator generalization analysis (generator-disjoint)
- efficiency analysis
- final comparisons

---

## 3. Block 2 completion definition

Block 2 is not complete when individual feature functions exist.

It is complete when:

```text
Block 1 processed data (train)
        ↓
 ┌──────┴──────┐
 ↓             ↓
Forensic      RGB
analysis      analysis
 ↓             ↓
selection     selection
 ↓             ↓
FOR_DATA      RGB_DATA
[COMPLETED]   [PENDING]
```

Both datasets must be reproducible and consumable independently by Block 3.

---

## 4. Forensic branch philosophy

The forensic branches cover five complementary low-level evidence domains:

- **Branch A (Frequency & Periodicity, 34 feats):** A1 global FFT spectral ratios + A2 Synthbuster per-channel cross-difference periodicity.
- **Branch B (Haar Wavelet, 30 feats):** 3-level 2D Haar DWT multiscale directional subband statistics.
- **Branch C_LBP (Local Texture, 16 feats):** Rotation-invariant uniform LBP histogram and summary statistics. (Ablations: C_GLCM=24, C_LBP_EDGE=16).
- **Branch D_MFR (Residual / Noise, 5 feats):** 3×3 median filter residual summary statistics. (Ablations: D_HIGHPASS=5, D_LAPLACIAN=5).
- **Branch E (JPEG / Compression-Aware, 26 feats):** E1 8×8 block DCT + E2 recompression response (Q95..Q60) + E3 phase stability + E4 canonical 8×8 grid.

Total canonical candidate bank = **111 features**.

Statistical overlap is expected; complementarity is measured empirically via 111×111 redundancy matrices, branch ablation, and mRMR selection.

---

## 5. RGB branch philosophy

The RGB branch represents conventional spatial/image information.

It exists as a separate path so that the project can answer:

- how much useful information is available from RGB learned representations?
- how does RGB compare with handcrafted forensic evidence?
- does combining RGB and forensic information provide measurable benefit?

Fusion is a later experiment, not a premise.

---

## 6. Persistent data principle

Each block produces a defined, immutable output artifact:

```text
Block 1 Output: data/processed/train/ (42,000 images, 256×256 RGB uint8)
       ↓
Block 2 Outputs:
  1. data/forensic_dataset/ (42,000 rows × 111 features + selected views 8..111)
  2. RGB persistent dataset (pending)
       ↓
Block 3: Consumes persistent datasets without recomputing upstream features
       ↓
Block 4: Evaluates models on frozen evaluation datasets
```

---

## 7. Future Extension Path

1. Complete and freeze the RGB pipeline (Block 2).
2. Finalize Block 2 closeout.
3. Generate deterministic Block 1 validation (9,000) and test (45,000) processed splits.
4. Generate corresponding Block 2 validation and test forensic datasets.
5. Train and compare planned forensic and RGB models in Block 3.
6. Perform Block 4 experiments (data budgets 1k..20k, feature budgets 8..111, model complexity, generator-disjoint generalization, compression robustness).
7. Future research extensions (larger feature banks with alternative branches, additional evidence families, dynamic/image-conditioned feature routing).

