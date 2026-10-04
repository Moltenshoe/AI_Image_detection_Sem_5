# Branch C — Texture

## Purpose

Branch C represents local spatial texture.

Three candidate pipelines exist, but they are alternatives rather than an automatic concatenation.

---

# C_LBP — Canonical branch

Count: 16.

Configuration:

- grayscale
- P = 8
- R = 1
- replicate padding
- rotation-invariant uniform LBP
- 10 histogram bins

Summary features:

- histogram statistics
- entropy
- uniformity
- dominant bin
- mean code
- standard deviation
- non-uniform ratio

C_LBP is the canonical C branch in the main 111-feature configuration.

---

# C_GLCM — Alternative

Count: 24.

Configuration:

- Ng = 16
- distances = 1, 2, 4
- angles = 0°, 45°, 90°, 135°
- symmetric
- normalized

Candidate statistics include:

- contrast
- dissimilarity
- homogeneity
- energy
- correlation
- entropy
- directional anisotropy statistics

C_GLCM is an alternative candidate pipeline for later ablation.

---

# C_LBP_EDGE — Alternative

Count: 16.

Pipeline:

```text
RGB
 ↓
grayscale
 ↓
Gaussian smoothing
 ↓
Canny edges
 ↓
LBP on edge pixels
```

Specified parameters:

- Gaussian σ = 1
- kernel = 5×5
- Canny = 50/100
- Sobel = 3×3

Candidate features include:

- edge LBP histogram
- edge summary statistics
- edge density

---

## Scientific role

C is intended to capture local texture/statistical structure rather than global spectrum.

Potential overlap with B and D is expected.

The purpose of feature selection is to determine whether C contributes information that remains useful after frequency and residual evidence are considered.

## Canonical choice

C_LBP = 16 features.

## Implementation & Verification Status

- **Status:** Implemented & Verified.
- **Modules:** `src/forensics/branch_c_texture/lbp.py`, `src/forensics/branch_c_texture/glcm.py`, `src/forensics/branch_c_texture/features.py`.
- **Verification:** 15/15 tests PASS in `src/forensics/tests/test_branch_c_texture.py`.
- **Dataset Integration:** 16 canonical C_LBP features materialized into `data/forensic_dataset/features.parquet`.
- **Ablation Candidates:** C_GLCM (24) and C_LBP_EDGE (16) implemented as independent alternatives for ablation experiments.

