# Branch F: Forensic Feature Analysis & Feature Selection

**Module**: Branch F (`src/forensics/branch_f/`)  
**Status**: VERIFIED & RESTRUCTURED  
**Date**: 2026-10-03  

---

## 1. Purpose & Relationship to Branches A–E

Branch F is a **meta-forensic stage** in Block 2. It does not extract new image-level features; instead, it operates downstream of Branches A–E to analyze, audit, and select subsets from the canonical 111-feature candidate pool.

```text
Processed Image [3, 256, 256]
      ↓
 ┌────┴───────────────────────────────┐
 A      B      C      D      E
(34)   (30)   (16)   (5)    (26)
 └────┬───────────────────────────────┘
      ↓
 111 Candidate Forensic Features
      ↓
 Branch F1 — Feature Analysis (Validity, Relevance, Redundancy, Complementarity, Stability, Compression)
      ↓
 Branch F2 — Feature Selection (Train-Only mRMR Baseline: 111 → 64 → 32 → 16 → 8)
      ↓
 Selected Forensic Feature Subsets & Manifests
      ↓
 Future Block 3 (Classifier Training)
```

---

## 2. Canonical 111-Feature Candidate Pool

The candidate feature representation is composed of:

| Branch | Domain Descriptor | Feature Count | Identifier |
|---|---|---:|---|
| **A** | Frequency & Periodicity (FFT + Synthbuster) | 34 | `fft_*`, `synth_*` |
| **B** | Multiscale Wavelet (3-level Haar DWT) | 30 | `LL3_*`, `{band}_*`, `detail_energy_ratio` |
| **C_LBP** | Local Texture (Uniform LBP) | 16 | `lbp_*` |
| **D_MFR** | Residual / Noise (3×3 Median Filter Residual) | 5 | `mfr_*` |
| **E** | Compression Forensics (DCT, Recompression, Phase, Grid) | 26 | `dct_*`, `ela_*`, `phase_*`, `grid_*` |
| **TOTAL** | **Canonical Candidate Pool** | **111** | |

Contract constraints verified:
- **0 duplicate feature names**.
- Deterministic extraction order.
- 100% finite values across real and AI images.

---

## 3. Branch F Architecture & Subpackages

Branch F is structured into two subpackages:

### F1: Feature Analysis (`src/forensics/branch_f/analysis/`)
1. **`validity.py` (`FeatureValidityAnalyzer`)**: Verifies 100% finite rates, variance, quantiles, and class-stratified statistics ($t$-tests).
2. **`relevance.py` (`FeatureRelevanceAnalyzer`)**: Evaluates univariate discriminative power using:
   - **Effective ROC-AUC**: $\max(\text{AUC}, 1 - \text{AUC}) \in [0.5, 1.0]$, treating inverse discriminators symmetrically.
   - **Mutual Information (MI)**: Non-parametric, nonlinear dependency estimation via $k$-nearest neighbors.
3. **`redundancy.py` (`FeatureRedundancyAnalyzer`)**: Computes $111 \times 111$ Pearson (linear) and Spearman (monotonic rank) correlation matrices. Flags redundant pairs with $|\rho_{\text{Spearman}}| \ge 0.90$.
4. **`complementarity.py` (`BranchComplementarityAnalyzer`)**: Aggregates domain-level relevance and generates 14 controlled branch ablation configurations.
5. **`generator_stability.py` (`GeneratorStabilityAnalyzer`)**: Audits per-feature effective AUC across the 5 individual AI generators (SD2.1, SDXL, SD3, DALL-E 3, Midjourney).
6. **`compression_analysis.py` (`CompressionSensitivityAnalyzer`)**: Measures degradation under symmetric JPEG compression across `Clean`, `Q95`, `Q80`, `Q60`, `Q40`, and `Q20`.

### F2: Feature Selection (`src/forensics/branch_f/selection/`)
1. **`mrmr.py` (`MRMRFeatureSelector`)**: Implements greedy Maximum Relevance Minimum Redundancy (mRMR) baseline:
   $$\text{score}(f_i) = \text{rel}(f_i) - \frac{1}{|S|} \sum_{f_s \in S} |\text{corr}(f_i, f_s)|$$
   - Fitted strictly on training data.
   - Generates deterministic ranked manifests for budgets: 111, 64, 32, 16, 8.
2. **`validation.py` (`DownstreamFeatureValidator`)**: Evaluates selected subsets using LightGBM on validation data to measure the performance–budget trade-off.

---

## 4. Leakage Controls & Rules

- **Train-Only Selection Rule**: All univariate relevance (MI/AUC), correlation matrices, and mRMR rankings are fitted exclusively on training set features ($N_{\text{train}} = 2,000$). Validation data never enters selection or thresholding.
- **Generator Identity Rule**: Generator identity (`Label_B`) is evaluation metadata only (used in Stage F stability diagnostics). It is never used as a detector feature or selection input.
- **Image-Only Rule**: No filenames, paths, captions, or image metadata enter the feature pipeline.
- **Symmetric Transformation Rule**: JPEG compression in Stage G is applied symmetrically to both real and AI-generated validation samples.

---

## 5. Measured & Verified Experimental Results

Measured on $N_{\text{train}} = 2,000$ (334 real, 1,666 AI), $N_{\text{val}} = 1,000$, seed=42.

### 5.1 Validity
- **111 / 111 features (100%)** produce finite values (0 NaN, 0 Inf).
- 0 constant features; 0 near-zero variance features.

### 5.2 Top Univariate Discriminators (Effective AUC & MI)
1. `dct_high_freq_ratio` (E): Eff AUC = 0.7196, MI = 0.0413
2. `lbp_bin_4` (C_LBP): Eff AUC = 0.7060, MI = 0.0403
3. `lbp_std_code` (C_LBP): Eff AUC = 0.6984, MI = 0.0417
4. `lbp_bin_0` (C_LBP): Eff AUC = 0.6939, MI = 0.0423
5. `lbp_nonuniform_ratio` (C_LBP): Eff AUC = 0.6901, MI = 0.0221
6. `dct_low_freq_ratio` (E): Eff AUC = 0.6818, MI = 0.0384
7. `synth_b_p8_y_mean` (A): Eff AUC = 0.6716, MI = 0.0487

### 5.3 Redundancy
- $111 \times 111$ correlation matrices are symmetric with unity diagonal.
- 104 pairs flagged at $|\rho| \ge 0.90$ (90 intra-branch, 14 cross-branch).
- Highest intra-branch redundancy: Branch D_MFR (mean $|\rho| = 0.7859$, 3 flagged pairs among 5 features).

### 5.4 Generator Stability
- ELA features (`ela_ratio_q90_q75`) and LBP features (`lbp_entropy`) exhibit high stability across all 5 generators (std $\le 0.054$).
- Branch A periodicity features exhibit larger variance across generators (std $\approx 0.08 - 0.13$).

### 5.5 Compression Sensitivity
- **62 / 111 features (55.9%)** retain $\ge 95\%$ of their AUC under Q60 JPEG compression.
- Most compression-robust features at Q20: `dct_mid_freq_ratio`, `dct_low_freq_ratio`, `grid_strength`, `LH3_absmean`, `LH3_energy`.

### 5.6 mRMR Selected Budgets & Downstream LightGBM Validation
- **Budget 8**: `synth_r_p8_y_mean` (A), `lbp_bin_8` (C), `synth_b_p8_x_mean` (A), `lbp_std_code` (C), `synth_b_p8_y_mean` (A), `grid_h_ratio` (E), `synth_r_p8_x_mean` (A), `synth_g_p8_y_mean` (A)
- **Downstream Validation Results**:

| Feature Budget | Feature Count | Val ROC-AUC | Val PR-AUC | Val F1 | Val Accuracy | TPR @ 1% FPR |
|---|---:|---:|---:|---:|---:|---:|
| **111 (Full)** | 111 | **0.9231** | 0.9828 | 0.9436 | 0.9020 | 0.5102 |
| **64** | 64 | 0.9144 | 0.9802 | 0.9402 | 0.8960 | 0.3469 |
| **32** | 32 | 0.9081 | 0.9785 | 0.9331 | 0.8840 | 0.2857 |
| **16** | 16 | 0.8272 | 0.9521 | 0.9235 | 0.8660 | 0.1633 |
| **8** | 8 | 0.7300 | 0.9214 | 0.9176 | 0.8540 | 0.0816 |

Ablation evidence:
- Removing Branch E causes the largest AUC drop: $-0.0199$ (from 0.9231 to 0.9032).
- Removing Branch B causes the second-largest drop: $-0.0161$ (to 0.9070).
- Multi-branch combinations strictly outperform any single branch (Branch A alone: 0.8548; Branch E alone: 0.8464).

---

## 6. Generated Persistent Artifacts

All 19 artifacts are preserved under `analysis/forensic_feature_analysis/`:

1. `feature_registry.csv` / `feature_registry.json`
2. `feature_validity.csv`
3. `univariate_relevance.csv`
4. `pearson_matrix.csv` / `spearman_matrix.csv`
5. `redundancy_pairs.csv` / `branch_redundancy_summary.csv`
6. `branch_complementarity.csv`
7. `generator_stability.csv`
8. `compression_analysis.csv`
9. `mrmr_rankings.csv`
10. `selected_features_8.json` / `_16.json` / `_32.json` / `_64.json` / `_111.json`
11. `validation_results.csv`
12. `run_metadata.json`

---

## 7. Limitations & Remaining Work

- **Baseline Status of mRMR**: mRMR is a baseline selection algorithm and does not capture complex feature interactions.
- **Sample Scaling**: Current artifacts were generated on a 2,000-sample training slice. Full-scale training set extraction (42,000 samples) can be executed when scaling up for Block 3 model training.
- **Future Scope**: Block 3 (Classifier Training), Block 4 (Evaluation & Robustness), and RGB baseline pipeline remain to be implemented.
