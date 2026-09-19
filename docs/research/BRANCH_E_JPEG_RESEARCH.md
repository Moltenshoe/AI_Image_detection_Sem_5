# docs/research/BRANCH_E_JPEG_RESEARCH.md

# Branch E: JPEG & Compression-Artifact Forensics — Literature & Scientific Design Record

**Module:** `src/forensics/branch_e/`  
**Research Block:** Block 2 (Forensic Feature Extraction)  
**Status:** IMPLEMENTED & VERIFIED (26 candidate features, 12/12 unit tests PASS, 61/61 forensic regression PASS)  
**Target Canonical Input:** RGB uint8 $256 \times 256$ tensor (Block 1 canonical representation, DEC-008 / DEC-010)

---

## 1. Executive Summary & Purpose

Branch E is the **JPEG / Compression-Artifact Forensics Branch** of the lightweight forensic detection pipeline.

### Scientific Objective
Investigate whether:
1. Local 2D DCT-domain basis coefficient distributions and directional fingerprints,
2. Spatial $8 \times 8$ block-grid boundary statistics on canonical images, and
3. Differential error and phase response curves under controlled in-memory JPEG compression

provide generalizable forensic evidence capable of discriminating real camera images from AI-generated images, while maintaining lightweight scalar representation (compatible with LightGBM and Tiny MLP) and zero reliance on fragile container metadata.

---

## 2. Primary Literature Review & Provenance Record

This section establishes the exact scientific literature informing Branch E's revised architecture.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                PRIMARY CITATION & PROVENANCE MAP                               │
├──────────────────────────┬──────────────────────┬──────────────────────────────────────────────┤
│ Literature Reference     │ Core Forensic Idea   │ Adaptation for Branch E                      │
├──────────────────────────┼──────────────────────┼──────────────────────────────────────────────┤
│ Pontorno et al.          │ Positional AC DCT    │ E1: Zonal and positional AC DCT fingerprints │
│ ICIP 2024                │ fingerprints in GenAI│ (low/mid/high/anisotropy)                    │
├──────────────────────────┼──────────────────────┼──────────────────────────────────────────────┤
│ Mandala                  │ Multi-quality JPEG   │ E2: Multi-quality controlled recompression   │
│ arXiv 2607.06615 (2026)  │ recompression curves │ response slope, cross-quality ratios, Gini   │
├──────────────────────────┼──────────────────────┼──────────────────────────────────────────────┤
│ Li et al.                │ Fourier phase        │ E3: Phase spectrum stability under controlled│
│ CVPR 2026                │ compression stability│ JPEG recompression                           │
├──────────────────────────┼──────────────────────┼──────────────────────────────────────────────┤
│ Fan & de Queiroz (2003)  │ 8x8 block boundary   │ E4: Canonical grid step discontinuity vs     │
│ Wang et al. (2009)       │ step discontinuity   │ intra-block texture (qualified)              │
├──────────────────────────┼──────────────────────┼──────────────────────────────────────────────┤
│ Grommelt et al. (2024)   │ JPEG & resolution    │ Section 4: Image-only representation, strict │
│ B-Free (CVPR 2025)       │ dataset confounding  │ symmetric transformation, metadata isolation │
└──────────────────────────┴──────────────────────┴──────────────────────────────────────────────┘
```

---

### A. DCT-Based AI-Generated Image Forensics

- **Citation:** Orazio Pontorno, Luca Guarnera, Sebastiano Battiato, *"On the Exploitation of DCT-Traces in the Generative-AI Domain"*, IEEE International Conference on Image Processing (ICIP), 2024. arXiv:2402.02209.
- **Concept Taken:** 
  - Generative synthesis models (GANs, Latent Diffusion Models) introduce characteristic non-uniformities across the 64 2D-DCT basis frequencies ($8 \times 8$ block DCT domain).
  - Individual AC DCT coefficient locations $(u, v)$ carry differing amounts of discriminative information rather than a uniform distribution.
  - Structural low-to-mid frequency AC traces survive subsequent lossy JPEG compression because they stem from generative upsampling/decoder architectures.
- **What is Adapted:** 
  - Instead of pooling all 63 AC coefficients into one monolithic average, we extract targeted zonal statistics: Low-frequency AC ($u+v \in [1, 3]$), Mid-frequency AC ($u+v \in [4, 7]$), High-frequency AC ($u+v \ge 8$), and directional horizontal vs vertical AC energy asymmetry.
  - Non-parametric distribution shape descriptors: AC energy, excess kurtosis, sparsity ratio, and Benford's First Digit Law divergence.
- **What is NOT Adopted:** 
  - Deep per-pixel feature maps or LIME interpretability masks.
  - 64 distinct per-coefficient models (which would inflate dimensionality and overfit).
- **Limitations:** On preprocessed/resized images, DCT coefficients reflect the canonical spatial representation; they do not recover the camera's original uncompressed DCT coefficients.

---

### B. Phase Spectrum Robustness Under Compression

- **Citation:** Kai Li, Wenqi Ren, Wei Wang, Xiaochun Cao, *"Detecting Compressed AI-Generated Images via Phase Spectrum Robustness"*, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.
- **Concept Taken:** 
  - Under lossy JPEG compression, Fourier magnitude spectra suffer severe degradation (high-frequency suppression and quantization noise), while the Fourier phase spectrum retains structural edge coherence and generative phase alignment.
- **What is Adapted:** 
  - We evaluate the *stability of the Fourier phase spectrum* when subjected to controlled in-memory JPEG compression ($Q=90$ and $Q=75$):
    $$\Delta\Phi_Q = \angle \mathcal{F}(I) - \angle \mathcal{F}(\text{JPEG}_Q(I))$$
  - Compact scalar descriptors: Phase cosine similarity / correlation, Phase difference angular energy, and High-frequency phase stability ratio.
- **What is NOT Adopted:** 
  - The heavy "Compression-Robust Phase-Harmonized Transformer" neural architecture or cross-modal attention layers.
- **Branch Assignment Rationale:** Because phase compression stability requires a *controlled recompression transformation*, it is functionally a compression-response feature (Branch E) rather than a static Fourier feature (Branch A).

---

### C. Multi-Quality Controlled Compression Response

- **Citation:** P. Mandala, *"Format-Controlled Multi-Scale JPEG Compression Response Analysis for Image-Level Forgery Screening"*, arXiv:2607.06615, 2026.
- **Concept Taken:** 
  - Applying controlled recompression across a standard set of JPEG qualities generates a deterministic *error response curve*.
  - Differential metrics across quality levels (e.g., error slope between $Q=90$ and $Q=75$, cross-quality ratios) decouple structural compressibility from absolute image contrast.
- **What is Adapted:** 
  - Controlled symmetric in-memory recompression across a compact quality grid: $Q \in \{95, 90, 75, 60\}$.
  - Extracting curve descriptors: Mean error at each $Q$, Error Energy at $Q90$, Compressibility Slope ($\frac{\Delta_{75} - \Delta_{90}}{15}$), Cross-Quality Ratio ($\Delta_{90} / \Delta_{75}$), and Spatial Error Sparsity (Gini coefficient / Entropy).
- **What is NOT Adopted:** 
  - 405-dimensional vector or multi-scale SRM residual combinations.
  - Splicing/forgery localization claims (AI generation is full-image synthesis, not localized copy-move).

---

### D. Dataset Biases, JPEG Confounders, and Leakage Control

- **Citations:**
  - Patrick Grommelt, Louis Weiss, Franz-Josef Pfreundt, Janis Keuper, *"Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets"*, ECCV Workshops, 2024. arXiv:2403.17608.
  - Davide Cozzolino et al., *"A Bias-Free Training Paradigm for More General AI-generated Image Detection"*, IEEE/CVF CVPR, 2025. arXiv:2412.17671.
- **Findings Relevant to Defactify:**
  - Detectors easily overfit to container format differences (e.g. Real=JPEG vs Fake=PNG) or resolution differences rather than AI synthesis artifacts.
  - In Defactify, observed JPEG quantization tables in the Defactify audit match the standard IJG Q75 table across all classes, but native resolutions vary sharply (SDXL=1024, SD2.1=768, DALL-E 3=270, Real=variable non-square).
- **Strict Methodological Rules Enforced in Branch E:**
  1. **DEC-003 / DEC-004:** Absolute prohibition of raw JPEG container parsing, EXIF reading, filename parsing, or quantization table inspection.
  2. **Canonical Input Only:** Input is strictly the preprocessed RGB uint8 $256 \times 256$ image (`cv2.INTER_AREA`).
  3. **Symmetric In-Memory Recompression:** Every transformation is applied identically to Real and AI images using the same codec.

---

## 3. Critical Scientific Correction on Data Representation

### The Preprocessing Gap
$$\text{Raw Dataset (Parquet bytes)} \xrightarrow{\text{Decode}} \text{Native RGB} \xrightarrow{\text{Centered Square Crop}} \text{Cropped RGB} \xrightarrow{\text{cv2.INTER\_AREA}} \text{Canonical RGB } 256 \times 256$$

Because the canonical image is decoded and resized:
1. **No Original JPEG Grid Preservation:** For an image downsampled from $1024 \times 1024$ to $256 \times 256$ (scale factor 4), the original $8 \times 8$ JPEG blocks are compressed to $2 \times 2$ pixel patches. For non-integer scalings (e.g., $640 \to 256$), the original grid is interpolated.
2. **No Quantization Dead-Zone Equivalence:** Computing an $8 \times 8$ DCT on the float canonical image does NOT reconstruct the integer coefficients of the camera's original JPEG codec. Small AC coefficients ($|C| < 1.0$) represent continuous spatial smoothness in the canonical frame, not integer quantization zeros.
3. **Qualified Terminology:** 
   - We do NOT label small DCT values as "JPEG quantization dead-zones".
   - We do NOT claim $8 \times 8$ boundary discontinuities are verified camera JPEG grid artifacts.
   - We define them as **canonical $8 \times 8$ block DCT energy distributions** and **canonical grid boundary step ratios**.

---

## 4. Evidence Classification & Scientific Status

In accordance with `AGENTS.md` Rule 7:

- **FACT:** The canonical input is RGB uint8 $256 \times 256$ produced by `cv2.INTER_AREA`.
- **FACT:** Observed JPEG quantization tables in the Defactify audit match the standard IJG Q75 table across all 96,000 images.
- **DECISION (DEC-003, DEC-008):** Detector inputs must be image-only without metadata, EXIF, or container headers.
- **HYPOTHESIS:** Zonal $8 \times 8$ DCT energy concentrations and multi-quality controlled recompression response curves provide complementary discriminative evidence when paired with spatial and frequency branches (A–D).
- **CANDIDATE:** The proposed 26 features are candidate descriptors; empirical selection/reduction is deferred to Block 4.
- **LIMITATION:** Linear spatial downsampling (`cv2.INTER_AREA`) alters high-frequency DCT harmonics, making cross-generator generalization testing mandatory.

---

## 5. Method Evaluation & Decision Matrix

| Candidate Method | Category | Decision | Technical Rationale |
|:---|:---|:---:|:---|
| **Zonal $8 \times 8$ AC DCT Energy Ratios** | DCT | **KEEP** | Compact, captures low/mid/high spectral decay across 64 DCT bases (Pontorno et al. 2024). |
| **AC Kurtosis & Sparsity** | DCT | **KEEP** | Non-parametric distribution shape (Laplacian vs Gaussian generative tails). |
| **Benford's First Digit Law SSD** | DCT | **KEEP** | Measures statistical conformity of rounded DCT coefficients to logarithmic distribution. |
| **Controlled Multi-Q ELA Error Curve** | Recompression | **KEEP** | $Q \in \{95, 90, 75, 60\}$ recompression slope characterizes structural compressibility. |
| **Cross-Quality ELA Error Ratio** | Recompression | **KEEP** | Content-normalized compressibility metric ($\Delta_{90}/\Delta_{75}$). |
| **Spatial Error Gini & Entropy** | Recompression | **KEEP** | Captures whether recompression error is localized to edges or diffuse across smooth regions. |
| **Fourier Phase Stability under JPEG** | Phase Response | **KEEP** | Evaluates structural phase preservation under lossy recompression (Li et al. CVPR 2026). |
| **Canonical $8 \times 8$ Boundary Step Ratio** | Grid Artifacts | **KEEP** | Fan & de Queiroz / Wang metric measuring boundary vs interior gradient discontinuity. |
| **Raw JPEG Quantization Table Parsing** | Metadata | **REJECT** | Direct violation of DEC-003 / DEC-004; introduces severe metadata bias (Grommelt 2024). |
| **63-bin / 256-bin Dense DCT Histograms** | High-Dim DCT | **REJECT** | Curse of dimensionality ($>16,000$ bins); high risk of LightGBM tabular overfitting. |
| **Double JPEG Grid Shift Estimation (64 shifts)**| Spatial Grid | **DEFER** | High compute cost ($\sim 100$ ms); fragile under continuous area downsampling. |
| **JPEG AI Neural Codec Decoders** | Deep Neural | **REJECT** | Violates lightweight scalar requirement; incompatible with CPU-based tabular pipeline. |

---

## 6. Revised Candidate Architecture: Branch E (26 Candidate Features)

```text
src/forensics/branch_e/
├── E1: DCT-Domain Fingerprints (10 candidate features)
├── E2: Controlled Multi-Quality Compression Response (8 candidate features)
├── E3: Compression-Stable Phase Response (4 candidate features)
└── E4: Canonical 8x8 Grid Boundary Artifacts (4 candidate features)
```

### Complete Candidate Feature Registry

| # | Feature Name | Sub-Family | Input | Range | Scientific Rationale | Redundancy Risk |
|:---:|:---|:---|:---:|:---:|:---|:---:|
| 1 | `dct_ac_mean_abs` | E1_DCT | Grayscale | $[0, \infty)$ | Mean AC magnitude across $8 \times 8$ blocks | Low |
| 2 | `dct_ac_energy` | E1_DCT | Grayscale | $[0, \infty)$ | Total AC energy per $8 \times 8$ block | Low |
| 3 | `dct_ac_kurtosis` | E1_DCT | Grayscale | $[-3, \infty)$ | Excess kurtosis of AC coefficient distribution | Low |
| 4 | `dct_sparsity_ratio` | E1_DCT | Grayscale | $[0, 1]$ | Fraction of near-zero AC coefficients ($<10^{-3}$) | Low |
| 5 | `dct_low_freq_ratio` | E1_DCT | Grayscale | $[0, 1]$ | Energy ratio in low-frequency AC bases ($1 \le u+v \le 3$) | Low |
| 6 | `dct_mid_freq_ratio` | E1_DCT | Grayscale | $[0, 1]$ | Energy ratio in mid-frequency AC bases ($4 \le u+v \le 7$) | Low |
| 7 | `dct_high_freq_ratio`| E1_DCT | Grayscale | $[0, 1]$ | Energy ratio in high-frequency AC bases ($u+v \ge 8$) | Low |
| 8 | `dct_anisotropy` | E1_DCT | Grayscale | $[0, 1]$ | Asymmetry between horizontal ($u > v$) and vertical ($v > u$) AC energy | Low |
| 9 | `dct_benford_ssd` | E1_DCT | Grayscale | $[0, \infty)$ | Sum of squared error against Benford First Digit Law | Low |
| 10 | `dct_block_var_mean` | E1_DCT | Grayscale | $[0, \infty)$ | Spatial variance of block-level AC energy across blocks | Low |
| 11 | `ela_q95_mean` | E2_RESP | RGB | $[0, 1]$ | Mean recompression error at near-lossless $Q=95$ | Low |
| 12 | `ela_q90_mean` | E2_RESP | RGB | $[0, 1]$ | Mean recompression error at high quality $Q=90$ | Low |
| 13 | `ela_q75_mean` | E2_RESP | RGB | $[0, 1]$ | Mean recompression error at baseline quality $Q=75$ | Low |
| 14 | `ela_q60_mean` | E2_RESP | RGB | $[0, 1]$ | Mean recompression error at aggressive quality $Q=60$ | Low |
| 15 | `ela_q90_energy` | E2_RESP | RGB | $[0, 1]$ | Mean squared error energy at $Q=90$ | Low |
| 16 | `ela_slope_q90_q75` | E2_RESP | RGB | $[0, 1]$ | Compressibility slope: $(\Delta_{75} - \Delta_{90}) / 15$ | Low |
| 17 | `ela_ratio_q90_q75` | E2_RESP | RGB | $[0, \infty)$ | Cross-quality response ratio: $\Delta_{90} / (\Delta_{75} + 10^{-6})$ | Low |
| 18 | `ela_q90_gini` | E2_RESP | RGB | $[0, 1]$ | Spatial concentration (Gini coefficient) of $Q=90$ error map | Low |
| 19 | `phase_corr_q90` | E3_PHASE | Grayscale | $[-1, 1]$ | Cosine similarity between clean & $Q=90$ Fourier phase spectra | Low |
| 20 | `phase_corr_q75` | E3_PHASE | Grayscale | $[-1, 1]$ | Cosine similarity between clean & $Q=75$ Fourier phase spectra | Low |
| 21 | `phase_diff_energy_q90` | E3_PHASE | Grayscale | $[0, \pi^2]$ | Mean squared angular phase difference at $Q=90$ | Low |
| 22 | `phase_hf_stability_q90`| E3_PHASE | Grayscale | $[0, 1]$ | High-frequency radius phase preservation ratio at $Q=90$ | Low |
| 23 | `grid_h_ratio` | E4_GRID | Grayscale | $[0, \infty)$ | Horizontal $8 \times 8$ boundary step vs interior step | Low |
| 24 | `grid_v_ratio` | E4_GRID | Grayscale | $[0, \infty)$ | Vertical $8 \times 8$ boundary step vs interior step | Low |
| 25 | `grid_strength` | E4_GRID | Grayscale | $[0, \infty)$ | Combined isotropic blocking factor | Low |
| 26 | `grid_anisotropy` | E4_GRID | Grayscale | $[0, 1]$ | Directional discrepancy in horizontal vs vertical boundary step | Low |

---

## 7. Redundancy & Complementarity Analysis Against Branches A–D

```
┌─────────────────────┬──────────────────┬────────────────────────────────────────────────────────────┐
│ Existing Branch     │ Risk of Overlap  │ Scientific & Mathematical Distinction in Branch E          │
├─────────────────────┼──────────────────┼────────────────────────────────────────────────────────────┤
│ Branch A1 (FFT)     │ Low/Mid/High     │ Branch A1 integrates continuous radial Fourier bands over  │
│                     │ energy ratios    │ the entire 256x256 image. Branch E1 evaluates local 8x8    │
│                     │                  │ discrete block DCT basis projections.                      │
├─────────────────────┼──────────────────┼────────────────────────────────────────────────────────────┤
│ Branch A2           │ Period 8 peak    │ Synthbuster analyzes continuous cross-difference FFT peaks │
│ (Synthbuster)       │ correlation      │ over the whole canvas. Branch E4 measures localized grid   │
│                     │                  │ boundary vs interior step discontinuities.                 │
├─────────────────────┼──────────────────┼────────────────────────────────────────────────────────────┤
│ Branch B            │ Multiscale       │ Haar DWT operates on dyadic multiscale spatial subbands    │
│ (Haar Wavelet)      │ subband energy   │ (128, 64, 32). Branch E1 evaluates 64 localized 8x8 DCT    │
│                     │                  │ basis frequencies and Benford conformity.                  │
├─────────────────────┼──────────────────┼────────────────────────────────────────────────────────────┤
│ Branch C            │ Local texture    │ LBP/GLCM measure arbitrary micro-texture transitions.      │
│ (LBP / GLCM)        │ co-occurrences   │ Branch E2/E3 measure differential response to controlled   │
│                     │                  │ in-memory quantization.                                    │
├─────────────────────┼──────────────────┼────────────────────────────────────────────────────────────┤
│ Branch D            │ Residual noise   │ Branch D extracts spatial continuous noise filters         │
│ (HP/Lap/MFR)        │ statistics       │ (Gaussian, Laplacian, Median). Branch E2/E3 extracts       │
│                     │                  │ non-linear quantization roundtrip error curves and phase.  │
└─────────────────────┴──────────────────┴────────────────────────────────────────────────────────────┘
```

---

## 8. Computational Cost & Implementation Plan

### Computational Estimate (per $256 \times 256$ sample)
- **E1 (DCT Fingerprints):** PyTorch `F.unfold(8, stride=8)` + batch matrix DCT: $\sim 1.8$ ms
- **E2 (Multi-Q Response):** In-memory `cv2.imencode`/`imdecode` ($Q \in \{95, 90, 75, 60\}$): $\sim 2.5$ ms
- **E3 (Phase Stability):** 2D FFT on recompressed buffers: $\sim 1.0$ ms
- **E4 (Grid Discontinuity):** Strided slice indexing in PyTorch: $\sim 0.5$ ms
- **Total Branch E Runtime:** **$\sim 5.8$ ms / sample (CPU single core)**

### Planned Module Layout
```text
src/forensics/branch_e/
├── __init__.py          # Public API exports
├── dct.py               # E1: 8x8 block DCT, zonal ratios, Benford SSD, sparsity
├── recompression.py     # E2: Controlled multi-Q JPEG recompression error curves & stats
├── phase_stability.py   # E3: Phase spectrum stability under controlled JPEG recompression
├── grid.py              # E4: Canonical 8x8 grid boundary vs interior step ratios
└── features.py          # Branch E composition layer exposing E1..E4 and unified 'E'

src/forensics/tests/
└── test_branch_e_forensics.py  # Complete 15+ test validation suite
```
