# CHANGELOG.md

# Project Change Log

This file records important project changes, discoveries, failures, fixes, and methodological updates.

It is historical documentation.

Do not rewrite history merely to make the project appear cleaner.

---

## 2026-09-19 — Project Documentation Initialized

### Added

- `AGENTS.md`
- `PROJECT.md`
- `DECISIONS.md`
- `CHANGELOG.md`

### Purpose

Established the initial operating rules, current project state, finalized decisions, and historical change log.

---

## Dataset Preparation

### Completed

- Defactify selected as the primary dataset.
- Dataset downloaded.
- Raw dataset stored under `data/defactify/`.
- Initial dataset integrity audit completed.

### Discovered

- Dataset contains substantial variation in native image resolution.
- Caption information is extensively reused across the dataset.
- Real and AI-generated classes are imbalanced.
- Generator/source metadata exists and must remain outside detector input.

### Methodological consequence

The dataset requires additional confound/leakage analysis before the main forensic experiments.

---

## Important Rule

Future entries should contain enough information to answer:

1. What changed?
2. Why did it change?
3. What was discovered?
4. What broke?
5. How was it fixed?
6. What was verified?
7. Did the change affect the methodology?

Do not record unverified claims as results.

---

## 2026-09-19 — Block 1 Phase 1: Defactify Confound & Leakage Audit (COMPLETE)

### Added

- `src/analysis/defactify_confound_audit.py` — 877-line full-dataset streaming audit.
- `src/analysis/defactify_confound_audit_report.json` — machine-readable full results.
- `src/analysis/defactify_confound_audit_summary.md` — human-readable summary.

### Key Findings (RESULT — verified from 96,000 images)

1. **Aspect ratio confound:** 100% of AI images are square; 97.46% of real images are non-square. A naive resize would create a geometric class-correlated confound.
2. **Deterministic AI resolutions:** SD2.1=768×768, SDXL=1024×1024, SD3=1024×1024, DALL-E3=270×270, Midjourney=436×436.
3. **Real image diversity:** 842 distinct resolutions; 73.5% landscape, 23.96% portrait, 2.54% square.
4. **Quantization tables:** Identical across all 96,000 images (IJG Q75, 4:2:0 chroma). Format/compression is not a usable detector signal.
5. **Exact duplicates:** 19 groups (71 images), all Midjourney v6. 7 train↔val, 8 train↔test, 3 val↔test cross-split duplicates. Zero real↔AI duplicates.
6. **Caption reuse:** All 5 AI generators share all 9,879 real-image captions. Captions cannot be used as real/AI discriminators.
7. **Accounting:** Perfect 16,000/16,000 per generator, zero nulls, stratified by generator.

### Methodological consequence

The aspect-ratio confound finding justified and confirmed the canonical preprocessing choice (largest centered square crop) as the scientifically defensible approach.

---

## 2026-09-19 — Block 1 Phase 2: Canonical Preprocessing Pipeline (COMPLETE)

### Added

- `src/data/__init__.py` — package init exposing preprocessing + loader.
- `src/data/preprocessing.py` — canonical crop+resize module.
- `src/data/loader.py` — `DefactifyDataset` PyTorch Dataset backed by Parquet.
- `src/data/tests/__init__.py` — test package init.
- `src/data/tests/test_preprocessing.py` — 7 verification tests.
- `src/data/tests/run_tests.py` — standalone test runner (no pytest required).

### Decision Formalized

DEC-008 added to DECISIONS.md: canonical preprocessing is centered square crop → bilinear resize 256×256. "exact spatial preprocessing method" removed from pending list.

### Verification (RESULT — all tests executed)

```
Block 1 — Preprocessing Test Suite
======================================================================
  PASS  Test 1 — Landscape 640×480 → crop 480×480 → resize 256×256
  PASS  Test 2 — Portrait 480×640 → crop 480×480 → resize 256×256
  PASS  Test 3 — Square 1024×1024 → identity crop → 256×256
  PASS  Test 4 — Square 270×270 → identity crop → 256×256
  PASS  Test 5 — RGB output, 256×256, deterministic
  PASS  Test 6 — Metadata isolation (path/caption/label_b blocked)
  PASS  Test 7 — Raw data integrity (git diff data/defactify/ == empty)
======================================================================
Result: 7 passed, 0 failed out of 7 tests.
```

Exit code: 0. No raw data modified.

---

## 2026-09-19 — Block 1 Correction: Loader Redesign + Extended Verification

### Problem (PROBLEM)

The initial `loader.py` used `pq.read_table(fpath)` followed by
`table.to_batches()`. This loaded each entire Parquet file into RAM first
and stored all JPEG byte arrays (for all rows in the split) in `self._rows`.
For the 7 GB Defactify dataset this is not acceptable.

### Fix (FIX)

Rewrote `loader.py` with a **lightweight index design**:

- `__init__` opens `pq.ParquetFile` handles (footer metadata only, kilobytes).
- `__init__` reads label columns (`Label_A`, optionally `Label_B`/`Caption`)
  row-group by row-group to build a flat `_IndexEntry` list.
- Each `_IndexEntry` stores only `(file_idx, rg_idx, row_in_rg, label_a)` — no bytes.
- `__getitem__` fetches the Image column for the required row group on demand,
  extracts one row's bytes, preprocesses, discards the byte object.

**Memory model verified (RESULT):**

```
_IndexEntry size:       80 bytes/entry
Total index (42k train): ~3.3 MB
JPEG bytes in RAM at init: 0
Decoded images in RAM: 0
Tensor cache: None
```

### Additional Problem (PROBLEM)

Test 7 used `git diff HEAD -- data/defactify/` as the immutability check.
`git check-ignore` confirmed that `data/defactify/` is excluded by
`.gitignore:7:/data/`, so an empty git diff proves nothing about these
untracked files.

### Fix (FIX)

Replaced Test 7 with a file-size + SHA-256 of first 64 KB check across all
17 Parquet files. Captures pre-test state and re-verifies within the same
process run. Does not hash all 7 GB.

### Documentation Fix

Corrected wording in `preprocessing.py` that claimed BILINEAR introduces
"no aliasing artifacts from random ops" — too strong for a forensic project.
Replaced with: "BILINEAR interpolation is deterministic and is applied
identically to every sample. The interpolation operation modifies image
frequency content and is therefore part of the standardized preprocessing
contract." Interpolation method unchanged.

### Added

- `src/data/tests/test_loader.py` — 7 loader tests (A–G) against real Parquet data.
- `run_tests.py` updated to run both preprocessing (1–7) and loader (A–G) suites.

### Verification (RESULT — all 14 tests executed)

```
Block 1 — Preprocessing Tests (1–7)
======================================================================
  PASS  Test 1 — Landscape 640×480 → crop 480×480 → resize 256×256
  PASS  Test 2 — Portrait 480×640 → crop 480×480 → resize 256×256
  PASS  Test 3 — Square 1024×1024 → identity crop → 256×256
  PASS  Test 4 — Square 270×270 → identity crop → 256×256
  PASS  Test 5 — RGB output, 256×256, deterministic
  PASS  Test 6 — Metadata isolation (path/caption/label_b blocked)
  PASS  Test 7 — Raw data immutability (file-size + 64KB SHA-256)

Block 1 — Loader Tests (A–G)
======================================================================
  PASS  Test A — Train split length == 42,000
  PASS  Test B — Validation split length == 9,000
  PASS  Test C — Test split length == 45,000
  PASS  Test D — Real Parquet sample: shape [3,256,256] float32 [0,1]
  PASS  Test E — Both classes (Label_A=0 and =1) loadable
  PASS  Test F — Detector mode returns exactly (image, label_a)
  PASS  Test G — Audit mode returns (image, label_a, label_b, caption)

Exit code: 0. No raw data modified (verified by file-size + SHA-256 check).

---

## 2026-09-19 — Block 1 Materialized Training Pipeline & Module Registry (COMPLETE)

### Architectural Refinement (DEC-009)

Separated the Block 1 responsibilities into an asymmetric architecture:
1. **Training:** Raw $\to$ Canonical Preprocessing $\to$ Materialized versioned Parquet dataset (`data/processed/train_v1/`) + `manifest.json`.
2. **Validation / Test:** Processed on demand via `DefactifyDataset` from raw Parquet files without permanent materialization.

### Added

- `src/data/materializer.py` — Training materialization pipeline with deterministic batch flushing (1,000 samples $\approx 196.6$ MB buffer), versioning, and JSON manifest generation.
- `src/data/processed_loader.py` — High-speed PyTorch dataset reader for materialized training Parquet data.
- `src/data/tests/test_materializer.py` — 9 unit tests covering materialization count, tensor shape/dtype/range, label alignment, absence of forbidden metadata, determinism, train/eval parity, corrupt input handling, idempotency, and manifest counts.
- `docs/MODULE_MAP.md` — Complete module inventory mapping responsibilities, inputs, outputs, interfaces, dependencies, and test coverage.

### Decision Formalized

DEC-009 added to `DECISIONS.md`: Materialized training dataset format (`image_rgb: binary` + `label_a: int32`) and on-demand evaluation path.

### Verification (RESULT — 23/23 tests PASS)

```
Suite 1 — Preprocessing Tests (1–7)
  PASS  Test 1 — Landscape 640×480 → crop 480×480 → resize 256×256
  PASS  Test 2 — Portrait 480×640 → crop 480×480 → resize 256×256
  PASS  Test 3 — Square 1024×1024 → identity crop → 256×256
  PASS  Test 4 — Square 270×270 → identity crop → 256×256
  PASS  Test 5 — RGB output, 256×256, deterministic
  PASS  Test 6 — Metadata isolation (path/caption/label_b blocked)
  PASS  Test 7 — Raw data immutability (file-size + 64KB SHA-256)

Suite 2 — Loader Tests (A–G)
  PASS  Test A — Train split length == 42,000
  PASS  Test B — Validation split length == 9,000
  PASS  Test C — Test split length == 45,000
  PASS  Test D — Real Parquet sample: shape [3,256,256] float32 [0,1]
  PASS  Test E — Both classes (Label_A=0 and =1) loadable
  PASS  Test F — Detector mode returns exactly (image, label_a)
  PASS  Test G — Audit mode returns (image, label_a, label_b, caption)

Suite 3 — Materializer Tests (C–K)
  PASS  Test C — Materialization: sample count matches SMOKE_N
  PASS  Test D — Materialization: shape [3,256,256] float32 [0,1]
  PASS  Test E — Materialization: labels match raw Parquet
  PASS  Test F — Materialization: no forbidden metadata columns
  PASS  Test G — Materialization: deterministic (byte-identical runs)
  PASS  Test H — Both paths: same canonical image for same source
  PASS  Test I — Corrupt image bytes: raised exception (caught)
  PASS  Test J — Overwrite idempotent (same output on repeat)
  PASS  Test K — Manifest counts match actual Parquet rows

TOTAL: 23 passed, 0 failed out of 23 tests.
```

- Real 50-sample materialization smoke test executed and verified (output verified, pixel delta with direct preprocessing = 0.0, cleaned up).
- Raw data immutability confirmed.
- Exit code: 0.

---

## 2026-09-19 — Block 2 Phase 1: Branch A Forensic Features (Frequency / Periodicity) (COMPLETE)

### Added

- `src/forensics/__init__.py` — Package root exposing `extract_branch_a_features`, `standard_fft_features`, `synthbuster_periodicity_features`, and core math helpers.
- `src/forensics/frequency.py` — Standard Fourier analysis module extracting normalized radial frequency energy bands (`fft_low_freq_ratio`, `fft_mid_freq_ratio`, `fft_high_freq_ratio`) and `fft_spectral_centroid` from DC-subtracted grayscale images, plus separated diagnostic maps generator.
- `src/forensics/synthbuster.py` — Synthbuster-inspired periodicity module extracting cross-difference residuals ($256 \times 256 \to 255 \times 255$), per-channel ($R, G, B$) 2D FFTs, directional frequency peak means ($x, y, d$) at periods $(2, 4, 8)$, and residual high-frequency ratios.
- `src/forensics/tests/__init__.py` — Test package init.
- `src/forensics/tests/test_frequency.py` — 10 comprehensive unit tests covering contracts, shapes, determinism, RGB independence, safety on zero/constant inputs, pattern discrimination, and real-image smoke extraction.
- `src/forensics/tests/run_tests.py` — Standalone test runner for Branch A.

### Reused / Refactored Historical Notebook Code

- Extracted and preserved mathematical logic from `analysis_800_v1(5).ipynb`:
  1. `frequency_radius()` — Vectorized 2D normalized frequency radius grid $[H, W]$.
  2. `standard_fft_features()` (refactored from `fft_features()`) — DC removal via mean subtraction, orthonormal 2D FFT (`norm="ortho"`), quadrant centering (`fftshift`), and radial energy band partitions.
  3. `cross_difference()` — Vectorized 2D residual operator producing $[B, C, H-1, W-1]$.
  4. `synthbuster_periodicity_features()` (refactored from `synthbuster_peak_features()`) — Per-channel 2D FFT, directional peak sampling at periods $(2, 4, 8)$ in $x, y, d$ offsets, and residual high-frequency ratio.
- Discarded exploratory $800+800$ dataset loading, plots, statistics, and AUC claims.
- Isolated diagnostic maps (`fft_logmag_map`, `fft_phase_map`, `fft_radial_power`) into a dedicated helper to ensure exactly 34 scalar features enter the detector contract.

### Contract Formalized (RESULT — Exactly 34 Scalar Features)

- **A1 Standard FFT (4 scalars):** `fft_low_freq_ratio`, `fft_mid_freq_ratio`, `fft_high_freq_ratio`, `fft_spectral_centroid`.
- **A2 Synthbuster-Inspired Periodicity (30 scalars):** `synth_{r,g,b}_p{2,4,8}_{x,y,d}_mean` (27 scalars) and `synth_{r,g,b}_fft_highfreq_ratio` (3 scalars).

### Verification (RESULT — All Tests Executed and Passing)

```
Block 2 — Branch A (Frequency / Periodicity) Test Suite
======================================================================
  PASS  test_01_a1_standard_fft_contract
  PASS  test_02_a2_synthbuster_contract
  PASS  test_03_unified_branch_a_contract
  PASS  test_04_cross_difference_geometry_and_arithmetic
  PASS  test_05_determinism
  PASS  test_06_rgb_channel_independence_in_a2
  PASS  test_07_numerical_safety_zero_and_constant_inputs
  PASS  test_08_distinct_images_produce_distinct_features
  PASS  test_09_diagnostics_isolation
  PASS  test_10_smoke_real_canonical_tensor
======================================================================
Result: 10 passed, 0 failed, 0 errors out of 10 tests (0.99s).
```

- Block 1 test suite re-verified: 23/23 tests PASS (`src/data/tests/run_tests.py`).
- Single-sample smoke test on Defactify train image verified: 34 features extracted, 100% finite, deterministic.
- Exit code: 0. No raw data modified.

---

## 2026-09-19 — Block 2 Phase 1: Structural Refactor into Final Forensic Architecture (COMPLETE)

### Purpose

Reorganized the working Block 2 — Branch A implementation into the final modular forensic architecture (`src/forensics/branch_a_frequency/`, `src/forensics/pipeline.py`, `src/forensics/run_forensic_pipeline.py`) to prepare for future branch additions (B, C, D, E) without cross-branch dependencies.

### Changes Made

- **Branch A Modularization:**
  - Created `src/forensics/branch_a_frequency/fft.py` (A1 Standard Fourier analysis + separated diagnostics).
  - Created `src/forensics/branch_a_frequency/synthbuster.py` (A2 Synthbuster-inspired periodicity analysis).
  - Created `src/forensics/branch_a_frequency/features.py` (Branch A composition layer).
  - Created `src/forensics/branch_a_frequency/__init__.py` (Branch A package API).
  - Removed flat files `src/forensics/frequency.py` and `src/forensics/synthbuster.py`.
- **Pipeline Orchestration:**
  - Created `src/forensics/pipeline.py` exposing `ForensicPipeline(branches=["A"])` with dynamic branch dispatch and deterministic feature combining.
  - Created `src/forensics/run_forensic_pipeline.py` providing CLI runner capabilities for dataset splits and synthetic smoke testing.
  - Updated `src/forensics/__init__.py` to export `ForensicPipeline` and unified feature extractors.
- **Tests Reorganization:**
  - Replaced `test_frequency.py` with `src/forensics/tests/test_branch_a_frequency.py` (11 tests covering all Branch A properties + `ForensicPipeline` branch validation and execution equivalence).
  - Updated `src/forensics/tests/run_tests.py`.

### Invariance & Numerical Equivalence Verified (RESULT)

- **Zero Mathematical Changes:** Grayscale weights, DC mean subtraction, orthonormal FFT, cross-difference geometry, peak coordinate indexing, and band definitions remain 100% identical.
- **Bit-Exact Numerical Parity:** Tested before vs after refactoring on deterministic seed tensor — Maximum delta across all 34 features = `0.0000000000e+00`.
- **Test Suite:** 11/11 PASS in `src/forensics/tests/run_tests.py` (0.94s).
- **Block 1 Regression Suite:** 23/23 PASS in `src/data/tests/run_tests.py` (exit code 0).
- **CLI Runner Smoke Test:** 3 samples from Defactify train split extracted at ~2.99 ms/image with 100% finite features.
- **Raw Data Immutability:** `data/defactify/` remains untouched.

---

## 2026-09-19 — Block 2 Phase 2: Branch B Forensic Features (Wavelet / 3-Level Haar DWT) (COMPLETE)

### Purpose

Implemented and verified Block 2 — Branch B: Wavelet feature extraction using a pure PyTorch 3-level 2D Haar Discrete Wavelet Transform (DWT) on canonical grayscale images.

### Added

- `src/forensics/branch_b_wavelet/haar.py` — Pure PyTorch 2D separable Haar wavelet decomposition module implementing `haar_2d_level` and `haar_dwt_3level` ($256 \to 128 \to 64 \to 32$).
- `src/forensics/branch_b_wavelet/features.py` — 30-feature scalar extraction module implementing `safe_entropy`, subband energy, absmean, entropy, and global `detail_energy_ratio`.
- `src/forensics/branch_b_wavelet/__init__.py` — Clean Branch B package API.
- `src/forensics/tests/test_branch_b_wavelet.py` — 11 comprehensive unit tests covering contracts, multiscale dimensions, Parseval energy conservation, ratio arithmetic, entropy properties, reference equivalence, determinism, numerical safety, pattern orientation sensitivity, real-image smoke extraction, and multi-branch pipeline integration.

### Modified

- `src/forensics/pipeline.py` — Registered Branch B (`"B"`) in `ForensicPipeline`, enabling `branches=["A"]` (34 features), `branches=["B"]` (30 features), and `branches=["A", "B"]` (64 features).
- `src/forensics/__init__.py` — Exported Branch B interfaces and helpers.
- `src/forensics/tests/run_tests.py` — Master test runner updated to execute all 22 tests across Branch A and Branch B.
- `src/forensics/tests/test_branch_a_frequency.py` — Updated unimplemented branch exception check from `"B"` to `"C"`.
- `docs/MODULE_MAP.md` — Updated module inventory and architectural specifications for Branch B.

### Haar Coefficient Convention & Mathematical Specifications

- **Separable 2D Orthonormal Haar Basis:**
  - Spatial 2x2 grid: $x_{00} = x[\dots, 0::2, 0::2]$, $x_{01} = x[\dots, 0::2, 1::2]$, $x_{10} = x[\dots, 1::2, 0::2]$, $x_{11} = x[\dots, 1::2, 1::2]$.
  - Approximation: $LL = (x_{00} + x_{01} + x_{10} + x_{11}) \times 0.5$
  - Horizontal detail: $LH = (x_{00} - x_{01} + x_{10} - x_{11}) \times 0.5$
  - Vertical detail: $HL = (x_{00} + x_{01} - x_{10} - x_{11}) \times 0.5$
  - Diagonal detail: $HH = (x_{00} - x_{01} - x_{10} + x_{11}) \times 0.5$
- **Sign Invariance:** Summary statistics ($\text{mean}(S^2)$, $\text{mean}(|S|)$, Shannon histogram entropy) are strictly sign-invariant, ensuring feature robustness.
- **Pure PyTorch Runtime:** Zero runtime dependency on external wavelet libraries (`pywt` used only as an optional verification reference in unit tests).
- **Detail Energy Ratio Definition:**
  - Numerator: $\text{total\_detail\_energy} = \sum_{l=1}^3 \left( \text{energy}(LH_l) + \text{energy}(HL_l) + \text{energy}(HH_l) \right)$
  - Denominator: $\text{total\_energy} = \text{energy}(LL_3) + \text{total\_detail\_energy}$
  - Ratio: $\text{detail\_energy\_ratio} = \frac{\text{total\_detail\_energy}}{\text{total\_energy} + 10^{-12}}$
- **Feature Set (Exactly 30 Scalars):**
  - $LL_3$: `LL3_energy`, `LL3_entropy` (2 scalars)
  - Detail subbands $LH_3, HL_3, HH_3, LH_2, HL_2, HH_2, LH_1, HL_1, HH_1$: `{band}_energy`, `{band}_absmean`, `{band}_entropy` (27 scalars)
  - Global ratio: `detail_energy_ratio` (1 scalar)

### Verification (RESULT — All Suites Executed and Passing)

```
Block 2 — Forensic Feature Extraction & Pipeline Test Suite
======================================================================
  PASS  test_01_a1_standard_fft_contract
  PASS  test_02_a2_synthbuster_contract
  PASS  test_03_unified_branch_a_contract
  PASS  test_04_cross_difference_geometry_and_arithmetic
  PASS  test_05_determinism
  PASS  test_06_rgb_channel_independence_in_a2
  PASS  test_07_numerical_safety_zero_and_constant_inputs
  PASS  test_08_distinct_images_produce_distinct_features
  PASS  test_09_diagnostics_isolation
  PASS  test_10_smoke_real_canonical_tensor
  PASS  test_11_forensic_pipeline_orchestration
  PASS  test_01_branch_b_contract
  PASS  test_02_haar_subband_shapes
  PASS  test_03_haar_parseval_energy_conservation
  PASS  test_04_detail_energy_ratio_math
  PASS  test_05_safe_entropy_properties
  PASS  test_06_reference_equivalence_and_pywt_comparison
  PASS  test_07_determinism
  PASS  test_08_numerical_safety
  PASS  test_09_pattern_discrimination
  PASS  test_10_smoke_real_canonical_tensor
  PASS  test_11_forensic_pipeline_multibranch
======================================================================
Result: 22 passed, 0 failed, 0 errors out of 22 tests (1.14s).
```

- **Branch A Invariance:** Bit-exact $\Delta = 0.0000000000e+00$ verified across all 34 Branch A features.
- **Block 1 Regression Suite:** 23/23 tests PASS (`src/data/tests/run_tests.py`).
- **Raw Data Immutability:** `data/defactify/` remains untouched (17 files, 7,509,031,418 bytes).
- **Exit Code:** 0.

---

## 2026-09-19 — Block 2 Phase 3: Branch C Forensic Features (Local Texture: LBP, GLCM, Edge-Guided LBP) (COMPLETE)

### Purpose

Implemented and verified Block 2 — Branch C: Local Texture feature extraction, introducing three independently selectable, alternative texture pipelines:
1. `C_LBP`: Standard rotation-invariant uniform Local Binary Pattern ($P=8, R=1$, 16 scalar features).
2. `C_GLCM`: Multi-distance Gray-Level Co-occurrence Matrix ($N_g=16, d \in \{1, 2, 4\}$, 24 scalar features).
3. `C_LBP_EDGE`: Canny edge-guided LBP candidate (16 scalar features).

In accordance with project decisions, alternative texture descriptors are isolated and never automatically combined into a single feature vector.

### Added

- `src/forensics/branch_c_texture/lbp.py` — Vectorized rotation-invariant uniform LBP computation, deterministic Canny edge detection, and masked LBP histogram and summary statistics extraction.
- `src/forensics/branch_c_texture/glcm.py` — Gray-Level Co-occurrence Matrix module with 16-level linear quantization, symmetrical multi-distance computation ($d \in \{1, 2, 4\}$ across 4 standard angles), and Haralick texture property extraction.
- `src/forensics/branch_c_texture/features.py` — Branch C composition layer exposing `extract_branch_c_lbp_features` (16 scalars), `extract_branch_c_glcm_features` (24 scalars), `extract_branch_c_lbp_edge_features` (16 scalars), and `rgb_to_gray`.
- `src/forensics/branch_c_texture/__init__.py` — Clean Branch C package API.
- `src/forensics/tests/test_branch_c_texture.py` — 12 comprehensive unit tests covering contracts, mathematical properties, edge cases, determinism, numerical safety on pathological inputs, texture discrimination, real-sample smoke extraction, pipeline isolation, and multi-branch composition.

### Modified

- `src/forensics/pipeline.py` — Registered `C_LBP`, `C_GLCM`, and `C_LBP_EDGE` in `ForensicPipeline`. Added validation requiring explicit selection if generic `"C"` is passed.
- `src/forensics/__init__.py` — Exported Branch C extractors and helpers.
- `src/forensics/run_forensic_pipeline.py` — Updated CLI runner with Branch C alternatives and defaults.
- `src/forensics/tests/run_tests.py` — Updated master test runner to execute all 34 forensic tests.
- `src/forensics/tests/test_branch_a_frequency.py` — Updated branch exception test to verify `ValueError` on `"C"` and `NotImplementedError` on `"D"`.
- `docs/MODULE_MAP.md` — Updated module inventory and architectural specifications for Branch C.

### Mathematical Specifications & Feature Sets

1. **Grayscale Standard:** ITU-R BT.601 luminance $Y = 0.299 R + 0.587 G + 0.114 B$ on canonical $256 \times 256$ images.
2. **Standard Uniform LBP (`C_LBP` — Exactly 16 Scalars):**
   - $P = 8, R = 1$, replicate 1-pixel padding.
   - Circular bit transitions $U \le 2$ mapped to bin index = number of 1s in pattern ($0 \dots 8$); $U > 2$ mapped to bin 9.
   - Bins: 10 normalized histogram probabilities (`lbp_bin_0` to `lbp_bin_8`, `lbp_bin_nonuniform`).
   - Summary stats: `lbp_entropy`, `lbp_uniformity`, `lbp_dominant_bin`, `lbp_mean_code`, `lbp_std_code`, `lbp_nonuniform_ratio`.
3. **Gray-Level Co-occurrence Matrix (`C_GLCM` — Exactly 24 Scalars):**
   - $N_g = 16$ gray levels, distances $d \in \{1, 2, 4\}$, angles $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$, symmetric and normalized.
   - 6 Haralick statistics: *Contrast, Dissimilarity, Homogeneity, Energy, Correlation, Entropy*.
   - 18 directional means across the 3 distances + 6 directional standard deviations at scale $d=1$ (measuring textural anisotropy).
4. **Edge-Guided LBP Candidate (`C_LBP_EDGE` — Exactly 16 Scalars):**
   - Grayscale $\to$ Canny edge detection ($\sigma=1.0, T_{\text{low}}=50, T_{\text{high}}=100$) $\to$ binary edge mask $M \in \{0, 1\}$.
   - LBP histogram and summary statistics extracted strictly on edge-located pixels ($M=1$) with uniform fallback for sparse masks ($|E| < 10$).
   - 10 edge histogram bins + 5 edge summary statistics + 1 edge pixel density metric (`lbp_edge_pixel_density`).

### Verification (RESULT — All Suites Executed and Passing)

```
Block 2 — Forensic Feature Extraction & Pipeline Test Suite
======================================================================
  PASS  test_01_a1_standard_fft_contract
  PASS  test_02_a2_synthbuster_contract
  PASS  test_03_unified_branch_a_contract
  PASS  test_04_cross_difference_geometry_and_arithmetic
  PASS  test_05_determinism
  PASS  test_06_rgb_channel_independence_in_a2
  PASS  test_07_numerical_safety_zero_and_constant_inputs
  PASS  test_08_distinct_images_produce_distinct_features
  PASS  test_09_diagnostics_isolation
  PASS  test_10_smoke_real_canonical_tensor
  PASS  test_11_forensic_pipeline_orchestration
  PASS  test_01_branch_b_contract
  PASS  test_02_haar_subband_shapes
  PASS  test_03_haar_parseval_energy_conservation
  PASS  test_04_detail_energy_ratio_math
  PASS  test_05_safe_entropy_properties
  PASS  test_06_reference_equivalence_and_pywt_comparison
  PASS  test_07_determinism
  PASS  test_08_numerical_safety
  PASS  test_09_pattern_discrimination
  PASS  test_10_smoke_real_canonical_tensor
  PASS  test_11_forensic_pipeline_multibranch
  PASS  test_01_c_lbp_contract
  PASS  test_02_c_glcm_contract
  PASS  test_03_c_lbp_edge_contract
  PASS  test_04_lbp_mathematical_properties_and_mapping
  PASS  test_05_glcm_mathematical_properties_and_quantization
  PASS  test_06_edge_detection_and_sparse_mask_safety
  PASS  test_07_determinism
  PASS  test_08_numerical_safety_pathological_inputs
  PASS  test_09_texture_discrimination
  PASS  test_10_smoke_real_defactify_sample
  PASS  test_11_forensic_pipeline_isolation_and_alternatives
  PASS  test_12_multi_branch_combination
======================================================================
Result: 34 passed, 0 failed, 0 errors out of 34 tests (1.44s).
```

- **Branch A & B Invariance:** Bit-exact $\Delta = 0.0000000000e+00$ verified across all Branch A (34) and Branch B (30) features.
- **Block 1 Regression Suite:** 23/23 tests PASS (`src/data/tests/run_tests.py`).
- **Raw Data Immutability:** `data/defactify/` remains untouched (17 files, 7,509,031,418 bytes).
- **Exit Code:** 0.

---

## 2026-09-19 — Block 2 Phase 4: Branch D Forensic Features (Residual / Error / Noise) (COMPLETE)

### Purpose

Implemented and verified Block 2 — Branch D: Residual / Error / Noise feature extraction, introducing three independent, alternative spatial residual candidate pipelines:
1. `D_HIGHPASS`: Linear Gaussian high-pass residual ($5 \times 5, \sigma=1.0$, 5 candidate scalar features).
2. `D_LAPLACIAN`: Linear discrete 8-neighbor Laplacian 2nd-order spatial-curvature residual ($3 \times 3$, 5 candidate scalar features).
3. `D_MFR`: Nonlinear $3 \times 3$ median filter residual implemented in pure PyTorch (5 candidate scalar features).

In accordance with the approved revised implementation plan:
- `D_HIGHPASS` and `D_LAPLACIAN` are treated as alternative linear residual representations with acknowledged potential overlap.
- `D_MFR` is a nonlinear rank-order residual candidate.
- None of the candidates are assumed to be complementary or proven useful prior to Block 4 empirical evaluation.
- `D_ELA` is deferred to future Branch E (JPEG/compression-aware evidence).
- Standalone cross-difference is omitted because Branch A already uses the identical cross-difference operator as a pre-filter.
- No mandatory composite feature vector is created.

### Added

- `src/forensics/branch_d_residual/highpass.py` — Gaussian high-pass residual module extracting `hp_absmean`, `hp_std`, `hp_energy`, `hp_kurtosis`, `hp_entropy`.
- `src/forensics/branch_d_residual/laplacian.py` — Discrete Laplacian residual module extracting `lap_absmean`, `lap_std`, `lap_energy`, `lap_kurtosis`, `lap_entropy`.
- `src/forensics/branch_d_residual/median_filter.py` — Median filter residual module extracting `mfr_absmean`, `mfr_std`, `mfr_energy`, `mfr_kurtosis`, `mfr_entropy`.
- `src/forensics/branch_d_residual/features.py` — Composition layer exposing `extract_branch_d_highpass_features`, `extract_branch_d_laplacian_features`, `extract_branch_d_mfr_features`.
- `src/forensics/branch_d_residual/__init__.py` — Clean Branch D package API.
- `src/forensics/tests/test_branch_d_residual.py` — 12 comprehensive unit tests covering contracts, shapes, constant-image zero response, filter weights, Scipy reference equivalence, determinism, numerical safety on pathological inputs, pattern discrimination, real-image smoke extraction, and ForensicPipeline integration.

### Modified

- `src/forensics/pipeline.py` — Registered `D_HIGHPASS`, `D_LAPLACIAN`, `D_MFR` and convenience alias `"D"` in `ForensicPipeline`.
- `src/forensics/__init__.py` — Exported Branch D public interfaces and helpers.
- `src/forensics/run_forensic_pipeline.py` — Updated CLI runner `--branches` argument choices and help.
- `src/forensics/tests/run_tests.py` — Master test runner updated to execute all 49 forensic tests.
- `src/forensics/tests/test_branch_a_frequency.py` — Updated unimplemented branch exception assertion from `"D"` to `"E"`.
- `docs/MODULE_MAP.md` — Updated module inventory and architectural specifications for Branch D.

### Mathematical Specifications & Candidate Sets

1. **Grayscale Standard:** ITU-R BT.601 luminance $Y = 0.299 R + 0.587 G + 0.114 B$ on canonical $256 \times 256$ images.
2. **Gaussian High-Pass (`D_HIGHPASS` — 5 Candidate Scalars):**
   - $R_{\text{HP}} = Y - G_{\sigma=1.0} * Y$ ($5 \times 5$, replicate padding).
   - Statistics: `hp_absmean`, `hp_std`, `hp_energy`, `hp_kurtosis`, `hp_entropy`.
3. **Discrete Laplacian (`D_LAPLACIAN` — 5 Candidate Scalars):**
   - $R_{\text{Lap}} = K_{\text{Lap}} * Y$ ($3 \times 3$, 8-neighbor, replicate padding).
   - Statistics: `lap_absmean`, `lap_std`, `lap_energy`, `lap_kurtosis`, `lap_entropy`.
4. **Median Filter Residual (`D_MFR` — 5 Candidate Scalars):**
   - $R_{\text{MFR}} = Y - \text{MedianFilter}_{3 \times 3}(Y)$ (pure PyTorch `F.unfold`, replicate padding).
   - Numerically bit-exact with `scipy.ndimage.median_filter(mode='nearest')` ($\Delta = 0.0$).
   - Statistics: `mfr_absmean`, `mfr_std`, `mfr_energy`, `mfr_kurtosis`, `mfr_entropy`.

### Verification (RESULT — All Suites Executed and Passing)

```
Block 2 — Forensic Feature Extraction & Pipeline Test Suite
======================================================================
  PASS  test_01_a1_standard_fft_contract
  PASS  test_02_a2_synthbuster_contract
  PASS  test_03_unified_branch_a_contract
  PASS  test_04_cross_difference_geometry_and_arithmetic
  PASS  test_05_determinism
  PASS  test_06_rgb_channel_independence_in_a2
  PASS  test_07_numerical_safety_zero_and_constant_inputs
  PASS  test_08_distinct_images_produce_distinct_features
  PASS  test_09_diagnostics_isolation
  PASS  test_10_smoke_real_canonical_tensor
  PASS  test_11_forensic_pipeline_orchestration
  PASS  test_01_branch_b_contract
  PASS  test_02_haar_subband_shapes
  PASS  test_03_haar_parseval_energy_conservation
  PASS  test_04_detail_energy_ratio_math
  PASS  test_05_safe_entropy_properties
  PASS  test_06_reference_equivalence_and_pywt_comparison
  PASS  test_07_determinism
  PASS  test_08_numerical_safety
  PASS  test_09_pattern_discrimination
  PASS  test_10_smoke_real_canonical_tensor
  PASS  test_11_forensic_pipeline_multibranch
  PASS  test_01_c_lbp_contract
  PASS  test_02_c_glcm_contract
  PASS  test_03_c_lbp_edge_contract
  PASS  test_04_lbp_mathematical_properties_and_mapping
  PASS  test_05_glcm_mathematical_properties_and_quantization
  PASS  test_06_edge_detection_and_sparse_mask_safety
  PASS  test_07_determinism
  PASS  test_08_numerical_safety_pathological_inputs
  PASS  test_09_texture_discrimination
  PASS  test_10_smoke_real_defactify_sample
  PASS  test_11_forensic_pipeline_isolation_and_alternatives
  PASS  test_12_multi_branch_combination
  PASS  test_13_mutual_exclusion_of_branch_c_alternatives
  PASS  test_14_empty_and_very_sparse_edge_masks
  PASS  test_15_c_lbp_edge_determinism_and_components
  PASS  test_01_d_highpass_contract
  PASS  test_02_d_laplacian_contract
  PASS  test_03_d_mfr_contract
  PASS  test_04_residual_maps_shapes_and_types
  PASS  test_05_constant_and_zero_image_behavior
  PASS  test_06_exact_kernel_behavior
  PASS  test_07_mfr_scipy_reference_equivalence
  PASS  test_08_determinism
  PASS  test_09_numerical_safety_pathological_inputs
  PASS  test_10_pattern_discrimination
  PASS  test_11_smoke_real_defactify_sample
  PASS  test_12_forensic_pipeline_integration_and_isolation
======================================================================
Result: 49 passed, 0 failed, 0 errors out of 49 tests (1.63s).
```

- **Branch A, B, C Invariance:** Bit-exact invariance verified across Branch A (34), Branch B (30), and Branch C (16/24/16) features.
- **Block 1 Regression Suite:** 23/23 tests PASS (`src/data/tests/run_tests.py`).
- **Raw Data Immutability:** `data/defactify/` remains untouched (17 files, 7,509,031,418 bytes).
- **Exit Code:** 0.

---

## 2026-09-19 — Block 1 Preprocessing Update: Area-Based Downsampling (cv2.INTER_AREA)

### Purpose

Updated the canonical spatial resize step from bilinear interpolation to area-based downsampling (`cv2.INTER_AREA`) in `src/data/preprocessing.py` (DEC-008).

### Decision

"Canonical preprocessing uses largest centered square crop followed by 256×256 area-based resizing (`cv2.INTER_AREA`)."

### Modified

- `src/data/preprocessing.py` — Replaced `PIL` bilinear resize in `resize_to_canonical` with `cv2.resize(..., interpolation=cv2.INTER_AREA)`.
- `src/data/materializer.py` — Updated manifest metadata preprocessing configuration `interpolation` field to `INTER_AREA`.
- `src/data/tests/test_preprocessing.py` — Added direct `cv2.INTER_AREA` reference verification on synthetic test image.
- `DECISIONS.md` — Updated DEC-008 to record area-based resizing (`cv2.INTER_AREA`).
- `PROJECT.md` — Updated DEC-008 summary.
- `docs/MODULE_MAP.md` — Updated preprocessing interface and interpolation constants.

### Verification (RESULT — All Suites Executed and Passing)

- **Block 1 Test Suite:** 23/23 tests PASS (`src/data/tests/run_tests.py`).
- **Block 2 Forensic Test Suite:** 49/49 tests PASS (`src/forensics/tests/run_tests.py`).
- **Real Defactify Smoke Test:** 5/5 sampled images successfully preprocessed with exact shape `[3, 256, 256]`, dtype float32, range in `[0.0, 1.0]`, deterministic and finite.
- **Raw Data Immutability:** `data/defactify/data/` remains untouched (17 files, 7,509,031,418 bytes).
- **Historical Analysis Directories:** `analysis_800x800/` and `first_analysis_own/` untouched.

---

## 2026-09-19 — Block 1 Phase 5: Rebuild Processed Training Dataset as train_v2


### Context

Following the preprocessing change from PIL bilinear resizing to `cv2.INTER_AREA` (see Phase 4 entry above), the canonical processed training dataset was rebuilt from scratch. The previous dataset `data/processed/train_v1/` was produced with OLD bilinear preprocessing and is no longer the canonical dataset.

### Changes

- `src/data/materializer.py` — Bumped `PREPROCESSING_VERSION` from `"v1"` to `"v2"`. The materializer now writes to `data/processed/train_v2/`.
- `DECISIONS.md` — Added DEC-010 recording `train_v2` as the canonical processed training dataset.

### Materialization (RESULT)

- **Output:** `data/processed/train_v2/`
- **Total samples:** 42,000
- **real (label_a=0):** 7,000 — matches raw train split exactly
- **AI (label_a=1):** 35,000 — matches raw train split exactly
- **Decode errors:** 0
- **Manifest:** `preprocessing_version = "v2"`, `interpolation = "INTER_AREA"`, `crop = "largest_centered_square"`
- **Schema:** `image_rgb` (binary, 256×256×3 uint8) + `label_a` (int32) only — no forbidden columns

### Verification (RESULT — All Steps Pass)

1. **Row count:** 42,000 — PASS
2. **Label distribution:** real=7,000, AI=35,000 — matches raw train split exactly — PASS
3. **Schema:** `{image_rgb, label_a}` only, no forbidden columns — PASS
4. **Manifest:** `version=v2`, `interpolation=INTER_AREA`, `total=42000` — PASS
5. **Image validation:** 1,000 sampled images all 256×256×3 uint8, finite — PASS
6. **Determinism:** 10 re-preprocessed raw samples match train_v2 byte-for-byte — PASS
7. **Bilinear vs INTER_AREA diff:** sample 0 (label_a=0) max_diff=46 — confirms preprocessing actually differs — PASS
8. **Raw data immutability:** 17 files, 7,509,031,418 bytes — unchanged — PASS
9. **Analysis directory immutability:** `analysis_800x800/` (1657 files), `first_analysis_own/` (155 files) — untouched — PASS
10. **Block 1 regression tests:** 23/23 PASS (`src/data/tests/run_tests.py`)
11. **Block 2 forensic regression tests:** 49/49 PASS (`src/forensics/tests/run_tests.py`)

### Notes

- `data/processed/train_v1/` is preserved (not deleted). It must not be used as the canonical training dataset.
- `ProcessedTrainDataset()` with no `processed_dir` argument now resolves to `train_v2/` via `get_processed_train_dir()`.
- The correct label distribution is real=7,000 / AI=35,000 — the previous session summary stated 8,000/34,000; that was incorrect. The raw train split has always been 7,000/35,000.

---

## 2026-09-19 — Block 1 Phase 6: Processed Training Dataset Directory Restructuring & Path Cleanup

### Context

To eliminate version ambiguity in runtime paths after rebuilding the processed training dataset with `cv2.INTER_AREA`, the directory structure under `data/processed/` was reorganized into a canonical path (`train/`) and an archival path (`train_old/`).

### Directory Renames (RESULT)

- `data/processed/train_v1/` $\to$ `data/processed/train_old/` (archival/audit dataset from former bilinear preprocessing).
- `data/processed/train_v2/` $\to$ `data/processed/train/` (canonical Block 1 processed training dataset using `cv2.INTER_AREA`).

### Code & Reference Updates

- `src/data/materializer.py`:
  - `materialize_training_dataset()` defaults output to `data/processed/train/`.
  - `get_processed_train_dir()` defaults resolution to `data/processed/train/`.
  - `PREPROCESSING_VERSION = "v2"` preserved for manifest version semantics.
- `src/data/processed_loader.py`: Docstring updated to reference canonical `data/processed/train/`.
- `DECISIONS.md`: DEC-010 updated to record `data/processed/train/` as canonical and `train_old/` as archival.
- `PROJECT.md`: Section 18 updated with DEC-010 canonical path.
- `docs/MODULE_MAP.md`: Updated materializer and loader path references to `data/processed/train/`.

### Verification (RESULT — All Steps Pass)

- **Directory Existence:** `data/processed/train/` and `data/processed/train_old/` verified.
- **Canonical Dataset Integrity (`data/processed/train/`):**
  - 42,000 rows (7,000 real, 35,000 AI)
  - Schema: `image_rgb` ($256 \times 256 \times 3$ uint8) + `label_a` (int32)
  - Manifest: version `v2`, interpolation `INTER_AREA`, 0 errors
- **Archival Dataset Integrity (`data/processed/train_old/`):**
  - 42,000 rows, manifest interpolation `BILINEAR`, version `v1`
- **Active References Audit:** 0 active runtime references to `train_v1` or `train_v2` in `src/`.
- **Default Resolution Smoke Test:** `ProcessedTrainDataset()` resolves directly to `data/processed/train/`, yielding tensors of shape `[3, 256, 256]` float32 in `[0, 1]` with exact label alignment.
- **Raw Data Immutability:** 17 files, 7,509,031,418 bytes byte-for-byte unchanged.
- **Block 1 Regression Tests:** 23/23 PASS (`src/data/tests/run_tests.py`).
- **Forensic A–D Regression Tests:** 49/49 PASS (`src/forensics/tests/run_tests.py`).

---

## 2026-09-20 — Block 2 Phase 5: Branch E Forensic Features (JPEG / Compression Forensics) (COMPLETE)

### Purpose

Implemented and verified Block 2 — Branch E: JPEG / Compression-Artifact forensic feature extraction, introducing four candidate sub-branches:
1. `E1_DCT`: 8×8 block DCT basis fingerprints, zonal energy ratios, non-parametric distributions, and Benford SSD (10 candidate scalar features).
2. `E2_RESP`: Controlled multi-quality in-memory JPEG recompression response curves across $Q \in \{95, 90, 75, 60\}$ (8 candidate scalar features).
3. `E3_PHASE`: Compression-stable Fourier phase spectrum response under controlled JPEG recompression at $Q \in \{90, 75\}$ (4 candidate scalar features).
4. `E4_GRID`: Canonical $8 \times 8$ grid boundary step discontinuities vs. interior gradients (4 candidate scalar features).

Total unified Branch E candidate pool: exactly 26 scalar float features.

### Added

- `docs/research/BRANCH_E_JPEG_RESEARCH.md` — Complete scientific literature review, provenance map (Pontorno et al. ICIP 2024, Li et al. CVPR 2026, Mandala 2026, Grommelt et al. 2024, B-Free CVPR 2025), data representation corrections, and leakage controls.
- `src/forensics/branch_e/dct.py` — Vectorized 2D orthonormal 8×8 block DCT-II transform (`compute_block_dct`) extracting `dct_ac_mean_abs`, `dct_ac_energy`, `dct_ac_kurtosis`, `dct_sparsity_ratio`, `dct_low_freq_ratio`, `dct_mid_freq_ratio`, `dct_high_freq_ratio`, `dct_anisotropy`, `dct_benford_ssd`, `dct_block_var_mean`.
- `src/forensics/branch_e/recompression.py` — Controlled in-memory JPEG recompression (`compute_recompression_error_map`) extracting `ela_q95_mean`, `ela_q90_mean`, `ela_q75_mean`, `ela_q60_mean`, `ela_q90_energy`, `ela_slope_q90_q75`, `ela_ratio_q90_q75`, `ela_q90_gini`.
- `src/forensics/branch_e/phase_stability.py` — Phase spectrum stability module extracting `phase_corr_q90`, `phase_corr_q75`, `phase_diff_energy_q90`, `phase_hf_stability_q90`.
- `src/forensics/branch_e/grid.py` — Canonical grid boundary discontinuity module extracting `grid_h_ratio`, `grid_v_ratio`, `grid_strength`, `grid_anisotropy`.
- `src/forensics/branch_e/features.py` — Branch E composition layer unifying E1..E4 into a 26-feature dictionary.
- `src/forensics/branch_e/__init__.py` — Clean Branch E package API.
- `src/forensics/tests/test_branch_e_forensics.py` — 12 comprehensive unit tests covering contracts, shapes, determinism, constant/pathological image safety, pattern discrimination, real-sample smoke extraction, and ForensicPipeline integration.

### Modified

- `src/forensics/pipeline.py` — Registered `E`, `E_DCT`, `E_RESP`, `E_PHASE`, `E_GRID` in `ForensicPipeline`, enabling 5-branch multi-branch combination ($A + B + C\_LBP + D\_MFR + E = 111$ candidate features).
- `src/forensics/__init__.py` — Exported Branch E extractors and helpers.
- `src/forensics/tests/run_tests.py` — Master test runner updated to execute all 61 forensic tests across Branches A, B, C, D, and E.
- `src/forensics/tests/test_branch_a_frequency.py` — Removed obsolete `NotImplementedError` check on `"E"` now that Branch E is implemented.
- `docs/MODULE_MAP.md` — Updated module inventory and architectural specifications for Branch E.

### Scientific Data Representation & Leakage Controls

1. **Canonical Input Adherence (DEC-003, DEC-008):** Branch E operates solely on preprocessed RGB uint8 $256 \times 256$ images (`cv2.INTER_AREA`). Zero container headers, EXIF metadata, filenames, or raw quantization tables are read.
2. **Qualified Terminology:** Canonical floating-point DCT and $8 \times 8$ grid steps are explicitly documented as canonical-frame representations, not reconstructions of original camera JPEG streams.
3. **Symmetric Execution (DEC-006):** In-memory JPEG recompression is applied identically to Real and AI images using the same OpenCV codec.

### Verification (RESULT — All Suites Executed and Passing)

```
Block 2 — Forensic Feature Extraction & Pipeline Test Suite (61 tests)
======================================================================
  PASS  Branch A Frequency / Periodicity Tests (11/11)
  PASS  Branch B Wavelet Haar DWT Tests (11/11)
  PASS  Branch C Local Texture Tests (15/15)
  PASS  Branch D Residual / Noise Tests (12/12)
  PASS  Branch E JPEG & Compression Forensics Tests (12/12)
======================================================================
Result: 61 passed, 0 failed, 0 errors out of 61 tests (2.53s).
```

- **Branch A–D Invariance:** All existing tests for Branches A, B, C, and D remain 100% passing without mathematical changes.
- **Block 1 Regression Suite:** 23/23 tests PASS (`src/data/tests/run_tests.py`).
- **Real Defactify Sample Smoke Extraction:** Extracted 26 finite features from canonical dataset sample (`DefactifyDataset(split="train")[0]`).
- **Pipeline Orchestration:** Verified 5-branch candidate suite `ForensicPipeline(branches=["A", "B", "C_LBP", "D_MFR", "E"])` produces exactly 111 deterministic, finite scalar features.
- **Measured CPU Runtime:** Benchmark on $256 \times 256$ image: E1_DCT = 3.10 ms, E2_RESP = 9.06 ms, E3_PHASE = 5.83 ms, E4_GRID = 0.20 ms, Unified Branch E = 18.68 ms.
- **Raw Data Immutability:** `data/defactify/` remains untouched (17 files, 7,509,031,418 bytes).
- **Exit Code:** 0.



