# RGB Branch B Research Harness — Implementation & Verification Report

**Repository:** `/home/moltenshoe/Desktop/ArchFiles/UBUNTU_FILES/projects/Minor`  
**Execution Date:** 2026-11-14  
**Python Environment:** Python 3.14.7 (`.venv/bin/python`), PyTorch 2.14.0+cu132, scikit-image 0.26.0  
**Status:** Implementation Complete, Verified & Isolated — Awaiting User Approval for Full Experiment Execution  

---

## 1. Executive Summary

The isolated research screening harness for the Block 2 RGB Spatial Pipeline (Branch B investigation) has been fully implemented and verified under `analysis/branch_b_screening/`.

All requirements specified in the research investigation, mathematical reconciliation, and verification plans have been strictly implemented without modifying production code or violating architectural invariants.

### Strict Invariants Maintained
1. **Production Code is Frozen:** `src/rgb/` (Branch A Lab, Branch B NQP baseline, Branch C Validation, Pipeline, Dataset, Materializer) and all production modules remain 100% untouched.
2. **DEC-014 Gate Preserved:** `analysis/rgb_nqp_pilot/pilot_report.json` remains untouched. The materialization gate in `src/rgb/materializer.py` remains active and unmodified.
3. **External Contracts Preserved:** All candidate representations accept canonical RGB tensors (`[3, 256, 256]` in `[0, 255]` or `[0.0, 1.0]`) and return signed float32 residual tensors of shape `[3, 256, 256]`.
4. **Complete Isolation:** All research candidates, configs, data loaders, metrics, runners, manifests, and test suites are isolated under `analysis/branch_b_screening/`.
5. **No Experiments or Model Training Executed:** In accordance with explicit instructions, full 100-image screening extraction, dataset materialization, model training, and expensive GPU benchmarks were **NOT** executed.

---

## 2. Completed Components

The research harness comprises the following components:

### 2.1 Candidate Representations (`analysis/branch_b_screening/candidates/`)
- **`raw_nqp.py` (Candidate A — Reference Control):**
  - Faithfully reproduces existing `src/rgb/branch_b_nqp/probe.py` raw uint8 behavior in an isolated module.
  - Implements deterministic SHA-256 seed derivation (`derive_sample_seed(base_seed=42, sample_id)`).
  - Implements `compute_raw_nqp_from_noise` for controlled synthetic threshold-crossing and cancellation tests.
  - Residual sign convention: $\Delta_{\text{A}} = I - \hat{I}$ (original minus reconstructed image).
- **`cvpr_color.py` (Candidate B — CVPR 2025 4-Bit Quantization):**
  - Uniform color quantizer $Q_{16}(x) = \operatorname{clamp}(\operatorname{round}(15x) / 15, 0.0, 1.0)$ with step $q = 1/15 \approx 0.0667$.
  - Reference reproduction `extract_cvpr_reference`: unseeded stochastic execution using `skimage.util.random_noise(..., mode="gaussian", clip=True, var=0.01)`.
  - Deterministic project adaptation `extract_cvpr_seeded`: PyTorch `torch.randn` with deterministic SHA-256 per-image seed.
  - Residual sign convention: $\Delta_{\text{CVPR}} = \hat{x} - x$ (reconstruction minus original continuous input).
  - Independent 2-stage clipping flags (`clip_noise` and `clip_quant`).
- **`pid_yuv.py` (Candidate C — PiD ICML 2025 Deterministic Decomposition):**
  - Standard ITU-R BT.601 forward matrix $M_{\text{YUV}}$ ($RGB \to YUV$).
  - Verified inverse matrix $M_{\text{inv}}$ corroborated by paper Figure 2 ($[1.0, 1.772, 0.0]$ in row index 2 reconstructing Blue $B = Y + 1.772U$).
  - Two discrete rounding stages: nearest-integer in YUV transformed space, and post-inverse nearest-integer in RGB space.
  - Table 5 matrix scaling ablation support: $s \in \{0.2, 0.5, 0.8, 1.0, 1.5, 2.0\}$.
  - Residual sign convention: $R_{\text{PiD}}(x) = x - \hat{x}$ (signed float32 in $[-255.0, 255.0]$).
  - Deterministic single-pass ($R=1$, zero RNG noise, no arbitrary clipping).
- **`coda_diagnostic.py` (Candidate D — CoDA 2026 Analytical Diagnostic):**
  - Separates Algorithm 1 (raw 8-bit Monte Carlo integer probe) from Section IV-C Equation 12 (normalized unit-step Fourier periodic bias model).
  - Strictly analytical diagnostic; explicitly flagged as NOT a candidate for Block 2 dataset materialization.

### 2.2 Data Handling & Stratified Manifest (`analysis/branch_b_screening/data/`)
- **`sampling.py`:**
  - Implements deterministic stratified selection of exactly 100 training samples matching the historical pilot quota:
    - Real (20)
    - SD2.1 (16)
    - SDXL (16)
    - SD3 (16)
    - DALL-E3 (16)
    - Midjourney v6 (16)
  - Exports immutable static manifest to `manifests/screening_samples.csv` with SHA-256 checksum verification (`a9a924279c2722cb1967dffb72c69f50f8482f02cf7895c5c7d363fd00c10f60`).
  - Precomputes Parquet part filename (`part-*.parquet`), row group index, and row-in-group index for fast $O(1)$ random access.
- **`sample_reader.py` (`ScreeningSampleReader`):**
  - On-demand single-sample reader loading individual $256 \times 256 \times 3$ uint8 arrays directly from Block 1 Parquet row groups.
  - Raises unhandled `DataIntegrityError` on missing parts, corrupt bytes, or row-length discrepancies (strictly no silent substitution).

### 2.3 Diagnostic Metrics & Analysis (`analysis/branch_b_screening/diagnostics/`)
- **`metrics.py`:**
  - Domain-aware numerical tolerances: $\tau_{\text{norm}} = 10^{-6}$ for normalized float32, $\tau_{\text{raw}} = 10^{-4}$ for raw uint8.
  - Level A Scale-Aware Degeneracy (Anti-Collapse): Exact-zero fraction ($f_0$), tolerance-based zero fraction ($f_\tau$), dynamic range ($\text{DR} = \max - \min$), maximum absolute residual ($\text{MaxAR}$), RMS, and empirical variance.
  - Sparse Representation Preservation: Representations with high sparsity ($f_\tau > 0.90$) are preserved as valid if $\text{DR} \gg \tau$ and $\text{MaxAR} \gg \tau$.
  - Level B Characterization: Pixel-level nonzero fraction, quantiles (p50, p95, p99), sign proportions, and unique-value cardinality.
  - Class- and generator-stratified aggregation.
- **`spatial_metrics.py`:**
  - Moran\'s I spatial autocorrelation via direct 4-connectivity slice cross-products in $O(HW)$ time without materializing $65,536 \times 65,536$ dense adjacency matrices.
  - Gini spatial concentration coefficient via $O(N \log N)$ sorting.
- **`intensity_profiles.py`:**
  - Intensity-conditioned expectation $E[\Delta \mid I \in B_k]$ and variance $\operatorname{Var}(\Delta \mid I \in B_k)$ across 16 equal-width intervals.
  - Distinguishes interior quantization behavior from boundary clipping ($B_0$ and $B_{15}$).
  - Explicitly represents empty bins as `None` (not `0.0`).
- **`visualizations.py`:**
  - Matplotlib non-interactive generation of side-by-side original image and R/G/B residual heatmaps, plus intensity-conditioned profile plots.

### 2.4 Experiment Configurations (`analysis/branch_b_screening/configs/`)
- `exp_a_baseline.json`: Raw NQP baseline ($R=50, \sigma=0.10$).
- `exp_b_cvpr_reference_500.json`: CVPR supplementary reproduction ($R=500, \text{var}=0.01$).
- `exp_b_cvpr_reference_50.json`: CVPR supplementary reproduction ($R=50, \text{var}=0.01$).
- `exp_b_cvpr_seeded_50.json`: CVPR deterministic project adaptation ($R=50, \sigma=0.10$, SHA-256 seed).
- `exp_b1_sweep.json`: Controlled noise-to-quantization ratio sweep ($\sigma/q \in \{0.1, 0.2, 0.3, 0.5, 1.5\}$).
- `exp_b2_clipping.json`: $2 \times 2$ factorial clipping disentanglement (noise clipping $\times$ quantizer clipping).
- `exp_c_pid.json`: PiD ICML 2025 deterministic decomposition ($s=1.0$ unscaled reference, six scaling factors).
- `exp_d_coda_diagnostic.json`: CoDA Algorithm 1 vs Eq 12 diagnostic analysis.

### 2.5 Screening Runner (`analysis/branch_b_screening/runners/run_screening.py`)
- Master CLI runner with support for single experiments, sweeps, factorials, and complete screening.
- Includes `--dry-run` pipeline verification mode testing manifest loading, Block 1 Parquet reading, and candidate execution on sample 0.
- Captures environment metadata (Python version, PyTorch version, CUDA hardware, Git commit hash `e0a08a7c0fc47c66e15a6ec41c3066e1bda62756`) to `metadata/environment.json`.
- Guards full execution behind explicit approval.

---

## 3. Automated Test Verification Results

All automated test suites were executed in the repository virtual environment (`.venv/bin/python`):

| Test Suite | File | Tests Run | Passed | Failed | Errors | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Branch B Research Suite** | `analysis/branch_b_screening/tests/run_tests.py` | 38 | 38 | 0 | 0 | **PASS** |
| • Raw NQP Baseline | `tests/test_raw_nqp.py` | 6 | 6 | 0 | 0 | **PASS** |
| • CVPR 2025 (incl. sweep & factorial) | `tests/test_cvpr_color.py` | 8 | 8 | 0 | 0 | **PASS** |
| • PiD ICML 2025 (matrices & scaling) | `tests/test_pid_yuv.py` | 6 | 6 | 0 | 0 | **PASS** |
| • CoDA Diagnostic Formulation | `tests/test_coda_diagnostic.py` | 4 | 4 | 0 | 0 | **PASS** |
| • Scale-Aware & Spatial Metrics | `tests/test_metrics.py` | 9 | 9 | 0 | 0 | **PASS** |
| • Sampling & Parquet Reader Integrity | `tests/test_sampling.py` | 5 | 5 | 0 | 0 | **PASS** |
| **Production RGB Pipeline Suite** | `src/rgb/tests/run_tests.py` | 18 | 18 | 0 | 0 | **PASS** |
| **Production Forensic Pipeline Suite**| `src/forensics/tests/run_tests.py` | 89 | 89 | 0 | 0 | **PASS** |
| **Production Block 1 Data Suite** | `src/data/tests/run_tests.py` | 23 | 23 | 0 | 0 | **PASS** |
| **TOTAL** | — | **168** | **168** | **0** | **0** | **100% PASS** |

### Dry-Run Smoke Execution Output
Running `python analysis/branch_b_screening/runners/run_screening.py --dry-run` succeeded with exit code 0:
- Loaded canonical sample index 792 (`train_000792`, Real).
- Verified raw image: shape `[3, 256, 256]`, dtype `uint8`, range `[0, 255]`.
- Verified normalized image: shape `[3, 256, 256]`, dtype `float32`, range `[0.0000, 1.0000]`.
- Exp A (Raw NQP): range `[0.0000, 0.0000]`.
- Exp B (CVPR Seeded): range `[-0.3235, 0.3824]`.
- Exp B1 (Sweep smoke MARs across ratios): `[0.0139, 0.0156, 0.0563]`.
- Exp B2 (Factorial smoke MARs across 4 clipping configs): `[0.0563, 0.0563, 0.0563, 0.0576]`.
- Exp C (PiD YUV): range `[-1.0000, 1.0000]`.
- Exp D (CoDA Diagnostic): status `coda_formulation_comparison`.

---

## 4. Unresolved Issues & Environmental Notes
1. **CUDA Driver & Execution Environment:** When running via CLI, CUDA device `cuda:0` is available and utilized for PyTorch tensor operations.
2. **Evaluation Splits:** Block 1 provides only the 42,000-image training split. Validation and test splits are not materialized in the repository. The 100 screening samples are strictly drawn from this training pool.
3. **No Unresolved Blockers:** All candidate algorithms, configurations, metrics, loaders, and test suites are complete, fully functional, and ready for deployment upon approval.
