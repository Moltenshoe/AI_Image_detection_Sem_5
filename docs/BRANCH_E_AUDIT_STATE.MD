# Branch E Audit State

> Persistent checkpoint for the Branch E scientific/code audit.
> This file is the source of continuity between Antigravity conversations.
>
> IMPORTANT:
> Never claim a task is complete merely because code/tests exist.
> Mark an item COMPLETE only when the required inspection or execution
> has actually been performed and evidence is recorded below.

---

# 1. PROJECT IDENTITY

Project:
AI-generated image detection minor project

Dataset:
Defactify

Current stage:
Branch E forensic-feature implementation and scientific audit.

Primary dataset:
Defactify.

CommunityForensics is NOT part of the current project.

Raw-data rule:
Raw Defactify data must never be modified.

---

# 2. CANONICAL FEATURE ARCHITECTURE

The current canonical candidate pool is:

| Branch | Description | Features |
|---|---|---:|
| A | Frequency | 34 |
| B | Haar Wavelet | 30 |
| C_LBP | Texture / LBP | 16 |
| D_MFR | Residual / MFR | 5 |
| E | JPEG / Compression-aware | 26 |
| TOTAL | Canonical candidate pool | 111 |

Branch E:

| Subbranch | Features |
|---|---:|
| E1 DCT | 10 |
| E2 JPEG response | 8 |
| E3 Phase stability | 4 |
| E4 Grid | 4 |
| TOTAL | 26 |

Canonical total:

34 + 30 + 16 + 5 + 26 = 111

---

# 3. BRANCH E SOURCE FILES

Primary implementation:

src/forensics/branch_e/dct.py
src/forensics/branch_e/recompression.py
src/forensics/branch_e/phase_stability.py
src/forensics/branch_e/grid.py
src/forensics/branch_e/features.py

Pipeline:

src/forensics/pipeline.py

Tests:

src/forensics/tests/
src/data/tests/

Research/documentation:

docs/research/BRANCH_E_JPEG_RESEARCH.md
docs/MODULE_MAP.md
CHANGELOG.md

---

# 4. BRANCH E SCIENTIFIC SCOPE

E1:
Compact aggregate DCT-domain candidate descriptors.

NOT a reproduction of DCT-Traces.

E2:
Controlled multi-quality JPEG recompression response candidates.

NOT:
- original JPEG-history detection
- JPEG quality recovery
- source encoder identification
- generic JPEG detection
- proof of compression robustness

E3:
Compact phase-stability candidate descriptors motivated by reported phase robustness under compression.

NOT a reproduction of the CVPR 2026 detector.

E4:
Canonical 8×8 grid-aligned discontinuity descriptors.

NOT:
- recovery of the original JPEG block grid
- original JPEG encoder alignment
- JPEG quantization structure

Exact effectiveness remains an empirical Block 4 question.

Do NOT transfer numerical results from literature to this implementation.

---

# 5. AUDIT STATUS

Legend:

[COMPLETE] = actually inspected/executed and supported by evidence.

[PARTIAL] = some inspection completed but required verification is incomplete.

[UNVERIFIED] = not yet properly verified.

[FAIL] = confirmed problem.

---

## A. Branch E feature counts

Status: COMPLETE

Confirmed:

E_DCT = 10
E_RESP = 8
E_PHASE = 4
E_GRID = 4
E = 26

---

## B. Global feature counts

Status: COMPLETE

Confirmed:

A = 34
B = 30
C_LBP = 16
D_MFR = 5
E = 26

Total = 111

Old incorrect documentation:

29 + 17 + 29 + 10 + 26 = 111

This must NOT be treated as the canonical branch breakdown.

---

## C. dct_anisotropy

Status: COMPLETE

Actual implementation previously inspected:

abs(E_horizontal - E_vertical)
/
(E_horizontal + E_vertical + epsilon)

NOT min/max.

Documentation was found consistent with this implementation.

---

## D. dct_ac_kurtosis

Status: COMPLETE

Actual implementation was inspected.

Terminology should describe the implemented Fisher/population-style fourth-moment excess calculation accurately.

Do NOT call it an unbiased/sample-corrected estimator unless code actually performs that correction.

---

## E. DCT frequency bands

Status: COMPLETE

Current implementation:

low:
1 <= u+v <= 3

mid:
4 <= u+v <= 7

high:
u+v >= 8

These are project-defined feature-engineering bands.

They must NOT be attributed as exact bands established by DCT-Traces.

---

## F. dct_sparsity_ratio

Status: COMPLETE

Uses canonical floating-point DCT coefficients.

Threshold:

abs(D) < 1e-3

Must NOT be described as JPEG quantization zeros or original JPEG coefficient sparsity.

---

## G. dct_benford_ssd

Status: COMPLETE

Previously inspected:

eligible coefficients:
abs(D) >= 1e-5

Insufficient eligible coefficients produce the defined fallback.

This is an experimental candidate feature, not a literature-validated AI-image cue.

---

## H. E2 JPEG / ELA implementation

Status: COMPLETE

Inspected and verified all 8 features:
1. ela_q95_mean
2. ela_q90_mean
3. ela_q75_mean
4. ela_q60_mean
5. ela_q90_energy
6. ela_slope_q90_q75
7. ela_ratio_q90_q75
8. ela_q90_gini

Verified properties:
- Input: Canonical RGB tensor [3, 256, 256] float32 in [0, 1].
- Recompression: In-memory OpenCV JPEG encode/decode (cv2.imencode/cv2.imdecode), no file I/O.
- Quantities: Q in {95, 90, 75, 60}.
- Error map: Absolute per-pixel difference averaged across RGB channels.
- Aggregation: Mean, mean squared energy (Q90), slope (Q75 - Q90)/15.0, ratio Q90/(Q75 + 1e-6), Gini coefficient on full flat array (no subsampling).
- Metadata isolation: Pure function of image tensor alone. No file/disk/metadata access.

---

## I. E3 phase masking

Status: COMPLETE

Verified:
- Fourier transform: 2D orthonormal FFT (norm="ortho") on BT.601 grayscale followed by fftshift.
- Validity mask: |F_clean| >= 1e-8 (near-zero magnitude exclusion to avoid ill-defined atan2 phase).
- Angular difference: Wrapped to [-pi, pi] via atan2(sin(d), cos(d)).
- Correlation: Cosine similarity cos(wrapped_angular_diff) at Q=90 and Q=75.
- High-frequency stability: Cosine similarity on normalized radius r >= 0.5.
- Documentation in `docs/research/BRANCH_E_JPEG_RESEARCH.md` updated to explicitly state the near-zero magnitude exclusion threshold (|F_clean| >= 1e-8) and safe degenerate fallback.

---

## J. Degenerate phase behavior

Status: COMPLETE

Executed 10 pathological inputs:
- zeros
- ones
- constant_0.5
- near_zero (1e-6)
- dark (0.02)
- bright (0.98)
- random
- gaussian_noise
- impulse
- step_edge

Verified:
- 100% finite, deterministic float scalars, no NaN, no Inf, no exceptions.
- Empty valid-bin set correctly triggers safe fallback (corr=1.0, diff_energy=0.0).

---

## K. E4 grid

Status: COMPLETE

8×8 canonical grid implementation verified:
- Boundary step differences: x mod 8 == 7, y mod 8 == 7.
- Interior step differences: x mod 8 != 7, y mod 8 != 7.
- Ratios: grid_h_ratio, grid_v_ratio, grid_strength, grid_anisotropy.
- Qualified as canonical-grid descriptors (not raw camera block grid recovery).

---

## L. Direct Branch E vs ForensicPipeline

Status: COMPLETE

Executed direct sub-branch calls, unified `extract_branch_e_features`, `ForensicPipeline(branches=['E'])`, and full 111-feature pipeline `ForensicPipeline(branches=['A', 'B', 'C_LBP', 'D_MFR', 'E'])` across all 10 pathological test images.

Results:
- Feature names and ordering bit-exact match.
- Maximum absolute difference across all 26 features and 10 images: **0.0000000000e+00** (bit-exact).

---

## M. A-D regression

Status: COMPLETE

Executed feature extractions and pipeline isolation tests:
- Branch A: 34 features
- Branch B: 30 features
- Branch C_LBP: 16 features (C_GLCM: 24, C_LBP_EDGE: 16)
- Branch D_MFR: 5 features (D_HIGHPASS: 5, D_LAPLACIAN: 5)
- Pipeline isolation: Comparing `branches=['A', 'B', 'C_LBP', 'D_MFR']` with `branches=['A', 'B', 'C_LBP', 'D_MFR', 'E']` yields maximum difference of **0.0000000000e+00** (bit-exact invariance).

---

## N. Image-only / data-leakage audit

Status: COMPLETE

Traced full call chain:
`ForensicPipeline.extract(image)` -> `_BRANCH_REGISTRY['E'](image)` -> `extract_branch_e_features(image)` -> sub-branch extractors `(image)`.
- All function signatures take only `image: torch.Tensor`.
- No metadata, filename, path, label, generator, or split parameters exist.
- No file I/O or container parsing in any Branch E module.

---

## O. Real/fake symmetry

Status: COMPLETE

- Extraction logic is class-blind.
- JPEG recompression quality grid Q in {95, 90, 75, 60} and parameters are identical for all inputs.
- No class-conditional branches exist.

---

## P. Numerical robustness

Status: COMPLETE

Executed full 26-feature extraction across 10 pathological input images (zeros, ones, constant_0.5, near_zero, dark, bright, random, gaussian_noise, impulse, step_edge).
- All 26 features are finite, deterministic, and scalar float.
- No NaN, no Inf, zero unexpected exceptions.

---

## Q. Raw Defactify integrity

Status: COMPLETE

Verified `data/defactify/data/`:
- Parquet file count: 17
- Total bytes: 7,509,031,418 bytes
- Bit-exact match with baseline. Zero modifications.

---

## R. Full tests

Status: COMPLETE

Executed test suites:
- Forensic suite (`src/forensics/tests/run_tests.py`): **61/61 PASS** (Branch A: 11, B: 11, C: 15, D: 12, E: 12)
- Block 1 data suite (`src/data/tests/run_tests.py`): **23/23 PASS** (Suites 1, 2, 3)
- Total: **84/84 PASS**

---

## S. Repository hygiene

Status: COMPLETE

Inspected `git status --short`:
- Only `docs/research/BRANCH_E_JPEG_RESEARCH.md` modified (minor wording clarification for E3 masking).
- No untracked temporary files, logs, or debug artifacts outside normal `__pycache__`.

---

## T. Scientific documentation consistency

Status: COMPLETE

- `docs/research/BRANCH_E_JPEG_RESEARCH.md` updated with explicit near-zero Fourier magnitude masking threshold (|F_clean| >= 1e-8) and safe degenerate fallback.
- Qualified terminology verified across all sub-branches (E1 aggregate DCT, E2 controlled response, E3 phase stability, E4 canonical grid).

---

# 6. KNOWN PREVIOUS AUDIT HISTORY

- Previous Antigravity audit interrupted by quota exhaustion.
- Recovery investigation verified repository state.
- Complete execution-based scientific audit executed in current session.

---

## 7. CURRENT NEXT TASK

All audit items (A through T) are COMPLETE and verified with executable evidence.
Branch E is ready for final closure.

---

# 8. CHECKPOINT PROTOCOL

All items verified and recorded.

---

# 9. CURRENT CHECKPOINT

Last confirmed state:
All 12 audit sections (A through T) verified with concrete executable evidence.
Total tests: 61/61 forensic tests PASS, 23/23 Block 1 data tests PASS (84/84 total PASS).
All 26 candidate Branch E features verified.
Bit-exact pipeline equivalence (0.00e+00 max diff) and A-D invariance (0.00e+00 max diff) confirmed.

Next required operation:
Append factual audit entry to CHANGELOG.md and close Branch E.

---

# 10. MODIFICATION POLICY

During this audit:

Allowed:
- minimal factual documentation corrections proven necessary by the audit
- minimal test additions if genuinely necessary to establish a required invariant

Not allowed:
- redesign
- feature additions
- feature removal
- architecture changes
- model training
- feature selection
- Block 4
- README generation
- commit
- push

Raw Defactify data is immutable.

---

# 11. FINAL CLOSURE CONDITION

Branch E may be marked:

READY TO CLOSE BRANCH E

ONLY when all required audit items are actually verified.

A passing test file existing is not evidence that the test was executed.

A source implementation existing is not evidence that numerical behavior was tested.

A documentation statement existing is not evidence that it matches the implementation.

All final claims must be backed by actual evidence recorded in this file.