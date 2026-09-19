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

Status: UNVERIFIED

Required:

Inspect and verify all 8:

ela_q95_mean
ela_q90_mean
ela_q75_mean
ela_q60_mean
ela_q90_energy
ela_slope_q90_q75
ela_ratio_q90_q75
ela_q90_gini

Must establish:

- RGB/grayscale behavior
- recompression implementation
- exact JPEG qualities
- absolute/signed difference
- aggregation
- energy
- slope denominator
- ratio denominator
- Gini implementation
- epsilon handling
- subsampling/clipping/resizing

Also verify image-only input.

---

## I. E3 phase masking

Status: PARTIAL

Implementation inspection found:

valid Fourier bins use:

abs(F_clean) >= 1e-8

This is a near-zero magnitude exclusion.

The previous wording had an ambiguity between "non-zero Fourier bins" and "high-magnitude bins".

Required:
make documentation explicitly match:

bins where |F_clean| >= 1e-8

Do NOT call this an arbitrary "high-magnitude" mask.

Documentation correction remains to be verified/applied.

---

## J. Degenerate phase behavior

Status: UNVERIFIED

Must actually execute tests for:

- zero image
- constant image
- all-one image
- near-zero image
- random image
- Gaussian noise
- dark image
- bright image
- impulse
- step

For all E3 features verify:

- finite
- no NaN
- no Inf
- deterministic
- scalar
- expected dtype
- no unexpected exception

Document intentional fallback behavior.

---

## K. E4 grid

Status: COMPLETE

8×8 canonical grid implementation previously inspected.

Current scope is canonical grid-aligned descriptors.

Do NOT describe it as recovering the original JPEG grid.

---

## L. Direct Branch E vs ForensicPipeline

Status: UNVERIFIED

IMPORTANT:

Existence of assertions in tests is NOT sufficient.

Must actually execute direct Branch E extraction and unified ForensicPipeline extraction on deterministic images.

Required images:

- random
- constant
- dark
- bright
- impulse
- step
- natural-looking image if available

Compare:

- feature names
- ordering
- shape
- dtype
- all numerical values

Required output:

maximum absolute difference across all 26 E features and all test images.

---

## M. A-D regression

Status: UNVERIFIED

Must actually run existing regression tests and verify that adding/integrating E did not numerically alter:

A
B
C
D

Do not merely inspect test definitions.

---

## N. Image-only / data-leakage audit

Status: UNVERIFIED

Trace:

ForensicPipeline
→ Branch E
→ E1/E2/E3/E4

Must prove Branch E cannot access:

- label
- generator
- split
- caption
- filename
- path
- file size
- EXIF
- JPEG header
- JPEG quantization table
- container metadata
- Parquet metadata
- source ID

Record the concrete call chain.

---

## O. Real/fake symmetry

Status: UNVERIFIED

Must verify:

- no class-specific processing
- no label access
- no generator access
- no split access
- same recompression qualities
- same preprocessing
- same feature computation

---

## P. Numerical robustness

Status: UNVERIFIED

Must execute Branch E tests for:

- zeros
- constant
- all-one
- random
- Gaussian noise
- dark
- bright
- step edge
- impulse

For EVERY Branch E feature:

- finite
- deterministic
- scalar
- expected dtype
- no unexpected exception

---

## Q. Raw Defactify integrity

Status: UNVERIFIED

Do NOT modify raw Defactify.

Required verification:

- file count
- total bytes
- byte-level comparison if a valid baseline/tool exists

Avoid unnecessarily hashing hundreds of GB if no baseline exists.

---

## R. Full tests

Status: UNVERIFIED

Must run:

./.venv/bin/python src/forensics/tests/run_tests.py

./.venv/bin/python src/data/tests/run_tests.py

Previous expected results:

Branch E: 12/12
Full forensic: 61/61
Data/Block 1: 23/23

These numbers must be re-established by execution.

---

## S. Repository hygiene

Status: PARTIAL

Repository currently has no useful committed baseline.

Do not interpret all `??` files as audit changes.

Must inspect for:

- generated files
- temporary files
- debug output
- test artifacts
- Python caches
- accidental datasets
- unexpected notebooks

Do not delete automatically.

---

## T. Scientific documentation consistency

Status: PARTIAL

Most documentation was previously spot-checked.

Known remaining issue:

E3 phase-mask wording.

Must perform final documentation cross-check after implementation audit.

---

# 6. KNOWN PREVIOUS AUDIT HISTORY

The previous Antigravity audit was interrupted by quota exhaustion.

A subsequent recovery conversation established:

- interrupted audit did not modify source/documentation
- repository has no useful Git baseline
- no obvious audit-generated junk was found
- counts and several implementation details were statically inspected
- execution-heavy verification was NOT completed

Do NOT repeat completed static inspections unnecessarily.

---

# 7. CURRENT NEXT TASK

The next agent MUST continue from the first UNVERIFIED/PARTIAL item above.

Priority order:

1. E2 implementation audit
2. E3 execution/pathological tests
3. direct Branch E vs Pipeline numerical equivalence
4. A-D regression
5. leakage/call-chain audit
6. real/fake symmetry
7. numerical robustness
8. raw Defactify integrity
9. full test suites
10. final documentation consistency
11. final Branch E decision

Do NOT start README/commit/push/Block 4.

---

# 8. CHECKPOINT PROTOCOL

THIS SECTION IS CRITICAL.

After completing EACH major audit item:

1. Update this file immediately.
2. Change its status from UNVERIFIED/PARTIAL to COMPLETE only if actually verified.
3. Record concrete evidence:
   - command executed
   - files inspected
   - test result
   - numerical result
   - relevant finding
4. If a problem is found, record it under that item.
5. Do NOT wait until the end of the conversation to update this file.

Before any likely quota exhaustion or context exhaustion:

- update this file
- record the exact current task
- record the last completed step
- record the next step
- record any commands already executed
- record any files modified
- record whether tests are running/passed/failed

A future agent MUST be able to continue from this file without relying on previous conversation history.

---

# 9. CURRENT CHECKPOINT

Last confirmed state:

Branch E implementation is mostly statically inspected.

Execution-heavy verification remains.

Current next item:

H — E2 JPEG / ELA implementation audit.

Last completed operation:

Recovery inspection of repository state and existing Branch E documentation/source/test structure.

Next required operation:

Perform the complete E2 code-level audit.

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