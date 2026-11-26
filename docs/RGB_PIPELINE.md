# RGB Pipeline --- Architecture and Implementation Specification

**Project:** AI-Generated Image Detection\
**Block:** 2 --- RGB analysis pipeline\
**Status:** Implemented and verified (DEC-014, DEC-015); Branch B active representation finalized as PiD YUV ($s=0.2$) residual map; NQP retained as research-only baseline.
**Framework:** Existing project Python environment; use PyTorch where
tensor/GPU processing is appropriate.

## 1. Purpose and scope

The RGB pipeline is the right-hand pipeline in Block 2. It derives
additional spatial representations from Block 1 images. It contains
exactly two parallel representation-extraction branches and one
validation stage:

-   **Branch A --- Lab Local Correlation Maps** (CHROMA-inspired).
-   **Branch B --- Projected Inversion Dynamics (PiD) YUV Residuals** (Active: PiD YUV $s=0.2$, DEC-015; Research baseline: NQP).
-   **Branch C --- Final Representation Validation** (quality-control
    stage, not another feature-extraction branch).

The pipeline enriches Block 1 data. It must not copy the original image
dataset.

Out of scope: model training (Block 3), final evaluation (Block 4),
extra RGB-space correlation maps, another RGB extraction branch, mRMR on
spatial maps, or a new data split.

## 2. Position in the four-block project

1.  **Block 1:** owns canonical preprocessed RGB images, labels, split
    assignments, stable sample IDs and metadata.
2.  **Block 2:** contains independent forensic and RGB pipelines.
3.  **Block 3:** trains models using original RGB and selected derived
    representations.
4.  **Block 4:** evaluates overall, per-generator, held-out-generator
    and per-perturbation performance.

Do not modify the forensic pipeline as part of this task.

## 3. Architecture

``` text
                    BLOCK 1
       Original RGB + labels + sample IDs + splits
                       |
             +---------+---------+
             |                   |
          BRANCH A            BRANCH B
       Lab correlation      PiD YUV (s=0.2)
           maps                 Δ map
             |                   |
             +---------+---------+
                       |
                    BRANCH C
              Final validation/QC
                       |
             Validated RGB outputs
                       |
          BLOCK 3 joins by sample ID
                       |
      Original RGB + selected derived maps
```

A and B process the same source sample independently. C validates both
outputs and their alignment with Block 1. C creates no new
representation.

## 4. Data contract: Block 1 remains the source of truth

Block 1 exclusively owns original images, labels, splits, sample IDs and
source metadata.

The RGB pipeline stores only derived representations and a compact
manifest/index. Do not copy original RGB images, recreate labels, or
create another split.

Each derived output must be associated with exactly one stable Block 1
`sample_id`. Block 3 must be able to retrieve the original image,
optional Lab maps, optional NQP map, label and split using that ID.

Preserve existing train/validation/test assignments. Do not use test-set
results to design or tune representations.

When Block 3 applies geometric augmentation, apply the same
crop/flip/rotation to the source image and every selected spatial map.
Do not independently randomize their geometry. Avoid colour jitter on
derived maps unless separately justified and tested.

## 5. Branch A --- Lab Local Correlation Maps

**Research inspiration:** Juan Pablo Sotelo et al., *CHROMA: Detecting
AI-Generated Images through Inter-Channel Color-Space Correlations*
(2026).

### Objective

Measure how Lab colour channels vary together within small local image
regions. In simple terms: examine local relationships between lightness
and colour, rather than measuring each channel in isolation.

### Input and conversion

Read the Block 1 RGB image. Verify actual dimensions, dtype and value
range from the real Block 1 contract; do not assume them. Convert RGB to
CIE Lab using one documented, consistent implementation.

Document: - Whether the conversion library expects RGB or BGR. - Input
value range and output Lab scaling. - Conversion
implementation/version. - Any required normalization.

Do not mix Lab conventions between training and inference.

### Correlation maps

Let the Lab channels be (L,a,b). At every pixel, use a 7 × 7 sliding
window (stride 1) and calculate Pearson correlation for:

1.  \(L\) versus (a).
2.  \(L\) versus (b).
3.  \(a\) versus (b).

For local vectors (X,Y):

\[ r\_{XY} = `\frac{\sum_i(X_i-\bar X)(Y_i-\bar Y)}`{=tex}
{`\sqrt{\sum_i(X_i-\bar X)^2}`{=tex}
`\sqrt{\sum_i(Y_i-\bar Y)^2}`{=tex}} \]

Each valid result lies in \[-1, 1\].

Outputs: - `lab_corr_La` - `lab_corr_Lb` - `lab_corr_ab`

Expected per-image shape: three maps of H × W, represented consistently
as either `3 × H × W` or `H × W × 3` according to project conventions.

### Boundaries and numerical stability

-   Preserve spatial dimensions.
-   Explicitly document padding/boundary handling.
-   Prefer matching the official CHROMA implementation where its
    behaviour can be verified.
-   A zero-variance or numerically negligible-variance window has
    undefined correlation: output 0.
-   Use epsilon only for numerical protection; do not let it turn an
    undefined flat-window correlation into a misleading valid value.
-   Clamp valid floating-point results to \[-1, 1\].
-   Never emit NaN or infinity.

Test constant and nearly constant windows. Do not create RGB-space
correlation maps.

### Output

Three spatial maps per image. Evaluate FP16 storage against FP32 after
validating numerical error.

## 6. Branch B --- Spatial Color Representation (PiD YUV & NQP)

Per **DEC-015**, Branch B has finalized its active representation as **Projected Inversion Dynamics (PiD) YUV Residuals** with scaling factor \(s=0.2\). The original **Noise Quantization Probe (NQP)** is preserved as a research-only baseline and reproducibility harness.

### 6.1 Active Representation: PiD YUV Residuals (\(s=0.2\)) [DEC-015]

**Research inspiration:** Projected Inversion Dynamics (PiD) color-space round-trip projection error.

#### Mathematical Definition
For an input RGB image \(x \in [0, 255]^{3 \times H \times W}\):
1. **Forward Transformation (RGB to Scaled YUV):**
   Using the standard ITU-R BT.601 conversion matrix \(M_{\text{YUV}}\), scale by factor \(s = 0.2\):
   \[
   y = s \cdot M_{\text{YUV}} \, x
   \]
   where
   \[
   M_{\text{YUV}} = \begin{bmatrix}
   0.299 & 0.587 & 0.114 \\
   -0.14713 & -0.28886 & 0.436 \\
   0.615 & -0.51499 & -0.10001
   \end{bmatrix}
   \]
2. **Quantization in YUV space:**
   \[
   y_q = \operatorname{round}(y)
   \]
3. **Inverse Transformation (Scaled YUV to RGB):**
   Invert using the Figure 2 inverse matrix \(M_{\text{RGB}}\) and scale by \(1/s\):
   \[
   \hat{x}_{\text{cont}} = \frac{1}{s} \cdot M_{\text{RGB}} \, y_q
   \]
   where
   \[
   M_{\text{RGB}} = \begin{bmatrix}
   1.0 & 0.0 & 1.13983 \\
   1.0 & -0.39465 & -0.58060 \\
   1.0 & 2.03211 & 0.0
   \end{bmatrix}
   \]
4. **Quantization in RGB space:**
   \[
   \hat{x} = \operatorname{clip}(\operatorname{round}(\hat{x}_{\text{cont}}), 0, 255)
   \]
5. **Signed Residual:**
   \[
   \Delta = x - \hat{x} \in [-255, 255]
   \]

#### Empirical & Operational Properties
- **Determinism:** Strictly deterministic and non-stochastic (single forward-inverse pass).
- **Density:** Yields \(\approx 15\%\) zero fraction (mean absolute residual \(\approx 1.95\)), avoiding both integer degeneracy (\(99.997\%\) zeros) and unscaled sparsity collapse (\(83.3\%\) zeros at \(s=1.0\)).
- **Efficiency:** Runs in \(\approx 1.2\)–\(1.5\text{ ms}\) per image on CPU (over \(300\times\) faster than 50-replica NQP).
- **Output:** Tensor of shape `[3, 256, 256]` float32 signed residuals.

---

### 6.2 Research Baseline: Noise Quantization Probe (NQP) [DEC-014, DEC-015]

**Research inspiration:** Zexi Jia et al., *CoDA: Color Distribution Probing for Efficient and Generalizable AI-Generated Image Detection* (2026).

#### Formulation
For \(I \in [0, 255]^{3 \times H \times W}\):
\[
I'_r = I + N_r, \quad N_r \sim \mathcal{N}(0, \sigma^2), \quad Y_r = \operatorname{clip}(\operatorname{round}(I'_r), 0, 255)
\]
\[
\hat{I} = \frac{1}{R} \sum_{r=1}^R Y_r, \quad \Delta = I - \hat{I}
\]
Paper settings: \(\sigma = 0.10\), \(R = 50\).

#### 100-Image Screening Findings & Decision
- **Integer Degeneracy:** On exact integer uint8 images (Defactify canonical dataset), \(\sigma=0.10\) produces \(99.9974\%\) zeros (\(0.0026\%\) nonzero scalars).
- **Stage 1 Screening:** An isolated 100-image screening harness evaluated 8 configurations (Exp A raw uint8, Exp B CVPR normalized \([0, 1]\) with \(\sigma=0.10\) and \(\sigma=0.01\), Exp B1 \(\sigma\) sweep from 0.001 to 0.50, Exp B2 clipping factorial, Exp C PiD, and Exp D CoDA diagnostic).
- **Mathematical & Structural Finding:** CVPR-style normalized NQP operated as a spatial high-pass / gradient filter (Moran's I \(\approx 0.296\)) rather than a pure point-wise color distribution probe.
- **Decision:** Excluded NQP from active RGB pipeline to eliminate stochastic overhead and integer degeneracy. Preserved NQP implementations (`src/rgb/branch_b_nqp/probe.py`, `analysis/branch_b_screening/`) for historical reproducibility.

---

## 7. Branch C --- Final Representation Validation

C is quality control, not a third feature extractor. It must not
transform or create another representation.

Validate:

**Structure** - Expected number of maps, spatial dimensions, channels
and dtype. - Documented value ranges.

**Numerical integrity** - No NaN or infinity. - Lab values within
\[-1,1\]. - Signed NQP residuals preserved. - Detect unexpected all-zero
or constant maps. - Assess NQP non-degeneracy using pilot evidence, not
arbitrary thresholds.

**Reproducibility** - Same sample/configuration yields the same
output. - Record configuration and implementation version.

**Dataset integrity** - Every output ID exists in Block 1. - No
unexpected duplicate or missing IDs. - Labels and splits are inherited
from Block 1. - A and B outputs align to the same source sample.

**Storage** - Read-after-write verification. - Values match pre-storage
values within documented tolerance. - Manifest paths resolve. -
Interrupted/partial outputs are detected and cannot be silently
accepted.

Produce a validation report with pass/fail status,
expected/processed/missing/invalid counts, numerical summaries, NQP
pilot status and storage verification. Only validated outputs are marked
ready for Block 3.

## 8. Output organization and on-the-fly execution

In accordance with project storage and non-duplication rules, the RGB pipeline operates on-the-fly during data loading:

- **Canonical Source:** Original images reside exclusively in Block 1 (`data/processed/train/part-*.parquet`).
- **On-the-Fly Processing:** `RGBDataset` generates Branch A (Lab local correlation maps) and Branch B (PiD YCbCr residuals) dynamically upon request during `__getitem__()`.
- **Zero Image Duplication:** No separate `rgb_dataset` directory or persistent feature maps are created on disk.
- **Optional Cache Support:** `src/rgb/materializer.py` remains available strictly as an optional offline caching utility (`data/rgb_cache/`), but is not invoked or required by the normal training path.

## 9. Block 3 integration contract


Block 2 does not train models. Block 3 must be able to run these
controlled ablations:

  Experiment   Inputs
  ------------ -----------------------------------
  A            Original RGB
  B            Original RGB + Lab maps
  C            Original RGB + PiD map (Branch B, s=0.2)
  D            Original RGB + Lab maps + PiD map

Original RGB comes from Block 1; maps come from the RGB pipeline and are
joined by sample ID.

Model architecture and fusion strategy are Block 3 decisions. Do not
blindly merge CHROMA's and CoDA's model architectures. Do not apply mRMR
to spatial maps; assess contribution through ablations.

## 10. Block 4 evaluation requirements

Document for later implementation; do not implement evaluation here: -
Overall metrics including ROC-AUC. - Per-generator metrics. -
Held-out-generator evaluation. - Particular attention to DALL-E 3. -
Per-perturbation results for perturbations present in the project test
data. - JPEG/compression robustness. - Comparisons across all four Block
3 ablations.

Design held-out-generator protocols without data leakage.

## 11. Research references and fidelity

### CHROMA

Juan Pablo Sotelo, Marina Gardella and Pablo Musé. *CHROMA: Detecting
AI-Generated Images through Inter-Channel Color-Space Correlations*
(2026). - Paper: https://arxiv.org/abs/2606.08864 - Official code:
https://github.com/JPSoteloSilva/CHROMA

Verify exact preprocessing, Lab scaling, local-correlation calculation,
padding and model-input conventions. This project adopts the Lab
local-correlation representation; it does not claim to reproduce the
complete CHROMA detector.

### CoDA

Zexi Jia et al. *CoDA: Color Distribution Probing for Efficient and
Generalizable AI-Generated Image Detection* (2026). - Paper:
https://arxiv.org/abs/2605.24306 - IEEE record:
https://doi.org/10.1109/TPAMI.2026.3696399

Verify exact NQP intensity, noise, quantization, clipping, averaging and
augmentation conventions against the paper and any official code. This
project adopts the NQP representation; it does not claim to reproduce
the complete CoDA detector.

### Fidelity rule

If paper, official code and this specification differ: 1. Identify the
discrepancy and source location. 2. Explain its practical consequence.
3. Do not silently choose an interpretation. 4. Record the final project
decision in `DECISIONS.md`.

Distinguish paper-defined settings from project-specific engineering
decisions. Do not invent paper claims or implementation details.

## 12. Documentation requirements

Keep documentation consistent with actual implementation. Preserve
existing history; append dated entries rather than overwriting unrelated
material.

Update: - `DECISIONS.md`: architecture, algorithms, parameters, data
contracts, numerical policies and deviations with rationale. -
`CHANGELOG.md`: meaningful implementation, test and documentation
changes. - `PROJECT.md`: RGB status and relationship to four-block
architecture. - `docs/RGB_PIPELINE.md`: this specification, updated when
decisions are finalized. - `docs/MODULE_MAP.md`: new RGB modules and
responsibilities. - `src/rgb/README.md`: usage, inputs, outputs,
execution and tests.

Follow existing documentation style and conventions.

## 13. Testing requirements

Follow the existing project test conventions and organize tests by
branch.

**Branch A** - RGB-to-Lab conversion and channel convention. - Correct
channel pairs. - 7 × 7 window/stride and output dimensions. - Constant
and near-constant window handling. - Correlation range and boundary
behaviour. - Determinism.

**Branch B** - Noise configuration, R=50, rounding, clipping, averaging
and signed residual. - Deterministic sample-specific seeds. -
Integer-input degeneracy detection and pilot metrics. - Memory-efficient
accumulation. - FP16/FP32 storage fidelity.

**Branch C** - Missing/duplicate IDs, split/label alignment. - Invalid
shapes/dtypes, NaN/infinity and degenerate maps. - Read-after-write
integrity and incomplete-output detection.

**Integration** - Small end-to-end fixture. - A/B sample alignment. -
Manifest correctness. - No duplicate original images. -
Resume/idempotency behaviour where supported.

After implementation, run existing Block 1 and forensic test suites
without modifying or weakening them. Report exact test counts and
results.

## 14. Implementation order and stop conditions

1.  Audit repository, documentation and current module conventions.
2.  Inspect actual Block 1 outputs and forensic structure.
3.  Create `src/rgb/` and branch-specific folders.
4.  Implement and test Branch A.
5.  Implement only the 100-image NQP pilot.
6.  **Stop for review if NQP is near-zero or its intensity convention is
    unresolved.**
7.  Implement full Branch B only after the convention is validated.
8.  Implement Branch C.
9.  Integrate manifests and storage using Block 1 sample IDs.
10. Run RGB, Block 1 and forensic tests.
11. Update required documentation.
12. Produce the completion report.

Branch A may be completed independently while NQP is under
investigation. Do not claim the full RGB pipeline is complete while the
NQP blocker remains unresolved.

## 15. Completion report

Report: - Actual final folder structure. - Responsibilities and files
for A, B and C. - Paper-to-implementation fidelity and deviations. - NQP
pilot measurements and decision. - Output shapes, dtypes and storage
format. - Processed, missing and invalid sample counts. - Test counts
and outcomes. - Documentation updated. - Remaining limitations/blockers.

**Core principle:** Block 1 owns original images and labels. RGB Block 2
owns only Lab and NQP derived representations plus validation/index
metadata. Block 3 joins representations to original samples by stable
sample ID.
