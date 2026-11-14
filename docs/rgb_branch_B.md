# RGB Branch B --- Research, Verification, and Replacement Specification

**Status:** Research specification; no production change authorized\
**Intended location:** `docs/rgb_branch_B.md`\
**Scope:** RGB pipeline Branch B only\
**Primary objective:** Determine whether a more useful
quantization/residual representation can replace the current Branch B
implementation without changing the RGB pipeline architecture or its
external contracts.

------------------------------------------------------------------------

## 1. Executive summary

The current Branch B implementation follows the 8-bit procedure
described in CoDA 2026 Algorithm 1:

-   Input image is represented in the 8-bit range `[0, 255]`.
-   Gaussian noise with standard deviation `0.10` is added.
-   Values are rounded to integer values and clipped to `[0, 255]`.
-   The operation is repeated `R = 50` times.
-   The quantized replicas are averaged.
-   A residual is computed between the original image and the averaged
    image.

The existing 100-image pilot found an extremely sparse residual:

  Metric                        Observed value
  --------------------------- ----------------
  Nonzero scalar count                     513
  Total scalar count                19,660,800
  Nonzero scalar fraction            0.002609%
  Nonzero pixel fraction             0.007828%
  Mean absolute residual              5.219e-7
  Maximum absolute residual           0.020004
  Unique residual values                     7

The pilot indicates that the current implementation is practically
degenerate for the intended feature-extraction purpose. It does **not**
establish that quantization-based forensic residuals are ineffective in
general.

The investigation will compare three independent directions:

1.  **Existing CoDA-style 8-bit implementation:** retain as a baseline
    and document its behavior.
2.  **CVPR 2025 4-bit quantization/restoration method:** reproduce the
    authors' supplementary implementation as closely as possible, then
    investigate its finite-replica and clipping behavior.
3.  **PiD-style deterministic color-space quantization residual:**
    reproduce as a separate candidate using the paper's actual
    transformation and quantization details.

A separate dataset-confound audit is required before classification
experiments.

**No architecture redesign is proposed.** Branch A, Branch C,
RGBPipeline integration, RGBDataset contracts, Block 3, and Block 4
remain unchanged. A candidate may only be considered for replacement
inside Branch B after it demonstrates a reliable complementary
improvement over an RGB-only baseline.

------------------------------------------------------------------------

## 2. Non-negotiable scope and constraints

### 2.1 In scope

-   Inspect and document the existing Branch B implementation.
-   Read the primary papers and supplementary material.
-   Reproduce candidate methods in an isolated research/scratch area.
-   Run synthetic numerical validation after approval.
-   Run a balanced numerical pilot after approval.
-   Audit Defactify for technical/source confounds.
-   Design and later run a controlled discriminative evaluation.
-   Recommend whether any candidate merits replacement of Branch B
    internals.

### 2.2 Out of scope

Do not redesign or modify:

-   Branch A --- Lab local correlation features.
-   Branch C --- validation and quality control.
-   `RGBPipeline` integration.
-   `RGBDataset` external contract.
-   Existing Block 1 data pipeline.
-   Forensic pipeline.
-   Block 3 model architecture or training system.
-   Block 4 evaluation system.

Do not modify production Branch B while conducting the research
investigation.

Do not remove Branch B, bypass its materialization gate, weaken tests,
regenerate datasets, overwrite existing processed data, reinstall
packages, or rebuild the Python environment.

### 2.3 Production replacement gate

A candidate is not eligible for production replacement merely because:

-   It produces nonzero residuals.
-   Its residual maps look visually interesting.
-   It has a larger residual magnitude.
-   It performs well on the 100-image numerical pilot.
-   It performs well as a residual-only classifier.

A candidate must demonstrate a reproducible improvement over the
RGB-only baseline in a controlled evaluation, including robustness and
generator-wise reporting. Any eventual replacement must preserve Branch
B's external input/output contract.

------------------------------------------------------------------------

## 3. Current implementation and mathematical diagnosis

### 3.1 Current Branch B behavior

The current implementation is understood to use:

-   Input: uint8 RGB image, values in `[0, 255]`.
-   Noise: Gaussian, `sigma = 0.10` in 8-bit intensity units.
-   Quantization: nearest-integer rounding.
-   Clipping: `[0, 255]`.
-   Replicas: `R = 50`.
-   Restoration: arithmetic mean of the quantized replicas.
-   Residual: original minus restored image.
-   Randomness: deterministic per-image seed.

Verify these details directly against the repository before experiments.
Treat the repository as authoritative for what is currently implemented;
do not assume this document replaces source inspection.

### 3.2 Why the residual is nearly zero

For an interior integer-valued scalar pixel $I = k$, and symmetric
Gaussian noise $N \sim \mathcal{N}(0, \sigma^2)$, nearest-integer rounding has symmetric outcomes
around $k$.

Ignoring clipping and tie events:

$$E[\operatorname{round}(k+N)] = k$$

Therefore, the expected residual is identically zero for interior integer-grid values:

$$E[k - \operatorname{round}(k+N)] = 0$$

This symmetry holds for any Gaussian standard deviation away from boundaries. Increasing
$\sigma$ increases the variability of individual quantized samples, but does not introduce a systematic non-zero bias for integer-valued inputs centered on decision intervals.

For $\sigma = 0.10$ raw intensity units, a rounding change occurs when the noise magnitude
reaches or exceeds $0.5$:

$$P(|N| \ge 0.5) = 2(1 - \Phi(5.0)) \approx 5.733 \times 10^{-7}$$

Across $R = 50$ independent Monte Carlo replicas, the probability of at least one crossing
for a single scalar is approximately:

$$1 - (1 - 5.733 \times 10^{-7})^{50} \approx 50 \times 5.733 \times 10^{-7} \approx 2.8665 \times 10^{-5} \quad (0.0028665\%)$$

This matches the observed nonzero scalar fraction of $0.002609\%$ (513 / 19,660,800).
- **Net Signed Difference vs. Single Flip:** An observed residual magnitude of $|\Delta| = 0.02 = 1/50$ establishes that the net signed sum of replica changes $\sum_{r=1}^{50} (Y_r - k)$ was $\pm 1$. While theoretically multiple cancelling flips could occur, their joint probability is negligible ($\approx 10^{-10}$), making single-replica flips overwhelmingly dominant.
- **Scalar vs. Pixel Fractions:** In the pilot, all 513 nonzero scalars occurred in 513 distinct pixels. Across 6.55 million pixels, zero pixels had multiple color channels perturbed simultaneously. The observed residuals represent isolated, uncorrelated Gaussian tail spikes rather than structured color features.
- **Boundary Bias:** At clipping boundaries ($k=0$ and $k=255$), one-sided clipping creates an asymmetric expected bias of magnitude $\approx 2.87 \times 10^{-7}$, which is negligible compared to normal representation values.

### 3.3 CoDA theoretical discrepancy and input-domain reconciliation

CoDA 2026 Algorithm 1 specifies an 8-bit image in $[0,255]$, Gaussian noise with $\sigma = 0.10$, integer rounding, and clipping. Its theoretical Section IV-C shifts to normalized intensity $x = I/256 \in [0,1)$ and writes Equation 12:

$$Y = \operatorname{round}(x + N)$$

where rounding is to the nearest integer $\mathbb{Z}$.

#### A. Resolution of the Mathematical Formulation
1. **Equation 12 as Printed:** Operates with a unit-step quantizer on normalized $x \in [0, 1)$. Its theoretical periodic bias function:
   $$\mathcal{B}(x; \sigma) = \sum_{m=1}^{\infty} \frac{(-1)^m}{m\pi} \exp(-2\pi^2\sigma^2 m^2) \sin(2\pi m x) \approx \frac{1}{\pi} \exp(-2\pi^2\sigma^2) \sin(2\pi x)$$
   is **generally non-zero** at $x = k/256$ because $\sin(2\pi k / 256) \neq 0$ for $k \notin \{0, 128, 256\}$.
2. **Non-Equivalence to Algorithm 1:** Equation 12 is a theoretical toy model with only a single decision boundary at $x = 0.5$ across the normalized range $[0, 1)$. It is **not** equivalent to Algorithm 1.
3. **Correctly Normalized Algorithm 1:** Transforming Algorithm 1 into normalized coordinates requires scaling both the quantizer step and the noise:
   $$q_{\text{norm}} = \frac{1}{256}, \quad \sigma_{\text{norm}} = \frac{0.10}{256} \approx 0.0003906, \quad Y_{\text{norm}} = \frac{\operatorname{round}(256x + N_{\text{raw}})}{256}$$
   Under this coordinate transformation, the noise-to-step ratio remains $\sigma/q = 0.10$.
4. **Input-Domain Consideration:** On exact integer uint8 inputs, interior expected residuals are zero. However, in the authors' training pipeline, extensive data augmentations (resizing, ColorJitter, rotation, cropping) may have introduced continuous floating-point values before the probe was applied. In the absence of official training code, the training-time tensor dtype remains an unresolved ambiguity. On our canonical Block 1 integer uint8 images, the raw 8-bit probe is demonstrably degenerate.

------------------------------------------------------------------------

## 4. Primary research references

Read the full papers, relevant equations, ablations, and supplementary
materials. Do not rely only on abstracts or secondary summaries.

### 4.1 CVPR 2025 --- Secret Lies in Color

Jia et al., **"Secret Lies in Color: Enhancing AI-Generated Images
Detection with Color Distribution Analysis"**, CVPR 2025.

-   Paper:
    https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Secret_Lies_in_Color_Enhancing_AI-Generated_Images_Detection_with_Color_CVPR_2025_paper.pdf
-   Supplementary material:
    https://openaccess.thecvf.com/content/CVPR2025/supplemental/Jia_Secret_Lies_in_CVPR_2025_supplemental.pdf
-   Author page: https://zexijia.github.io/

The supplementary material includes color-feature extraction code. The
implementation described in the supplied code uses a quantizer of the
form:

\[ Q_L(x)=`\frac{\operatorname{round}((L-1)x)}{L-1}`{=tex} \]

with clipping to `[0,1]`.

For `L = 16`:

\[ Q\_{16}(x)=`\frac{\operatorname{round}(15x)}{15}`{=tex} \]

The nominal normalized quantization interval is `1/15`; expressed over a
0--255 range, it is `255/15 = 17`, not 16.

The supplied code adds Gaussian noise to the normalized image using
`skimage.util.random_noise`, quantizes each noisy replica, averages the
processed replicas, and computes a residual with the sign:

\[ `\Delta`{=tex}\_{`\mathrm{CVPR}`{=tex}}=`\hat `{=tex}I-I \]

The existing Branch B uses the opposite sign. The experimental
reproduction must document this. A comparison adapter may convert the
candidate residual to the sign expected by the existing Branch B
interface, but the reproduction itself must preserve and report the
authors' convention.

The supplied code also computes a quantized original image that appears
not to be used in the subsequent averaging/residual calculation. Inspect
the exact code and report this rather than silently deleting or changing
the operation.

The scikit-image `random_noise` documentation gives a default Gaussian
variance of `0.01` for the Gaussian mode, corresponding to standard
deviation `0.1` in normalized units. Verify the exact version, function
signature, default behavior, clipping behavior, dtype/range handling,
and random-state behavior in the installed environment before relying on
these defaults.

Relevant documentation:
https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise

The CVPR paper reports a 4-bit setting outperforming its tested 8-bit
and 2-bit settings on its evaluated benchmark. In Table 5 (rows 10–12), reported FakeART
accuracies are 66.5% (4-bit), 63.9% (8-bit), and 61.1% (2-bit).

- **Replica Count Parameter:** The supplementary function `process_image_batch` accepts `repetition_count` as an explicit argument with **no default value**.
- **Reported Experimental Configuration:** Table 5 in the paper compares 200, 300, and 500 noise images, achieving best accuracy at **500 noise images** ($R = 500$). This is the **strongest reported CVPR ablation configuration**, not an official software default or necessarily the authors' universal production default.
- **Noise-to-Step Ratio:** With $q = 1/15 \approx 0.0667$ and scikit-image default $\sigma = 0.10$, the ratio is $\sigma/q = \mathbf{1.50}$.
- **Dual Clipping:** Clamping occurs both at the noise injection stage (`clip=True` in `util.random_noise`) and at the quantizer output (`.clamp(0, 1)`).
- **Residual Sign:** $\Delta_{\text{CVPR}} = \hat{I} - I$ (reconstruction minus original), opposite to our Branch B.

### 4.2 CoDA 2026

**Paper:** Zexi Jia, Zhiqiang Yuan, Xiaoyue Duan, Jinchao Zhang, Jie Zhou, Anil K. Jain, *"CoDA: Color Distribution Probing for Efficient and Generalizable AI-Generated Image Detection"*, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Volume 48, Issue 10, October 2026, pages 12100–12115, DOI: [10.1109/TPAMI.2026.3696399](https://doi.org/10.1109/TPAMI.2026.3696399) (Preprint: arXiv:2605.24306v1, 26 May 2026).

1. **Algorithm 1 (Section IV-B):**
   - Operates on raw RGB $I \in [0, 255]$ with Gaussian noise $\sigma = 0.10$ raw intensity units.
   - Quantization: nearest-integer rounding with clipping to $[0, 255]$.
   - Monte Carlo averaging over $R = 50$ replicas.
   - Residual: $\Delta = I - \hat{I}$ (matches current Branch B).
   - Expected residual on interior integer pixels is identically zero ($E[\Delta] = 0$).
2. **Theoretical Model (Section IV-C, Equation 12):**
   - Operates on normalized intensity $x = I/256 \in [0, 1)$ with $Y = \operatorname{round}(x + N)$ rounding to the nearest integer $\mathbb{Z}$ (unit quantization step).
   - Derives a period-one Fourier bias: $\mathcal{B}(x; \sigma) \approx \frac{1}{\pi} e^{-2\pi^2\sigma^2} \sin(2\pi x)$.
   - This theoretical bias does **not** vanish at $x = k/256$ because $\sin(2\pi k / 256) \neq 0$ for $k \notin \{0, 128, 256\}$.
   - Equation 12 defines a unit-step round on a normalized variable in $[0, 1)$, which differs in quantization step and coordinate scaling from the discrete 8-bit integer operations of Algorithm 1.
3. **Public Implementation Status:** No official public codebase was released for CoDA. Whether the original authors applied data augmentations (resizing, rotation, color jitter) that introduced fractional float values prior to the probe remains an unresolved ambiguity.

### 4.3 PiD — ICML 2025

**Paper:** Xinghe Fu, Zhiyuan Yan, Zheng Yang, Taiping Yao, Yandan Zhao, Shouhong Ding, Xi Li, *"PiD: Generalized AI-Generated Images Detection with Pixelwise Decomposition Residuals"*, ICML 2025, PMLR 267:17894–17908.

PiD is a completely deterministic, single-pass linear color-space decomposition method ($R = 1$, zero noise):
1. **Algorithmic Operation:**
   - Input: $x_i \in [0, 255]^3$.
   - Forward transformation: $x'_i = x_i \times_1 M_t$ (default is $M_{YUV}$, mapping RGB to YUV).
   - Quantization in YUV: $x'_i = \operatorname{round}(x'_i)$ (nearest integer).
   - Inverse transformation: $x'_i = x'_i \times_1 M_t^{-1}$.
   - Post-inverse RGB rounding: $x'_i \leftarrow \operatorname{round}(x'_i)$ (explicitly applied in RGB space by default).
   - Residual: $R_{\text{pid}}(x) = x_i - x'_i$ (original minus reconstructed).
2. **Corroborated Inverse Matrix Erratum:**
   - The paper's displayed forward matrix $M_{YUV}$ is the standard ITU-R BT.601 matrix:
     $$\begin{bmatrix} Y \\ U \\ V \end{bmatrix} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \\ -0.168736 & -0.331264 & 0.5 \\ 0.5 & -0.418688 & -0.081312 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix}$$
   - **Vector Conventions:** Under column-vector convention $[R,G,B]^T = M^{-1} [Y,U,V]^T$, the third row of $M^{-1}$ reconstructs the Blue channel ($B = 1.0 \cdot Y + 1.772 \cdot U + 0.0 \cdot V$). In tensor/row-vector notation ($x' = x M^T$ and $x = x' (M^{-1})^T$), the third channel $B$ is identically reconstructed by the third row of $M^{-1}$.
   - **Figure 2 Corroboration:** In Figure 2 of the main published paper, the pipeline diagram explicitly displays the mathematically correct YUV inverse matrix, with the **third row (index 2 in 0-based indexing)** showing $[1.0, 1.772, 0.0]$:
     $$M_{\text{true}}^{-1} = \begin{bmatrix} 1.0 & 0.0 & 1.402 \\ 1.0 & -0.344136 & -0.714136 \\ 1.0 & 1.772 & 0.0 \end{bmatrix}$$
   - **Appendix A Table Typo:** In Appendix A, the third row (index 2 in 0-based indexing) of $M_{\text{printed}}^{-1}$ is printed as `[0.5, -0.418688, -0.081312]`, which is an apparent typographical copy-paste duplicate of the third row of the forward matrix $M_{YUV}$.
   - **Distinction:** The published Figure 2 matrix matches the independently calculated numerical inverse of $M_{YUV}$ (third row $[1.0, 1.772, 0.0]$). Candidate experiments must use this verified inverse.
3. **Matrix Norm & Scaling Ambiguity:**
   - Appendix A states *"The l1-norm of these matrices are the same ($\|M_t\| = 1$)"*.
   - Inspection of all Appendix A matrices shows that for $M_{YUV}$, absolute row sums equal $1.0$ while column sums reach $\approx 1.337$; for $M_{XYZ}$ and $M_{\text{custom}}$, neither the maximum column sum nor the maximum row sum equals $1.0$.
   - The exact mathematical norm intended by the authors is an **unresolved ambiguity**. We do not claim that the paper's reported matrix norm is a particular induced norm.
   - The paper's matrix scaling ablation is reported in **Table 5** (Table 6 compares transformation matrices). Table 5 evaluates scaling factors $s \in \{0.2, 0.5, 0.8, 1.0, 1.5, 2.0\}$ on $M_s = s M_{\text{YUV}}$, showing $s = 1.0$ achieves $96.83\%$ mean accuracy, while $s > 1.0$ causes severe clipping degradation.
   - Retain $s = 1.0$ as the unscaled reference default, with the paper's six scaling factors $s \in \{0.2, 0.5, 0.8, 1.0, 1.5, 2.0\}$ as sensitivity controls. Output is returned as signed float32 without arbitrary clipping or silent YUV clamping.

------------------------------------------------------------------------

## 5. Candidate experiments

Keep candidates independent. Do not merge CVPR, CoDA, and PiD into one
hybrid method during initial investigation.

### Experiment A --- Existing Branch B baseline

Reproduce the current implementation in an isolated research script
without modifying production code.

Purpose:

-   Confirm current behavior.
-   Verify the numerical pilot.
-   Establish the reference residual distribution.
-   Verify deterministic seeding and runtime.

Use the repository's actual implementation details, not assumptions in
this document.

### Experiment B --- CVPR 2025 4-bit reproduction

This is the first priority.

Reproduce the authors' supplementary implementation as closely as
possible:

-   Normalize input to `[0,1]`.
-   Use 16 quantization levels.
-   Use the exact `(levels - 1)` quantizer.
-   Add Gaussian noise using the documented scikit-image procedure.
-   Quantize each noisy replica.
-   Average quantized replicas.
-   Compute the residual using the paper's sign convention.

First reproduce the supplied code behavior. Then create a deterministic,
explicitly seeded research version. Clearly document any differences
introduced to make it deterministic.

Verify the actual replica count used by the authors' code and tables. Do
not assume `R = 50` is the CVPR setting.

For controlled comparison on our pilot, include at least `R = 50` and
`R = 500`, while separately documenting the paper-reported settings. If
the paper's code or ablation requires additional values, include them
only when justified and computationally feasible.

### Experiment B1 --- Quantization and noise analysis

For 16 levels:

\[ q=`\frac{1}{15}`{=tex} \]

At normalized noise standard deviation `sigma = 0.1`:

\[ `\frac{\sigma}{q}`{=tex}=1.5 \]

For the idealized periodic quantization-bias component, the Gaussian
attenuation term is:

\[
A=`\exp`{=tex}`\left`{=tex}(-2`\pi`{=tex}^2`\left`{=tex}(`\frac{\sigma}{q}`{=tex}`\right`{=tex})^2`\right`{=tex})
\]

At `sigma/q = 1.5`, this term is approximately `5.16e-20`. This suggests
that the periodic theoretical component may be strongly suppressed at
the stated parameters.

Treat this formula as a theoretical diagnostic, not proof that the
entire residual map is zero. Finite-replica sampling variance and
clipping/saturation can still produce residuals.

Investigate normalized noise-to-step ratios:

-   `sigma/q = 0.1` ($\sigma = 0.0067$)
-   `sigma/q = 0.2` ($\sigma = 0.0133$)
-   `sigma/q = 0.3` ($\sigma = 0.0200$)
-   `sigma/q = 0.5` ($\sigma = 0.0333$)
-   `sigma/q = 1.5` ($\sigma = 0.1000$ — **paper-code default implied by scikit-image default variance**)

Note that $\exp(-2\pi^2(\sigma/q)^2)$ is an idealized periodic-component attenuation expression under unconstrained Fourier analysis; it must **not** be interpreted as total residual magnitude, total finite-replica variance, or full reconstruction error.

Record both `sigma` in normalized units and `sigma/q`; never report one without the other.

### Experiment B2 --- 2x2 Factorial clipping study

Use a rigorous $2 \times 2$ factorial design to disentangle noise-stage clipping from quantizer-output saturation:

| Configuration | Noise-Stage Clipping (`clip=True`) | Quantizer-Output Clipping (`.clamp(0, 1)`) | Description |
| :---: | :---: | :---: | :--- |
| **Config 1** | Enabled | Enabled | Paper supplementary default |
| **Config 2** | Enabled | Disabled | Diagnostic |
| **Config 3** | Disabled | Enabled | Diagnostic |
| **Config 4** | Disabled | Disabled | Pure unconstrained quantizer |

Measure residuals conditioned on original intensity. Use intensity bins
across `[0,1]` and plot:

\[ E\[\Delta \mid I \in \text{bin}\] \]

Also report conditional variance and nonzero rates. Determine whether
residual behavior is concentrated near black/white boundaries or
exhibits periodic structure throughout the intensity range.

### Experiment C --- PiD deterministic residual

Reproduce PiD independently using the exact paper method.

Purpose:

-   Test a deterministic alternative.
-   Avoid random noise and replica averaging.
-   Determine whether color-space quantization creates a useful
    complementary residual on Defactify.

Do not combine PiD with CVPR NQP in this first stage.

If the paper supports multiple quantizers, test the specified default
first and then the documented ablation. Do not create arbitrary
transformation or quantization variants.

### Experiment D --- CoDA Equation 12 diagnostic

Keep this as a paper-consistency investigation only.

Document the literal written equation, its input range, units, rounding
behavior, and relationship to Algorithm 1. Do not silently rewrite it as
a `1/256` quantizer or claim that it is the confirmed intended
implementation.

Do not prioritize it as a production candidate unless a source or author
clarification resolves the ambiguity.

------------------------------------------------------------------------

## 6. Synthetic numerical validation

Before using Defactify, design and run synthetic tests after approval.

Include:

1.  Constant black images.
2.  Constant white images.
3.  Mid-range constant images.
4.  Smooth intensity gradients.
5.  Uniform random images.
6.  Restricted color palettes.
7.  Large flat regions.
8.  Values immediately below and above quantization boundaries.
9.  Values near clipping boundaries.

Test uint8 and float32 inputs where applicable.

Verify:

-   Input range and normalization.
-   RGB/BGR channel order.
-   Quantization levels and boundaries.
-   Noise variance and standard deviation.
-   Clipping and saturation.
-   Replica averaging.
-   Residual sign.
-   Output shape and dtype.
-   Finite values.
-   Deterministic behavior.
-   Random-number generator and seed policy.
-   Expected boundary behavior.
-   Agreement with the supplied author code where reproducible.

For the CVPR method, distinguish: - Close reproduction of the supplied
stochastic code. - Deterministic seeded implementation for controlled
comparisons.

Do not claim bitwise equality when different random-number generators or
library versions are used.

------------------------------------------------------------------------

## 7. Numerical pilot

Use the existing balanced 100-image pilot:

-   20 real images.
-   16 images from each of the five generator categories.

Use identical image IDs across all applicable candidates. Record the
split and image IDs.

Compare:

-   Nonzero scalar fraction.
-   Nonzero pixel fraction.
-   Mean absolute residual.
-   Maximum absolute residual.
-   Residual variance and standard deviation.
-   Residual histograms.
-   Conditional mean/variance by original intensity.
-   Per-class statistics.
-   Per-generator statistics.
-   Spatial residual visualizations.
-   Runtime.
-   Peak memory.

Compare residuals in common units and document residual sign. Do not
compare normalized and 8-bit values without conversion.

The 100-image pilot is for numerical behavior and implementation
validation only. It is not sufficient to establish classification
generalization.

------------------------------------------------------------------------

## 8. Defactify dataset-confound audit

This audit must happen before classifier experiments.

Investigate whether real and generated classes differ systematically in
technical properties that could become shortcuts.

Inspect:

-   File formats: JPEG, PNG, WebP and others.
-   JPEG quantization/compression characteristics where recoverable.
-   Original image dimensions and aspect ratios.
-   Resizing and interpolation history where recoverable.
-   Color mode: RGB, RGBA, grayscale, palette.
-   Bit depth.
-   EXIF metadata.
-   ICC profiles.
-   Software/encoder metadata.
-   File size distributions.
-   Image-level technical statistics.
-   Duplicate and near-duplicate images.
-   Generator-specific and class-specific acquisition patterns.

Report distributions by real/generated class and by each generator.

Run a metadata/technical-properties-only baseline classifier, if
feasible, to determine whether these properties alone predict the
labels.

Design matched-subset or standardized-preprocessing controls. Ensure
every candidate uses the same input decoding and preprocessing path.

Do not assume that converting every image to RGB and resizing to a
common dimension eliminates all encoding or compression confounds.

Prevent source-image or near-duplicate leakage across train, validation,
and test splits.

------------------------------------------------------------------------

## 9. Discriminative evaluation

Only after numerical validation and confound auditing should the plan
proceed to classifier evaluation.

Compare:

  Model         Input
  ------------- ---------------------------
  Baseline      RGB image only
  Candidate 1   RGB image + CVPR residual
  Candidate 2   RGB image + PiD residual
  Diagnostic    CVPR residual only
  Diagnostic    PiD residual only

The residual-only models are diagnostics, not the main success
criterion.

Use the same:

-   Dataset split.
-   Backbone or model capacity.
-   Training budget.
-   Optimization policy.
-   Augmentation policy.
-   Evaluation protocol.

Use a CNN or another justified spatial model that processes the original
image and residual as complementary inputs. Do not rely on flattening
full-resolution residual maps into a very high-dimensional vector as the
primary evaluation.

Report:

-   Balanced accuracy.
-   Precision.
-   Recall.
-   F1.
-   ROC-AUC.
-   Confusion matrix.
-   Per-generator performance.
-   Confidence intervals or repeated-run variability.
-   Runtime and memory.
-   Results under JPEG compression.
-   Results under resizing and other realistic post-processing.

Use training data for fitting. Keep validation separate for model
selection. Keep the test split untouched until the methodology and model
are finalized.

The main question is whether adding a residual provides a reliable
improvement over RGB-only, particularly on unseen generators and under
post-processing.

A larger residual or a better residual-only score is not sufficient
evidence.

------------------------------------------------------------------------

## 10. Reproducibility requirements

Record:

-   Python version.
-   PyTorch version.
-   NumPy version.
-   OpenCV version.
-   scikit-image version.
-   CUDA availability and version.
-   GPU model.
-   Device used.
-   All experiment parameters and units.
-   Quantization level and step.
-   Noise variance and standard deviation.
-   Replica count.
-   Random-number generator.
-   Seed policy and per-image seeds.
-   Sample IDs and split membership.
-   Runtime.
-   Peak memory.
-   Raw metrics.
-   Visualizations.
-   Exact commands.
-   Any deviations from paper code.

Retain experiment scripts, configuration files, raw results, logs, and
seed records. Do not automatically delete intermediate evidence.

------------------------------------------------------------------------

## 11. Decision criteria for a Branch B replacement

A candidate may be recommended for replacement only if all of the
following are satisfied:

1.  Its implementation is faithful to a documented method or its
    deviations are explicit.
2.  Synthetic tests validate the expected numerical behavior.
3.  Its output is deterministic under the defined seed policy, or its
    stochastic variability is quantified.
4.  It is not relying primarily on dataset encoding or metadata
    shortcuts.
5.  It improves the RGB-only baseline under a fair evaluation.
6.  Its benefit is not confined to one generator.
7.  Its benefit remains meaningful under JPEG compression and resizing.
8.  Runtime and memory are acceptable for the project.
9.  It can preserve the existing Branch B external input/output
    contract.

If no candidate meets these conditions, retain the current Branch B code
and gate. Do not force a replacement.

------------------------------------------------------------------------

## 12. Sources

-   Jia et al., CVPR 2025, *Secret Lies in Color*:\
    https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Secret_Lies_in_Color_Enhancing_AI-Generated_Images_Detection_with_Color_CVPR_2025_paper.pdf
-   CVPR 2025 supplementary material:\
    https://openaccess.thecvf.com/content/CVPR2025/supplemental/Jia_Secret_Lies_in_CVPR_2025_supplemental.pdf
-   Jia et al., CoDA 2026:\
    https://arxiv.org/abs/2605.24306
-   Fu et al., PiD, ICML 2025:\
    https://proceedings.mlr.press/v267/fu25i.html
-   scikit-image `random_noise` documentation:\
    https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise

------------------------------------------------------------------------

## 13. Current authorization and stopping condition

This document is a research specification. It does not authorize
implementation or production changes.

Before any experiment is executed, the user must approve a separate
implementation/execution plan.

Until then:

-   Do not modify production source code.
-   Do not modify Branch A, Branch C, or integration.
-   Do not modify production Branch B.
-   Do not modify tests.
-   Do not modify datasets.
-   Do not execute pilots or classifiers.
-   Do not bypass the materialization gate.
-   Do not reinstall packages or rebuild the environment.

The first Antigravity task using this document should be a repository
inspection and plan-preparation task only. It must report findings and
wait for approval.
