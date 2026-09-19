# Version History / Changelog

## Project: AI Image Forensics

This file records the major methodological changes made to the image-forensics notebook.

The versions are intended to document **what changed in the analysis pipeline**, why it changed, and which numerical results belong to which implementation.

---

# v3 — Color Quaternion Wavelet Extension

## Status

Current research version.

## Main change

v3 keeps the refined PyTorch/TorchVision/Kornia pipeline from v2 and adds an exploratory **color-quaternion Haar wavelet analysis**.

This is the largest conceptual addition to the project so far.

## Added

### Color-quaternion wavelet features

RGB coefficients are kept together and represented as:

`q = R i + G j + B k`

The module extracts:

- quaternion magnitude
- magnitude mean
- magnitude standard deviation
- magnitude energy
- magnitude entropy
- RGB pairwise angular coordinates
- circular coherence
- angular standard deviation
- magnitude skewness
- magnitude kurtosis
- multiscale detail-energy summaries

The module produces **145 scalar QWT-inspired features**.

### QWT/CQWT motivation

The new analysis is motivated by published quaternion-wavelet forensic research.

The notebook explicitly distinguishes the implementation from a canonical analytic QWT/CQWT:

- it uses a tensor Haar decomposition;
- corresponding RGB coefficients are combined into a pure quaternion;
- the angular quantities are pairwise phase proxies;
- it is not claimed to reproduce the exact published algorithms.

This distinction is intentional to avoid overclaiming.

### QWT visualizations

Added visual comparison of:

- quaternion magnitude
- RG phase proxy
- RB phase proxy
- GB phase proxy

for wavelet detail coefficients.

---

## v3 retained

The following v2 analyses remain:

- FFT
- 3-level Haar DWT
- 8 × 8 block DCT
- Gaussian high-pass residual
- cross-difference residual
- Synthbuster-inspired periodic FFT features
- Sobel gradient
- Laplacian
- Canny
- contours
- LBP
- GLCM
- intensity statistics
- RGB statistics
- HSV statistics
- entropy
- ELA
- paired differences
- Cohen's paired effect size
- exact sign-flip testing
- Benjamini–Hochberg FDR
- paired plots
- relative-change heatmap
- final effect-size ranking

---

## v3 results

The final feature table contains **539 scalar features per image**.

The paired-difference table contains **1077 columns**.

The strongest final feature is:

`qwt_HL2_magnitude_entropy`

with:

- Mean AI − Real: **-0.868330**
- Cohen's dz: **-3.063366**
- 8/8 pairs in the same direction
- exact sign-flip p: **0.007812**
- FDR q: **0.02532**

Other important new QWT observations include:

- `qwt_LH2_magnitude_entropy`: dz **-2.454539**
- `qwt_HH1_phase_gb_circular_coherence`: dz **+2.115547**
- `qwt_HH1_phase_rb_circular_coherence`: dz **+2.107296**
- `qwt_HH1_phase_rg_circular_coherence`: dz **+2.096064**

These results are exploratory because the dataset contains only 8 matched pairs.

---

# v2 — Clean PyTorch / Tensor-Native Pipeline

## Main goal

v2 refactored the original analysis so that most image processing is performed with tensors and can remain on the GPU.

The goal was to make the pipeline:

- cleaner
- easier to inspect
- more consistent
- GPU-friendly
- less dependent on unnecessary NumPy conversions

## Major changes

### Image loading

Changed from PIL-based loading to TorchVision:

- `torchvision.io.decode_image`
- EXIF orientation handling
- TorchVision v2 transforms

The analysis preprocessing changed from directly stretching images to 512 × 512 to:

1. aspect-preserving resize;
2. center crop to 512 × 512;
3. conversion to float32 `[0,1]`.

This means v2/v3 numerical results are **not directly interchangeable with the original notebook's results**.

---

### FFT

Changed to PyTorch FFT:

- `torch.fft.fft2`
- `torch.fft.fftshift`
- `torch.fft.fftfreq`

The radial frequency calculation was also changed.

Therefore FFT numerical values from v1 and v2/v3 should not be mixed.

---

### DWT

Changed from PyWavelets to `ptwt`.

The configuration remained:

- Haar
- level 3
- symmetric boundary handling

The resulting DWT measurements remained very close to the original implementation.

---

### DCT

Changed to `torch-dct`.

More importantly, the original DCT implementation contained a methodological problem:

> the block mean was subtracted before applying DCT.

That effectively suppressed the DC coefficient.

v2 fixes this by applying the DCT directly to the block.

The new DCT implementation also exposes DC-energy statistics.

---

### Residual analysis

The cross-difference implementation was changed to use the **absolute value** of the cross-difference.

v2 also added the Synthbuster-inspired periodic FFT analysis.

---

### Synthbuster-inspired features

Added residual Fourier measurements at periods:

- 2
- 4
- 8

for:

- R
- G
- B

and directions:

- x
- y
- diagonal

This is an exploratory implementation inspired by diffusion-image forensic work.

It is not a full reproduction of Synthbuster.

---

### Edge analysis

Moved major edge operations to Kornia:

- Sobel
- Laplacian
- Canny

OpenCV remains for contour extraction.

The raw numerical scale of some gradient features changed, although the direction of several paired effects remained similar.

---

### LBP

The original scikit-image LBP implementation was replaced by a custom PyTorch implementation.

The implementation currently supports:

- P = 8
- R = 1
- uniform LBP

Because the implementation changed, the exact LBP values from v1 and v2/v3 should not be treated as identical measurements.

---

### GLCM

GLCM remained based on scikit-image.

Its configuration was kept consistent, so GLCM results remained highly comparable.

---

### ELA

ELA was moved to operate on the **original file before the 512 × 512 analysis resize**.

This avoids making the ELA measurement dependent on the analysis resize.

---

### Statistical analysis

The paired statistical framework was retained and expanded to the larger feature set.

The notebook uses:

- paired AI − Real differences
- Cohen's dz
- exact sign-flip test
- Benjamini–Hochberg FDR

Because v2 added many features, the FDR correction is applied across the expanded hypothesis set.

---

# v1 — Original Forensic Analysis

## Main goal

The original notebook established the initial multi-method forensic analysis of the 8 real/AI image pairs.

## Analyses included

- FFT
- DWT
- DCT
- residual analysis
- cross-difference
- edge/gradient analysis
- contours
- LBP
- GLCM
- intensity/colour analysis
- ELA
- paired feature differences
- effect sizes
- exact sign-flip testing
- FDR correction

## Original implementation characteristics

### Image loading

Images were loaded using PIL and resized directly to:

**512 × 512**

This could stretch images when their aspect ratio was not 1:1.

### FFT

Used SciPy FFT with a pixel-coordinate radial frequency calculation.

### DWT

Used PyWavelets.

### DCT

Used SciPy DCT in 8 × 8 blocks.

The original implementation subtracted the block mean before the DCT, which suppressed the DC component.

### Residual

Used OpenCV Gaussian blur and a signed cross-difference.

### Edges

Used OpenCV Sobel, Laplacian, Canny and contour analysis.

### LBP

Used scikit-image LBP.

### GLCM

Used scikit-image GLCM.

### ELA

ELA was performed after the analysis image had already been resized.

---

# Important Versioning Rule

## Do not mix numerical results between versions

v1, v2 and v3 are **not numerically identical pipelines**.

Some changes affect only implementation efficiency.

Others change the actual mathematical input or feature definition.

The most important changes affecting numerical comparability are:

| Analysis | v1 | v2/v3 |
|---|---|---|
| Image preprocessing | Direct 512×512 stretch | Aspect-preserving resize + center crop |
| FFT | SciPy + pixel-coordinate radial bins | PyTorch FFT + `fftfreq` |
| DWT | PyWavelets | `ptwt` |
| DCT | Mean-subtracted blocks | Correct DC-preserving block DCT |
| Cross-difference | Signed | Absolute |
| Synthbuster features | Not present | Added |
| Edge operations | OpenCV | Kornia + OpenCV contours |
| LBP | scikit-image | Custom PyTorch |
| GLCM | scikit-image | Same basic implementation |
| ELA | After resize | Original file before resize |
| QWT | Not present | Added in v3 |

Therefore:

> A feature's numerical value in v1 should not be copied into a v2/v3 report unless the v2/v3 notebook is rerun and that feature is verified under the new pipeline.

---

# Version Relationship

```text
v1
│
│  Initial multi-method forensic analysis
│
▼
v2
│
├── PyTorch/TorchVision tensor pipeline
├── Kornia computer vision
├── ptwt wavelets
├── torch-dct
├── corrected DCT handling
├── revised preprocessing
├── revised residual handling
├── Synthbuster-inspired periodic features
└── GPU-friendly extraction
│
▼
v3
│
└── Color-quaternion wavelet analysis
    ├── quaternion magnitude
    ├── magnitude entropy
    ├── colour phase proxies
    ├── circular coherence
    ├── quaternion skewness
    └── quaternion kurtosis
```

---

# Current Version: v3

The current notebook should be described as:

> **AI Image Forensics — Clean PyTorch Pipeline (v3)**

Its purpose is:

> **feature analysis and forensic hypothesis generation, not a finished AI-image detector.**

The next major version should ideally focus on **validation and generalization**, rather than continually adding features to the same 8-image dataset.
