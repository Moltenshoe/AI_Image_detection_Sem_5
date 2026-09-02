# AI Image Forensic Analysis — v3 (Analysis Types & Interpretation Guide)

Dataset: 8 matched real/AI (img2img) pairs, ~539 features per image, paired stats + BH-FDR correction.
Below: each analysis, what it computes, and how to read the metric (high vs low → what it suggests).

---

## 1. FFT / Fourier Frequency Analysis
**What it does:** Splits the image into spatial frequencies (low = smooth areas, high = fine detail/noise). Measures how power is distributed across radial frequency bands.

**Reading the metrics:**
- `high_freq_ratio` **high** → lots of fine detail/texture/noise retained. **Low** → image over-smoothed, common in some AI/denoised pipelines.
- `low_freq_ratio` **high** → dominated by broad shapes/flat regions, little detail.
- `spectral_centroid` **high** → energy skewed toward fine detail; **low** → energy concentrated in smooth/coarse structure.
- Not among the top statistically significant features in this run — treat as supporting evidence, not a standalone signal.

---

## 2. DWT — 3-Level Haar Wavelet Analysis
**What it does:** Multi-scale decomposition into an approximation (LL) and directional detail bands (LH, HL, HH) at 3 zoom levels.

**Reading the metrics:**
- Detail-band **entropy** (e.g. `HH2_entropy`, `HL1_entropy`, `LH2_entropy`) **low** → coefficients are less varied/predictable — fine texture is more "flattened," a pattern this dataset found consistently in AI images (dz around -1.8 to -2.3, 8/8 pairs).
- Detail-band **entropy high** → richer, more chaotic fine-scale texture, more typical of real camera noise/grain here.
- `detail_energy_ratio` **high** → more of the image's total energy sits in fine detail vs the smooth base; **low** → most energy sits in the coarse/blurry LL band.

**This was one of the strongest and most consistent findings in the notebook.**

---

## 3. Color-Quaternion Wavelet Analysis (QWT-inspired, v3 addition)
**What it does:** Runs Haar wavelet on R, G, B separately, then bundles matching R/G/B coefficients into one quaternion value per pixel so channel relationships can be measured together. Not a reproduction of the published QWT/CQWT algorithms — treat as an exploratory, custom method.

**Reading the metrics:**
- `qwt_*_magnitude_entropy` **low** → color-wavelet coefficients are less diverse/predictable at that band. This was the single strongest effect in the whole notebook (`qwt_HL2_magnitude_entropy`, dz = -3.06, 8/8 pairs lower in AI).
- `qwt_*_phase_rg/rb/gb_circular_coherence` **high** → R, G, B channel edges/phases are moving together (aligned); **low** → channels behave more independently. AI images showed **higher** coherence consistently (dz ≈ +2.0 to +2.1) — channels more "in sync" than real photos.
- `qwt_*_phase_*_std` **low** → phase angles are tightly clustered (matches high coherence above); **high** → phases scattered/independent.
- `qwt_*_quaternion_skewness` / `kurtosis` — shape of the magnitude distribution; large deviations from ~0 skew or ~3 kurtosis suggest a non-natural, lopsided or spiky coefficient distribution.

---

## 4. 8×8 Block DCT Analysis
**What it does:** JPEG-style block transform. Measures how energy is split between the DC (average brightness) term and higher-frequency block coefficients.

**Reading the metrics:**
- `dct_dc_ratio_mean` **high** → blocks are mostly flat/uniform, little internal structure. **Low** → blocks carry more internal detail.
- `dct_highfreq_ratio_mean` **high** → strong fine-grained block texture (real photo noise, compression artifacts). **Low** → smoother blocks, consistent with softened/AI-processed texture.
- Not a top-ranked significant feature here, but useful as a cross-check against the wavelet findings.

---

## 5. Gaussian High-Pass Residual Analysis
**What it does:** Blurs the image with a 5×5 Gaussian kernel (sigma = 1.0), subtracts the blur from the original. What's left is the fine texture/noise the blur removed.

**Reading the metrics:**
- `hp_std` **high** → lots of fine-grained residual energy (busy texture/noise). **Low** → image was already smooth, little left after blur removal.
- `hp_kurtosis` **high** → residual is mostly near-zero (flat) with a few sharp outlier spikes — a "flat plus occasional artifact" pattern. This was noticeably **higher in AI images** (Δ ≈ +9.8 on average). **Low/near-zero** kurtosis → residual spread more evenly, closer to natural Gaussian sensor noise, as typically seen in real photos.

---

## 6. 2×2 Cross-Difference Analysis
**What it does:** For every 2×2 pixel block: (top-left + bottom-right) − (top-right + bottom-left), then absolute value. A stride-1 filter; output shape is `[H-1, W-1]`.

**Reading the metrics:**
- `crossdiff_std` / `crossdiff_absmean` **high** → strong local diagonal asymmetry, i.e. checkerboard/grid-style artifacts present. **Low** → pixels vary smoothly with no grid artifact.
- This is a detector for pixel-grid artifacts left by upsampling/deconvolution steps common in AI image pipelines — a spike here is a flag, not a diagnosis on its own.

---

## 7. Synthbuster-Inspired Periodic Residual-Frequency Analysis
**What it does:** Takes the cross-difference residual per color channel, FFTs it, and checks the exact frequency points matching repeat-periods of 2, 4, and 8 pixels (horizontal, vertical, diagonal). Targets the periodic "combs" that many AI upsampling/decoding steps leave behind.

**Reading the metrics:**
- `synth_r/g/b_p2_x_mean` (and p4, p8 variants) **high** → a repeating pattern at that exact pixel spacing is present — a classic AI-generator/upscaler fingerprint. **Low/near baseline** → no such periodic artifact detected.
- In this dataset, the **period-2** peaks were consistently and significantly **higher in AI images** across all 3 color channels (dz ≈ +1.8 to +2.0, 8/8 pairs) — the clearest "generation artifact" signal found.
- `synth_*_fft_highfreq_ratio` **high** → residual itself carries more high-frequency energy overall.

---

## 8. Sobel Gradient, Laplacian, Canny & Contour Analysis
**What it does:** Standard edge-detection family. Sobel = edge strength/direction. Laplacian = sharpness/curvature of intensity change. Canny = binary edge map. Contours = traced outlines from Canny, summarizing shape complexity.

**Reading the metrics:**
- `gradient_mean` / `laplacian_absmean` **high** → strong, sharp edges throughout the image. **Low** → softer, more gradual transitions (over-smoothing).
- `edge_density` **high** → many detected edges (busy/detailed scene or noisy edge detection). **Low** → few edges, flatter image.
- `contour_perimeter_per_area` **high** → complex, jagged outlines. **Low** → simpler, smoother shapes.
- In this dataset these leaned slightly lower in AI images but were not among the strongest/most significant results — best used as supporting, not primary, evidence.

---

## 9. LBP — Local Binary Pattern Texture Analysis
**What it does:** For each pixel, compares its 8 neighbors (brighter/darker → 1/0) into an 8-bit code describing local micro-texture. Codes with ≤2 bit-transitions ("uniform" patterns like edges/corners) are kept as counts 0–8; noisier/busier codes are grouped into one bucket. Result: a 10-bin texture histogram per image.

**Reading the metrics:**
- `lbp_mean_code` **high** (shifted toward more edge/corner-like uniform codes) → texture skews toward simpler, more structured micro-patterns. This was **significantly higher in AI images** here (dz ≈ +2.26, 8/8 pairs).
- `lbp_hist_entropy` **high** → micro-texture patterns are diverse/spread across many codes (complex texture). **Low** → texture is dominated by a few pattern types (more uniform/simplified texture).
- `lbp_dominant_bin` **high** → one texture pattern dominates the whole image (low variety).

---

## 10. GLCM — Gray-Level Co-occurrence Matrix Texture Analysis
**What it does:** Quantizes the image to 32 gray levels, then builds a co-occurrence matrix counting how often pairs of gray levels appear next to each other, at distances 1/2/4 px and angles 0°/45°/90°/135° (12 matrices, averaged). From this, computes standard Haralick texture properties.

**Reading the metrics:**
- `glcm_contrast_mean` / `glcm_dissimilarity_mean` **high** → neighboring pixels differ a lot (rough, high-variation texture). **Low** → neighboring pixels are very similar (smooth, low-variation texture). This was **lower in AI images** here — smoother local texture than real photos.
- `glcm_homogeneity_mean` **high** → texture is uniform/self-similar. **Low** → texture is varied/rough. This was **higher in AI images** here — consistent with the dissimilarity drop above.
- `glcm_energy_mean` / `ASM` **high** → texture is repetitive/orderly (few dominant gray-level pairs). **Low** → texture is more random/varied.
- `glcm_correlation_mean` **high** → pixel values are linearly predictable from neighbors. **Low** → less predictable, more locally random.
- `glcm_entropy_mean` **high** → co-occurrence pattern itself is disordered/random. **Low** → pattern is more concentrated/predictable.

---

## 11. Intensity, RGB, HSV & Entropy Analysis
**What it does:** Basic distributional statistics — brightness mean/spread, correlation between R/G/B channels, hue/saturation stats, and pixel-value entropy (overall randomness).

**Reading the metrics:**
- `intensity_entropy` **high** → wide, varied use of brightness values (natural, high dynamic range). **Low** → values cluster narrowly (flatter, less varied tonal range).
- `rgb_corr_rg` / `rgb_corr_rb` / `rgb_corr_gb` **high** (close to 1) → channels track each other closely (e.g. desaturated/monochrome-leaning content, or channels generated jointly). **Low/near 0** → channels vary more independently, typical of natural sensor color response.
- These were not top significant features individually here, but channel-correlation direction supports the quaternion-coherence findings above (channels moving together more in AI).

---

## 12. Error Level Analysis (ELA)
**What it does:** Re-saves the image as JPEG at a fixed quality, then diffs it against the original. Regions that behave inconsistently under re-compression often indicate edits, splices, or different processing history.

**Reading the metrics:**
- `ela_mean` **high** (and uneven spatially) → strong recompression error, potentially indicating spliced/edited content or a different original compression history. **Low and uniform** → the image responds consistently to recompression, typical of a single-source, unedited image.
- ELA is a classic **splice/edit** detector, less directly meaningful for a fully AI-generated (not spliced) image — included here mainly as a standard forensic baseline, not a top finding in this run.

---

## 13. Paired Real-vs-AI Feature Differences
**What it does:** For each of the 8 matched pairs, computes AI value − Real value (and relative % change) for every feature above, isolating the effect of the AI transformation from differences in scene content.

**Reading the metrics:**
- A **large, consistent** delta across all 8 pairs (same sign every time) is a strong exploratory signal.
- A delta that flips sign across pairs means the effect is scene-dependent, not a reliable AI signature.

---

## 14. Cohen's Paired Effect Size (dz)
**What it does:** Standardizes the paired mean difference by its spread, giving a comparable "how big and how consistent" score per feature.

**Reading the metrics:**
- `|dz|` **> 2.0** → very large, highly consistent effect (as seen in the top wavelet/QWT/LBP features here, up to dz = -3.06).
- `|dz|` **0.5–2.0** → moderate effect, worth further validation.
- `|dz|` **< 0.5** → weak/negligible effect, likely noise at this sample size.

---

## 15. Exact Paired Sign-Flip Test + Benjamini–Hochberg FDR Correction
**What it does:** With only 8 pairs, tests every possible +/- sign combination of the observed differences to get an exact p-value (no large-sample assumptions). BH-FDR then corrects p-values across all ~539 tested features to control false positives from testing so many features at once.

**Reading the metrics:**
- `exact_signflip_p` **= 0.0078** → the smallest possible p-value with n=8 (the effect went the same direction in all 8/8 pairs). This is the ceiling of what 8 pairs can prove — it means "100% consistent here," not "proven in general."
- `q_fdr` **low (surviving correction)** → the feature remains notable even after correcting for testing hundreds of features at once.
- **Caveat:** significance here reflects small-sample consistency, not generalizability. All top findings in this notebook sit at this p=0.0078 ceiling.

---

## Summary — Strongest Convergent Findings

| Analysis | Metric | Direction in AI | Effect (dz) |
|---|---|---|---|
| Quaternion Wavelet | `qwt_HL2_magnitude_entropy` | lower | -3.06 |
| DWT | `HH2_entropy`, `HL2_entropy`, `LH2_entropy` | lower | -2.3 to -2.4 |
| LBP | `lbp_mean_code` | higher | +2.26 |
| Quaternion Wavelet | phase circular coherence (HH1/HL1) | higher | +2.0 to +2.1 |
| Synthbuster | period-2 peaks (r/g/b) | higher | +1.8 to +2.0 |
| GLCM | dissimilarity ↓ / homogeneity ↑ | smoother | moderate |

**Bottom line:** AI images in this dataset showed less fine-scale randomness (wavelet/LBP/GLCM agree), unusually synchronized color channels at fine detail (quaternion phase coherence), and a faint repeating 2-pixel artifact (Synthbuster) — all consistent 8/8 pairs. With only 8 pairs this is a **hypothesis-generating result**, not a validated AI detector.