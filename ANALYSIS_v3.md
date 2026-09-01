# AI Image Forensic Analysis — v3 Concepts and Results

## 1. Overview

This study compares **8 real photographs** with their corresponding **AI-generated img2img versions**.

The purpose of the experiment is not to claim that a single mathematical operation can identify an AI image. Instead, the notebook investigates whether an AI/img2img transformation produces measurable changes in different representations of the same image.

The v3 notebook extends the earlier analysis with a **color-aware quaternion wavelet analysis** while keeping the existing spatial, texture, frequency, residual and compression analyses.

The analyses performed are:

- FFT / Fourier frequency analysis
- 3-level Haar DWT
- 8×8 block DCT
- Gaussian high-pass residual analysis
- 2×2 cross-difference analysis
- Synthbuster-inspired periodic residual-frequency features
- Sobel gradient analysis
- Laplacian analysis
- Canny edge analysis
- contour analysis
- LBP local-texture analysis
- GLCM texture statistics
- intensity and entropy analysis
- RGB colour statistics
- HSV statistics
- Error Level Analysis (ELA)
- **color-quaternion Haar wavelet analysis (QWT-inspired)**
- paired real-vs-AI feature differences
- Cohen's paired effect size
- exact paired sign-flip testing
- Benjamini–Hochberg false-discovery-rate correction

The v3 pipeline extracted **539 scalar features per image**, including the new quaternion-wavelet features. The paired-difference table therefore contains the real value, AI value, absolute difference and relative difference for the extracted features.

The results should be interpreted as observations from a **small controlled experiment**, not as universal rules for AI-generated images.

---

## 2. Dataset and Experimental Design

### Dataset

The experiment contains:

- **8 real images**
- **8 corresponding AI/img2img images**
- **8 matched real-AI pairs**

The matching is performed using filename stems. A real image is therefore compared only with its corresponding AI version.

This paired design is important because the images contain very different subjects. For example, a forest image naturally has very different texture and frequency statistics from a photograph of fabric or text.

Instead of asking:

> "Are AI images different from real images on average?"

the experiment asks:

> "For the same original photograph, how did the measured features change after the img2img transformation?"

This substantially reduces the effect of differences in image content.

---

## 3. Image Preprocessing

The v3 notebook uses a tensor-native PyTorch/TorchVision pipeline.

Images are:

1. decoded directly into RGB tensors;
2. EXIF orientation is applied during decoding;
3. resized while preserving aspect ratio;
4. center-cropped to **512 × 512**;
5. converted to float32 in the range **[0, 1]**.

The resulting analysis tensor has the form:

`[1, 3, 512, 512]`

and the executed notebook confirmed:

- PyTorch: **2.13.0+cu130**
- TorchVision: **0.28.0+cu130**
- Kornia: **0.8.3**
- device: **CUDA**
- GPU: **NVIDIA GeForce RTX 3050 A Laptop GPU**

This version keeps most tensor operations on the GPU until scalar measurements are required.

Specialized operations remain on CPU where appropriate, particularly GLCM, contours and statistical tests.

---

# 4. FFT / Fourier Frequency Analysis

## Concept

An image is normally represented as pixels.

FFT provides a different representation: it describes the image in terms of **spatial frequencies**.

A useful intuitive interpretation is:

- **low frequencies** → broad shapes, smooth illumination and slowly changing structures;
- **middle frequencies** → ordinary edges and medium-scale structures;
- **high frequencies** → fine texture, sharp transitions and very small details.

The notebook first subtracts the image mean so that the DC component does not dominate the analysis.

It then calculates:

- FFT magnitude;
- logarithmic FFT magnitude map;
- FFT phase map;
- radial frequency power distribution;
- low-frequency power ratio;
- mid-frequency power ratio;
- high-frequency power ratio;
- spectral centroid.

## What the result means

The radial FFT representation provides a compact way of asking whether the AI transformation changed where image energy is concentrated in frequency space.

The notebook produces both:

- a visual frequency representation, and
- numerical frequency features that can be compared pair-by-pair.

## Interpretation

The FFT results show that the img2img transformation changes the frequency structure of the photographs.

However, the ordinary FFT is not sufficient to say:

> "This frequency pattern proves that an image is AI-generated."

Natural photographs can have very different frequency distributions depending on their subject, camera, focus, lighting and texture.

Therefore FFT is best treated as a **supporting forensic representation**.

---

# 5. DWT — Three-Level Haar Wavelet Analysis

## Concept

The Discrete Wavelet Transform is particularly useful because it separates image information by both **scale** and **orientation**.

The v3 notebook performs a **3-level Haar DWT**.

Each level contains:

- **LL** — approximation / low-frequency information;
- **LH** — one directional detail component;
- **HL** — the other directional detail component;
- **HH** — diagonal/high-detail information.

Unlike an ordinary global FFT, wavelets preserve spatial-scale information.

In simple terms:

> FFT tells us how much information exists at different frequencies, while DWT also tells us how that information is distributed across different scales and directional detail bands.

## Features measured

For the wavelet bands the notebook measures:

- energy;
- absolute mean;
- entropy;
- approximation energy and entropy;
- total detail-energy ratio.

## Results

The ordinary DWT produced some of the strongest and most consistent differences in the experiment.

Important features include:

| Feature | Mean AI − Real | Cohen's dz | Direction |
|---|---:|---:|---|
| LH2 entropy | -0.961482 | -2.364 | lower in AI |
| HL2 entropy | -0.897274 | -2.303 | lower in AI |
| HH2 entropy | -1.313646 | -2.300 | lower in AI |
| HL1 entropy | -1.135556 | -1.880 | lower in AI |
| LH1 entropy | -1.199208 | -1.757 | lower in AI |
| LH3 entropy | -0.386578 | -1.820 | lower in AI |

These changes occurred in the same direction for all 8 matched pairs for the listed features.

## Interpretation

The repeated decrease in wavelet-detail entropy is important.

Entropy here describes how broadly the wavelet coefficients are distributed.

A lower value therefore suggests that the AI versions have a **less diverse distribution of coefficients in these fine-scale detail bands**.

This does not mean that AI images universally have lower wavelet entropy. It means that, in this controlled img2img experiment, the transformation consistently changed the fine-scale coefficient distribution.

This is one of the strongest conventional analyses in the notebook.

---

# 6. Color-Quaternion Wavelet Analysis — v3 Addition

## Why this analysis was added

The previous DWT operates on a grayscale image.

That means the relationship between the R, G and B channels is largely removed before the wavelet analysis.

The v3 notebook adds a **color-aware quaternion-valued wavelet analysis** so that the three color channels can be represented together.

The motivation comes from the QWT/CQWT forensic literature, where quaternion wavelet representations have been investigated for distinguishing photographic images from computer-generated images.

The v3 notebook deliberately documents an important limitation:

> The implemented method is a quaternion-valued Haar-wavelet analysis inspired by QWT/CQWT, but it is **not claimed to be an exact reproduction of the published 2017/2019 QWT/CQWT algorithms**.

Published QWT definitions vary, and a true analytic QWT uses a more specific construction than simply attaching R, G and B coefficients to quaternion units.

Therefore the correct description of this notebook is:

**color-quaternion Haar wavelet analysis / QWT-inspired exploratory analysis**

rather than:

**exact implementation of the published QWT detector**.

---

## 6.1 How the v3 Quaternion Wavelet Works

For every wavelet subband, the notebook performs the same Haar decomposition independently on:

- R;
- G;
- B.

Corresponding coefficients are then grouped as a pure quaternion:

\[
q = R i + G j + B k
\]

The quaternion magnitude is:

\[
|q| = \sqrt{R^2 + G^2 + B^2}
\]

This gives a color-aware measure of wavelet coefficient strength.

The notebook then calculates:

- quaternion magnitude mean;
- quaternion magnitude standard deviation;
- quaternion magnitude energy;
- quaternion magnitude entropy;
- three pairwise angular/phase-proxy measures;
- circular coherence of those phase proxies;
- phase-proxy standard deviations;
- magnitude-based skewness;
- magnitude-based kurtosis;
- multiscale detail magnitude-energy ratio.

The v3 module produces **145 quaternion-wavelet features**.

---

# 7. Quaternion Phase Proxies

The notebook also calculates three pairwise angular quantities:

\[
\phi_{RG} = \operatorname{atan2}(G,R)
\]

\[
\phi_{RB} = \operatorname{atan2}(B,R)
\]

\[
\phi_{GB} = \operatorname{atan2}(B,G)
\]

These are explicitly called **phase proxies** in the notebook.

They should not be confused with the canonical three phase angles of a full analytic quaternion wavelet transform.

The notebook summarizes the angular distribution using circular resultant concentration:

\[
C =
\sqrt{
(\operatorname{mean}(\cos\phi))^2+
(\operatorname{mean}(\sin\phi))^2
}
\]

where:

- values near **1** indicate strong angular concentration;
- values near **0** indicate a more dispersed angular distribution.

This provides an exploratory measurement of how the relationships between color-channel wavelet coefficients are distributed.

---

# 8. Quaternion Wavelet Sanity-Check Result

For the first example pair, the notebook produced 145 quaternion-wavelet features.

Some example values were:

| Feature | Real | AI |
|---|---:|---:|
| LL1 magnitude mean | 1.489182 | 1.489436 |
| LL1 magnitude std | 0.757942 | 0.690731 |
| LL1 magnitude energy | 2.792130 | 2.695521 |
| LL1 magnitude entropy | 7.727242 | 7.241254 |
| LL1 RG circular coherence | 0.982918 | 0.985936 |
| LL1 RB circular coherence | 0.960515 | 0.970835 |
| LL1 GB circular coherence | 0.988198 | 0.991815 |
| LL1 RG phase std | 0.185528 | 0.168157 |
| LL1 RB phase std | 0.283010 | 0.242759 |
| LL1 GB phase std | 0.154084 | 0.128225 |
| LL1 quaternion skewness | -0.058461 | 0.040880 |
| LL1 quaternion kurtosis | 1.824152 | 2.148906 |

These values show that the quaternion representation is capable of detecting changes in both:

- the strength/distribution of color-wavelet coefficients;
- the relationships between the color components.

---

# 9. Quaternion Wavelet Results

The most interesting result in v3 is that several quaternion-wavelet features moved into the strongest effect sizes in the entire experiment.

The strongest feature was:

### qwt_HL2_magnitude_entropy

- mean AI − Real: **-0.868330**
- median change: **-0.879178**
- Cohen's dz: **-3.063366**
- exact sign-flip p: **0.007812**
- FDR q: **0.02532**

The second strongest was:

### qwt_LH2_magnitude_entropy

- mean AI − Real: **-0.748655**
- median change: **-0.669618**
- Cohen's dz: **-2.454539**
- exact sign-flip p: **0.007812**
- FDR q: **0.02532**

Another important result was:

### qwt_HH2_magnitude_entropy

- mean AI − Real: **-1.121478**
- median change: **-1.165889**
- Cohen's dz: **-1.915381**
- exact sign-flip p: **0.007812**
- FDR q: **0.02532**

These results are particularly interesting because the ordinary grayscale DWT also showed strong decreases in the corresponding fine-detail entropy measurements.

That agreement suggests that the observed wavelet effect is not restricted to a single grayscale measurement.

---

# 10. Quaternion Phase-Coherence Results

Several phase-proxy coherence features also showed large paired effects.

Important examples were:

| Feature | Mean AI − Real | Cohen's dz | Consistency |
|---|---:|---:|---:|
| qwt_HH1 RG circular coherence | +0.440841 | +2.096 | 8/8 |
| qwt_HH1 RB circular coherence | +0.447285 | +2.107 | 8/8 |
| qwt_HH1 GB circular coherence | +0.443342 | +2.116 | 8/8 |
| qwt_HL1 RG circular coherence | +0.418054 | +2.049 | 8/8 |
| qwt_HL1 RB circular coherence | +0.418595 | +2.021 | 8/8 |
| qwt_HL1 GB circular coherence | +0.422071 | +2.071 | 8/8 |

The corresponding phase-proxy standard deviations decreased.

For example:

- HH1 RG phase std: **-0.489380**
- HH1 RB phase std: **-0.501988**
- HH1 GB phase std: **-0.494866**

## Interpretation

The result suggests that the AI images produced a more concentrated distribution of these RGB wavelet coefficient angle proxies in this dataset.

That is an interesting observation because it introduces a **color relationship** that ordinary grayscale DWT cannot capture.

However, this is the part of the analysis that requires the greatest caution.

The current phase measurements are **exploratory pairwise angle proxies**, not the canonical phase variables of an analytic QWT.

Therefore the scientifically correct statement is:

> The v3 experiment found strong and consistent changes in color-wavelet angular relationships using its quaternion phase-proxy features.

It should **not** yet be stated as:

> "Chromatic phase is proven to be an AI fingerprint."

A larger dataset and a more faithful analytic QWT/CQWT implementation would be required before making that claim.

---

# 11. DCT — 8×8 Block Discrete Cosine Analysis

## Concept

The Discrete Cosine Transform represents local image regions using cosine-frequency components.

The notebook uses non-overlapping **8×8 blocks**, similar in spirit to the block representation used in JPEG compression.

This makes DCT useful for studying how image information is distributed between:

- low-frequency smooth components;
- high-frequency local detail components.

## Features

The notebook measures:

- DC-energy ratio;
- high-frequency energy ratio;
- mean absolute DCT coefficient;
- mean DCT coefficient map.

An important improvement in v3 is that the DCT is applied to the actual image blocks without the earlier block-mean subtraction that had effectively removed the DC component.

## Result

The mean absolute DCT coefficient decreased:

- mean AI − Real: **-0.007226**
- Cohen's dz: **-1.499750**
- direction: AI lower than real for all 8 pairs

The high-frequency DCT representation also continues to support the broader observation that the img2img transformation changes fine-scale image information.

## Interpretation

DCT supports the wavelet findings:

> the AI transformation changed the distribution of local frequency information.

DCT is nevertheless a supporting feature family rather than a standalone AI detector.

---

# 12. Residual / High-Pass Analysis

## Concept

A residual is the information remaining after suppressing part of the normal image content.

The notebook uses a Gaussian blur and subtracts it from the original grayscale image:

\[
R = I - G_\sigma(I)
\]

This suppresses broad structures and emphasizes local variations.

The notebook measures:

- residual standard deviation;
- residual absolute mean;
- residual kurtosis.

## Result

The residual absolute mean changed by:

**-0.005894**

with:

- Cohen's dz: **-1.115037**
- 8/8 pairs lower in AI.

Residual kurtosis changed by:

**+9.759085**

with:

- Cohen's dz: **+1.056761**
- 8/8 pairs higher in AI.

## Interpretation

The residual distribution changed substantially after the img2img transformation.

This is interesting because residuals are designed to suppress ordinary visual content and expose smaller-scale processing differences.

However, residual statistics are not automatically "AI fingerprints." Camera processing, sharpening, denoising, JPEG compression and resizing can also change residuals.

---

# 13. Cross-Difference Analysis

## Concept

The notebook uses a 2×2 cross-difference:

\[
D =
|I_{00}+I_{11}-I_{10}-I_{01}|
\]

This is a local second-order difference.

It suppresses some slowly varying information and emphasizes local changes.

The absolute value is used in v3 so that positive and negative local changes do not cancel each other.

## Why it is useful

The ordinary image contains a large amount of content that is unrelated to how the image was generated.

Cross-difference attempts to reduce some of that content and make subtle local processing patterns easier to study.

The resulting cross-difference is also examined in the frequency domain.

This gives:

- cross-difference standard deviation;
- cross-difference absolute mean;
- cross-difference FFT high-frequency ratio;
- cross-difference FFT visualization.

---

# 14. Synthbuster-Inspired Periodic Residual-Frequency Analysis

## Concept

The v3 notebook adds a simplified set of residual-frequency features motivated by the general idea behind **Synthbuster-style forensic analysis**.

The goal is to look for unusually structured or periodic frequency energy after applying a high-pass/cross-difference operation.

The notebook performs the analysis separately for:

- R;
- G;
- B.

It then measures FFT magnitude at selected periodic locations corresponding to periods:

- 2;
- 4;
- 8.

Measurements are taken along:

- x;
- y;
- diagonal directions.

A high-frequency residual ratio is also calculated.

## Important qualification

This is **not a full implementation or reproduction of Synthbuster**.

It is a simplified, Synthbuster-inspired feature family intended to investigate whether residual Fourier structure differs between the matched real and AI images.

## Results

Several periodic features produced relatively large effects.

Examples include:

| Feature | Mean AI − Real | Cohen's dz |
|---|---:|---:|
| synth_r_p2_x_mean | +0.128064 | +1.977 |
| synth_b_p2_x_mean | +0.127869 | +1.937 |
| synth_g_p2_x_mean | +0.118991 | +1.833 |
| synth_b_p4_x_mean | +0.269937 | +1.636 |
| synth_r_p4_x_mean | +0.264977 | +1.596 |
| synth_g_p4_x_mean | +0.260674 | +1.541 |

These results suggest that the residual frequency representation contains structured differences between the real and AI images.

Again, these features should be treated as **exploratory forensic indicators**, not as proof of a universal diffusion fingerprint.

---

# 15. Edge, Gradient, Laplacian and Contour Analysis

## Concept

Edges are places where image values change rapidly.

The notebook uses:

- Sobel gradient magnitude;
- Laplacian magnitude;
- Canny edges;
- contour extraction.

These measurements capture different aspects of local structure.

For example:

- Sobel measures local intensity change;
- Laplacian emphasizes second-order changes;
- Canny identifies likely edges;
- contours summarize connected edge structures.

## Results

The mean gradient changed by:

**-0.006924**

with:

- Cohen's dz: **-1.372226**
- AI lower than real in all 8 pairs.

The Laplacian absolute mean changed by:

**-0.003417**

with:

- Cohen's dz: **-0.995488**

Edge density changed by:

**-0.015447**

with:

- Cohen's dz: **-0.742960**

Contour perimeter per image area changed by:

**-0.028455**

with:

- Cohen's dz: **-0.754686**

## Interpretation

The AI images in this controlled dataset tended to have lower measured local variation and edge activity.

This agrees with the texture and wavelet observations.

However, "AI images are smoother" would be too strong as a general conclusion.

The safer conclusion is:

> The particular img2img transformation used in this experiment consistently altered local gradient, edge and fine-structure measurements.

---

# 16. LBP — Local Binary Pattern Texture Analysis

## Concept

LBP describes local micro-texture.

For every pixel, it compares neighbouring pixels with the centre pixel and encodes the resulting pattern.

The v3 implementation uses:

- P = 8 neighbours;
- radius R = 1;
- uniform LBP representation.

This makes LBP sensitive to small texture structures.

It is particularly useful for studying:

- fur;
- fabric;
- wood;
- small repeated patterns;
- fine surfaces.

## Features

The notebook measures:

- LBP histogram entropy;
- dominant histogram bin;
- mean LBP code;
- standard deviation of LBP code.

## Result

LBP mean code was one of the strongest conventional features:

- mean AI − Real: **+0.474090**
- median: **+0.481371**
- Cohen's dz: **+2.256247**
- direction consistency: **8/8**
- exact sign-flip p: **0.007812**
- FDR q: **0.02532**

LBP therefore provides a strong and consistent indication that the local micro-texture representation changed after img2img processing.

The change in LBP code distribution is complementary to the DWT results because LBP focuses directly on local neighbourhood structure rather than wavelet coefficients.

---

# 17. GLCM — Gray-Level Co-occurrence Texture Analysis

## Concept

GLCM describes how often pairs of gray levels occur next to each other.

It provides statistics related to:

- local similarity;
- local variation;
- texture regularity;
- contrast;
- homogeneity.

The notebook uses:

- 32 gray levels;
- distances 1, 2 and 4;
- angles 0°, 45°, 90° and 135°;
- symmetric normalized GLCMs.

## Results

Two particularly clear features were:

### GLCM homogeneity

Mean AI − Real:

**+0.087996**

Cohen's dz:

**+1.416785**

Direction consistency:

**8/8**

### GLCM dissimilarity

Mean AI − Real:

**-0.355536**

Cohen's dz:

**-1.281584**

Direction consistency:

**8/8**

## Interpretation

The two statistics point in the same broad direction.

The AI versions became:

- more homogeneous;
- less locally dissimilar.

This supports the LBP result that the local texture distribution changed.

The agreement between LBP and GLCM is more informative than either feature alone because they measure texture in different ways.

---

# 18. Intensity, RGB Colour, HSV and Entropy Analysis

## Concept

Basic intensity and colour statistics are included as baseline controls.

The notebook measures:

### Grayscale

- mean;
- standard deviation;
- 1st percentile;
- median;
- 99th percentile;
- entropy.

### RGB

For R, G and B:

- mean;
- standard deviation;
- entropy.

It also calculates:

- R-G correlation;
- R-B correlation;
- G-B correlation.

### HSV

The notebook measures:

- mean saturation;
- mean value;
- saturation standard deviation.

## Interpretation

These features are useful because they tell us whether the AI transformation changed basic image appearance.

However, they are weaker forensic evidence.

A model can change:

- brightness;
- saturation;
- colour balance;
- contrast;

without those changes being uniquely related to AI generation.

Therefore these measurements should mainly be treated as **baseline/control features**.

---

# 19. Error Level Analysis (ELA)

## Concept

ELA compares an original image with a JPEG-recompressed version.

The difference highlights regions that respond differently to JPEG compression.

The v3 notebook improves the experimental setup by performing ELA on the **original file before the 512×512 analysis resize**.

This is preferable to first resizing the image and then performing JPEG recompression because resizing itself can alter compression-related measurements.

## Features

The notebook measures:

- ELA mean;
- ELA standard deviation;
- ELA 95th percentile;
- ELA maximum.

## Interpretation

ELA produced measurable differences between the real and AI images.

However, ELA is strongly affected by:

- JPEG history;
- previous compression;
- image format;
- quality settings;
- resizing;
- repeated editing.

Therefore ELA should not be interpreted as an AI detector.

A difference in ELA means:

> the images respond differently to recompression.

It does not automatically mean:

> one image is AI-generated.

---

# 20. Paired Feature Differences

After feature extraction, every feature is compared between the matched images.

For every feature:

\[
\Delta = AI - Real
\]

The notebook also calculates:

\[
\text{relative difference}
=
\frac{AI-Real}{|Real|+\epsilon}
\]

This creates a paired difference table.

For example, if a feature has:

- Real = 2.0
- AI = 1.5

then:

\[
\Delta = -0.5
\]

and the AI image has a 25% reduction relative to the real image.

The paired design is particularly important because each AI image has a specific original counterpart.

---

# 21. Statistical Testing

The notebook does not rely only on visual inspection.

For every scalar feature it calculates:

- number of valid pairs;
- mean difference;
- median difference;
- standard deviation of differences;
- Cohen's paired effect size \(d_z\);
- direction consistency;
- exact sign-flip p-value;
- Benjamini–Hochberg FDR-adjusted q-value.

---

# 22. Cohen's Paired Effect Size

The notebook calculates:

\[
d_z =
\frac{\operatorname{mean}(\Delta)}
{\operatorname{std}(\Delta)}
\]

where:

\[
\Delta = AI - Real
\]

The sign tells us the direction:

- positive → AI tends to be higher;
- negative → AI tends to be lower.

The absolute value tells us how large the standardized paired difference is.

For example:

- \(d_z = +2.0\) → strong increase;
- \(d_z = -2.0\) → strong decrease.

The effect size is useful because it accounts for the variability between the eight image pairs.

---

# 23. Direction Consistency

Direction consistency answers:

> In how many image pairs did the AI feature exceed the real feature?

For example:

- 8/8 increases → very consistent;
- 7/8 increases → mostly consistent;
- 4/8 increases → mixed behaviour.

This is especially useful with only eight pairs.

A large effect caused by one unusual image would be less convincing than a similar effect repeated across all eight pairs.

---

# 24. Exact Paired Sign-Flip Test

Because the experiment contains only 8 pairs, the notebook uses an **exact sign-flip permutation test**.

For eight differences there are:

\[
2^8 = 256
\]

possible assignments of signs.

The observed mean difference is compared with all possible sign-flipped versions.

The exact two-sided p-value therefore does not depend on a large-sample approximation.

### Important consequence

For this dataset, when every one of the eight pairs moves in the same direction, the smallest possible two-sided p-value from this test is:

\[
\frac{2}{256}
=
0.0078125
\]

That explains why many perfectly consistent features have:

**p = 0.007812**

This is the most extreme value this particular exact test can produce with only eight pairs.

It should therefore not be interpreted as evidence equivalent to having hundreds of independent samples.

---

# 25. Benjamini–Hochberg FDR Correction

The notebook tests hundreds of features.

If hundreds of hypotheses are tested independently, some will appear significant by chance.

The Benjamini–Hochberg procedure controls the expected false-discovery rate among the declared discoveries.

The strongest reported features in v3 have:

**q ≈ 0.02532**

This means that the multiple-comparison correction still leaves the strongest effects below a conventional 0.05 FDR threshold.

However, the small number of independent image pairs remains the major limitation.

---

# 26. Strongest Results in v3

The following table summarizes some of the strongest effects reported by the executed notebook.

| Rank | Feature | Mean AI − Real | Cohen's dz | Direction |
|---:|---|---:|---:|---|
| 1 | qwt_HL2_magnitude_entropy | -0.868330 | **-3.063** | AI lower |
| 2 | qwt_LH2_magnitude_entropy | -0.748655 | **-2.455** | AI lower |
| 3 | LH2_entropy | -0.961482 | **-2.364** | AI lower |
| 4 | HL2_entropy | -0.897274 | **-2.303** | AI lower |
| 5 | HH2_entropy | -1.313646 | **-2.300** | AI lower |
| 6 | lbp_mean_code | +0.474090 | **+2.256** | AI higher |
| 7 | qwt_HH1_GB coherence | +0.443342 | **+2.116** | AI higher |
| 8 | qwt_HH1_RB coherence | +0.447285 | **+2.107** | AI higher |
| 9 | qwt_HH1_RG coherence | +0.440841 | **+2.096** | AI higher |
| 10 | qwt_HL1_GB coherence | +0.422071 | **+2.071** | AI higher |
| 11 | qwt_HL1_RG coherence | +0.418054 | **+2.049** | AI higher |
| 12 | qwt_LH1_magnitude_entropy | -1.115397 | **-2.040** | AI lower |
| 13 | qwt_HL1_RB coherence | +0.418595 | **+2.021** | AI higher |
| 14 | synth_r_p2_x_mean | +0.128064 | **+1.977** | AI higher |
| 15 | synth_b_p2_x_mean | +0.127869 | **+1.937** | AI higher |
| 16 | qwt_HH2_magnitude_entropy | -1.121478 | **-1.915** | AI lower |
| 17 | HL1_entropy | -1.135556 | **-1.880** | AI lower |
| 18 | qwt_HL1_magnitude_entropy | -1.129334 | **-1.878** | AI lower |
| 19 | LH1_absmean | -0.006025 | **-1.842** | AI lower |
| 20 | synth_g_p2_x_mean | +0.118991 | **+1.833** | AI higher |

The strongest features shown in the executed report had:

- **n = 8 pairs**
- exact sign-flip **p = 0.007812**
- FDR-adjusted **q = 0.02532**

where applicable.

---

# 27. What the v3 Results Suggest

The analyses do not all measure the same thing.

Instead, several independent representations point toward related changes.

## 27.1 Fine-scale wavelet information changed

The ordinary DWT and the color-quaternion wavelet analysis both show strong decreases in entropy of fine-detail representations.

This is one of the clearest patterns in the experiment.

The ordinary grayscale wavelet result is important because it does not depend on the new quaternion module.

The quaternion magnitude-entropy result then extends the observation into a color-aware representation.

---

## 27.2 Local texture changed

LBP and GLCM both show substantial changes.

LBP mean code increased strongly.

GLCM homogeneity increased while dissimilarity decreased.

Together, these suggest that the AI transformation changed the local texture statistics of the photographs.

---

## 27.3 RGB wavelet relationships changed

The new quaternion analysis indicates that the relationships among RGB wavelet coefficients changed.

The phase-proxy circular coherence increased consistently in several fine-detail bands.

This is potentially one of the most interesting new observations in v3.

But it remains exploratory until tested with:

- a larger dataset;
- multiple generators;
- multiple image types;
- a more faithful analytic QWT/CQWT implementation.

---

## 27.4 Residual frequency structure changed

The Synthbuster-inspired periodic residual features produced several strong effects.

This suggests that after suppressing much of the ordinary image content, the remaining local signal has structured frequency differences between the real and AI images.

This is especially interesting for future forensic work because generation pipelines can leave subtle processing traces that are difficult to see in the normal RGB image.

---

## 27.5 Local edge behaviour changed

Gradient, Laplacian and edge measurements generally decreased.

This agrees with the broader observation that the img2img transformation altered local high-frequency structure.

---

## 27.6 Basic colour and intensity statistics are weaker evidence

Brightness, colour and HSV measurements can change simply because the AI transformation changes the appearance of the photograph.

Therefore they are useful baselines but should not be given the same forensic importance as the more structural representations.

---

## 27.7 ELA is supportive rather than decisive

ELA detected differences but is heavily influenced by compression history.

It should therefore remain a supporting analysis.

---

# 28. Why the Quaternion Results Are Important

The most important development from the previous notebook to v3 is that the strongest effect in the complete report is now a **quaternion-wavelet magnitude entropy feature**:

**qwt_HL2_magnitude_entropy, Cohen's dz = -3.063**

This is larger than the strongest conventional DWT effect.

More importantly, the result is not isolated.

The following all show strong effects:

- qwt HL2 magnitude entropy;
- qwt LH2 magnitude entropy;
- qwt HH2 magnitude entropy;
- ordinary LH2 entropy;
- ordinary HL2 entropy;
- ordinary HH2 entropy.

This produces a coherent pattern:

> multiple wavelet representations are detecting changes in fine-scale coefficient distributions after the img2img transformation.

That is more meaningful than finding one isolated high-effect feature.

---

# 29. What We Should NOT Claim

The current experiment does **not** justify the following claims:

### Not justified:

> "QWT proves that an image is AI-generated."

### Not justified:

> "The phase-coherence feature is a universal AI fingerprint."

### Not justified:

> "All AI images have lower wavelet entropy."

### Not justified:

> "The notebook is a trained AI detector."

### Not justified:

> "A p-value of 0.007812 means the result is universally proven."

The experiment only contains eight matched pairs and one controlled img2img setup.

The correct scientific interpretation is:

> The experiment found strong and consistent feature differences between these matched real photographs and their corresponding AI/img2img versions.

---

# 30. What We CAN Reasonably Claim

The experiment provides evidence that the particular img2img transformation used here produces measurable changes in:

- fine-scale wavelet structure;
- color-wavelet coefficient distributions;
- RGB wavelet relationships;
- local micro-texture;
- GLCM texture;
- local gradients and edges;
- residual statistics;
- residual frequency structure;
- block-DCT coefficients.

The most compelling results are those where multiple measurements agree.

For example:

**DWT + QWT**

both show reductions in fine-detail entropy.

**LBP + GLCM**

both indicate altered local texture.

**Residual + Synthbuster-inspired FFT**

both indicate changes in high-frequency residual structure.

This convergence is the strongest aspect of the current experiment.

---

# 31. Limitations

The most important limitations are:

## 31.1 Only eight image pairs

This is the largest limitation.

Eight pairs are enough for an exploratory paired experiment but not enough to establish a general AI-image detector.

---

## 31.2 Controlled img2img source

The AI images come from a particular controlled workflow.

A different generator, model version, sampler, prompt or denoising strength could produce different forensic patterns.

---

## 31.3 Image content

The eight photographs contain different subjects.

Texture and frequency statistics naturally depend on image content.

The paired design reduces this problem but does not eliminate it.

---

## 31.4 Resizing and cropping

The main analysis uses a 512×512 center crop.

Although v3 improves preprocessing by preserving aspect ratio before cropping, the crop can still remove image content.

---

## 31.5 Compression

JPEG compression can affect:

- FFT;
- DCT;
- residuals;
- edges;
- ELA;
- wavelet coefficients.

Therefore future testing should deliberately vary compression.

---

## 31.6 Multiple comparisons

Hundreds of features are tested.

FDR correction helps control false discoveries, but it does not increase the number of independent image pairs.

---

## 31.7 QWT implementation

The quaternion-wavelet module is intentionally transparent and GPU-friendly, but it is not a drop-in implementation of a canonical published analytic QWT/CQWT detector.

The phase variables are explicitly **phase proxies**.

This distinction should remain in any report or presentation.

---

# 32. Recommended Next Step

The next stage should not immediately be to add many more unrelated transforms.

The strongest direction is to validate the wavelet findings.

A good progression is:

### Step 1 — Increase the dataset

Move from 8 pairs to a substantially larger number of matched real/AI pairs.

### Step 2 — Add multiple AI generators

Test whether the same features appear across different image-generation systems.

### Step 3 — Test different image categories

Include:

- portraits;
- landscapes;
- objects;
- text;
- animals;
- buildings;
- textures;
- indoor scenes.

### Step 4 — Test compression robustness

Evaluate the features after:

- PNG;
- high-quality JPEG;
- medium-quality JPEG;
- stronger JPEG compression.

### Step 5 — Implement a more faithful QWT/CQWT formulation

The current quaternion Haar module should be treated as an exploratory result.

The next research step would be to implement a more faithful analytic QWT/CQWT representation and compare:

1. ordinary grayscale DWT;
2. current color-quaternion Haar features;
3. literature-style analytic QWT/CQWT features.

### Step 6 — Feature selection

Once the dataset is large enough, identify which features remain useful without relying on the entire 539-feature set.

### Step 7 — Train a classifier

Only after the feature behaviour has been validated should the project move toward a classifier.

Possible models include:

- logistic regression;
- SVM;
- random forest;
- gradient boosting.

The classifier should be evaluated using images and generators that were not used during training.

---

# 33. Final Conclusion

The v3 experiment provides a stronger forensic picture than the original analysis because it combines several complementary representations.

The most important result is the strong change in **wavelet-domain structure**.

Ordinary DWT shows large, consistent decreases in detail entropy.

The new color-quaternion wavelet analysis finds an even stronger effect in:

**qwt_HL2_magnitude_entropy**

with:

**Cohen's dz = -3.063**

and also finds strong effects in other quaternion magnitude-entropy features.

The quaternion phase-proxy measurements additionally show consistent changes in RGB wavelet relationships.

At the same time:

- LBP shows a strong change in micro-texture;
- GLCM shows increased homogeneity and reduced dissimilarity;
- residual analysis shows changed high-frequency residual statistics;
- Synthbuster-inspired features show structured periodic residual-frequency differences;
- gradient and edge measurements change consistently;
- DCT supports changes in local frequency information.

The important scientific conclusion is therefore not:

> "We found the one feature that detects AI."

Instead, it is:

> **The img2img transformation produced coordinated changes across spatial, texture, frequency, residual and color-wavelet representations of the matched photographs.**

The v3 quaternion analysis makes the project particularly interesting because it suggests that the transformation may alter not only the amount of fine-scale information, but also the **relationships between color channels within fine-scale wavelet structures**.

That observation is promising, but it is still exploratory.

The next meaningful test is not more statistical analysis on the same eight images. It is **validation on a larger and more diverse dataset**, ideally with multiple generators and a more faithful QWT/CQWT implementation.

---

# 34. Short Summary for Presentation

If the entire analysis has to be explained to a professor in a few sentences:

> We compared eight real photographs with their corresponding AI-generated img2img versions using several image-forensic representations. We examined frequency information using FFT and DCT, multiscale detail using DWT, local texture using LBP and GLCM, processing residuals, edges, compression behaviour using ELA, and finally a color-aware quaternion wavelet representation. The strongest and most consistent differences appeared in fine-scale wavelet entropy, with the new quaternion-wavelet feature qwt_HL2_magnitude_entropy producing a paired Cohen's dz of approximately -3.06. LBP, GLCM, residual-frequency and RGB wavelet phase-proxy features also showed consistent changes. These results suggest that the img2img process leaves measurable changes across several complementary representations, but the dataset is only eight pairs, so the findings are preliminary and must be validated on a larger multi-generator dataset before being considered a general AI-image detection method.
