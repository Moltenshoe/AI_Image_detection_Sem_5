# AI Image Detection -- Semester 5 Minor Project

This project investigates the differences between real photographs and
AI-generated images using image processing, texture analysis, and
frequency-domain techniques.

## Repository Structure

-   `real/` --- Real photographs used for analysis
-   `ai/` --- AI-generated images corresponding to the real photographs
-   `original_photos/` --- Original captured photographs
-   `analysis_outputs/` --- Generated analysis results
-   `Minor/` --- Project-related files
-   `analysis.ipynb` --- Earlier analysis notebook
-   `new_analysis.ipynb` --- Main image-analysis notebook
-   `crop_dataset.py` --- Dataset/image preprocessing script

## Dataset

The dataset consists of real photographs and AI-generated counterparts.

### Original Photos

[View Original Photos](./original_photos/)

### Real Images

[View Real Images](./real/)

### AI-Generated Images

[View AI-Generated Images](./ai/)

## Image Analysis

The notebook applies multiple techniques to examine characteristics that
may differ between real and AI-generated images:

-   Fast Fourier Transform (FFT)
-   Discrete Wavelet Transform (DWT)
-   Discrete Cosine Transform (DCT)
-   Edge and contour analysis
-   Local Binary Patterns (LBP)
-   Gray-Level Co-occurrence Matrix (GLCM) texture analysis
-   Intensity analysis
-   Error Level Analysis (ELA)
-   Cross-difference analysis
-   Frequency-domain analysis

The generated results are stored in:

[View Analysis Outputs](./analysis_outputs/)

The Analysis of the images is stored in the analysis.md file

[View Analysis.md file](./ANALYSIS.md)

New analysis with respect to v3

[View Analysis_v3.md file](./ANALYSIS_v3.md)

## Running the Analysis

1.  Clone the repository.
2.  Install the required Python libraries.
3.  Open `new_analysis.ipynb` using Jupyter Notebook or JupyterLab.
4.  Make sure the `real/` and `ai/` folders contain the required images.
5.  Run the notebook cells to perform the analysis.

Table for the types of analysis used:

| Analysis                                    | What we actually did                                                                                                                  | Establishedness / amount of research                                                                                                                                                                   | Is the method itself good in the field?                                                                                                                                                                                                                                 | Bleeding-edge / future potential?                                                                                  | Your result                                                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **FFT / Fourier analysis**                  | 2-D FFT, magnitude, phase, radial power, low/mid/high-frequency energy, spectral centroid                                             | **Very high** — Fourier analysis is foundational signal processing; frequency-domain image forensics has been studied for years                                                                        | **Yes, but context-dependent.** Frequency artifacts can be useful, but simple spectral statistics alone are not robust universal detectors. Current surveys still treat frequency-domain analysis as a major AI-image-forensics family. ([DOI][1])                      | **High** when combined with learned/fingerprint methods; particularly interesting for diffusion artifacts          | **Meaningful.** Several frequency statistics changed consistently between Real and AI.                                           |
| **FFT phase**                               | Fourier phase spectrum visualized/compared alongside magnitude                                                                        | **Very high** as a signal-processing concept                                                                                                                                                           | **Useful supporting evidence**, but not generally a standalone AI detector                                                                                                                                                                                              | **Moderate**                                                                                                       | **Some difference**, but not one of our strongest quantitative signals                                                           |
| **DWT / Haar wavelets**                     | 3-level 2-D Haar decomposition; LL/LH/HL/HH energy, entropy and statistics                                                            | **Very high** — wavelets have been used in image processing/forensics for decades                                                                                                                      | **Yes.** Wavelet-domain features are well established in classical image forensics, particularly for multiscale/noise/texture analysis. ([IET][2])                                                                                                                      | **Moderate.** The basic Haar DWT itself isn't bleeding-edge; learned/multiscale wavelet features are more current  | **Very meaningful.** Several level-2 detail-band entropy features showed highly consistent Real→AI decreases                     |
| **DCT**                                     | 8×8 block DCT; DC energy, coefficient magnitude and high-frequency energy                                                             | **Very high** — especially because JPEG itself is based on block DCT; DCT-based forensic analysis has extensive literature. ([IET][3])                                                                 | **Yes**, particularly for compression/double-compression/JPEG forensics; less powerful as a generic AI detector                                                                                                                                                         | **Low–moderate** for raw handcrafted DCT statistics; **higher** when DCT features are combined with learned models | **Meaningful supporting result**, but not among our strongest signals                                                            |
| **Residual / high-pass analysis**           | Gaussian high-pass residual; residual std, absmean, kurtosis and high-frequency energy                                                | **High** — residual/noise analysis is a core classical forensic strategy. Recent surveys explicitly describe residual filtering as a fundamental stage in classical manipulation detection. ([DOI][4]) | **Yes.** Very useful because forensic traces are often easier to see after suppressing scene content                                                                                                                                                                    | **High** when residuals are combined with learned fingerprints                                                     | **Meaningful.** AI images showed systematic residual differences                                                                 |
| **Cross-difference residual**               | 2×2 diagonal cross-difference filter followed by statistics and Fourier analysis                                                      | **Moderate–high** in forensic signal processing; the exact filter is much more specialized than generic high-pass filtering                                                                            | **Good as a forensic feature extractor**, but not a standalone detector                                                                                                                                                                                                 | **High when paired with spectral analysis**, especially because this connects directly to diffusion fingerprints   | **Meaningful**                                                                                                                   |
| **Synthbuster-inspired spectral features**  | Cross-difference residual → FFT → periodic peaks around periods 2/4/8 for RGB channels                                                | **Recent** — Synthbuster was published in 2024 and specifically targets diffusion-generated images through residual Fourier artifacts. ([IEEE Signal Processing Society][5])                           | **Yes — this is one of the most relevant analyses in our project.** Synthbuster demonstrated that diffusion models can leave exploitable frequency artifacts and reported generalization to unknown models/mild JPEG compression. ([IEEE Signal Processing Society][5]) | **Very high.** This is the clearest **bleeding-edge direction** among our current analyses                         | **Very meaningful in our dataset.** Several periodic features had strong and consistent Real→AI effects                          |
| **Sobel / gradient**                        | Spatial gradient magnitude; mean/std/p95                                                                                              | **Very high** — Sobel dates to the classical era of computer vision. ([MDPI][6])                                                                                                                       | **Useful supporting feature**, but weak as a standalone AI detector because natural image content strongly affects gradients                                                                                                                                            | **Low** for the basic operator; **higher** when incorporated into learned forensic representations                 | **Meaningful supporting result**; AI generally showed lower gradient statistics                                                  |
| **Laplacian**                               | Second-order derivative; absolute mean/std                                                                                            | **Very high** — classic image-processing operation                                                                                                                                                     | **Useful for detecting sharpness/high-frequency differences**, but not a specialized AI detector                                                                                                                                                                        | **Low** by itself                                                                                                  | **Some signal**, mainly supporting the texture/sharpness story                                                                   |
| **Canny edges**                             | Canny edge map and edge density                                                                                                       | **Very high** — Canny was introduced in 1986 and remains foundational. ([MDPI][6])                                                                                                                     | **Excellent general edge detector**, but **not particularly strong as an AI-forensics method by itself**                                                                                                                                                                | **Low** for classical Canny; modern learned edge/feature extractors have greater potential                         | **Useful supporting evidence**, not a major discriminator                                                                        |
| **Contours**                                | Canny/threshold-derived contours; contour count, perimeter and area                                                                   | **High** as classical computer vision                                                                                                                                                                  | **Good for object/shape analysis**, but weak for distinguishing real photography from img2img AI because scene composition dominates                                                                                                                                    | **Low** in this application                                                                                        | **Weak/not particularly meaningful** for our AI-vs-real question                                                                 |
| **LBP**                                     | Local Binary Pattern map; histogram, entropy, dominant bin, mean/std code                                                             | **High** — LBP was introduced in 1996 and has extensive texture/forensics literature. ([Taylor & Francis Online][7])                                                                                   | **Yes, as a texture descriptor.** It has been directly investigated for image-forgery detection. ([Taylor & Francis Online][7])                                                                                                                                         | **Low–moderate** for raw LBP; higher when combined with richer learned texture representations                     | **One of our strongest results.** Mean LBP code changed consistently across all 8 pairs; histogram entropy also changed strongly |
| **GLCM / Haralick texture**                 | GLCM at distances 1,2,4 and angles 0°,45°,90°,135°; contrast, dissimilarity, homogeneity, ASM, energy, correlation, entropy           | **Very high** — GLCM is a classic texture-analysis technique with decades of use                                                                                                                       | **Yes as a texture feature**, including demonstrated use in image-splicing forensics. ([ScienceDirect][8])                                                                                                                                                              | **Low–moderate** by itself; more potential when fused with modern learned features                                 | **One of our strongest supporting results.** Homogeneity increased and dissimilarity decreased consistently                      |
| **Intensity statistics**                    | Mean/std/percentiles and intensity entropy                                                                                            | **Very high** — basic image statistics                                                                                                                                                                 | **Weak as forensic evidence** because lighting/exposure/content naturally changes them                                                                                                                                                                                  | **Very low**                                                                                                       | **Not particularly meaningful as a detector**; useful baseline/control                                                           |
| **RGB / color statistics**                  | RGB mean/std/entropy and channel correlations                                                                                         | **Very high**                                                                                                                                                                                          | **Weak–moderate**; useful for identifying systematic processing/color shifts but not a reliable AI fingerprint                                                                                                                                                          | **Low** for handcrafted statistics                                                                                 | **Some differences, but weak forensic evidence**                                                                                 |
| **HSV saturation/value**                    | Saturation/value mean/std/percentiles                                                                                                 | **High** as standard image analysis                                                                                                                                                                    | **Weak as an AI-forensics method**                                                                                                                                                                                                                                      | **Low**                                                                                                            | **Supporting only**                                                                                                              |
| **Entropy**                                 | Shannon entropy over intensity/texture/frequency-derived representations                                                              | **Very high** as an information/texture statistic                                                                                                                                                      | **Useful as a descriptor**, but not a detector by itself                                                                                                                                                                                                                | **Low** alone; potentially useful inside learned forensic feature sets                                             | **Meaningful mainly inside DWT/LBP/residual analyses**                                                                           |
| **ELA**                                     | JPEG re-save at quality 90 → absolute difference → mean/std/p95 + map                                                                 | **High/established** — ELA dates to Krawetz's 2007 work and has been widely used in practical image forensics. ([forensics.media][9])                                                                  | **Limited.** Useful as a JPEG/compression diagnostic, **not strong evidence of AI generation**. It is sensitive to resaving/compression history. ([forensics.media][9])                                                                                                 | **Low** as a bleeding-edge AI detector                                                                             | **It showed a consistent difference in our dataset, but I would classify it as supporting evidence, not a major AI signal**      |
| **Paired effect-size/statistical analysis** | AI−Real deltas, relative deltas, exact sign-flip permutation tests, Cohen's paired `d_z`, direction consistency and BH-FDR correction | **Very established statistical methodology**                                                                                                                                                           | **Very good for evaluating our experiment**, but it isn't itself an image-forensics technique                                                                                                                                                                           | **Low as a technology; high importance scientifically**                                                            | **Very meaningful because it tells us which image features are consistently different across the 8 matched pairs**               |

[1]: https://doi.org/10.1016/j.cosrev.2026.100908?utm_source=chatgpt.com "Methods and trends in detecting AI-generated images: A comprehensive review - ScienceDirect"
[2]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2016.0322?utm_source=chatgpt.com "Review, analysis and parameterisation of techniques for copy–move forgery detection in digital images - Dixit - 2017 - IET Image Processing - Wiley Online Library"
[3]: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/el.2019.2719?utm_source=chatgpt.com "End‐to‐end double JPEG detection with a 3D convolutional network in the DCT domain - Ahn - 2020 - Electronics Letters - Wiley Online Library"
[4]: https://doi.org/10.1145/3731243?utm_source=chatgpt.com "Unravelling Digital Forgeries: A Systematic Survey on Image Manipulation Detection and Localization | ACM Computing Surveys"
[5]: https://signalprocessingsociety.org/index.php/publications-resources/ieee-open-journal-signal-processing/2024/01/synthbuster-towards-detection?utm_source=chatgpt.com "Synthbuster: Towards Detection of Diffusion Model Generated Images | IEEE Signal Processing Society"
[6]: https://www.mdpi.com/2227-7390/13/15/2464?utm_source=chatgpt.com "A Mathematical Survey of Image Deep Edge Detection Algorithms: From Convolution to Attention"
[7]: https://www.tandfonline.com/doi/full/10.1080/19361610.2017.1422367?utm_source=chatgpt.com "Blind Forensics of Images using Higher Order Local Binary Pattern: Journal of Applied Security Research: Vol 13 , No 2 - Get Access"
[8]: https://www.sciencedirect.com/science/article/abs/pii/S0923596524000353?utm_source=chatgpt.com "Image splicing detection using low-dimensional feature vector of texture features and Haralick features based on Gray Level Co-occurrence Matrix - ScienceDirect"
[9]: https://forensics.media/articles/is-error-level-analysis-reliable/?utm_source=chatgpt.com "Is Error Level Analysis reliable? — forensics.media"


## Technologies

-   Python
-   NumPy
-   OpenCV
-   SciPy
-   scikit-image
-   PyWavelets
-   Matplotlib
-   Jupyter Notebook

## Objective

The objective of this minor project is to investigate whether
image-processing and frequency-domain features can reveal measurable
differences between real photographs and AI-generated images.

This is the img-to-img model I have used for creating the AI counterparts of the Real images

[Img2Img model](https://huggingface.co/spaces/diffusers/unofficial-SDXL-Turbo-i2i-t2i)

Base model: stabilityai/sdxl-turbo

1. Distilled from SDXL 1.0 via Adversarial Diffusion Distillation (ADD) — a student net trained w/ combined score-distillation + adversarial loss, so it generates coherent images in 1-4 steps instead of SDXL's normal 20-50.
2. ~3.1B params, UNet+CNN backbone (not a transformer-diffusion arch like SD3/FLUX).
3. No classifier-free guidance at inference (guidance_scale=0.0) — that's why it's fast, but also why prompt adherence is looser than full SDXL.
4. Non-commercial license (Stability AI research license).

This specific model:

1. Runs on ZeroGPU (shared/queued GPU, not dedicated), fp16, diffusers.AutoPipelineForImage2Image.
2. Forces width=512, height=512 — matches your 512x512 need exactly.
3. Session model: click "Start" → holds GPU 60s, streams new output on every param change (prompt/strength/steps/seed) — realtime-ish, not per-request queue.
4. Strength slider 0-1, steps 1-10 (paper recommends 2-4 for i2i quality).
5. No safety checker loaded (safety_checker=None).

## Project Status

**Semester 5 Minor Project --- In Progress**
