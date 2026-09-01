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

| Analysis | What We Did | Main Result |
|---|---|---|
| **FFT / Frequency** | Analyzed image magnitude, phase and frequency distribution | **Meaningful differences** in frequency structure |
| **DWT / Haar Wavelets** | 3-level Haar decomposition with detail-band energy and entropy | **Very strong differences**, especially level-2 detail entropy |
| **QWT / Color-Quaternion Wavelets** | RGB-aware Haar coefficients with quaternion magnitude and phase-proxy features | **Strongest results**; QWT HL2 entropy reached dz ≈ **-3.06** |
| **DCT** | 8×8 block DCT and high-frequency energy | **Meaningful supporting differences** |
| **Residual + Synthbuster** | High-pass/cross-difference residuals and periodic FFT peaks | **Strong periodic differences** in several RGB features |
| **Edges / Gradients** | Sobel, Laplacian, Canny and contours | **Moderate supporting differences**; AI images generally had lower gradients |
| **LBP** | Local binary patterns for micro-texture | **Strong and consistent texture differences** |
| **GLCM** | Contrast, homogeneity, dissimilarity and other texture statistics | **Consistent texture changes** |
| **Intensity / Color** | RGB, HSV, brightness and entropy statistics | **Some differences**, but weak as forensic evidence |
| **ELA** | JPEG recompression error analysis | **Small supporting differences**; compression-dependent |
| **Statistical Analysis** | Paired differences, Cohen's *d*, sign-flip tests and FDR | Identified the **most consistent Real vs AI features** |

Data outputs numerical table

| Analysis / Feature                  | AI − Real | Cohen's dz |
| ----------------------------------- | --------: | ---------: |
| QWT HL2 magnitude entropy           |    -0.868 |  **-3.06** |
| QWT LH2 magnitude entropy           |    -0.749 |  **-2.45** |
| DWT LH2 entropy                     |    -0.961 |  **-2.36** |
| DWT HL2 entropy                     |    -0.897 |  **-2.30** |
| DWT HH2 entropy                     |    -1.314 |  **-2.30** |
| LBP mean code                       |    +0.474 |  **+2.26** |
| QWT HH1 phase coherence             |    +0.443 |  **+2.12** |
| Synthbuster periodic peak (R, p2-x) |    +0.128 |  **+1.98** |
| DCT high-frequency ratio            |    -0.080 |  **-1.37** |

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

[CHANGELOG.MD FILE](./CHANGELOG.md)
