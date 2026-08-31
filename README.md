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

## Running the Analysis

1.  Clone the repository.
2.  Install the required Python libraries.
3.  Open `new_analysis.ipynb` using Jupyter Notebook or JupyterLab.
4.  Make sure the `real/` and `ai/` folders contain the required images.
5.  Run the notebook cells to perform the analysis.

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

## Project Status

**Semester 5 Minor Project --- In Progress**
