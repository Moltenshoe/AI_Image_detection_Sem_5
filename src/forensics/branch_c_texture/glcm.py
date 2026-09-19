"""Gray-Level Co-occurrence Matrix (GLCM) computation module (Block 2 — Branch C).

Provides GLCM texture feature extraction using quantized spatial co-occurrence matrices
and Haralick texture properties.

Mathematical & Implementation Specifications:
1. Grayscale Quantization:
   Continuous luminance Y in [0.0, 1.0] is linearly quantized into Ng = 16 discrete levels:
       Quantized(x, y) = min(floor(Y(x, y) * Ng), Ng - 1) in {0, 1, ..., 15}.
   Rationale: Reduces GLCM matrix dimensionality from 256x256 (65,536 bins) to 16x16 (256 bins),
   eliminating matrix sparsity, providing robustness to sensor noise, and speeding up computation.

2. Displacements & Orientations:
   - Multi-scale pixel distances: d in {1, 2, 4} (capturing micro-texture, intermediate, and macro scales).
   - Standard 2D directions: theta in {0, pi/4, pi/2, 3*pi/4} (0 deg, 45 deg, 90 deg, 135 deg).
   - Matrix properties: Symmetrical (P(i, j) = P(j, i)), Normalized (sum_{i,j} P(i, j) = 1.0).

3. Haralick Texture Statistics:
   For each (d, theta) co-occurrence matrix P:
   - Contrast:      sum_{i,j} |i - j|^2 * P(i, j)
   - Dissimilarity: sum_{i,j} |i - j| * P(i, j)
   - Homogeneity:   sum_{i,j} P(i, j) / (1 + (i - j)^2)
   - Energy (ASM):  sqrt(sum_{i,j} P(i, j)^2)
   - Correlation:   sum_{i,j} ((i - mu_i) * (j - mu_j) * P(i, j)) / (sigma_i * sigma_j)
   - Entropy:       -sum_{i,j, P(i,j)>0} P(i, j) * log2(P(i, j) + 1e-12)

4. Aggregation Strategy & Candidate Feature Justification:
   - Directional Means (18 scalars = 6 statistics x 3 distances d in {1, 2, 4}):
     Averages across the 4 angles to achieve primary rotation invariance across multiple spatial scales.
   - Directional Standard Deviations at d=1 (6 scalars = 6 statistics x 1 scale):
     Captures textural directionality / anisotropy (variance across angles theta) at the immediate
     micro-pixel scale (d=1), which is particularly sensitive to directional raster artifacts
     or oriented generative grid patterns.
     *Candidate Note:* Restricting directional standard deviations to d=1 maintains a compact
     24-feature vector (rather than 36). This is an explicit candidate design for Block 4 evaluation,
     not a theoretically mandatory choice.

Total: Exactly 24 scalar float features.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from skimage.feature import graycomatrix, graycoprops
import torch


def quantize_grayscale(gray: torch.Tensor, levels: int = 16) -> np.ndarray:
    """Quantize grayscale tensor into discrete integer levels.

    Args:
        gray: Grayscale tensor [..., 1, H, W] or [..., H, W], float32 in [0.0, 1.0].
        levels: Number of discrete quantization levels Ng (default: 16).

    Returns:
        2D uint8 numpy array of shape [H, W] with integer values in [0, levels - 1].
    """
    if gray.ndim >= 3 and gray.shape[-3] == 1:
        img_tensor = gray.squeeze(-3)
    else:
        img_tensor = gray

    if img_tensor.ndim > 2:
        img_tensor = img_tensor[0]  # Take first sample if batched

    # Clamp to [0.0, 1.0] and quantize
    clamped = img_tensor.clamp(0.0, 1.0)
    quantized = torch.floor(clamped * levels).to(torch.int64)
    quantized = quantized.clamp_max(levels - 1)

    return quantized.detach().cpu().numpy().astype(np.uint8)


def compute_glcm_matrix(
    image_quantized: np.ndarray,
    levels: int = 16,
    distances: Sequence[int] = (1, 2, 4),
    angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> np.ndarray:
    """Compute normalized symmetrical Gray-Level Co-occurrence Matrix.

    Args:
        image_quantized: 2D uint8 numpy array [H, W] with values in [0, levels - 1].
        levels: Number of gray levels Ng (default: 16).
        distances: Sequence of integer pixel offsets (default: (1, 2, 4)).
        angles: Sequence of orientation angles in radians (default: 4 standard angles).

    Returns:
        4D numpy array of shape [levels, levels, len(distances), len(angles)], float64.
    """
    glcm = graycomatrix(
        image_quantized,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )
    return glcm


def compute_glcm_features(
    gray: torch.Tensor,
    levels: int = 16,
    distances: Sequence[int] = (1, 2, 4),
    angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> Dict[str, float]:
    """Extract all 24 scalar detector features from Gray-Level Co-occurrence Matrix.

    Args:
        gray: Canonical grayscale image tensor [..., 1, 256, 256] or [..., 256, 256], float32 in [0, 1].
        levels: Quantization levels Ng (default: 16).
        distances: Distance offsets (default: (1, 2, 4)).
        angles: Directional angles (default: 4 standard angles).

    Returns:
        Dictionary of exactly 24 scalar float features in deterministic order:
            - For each distance d in (1, 2, 4):
                glcm_d{d}_contrast_mean
                glcm_d{d}_dissimilarity_mean
                glcm_d{d}_homogeneity_mean
                glcm_d{d}_energy_mean
                glcm_d{d}_correlation_mean
                glcm_d{d}_entropy_mean
            - Directional standard deviations across angles at d=1:
                glcm_d1_contrast_std
                glcm_d1_dissimilarity_std
                glcm_d1_homogeneity_std
                glcm_d1_energy_std
                glcm_d1_correlation_std
                glcm_d1_entropy_std
    """
    quantized = quantize_grayscale(gray, levels=levels)
    glcm = compute_glcm_matrix(quantized, levels=levels, distances=distances, angles=angles)

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    ]

    # Compute standard scikit-image Haralick properties
    # Each returns shape [len(distances), len(angles)]
    prop_values: Dict[str, np.ndarray] = {}
    for prop in properties:
        vals = graycoprops(glcm, prop)
        # Handle potential NaNs in correlation for constant/flat images
        if prop == "correlation":
            vals = np.nan_to_num(vals, nan=1.0)
        prop_values[prop] = vals

    # Compute GLCM Shannon Entropy across (levels, levels) for each (d, theta)
    # Shape: [len(distances), len(angles)]
    entropy_matrix = -np.sum(
        np.where(glcm > 0, glcm * np.log2(glcm + 1e-12), 0.0),
        axis=(0, 1),
    )
    prop_values["entropy"] = entropy_matrix

    all_stat_names = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "entropy",
    ]

    features: Dict[str, float] = {}

    # 1. Multi-distance Directional Means (18 scalars = 6 statistics x 3 distances)
    for d_idx, d in enumerate(distances):
        for stat in all_stat_names:
            vals_d = prop_values[stat][d_idx, :]  # Shape: [len(angles)]
            mean_val = float(np.mean(vals_d))
            features[f"glcm_d{d}_{stat}_mean"] = mean_val

    # 2. Directional Standard Deviations at d=1 (6 scalars = 6 statistics x 1 scale)
    # Captures orientation anisotropy across the 4 angles at micro-scale d=1
    d1_idx = 0  # index for distance d=1
    for stat in all_stat_names:
        vals_d1 = prop_values[stat][d1_idx, :]  # Shape: [len(angles)]
        std_val = float(np.std(vals_d1))
        features[f"glcm_d1_{stat}_std"] = std_val

    return features
