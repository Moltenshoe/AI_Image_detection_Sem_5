"""Local Texture Feature Extractors (Block 2 — Branch C).

Provides three independent, alternative local texture extraction pipelines:
1. `extract_branch_c_lbp_features`: Standard rotation-invariant uniform LBP (16 scalars).
2. `extract_branch_c_glcm_features`: Multi-distance Gray-Level Co-occurrence Matrix (24 scalars).
3. `extract_branch_c_lbp_edge_features`: Canny edge-guided LBP candidate (16 scalars).

These pipelines are alternative descriptors that can be evaluated independently or in multi-branch
combinations in Block 4. They are NOT automatically combined into a single feature vector.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from src.forensics.branch_c_texture.glcm import compute_glcm_features
from src.forensics.branch_c_texture.lbp import (
    compute_canny_edges,
    compute_lbp_map,
    extract_lbp_statistics,
)


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor to grayscale using standard luminance weights.

    Formula: 0.299 * R + 0.587 * G + 0.114 * B (ITU-R BT.601 standard weights).

    Args:
        image: Tensor of shape [..., 3, H, W] or [..., 1, H, W].

    Returns:
        Grayscale tensor of shape [..., 1, H, W].
    """
    if image.shape[-3] == 1:
        return image
    if image.shape[-3] != 3:
        raise ValueError(
            f"Expected image with 1 or 3 channels at dim -3, got shape {tuple(image.shape)}"
        )
    return (
        0.299 * image[..., 0:1, :, :]
        + 0.587 * image[..., 1:2, :, :]
        + 0.114 * image[..., 2:3, :, :]
    )


def extract_branch_c_lbp_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract standard rotation-invariant uniform LBP features (C_LBP alternative).

    Pipeline:
        RGB [3, 256, 256] -> Grayscale -> Uniform LBP map -> 10-bin histogram + 6 statistics.

    Args:
        image: Canonical image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of exactly 16 scalar float features:
            - lbp_bin_0 to lbp_bin_8 (9 uniform pattern frequency bins)
            - lbp_bin_nonuniform (non-uniform pattern frequency bin)
            - lbp_entropy (Shannon entropy of 10-bin distribution in bits)
            - lbp_uniformity (sum of squared probabilities)
            - lbp_dominant_bin (maximum bin probability)
            - lbp_mean_code (expected code value)
            - lbp_std_code (standard deviation of codes)
            - lbp_nonuniform_ratio (explicit ratio of non-uniform patterns)
    """
    gray = rgb_to_gray(image)
    lbp_map = compute_lbp_map(gray, p=8, r=1)
    features = extract_lbp_statistics(lbp_map, mask=None, prefix="lbp")
    return features


def extract_branch_c_glcm_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract Gray-Level Co-occurrence Matrix features (C_GLCM alternative).

    Pipeline:
        RGB [3, 256, 256] -> Grayscale -> 16-level Quantization -> Symmetrical GLCM (d={1,2,4}, 4 angles)
        -> 18 directional means + 6 directional standard deviations at d=1.

    Args:
        image: Canonical image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of exactly 24 scalar float features:
            - For d in (1, 2, 4):
                glcm_d{d}_contrast_mean
                glcm_d{d}_dissimilarity_mean
                glcm_d{d}_homogeneity_mean
                glcm_d{d}_energy_mean
                glcm_d{d}_correlation_mean
                glcm_d{d}_entropy_mean
            - At scale d=1 (anisotropy):
                glcm_d1_contrast_std
                glcm_d1_dissimilarity_std
                glcm_d1_homogeneity_std
                glcm_d1_energy_std
                glcm_d1_correlation_std
                glcm_d1_entropy_std
    """
    gray = rgb_to_gray(image)
    features = compute_glcm_features(gray, levels=16, distances=(1, 2, 4))
    return features


def extract_branch_c_lbp_edge_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract Canny edge-guided LBP candidate features (C_LBP_EDGE alternative).

    Pipeline:
        RGB [3, 256, 256] -> Grayscale -> Gaussian pre-filter (sigma=1.0, 5x5) -> Canny edge detection (low=50, high=100)
        -> Uniform LBP map -> Edge-masked 10-bin histogram + 5 statistics + edge density.

    Args:
        image: Canonical image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of exactly 16 scalar float features:
            - lbp_edge_bin_0 to lbp_edge_bin_8
            - lbp_edge_bin_nonuniform
            - lbp_edge_entropy
            - lbp_edge_uniformity
            - lbp_edge_dominant_bin
            - lbp_edge_mean_code
            - lbp_edge_std_code
            - lbp_edge_pixel_density (fraction of edge pixels in 256x256 image)
    """
    gray = rgb_to_gray(image)
    edge_mask = compute_canny_edges(gray, low_threshold=50.0, high_threshold=100.0, sigma=1.0)
    lbp_map = compute_lbp_map(gray, p=8, r=1)
    features = extract_lbp_statistics(lbp_map, mask=edge_mask, prefix="lbp_edge")
    return features
