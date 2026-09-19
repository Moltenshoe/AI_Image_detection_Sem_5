"""Forensic Feature Extraction Package (Block 2).

Exposes modular forensic branches and multi-branch orchestration pipeline for AI-generated image detection.
Currently implemented:
    - Branch A (Frequency / Periodicity)
    - Branch B (Wavelet / 3-level Haar DWT)
    - Branch C (Local Texture: C_LBP, C_GLCM, C_LBP_EDGE)
    - Branch D (Residual / Noise: D_HIGHPASS, D_LAPLACIAN, D_MFR)
"""

from __future__ import annotations

from src.forensics.branch_a_frequency import (
    compute_fft_diagnostics,
    cross_difference,
    extract_branch_a_features,
    frequency_radius,
    rgb_to_gray,
    standard_fft_features,
    synthbuster_periodicity_features,
)
from src.forensics.branch_b_wavelet import (
    extract_branch_b_features,
    haar_2d_level,
    haar_dwt_3level,
    safe_entropy,
)
from src.forensics.branch_c_texture import (
    compute_canny_edges,
    compute_glcm_features,
    compute_glcm_matrix,
    compute_lbp_map,
    extract_branch_c_glcm_features,
    extract_branch_c_lbp_edge_features,
    extract_branch_c_lbp_features,
    extract_lbp_statistics,
    quantize_grayscale,
)
from src.forensics.branch_d_residual import (
    compute_highpass_residual,
    compute_laplacian_residual,
    compute_median_filter_residual,
    extract_branch_d_highpass_features,
    extract_branch_d_laplacian_features,
    extract_branch_d_mfr_features,
    extract_highpass_features,
    extract_laplacian_features,
    extract_mfr_features,
)
from src.forensics.branch_e import (
    compute_block_dct,
    compute_grid_discontinuities,
    compute_recompression_error_map,
    extract_branch_e_dct_features,
    extract_branch_e_features,
    extract_branch_e_grid_features,
    extract_branch_e_phase_features,
    extract_branch_e_recompression_features,
)
from src.forensics.pipeline import DEFAULT_BRANCHES, ForensicPipeline

__all__ = [
    "ForensicPipeline",
    "DEFAULT_BRANCHES",
    # Branch A
    "extract_branch_a_features",
    "standard_fft_features",
    "synthbuster_periodicity_features",
    "cross_difference",
    "frequency_radius",
    "rgb_to_gray",
    "compute_fft_diagnostics",
    # Branch B
    "extract_branch_b_features",
    "haar_2d_level",
    "haar_dwt_3level",
    "safe_entropy",
    # Branch C
    "extract_branch_c_lbp_features",
    "extract_branch_c_glcm_features",
    "extract_branch_c_lbp_edge_features",
    "compute_lbp_map",
    "compute_canny_edges",
    "extract_lbp_statistics",
    "quantize_grayscale",
    "compute_glcm_matrix",
    "compute_glcm_features",
    # Branch D
    "extract_branch_d_highpass_features",
    "extract_branch_d_laplacian_features",
    "extract_branch_d_mfr_features",
    "compute_highpass_residual",
    "compute_laplacian_residual",
    "compute_median_filter_residual",
    "extract_highpass_features",
    "extract_laplacian_features",
    "extract_mfr_features",
    # Branch E
    "extract_branch_e_features",
    "extract_branch_e_dct_features",
    "extract_branch_e_recompression_features",
    "extract_branch_e_phase_features",
    "extract_branch_e_grid_features",
    "compute_block_dct",
    "compute_recompression_error_map",
    "compute_grid_discontinuities",
]
