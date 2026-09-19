"""Block 2 — Branch C: Local Texture Extraction Package.

Exposes independent, alternative local texture feature extractors:
- `extract_branch_c_lbp_features`: Standard rotation-invariant uniform LBP (16 features).
- `extract_branch_c_glcm_features`: Gray-Level Co-occurrence Matrix (24 features).
- `extract_branch_c_lbp_edge_features`: Edge-guided LBP candidate (16 features).
"""

from src.forensics.branch_c_texture.features import (
    extract_branch_c_glcm_features,
    extract_branch_c_lbp_edge_features,
    extract_branch_c_lbp_features,
    rgb_to_gray,
)
from src.forensics.branch_c_texture.glcm import (
    compute_glcm_features,
    compute_glcm_matrix,
    quantize_grayscale,
)
from src.forensics.branch_c_texture.lbp import (
    compute_canny_edges,
    compute_lbp_map,
    extract_lbp_statistics,
)

__all__ = [
    "extract_branch_c_lbp_features",
    "extract_branch_c_glcm_features",
    "extract_branch_c_lbp_edge_features",
    "rgb_to_gray",
    "compute_lbp_map",
    "compute_canny_edges",
    "extract_lbp_statistics",
    "quantize_grayscale",
    "compute_glcm_matrix",
    "compute_glcm_features",
]
