"""Branch A — Lab Local Correlation Maps (CHROMA-inspired).

Public API:
    - extract_branch_a_lab_maps: High-level extractor returning [3, H, W] tensor.
    - rgb_to_cielab: Color space converter RGB -> CIE Lab.
    - compute_local_correlation_maps: Vectorized 7x7 Pearson sliding window correlation.
"""

from .color_conversion import rgb_to_cielab
from .correlation import compute_local_correlation_maps
from .features import extract_branch_a_lab_maps

__all__ = [
    "extract_branch_a_lab_maps",
    "rgb_to_cielab",
    "compute_local_correlation_maps",
]
