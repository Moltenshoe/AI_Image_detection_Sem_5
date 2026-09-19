"""Branch B — Wavelet feature extraction package (Block 2).

Exposes 3-level 2D Haar discrete wavelet transform and the 30-feature scalar extractor.
"""

from __future__ import annotations

from src.forensics.branch_b_wavelet.features import (
    extract_branch_b_features,
    rgb_to_gray,
    safe_entropy,
)
from src.forensics.branch_b_wavelet.haar import (
    haar_2d_level,
    haar_dwt_3level,
)

__all__ = [
    "extract_branch_b_features",
    "rgb_to_gray",
    "safe_entropy",
    "haar_2d_level",
    "haar_dwt_3level",
]
