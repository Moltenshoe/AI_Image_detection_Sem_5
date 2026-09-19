"""Branch E — JPEG & Compression-Artifact Forensics Package.

Exports:
    - extract_branch_e_features: Unified 26-feature extractor (E1 + E2 + E3 + E4).
    - extract_branch_e_dct_features: E1_DCT 10-feature extractor.
    - extract_branch_e_recompression_features: E2_RESP 8-feature extractor.
    - extract_branch_e_phase_features: E3_PHASE 4-feature extractor.
    - extract_branch_e_grid_features: E4_GRID 4-feature extractor.
    - compute_block_dct: 2D 8×8 block DCT tensor computation.
    - compute_recompression_error_map: In-memory JPEG recompression error map computation.
    - compute_grid_discontinuities: Canonical 8×8 boundary vs interior step differences.
"""

from __future__ import annotations

from src.forensics.branch_e.dct import (
    compute_block_dct,
    extract_branch_e_dct_features,
)
from src.forensics.branch_e.features import extract_branch_e_features
from src.forensics.branch_e.grid import (
    compute_grid_discontinuities,
    extract_branch_e_grid_features,
)
from src.forensics.branch_e.phase_stability import extract_branch_e_phase_features
from src.forensics.branch_e.recompression import (
    compute_recompression_error_map,
    extract_branch_e_recompression_features,
)

__all__ = [
    "extract_branch_e_features",
    "extract_branch_e_dct_features",
    "extract_branch_e_recompression_features",
    "extract_branch_e_phase_features",
    "extract_branch_e_grid_features",
    "compute_block_dct",
    "compute_recompression_error_map",
    "compute_grid_discontinuities",
]
