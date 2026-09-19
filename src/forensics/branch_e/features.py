"""Branch E composition layer unifying candidate JPEG & compression-artifact forensic features.

Orchestrates the four candidate sub-branches:
    - E1_DCT    (10 features): 8×8 block DCT basis fingerprints and zonal ratios
    - E2_RESP   (8 features) : Controlled multi-quality in-memory JPEG recompression response
    - E3_PHASE  (4 features) : Fourier phase spectrum stability under controlled JPEG recompression
    - E4_GRID   (4 features) : Canonical 8×8 grid boundary step discontinuities

Total Unified Branch E: Exactly 26 scalar float features.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.forensics.branch_e.dct import extract_branch_e_dct_features
from src.forensics.branch_e.grid import extract_branch_e_grid_features
from src.forensics.branch_e.phase_stability import extract_branch_e_phase_features
from src.forensics.branch_e.recompression import extract_branch_e_recompression_features


def extract_branch_e_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract all 26 unified candidate Branch E forensic features from canonical image tensor.

    Sub-branches combined in deterministic order:
        E1_DCT (1..10):
            dct_ac_mean_abs, dct_ac_energy, dct_ac_kurtosis, dct_sparsity_ratio,
            dct_low_freq_ratio, dct_mid_freq_ratio, dct_high_freq_ratio,
            dct_anisotropy, dct_benford_ssd, dct_block_var_mean
        E2_RESP (11..18):
            ela_q95_mean, ela_q90_mean, ela_q75_mean, ela_q60_mean,
            ela_q90_energy, ela_slope_q90_q75, ela_ratio_q90_q75, ela_q90_gini
        E3_PHASE (19..22):
            phase_corr_q90, phase_corr_q75, phase_diff_energy_q90, phase_hf_stability_q90
        E4_GRID (23..26):
            grid_h_ratio, grid_v_ratio, grid_strength, grid_anisotropy

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of exactly 26 scalar float features with deterministic ordering.
        All values are guaranteed to be finite (no NaN, no Inf).
    """
    features: Dict[str, float] = {}

    # E1_DCT (10 features)
    features.update(extract_branch_e_dct_features(image))

    # E2_RESP (8 features)
    features.update(extract_branch_e_recompression_features(image))

    # E3_PHASE (4 features)
    features.update(extract_branch_e_phase_features(image))

    # E4_GRID (4 features)
    features.update(extract_branch_e_grid_features(image))

    return features
