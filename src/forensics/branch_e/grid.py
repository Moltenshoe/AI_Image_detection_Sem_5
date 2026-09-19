"""Canonical 8×8 grid boundary feature extraction (Block 2 — Branch E candidate: E4_GRID).

Mathematical & Forensic Formulation
-----------------------------------
Input:
    Canonical RGB image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

Grayscale conversion (ITU-R BT.601):
    Y = 0.299 * R + 0.587 * G + 0.114 * B,  Y in [0, 1]

Canonical Grid Boundary Discontinuity:
    For zero-indexed image Y(y, x) of size H × W (256 × 256):
    - Block boundaries occur across columns x = 8k - 1 and x = 8k for k in {1, ..., W/8 - 1}.
    - Block interior transitions occur across adjacent columns within the same block:
      x in {0, ..., W - 2} where (x mod 8) != 7.

    Boundary step differences:
        D_{h, bound} = mean_{k=1..31, y=0..255} |Y(y, 8k - 1) - Y(y, 8k)|
        D_{v, bound} = mean_{k=1..31, x=0..255} |Y(8k - 1, x) - Y(8k, x)|

    Interior step differences:
        D_{h, int}   = mean_{(x mod 8) != 7, y=0..255} |Y(y, x) - Y(y, x + 1)|
        D_{v, int}   = mean_{(y mod 8) != 7, x=0..255} |Y(y, x) - Y(y + 1, x)|

Candidate Features (4 scalars — prefix: grid_*):
------------------------------------------------
    23. grid_h_ratio    : D_{h, bound} / (D_{h, int} + 1e-6)
    24. grid_v_ratio    : D_{v, bound} / (D_{v, int} + 1e-6)
    25. grid_strength   : (D_{h, bound} + D_{v, bound}) / (D_{h, int} + D_{v, int} + 1e-6)
    26. grid_anisotropy : |D_{h, bound} - D_{v, bound}| / (D_{h, bound} + D_{v, bound} + 1e-6)

CRITICAL SCIENTIFIC QUALIFICATIONS
----------------------------------
1. Canonical Frame Interpretation: These features measure 8×8-aligned step discontinuities
   in the canonical 256×256 frame. Because preprocessing rescales original images, these
   are NOT claimed to be verified artifacts of the camera's original JPEG grid.
2. Numerical Safety: Safe epsilon addition prevents division by zero on uniform/constant images.
"""

from __future__ import annotations

from typing import Dict

import torch


def compute_grid_discontinuities(image: torch.Tensor) -> Dict[str, float]:
    """Compute boundary and interior step differences along horizontal and vertical axes.

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary containing dh_bound, dh_int, dv_bound, dv_int as float scalars.
    """
    if image.dim() == 3:
        img = image.unsqueeze(0)
    else:
        img = image

    # BT.601 grayscale
    gray = (
        0.299 * img[:, 0:1, :, :]
        + 0.587 * img[:, 1:2, :, :]
        + 0.114 * img[:, 2:3, :, :]
    )  # [B, 1, H, W]

    B, C, H, W = gray.shape

    # Horizontal step differences between column x and column x+1: shape [B, 1, H, W-1]
    diff_h = (gray[:, :, :, :-1] - gray[:, :, :, 1:]).abs()
    # Vertical step differences between row y and row y+1: shape [B, 1, H-1, W]
    diff_v = (gray[:, :, :-1, :] - gray[:, :, 1:, :]).abs()

    # Column indices 0..W-2
    x_indices = torch.arange(W - 1, device=gray.device)
    bound_x_mask = (x_indices % 8) == 7
    int_x_mask = (x_indices % 8) != 7

    # Row indices 0..H-2
    y_indices = torch.arange(H - 1, device=gray.device)
    bound_y_mask = (y_indices % 8) == 7
    int_y_mask = (y_indices % 8) != 7

    dh_bound = float(diff_h[:, :, :, bound_x_mask].mean())
    dh_int = float(diff_h[:, :, :, int_x_mask].mean())

    dv_bound = float(diff_v[:, :, bound_y_mask, :].mean())
    dv_int = float(diff_v[:, :, int_y_mask, :].mean())

    return {
        "dh_bound": dh_bound,
        "dh_int": dh_int,
        "dv_bound": dv_bound,
        "dv_int": dv_int,
    }


def extract_branch_e_grid_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract 4 candidate E4_GRID scalar features from canonical image tensor.

    Features:
        23. grid_h_ratio
        24. grid_v_ratio
        25. grid_strength
        26. grid_anisotropy

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of exactly 4 scalar float features in deterministic order.
    """
    disc = compute_grid_discontinuities(image)
    eps = 1e-6

    dh_bound = disc["dh_bound"]
    dh_int = disc["dh_int"]
    dv_bound = disc["dv_bound"]
    dv_int = disc["dv_int"]

    grid_h_ratio = dh_bound / (dh_int + eps)
    grid_v_ratio = dv_bound / (dv_int + eps)
    grid_strength = (dh_bound + dv_bound) / (dh_int + dv_int + eps)
    grid_anisotropy = abs(dh_bound - dv_bound) / (dh_bound + dv_bound + eps)

    features: Dict[str, float] = {
        "grid_h_ratio": grid_h_ratio,
        "grid_v_ratio": grid_v_ratio,
        "grid_strength": grid_strength,
        "grid_anisotropy": grid_anisotropy,
    }

    return features
