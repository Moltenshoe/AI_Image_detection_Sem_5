"""Local Binary Pattern (LBP) computation module (Block 2 — Branch C).

Provides core algorithms for:
1. Standard rotation-invariant uniform LBP computation (P=8, R=1).
2. Deterministic Canny edge detection on canonical grayscale images.
3. Masked/edge-guided LBP histogram and summary statistics extraction.

Mathematical & Implementation Specifications:
- Input: Grayscale tensor [..., 1, 256, 256] or [..., 256, 256] with float32 values in [0.0, 1.0].
- Radius & Neighbors: R = 1, P = 8 on a circular/8-connected grid.
- Replicate 1-pixel boundary padding to preserve exact (256, 256) spatial resolution.
- Shifts: 8 neighbors in order:
    (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)
- Binary threshold: s(g_p - g_c) = 1 if g_p >= g_c else 0.
- Raw code: sum_{k=0}^7 bit_k * (1 << k) in [0, 255].
- Circular bit transitions: U = sum_{k=0}^7 |bit_k - bit_{(k+1)%8}|.
- Uniform mapping (U <= 2):
    - If U <= 2, uniform bin index = sum_{k=0}^7 bit_k (number of 1s in pattern, in [0, 8]).
    - If U > 2, non-uniform bin index = 9 (P + 1).
- Total LBP bins: Exactly 10 bins (bins 0-8 for uniform patterns, bin 9 for non-uniform).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compute_lbp_map(gray: torch.Tensor, p: int = 8, r: int = 1) -> torch.Tensor:
    """Compute rotation-invariant uniform LBP code map.

    Args:
        gray: Grayscale tensor [..., 1, H, W] or [..., H, W], float32 in [0.0, 1.0].
        p: Number of circular neighbor sampling points (default: 8).
        r: Radius of circle (default: 1).

    Returns:
        Tensor of same spatial shape [..., H, W] with integer bin indices in [0, 9].
    """
    if p != 8 or r != 1:
        raise ValueError(f"Currently supports p=8, r=1; got p={p}, r={r}")

    # Squeeze channel dimension if present: [..., 1, H, W] -> [..., H, W]
    if gray.ndim >= 3 and gray.shape[-3] == 1:
        x = gray.squeeze(-3)
    else:
        x = gray

    # Ensure 3D or 2D tensor
    is_batched = x.ndim > 2
    if not is_batched:
        x = x.unsqueeze(0)  # [1, H, W]

    b, h, w = x.shape
    x_clamped = (x.clamp(0.0, 1.0) * 255.0).float()

    # Replicate pad 1 pixel on all sides: [B, H+2, W+2]
    padded = F.pad(x_clamped.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    center = padded[:, 1:-1, 1:-1]

    # 8 neighbor shifts: (dy, dx)
    shifts = [
        (-1, -1), (-1, 0), (-1, 1), (0, 1),
        (1, 1), (1, 0), (1, -1), (0, -1),
    ]

    # Extract 8 neighbor bit values: [8, B, H, W]
    neighbor_bits = []
    for dy, dx in shifts:
        nb = padded[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        bit = (nb >= center).to(torch.int64)
        neighbor_bits.append(bit)

    # Stack neighbor bits along dimension 0: [8, B, H, W]
    bits = torch.stack(neighbor_bits, dim=0)

    # Compute circular bit transitions U: |bit_k - bit_{(k+1)%8}|
    rolled_bits = torch.roll(bits, shifts=-1, dims=0)
    transitions = (bits != rolled_bits).to(torch.int64).sum(dim=0)  # [B, H, W]

    # Uniform mapping:
    # If transitions <= 2 -> number of 1-bits (sum across dim 0 in [0, 8])
    # If transitions > 2 -> bin 9 (P + 1)
    ones_count = bits.sum(dim=0)  # [B, H, W], values in [0, 8]
    uniform_map = torch.where(
        transitions <= 2,
        ones_count,
        torch.full_like(transitions, 9, dtype=torch.int64),
    )

    if not is_batched:
        uniform_map = uniform_map.squeeze(0)

    return uniform_map


def compute_canny_edges(
    gray: torch.Tensor,
    low_threshold: float = 50.0,
    high_threshold: float = 100.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """Compute deterministic edge mask using Gaussian smoothing pre-filter + Canny edge detection.

    Steps:
        1. Grayscale luminance in [0.0, 1.0] scaled to uint8 in [0, 255].
        2. Gaussian smoothing pre-filter with sigma=1.0 (kernel size 5x5).
        3. Canny edge detection with hysteresis thresholds [low_threshold, high_threshold].

    Args:
        gray: Grayscale tensor [..., 1, H, W] or [..., H, W], float32 in [0.0, 1.0].
        low_threshold: Canny lower hysteresis threshold in [0, 255] (default: 50.0).
        high_threshold: Canny upper hysteresis threshold in [0, 255] (default: 100.0).
        sigma: Standard deviation for Gaussian smoothing pre-filter (default: 1.0).

    Returns:
        Binary numpy array of shape [H, W] where 1 indicates an edge pixel, 0 non-edge.
    """
    if gray.ndim >= 3 and gray.shape[-3] == 1:
        img_tensor = gray.squeeze(-3)
    else:
        img_tensor = gray

    if img_tensor.ndim > 2:
        img_tensor = img_tensor[0]  # Take first sample if batched

    # Convert to uint8 numpy image in [0, 255]
    img_np = (img_tensor.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)

    # Apply Gaussian smoothing if sigma > 0
    if sigma > 0:
        ksize = int(2 * round(2 * sigma) + 1)
        blurred = cv2.GaussianBlur(img_np, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    else:
        blurred = img_np

    edges = cv2.Canny(
        blurred,
        threshold1=low_threshold,
        threshold2=high_threshold,
        L2gradient=False,
    )

    # Convert to binary mask {0, 1}
    binary_mask = (edges > 0).astype(np.uint8)
    return binary_mask


def extract_lbp_statistics(
    lbp_map: torch.Tensor,
    mask: Optional[np.ndarray] = None,
    prefix: str = "lbp",
) -> Dict[str, float]:
    """Extract normalized 10-bin histogram and scalar summary statistics from LBP map.

    Args:
        lbp_map: Tensor with integer values in [0, 9] (rotation-invariant uniform codes).
        mask: Optional binary mask [H, W] (e.g. Canny edge mask). If provided, LBP
              histogram is computed strictly over pixels where mask == 1.
        prefix: Prefix for dictionary keys (default: 'lbp', or 'lbp_edge').

    Returns:
        Dictionary of scalar float features:
            - {prefix}_bin_0 to {prefix}_bin_8
            - {prefix}_bin_nonuniform
            - {prefix}_entropy
            - {prefix}_uniformity
            - {prefix}_dominant_bin
            - {prefix}_mean_code
            - {prefix}_std_code
            - {prefix}_nonuniform_ratio (for standard LBP) OR {prefix}_pixel_density (for edge LBP)
    """
    flat_codes = lbp_map.reshape(-1)

    if mask is not None:
        mask_flat = torch.from_numpy(mask).to(device=lbp_map.device).reshape(-1)
        valid_codes = flat_codes[mask_flat > 0]
        n_edge_pixels = int(valid_codes.numel())
        total_pixels = int(flat_codes.numel())
        edge_density = float(n_edge_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

        # Numerical safety: if edge mask is sparse (< 10 edge pixels) or empty
        if n_edge_pixels < 10:
            # Fallback to uniform distribution over 10 bins (0.1 each)
            hist = torch.full((10,), 0.1, dtype=torch.float32, device=lbp_map.device)
            # Codes for mean/std fallback: uniformly distributed 0..9
            valid_codes = torch.arange(10, dtype=torch.float32, device=lbp_map.device)
        else:
            hist = torch.bincount(valid_codes.long(), minlength=10)[:10].float()
            hist = hist / (hist.sum() + 1e-12)
    else:
        n_edge_pixels = None
        edge_density = None
        valid_codes = flat_codes
        hist = torch.bincount(flat_codes.long(), minlength=10)[:10].float()
        hist = hist / (hist.sum() + 1e-12)

    # 1. 10 normalized histogram bins
    features: Dict[str, float] = {}
    for i in range(9):
        features[f"{prefix}_bin_{i}"] = float(hist[i].detach().cpu())
    features[f"{prefix}_bin_nonuniform"] = float(hist[9].detach().cpu())

    # 2. Shannon Entropy of 10-bin histogram in bits
    p = hist[hist > 0]
    entropy = float(-(p * torch.log2(p)).sum().detach().cpu())
    features[f"{prefix}_entropy"] = entropy

    # 3. Uniformity / Energy (sum of squared probabilities)
    features[f"{prefix}_uniformity"] = float((hist.square()).sum().detach().cpu())

    # 4. Dominant bin probability
    features[f"{prefix}_dominant_bin"] = float(hist.max().detach().cpu())

    # 5. Mean code of sampled pixels
    features[f"{prefix}_mean_code"] = float(valid_codes.float().mean().detach().cpu())

    # 6. Standard deviation of sampled codes
    features[f"{prefix}_std_code"] = float(valid_codes.float().std().detach().cpu())

    # 7. Non-uniform ratio (standard LBP) or Edge density (edge LBP)
    if mask is not None:
        features[f"{prefix}_pixel_density"] = edge_density
    else:
        # Explicit non-uniform ratio
        features[f"{prefix}_nonuniform_ratio"] = float(hist[9].detach().cpu())

    return features
