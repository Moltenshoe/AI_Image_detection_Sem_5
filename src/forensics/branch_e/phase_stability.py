"""Compression-stable Fourier phase feature extraction (Block 2 — Branch E candidate: E3_PHASE).

Mathematical & Forensic Formulation
-----------------------------------
Input:
    Canonical RGB image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

Grayscale conversion (ITU-R BT.601):
    Y = 0.299 * R + 0.587 * G + 0.114 * B,  Y in [0, 1]

Fourier Phase Spectrum Extraction:
    For clean grayscale Y and recompressed Y_Q at quality Q in {90, 75}:
        F_clean = torch.fft.fft2(Y, norm="ortho")
        F_Q     = torch.fft.fft2(Y_Q, norm="ortho")
        Phi_clean = torch.angle(F_clean)
        Phi_Q     = torch.angle(F_Q)

Phase Angular Difference & Wrapping:
    The angular difference delta_Q is wrapped to [-pi, pi]:
        delta_Q = atan2(sin(Phi_clean - Phi_Q), cos(Phi_clean - Phi_Q))

Zero-Magnitude Masking:
    For bins with |F_clean| < 1e-8, phase is numerically ill-defined.
    These bins are masked out when computing phase correlations and differences.

Candidate Features (4 scalars — prefix: phase_*):
-------------------------------------------------
    19. phase_corr_q90          : Phase cosine similarity mean(cos(delta_90)) over valid bins in [-1, 1].
    20. phase_corr_q75          : Phase cosine similarity mean(cos(delta_75)) over valid bins in [-1, 1].
    21. phase_diff_energy_q90   : Mean squared angular difference mean(delta_90^2) in [0, pi^2].
    22. phase_hf_stability_q90  : Phase cosine similarity restricted to high-frequency radius (r >= 0.5 r_max).

CRITICAL SCIENTIFIC QUALIFICATIONS
----------------------------------
1. Compression-Response Metric: Evaluates phase stability under controlled in-memory recompression,
   distinguishing it from static Branch A Fourier magnitude/periodicity features.
2. Numerical Safety: Bounded in [-1, 1] and [0, pi^2]; constant/zero inputs return exact safe limits.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from src.forensics.branch_e.recompression import compute_recompression_error_map


# ---------------------------------------------------------------------------
# Frequency Radius Grid Helper
# ---------------------------------------------------------------------------

def _frequency_radius_grid(
    h: int, w: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Compute normalized 2D radial frequency grid [0, 1] centered at (H//2, W//2)."""
    y = torch.arange(h, device=device, dtype=dtype) - h // 2
    x = torch.arange(w, device=device, dtype=dtype) - w // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(yy ** 2 + xx ** 2)
    max_r = math.sqrt((h // 2) ** 2 + (w // 2) ** 2)
    return r / (max_r + 1e-12)


# ---------------------------------------------------------------------------
# Public Extraction Function
# ---------------------------------------------------------------------------

def extract_branch_e_phase_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract 4 candidate E3_PHASE scalar features across Q in {90, 75}.

    Features:
        19. phase_corr_q90
        20. phase_corr_q75
        21. phase_diff_energy_q90
        22. phase_hf_stability_q90

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of exactly 4 scalar float features in deterministic order.
    """
    if image.dim() == 3:
        img = image.unsqueeze(0)
    else:
        img = image

    # Compute recompressed RGB tensors
    _, recomp_q90 = compute_recompression_error_map(img, quality=90)
    _, recomp_q75 = compute_recompression_error_map(img, quality=75)

    # Convert to grayscale (BT.601)
    def _to_gray(t: torch.Tensor) -> torch.Tensor:
        return 0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3]

    gray_clean = _to_gray(img)
    gray_q90 = _to_gray(recomp_q90)
    gray_q75 = _to_gray(recomp_q75)

    # Compute 2D orthonormal FFTs
    f_clean = torch.fft.fftshift(torch.fft.fft2(gray_clean, norm="ortho"), dim=(-2, -1))
    f_q90 = torch.fft.fftshift(torch.fft.fft2(gray_q90, norm="ortho"), dim=(-2, -1))
    f_q75 = torch.fft.fftshift(torch.fft.fft2(gray_q75, norm="ortho"), dim=(-2, -1))

    mag_clean = f_clean.abs()
    valid_mask = mag_clean >= 1e-8  # Mask near-zero magnitude bins

    phi_clean = torch.angle(f_clean)
    phi_q90 = torch.angle(f_q90)
    phi_q75 = torch.angle(f_q75)

    # Wrapped angular differences in [-pi, pi]
    diff_q90 = torch.atan2(torch.sin(phi_clean - phi_q90), torch.cos(phi_clean - phi_q90))
    diff_q75 = torch.atan2(torch.sin(phi_clean - phi_q75), torch.cos(phi_clean - phi_q75))

    # Cosine similarities
    cos_q90 = torch.cos(diff_q90)
    cos_q75 = torch.cos(diff_q75)

    # Radial frequency mask for high-frequency region (r >= 0.5)
    B, C, H, W = gray_clean.shape
    r_grid = _frequency_radius_grid(H, W, device=gray_clean.device, dtype=gray_clean.dtype)
    hf_mask = (r_grid >= 0.5).unsqueeze(0).unsqueeze(0) & valid_mask

    # Extract scalar features safely
    valid_cos_q90 = cos_q90[valid_mask]
    valid_cos_q75 = cos_q75[valid_mask]
    valid_diff_q90 = diff_q90[valid_mask]
    hf_cos_q90 = cos_q90[hf_mask]

    phase_corr_q90 = float(valid_cos_q90.mean()) if len(valid_cos_q90) > 0 else 1.0
    phase_corr_q75 = float(valid_cos_q75.mean()) if len(valid_cos_q75) > 0 else 1.0
    phase_diff_energy_q90 = float(valid_diff_q90.pow(2).mean()) if len(valid_diff_q90) > 0 else 0.0
    phase_hf_stability_q90 = float(hf_cos_q90.mean()) if len(hf_cos_q90) > 0 else 1.0

    features: Dict[str, float] = {
        "phase_corr_q90": phase_corr_q90,
        "phase_corr_q75": phase_corr_q75,
        "phase_diff_energy_q90": phase_diff_energy_q90,
        "phase_hf_stability_q90": phase_hf_stability_q90,
    }

    return features
