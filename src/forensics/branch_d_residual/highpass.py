"""Gaussian high-pass residual feature extraction (Block 2 — Branch D candidate: D_HIGHPASS).

Mathematical Definition
-----------------------
Input:
    Canonical RGB image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

Grayscale conversion (ITU-R BT.601):
    Y = 0.299 * R + 0.587 * G + 0.114 * B,  Y in [0, 1]

High-pass residual:
    R_HP = Y - GaussianBlur(Y)

where GaussianBlur uses an exact 5×5 Gaussian kernel with sigma=1.0 and
replicate boundary padding (1-pixel border on each side).

Output residual R_HP is signed with approximate range [-1, 1].  It is
zero for constant-valued inputs because the Gaussian blur of a constant
is the constant itself.

Candidate Statistics (5 scalars — prefix: hp_*)
-------------------------------------------------
    hp_absmean   : mean(|R_HP|)                — average residual amplitude
    hp_std       : std(R_HP)                   — residual standard deviation
    hp_energy    : mean(R_HP²)                 — mean squared residual energy
    hp_kurtosis  : Fisher excess kurtosis of R_HP distribution
    hp_entropy   : Shannon entropy (256-bin histogram) of R_HP

All statistics are defined as CANDIDATES.  The final retained subset is
determined experimentally in Block 4 using training-data-only analysis.
Feature counts and subset may change after Block 4.

Forensic Hypothesis
-------------------
Real camera images contain characteristic high-frequency residuals arising from
optical PSF, CFA demosaicing, and JPEG quantization ringing.  AI-generated images
produced by iterative synthesis (diffusion, GAN, auto-regressive) may have
statistically different high-frequency residual distributions.

This candidate may overlap partially with Branch B wavelet level-1 detail subband
statistics.  That overlap is acknowledged and must be empirically tested in Block 4.
D_HIGHPASS and D_LAPLACIAN are both linear residual operators and may be redundant
with each other.  Do not concatenate them by default without Block 4 evidence.

Numerical Safety
----------------
All statistics guard against zero-variance inputs (constant images):
    - kurtosis returns 0.0 when std < 1e-12
    - entropy returns 0.0 when input range < 1e-12

Boundary Convention
-------------------
Padding: replicate (border pixels are repeated at the edge).
This avoids introducing artificial high-frequency artifacts at image borders
that would be caused by zero-padding.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gaussian kernel construction
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Compute a 1D normalized Gaussian kernel.

    Args:
        size: Kernel length (must be odd and positive).
        sigma: Standard deviation of the Gaussian.

    Returns:
        1D float32 tensor of shape [size], normalized to sum 1.0.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def _build_gaussian_kernel_2d(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Build a 2D separable Gaussian kernel from two 1D kernels.

    Args:
        size: Kernel side length (default: 5).
        sigma: Gaussian standard deviation (default: 1.0).

    Returns:
        2D float32 tensor of shape [1, 1, size, size].
    """
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d[:, None] * k1d[None, :]          # outer product
    return k2d.view(1, 1, size, size)


# Module-level constant — kernel is deterministic, computed once.
# Exact parameters: 5×5, sigma=1.0.  Do NOT change these without updating
# the module docstring, tests, and CHANGELOG.
_GAUSSIAN_KERNEL_5x5_SIGMA1: torch.Tensor = _build_gaussian_kernel_2d(size=5, sigma=1.0)


# ---------------------------------------------------------------------------
# Numerical safety helpers
# ---------------------------------------------------------------------------

def _safe_kurtosis(flat: torch.Tensor) -> float:
    """Compute Fisher excess kurtosis (mu4/sigma^4 - 3) safely.

    Returns 0.0 when standard deviation is below 1e-12 (constant input).

    Args:
        flat: 1D float32 tensor.

    Returns:
        Scalar float. 0.0 for near-constant inputs.
    """
    flat = flat.float()
    mu = flat.mean()
    sigma = flat.std()
    if float(sigma) < 1e-12:
        return 0.0
    z = (flat - mu) / sigma
    return float(z.pow(4).mean()) - 3.0


def _safe_entropy(x: torch.Tensor, bins: int = 256) -> float:
    """Compute histogram-based Shannon entropy safely (base 2, in bits).

    Returns 0.0 for constant-valued tensors.

    Args:
        x: Tensor of arbitrary shape.
        bins: Number of histogram bins.

    Returns:
        Scalar float >= 0.0.
    """
    flat = x.reshape(-1).float()
    lo = flat.min()
    hi = flat.max()
    if float((hi - lo).abs()) < 1e-12:
        return 0.0
    hist = torch.histc(flat, bins=bins, min=float(lo), max=float(hi))
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_highpass_residual(image: torch.Tensor) -> torch.Tensor:
    """Compute the Gaussian high-pass residual map from a canonical image tensor.

    Converts to grayscale, applies a 5×5 Gaussian blur (sigma=1.0, replicate
    padding), then returns the signed residual R_HP = Y - GaussianBlur(Y).

    Args:
        image: Canonical RGB tensor [3, H, W] or [B, 3, H, W], float32 in [0, 1].

    Returns:
        Signed residual tensor [1, H, W] or [B, 1, H, W], float32.
        Shape is the same spatial size as input.
    """
    # Ensure batch dimension
    if image.dim() == 3:
        img = image.unsqueeze(0)           # [1, 3, H, W]
        squeeze_out = True
    else:
        img = image
        squeeze_out = False

    # Grayscale conversion — ITU-R BT.601
    gray = (
        0.299 * img[:, 0:1, :, :]
        + 0.587 * img[:, 1:2, :, :]
        + 0.114 * img[:, 2:3, :, :]
    )                                       # [B, 1, H, W]

    # Gaussian blur with replicate padding
    kernel = _GAUSSIAN_KERNEL_5x5_SIGMA1.to(device=gray.device, dtype=gray.dtype)
    # Replicate-pad 2 pixels on each side for the 5×5 kernel
    padded = F.pad(gray, (2, 2, 2, 2), mode="replicate")
    blurred = F.conv2d(padded, kernel, stride=1, padding=0)

    residual = gray - blurred               # [B, 1, H, W]

    if squeeze_out:
        residual = residual.squeeze(0)      # [1, H, W]

    return residual


def extract_highpass_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract candidate D_HIGHPASS scalar statistics from the Gaussian high-pass residual.

    Candidate statistics (5 scalars, prefix: hp_*):
        hp_absmean  : mean(|R_HP|)
        hp_std      : std(R_HP)
        hp_energy   : mean(R_HP^2)
        hp_kurtosis : Fisher excess kurtosis of R_HP
        hp_entropy  : Shannon entropy (256-bin) of R_HP

    NOTE: These are CANDIDATE statistics.  The final retained subset is
    determined in Block 4 using training-data-only analysis.  Feature count
    and subset may change.

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of 5 scalar float features in deterministic order.
        All values are finite (no NaN, no Inf).
    """
    residual = compute_highpass_residual(image)   # [1, H, W] or [B, 1, H, W]
    flat = residual.reshape(-1).detach().cpu()

    features: Dict[str, float] = {}
    features["hp_absmean"]  = float(flat.abs().mean())
    features["hp_std"]      = float(flat.std())
    features["hp_energy"]   = float(flat.pow(2).mean())
    features["hp_kurtosis"] = _safe_kurtosis(flat)
    features["hp_entropy"]  = _safe_entropy(flat)

    return features
