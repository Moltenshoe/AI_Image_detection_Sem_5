"""Pure PyTorch 2D Haar Discrete Wavelet Transform (Block 2 — Branch B).

Implements separable 2D Haar wavelet decomposition directly using PyTorch tensor operations.
Does not depend on external wavelet libraries at runtime.

Haar Coefficient Convention:
For a 2D spatial block of 2x2 pixels:
    x00 = x[..., 0::2, 0::2]  (row even, col even)
    x01 = x[..., 0::2, 1::2]  (row even, col odd)
    x10 = x[..., 1::2, 0::2]  (row odd, col even)
    x11 = x[..., 1::2, 1::2]  (row odd, col odd)

The separable 2D orthonormal Haar basis scaled by 0.5 (1/sqrt(2) * 1/sqrt(2)) computes:
    LL = (x00 + x01 + x10 + x11) * 0.5   (Approximation: Low-Low)
    LH = (x00 - x01 + x10 - x11) * 0.5   (Horizontal detail: Highpass columns, Lowpass rows)
    HL = (x00 + x01 - x10 - x11) * 0.5   (Vertical detail: Lowpass columns, Highpass rows)
    HH = (x00 - x01 - x10 + x11) * 0.5   (Diagonal detail: Highpass columns, Highpass rows)

Sign-Invariance Note:
Third-party libraries (such as PyWavelets) may adopt alternative sign or axis filter ordering
conventions for detail bands (e.g. -x00 + x01). Because all downstream Branch B summary
statistics (energy, absmean, entropy) are strictly sign-invariant, feature-level outputs
are invariant to such sign differences.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def haar_2d_level(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one level of 2D separable Haar wavelet transform.

    Args:
        x: Tensor of shape [..., H, W] where H and W are even.

    Returns:
        Tuple of (LL, LH, HL, HH), each of shape [..., H//2, W//2].
    """
    if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
        raise ValueError(
            f"Spatial dimensions must be even for Haar transform, got shape {tuple(x.shape)}"
        )

    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5

    return ll, lh, hl, hh


def haar_dwt_3level(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute 3-level 2D Haar Discrete Wavelet Transform.

    Transforms a 256x256 image successively into:
        Level 1: 256x256 -> LL1, LH1, HL1, HH1 (each 128x128)
        Level 2: 128x128 (from LL1) -> LL2, LH2, HL2, HH2 (each 64x64)
        Level 3: 64x64 (from LL2) -> LL3, LH3, HL3, HH3 (each 32x32)

    Args:
        x: Tensor of shape [..., H, W] where H and W are multiples of 8 (e.g. 256x256).

    Returns:
        Dictionary mapping subband names to coefficient tensors:
            'LL3': [..., 32, 32]
            'LH3', 'HL3', 'HH3': [..., 32, 32]
            'LH2', 'HL2', 'HH2': [..., 64, 64]
            'LH1', 'HL1', 'HH1': [..., 128, 128]
    """
    ll1, lh1, hl1, hh1 = haar_2d_level(x)
    ll2, lh2, hl2, hh2 = haar_2d_level(ll1)
    ll3, lh3, hl3, hh3 = haar_2d_level(ll2)

    return {
        "LL3": ll3,
        "LH3": lh3,
        "HL3": hl3,
        "HH3": hh3,
        "LH2": lh2,
        "HL2": hl2,
        "HH2": hh2,
        "LH1": lh1,
        "HL1": hl1,
        "HH1": hh1,
    }
