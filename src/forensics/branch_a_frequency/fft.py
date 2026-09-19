"""Standard Fourier / Frequency analysis module (Block 2 — Branch A1).

Provides modular, deterministic 2D FFT feature extraction on canonical image tensors.
Extracts normalized radial frequency energy bands and spectral centroid from grayscale representations.

Reference implementation refactored from historical exploratory notebook analysis_800_v1.ipynb.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch


def frequency_radius(
    h: int,
    w: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute a 2D normalized frequency radius grid centered at (h//2, w//2).

    Radius values are normalized such that the maximum frequency corner is 1.0:
        max_radius = sqrt(0.5^2 + 0.5^2) = sqrt(0.5)

    Args:
        h: Height of the spatial/frequency grid.
        w: Width of the spatial/frequency grid.
        device: PyTorch device on which to construct the grid.
        dtype: PyTorch float dtype (defaults to torch.float32).

    Returns:
        torch.Tensor of shape [h, w] with values in [0.0, 1.0].
    """
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=device, dtype=dtype))
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=device, dtype=dtype))
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    max_radius = math.sqrt(0.5**2 + 0.5**2)
    return torch.sqrt(xx.square() + yy.square()) / max_radius


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor to grayscale using standard luminance weights.

    Formula: 0.299 * R + 0.587 * G + 0.114 * B (ITU-R BT.601 standard weights).

    Args:
        image: Tensor of shape [..., 3, H, W] or [..., 1, H, W].

    Returns:
        Grayscale tensor of shape [..., 1, H, W].
    """
    if image.shape[-3] == 1:
        return image
    if image.shape[-3] != 3:
        raise ValueError(
            f"Expected image with 1 or 3 channels at dim -3, got shape {tuple(image.shape)}"
        )
    return (
        0.299 * image[..., 0:1, :, :]
        + 0.587 * image[..., 1:2, :, :]
        + 0.114 * image[..., 2:3, :, :]
    )


def standard_fft_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract standard Fourier / frequency scalar detector features (Branch A1).

    Pipeline:
        1. Convert RGB to grayscale via standard luminance weights.
        2. Remove DC component (spatial mean subtraction).
        3. 2D FFT with orthonormal normalization (`torch.fft.fft2(..., norm="ortho")`).
        4. Quadrant shift to center zero frequency (`torch.fft.fftshift`).
        5. Compute power spectrum and normalized frequency radius.
        6. Compute radial energy band ratios (low < 0.20, mid [0.20, 0.50), high >= 0.50)
           and spectral centroid.

    Args:
        image: Canonical image tensor, shape [3, H, W] or [B, 3, H, W], float32 in [0, 1].

    Returns:
        Dictionary of exactly 4 scalar float features:
            - fft_low_freq_ratio
            - fft_mid_freq_ratio
            - fft_high_freq_ratio
            - fft_spectral_centroid
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")

    orig_ndim = image.ndim
    if orig_ndim == 3:
        x_in = image.unsqueeze(0)
    elif orig_ndim == 4:
        x_in = image
    else:
        raise ValueError(
            f"Expected 3D [C, H, W] or 4D [B, C, H, W] tensor, got ndim={orig_ndim}"
        )

    gray = rgb_to_gray(x_in)
    h, w = gray.shape[-2:]

    # Remove DC component from the spatial image before measuring spectral structure
    x_centered = gray - gray.mean(dim=(-2, -1), keepdim=True)

    # 2D Orthonormal FFT and centering
    spectrum = torch.fft.fft2(x_centered, norm="ortho")
    shifted = torch.fft.fftshift(spectrum, dim=(-2, -1))

    magnitude = shifted.abs()
    power = magnitude.square()

    radius = frequency_radius(h, w, device=gray.device, dtype=gray.dtype)
    radius_expanded = radius.view(1, 1, h, w)

    # Numerical safety: total power denominator
    total_power = power.sum(dim=(-2, -1), keepdim=True)
    denom = total_power + 1e-12

    # Band masks
    low_mask = (radius_expanded < 0.20).float()
    mid_mask = ((radius_expanded >= 0.20) & (radius_expanded < 0.50)).float()
    high_mask = (radius_expanded >= 0.50).float()

    low_ratio = (power * low_mask).sum(dim=(-2, -1), keepdim=True) / denom
    mid_ratio = (power * mid_mask).sum(dim=(-2, -1), keepdim=True) / denom
    high_ratio = (power * high_mask).sum(dim=(-2, -1), keepdim=True) / denom
    centroid = (power * radius_expanded).sum(dim=(-2, -1), keepdim=True) / denom

    return {
        "fft_low_freq_ratio": float(low_ratio[0, 0, 0, 0].detach().cpu()),
        "fft_mid_freq_ratio": float(mid_ratio[0, 0, 0, 0].detach().cpu()),
        "fft_high_freq_ratio": float(high_ratio[0, 0, 0, 0].detach().cpu()),
        "fft_spectral_centroid": float(centroid[0, 0, 0, 0].detach().cpu()),
    }


def compute_fft_diagnostics(
    image: torch.Tensor, radial_bins: int = 64
) -> Dict[str, torch.Tensor]:
    """Compute diagnostic 2D FFT maps and 1D radial power spectrum.

    NOTE: Diagnostic maps are strictly for visualization and debugging.
    They are NOT detector features and must NOT enter the classifier pipeline.

    Args:
        image: Canonical image tensor [3, H, W] or [1, 3, H, W].
        radial_bins: Number of radial bins for 1D power spectrum accumulation.

    Returns:
        Dictionary containing:
            - fft_logmag_map: log(1 + magnitude) 2D tensor [1, 1, H, W]
            - fft_phase_map: angle/phase 2D tensor [1, 1, H, W] in [-pi, pi]
            - fft_radial_power: 1D radial power profile [radial_bins]
            - fft_radial_r: 1D radial bin coordinates [radial_bins]
    """
    orig_ndim = image.ndim
    if orig_ndim == 3:
        x_in = image.unsqueeze(0)
    elif orig_ndim == 4:
        x_in = image
    else:
        raise ValueError(
            f"Expected 3D [C, H, W] or 4D [B, C, H, W] tensor, got ndim={orig_ndim}"
        )

    gray = rgb_to_gray(x_in)
    h, w = gray.shape[-2:]

    x_centered = gray - gray.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fft2(x_centered, norm="ortho")
    shifted = torch.fft.fftshift(spectrum, dim=(-2, -1))

    magnitude = shifted.abs()
    power = magnitude.square()

    radius = frequency_radius(h, w, device=gray.device, dtype=gray.dtype).view(1, 1, h, w)
    total = power.sum() + 1e-12

    # Vectorized radial accumulation (from historical notebook)
    bin_index = torch.clamp(
        (radius * radial_bins).long(),
        min=0,
        max=radial_bins - 1,
    )
    flat_bins = bin_index.expand_as(power).reshape(-1)
    flat_power = power.reshape(-1)
    radial_sum = torch.zeros(
        radial_bins,
        device=gray.device,
        dtype=power.dtype,
    )
    radial_sum.scatter_add_(0, flat_bins, flat_power)
    radial_power = radial_sum / total

    radial_r = torch.linspace(
        0.5 / radial_bins,
        1.0 - 0.5 / radial_bins,
        radial_bins,
        device=gray.device,
        dtype=power.dtype,
    )

    return {
        "fft_logmag_map": torch.log1p(magnitude),
        "fft_phase_map": torch.angle(shifted),
        "fft_radial_power": radial_power,
        "fft_radial_r": radial_r,
    }
