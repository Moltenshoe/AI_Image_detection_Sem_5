"""Wavelet-domain feature extraction module (Block 2 — Branch B).

Extracts 30 multiscale spatial-frequency scalar features using a 3-level 2D Haar DWT
on canonical grayscale image representations.

Mathematical Definitions:
1. Grayscale Conversion:
   Canonical RGB [3, 256, 256] -> Grayscale [1, 256, 256] via ITU-R BT.601:
       Gray = 0.299 * R + 0.587 * G + 0.114 * B

2. Multiscale 2D Haar DWT:
   3-level decomposition yielding 10 subbands:
       LL3 (32x32)
       LH3, HL3, HH3 (32x32)
       LH2, HL2, HH2 (64x64)
       LH1, HL1, HH1 (128x128)

3. Subband Statistics (29 features):
   For approximation subband LL3:
       LL3_energy  = mean(LL3^2)
       LL3_entropy = safe_entropy(LL3)
   For each detail subband S in (LH3, HL3, HH3, LH2, HL2, HH2, LH1, HL1, HH1):
       {S}_energy  = mean(S^2)
       {S}_absmean = mean(|S|)
       {S}_entropy = safe_entropy(S)

4. Global Detail Energy Ratio (1 feature):
   Numerator (Total Detail Energy):
       total_detail_energy = sum(energy(S) for S in [LH3, HL3, HH3, LH2, HL2, HH2, LH1, HL1, HH1])
   Denominator (Total Energy):
       total_energy = energy(LL3) + total_detail_energy
   Ratio:
       detail_energy_ratio = total_detail_energy / (total_energy + 1e-12)

Total: Exactly 30 scalar float features.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.forensics.branch_b_wavelet.haar import haar_dwt_3level


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


def safe_entropy(x: torch.Tensor, bins: int = 256) -> float:
    """Compute histogram-based Shannon entropy safely.

    Calculates Shannon entropy in bits (base 2) across the dynamic range [min(x), max(x)].
    Guards against zero dynamic range (constant inputs) and zero probabilities.

    Args:
        x: Tensor of arbitrary shape.
        bins: Number of histogram bins (default: 256).

    Returns:
        Scalar float entropy in bits >= 0.0. Returns 0.0 for constant inputs.
    """
    flat = x.reshape(-1).float()
    lo = flat.min()
    hi = flat.max()

    if float((hi - lo).abs()) < 1e-12:
        return 0.0

    hist = torch.histc(flat, bins=bins, min=float(lo), max=float(hi))
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum().detach().cpu())


def extract_branch_b_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract all 30 scalar detector features from Branch B (Wavelet / Haar DWT).

    Args:
        image: Canonical image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of exactly 30 scalar float features in deterministic order:
            - LL3_energy, LL3_entropy
            - For each level l in (3, 2, 1) and orientation band in (LH, HL, HH):
                {band}{l}_energy, {band}{l}_absmean, {band}{l}_entropy
            - detail_energy_ratio
    """
    gray = rgb_to_gray(image)
    subbands = haar_dwt_3level(gray)

    features: Dict[str, float] = {}

    # 1. Approximation subband LL3 (2 scalars)
    ll3 = subbands["LL3"]
    ll3_energy = float(ll3.square().mean().detach().cpu())
    features["LL3_energy"] = ll3_energy
    features["LL3_entropy"] = safe_entropy(ll3)

    # 2. Detail subbands (27 scalars: 9 subbands x 3 statistics)
    detail_bands = [
        "LH3", "HL3", "HH3",
        "LH2", "HL2", "HH2",
        "LH1", "HL1", "HH1",
    ]

    total_detail_energy = 0.0
    for band_name in detail_bands:
        band = subbands[band_name]
        band_energy = float(band.square().mean().detach().cpu())
        band_absmean = float(band.abs().mean().detach().cpu())
        band_entropy = safe_entropy(band)

        features[f"{band_name}_energy"] = band_energy
        features[f"{band_name}_absmean"] = band_absmean
        features[f"{band_name}_entropy"] = band_entropy

        total_detail_energy += band_energy

    # 3. Global detail energy ratio (1 scalar)
    total_energy = ll3_energy + total_detail_energy
    features["detail_energy_ratio"] = total_detail_energy / (total_energy + 1e-12)

    return features
