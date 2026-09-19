"""Branch A feature extraction composition layer (Block 2 — Frequency / Periodicity).

Combines A1 (Standard Fourier analysis) and A2 (Synthbuster-inspired periodicity analysis)
into the canonical 34-feature Branch A dictionary.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from src.forensics.branch_a_frequency.fft import standard_fft_features
from src.forensics.branch_a_frequency.synthbuster import synthbuster_periodicity_features


def extract_branch_a_features(
    image: torch.Tensor,
    periods: Sequence[int] = (2, 4, 8),
) -> Dict[str, float]:
    """Extract all 34 scalar detector features from Branch A (Frequency / Periodicity).

    Combines:
        - A1: Standard Fourier / Frequency features (4 scalars):
            fft_low_freq_ratio, fft_mid_freq_ratio, fft_high_freq_ratio, fft_spectral_centroid
        - A2: Synthbuster-inspired periodicity features (30 scalars):
            synth_{r,g,b}_p{2,4,8}_{x,y,d}_mean (27 scalars)
            synth_{r,g,b}_fft_highfreq_ratio (3 scalars)

    Args:
        image: Canonical RGB image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].
        periods: Synthbuster period sequence (default: (2, 4, 8)).

    Returns:
        Dictionary of exactly 34 scalar float features in deterministic order.
    """
    features: Dict[str, float] = {}
    features.update(standard_fft_features(image))
    features.update(synthbuster_periodicity_features(image, periods=periods))
    return features
