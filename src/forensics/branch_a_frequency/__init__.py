"""Branch A — Frequency / Periodicity Forensic Extraction Package (Block 2).

Exposes modular frequency-domain forensic tools:
- A1: Standard Fourier / Frequency analysis (`standard_fft_features`)
- A2: Synthbuster-inspired periodicity analysis (`synthbuster_periodicity_features`)
- Unified Branch A feature extraction (`extract_branch_a_features`)
"""

from __future__ import annotations

from src.forensics.branch_a_frequency.features import extract_branch_a_features
from src.forensics.branch_a_frequency.fft import (
    compute_fft_diagnostics,
    frequency_radius,
    rgb_to_gray,
    standard_fft_features,
)
from src.forensics.branch_a_frequency.synthbuster import (
    cross_difference,
    synthbuster_periodicity_features,
)

__all__ = [
    "extract_branch_a_features",
    "standard_fft_features",
    "synthbuster_periodicity_features",
    "cross_difference",
    "frequency_radius",
    "rgb_to_gray",
    "compute_fft_diagnostics",
]
