"""Branch D composition layer (Block 2 — Residual / Error / Noise).

Exposes three independent, alternative pixel-domain residual extraction pipelines:

1. extract_branch_d_highpass_features (D_HIGHPASS)
   Gaussian high-pass residual.  Linear subtractive low-pass operator.
   Candidate statistics: hp_absmean, hp_std, hp_energy, hp_kurtosis, hp_entropy.
   Feature count: 5 scalars (CANDIDATE — final subset determined in Block 4).

2. extract_branch_d_laplacian_features (D_LAPLACIAN)
   Discrete 8-neighbor Laplacian residual.  Linear 2nd-order curvature operator.
   Candidate statistics: lap_absmean, lap_std, lap_energy, lap_kurtosis, lap_entropy.
   Feature count: 5 scalars (CANDIDATE — final subset determined in Block 4).

3. extract_branch_d_mfr_features (D_MFR)
   Median filter residual.  Nonlinear rank-order operator.
   Candidate statistics: mfr_absmean, mfr_std, mfr_energy, mfr_kurtosis, mfr_entropy.
   Feature count: 5 scalars (CANDIDATE — final subset determined in Block 4).

IMPORTANT — Candidate Relationships
-------------------------------------
D_HIGHPASS and D_LAPLACIAN are both LINEAR high-frequency residual operators.
Their scalar statistics may be substantially correlated.  They are treated as
ALTERNATIVE candidates.  Do not concatenate them by default.

D_MFR is NONLINEAR (rank-order based) and has a clearer theoretical distinction
from Branches A (FFT) and B (wavelet), which are both linear.  D_MFR is a
POTENTIALLY COMPLEMENTARY candidate, but complementarity is not established
until Block 4 experiments demonstrate incremental information contribution.

NOT implemented here (by design):
- D_ELA: ELA/JPEG recompression evidence deferred to future Branch E.
- Standalone cross-difference: Branch A already uses this operator as a
  pre-filter in its Synthbuster periodicity pipeline; duplicating it would
  re-compute the same residual map without adding distinct evidence.
- Any composite "D_RESIDUAL" combining all candidates: premature concatenation
  before empirical validation of complementarity is prohibited by AGENTS.md.

Branch Responsibility Map
--------------------------
Branch A — Frequency  : Global spectral distribution + periodic artifacts (FFT).
Branch B — Wavelet    : Multi-scale localized frequency / subband energy (Haar DWT).
Branch C — Texture    : Local spatial texture patterns (LBP, GLCM).
Branch D — Residual   : Pixel-domain prediction error / noise residual behavior.
Branch E — JPEG/comp  : Compression, quantization, recompression behavior (future).
"""

from __future__ import annotations

from typing import Dict

import torch

from src.forensics.branch_d_residual.highpass import extract_highpass_features
from src.forensics.branch_d_residual.laplacian import extract_laplacian_features
from src.forensics.branch_d_residual.median_filter import extract_mfr_features


# ---------------------------------------------------------------------------
# Named extractor aliases matching pipeline branch identifiers
# ---------------------------------------------------------------------------

def extract_branch_d_highpass_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract D_HIGHPASS candidate features (Gaussian high-pass residual).

    Pipeline:
        RGB [3, 256, 256] -> Grayscale (BT.601) -> R_HP = Y - GaussBlur(Y, 5x5, sigma=1.0)
        -> 5 candidate scalar statistics (hp_absmean, hp_std, hp_energy,
           hp_kurtosis, hp_entropy).

    NOTE: This is a CANDIDATE representation.  Its incremental contribution
    beyond Branches A/B is not yet established.  D_HIGHPASS and D_LAPLACIAN
    may be redundant with each other; Block 4 determines whether both are needed.

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of 5 scalar float features. All finite.
    """
    return extract_highpass_features(image)


def extract_branch_d_laplacian_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract D_LAPLACIAN candidate features (discrete Laplacian residual).

    Pipeline:
        RGB [3, 256, 256] -> Grayscale (BT.601) -> R_Lap = K_Lap * Y (8-neighbor, 3x3)
        -> 5 candidate scalar statistics (lap_absmean, lap_std, lap_energy,
           lap_kurtosis, lap_entropy).

    NOTE: This is a CANDIDATE representation.  ALTERNATIVE to D_HIGHPASS.
    Both are linear residual operators; their statistics may be correlated.
    Do not concatenate with D_HIGHPASS by default.  Block 4 determines whether
    both are needed.

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of 5 scalar float features. All finite.
    """
    return extract_laplacian_features(image)


def extract_branch_d_mfr_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract D_MFR candidate features (3×3 median filter residual).

    Pipeline:
        RGB [3, 256, 256] -> Grayscale (BT.601)
        -> R_MFR = Y - MedianFilter(Y, 3x3, replicate padding)
        -> 5 candidate scalar statistics (mfr_absmean, mfr_std, mfr_energy,
           mfr_kurtosis, mfr_entropy).

    NOTE: This is a CANDIDATE representation.  Nonlinear; most theoretically
    distinct from Branches A and B among the three D candidates.  Still requires
    empirical validation in Block 4.  Computationally more expensive than
    D_HIGHPASS / D_LAPLACIAN due to per-pixel median sorting.

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of 5 scalar float features. All finite.
    """
    return extract_mfr_features(image)
