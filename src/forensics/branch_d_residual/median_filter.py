"""Median filter residual feature extraction (Block 2 — Branch D candidate: D_MFR).

Mathematical Definition
-----------------------
Input:
    Canonical RGB image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

Grayscale conversion (ITU-R BT.601):
    Y = 0.299 * R + 0.587 * G + 0.114 * B,  Y in [0, 1]

Median filter residual:
    R_MFR = Y - MedianFilter(Y, kernel=3×3)

The 3×3 median filter is a rank-order (nonlinear) operator.  It replaces each
pixel with the median of its 3×3 neighborhood.  The residual isolates the
component of the signal that cannot be predicted by local intensity ranking.

Boundary Convention
-------------------
Padding: replicate (1-pixel border on each side for the 3×3 neighborhood).
Consistent with D_HIGHPASS and D_LAPLACIAN boundary handling.

Implementation
--------------
Pure PyTorch implementation using F.unfold followed by torch.median on the
patch dimension.  This avoids OpenCV/SciPy dependency in the feature extraction
critical path.  The unfolding approach is deterministic.

    padded = F.pad(Y, (1,1,1,1), mode="replicate")   # [B, 1, H+2, W+2]
    patches = F.unfold(padded, kernel_size=3)          # [B, 9, H*W]
    median_flat = patches.median(dim=1).values         # [B, H*W]
    median_map = median_flat.view(B, 1, H, W)
    R_MFR = Y - median_map

Candidate Statistics (5 scalars — prefix: mfr_*)
--------------------------------------------------
    mfr_absmean  : mean(|R_MFR|)              — average noise amplitude
    mfr_std      : std(R_MFR)                 — noise standard deviation
    mfr_energy   : mean(R_MFR²)              — mean squared noise energy
    mfr_kurtosis : Fisher excess kurtosis of R_MFR distribution
    mfr_entropy  : Shannon entropy (256-bin histogram) of R_MFR

All statistics are defined as CANDIDATES.  The final retained subset is
determined experimentally in Block 4 using training-data-only analysis.

Forensic Hypothesis
-------------------
The median filter is edge-preserving: it suppresses Gaussian-like random noise
while preserving step-function edges.  Therefore R_MFR isolates impulsive and
non-smooth residual components that survive rank-order filtering.  This includes
pixel-level grid artifacts, quantization ringing, and synthesis noise that AI
generative models may introduce as part of their generation process.

Distinction from D_HIGHPASS and D_LAPLACIAN
--------------------------------------------
D_HIGHPASS and D_LAPLACIAN are LINEAR operators.  Their statistics can in
principle be predicted from the Fourier spectrum (Branch A) or wavelet subbands
(Branch B).  D_MFR is NONLINEAR: its output depends on local pixel value rank,
not on linear projections.  This gives D_MFR a clearer theoretical justification
for potentially carrying information not already in Branches A/B.

Despite this distinction, whether D_MFR contributes information beyond Branches
A/B/C in practice is an empirical question deferred to Block 4.  D_MFR may
correlate with C_LBP features (which are also rank-order sensitive).  This
possible correlation must be tested in Block 4 before concluding complementarity.

Computational Note
------------------
The F.unfold + .median() approach is significantly more expensive than convolution-
based high-pass or Laplacian operations because sorting 9 elements per pixel
involves sequential comparisons.  Runtime is measured during verification and
documented in CHANGELOG.  Computational cost is a factor in Block 4 evaluation.

Numerical Safety
----------------
    - kurtosis returns 0.0 when std < 1e-12
    - entropy returns 0.0 when input range < 1e-12
    - For a constant-valued image, MedianFilter(c) = c, so R_MFR = 0 exactly.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.forensics.branch_d_residual.highpass import _safe_entropy, _safe_kurtosis


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_median_filter_residual(image: torch.Tensor) -> torch.Tensor:
    """Compute the 3×3 median filter residual map from a canonical image tensor.

    Converts to grayscale, applies a 3×3 median filter with replicate padding
    using pure PyTorch (F.unfold + median), and returns Y - MedianFilter(Y).

    Args:
        image: Canonical RGB tensor [3, H, W] or [B, 3, H, W], float32 in [0, 1].

    Returns:
        Signed residual tensor [1, H, W] or [B, 1, H, W], float32.
        Shape is the same spatial size as input.
    """
    if image.dim() == 3:
        img = image.unsqueeze(0)            # [1, 3, H, W]
        squeeze_out = True
    else:
        img = image
        squeeze_out = False

    B, _, H, W = img.shape

    # Grayscale — ITU-R BT.601
    gray = (
        0.299 * img[:, 0:1, :, :]
        + 0.587 * img[:, 1:2, :, :]
        + 0.114 * img[:, 2:3, :, :]
    )                                       # [B, 1, H, W]

    # Replicate-pad 1 pixel on each side for the 3×3 neighbourhood
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")   # [B, 1, H+2, W+2]

    # Extract 3×3 patches for every pixel: [B, 9, H*W]
    patches = F.unfold(padded, kernel_size=3, stride=1, padding=0)

    # Compute per-pixel median across the 9 neighbourhood values
    median_flat = patches.median(dim=1).values              # [B, H*W]
    median_map = median_flat.view(B, 1, H, W)              # [B, 1, H, W]

    residual = gray - median_map                            # [B, 1, H, W]

    if squeeze_out:
        residual = residual.squeeze(0)      # [1, H, W]

    return residual


def extract_mfr_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract candidate D_MFR scalar statistics from the median filter residual.

    Candidate statistics (5 scalars, prefix: mfr_*):
        mfr_absmean  : mean(|R_MFR|)
        mfr_std      : std(R_MFR)
        mfr_energy   : mean(R_MFR^2)
        mfr_kurtosis : Fisher excess kurtosis of R_MFR
        mfr_entropy  : Shannon entropy (256-bin) of R_MFR

    NOTE: These are CANDIDATE statistics.  The final retained subset is
    determined in Block 4 using training-data-only analysis.

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of 5 scalar float features in deterministic order.
        All values are finite (no NaN, no Inf).
    """
    residual = compute_median_filter_residual(image)    # [1, H, W] or [B, 1, H, W]
    flat = residual.reshape(-1).detach().cpu()

    features: Dict[str, float] = {}
    features["mfr_absmean"]  = float(flat.abs().mean())
    features["mfr_std"]      = float(flat.std())
    features["mfr_energy"]   = float(flat.pow(2).mean())
    features["mfr_kurtosis"] = _safe_kurtosis(flat)
    features["mfr_entropy"]  = _safe_entropy(flat)

    return features
