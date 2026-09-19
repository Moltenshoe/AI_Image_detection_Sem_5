"""Discrete Laplacian residual feature extraction (Block 2 — Branch D candidate: D_LAPLACIAN).

Mathematical Definition
-----------------------
Input:
    Canonical RGB image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

Grayscale conversion (ITU-R BT.601):
    Y = 0.299 * R + 0.587 * G + 0.114 * B,  Y in [0, 1]

Laplacian residual:
    R_Lap = K_Lap * Y

where K_Lap is the discrete 8-neighbor Laplacian kernel:

        K_Lap = [[1,  1, 1],
                 [1, -8, 1],
                 [1,  1, 1]]

The kernel sum is zero, so a constant-valued input produces zero output.
Output range is approximately [-8, 8] for input in [0, 1], but in practice
confined by local intensity differences in natural images.

Boundary Convention
-------------------
Padding: replicate (1-pixel border on each side for the 3×3 kernel).
This avoids zero-padding boundary artifacts.

Candidate Statistics (5 scalars — prefix: lap_*)
--------------------------------------------------
    lap_absmean  : mean(|R_Lap|)               — average curvature magnitude
    lap_std      : std(R_Lap)                  — curvature variance
    lap_energy   : mean(R_Lap²)               — mean squared curvature
    lap_kurtosis : Fisher excess kurtosis of R_Lap distribution
    lap_entropy  : Shannon entropy (256-bin histogram) of R_Lap

All statistics are defined as CANDIDATES.  The final retained subset is
determined experimentally in Block 4 using training-data-only analysis.

Forensic Hypothesis
-------------------
The Laplacian measures spatial curvature: the difference between a pixel and
the average of its 8 neighbors.  Real camera images exhibit curvature patterns
tied to optical blur, anti-aliasing, and JPEG ringing at edges.  AI-generated
images from different synthesis processes may produce distinct curvature
distributions.

Relationship to D_HIGHPASS
---------------------------
Both D_HIGHPASS and D_LAPLACIAN are LINEAR high-frequency residual operators
on the same grayscale input.  Their aggregate scalar statistics may be highly
correlated.  They are therefore classified as ALTERNATIVE residual candidates,
not as complementary components.  Whether one, both, or neither contributes
information beyond Branch B wavelet features is an empirical question deferred
to Block 4.  Do NOT concatenate D_HIGHPASS + D_LAPLACIAN by default.

Relationship to C_GLCM
-----------------------
The GLCM contrast statistic (weighted by squared intensity differences) is
conceptually related to second-order spatial differences.  Some correlation
between lap_* features and glcm contrast statistics is therefore expected and
must be tested in Block 4.

Numerical Safety
----------------
    - kurtosis returns 0.0 when std < 1e-12 (constant residual)
    - entropy returns 0.0 when input range < 1e-12
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.forensics.branch_d_residual.highpass import _safe_entropy, _safe_kurtosis


# ---------------------------------------------------------------------------
# Kernel definition — module-level constant
# ---------------------------------------------------------------------------

# 8-neighbor discrete Laplacian kernel.  Sum = 0, center = -8.
# Do NOT substitute a 4-neighbor kernel (center=-4) or any other variant
# without updating the module docstring, tests, and CHANGELOG.
_LAPLACIAN_KERNEL: torch.Tensor = torch.tensor(
    [[1.0, 1.0, 1.0],
     [1.0, -8.0, 1.0],
     [1.0, 1.0, 1.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_laplacian_residual(image: torch.Tensor) -> torch.Tensor:
    """Compute the discrete Laplacian residual map from a canonical image tensor.

    Converts to grayscale, applies the 8-neighbor 3×3 Laplacian kernel with
    replicate padding, and returns the signed response map.

    Args:
        image: Canonical RGB tensor [3, H, W] or [B, 3, H, W], float32 in [0, 1].

    Returns:
        Signed residual tensor [1, H, W] or [B, 1, H, W], float32.
        Shape is the same spatial size as input.  Range is approximately [-8, 8]
        but practically smaller for natural images.
    """
    if image.dim() == 3:
        img = image.unsqueeze(0)
        squeeze_out = True
    else:
        img = image
        squeeze_out = False

    # Grayscale — ITU-R BT.601
    gray = (
        0.299 * img[:, 0:1, :, :]
        + 0.587 * img[:, 1:2, :, :]
        + 0.114 * img[:, 2:3, :, :]
    )                                       # [B, 1, H, W]

    # Apply Laplacian with replicate padding (1 pixel for 3×3 kernel)
    kernel = _LAPLACIAN_KERNEL.to(device=gray.device, dtype=gray.dtype)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    residual = F.conv2d(padded, kernel, stride=1, padding=0)   # [B, 1, H, W]

    if squeeze_out:
        residual = residual.squeeze(0)      # [1, H, W]

    return residual


def extract_laplacian_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract candidate D_LAPLACIAN scalar statistics from the Laplacian residual map.

    Candidate statistics (5 scalars, prefix: lap_*):
        lap_absmean  : mean(|R_Lap|)
        lap_std      : std(R_Lap)
        lap_energy   : mean(R_Lap^2)
        lap_kurtosis : Fisher excess kurtosis of R_Lap
        lap_entropy  : Shannon entropy (256-bin) of R_Lap

    NOTE: These are CANDIDATE statistics.  The final retained subset is
    determined in Block 4 using training-data-only analysis.  D_HIGHPASS and
    D_LAPLACIAN are alternative candidates: do not concatenate them by default.

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of 5 scalar float features in deterministic order.
        All values are finite (no NaN, no Inf).
    """
    residual = compute_laplacian_residual(image)   # [1, H, W] or [B, 1, H, W]
    flat = residual.reshape(-1).detach().cpu()

    features: Dict[str, float] = {}
    features["lap_absmean"]  = float(flat.abs().mean())
    features["lap_std"]      = float(flat.std())
    features["lap_energy"]   = float(flat.pow(2).mean())
    features["lap_kurtosis"] = _safe_kurtosis(flat)
    features["lap_entropy"]  = _safe_entropy(flat)

    return features
