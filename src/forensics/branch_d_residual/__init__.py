"""Branch D — Residual / Error / Noise feature extraction package (Block 2).

Exposes three independent, alternative pixel-domain residual candidate extractors:

  D_HIGHPASS  : Gaussian high-pass residual (linear, 5 candidate scalars)
  D_LAPLACIAN : Discrete Laplacian residual (linear, 5 candidate scalars)
  D_MFR       : Median filter residual (nonlinear, 5 candidate scalars)

Candidate Status
----------------
All three are CANDIDATE representations.  None has been experimentally validated
as useful for AI-image detection.  Complementarity between candidates is not
assumed.  D_HIGHPASS and D_LAPLACIAN are treated as potentially redundant
(both are linear residuals).  D_MFR is the most theoretically distinct (nonlinear).

Block 4 will determine, using training-data-only analysis, which candidates and
which specific statistics contribute meaningful information beyond Branches A/B/C.

Not Implemented Here
--------------------
  ELA: deferred to future Branch E (JPEG/compression-aware evidence).
  Cross-difference: excluded (Branch A already uses this operator).
"""

from __future__ import annotations

from src.forensics.branch_d_residual.features import (
    extract_branch_d_highpass_features,
    extract_branch_d_laplacian_features,
    extract_branch_d_mfr_features,
)
from src.forensics.branch_d_residual.highpass import (
    compute_highpass_residual,
    extract_highpass_features,
)
from src.forensics.branch_d_residual.laplacian import (
    compute_laplacian_residual,
    extract_laplacian_features,
)
from src.forensics.branch_d_residual.median_filter import (
    compute_median_filter_residual,
    extract_mfr_features,
)

__all__ = [
    # Pipeline-facing extractors (named by branch identifier)
    "extract_branch_d_highpass_features",
    "extract_branch_d_laplacian_features",
    "extract_branch_d_mfr_features",
    # Low-level compute functions (residual maps)
    "compute_highpass_residual",
    "compute_laplacian_residual",
    "compute_median_filter_residual",
    # Low-level scalar-extraction functions
    "extract_highpass_features",
    "extract_laplacian_features",
    "extract_mfr_features",
]
