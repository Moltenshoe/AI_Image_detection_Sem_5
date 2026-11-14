"""Branch B — Active Representation: PiD YUV Residuals (s=0.2) [DEC-015].

Active Pipeline API:
    - compute_pid_residual: Active Branch B representation (PiD YUV, s=0.2).
    - compute_branch_b_residual: Canonical alias for the active Branch B representation.

Research & Reproducibility API (NQP, CoDA-inspired):
    - compute_nqp_residual: Research-only baseline probe.
    - derive_sample_seed: Deterministic sample seed generator for research experiments.
    - run_nqp_pilot: Mandatory 100-image pilot runner.
"""

from .pid import compute_branch_b_residual, compute_pid_residual
from .pilot import run_nqp_pilot
from .probe import compute_nqp_residual, derive_sample_seed

__all__ = [
    "compute_branch_b_residual",
    "compute_pid_residual",
    "compute_nqp_residual",
    "derive_sample_seed",
    "run_nqp_pilot",
]

