"""Candidate representations for Branch B numerical screening."""

from __future__ import annotations

from .raw_nqp import compute_raw_nqp_residual, compute_raw_nqp_from_noise
from .cvpr_color import (
    extract_cvpr_reference,
    extract_cvpr_seeded,
    compute_cvpr_from_noise,
    quantize_cvpr,
)
from .pid_yuv import compute_pid_residual, forward_yuv, inverse_yuv
from .coda_diagnostic import (
    coda_algorithm_1,
    coda_equation_12,
    compare_coda_formulations,
)

__all__ = [
    "compute_raw_nqp_residual",
    "compute_raw_nqp_from_noise",
    "extract_cvpr_reference",
    "extract_cvpr_seeded",
    "compute_cvpr_from_noise",
    "quantize_cvpr",
    "compute_pid_residual",
    "forward_yuv",
    "inverse_yuv",
    "coda_algorithm_1",
    "coda_equation_12",
    "compare_coda_formulations",
]
