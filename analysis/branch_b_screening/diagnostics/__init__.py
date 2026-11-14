"""Diagnostic metrics, spatial statistics, and visualizers for Branch B screening."""

from __future__ import annotations

from .metrics import (
    compute_scalar_metrics,
    compute_per_image_metrics,
    aggregate_screening_metrics,
    assess_numerical_degeneracy,
)
from .intensity_profiles import (
    compute_intensity_conditioned_profile,
    format_intensity_profile_table,
)
from .spatial_metrics import (
    compute_morans_i,
    compute_gini_coefficient,
)

__all__ = [
    "compute_scalar_metrics",
    "compute_per_image_metrics",
    "aggregate_screening_metrics",
    "assess_numerical_degeneracy",
    "compute_intensity_conditioned_profile",
    "format_intensity_profile_table",
    "compute_morans_i",
    "compute_gini_coefficient",
]
