"""Branch F2: Feature Selection Subpackage.

Provides feature selection baselines and downstream validation:
- MRMRFeatureSelector: Train-only Maximum Relevance Minimum Redundancy static selection.
- DownstreamFeatureValidator: LightGBM tabular validation on feature budgets and ablations.
"""

from src.forensics.branch_f.selection.mrmr import (
    MRMRFeatureSelector,
    MRMRRankedFeature,
)
from src.forensics.branch_f.selection.validation import (
    DownstreamFeatureValidator,
    ValidationExperimentResult,
)

__all__ = [
    "MRMRFeatureSelector",
    "MRMRRankedFeature",
    "DownstreamFeatureValidator",
    "ValidationExperimentResult",
]
