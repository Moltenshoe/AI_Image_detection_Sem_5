"""Branch F: Forensic Feature Analysis & Feature Selection (Meta-Forensic Stage).

Branch F operates downstream of Branches A–E. It inspects, audits, and selects
compact subsets from the canonical 111-feature candidate bank:
- F1 (Analysis): Validity, Relevance (ROC-AUC/MI), Redundancy (Pearson/Spearman),
  Branch Complementarity, Generator Stability, and Compression Sensitivity.
- F2 (Selection): Train-only baseline mRMR selection across budgets 111, 64, 32, 16, 8,
  and downstream LightGBM tabular validation.
"""

from src.forensics.branch_f.analysis import (
    BranchAblationConfig,
    BranchComplementarityAnalyzer,
    BranchRedundancySummary,
    BranchRelevanceSummary,
    CompressionSensitivityAnalyzer,
    FeatureCompressionStats,
    FeatureRedundancyAnalyzer,
    FeatureRelevanceAnalyzer,
    FeatureRelevanceStats,
    FeatureValidityAnalyzer,
    FeatureValidityStats,
    GeneratorStabilityAnalyzer,
    GeneratorStabilityStats,
    RedundantPair,
    apply_symmetric_jpeg_compression,
)
from src.forensics.branch_f.pipeline import BranchFPipeline
from src.forensics.branch_f.registry import (
    CANONICAL_BRANCHES,
    EXPECTED_BRANCH_COUNTS,
    EXPECTED_FEATURE_COUNT,
    FeatureMetadata,
    FeatureRegistry,
)
from src.forensics.branch_f.runner import FeatureAnalysisRunner
from src.forensics.branch_f.selection import (
    DownstreamFeatureValidator,
    MRMRFeatureSelector,
    MRMRRankedFeature,
    ValidationExperimentResult,
)

__all__ = [
    "FeatureRegistry",
    "FeatureMetadata",
    "CANONICAL_BRANCHES",
    "EXPECTED_FEATURE_COUNT",
    "EXPECTED_BRANCH_COUNTS",
    "FeatureValidityAnalyzer",
    "FeatureValidityStats",
    "FeatureRelevanceAnalyzer",
    "FeatureRelevanceStats",
    "FeatureRedundancyAnalyzer",
    "RedundantPair",
    "BranchRedundancySummary",
    "BranchComplementarityAnalyzer",
    "BranchRelevanceSummary",
    "BranchAblationConfig",
    "GeneratorStabilityAnalyzer",
    "GeneratorStabilityStats",
    "CompressionSensitivityAnalyzer",
    "FeatureCompressionStats",
    "apply_symmetric_jpeg_compression",
    "MRMRFeatureSelector",
    "MRMRRankedFeature",
    "DownstreamFeatureValidator",
    "ValidationExperimentResult",
    "FeatureAnalysisRunner",
    "BranchFPipeline",
]
