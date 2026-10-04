"""Branch F1: Feature Analysis Subpackage.

Provides specialized analyzers for the canonical 111 candidate forensic feature bank:
- FeatureValidityAnalyzer: Distribution statistics, variance, NaN/Inf checks, class stats.
- FeatureRelevanceAnalyzer: Univariate effective ROC-AUC and Mutual Information estimation.
- FeatureRedundancyAnalyzer: 111x111 Pearson and Spearman correlation matrices, redundancy flagging.
- BranchComplementarityAnalyzer: Evidence domain summaries and controlled ablation generation.
- GeneratorStabilityAnalyzer: Diagnostic evaluation across the 5 Defactify AI generators.
- CompressionSensitivityAnalyzer: Symmetric JPEG degradation across clean and Q95..Q20.
"""

from src.forensics.branch_f.analysis.complementarity import (
    BranchAblationConfig,
    BranchComplementarityAnalyzer,
    BranchRelevanceSummary,
)
from src.forensics.branch_f.analysis.compression_analysis import (
    CompressionSensitivityAnalyzer,
    FeatureCompressionStats,
    apply_symmetric_jpeg_compression,
)
from src.forensics.branch_f.analysis.generator_stability import (
    GeneratorStabilityAnalyzer,
    GeneratorStabilityStats,
)
from src.forensics.branch_f.analysis.redundancy import (
    BranchRedundancySummary,
    FeatureRedundancyAnalyzer,
    RedundantPair,
)
from src.forensics.branch_f.analysis.relevance import (
    FeatureRelevanceAnalyzer,
    FeatureRelevanceStats,
)
from src.forensics.branch_f.analysis.validity import (
    FeatureValidityAnalyzer,
    FeatureValidityStats,
)

__all__ = [
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
]
