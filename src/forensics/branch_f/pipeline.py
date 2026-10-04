"""Branch F: Meta-Forensic Feature Analysis and Selection Pipeline.

Branch F operates AFTER Branches A–E feature extraction. It does not extract
new image-level features; instead, it provides structured analysis (F1) and
deterministic selection (F2) on the canonical 111-feature bank.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from src.forensics.branch_f.registry import FeatureRegistry, CANONICAL_BRANCHES
from src.forensics.branch_f.analysis.validity import FeatureValidityAnalyzer
from src.forensics.branch_f.analysis.relevance import FeatureRelevanceAnalyzer
from src.forensics.branch_f.analysis.redundancy import FeatureRedundancyAnalyzer
from src.forensics.branch_f.analysis.complementarity import BranchComplementarityAnalyzer
from src.forensics.branch_f.analysis.generator_stability import GeneratorStabilityAnalyzer
from src.forensics.branch_f.analysis.compression_analysis import CompressionSensitivityAnalyzer
from src.forensics.branch_f.selection.mrmr import MRMRFeatureSelector
from src.forensics.branch_f.selection.validation import DownstreamFeatureValidator
from src.forensics.branch_f.runner import FeatureAnalysisRunner
from src.forensics.pipeline import ForensicPipeline


class BranchFPipeline:
    """Unified facade for Branch F (Analysis & Selection)."""

    def __init__(
        self,
        output_dir: str = "analysis/forensic_feature_analysis",
        random_state: int = 42,
    ) -> None:
        self.output_dir = output_dir
        self.random_state = random_state
        self.registry = FeatureRegistry()
        self.runner = FeatureAnalysisRunner(
            output_dir=output_dir,
            random_state=random_state,
        )

    @property
    def feature_names(self) -> List[str]:
        """List of all 111 canonical feature names."""
        return self.registry.feature_names

    def run_analysis_and_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        generator_labels_val: Optional[np.ndarray] = None,
        generator_names_map: Optional[Dict[int, str]] = None,
        compression_sample_tensors: Optional[List[torch.Tensor]] = None,
        compression_sample_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """Execute full F1 analysis + F2 selection workflow and export artifacts."""
        return self.runner.run_analysis(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            generator_labels_val=generator_labels_val,
            generator_names_map=generator_names_map,
            compression_sample_tensors=compression_sample_tensors,
            compression_sample_labels=compression_sample_labels,
        )
