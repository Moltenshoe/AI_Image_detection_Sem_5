"""Feature Redundancy & Pairwise Correlation Analysis Module.

Computes 111x111 Pearson and Spearman correlation matrices, intra-branch and inter-branch
redundancy summaries, and identifies highly correlated feature pairs (|rho| >= 0.90) for diagnostic analysis.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class RedundantPair:
    """A flagged pair of features with correlation exceeding the redundancy threshold."""

    feature_1: str
    feature_2: str
    branch_1: str
    branch_2: str
    is_cross_branch: bool
    spearman_rho: float
    pearson_r: float
    abs_spearman: float


@dataclass(frozen=True)
class BranchRedundancySummary:
    """Summary of intra-branch or inter-branch redundancy."""

    branch_group: str
    is_cross_branch: bool
    num_pairs: int
    mean_abs_spearman: float
    max_abs_spearman: float
    mean_abs_pearson: float
    max_abs_pearson: float
    num_flagged_pairs: int


class FeatureRedundancyAnalyzer:
    """Computes pairwise correlation matrices and intra/inter-branch redundancy summaries."""

    def __init__(self, high_correlation_threshold: float = 0.90) -> None:
        """Initialize redundancy analyzer.

        Args:
            high_correlation_threshold: Absolute correlation threshold (|rho| >= threshold) for flagging pairs.
        """
        self.threshold = high_correlation_threshold

    def compute_correlation_matrices(
        self,
        features_matrix: np.ndarray,
        feature_names: Sequence[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute Pearson and Spearman correlation DataFrames.

        Args:
            features_matrix: 2D array [N, D].
            feature_names: List of D feature names.

        Returns:
            (pearson_df, spearman_df)
        """
        df = pd.DataFrame(features_matrix, columns=feature_names)
        # Handle zero-variance columns cleanly
        pearson_df = df.corr(method="pearson").fillna(0.0)
        spearman_df = df.corr(method="spearman").fillna(0.0)
        return pearson_df, spearman_df

    def find_redundant_pairs(
        self,
        spearman_df: pd.DataFrame,
        pearson_df: pd.DataFrame,
        branch_map: Dict[str, str],
    ) -> List[RedundantPair]:
        """Identify all feature pairs with |Spearman rho| >= threshold.

        Args:
            spearman_df: DxD Spearman correlation DataFrame.
            pearson_df: DxD Pearson correlation DataFrame.
            branch_map: Mapping from feature name to its branch identifier.
        """
        features = list(spearman_df.columns)
        D = len(features)
        flagged_pairs: List[RedundantPair] = []

        for i in range(D):
            for j in range(i + 1, D):
                f1 = features[i]
                f2 = features[j]
                rho = float(spearman_df.iloc[i, j])
                r = float(pearson_df.iloc[i, j])
                abs_rho = abs(rho)

                if abs_rho >= self.threshold:
                    b1 = branch_map.get(f1, "UNKNOWN")
                    b2 = branch_map.get(f2, "UNKNOWN")
                    flagged_pairs.append(
                        RedundantPair(
                            feature_1=f1,
                            feature_2=f2,
                            branch_1=b1,
                            branch_2=b2,
                            is_cross_branch=(b1 != b2),
                            spearman_rho=rho,
                            pearson_r=r,
                            abs_spearman=abs_rho,
                        )
                    )

        # Sort descending by absolute Spearman correlation
        flagged_pairs.sort(key=lambda p: p.abs_spearman, reverse=True)
        return flagged_pairs

    def summarize_branch_redundancy(
        self,
        spearman_df: pd.DataFrame,
        pearson_df: pd.DataFrame,
        branch_map: Dict[str, str],
    ) -> List[BranchRedundancySummary]:
        """Compute intra-branch and inter-branch redundancy summaries across all branch combinations."""
        features = list(spearman_df.columns)
        D = len(features)
        branches = sorted(list(set(branch_map.values())))

        summaries: List[BranchRedundancySummary] = []

        # 1. Intra-branch summaries
        for b in branches:
            b_indices = [i for i, f in enumerate(features) if branch_map.get(f) == b]
            if len(b_indices) < 2:
                continue

            rhos = []
            rs = []
            for idx1 in range(len(b_indices)):
                for idx2 in range(idx1 + 1, len(b_indices)):
                    i, j = b_indices[idx1], b_indices[idx2]
                    rhos.append(abs(float(spearman_df.iloc[i, j])))
                    rs.append(abs(float(pearson_df.iloc[i, j])))

            rhos_arr = np.array(rhos)
            rs_arr = np.array(rs)
            num_flagged = int(np.sum(rhos_arr >= self.threshold))

            summaries.append(
                BranchRedundancySummary(
                    branch_group=f"Intra-{b}",
                    is_cross_branch=False,
                    num_pairs=len(rhos),
                    mean_abs_spearman=float(np.mean(rhos_arr)) if len(rhos_arr) > 0 else 0.0,
                    max_abs_spearman=float(np.max(rhos_arr)) if len(rhos_arr) > 0 else 0.0,
                    mean_abs_pearson=float(np.mean(rs_arr)) if len(rs_arr) > 0 else 0.0,
                    max_abs_pearson=float(np.max(rs_arr)) if len(rs_arr) > 0 else 0.0,
                    num_flagged_pairs=num_flagged,
                )
            )

        # 2. Inter-branch summaries
        for b1_idx in range(len(branches)):
            for b2_idx in range(b1_idx + 1, len(branches)):
                b1 = branches[b1_idx]
                b2 = branches[b2_idx]
                b1_indices = [i for i, f in enumerate(features) if branch_map.get(f) == b1]
                b2_indices = [i for i, f in enumerate(features) if branch_map.get(f) == b2]

                rhos = []
                rs = []
                for i in b1_indices:
                    for j in b2_indices:
                        rhos.append(abs(float(spearman_df.iloc[i, j])))
                        rs.append(abs(float(pearson_df.iloc[i, j])))

                rhos_arr = np.array(rhos)
                rs_arr = np.array(rs)
                num_flagged = int(np.sum(rhos_arr >= self.threshold))

                summaries.append(
                    BranchRedundancySummary(
                        branch_group=f"Inter-{b1}<->{b2}",
                        is_cross_branch=True,
                        num_pairs=len(rhos),
                        mean_abs_spearman=float(np.mean(rhos_arr)) if len(rhos_arr) > 0 else 0.0,
                        max_abs_spearman=float(np.max(rhos_arr)) if len(rhos_arr) > 0 else 0.0,
                        mean_abs_pearson=float(np.mean(rs_arr)) if len(rs_arr) > 0 else 0.0,
                        max_abs_pearson=float(np.max(rs_arr)) if len(rs_arr) > 0 else 0.0,
                        num_flagged_pairs=num_flagged,
                    )
                )

        return summaries

    @staticmethod
    def pairs_to_dataframe(pairs: Sequence[RedundantPair]) -> pd.DataFrame:
        """Convert redundant pairs list to DataFrame."""
        return pd.DataFrame([asdict(p) for p in pairs])

    @staticmethod
    def summaries_to_dataframe(summaries: Sequence[BranchRedundancySummary]) -> pd.DataFrame:
        """Convert branch summaries list to DataFrame."""
        return pd.DataFrame([asdict(s) for s in summaries])
