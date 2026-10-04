"""Minimum Redundancy Maximum Relevance (mRMR) Feature Selection Baseline Module.

Implements the canonical mRMR feature selection algorithm for the 111 candidate forensic bank.
Generates ranked feature orderings and selects compact feature subsets for budgets:
111, 64, 32, 16, 8.

LEAKAGE RULE:
mRMR feature selection must be fitted strictly on TRAINING DATA ONLY.
Validation/test sets must never influence relevance estimation, correlation calculation, or ranking.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MRMRRankedFeature:
    """Metadata and selection score for a single feature ranked by mRMR."""

    rank: int  # 1-indexed rank
    feature_index: int
    name: str
    branch: str
    relevance_score: float
    redundancy_score: float
    mrmr_score: float
    in_budget_8: bool
    in_budget_16: bool
    in_budget_32: bool
    in_budget_64: bool
    in_budget_111: bool


class MRMRFeatureSelector:
    """Maximum Relevance Minimum Redundancy (mRMR) static feature selection engine."""

    BUDGETS: Tuple[int, ...] = (8, 16, 32, 64, 111)

    def __init__(
        self,
        relevance_method: str = "mutual_info",
        redundancy_method: str = "spearman",
        branch_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize mRMR selector.

        Args:
            relevance_method: "mutual_info" or "effective_auc".
            redundancy_method: "spearman" or "pearson".
            branch_map: Mapping from feature name to branch string.
        """
        self.relevance_method = relevance_method
        self.redundancy_method = redundancy_method
        self.branch_map = branch_map or {}

    def fit_ranking(
        self,
        relevance_scores: np.ndarray,
        correlation_matrix: np.ndarray,
        feature_names: Sequence[str],
    ) -> List[MRMRRankedFeature]:
        """Compute greedy mRMR ranking across all features.

        Args:
            relevance_scores: 1D array [D] of relevance values (e.g. MI or eff AUC).
            correlation_matrix: 2D array [D, D] of absolute correlation values in [0, 1].
            feature_names: List of D feature names.

        Returns:
            List of MRMRRankedFeature sorted by selection order (rank 1 to D).
        """
        D = len(feature_names)
        if len(relevance_scores) != D or correlation_matrix.shape != (D, D):
            raise ValueError("Dimension mismatch among relevance, correlation matrix, and feature names.")

        rel = np.asarray(relevance_scores, dtype=np.float64)
        corr = np.abs(np.asarray(correlation_matrix, dtype=np.float64))

        # Normalize relevance to [0, 1] for balanced scale with correlation if needed
        max_rel = np.max(rel)
        if max_rel > 0:
            norm_rel = rel / max_rel
        else:
            norm_rel = rel

        selected_indices: List[int] = []
        unselected_indices: Set[int] = set(range(D))
        ranked_records: List[MRMRRankedFeature] = []

        for step in range(1, D + 1):
            if step == 1:
                # First feature: highest relevance, tie-broken by lowest index
                best_idx = int(np.argmax(norm_rel))
                best_rel = float(rel[best_idx])
                best_red = 0.0
                best_mrmr = float(norm_rel[best_idx])
            else:
                best_idx = -1
                best_mrmr = -float("inf")
                best_rel = 0.0
                best_red = 0.0

                # Evaluate unselected features deterministically by index
                for candidate_idx in sorted(list(unselected_indices)):
                    cand_rel = float(norm_rel[candidate_idx])
                    # Redundancy is average absolute correlation with previously selected features
                    cand_red = float(np.mean([corr[candidate_idx, s] for s in selected_indices]))
                    score = cand_rel - cand_red

                    # Strict greater-than ensures deterministic tie-breaking (smaller index preferred on exact ties)
                    if score > best_mrmr:
                        best_mrmr = score
                        best_idx = candidate_idx
                        best_rel = float(rel[candidate_idx])
                        best_red = cand_red

            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

            feat_name = feature_names[best_idx]
            ranked_records.append(
                MRMRRankedFeature(
                    rank=step,
                    feature_index=best_idx,
                    name=feat_name,
                    branch=self.branch_map.get(feat_name, "UNKNOWN"),
                    relevance_score=best_rel,
                    redundancy_score=best_red,
                    mrmr_score=best_mrmr,
                    in_budget_8=(step <= 8),
                    in_budget_16=(step <= 16),
                    in_budget_32=(step <= 32),
                    in_budget_64=(step <= 64),
                    in_budget_111=(step <= 111),
                )
            )

        return ranked_records

    def get_selected_subsets(
        self,
        ranked_features: Sequence[MRMRRankedFeature],
        budgets: Sequence[int] = BUDGETS,
    ) -> Dict[int, List[str]]:
        """Extract lists of selected feature names for specified budgets."""
        subsets: Dict[int, List[str]] = {}
        for b in budgets:
            subsets[b] = [f.name for f in ranked_features[:b]]
        return subsets

    def export_subsets_json(
        self,
        ranked_features: Sequence[MRMRRankedFeature],
        output_dir: str,
        budgets: Sequence[int] = BUDGETS,
    ) -> Dict[int, str]:
        """Save selected feature subsets as JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        subsets = self.get_selected_subsets(ranked_features, budgets)
        exported_paths: Dict[int, str] = {}

        for b, feat_list in subsets.items():
            path = os.path.join(output_dir, f"selected_features_{b}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "budget": b,
                        "feature_count": len(feat_list),
                        "features": feat_list,
                    },
                    f,
                    indent=2,
                )
            exported_paths[b] = path

        return exported_paths

    @staticmethod
    def to_dataframe(ranked_features: Sequence[MRMRRankedFeature]) -> pd.DataFrame:
        """Convert list of MRMRRankedFeature to a pandas DataFrame."""
        return pd.DataFrame([asdict(f) for f in ranked_features])
