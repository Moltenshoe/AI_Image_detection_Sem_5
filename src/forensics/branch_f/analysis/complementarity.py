"""Branch Complementarity & Multi-Domain Synergy Analysis Module.

Analyzes the 5 forensic evidence domains (A, B, C_LBP, D_MFR, E), calculating
branch-level relevance summaries, domain contributions, and controlled branch ablation configurations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from src.forensics.branch_f.analysis.relevance import FeatureRelevanceStats
from src.forensics.branch_f.registry import FeatureRegistry


@dataclass(frozen=True)
class BranchRelevanceSummary:
    """Summary of univariate discriminative power for an entire forensic branch."""

    branch: str
    feature_count: int
    mean_effective_auc: float
    max_effective_auc: float
    min_effective_auc: float
    mean_mutual_info: float
    max_mutual_info: float
    top_feature: str


@dataclass(frozen=True)
class BranchAblationConfig:
    """Definition of a branch ablation or combination experiment."""

    config_name: str
    config_type: str  # "single", "additive", "leave_one_out", "full"
    branches: Tuple[str, ...]
    feature_count: int
    feature_names: List[str]
    feature_indices: List[int]


class BranchComplementarityAnalyzer:
    """Analyzes branch-level summaries and generates controlled ablation configurations."""

    def __init__(self, registry: Optional[FeatureRegistry] = None) -> None:
        """Initialize with canonical FeatureRegistry."""
        self.registry = registry if registry is not None else FeatureRegistry()
        self.branch_map: Dict[str, str] = {e.name: e.branch for e in self.registry.entries}

    def summarize_branch_relevance(
        self,
        relevance_stats: Sequence[FeatureRelevanceStats],
    ) -> List[BranchRelevanceSummary]:
        """Aggregate feature relevance statistics up to the branch level."""
        # Group by branch
        branch_groups: Dict[str, List[FeatureRelevanceStats]] = {}
        for stat in relevance_stats:
            b = self.branch_map.get(stat.name, "UNKNOWN")
            if b not in branch_groups:
                branch_groups[b] = []
            branch_groups[b].append(stat)

        summaries: List[BranchRelevanceSummary] = []
        for b, stats_list in sorted(branch_groups.items()):
            aucs = [s.effective_auc for s in stats_list]
            mis = [s.mutual_info for s in stats_list]
            top_stat = max(stats_list, key=lambda s: s.effective_auc)

            summaries.append(
                BranchRelevanceSummary(
                    branch=b,
                    feature_count=len(stats_list),
                    mean_effective_auc=float(np.mean(aucs)),
                    max_effective_auc=float(np.max(aucs)),
                    min_effective_auc=float(np.min(aucs)),
                    mean_mutual_info=float(np.mean(mis)),
                    max_mutual_info=float(np.max(mis)),
                    top_feature=top_stat.name,
                )
            )

        return summaries

    def generate_ablation_configs(self) -> List[BranchAblationConfig]:
        """Generate canonical branch ablation and combination experiment definitions."""
        all_branches = ("A", "B", "C_LBP", "D_MFR", "E")
        configs: List[BranchAblationConfig] = []

        # 1. Full 111-feature candidate bank
        all_names = self.registry.feature_names
        all_indices = list(range(len(all_names)))
        configs.append(
            BranchAblationConfig(
                config_name="FULL_111",
                config_type="full",
                branches=all_branches,
                feature_count=len(all_names),
                feature_names=all_names,
                feature_indices=all_indices,
            )
        )

        # 2. Individual branch baselines
        for b in all_branches:
            b_entries = self.registry.get_features_by_branch(b)
            b_names = [e.name for e in b_entries]
            b_indices = [e.index for e in b_entries]
            configs.append(
                BranchAblationConfig(
                    config_name=f"BRANCH_{b}",
                    config_type="single",
                    branches=(b,),
                    feature_count=len(b_names),
                    feature_names=b_names,
                    feature_indices=b_indices,
                )
            )

        # 3. Additive combinations
        accum_branches: List[str] = []
        for b in all_branches[:-1]:  # Exclude last because that equals FULL_111
            accum_branches.append(b)
            combo_name = "+".join(accum_branches)
            combo_entries = [e for e in self.registry.entries if e.branch in accum_branches]
            combo_names = [e.name for e in combo_entries]
            combo_indices = [e.index for e in combo_entries]
            if len(accum_branches) > 1:
                configs.append(
                    BranchAblationConfig(
                        config_name=f"ADD_{combo_name}",
                        config_type="additive",
                        branches=tuple(accum_branches),
                        feature_count=len(combo_names),
                        feature_names=combo_names,
                        feature_indices=combo_indices,
                    )
                )

        # 4. Leave-one-branch-out ablations
        for b in all_branches:
            rem_branches = tuple(x for x in all_branches if x != b)
            rem_entries = [e for e in self.registry.entries if e.branch != b]
            rem_names = [e.name for e in rem_entries]
            rem_indices = [e.index for e in rem_entries]
            configs.append(
                BranchAblationConfig(
                    config_name=f"WITHOUT_{b}",
                    config_type="leave_one_out",
                    branches=rem_branches,
                    feature_count=len(rem_names),
                    feature_names=rem_names,
                    feature_indices=rem_indices,
                )
            )

        return configs

    @staticmethod
    def summaries_to_dataframe(summaries: Sequence[BranchRelevanceSummary]) -> pd.DataFrame:
        """Convert branch summaries list to DataFrame."""
        return pd.DataFrame([asdict(s) for s in summaries])
