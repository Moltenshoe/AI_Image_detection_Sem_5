"""Generator Stability Diagnostic Analysis Module.

Evaluates candidate forensic feature discriminative power across individual AI generators
(e.g., SD2.1, SDXL, SD3, DALL-E 3, Midjourney).

LEAKAGE & SCIENTIFIC CONTROL RULE:
Generator identity is STRICTLY diagnostic evaluation metadata.
It must NEVER enter feature extraction or classifier training as an input feature.
The purpose of this module is solely diagnostic auditing: identifying features that
generalize across all generators vs. those that exploit generator-specific shortcuts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class GeneratorStabilityStats:
    """Stability statistics for a single forensic feature across generator categories."""

    index: int
    name: str
    branch: str
    overall_effective_auc: float
    mean_generator_auc: float
    worst_generator_auc: float
    best_generator_auc: float
    generator_auc_std: float
    generator_auc_range: float
    # Per-generator effective AUC breakdown
    generator_aucs: Dict[str, float]


class GeneratorStabilityAnalyzer:
    """Computes generator-stratified discriminative AUCs and stability metrics."""

    def __init__(self, branch_map: Optional[Dict[str, str]] = None) -> None:
        """Initialize stability analyzer."""
        self.branch_map = branch_map or {}

    def compute_feature_generator_stability(
        self,
        index: int,
        name: str,
        feature_values: np.ndarray,
        binary_labels: np.ndarray,
        generator_labels: np.ndarray,
        generator_names: Optional[Dict[int, str]] = None,
    ) -> GeneratorStabilityStats:
        """Compute per-generator effective AUCs and stability metrics for one feature.

        Args:
            index: Feature column index.
            name: Feature identifier string.
            feature_values: 1D array of feature values across all samples.
            binary_labels: 1D array of binary class labels (0=real, 1=AI).
            generator_labels: 1D array of generator IDs (0=real, 1..K=AI generators).
            generator_names: Mapping from generator ID to human-readable string.
        """
        vals = np.asarray(feature_values, dtype=np.float64)
        y_bin = np.asarray(binary_labels, dtype=np.int32)
        y_gen = np.asarray(generator_labels)

        # 1. Overall effective AUC (real vs all AI)
        real_mask = (y_bin == 0) & np.isfinite(vals)
        ai_mask = (y_bin == 1) & np.isfinite(vals)

        if np.sum(real_mask) < 2 or np.sum(ai_mask) < 2:
            overall_auc = 0.5
        else:
            y_eval = np.concatenate([np.zeros(np.sum(real_mask)), np.ones(np.sum(ai_mask))])
            v_eval = np.concatenate([vals[real_mask], vals[ai_mask]])
            try:
                raw_auc = float(roc_auc_score(y_eval, v_eval))
                overall_auc = max(raw_auc, 1.0 - raw_auc)
            except Exception:
                overall_auc = 0.5

        # 2. Per-generator AUC (real vs specific AI generator)
        unique_ai_gens = sorted([g for g in np.unique(y_gen[ai_mask])])
        gen_aucs: Dict[str, float] = {}

        for g in unique_ai_gens:
            g_name = generator_names.get(g, f"Gen_{g}") if generator_names else f"Gen_{g}"
            g_ai_mask = (y_gen == g) & (y_bin == 1) & np.isfinite(vals)

            if np.sum(real_mask) < 2 or np.sum(g_ai_mask) < 2:
                gen_aucs[g_name] = 0.5
                continue

            y_sub = np.concatenate([np.zeros(np.sum(real_mask)), np.ones(np.sum(g_ai_mask))])
            v_sub = np.concatenate([vals[real_mask], vals[g_ai_mask]])

            try:
                raw_sub_auc = float(roc_auc_score(y_sub, v_sub))
                eff_sub_auc = float(max(raw_sub_auc, 1.0 - raw_sub_auc))
            except Exception:
                eff_sub_auc = 0.5

            if math.isnan(eff_sub_auc):
                eff_sub_auc = 0.5

            gen_aucs[g_name] = eff_sub_auc

        auc_values = list(gen_aucs.values()) if gen_aucs else [overall_auc]
        mean_auc = float(np.mean(auc_values))
        worst_auc = float(np.min(auc_values))
        best_auc = float(np.max(auc_values))
        std_auc = float(np.std(auc_values))
        range_auc = best_auc - worst_auc

        return GeneratorStabilityStats(
            index=index,
            name=name,
            branch=self.branch_map.get(name, "UNKNOWN"),
            overall_effective_auc=overall_auc,
            mean_generator_auc=mean_auc,
            worst_generator_auc=worst_auc,
            best_generator_auc=best_auc,
            generator_auc_std=std_auc,
            generator_auc_range=range_auc,
            generator_aucs=gen_aucs,
        )

    def analyze_dataset(
        self,
        features_matrix: np.ndarray,
        feature_names: Sequence[str],
        binary_labels: np.ndarray,
        generator_labels: np.ndarray,
        generator_names: Optional[Dict[int, str]] = None,
    ) -> List[GeneratorStabilityStats]:
        """Analyze generator stability for all candidate features."""
        N, D = features_matrix.shape
        if len(feature_names) != D:
            raise ValueError(f"Shape mismatch: {D} columns but {len(feature_names)} names.")

        results = []
        for j in range(D):
            res = self.compute_feature_generator_stability(
                index=j,
                name=feature_names[j],
                feature_values=features_matrix[:, j],
                binary_labels=binary_labels,
                generator_labels=generator_labels,
                generator_names=generator_names,
            )
            results.append(res)
        return results

    @staticmethod
    def to_dataframe(stats_list: Sequence[GeneratorStabilityStats]) -> pd.DataFrame:
        """Convert list of GeneratorStabilityStats to a flattened pandas DataFrame."""
        rows = []
        for s in stats_list:
            row = {
                "index": s.index,
                "name": s.name,
                "branch": s.branch,
                "overall_effective_auc": s.overall_effective_auc,
                "mean_generator_auc": s.mean_generator_auc,
                "worst_generator_auc": s.worst_generator_auc,
                "best_generator_auc": s.best_generator_auc,
                "generator_auc_std": s.generator_auc_std,
                "generator_auc_range": s.generator_auc_range,
            }
            # Flatten per-generator AUCs
            for g_name, auc in s.generator_aucs.items():
                row[f"auc_{g_name.lower().replace(' ', '_').replace('-', '_')}"] = auc
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.sort_values(by="mean_generator_auc", ascending=False).reset_index(drop=True)
