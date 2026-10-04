"""Univariate Feature Relevance Analysis Module.

Calculates discriminative relevance metrics between individual candidate forensic features
and the binary real/AI target (Label_A).

Metrics:
- Raw ROC-AUC
- Effective (direction-invariant) ROC-AUC: max(AUC, 1 - AUC)
- Direction (+1 for positive AI correlation, -1 for inverse)
- Mutual Information (continuous feature / discrete target)

LEAKAGE RULE:
All relevance calculations and estimations must be computed strictly on TRAINING DATA ONLY.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class FeatureRelevanceStats:
    """Univariate relevance metrics for a single forensic feature against the binary target."""

    index: int
    name: str
    raw_auc: float
    effective_auc: float
    direction: int
    mutual_info: float


class FeatureRelevanceAnalyzer:
    """Computes univariate relevance scores (effective AUC and Mutual Information)."""

    def __init__(self, random_state: int = 42, mi_n_neighbors: int = 3) -> None:
        """Initialize relevance analyzer.

        Args:
            random_state: Fixed random seed for deterministic MI estimation.
            mi_n_neighbors: Number of nearest neighbors for Kraskov MI estimator.
        """
        self.random_state = random_state
        self.mi_n_neighbors = mi_n_neighbors

    def compute_feature_auc(self, values: np.ndarray, labels: np.ndarray) -> Tuple[float, float, int]:
        """Compute raw AUC, effective AUC, and direction for a single feature vector.

        Returns:
            (raw_auc, effective_auc, direction)
        """
        y_true = np.asarray(labels, dtype=np.int32)
        y_score = np.asarray(values, dtype=np.float64)

        # Filter non-finite values if any
        mask = np.isfinite(y_score) & np.isfinite(y_true)
        if np.sum(mask) < 2:
            return 0.5, 0.5, 1

        y_true_f = y_true[mask]
        y_score_f = y_score[mask]

        # Check for single class in y_true or constant feature in y_score
        unique_classes = np.unique(y_true_f)
        if len(unique_classes) < 2 or np.all(y_score_f == y_score_f[0]):
            return 0.5, 0.5, 1

        try:
            raw_auc = float(roc_auc_score(y_true_f, y_score_f))
        except Exception:
            raw_auc = 0.5

        if math.isnan(raw_auc):
            raw_auc = 0.5

        effective_auc = float(max(raw_auc, 1.0 - raw_auc))
        direction = 1 if raw_auc >= 0.5 else -1

        return raw_auc, effective_auc, direction

    def compute_mutual_information(
        self,
        features_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Estimate mutual information between continuous features and discrete binary target.

        Args:
            features_matrix: 2D array [N, D] of feature values.
            labels: 1D array [N] of binary targets (0=real, 1=AI).

        Returns:
            1D array [D] of estimated MI values in nats (non-negative).
        """
        X = np.asarray(features_matrix, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int32)

        # Replace non-finite with column median for safe MI calculation
        X_clean = np.copy(X)
        for col in range(X_clean.shape[1]):
            col_vals = X_clean[:, col]
            finite_mask = np.isfinite(col_vals)
            if np.all(~finite_mask):
                X_clean[:, col] = 0.0
            elif np.any(~finite_mask):
                median_val = np.median(col_vals[finite_mask])
                X_clean[~finite_mask, col] = median_val

        mi_scores = mutual_info_classif(
            X_clean,
            y,
            discrete_features=False,
            n_neighbors=self.mi_n_neighbors,
            random_state=self.random_state,
        )
        return np.maximum(0.0, mi_scores)

    def analyze_dataset(
        self,
        features_matrix: np.ndarray,
        feature_names: Sequence[str],
        labels: np.ndarray,
    ) -> List[FeatureRelevanceStats]:
        """Compute full univariate relevance for all candidate features.

        Args:
            features_matrix: 2D array of shape [N, D].
            feature_names: List of D feature names.
            labels: 1D array of N binary targets.

        Returns:
            List of FeatureRelevanceStats.
        """
        N, D = features_matrix.shape
        if len(feature_names) != D:
            raise ValueError(f"Shape mismatch: {D} columns but {len(feature_names)} names provided.")

        mi_scores = self.compute_mutual_information(features_matrix, labels)

        results = []
        for j in range(D):
            raw_auc, eff_auc, direction = self.compute_feature_auc(features_matrix[:, j], labels)
            entry = FeatureRelevanceStats(
                index=j,
                name=feature_names[j],
                raw_auc=raw_auc,
                effective_auc=eff_auc,
                direction=direction,
                mutual_info=float(mi_scores[j]),
            )
            results.append(entry)

        return results

    @staticmethod
    def to_dataframe(stats_list: Sequence[FeatureRelevanceStats]) -> pd.DataFrame:
        """Convert relevance stats to pandas DataFrame sorted by effective AUC descending."""
        df = pd.DataFrame([asdict(s) for s in stats_list])
        return df.sort_values(by="effective_auc", ascending=False).reset_index(drop=True)
