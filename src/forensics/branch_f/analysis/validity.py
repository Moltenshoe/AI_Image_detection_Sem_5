"""Feature Validity Diagnostics & Summary Statistics Module.

Calculates numerical validity, distribution properties, variance checks, and class-wise
(Real vs AI) descriptive statistics across the 111 candidate forensic features.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class FeatureValidityStats:
    """Descriptive and validity statistics for a single forensic feature."""

    index: int
    name: str
    total_samples: int
    finite_count: int
    finite_rate: float
    nan_count: int
    nan_rate: float
    inf_count: int
    inf_rate: float
    mean: float
    std: float
    variance: float
    min: float
    p25: float
    median: float
    p75: float
    max: float
    is_constant: bool
    near_zero_variance: bool
    # Class-wise statistics
    mean_real: float
    std_real: float
    median_real: float
    mean_ai: float
    std_ai: float
    median_ai: float
    delta_mean: float
    t_statistic: float
    t_pvalue: float


class FeatureValidityAnalyzer:
    """Analyzes numerical validity and distribution properties of candidate features."""

    def __init__(self, variance_threshold: float = 1e-6) -> None:
        """Initialize analyzer with variance threshold."""
        self.variance_threshold = variance_threshold

    def compute_feature_stats(
        self,
        index: int,
        name: str,
        values: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> FeatureValidityStats:
        """Compute comprehensive validity and class-stratified statistics for one feature vector.

        Args:
            index: Column index of the feature.
            name: Feature identifier.
            values: 1D NumPy array of numerical values for this feature across samples.
            labels: Optional 1D binary target array (0=real, 1=AI).
        """
        values_f = np.asarray(values, dtype=np.float64)
        total = len(values_f)
        if total == 0:
            raise ValueError(f"Feature array for '{name}' is empty.")

        is_nan = np.isnan(values_f)
        is_inf = np.isinf(values_f)
        is_finite = np.isfinite(values_f)

        nan_count = int(np.sum(is_nan))
        inf_count = int(np.sum(is_inf))
        finite_count = int(np.sum(is_finite))

        nan_rate = nan_count / total
        inf_rate = inf_count / total
        finite_rate = finite_count / total

        finite_vals = values_f[is_finite]

        if len(finite_vals) > 0:
            mean_val = float(np.mean(finite_vals))
            std_val = float(np.std(finite_vals))
            var_val = float(np.var(finite_vals))
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
            p25_val = float(np.percentile(finite_vals, 25))
            median_val = float(np.percentile(finite_vals, 50))
            p75_val = float(np.percentile(finite_vals, 75))
            is_const = bool(min_val == max_val or var_val == 0.0)
            near_zero_var = bool(var_val < self.variance_threshold)
        else:
            mean_val = std_val = var_val = min_val = max_val = p25_val = median_val = p75_val = float("nan")
            is_const = True
            near_zero_var = True

        # Class-wise statistics if labels provided
        if labels is not None and len(labels) == total:
            labels_arr = np.asarray(labels)
            mask_real = (labels_arr == 0) & is_finite
            mask_ai = (labels_arr == 1) & is_finite

            real_vals = values_f[mask_real]
            ai_vals = values_f[mask_ai]

            mean_real = float(np.mean(real_vals)) if len(real_vals) > 0 else float("nan")
            std_real = float(np.std(real_vals)) if len(real_vals) > 0 else float("nan")
            median_real = float(np.median(real_vals)) if len(real_vals) > 0 else float("nan")

            mean_ai = float(np.mean(ai_vals)) if len(ai_vals) > 0 else float("nan")
            std_ai = float(np.std(ai_vals)) if len(ai_vals) > 0 else float("nan")
            median_ai = float(np.median(ai_vals)) if len(ai_vals) > 0 else float("nan")

            delta_mean = mean_ai - mean_real if not math.isnan(mean_ai) and not math.isnan(mean_real) else float("nan")

            if len(real_vals) >= 2 and len(ai_vals) >= 2 and std_real > 0 and std_ai > 0:
                t_res = stats.ttest_ind(ai_vals, real_vals, equal_var=False)
                t_stat = float(t_res.statistic)
                t_pval = float(t_res.pvalue)
            else:
                t_stat = float("nan")
                t_pval = float("nan")
        else:
            mean_real = std_real = median_real = mean_ai = std_ai = median_ai = delta_mean = t_stat = t_pval = float("nan")

        return FeatureValidityStats(
            index=index,
            name=name,
            total_samples=total,
            finite_count=finite_count,
            finite_rate=finite_rate,
            nan_count=nan_count,
            nan_rate=nan_rate,
            inf_count=inf_count,
            inf_rate=inf_rate,
            mean=mean_val,
            std=std_val,
            variance=var_val,
            min=min_val,
            p25=p25_val,
            median=median_val,
            p75=p75_val,
            max=max_val,
            is_constant=is_const,
            near_zero_variance=near_zero_var,
            mean_real=mean_real,
            std_real=std_real,
            median_real=median_real,
            mean_ai=mean_ai,
            std_ai=std_ai,
            median_ai=median_ai,
            delta_mean=delta_mean,
            t_statistic=t_stat,
            t_pvalue=t_pval,
        )

    def analyze_dataset(
        self,
        features_matrix: np.ndarray,
        feature_names: Sequence[str],
        labels: Optional[np.ndarray] = None,
    ) -> List[FeatureValidityStats]:
        """Analyze full feature matrix [N, D].

        Args:
            features_matrix: 2D array of shape [N, D].
            feature_names: List of D feature names.
            labels: Optional 1D array of N binary targets.

        Returns:
            List of FeatureValidityStats for each feature.
        """
        N, D = features_matrix.shape
        if len(feature_names) != D:
            raise ValueError(f"Shape mismatch: {D} columns but {len(feature_names)} names provided.")

        results = []
        for j in range(D):
            stats_entry = self.compute_feature_stats(
                index=j,
                name=feature_names[j],
                values=features_matrix[:, j],
                labels=labels,
            )
            results.append(stats_entry)
        return results

    @staticmethod
    def to_dataframe(stats_list: Sequence[FeatureValidityStats]) -> pd.DataFrame:
        """Convert list of FeatureValidityStats to a pandas DataFrame."""
        return pd.DataFrame([asdict(s) for s in stats_list])
