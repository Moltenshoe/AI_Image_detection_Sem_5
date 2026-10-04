"""Lightweight Downstream Tabular Validation Module.

Uses LightGBM as a controlled evaluation instrument to measure information retention
across selected feature budgets (111, 64, 32, 16, 8) and branch ablation configurations.

SCIENTIFIC / LEAKAGE CONTROL:
LightGBM validation uses the feature subsets determined strictly by TRAIN-ONLY selection.
Validation and test sets are used solely for evaluation.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ValidationExperimentResult:
    """Downstream validation metrics for a single feature subset configuration."""

    experiment_name: str
    feature_budget: int
    feature_count: int
    train_samples: int
    val_samples: int
    roc_auc: float
    pr_auc: float
    f1: float
    accuracy: float
    tpr_at_1pct_fpr: float
    train_time_sec: float
    inference_time_ms_per_1k: float
    feature_names: List[str]


def compute_tpr_at_fixed_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """Compute True Positive Rate at a fixed False Positive Rate threshold (e.g. FPR=1%)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


class DownstreamFeatureValidator:
    """Evaluates feature budgets and branch subsets using lightweight LightGBM models."""

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
    ) -> None:
        """Initialize validator with fixed model hyper-parameters."""
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def create_model(self) -> lgb.LGBMClassifier:
        """Create standard LightGBM classifier."""
        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def evaluate_feature_subset(
        self,
        experiment_name: str,
        feature_budget: int,
        feature_names: Sequence[str],
        feature_indices: Sequence[int],
        X_train_full: np.ndarray,
        y_train: np.ndarray,
        X_val_full: np.ndarray,
        y_val: np.ndarray,
    ) -> ValidationExperimentResult:
        """Train LightGBM on selected feature columns and evaluate on validation data."""
        indices = list(feature_indices)
        X_tr = X_train_full[:, indices]
        X_v = X_val_full[:, indices]
        y_tr = np.asarray(y_train, dtype=np.int32)
        y_v = np.asarray(y_val, dtype=np.int32)

        model = self.create_model()

        # Train with timing
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        # Inference timing on validation set
        t0_inf = time.perf_counter()
        probs = model.predict_proba(X_v)[:, 1]
        preds = (probs >= 0.5).astype(np.int32)
        inf_time = time.perf_counter() - t0_inf

        inf_time_per_1k = (inf_time / max(1, len(y_v))) * 1000.0 * 1000.0  # in ms per 1000 samples

        # Metrics
        auc = float(roc_auc_score(y_v, probs)) if len(np.unique(y_v)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_v, probs)) if len(np.unique(y_v)) > 1 else 0.0
        f1 = float(f1_score(y_v, preds, zero_division=0))
        acc = float(accuracy_score(y_v, preds))
        tpr_1pct = compute_tpr_at_fixed_fpr(y_v, probs, target_fpr=0.01)

        return ValidationExperimentResult(
            experiment_name=experiment_name,
            feature_budget=feature_budget,
            feature_count=len(indices),
            train_samples=len(y_tr),
            val_samples=len(y_v),
            roc_auc=auc,
            pr_auc=pr_auc,
            f1=f1,
            accuracy=acc,
            tpr_at_1pct_fpr=tpr_1pct,
            train_time_sec=train_time,
            inference_time_ms_per_1k=inf_time_per_1k,
            feature_names=list(feature_names),
        )

    def run_budget_experiments(
        self,
        mrmr_selected_subsets: Dict[int, List[str]],
        all_feature_names: Sequence[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> List[ValidationExperimentResult]:
        """Run validation experiments across all mRMR budget subsets."""
        name_to_idx = {name: i for i, name in enumerate(all_feature_names)}
        results: List[ValidationExperimentResult] = []

        # Sort budgets descending (111, 64, 32, 16, 8)
        for budget in sorted(mrmr_selected_subsets.keys(), reverse=True):
            feat_subset = mrmr_selected_subsets[budget]
            indices = [name_to_idx[f] for f in feat_subset if f in name_to_idx]
            res = self.evaluate_feature_subset(
                experiment_name=f"mRMR_Budget_{budget}",
                feature_budget=budget,
                feature_names=feat_subset,
                feature_indices=indices,
                X_train_full=X_train,
                y_train=y_train,
                X_val_full=X_val,
                y_val=y_val,
            )
            results.append(res)

        return results

    @staticmethod
    def to_dataframe(results: Sequence[ValidationExperimentResult]) -> pd.DataFrame:
        """Convert list of ValidationExperimentResult to a pandas DataFrame."""
        rows = []
        for r in results:
            d = asdict(r)
            d["feature_names"] = ", ".join(d["feature_names"])
            rows.append(d)
        return pd.DataFrame(rows)
