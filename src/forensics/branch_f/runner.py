"""End-to-End Orchestrator for Forensic Feature Analysis (Block 2).

Coordinates Feature Registry, Validity, Univariate Relevance, Redundancy, Branch Complementarity,
Generator Stability, Compression Sensitivity, mRMR Selection, and Downstream Tabular Validation.
Produces persistent, versionable, reproducible research artifacts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.forensics.branch_f.analysis.complementarity import BranchComplementarityAnalyzer
from src.forensics.branch_f.analysis.compression_analysis import CompressionSensitivityAnalyzer
from src.forensics.branch_f.analysis.generator_stability import GeneratorStabilityAnalyzer
from src.forensics.branch_f.analysis.redundancy import FeatureRedundancyAnalyzer
from src.forensics.branch_f.analysis.relevance import FeatureRelevanceAnalyzer
from src.forensics.branch_f.analysis.validity import FeatureValidityAnalyzer
from src.forensics.branch_f.registry import FeatureRegistry
from src.forensics.branch_f.selection.mrmr import MRMRFeatureSelector
from src.forensics.branch_f.selection.validation import DownstreamFeatureValidator
from src.forensics.pipeline import ForensicPipeline


class FeatureAnalysisRunner:
    """Orchestrates end-to-end execution of the Forensic Feature Analysis stage."""

    DEFAULT_OUTPUT_DIR: str = "analysis/forensic_feature_analysis"

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        random_state: int = 42,
        high_corr_threshold: float = 0.90,
    ) -> None:
        """Initialize orchestrator."""
        self.output_dir = output_dir
        self.random_state = random_state
        self.high_corr_threshold = high_corr_threshold

        self.registry = FeatureRegistry()
        self.pipeline = ForensicPipeline(["A", "B", "C_LBP", "D_MFR", "E"])
        self.feature_names = self.registry.feature_names
        self.branch_map = {e.name: e.branch for e in self.registry.entries}
        self.sub_branch_map = {e.name: e.sub_branch for e in self.registry.entries}

    def run_analysis(
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
        """Execute all analysis stages and export persistent artifacts to output_dir.

        Args:
            X_train: [N_train, 111] feature matrix for training.
            y_train: [N_train] binary target array.
            X_val: [N_val, 111] feature matrix for validation (used solely for evaluation).
            y_val: [N_val] binary target array for validation.
            generator_labels_val: Generator IDs for validation images (used solely for generator stability audit).
            generator_names_map: Mapping of generator ID to name.
            compression_sample_tensors: Sample image tensors for symmetric compression analysis.
            compression_sample_labels: Binary labels for compression sample images.

        Returns:
            Dictionary mapping artifact identifier to its output file path.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        t_start = time.perf_counter()
        artifacts: Dict[str, str] = {}

        print("=" * 70)
        print("Block 2 — Forensic Feature Analysis Pipeline")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"Candidate feature bank: {len(self.feature_names)} features")
        print(f"Train samples: {len(y_train)}")
        if y_val is not None:
            print(f"Val samples: {len(y_val)}")

        # ------------------------------------------------------------------
        # Stage A: Feature Registry
        # ------------------------------------------------------------------
        print("\n[Stage A] Exporting Feature Registry...")
        reg_csv = os.path.join(self.output_dir, "feature_registry.csv")
        reg_json = os.path.join(self.output_dir, "feature_registry.json")
        self.registry.export_csv(reg_csv)
        self.registry.export_json(reg_json)
        artifacts["feature_registry_csv"] = reg_csv
        artifacts["feature_registry_json"] = reg_json

        # ------------------------------------------------------------------
        # Stage B: Feature Validity
        # ------------------------------------------------------------------
        print("[Stage B] Computing Feature Validity & Descriptive Statistics...")
        val_analyzer = FeatureValidityAnalyzer()
        validity_stats = val_analyzer.analyze_dataset(
            features_matrix=X_train,
            feature_names=self.feature_names,
            labels=y_train,
        )
        val_df = val_analyzer.to_dataframe(validity_stats)
        val_csv = os.path.join(self.output_dir, "feature_validity.csv")
        val_df.to_csv(val_csv, index=False)
        artifacts["feature_validity_csv"] = val_csv

        # ------------------------------------------------------------------
        # Stage C: Univariate Relevance (Train-Only)
        # ------------------------------------------------------------------
        print("[Stage C] Estimating Univariate Relevance (Effective ROC-AUC & MI)...")
        rel_analyzer = FeatureRelevanceAnalyzer(random_state=self.random_state)
        relevance_stats = rel_analyzer.analyze_dataset(
            features_matrix=X_train,
            feature_names=self.feature_names,
            labels=y_train,
        )
        rel_df = rel_analyzer.to_dataframe(relevance_stats)
        rel_csv = os.path.join(self.output_dir, "univariate_relevance.csv")
        rel_df.to_csv(rel_csv, index=False)
        artifacts["univariate_relevance_csv"] = rel_csv

        # ------------------------------------------------------------------
        # Stage D: Redundancy & Correlation Analysis (Train-Only)
        # ------------------------------------------------------------------
        print("[Stage D] Computing Pearson & Spearman Redundancy Matrices...")
        red_analyzer = FeatureRedundancyAnalyzer(high_correlation_threshold=self.high_corr_threshold)
        p_df, s_df = red_analyzer.compute_correlation_matrices(X_train, self.feature_names)

        p_csv = os.path.join(self.output_dir, "pearson_matrix.csv")
        s_csv = os.path.join(self.output_dir, "spearman_matrix.csv")
        p_df.to_csv(p_csv)
        s_df.to_csv(s_csv)
        artifacts["pearson_matrix_csv"] = p_csv
        artifacts["spearman_matrix_csv"] = s_csv

        pairs = red_analyzer.find_redundant_pairs(s_df, p_df, self.branch_map)
        pairs_df = red_analyzer.pairs_to_dataframe(pairs)
        pairs_csv = os.path.join(self.output_dir, "redundancy_pairs.csv")
        pairs_df.to_csv(pairs_csv, index=False)
        artifacts["redundancy_pairs_csv"] = pairs_csv

        branch_red_summaries = red_analyzer.summarize_branch_redundancy(s_df, p_df, self.branch_map)
        branch_red_df = red_analyzer.summaries_to_dataframe(branch_red_summaries)
        branch_red_csv = os.path.join(self.output_dir, "branch_redundancy_summary.csv")
        branch_red_df.to_csv(branch_red_csv, index=False)
        artifacts["branch_redundancy_csv"] = branch_red_csv

        # ------------------------------------------------------------------
        # Stage E: Branch Complementarity
        # ------------------------------------------------------------------
        print("[Stage E] Summarizing Branch Complementarity...")
        comp_analyzer = BranchComplementarityAnalyzer(self.registry)
        branch_rel_summaries = comp_analyzer.summarize_branch_relevance(relevance_stats)
        branch_comp_df = comp_analyzer.summaries_to_dataframe(branch_rel_summaries)
        branch_comp_csv = os.path.join(self.output_dir, "branch_complementarity.csv")
        branch_comp_df.to_csv(branch_comp_csv, index=False)
        artifacts["branch_complementarity_csv"] = branch_comp_csv

        # ------------------------------------------------------------------
        # Stage F: Generator Stability Diagnostics
        # ------------------------------------------------------------------
        if generator_labels_val is not None and X_val is not None and y_val is not None:
            print("[Stage F] Auditing Generator Stability across AI Generators...")
            gen_analyzer = GeneratorStabilityAnalyzer(self.branch_map)
            gen_stats = gen_analyzer.analyze_dataset(
                features_matrix=X_val,
                feature_names=self.feature_names,
                binary_labels=y_val,
                generator_labels=generator_labels_val,
                generator_names=generator_names_map,
            )
            gen_df = gen_analyzer.to_dataframe(gen_stats)
            gen_csv = os.path.join(self.output_dir, "generator_stability.csv")
            gen_df.to_csv(gen_csv, index=False)
            artifacts["generator_stability_csv"] = gen_csv

        # ------------------------------------------------------------------
        # Stage G: Symmetric Compression Sensitivity
        # ------------------------------------------------------------------
        if compression_sample_tensors is not None and compression_sample_labels is not None:
            print("[Stage G] Evaluating Symmetric JPEG Compression Degradation (Clean, Q95..Q20)...")
            comp_analyzer = CompressionSensitivityAnalyzer(
                pipeline=self.pipeline,
                branch_map=self.branch_map,
                sub_branch_map=self.sub_branch_map,
            )
            comp_matrices = comp_analyzer.extract_features_under_compression(compression_sample_tensors)
            comp_stats = comp_analyzer.compute_compression_stability(
                comp_matrices,
                compression_sample_labels,
            )
            comp_df = comp_analyzer.to_dataframe(comp_stats)
            comp_csv = os.path.join(self.output_dir, "compression_analysis.csv")
            comp_df.to_csv(comp_csv, index=False)
            artifacts["compression_analysis_csv"] = comp_csv

        # ------------------------------------------------------------------
        # Stage H: Train-Only mRMR Feature Selection Baseline
        # ------------------------------------------------------------------
        print("[Stage H] Computing Train-Only mRMR Selection (Budgets: 111, 64, 32, 16, 8)...")
        # Build relevance array ordered by feature_names
        rel_map = {s.name: s.mutual_info for s in relevance_stats}
        relevance_vector = np.array([rel_map[f] for f in self.feature_names])
        corr_matrix_raw = s_df.values

        mrmr_selector = MRMRFeatureSelector(branch_map=self.branch_map)
        ranked_features = mrmr_selector.fit_ranking(
            relevance_scores=relevance_vector,
            correlation_matrix=corr_matrix_raw,
            feature_names=self.feature_names,
        )
        mrmr_df = mrmr_selector.to_dataframe(ranked_features)
        mrmr_csv = os.path.join(self.output_dir, "mrmr_rankings.csv")
        mrmr_df.to_csv(mrmr_csv, index=False)
        artifacts["mrmr_rankings_csv"] = mrmr_csv

        # Export budget JSON subsets
        subset_paths = mrmr_selector.export_subsets_json(ranked_features, self.output_dir)
        for b, path in subset_paths.items():
            artifacts[f"selected_features_{b}"] = path

        # ------------------------------------------------------------------
        # Stage I: Downstream Tabular Validation (LightGBM)
        # ------------------------------------------------------------------
        if X_val is not None and y_val is not None:
            print("[Stage I] Running Downstream LightGBM Validation across Budgets & Ablations...")
            validator = DownstreamFeatureValidator(random_state=self.random_state)
            selected_subsets = mrmr_selector.get_selected_subsets(ranked_features)

            # Budget experiments
            budget_results = validator.run_budget_experiments(
                mrmr_selected_subsets=selected_subsets,
                all_feature_names=self.feature_names,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            # Branch ablation experiments
            branch_comp_analyzer = BranchComplementarityAnalyzer(self.registry)
            ablation_configs = branch_comp_analyzer.generate_ablation_configs()
            name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
            ablation_results = []
            for cfg in ablation_configs:
                if cfg.config_name == "FULL_111":
                    continue  # Already covered in budget 111
                res = validator.evaluate_feature_subset(
                    experiment_name=cfg.config_name,
                    feature_budget=cfg.feature_count,
                    feature_names=cfg.feature_names,
                    feature_indices=cfg.feature_indices,
                    X_train_full=X_train,
                    y_train=y_train,
                    X_val_full=X_val,
                    y_val=y_val,
                )
                ablation_results.append(res)

            all_val_results = budget_results + ablation_results
            val_results_df = validator.to_dataframe(all_val_results)
            val_results_csv = os.path.join(self.output_dir, "validation_results.csv")
            val_results_df.to_csv(val_results_csv, index=False)
            artifacts["validation_results_csv"] = val_results_csv

        # ------------------------------------------------------------------
        # Stage J: Run Metadata
        # ------------------------------------------------------------------
        total_time = time.perf_counter() - t_start
        meta = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_execution_time_sec": total_time,
            "random_state": self.random_state,
            "feature_bank_count": len(self.feature_names),
            "branches": list(self.pipeline.branches),
            "train_samples": len(y_train),
            "val_samples": len(y_val) if y_val is not None else 0,
            "artifacts_generated": artifacts,
        }
        meta_json = os.path.join(self.output_dir, "run_metadata.json")
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        artifacts["run_metadata_json"] = meta_json

        print("\n" + "=" * 70)
        print(f"Feature Analysis Pipeline Complete in {total_time:.2f}s!")
        print(f"Generated {len(artifacts)} persistent artifacts under: {self.output_dir}")
        print("=" * 70)

        return artifacts
