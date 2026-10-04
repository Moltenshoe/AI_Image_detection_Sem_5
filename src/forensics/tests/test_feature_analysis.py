"""Unit Test Suite for Forensic Feature Analysis & Selection (Block 2).

Verifies the 111-feature registry contract, validity analyzer, univariate relevance (effective AUC & MI),
redundancy & correlation matrices, branch complementarity, generator stability, symmetric compression,
train-only mRMR ranking, and downstream LightGBM validation.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.forensics.branch_f.analysis.complementarity import (
    BranchComplementarityAnalyzer,
    BranchRelevanceSummary,
)
from src.forensics.branch_f.analysis.compression_analysis import (
    CompressionSensitivityAnalyzer,
    apply_symmetric_jpeg_compression,
)
from src.forensics.branch_f.analysis.generator_stability import (
    GeneratorStabilityAnalyzer,
)
from src.forensics.branch_f.analysis.redundancy import (
    FeatureRedundancyAnalyzer,
)
from src.forensics.branch_f.analysis.relevance import (
    FeatureRelevanceAnalyzer,
)
from src.forensics.branch_f.analysis.validity import (
    FeatureValidityAnalyzer,
)
from src.forensics.branch_f.pipeline import BranchFPipeline
from src.forensics.branch_f.registry import (
    CANONICAL_BRANCHES,
    EXPECTED_BRANCH_COUNTS,
    EXPECTED_FEATURE_COUNT,
    FeatureRegistry,
)
from src.forensics.branch_f.runner import (
    FeatureAnalysisRunner,
)
from src.forensics.branch_f.selection.mrmr import (
    MRMRFeatureSelector,
)
from src.forensics.branch_f.selection.validation import (
    DownstreamFeatureValidator,
)
from src.forensics.pipeline import ForensicPipeline


class TestFeatureAnalysis(unittest.TestCase):
    """Test suite for Block 2 Forensic Feature Analysis components."""

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_feature_registry_contract(self):
        """Verify feature registry contains exactly 111 canonical features with correct branch counts."""
        registry = FeatureRegistry()
        self.assertEqual(len(registry.feature_names), EXPECTED_FEATURE_COUNT)
        self.assertEqual(len(registry.entries), EXPECTED_FEATURE_COUNT)

        # Unique feature names
        self.assertEqual(len(set(registry.feature_names)), EXPECTED_FEATURE_COUNT)

        # Check individual branch counts
        for branch, expected in EXPECTED_BRANCH_COUNTS.items():
            feats = registry.get_features_by_branch(branch)
            self.assertEqual(
                len(feats),
                expected,
                f"Branch {branch} expected {expected} features, got {len(feats)}",
            )

    def test_02_feature_registry_metadata_and_export(self):
        """Verify registry metadata fields and export to CSV / JSON."""
        registry = FeatureRegistry()
        entry_0 = registry.get_by_index(0)
        self.assertIsNotNone(entry_0.name)
        self.assertIsNotNone(entry_0.branch)
        self.assertIsNotNone(entry_0.sub_branch)
        self.assertEqual(entry_0.data_type, "float32")

        csv_path = os.path.join(self.temp_dir, "test_registry.csv")
        json_path = os.path.join(self.temp_dir, "test_registry.json")
        registry.export_csv(csv_path)
        registry.export_json(json_path)

        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), EXPECTED_FEATURE_COUNT)

    def test_03_validity_analyzer_finite_and_stats(self):
        """Verify validity statistics, quantiles, and class-stratified metrics."""
        analyzer = FeatureValidityAnalyzer()
        X = np.random.randn(100, 3)
        y = np.array([0] * 50 + [1] * 50)
        names = ["feat_a", "feat_b", "feat_c"]

        stats = analyzer.analyze_dataset(X, names, y)
        self.assertEqual(len(stats), 3)

        df = analyzer.to_dataframe(stats)
        self.assertEqual(len(df), 3)
        self.assertTrue(np.all(df["finite_rate"] == 1.0))
        self.assertFalse(df["is_constant"].iloc[0])
        self.assertIn("mean_real", df.columns)
        self.assertIn("mean_ai", df.columns)

    def test_04_validity_analyzer_pathological_inputs(self):
        """Verify validity analyzer safely handles constant, zero, NaN, and Inf inputs."""
        analyzer = FeatureValidityAnalyzer()
        X = np.zeros((50, 2))
        X[0, 0] = np.nan
        X[1, 0] = np.inf
        y = np.array([0] * 25 + [1] * 25)

        stats = analyzer.analyze_dataset(X, ["nan_inf_feat", "constant_zero"], y)
        self.assertEqual(stats[0].nan_count, 1)
        self.assertEqual(stats[0].inf_count, 1)
        self.assertTrue(stats[1].is_constant)
        self.assertTrue(stats[1].near_zero_variance)

    def test_05_relevance_analyzer_effective_auc_symmetry(self):
        """Verify effective AUC correctly maps both positive and inverse discriminators to high AUC."""
        analyzer = FeatureRelevanceAnalyzer(random_state=42)
        X = np.array([
            [1.0, 10.0, 5.0],
            [2.0, 8.0, 5.0],
            [3.0, 6.0, 5.0],
            [4.0, 4.0, 5.0],
            [5.0, 2.0, 5.0],
            [6.0, 0.0, 5.0],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        names = ["pos_corr", "inv_corr", "constant"]

        stats = analyzer.analyze_dataset(X, names, y)
        df = analyzer.to_dataframe(stats)

        pos_stat = df[df["name"] == "pos_corr"].iloc[0]
        inv_stat = df[df["name"] == "inv_corr"].iloc[0]
        const_stat = df[df["name"] == "constant"].iloc[0]

        self.assertEqual(pos_stat["raw_auc"], 1.0)
        self.assertEqual(pos_stat["effective_auc"], 1.0)
        self.assertEqual(pos_stat["direction"], 1)

        self.assertEqual(inv_stat["raw_auc"], 0.0)
        self.assertEqual(inv_stat["effective_auc"], 1.0)
        self.assertEqual(inv_stat["direction"], -1)

        self.assertEqual(const_stat["effective_auc"], 0.5)

    def test_06_relevance_analyzer_mutual_information(self):
        """Verify mutual information estimation produces non-negative, finite scores."""
        analyzer = FeatureRelevanceAnalyzer(random_state=42)
        X = np.random.randn(80, 4)
        y = np.random.randint(0, 2, size=80)
        names = [f"f_{i}" for i in range(4)]

        stats = analyzer.analyze_dataset(X, names, y)
        for s in stats:
            self.assertGreaterEqual(s.mutual_info, 0.0)
            self.assertTrue(np.isfinite(s.mutual_info))

    def test_07_redundancy_analyzer_matrices(self):
        """Verify 111x111 Pearson and Spearman correlation matrices are symmetric with unity diagonal."""
        analyzer = FeatureRedundancyAnalyzer(high_correlation_threshold=0.90)
        names = [f"feat_{i}" for i in range(111)]
        X = np.random.randn(50, 111)

        p_df, s_df = analyzer.compute_correlation_matrices(X, names)
        self.assertEqual(p_df.shape, (111, 111))
        self.assertEqual(s_df.shape, (111, 111))

        # Check diagonal
        np.testing.assert_allclose(np.diag(p_df.values), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.diag(s_df.values), 1.0, atol=1e-5)

        # Check symmetry
        np.testing.assert_allclose(p_df.values, p_df.values.T, atol=1e-5)
        np.testing.assert_allclose(s_df.values, s_df.values.T, atol=1e-5)

    def test_08_redundancy_analyzer_pair_flagging(self):
        """Verify high-redundancy pair detection correctly flags |rho| >= threshold."""
        analyzer = FeatureRedundancyAnalyzer(high_correlation_threshold=0.90)
        names = ["f0", "f1", "f2"]
        X = np.random.randn(60, 3)
        X[:, 1] = X[:, 0] * 1.0  # Perfect correlation
        b_map = {"f0": "A", "f1": "A", "f2": "B"}

        p_df, s_df = analyzer.compute_correlation_matrices(X, names)
        pairs = analyzer.find_redundant_pairs(s_df, p_df, b_map)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].feature_1, "f0")
        self.assertEqual(pairs[0].feature_2, "f1")
        self.assertAlmostEqual(pairs[0].abs_spearman, 1.0, places=4)

    def test_09_branch_complementarity_summaries_and_ablations(self):
        """Verify branch ablation configurations generate expected counts."""
        registry = FeatureRegistry()
        analyzer = BranchComplementarityAnalyzer(registry)
        configs = analyzer.generate_ablation_configs()

        self.assertEqual(len(configs), 14)
        config_names = [c.config_name for c in configs]
        self.assertIn("FULL_111", config_names)
        self.assertIn("BRANCH_A", config_names)
        self.assertIn("BRANCH_B", config_names)
        self.assertIn("BRANCH_C_LBP", config_names)
        self.assertIn("BRANCH_D_MFR", config_names)
        self.assertIn("BRANCH_E", config_names)
        self.assertIn("WITHOUT_A", config_names)
        self.assertIn("WITHOUT_E", config_names)

    def test_10_generator_stability_diagnostic(self):
        """Verify generator stability calculations across distinct generator classes."""
        analyzer = GeneratorStabilityAnalyzer({"f0": "A", "f1": "B"})
        X = np.random.randn(60, 2)
        y_bin = np.array([0] * 20 + [1] * 40)
        y_gen = np.array([0] * 20 + [1] * 20 + [2] * 20)
        names = ["f0", "f1"]
        gen_names = {0: "Real", 1: "SD21", 2: "Midjourney"}

        stats = analyzer.analyze_dataset(X, names, y_bin, y_gen, gen_names)
        self.assertEqual(len(stats), 2)
        df = analyzer.to_dataframe(stats)

        self.assertIn("auc_sd21", df.columns)
        self.assertIn("auc_midjourney", df.columns)
        self.assertIn("mean_generator_auc", df.columns)
        self.assertIn("worst_generator_auc", df.columns)

    def test_11_compression_sensitivity_symmetry(self):
        """Verify symmetric JPEG compression generates valid [3, 256, 256] tensors in [0, 1]."""
        x = torch.rand(3, 256, 256, dtype=torch.float32)
        for q in (95, 80, 60, 40, 20):
            compressed = apply_symmetric_jpeg_compression(x, q)
            self.assertEqual(compressed.shape, (3, 256, 256))
            self.assertEqual(compressed.dtype, torch.float32)
            self.assertGreaterEqual(float(compressed.min()), 0.0)
            self.assertLessEqual(float(compressed.max()), 1.0)

    def test_12_mrmr_selector_deterministic_ranking_and_budgets(self):
        """Verify mRMR selector produces deterministic rankings and exact budget subsets."""
        selector = MRMRFeatureSelector(branch_map={"f0": "A", "f1": "A", "f2": "B", "f3": "C", "f4": "D"})
        names = ["f0", "f1", "f2", "f3", "f4"]
        rel = np.array([0.9, 0.88, 0.5, 0.7, 0.3])
        corr = np.array([
            [1.0, 0.99, 0.1, 0.2, 0.05],
            [0.99, 1.0, 0.1, 0.2, 0.05],
            [0.1, 0.1, 1.0, 0.05, 0.1],
            [0.2, 0.2, 0.05, 1.0, 0.05],
            [0.05, 0.05, 0.1, 0.05, 1.0],
        ])

        ranked_1 = selector.fit_ranking(rel, corr, names)
        ranked_2 = selector.fit_ranking(rel, corr, names)

        # Determinism
        self.assertEqual([f.name for f in ranked_1], [f.name for f in ranked_2])

        # Subsets
        subsets = selector.get_selected_subsets(ranked_1, budgets=(2, 4))
        self.assertEqual(len(subsets[2]), 2)
        self.assertEqual(len(subsets[4]), 4)
        self.assertEqual(len(set(subsets[4])), 4)

    def test_13_mrmr_train_only_leakage_isolation(self):
        """Verify that mRMR ranking depends strictly on train relevance and correlation."""
        selector = MRMRFeatureSelector()
        names = ["f0", "f1", "f2"]
        train_rel = np.array([0.8, 0.2, 0.5])
        train_corr = np.eye(3)

        ranked = selector.fit_ranking(train_rel, train_corr, names)
        # In uncorrelated setting, order must follow train relevance exactly
        self.assertEqual([f.name for f in ranked], ["f0", "f2", "f1"])

    def test_14_downstream_validator_lightgbm(self):
        """Verify LightGBM downstream validator runs without error and returns all metrics."""
        validator = DownstreamFeatureValidator(n_estimators=10)
        X_tr = np.random.randn(80, 5)
        y_tr = np.array([0] * 40 + [1] * 40)
        X_v = np.random.randn(40, 5)
        y_v = np.array([0] * 20 + [1] * 20)
        names = [f"f_{i}" for i in range(5)]

        subsets = {5: names, 2: names[:2]}
        results = validator.run_budget_experiments(subsets, names, X_tr, y_tr, X_v, y_v)

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertGreaterEqual(r.roc_auc, 0.0)
            self.assertLessEqual(r.roc_auc, 1.0)
            self.assertGreater(r.train_time_sec, 0.0)

    def test_15_runner_end_to_end_smoke(self):
        """Verify FeatureAnalysisRunner end-to-end execution on synthetic feature matrices."""
        runner = FeatureAnalysisRunner(output_dir=os.path.join(self.temp_dir, "analysis_out"))
        N_tr, N_v = 60, 30
        D = 111
        X_tr = np.random.randn(N_tr, D)
        y_tr = np.array([0] * 30 + [1] * 30)
        X_v = np.random.randn(N_v, D)
        y_v = np.array([0] * 15 + [1] * 15)
        gen_v = np.array([0] * 15 + [1] * 5 + [2] * 5 + [3] * 5)

        comp_tensors = [torch.rand(3, 256, 256) for _ in range(4)]
        comp_labels = np.array([0, 0, 1, 1])

        artifacts = runner.run_analysis(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_v,
            y_val=y_v,
            generator_labels_val=gen_v,
            compression_sample_tensors=comp_tensors,
            compression_sample_labels=comp_labels,
        )

        self.assertIn("feature_registry_csv", artifacts)
        self.assertIn("feature_validity_csv", artifacts)
        self.assertIn("univariate_relevance_csv", artifacts)
        self.assertIn("pearson_matrix_csv", artifacts)
        self.assertIn("spearman_matrix_csv", artifacts)
        self.assertIn("redundancy_pairs_csv", artifacts)
        self.assertIn("branch_redundancy_csv", artifacts)
        self.assertIn("branch_complementarity_csv", artifacts)
        self.assertIn("generator_stability_csv", artifacts)
        self.assertIn("compression_analysis_csv", artifacts)
        self.assertIn("mrmr_rankings_csv", artifacts)
        self.assertIn("selected_features_8", artifacts)
        self.assertIn("selected_features_64", artifacts)
        self.assertIn("validation_results_csv", artifacts)
        self.assertIn("run_metadata_json", artifacts)

        for path in artifacts.values():
            self.assertTrue(os.path.exists(path), f"Artifact missing: {path}")

    def test_16_branch_f_pipeline_facade(self):
        """Verify BranchFPipeline facade exposes feature names and runs end-to-end."""
        pipeline = BranchFPipeline(output_dir=os.path.join(self.temp_dir, "facade_out"))
        self.assertEqual(len(pipeline.feature_names), 111)

        N_tr, N_v = 40, 20
        D = 111
        X_tr = np.random.randn(N_tr, D)
        y_tr = np.array([0] * 20 + [1] * 20)
        X_v = np.random.randn(N_v, D)
        y_v = np.array([0] * 10 + [1] * 10)

        artifacts = pipeline.run_analysis_and_selection(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_v,
            y_val=y_v,
        )
        self.assertIn("mrmr_rankings_csv", artifacts)
        self.assertTrue(os.path.exists(artifacts["mrmr_rankings_csv"]))


if __name__ == "__main__":
    unittest.main()
