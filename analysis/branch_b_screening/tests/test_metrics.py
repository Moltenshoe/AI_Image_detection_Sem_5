"""Unit tests for Scale-Aware Metrics, Anti-Collapse, and Spatial Statistics."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from analysis.branch_b_screening.diagnostics.metrics import (
    compute_scalar_metrics,
    compute_per_image_metrics,
    assess_numerical_degeneracy,
)
from analysis.branch_b_screening.diagnostics.intensity_profiles import (
    compute_intensity_conditioned_profile,
)
from analysis.branch_b_screening.diagnostics.spatial_metrics import (
    compute_morans_i,
    compute_gini_coefficient,
)


class TestScreeningMetrics(unittest.TestCase):
    """Test suite for metrics, degeneracy assessment, and spatial statistics."""

    def test_01_all_zero_map_degeneracy(self):
        """Verify all-zero residual map is flagged as degenerate."""
        zero_map = torch.zeros((3, 32, 32), dtype=torch.float32)
        res = compute_per_image_metrics(zero_map, is_normalized=True)
        self.assertTrue(res["is_degenerate"])
        self.assertEqual(res["combined"]["exact_zero_fraction"], 1.0)
        self.assertEqual(res["combined"]["tolerance_zero_fraction"], 1.0)
        self.assertEqual(res["combined"]["dynamic_range"], 0.0)
        self.assertEqual(res["pixel_nonzero_fraction"], 0.0)

    def test_02_constant_nonzero_map_degeneracy(self):
        """Verify constant nonzero map has zero dynamic range and is degenerate."""
        const_map = torch.full((3, 32, 32), 0.5, dtype=torch.float32)
        res = compute_per_image_metrics(const_map, is_normalized=True)
        self.assertTrue(res["is_degenerate"])
        self.assertEqual(res["combined"]["dynamic_range"], 0.0)
        self.assertIn("collapsed_dynamic_range(DR=0.00e+00<=tau)", res["combined"]["degeneracy_reasons"])

    def test_03_sparse_representation_preservation(self):
        """Verify valid sparse representation is preserved (non-degenerate)."""
        # 98% zeros, 2% high-magnitude structural edges (|Delta| = 0.5 >> tau)
        sparse_map = torch.zeros((3, 100, 100), dtype=torch.float32)
        sparse_map[:, :2, :50] = 0.5  # 2% active pixels
        res = compute_per_image_metrics(sparse_map, is_normalized=True)

        # Must NOT be classified as degenerate because DR >> tau and MaxAR >> tau
        self.assertFalse(res["is_degenerate"])
        self.assertGreater(res["combined"]["max_ar"], 0.1)
        self.assertGreater(res["combined"]["dynamic_range"], 0.1)
        self.assertGreater(res["combined"]["tolerance_zero_fraction"], 0.95)

    def test_04_single_outlier_map(self):
        """Verify single outlier pixel is tracked in MaxAR and DR."""
        outlier_map = torch.zeros((3, 32, 32), dtype=torch.float32)
        outlier_map[0, 5, 5] = 1.0
        res = compute_per_image_metrics(outlier_map, is_normalized=True)
        self.assertFalse(res["is_degenerate"])
        self.assertEqual(res["combined"]["max_ar"], 1.0)
        self.assertEqual(res["pixel_nonzero_count"], 1)

    def test_05_nan_and_inf_detection(self):
        """Verify presence of NaN or Inf is explicitly detected."""
        bad_map = torch.zeros((3, 16, 16), dtype=torch.float32)
        bad_map[0, 0, 0] = float("nan")
        bad_map[1, 1, 1] = float("inf")
        res = compute_scalar_metrics(bad_map, is_normalized=True)
        self.assertEqual(res["nan_count"], 1)
        self.assertEqual(res["inf_count"], 1)

    def test_06_intensity_profile_empty_bins_handling(self):
        """Verify empty intensity bins return None, not 0.0."""
        # Image with pixels only in [0.0, 0.1]
        img = torch.full((3, 32, 32), 0.05, dtype=torch.float32)
        res = torch.full((3, 32, 32), 0.01, dtype=torch.float32)
        profile = compute_intensity_conditioned_profile(img, res, is_normalized=True, num_bins=16)

        # Bin 0 has all pixels
        self.assertTrue(profile["bins"][0]["combined"]["available"])
        self.assertEqual(profile["bins"][0]["combined"]["count"], 3 * 32 * 32)
        # Bins 1..15 are empty and must have mean=None
        for k in range(1, 16):
            self.assertFalse(profile["bins"][k]["combined"]["available"])
            self.assertIsNone(profile["bins"][k]["combined"]["mean"])
            self.assertIsNone(profile["bins"][k]["combined"]["var"])

    def test_07_spatial_morans_i_calculation(self):
        """Verify Moran's I on clustered vs random spatial patterns."""
        # Clustered pattern (half positive, half negative) -> high positive Moran's I
        clustered = torch.zeros((64, 64), dtype=torch.float32)
        clustered[:32, :] = 1.0
        clustered[32:, :] = -1.0
        m_clustered = compute_morans_i(clustered)
        self.assertEqual(m_clustered["status"], "valid")
        self.assertGreater(m_clustered["morans_i"], 0.8)

        # Constant map -> undefined status
        const_map = torch.ones((32, 32), dtype=torch.float32)
        m_const = compute_morans_i(const_map)
        self.assertIsNone(m_const["morans_i"])
        self.assertEqual(m_const["status"], "undefined_constant_or_zero_variance")

    def test_08_spatial_gini_coefficient(self):
        """Verify Gini coefficient on equal vs concentrated maps."""
        # Perfectly equal nonzero values -> Gini = 0.0
        equal_map = torch.ones((32, 32), dtype=torch.float32)
        g_equal = compute_gini_coefficient(equal_map)
        self.assertAlmostEqual(g_equal["gini"], 0.0, places=4)

        # Single active pixel among zeros -> Gini ~ 1.0
        extreme_sparse = torch.zeros((32, 32), dtype=torch.float32)
        extreme_sparse[0, 0] = 100.0
        g_sparse = compute_gini_coefficient(extreme_sparse)
        self.assertGreater(g_sparse["gini"], 0.95)

    def test_09_row_standardized_morans_i_analytical_known_cases(self):
        """Verify row-standardized Moran's I matches analytical solutions on known grids."""
        # 1. 2x2 Checkerboard: [[1, 0], [0, 1]] -> analytical Moran's I is exactly -1.0
        cb_2x2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        res_cb = compute_morans_i(cb_2x2)
        self.assertEqual(res_cb["status"], "valid")
        self.assertAlmostEqual(res_cb["morans_i"], -1.0, places=6)

        # 2. 2x2 Equal Rows: [[1, 1], [0, 0]] -> analytical Moran's I is exactly 0.0
        rows_2x2 = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        res_rows = compute_morans_i(rows_2x2)
        self.assertEqual(res_rows["status"], "valid")
        self.assertAlmostEqual(res_rows["morans_i"], 0.0, places=6)

        # 3. 3x3 Grid: verify fast slice computation exactly matches dense matrix definition
        grid_3x3 = np.array([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ], dtype=np.float64)
        H, W = grid_3x3.shape
        N = H * W
        z = grid_3x3.flatten() - np.mean(grid_3x3)
        ss = np.sum(z ** 2)

        # Construct explicit dense row-standardized weight matrix
        W_dense = np.zeros((N, N), dtype=np.float64)
        for r in range(H):
            for c in range(W):
                i = r * W + c
                neighbors = []
                if r > 0: neighbors.append((r - 1) * W + c)
                if r < H - 1: neighbors.append((r + 1) * W + c)
                if c > 0: neighbors.append(r * W + (c - 1))
                if c < W - 1: neighbors.append(r * W + (c + 1))
                for j in neighbors:
                    W_dense[i, j] = 1.0 / float(len(neighbors))

        expected_morans_i = float(np.sum(z * (W_dense @ z)) / ss)
        res_fast = compute_morans_i(grid_3x3)
        self.assertAlmostEqual(res_fast["morans_i"], expected_morans_i, places=10)


if __name__ == "__main__":
    unittest.main()
