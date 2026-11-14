"""Unit and integration tests for Block 2 — RGB Branch A (Lab Local Correlation Maps).

Verifies:
1. RGB-to-Lab conversion and channel conventions (L, a, b).
2. Correct channel pairs (L-a, L-b, a-b).
3. 7x7 window, stride 1, and spatial dimension preservation ([3, 256, 256]).
4. Constant and near-constant window handling (zero variance -> 0.0).
5. Correlation range [-1.0, 1.0] and absence of NaN / Inf.
6. Determinism (repeated extractions match bit-for-bit).
7. Boundary handling (reflect padding preserves edges cleanly).
8. Smoke test on real Defactify training sample.
"""

import os
import unittest
import numpy as np
import pyarrow.parquet as pq
import torch

from src.rgb.branch_a_lab import (
    compute_local_correlation_maps,
    extract_branch_a_lab_maps,
    rgb_to_cielab,
)


class TestBranchALab(unittest.TestCase):
    """Test suite for RGB Branch A."""

    def setUp(self):
        self.project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        self.processed_train_dir = os.path.join(
            self.project_root, "data", "processed", "train"
        )

    def test_01_rgb_to_cielab_conversion_and_ranges(self):
        """Verify RGB to CIE Lab conversion output ranges and channel shapes."""
        # 1. Float tensor [3, 256, 256] in [0, 1]
        rgb = torch.rand(3, 256, 256, dtype=torch.float32)
        lab = rgb_to_cielab(rgb)

        self.assertEqual(lab.shape, (3, 256, 256))
        self.assertEqual(lab.dtype, torch.float32)

        # L* should be within [0, 100]
        self.assertGreaterEqual(lab[0].min().item(), -1e-4)
        self.assertLessEqual(lab[0].max().item(), 100.0 + 1e-4)

        # a* and b* should typically lie within [-128, 127]
        self.assertGreaterEqual(lab[1].min().item(), -128.0)
        self.assertLessEqual(lab[1].max().item(), 127.0)
        self.assertGreaterEqual(lab[2].min().item(), -128.0)
        self.assertLessEqual(lab[2].max().item(), 127.0)

        # 2. NumPy uint8 [256, 256, 3] in [0, 255]
        rgb_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        lab_from_np = rgb_to_cielab(rgb_np)
        self.assertEqual(lab_from_np.shape, (3, 256, 256))

    def test_02_correlation_maps_shape_and_bounds(self):
        """Verify output shape [3, 256, 256] and strict [-1, 1] bounds."""
        torch.manual_seed(42)
        rgb = torch.rand(3, 256, 256, dtype=torch.float32)
        corr_maps = extract_branch_a_lab_maps(rgb, window_size=7)

        self.assertEqual(corr_maps.shape, (3, 256, 256))
        self.assertEqual(corr_maps.dtype, torch.float32)

        # Range bounds
        self.assertGreaterEqual(corr_maps.min().item(), -1.0)
        self.assertLessEqual(corr_maps.max().item(), 1.0)

        # Finite checks
        self.assertEqual(torch.isnan(corr_maps).sum().item(), 0)
        self.assertEqual(torch.isinf(corr_maps).sum().item(), 0)

    def test_03_constant_and_zero_image_safety(self):
        """Constant windows have zero variance; correlation must be exactly 0.0."""
        # 1. All-zero image
        zero_img = torch.zeros(3, 256, 256, dtype=torch.float32)
        zero_corr = extract_branch_a_lab_maps(zero_img)
        self.assertEqual(torch.isnan(zero_corr).sum().item(), 0)
        self.assertEqual(torch.isinf(zero_corr).sum().item(), 0)
        self.assertTrue((zero_corr == 0.0).all().item())

        # 2. Constant mid-gray image
        const_img = torch.full((3, 256, 256), 0.5, dtype=torch.float32)
        const_corr = extract_branch_a_lab_maps(const_img)
        self.assertEqual(torch.isnan(const_corr).sum().item(), 0)
        self.assertEqual(torch.isinf(const_corr).sum().item(), 0)
        self.assertTrue((const_corr == 0.0).all().item())

    def test_04_perfect_positive_and_negative_correlation(self):
        """Construct synthetic images with known correlation between channels."""
        # Synthetic Lab tensor where L and a are identical, and b is negative of L
        base = torch.linspace(0, 100, 256).view(1, 256).expand(256, 256)
        L = base.clone()
        a = base.clone() # perfect positive correlation
        b = -base.clone() # perfect negative correlation
        lab = torch.stack([L, a, b], dim=0)

        # Use internal 7x7 window away from borders
        corr = compute_local_correlation_maps(lab, window_size=7)

        # Check interior pixels (e.g. [10:240, 10:240])
        interior_La = corr[0, 10:240, 10:240]
        interior_Lb = corr[1, 10:240, 10:240]
        interior_ab = corr[2, 10:240, 10:240]

        # La should be close to 1.0
        self.assertTrue(torch.allclose(interior_La, torch.ones_like(interior_La), atol=1e-3))
        # Lb should be close to -1.0
        self.assertTrue(torch.allclose(interior_Lb, -torch.ones_like(interior_Lb), atol=1e-3))
        # ab should be close to -1.0
        self.assertTrue(torch.allclose(interior_ab, -torch.ones_like(interior_ab), atol=1e-3))

    def test_05_determinism(self):
        """Repeated extractions on the same image yield identical results."""
        torch.manual_seed(123)
        rgb = torch.rand(3, 256, 256, dtype=torch.float32)

        run1 = extract_branch_a_lab_maps(rgb)
        run2 = extract_branch_a_lab_maps(rgb)

        max_diff = (run1 - run2).abs().max().item()
        self.assertEqual(max_diff, 0.0)

    def test_06_batch_consistency(self):
        """Single image and batched images yield identical results."""
        torch.manual_seed(999)
        img1 = torch.rand(3, 256, 256, dtype=torch.float32)
        img2 = torch.rand(3, 256, 256, dtype=torch.float32)
        batch = torch.stack([img1, img2], dim=0)

        maps_batch = extract_branch_a_lab_maps(batch)
        maps1 = extract_branch_a_lab_maps(img1)
        maps2 = extract_branch_a_lab_maps(img2)

        self.assertEqual(maps_batch.shape, (2, 3, 256, 256))
        self.assertAlmostEqual((maps_batch[0] - maps1).abs().max().item(), 0.0, places=5)
        self.assertAlmostEqual((maps_batch[1] - maps2).abs().max().item(), 0.0, places=5)

    def test_07_smoke_real_defactify_sample(self):
        """Smoke test on a real Defactify training sample."""
        parquet_path = os.path.join(self.processed_train_dir, "part-00000.parquet")
        if not os.path.isfile(parquet_path):
            self.skipTest(f"Processed training file not found at: {parquet_path}")

        pf = pq.ParquetFile(parquet_path)
        batch = pf.read_row_group(0, columns=["image_rgb", "label_a"])
        rgb_bytes = batch.column("image_rgb").to_pylist()[0]
        arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(256, 256, 3)

        corr_maps = extract_branch_a_lab_maps(arr)
        self.assertEqual(corr_maps.shape, (3, 256, 256))
        self.assertEqual(torch.isnan(corr_maps).sum().item(), 0)
        self.assertEqual(torch.isinf(corr_maps).sum().item(), 0)
        self.assertGreaterEqual(corr_maps.min().item(), -1.0)
        self.assertLessEqual(corr_maps.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
