"""Unit tests for Candidate C: PiD ICML 2025 Deterministic Color Decomposition."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from analysis.branch_b_screening.candidates.pid_yuv import (
    M_YUV_BT601,
    M_INV_VERIFIED,
    forward_yuv,
    inverse_yuv,
    compute_pid_residual,
)


class TestPiDYUV(unittest.TestCase):
    """Test suite for PiD color-space transformation matrices and residuals."""

    def test_01_matrix_inverse_identity(self):
        """Verify M_inv * M_fwd == I within numerical tolerance in float64."""
        product = np.matmul(M_INV_VERIFIED, M_YUV_BT601)
        identity = np.eye(3, dtype=np.float64)
        max_error = np.max(np.abs(product - identity))
        self.assertLess(max_error, 1e-4)

    def test_02_verified_inverse_third_row_blue_reconstruction(self):
        """Corroborate that third row of M_inv is [1.0, 1.772, 0.0] reconstructing Blue."""
        row3 = M_INV_VERIFIED[2]
        self.assertAlmostEqual(row3[0], 1.0, places=5)
        self.assertAlmostEqual(row3[1], 1.772, places=3)
        self.assertAlmostEqual(row3[2], 0.0, places=5)

        # For pure Blue [0, 0, 255]:
        rgb = np.array([0.0, 0.0, 255.0])
        yuv = np.dot(M_YUV_BT601, rgb)
        # B = 1.0*Y + 1.772*U + 0.0*V
        b_recon = 1.0 * yuv[0] + 1.772 * yuv[1] + 0.0 * yuv[2]
        self.assertAlmostEqual(b_recon, 255.0, places=1)

    def test_03_known_basis_vectors_roundtrip(self):
        """Test roundtrip transformations on known primary and neutral colors."""
        test_colors = {
            "black": torch.tensor([0, 0, 0], dtype=torch.uint8),
            "white": torch.tensor([255, 255, 255], dtype=torch.uint8),
            "gray": torch.tensor([128, 128, 128], dtype=torch.uint8),
            "red": torch.tensor([255, 0, 0], dtype=torch.uint8),
            "green": torch.tensor([0, 255, 0], dtype=torch.uint8),
            "blue": torch.tensor([0, 0, 255], dtype=torch.uint8),
        }

        for name, color in test_colors.items():
            img = color.view(3, 1, 1).repeat(1, 4, 4)
            res_dict = compute_pid_residual(img, s=1.0)
            res = res_dict["residual"]
            # Residual must be finite and bounded in [-255, 255]
            self.assertTrue(torch.all(torch.isfinite(res)))
            self.assertGreaterEqual(res.min().item(), -255.0)
            self.assertLessEqual(res.max().item(), 255.0)

            # Black and White are exact fixed points under integer rounding
            if name in ("black", "white", "gray"):
                self.assertLess(res.abs().max().item(), 2.0)

    def test_04_residual_sign_convention(self):
        """Verify residual sign convention is R_PiD = x - x_hat (original minus reconstructed)."""
        img = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
        res_dict = compute_pid_residual(img, s=1.0)
        self.assertEqual(res_dict["residual_sign"], "x - x_hat")
        self.assertEqual(res_dict["candidate_name"], "pid_yuv")

    def test_05_scaling_factors_ablation(self):
        """Verify execution across all six Table 5 scaling factors."""
        img = torch.randint(10, 240, (3, 16, 16), dtype=torch.uint8)
        scaling_factors = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]

        for s in scaling_factors:
            res_dict = compute_pid_residual(img, s=s)
            res = res_dict["residual"]
            self.assertEqual(res.shape, (3, 16, 16))
            self.assertTrue(torch.all(torch.isfinite(res)))
            self.assertEqual(res_dict["config"]["scale_factor_s"], s)

    def test_06_determinism(self):
        """Verify PiD is 100% deterministic (zero noise, R=1)."""
        img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        run1 = compute_pid_residual(img, s=1.0)["residual"]
        run2 = compute_pid_residual(img, s=1.0)["residual"]
        diff = torch.max(torch.abs(run1 - run2)).item()
        self.assertEqual(diff, 0.0)


if __name__ == "__main__":
    unittest.main()
