"""Unit tests for Candidate A: Raw NQP Baseline."""

from __future__ import annotations

import unittest
import torch

from analysis.branch_b_screening.candidates.raw_nqp import (
    compute_raw_nqp_residual,
    compute_raw_nqp_from_noise,
    derive_sample_seed,
)


class TestRawNQP(unittest.TestCase):
    """Test suite for Raw NQP mathematical operations and properties."""

    def test_01_seed_derivation_determinism(self):
        """Verify SHA-256 seed derivation is deterministic and sample-dependent."""
        seed1 = derive_sample_seed(42, "img_100")
        seed2 = derive_sample_seed(42, "img_100")
        seed3 = derive_sample_seed(42, "img_101")
        self.assertEqual(seed1, seed2)
        self.assertNotEqual(seed1, seed3)
        self.assertIsInstance(seed1, int)
        self.assertGreaterEqual(seed1, 0)
        self.assertLessEqual(seed1, 2**32 - 1)

    def test_02_contract_shapes_and_dtype(self):
        """Verify output contract: residual shape [3, 256, 256], dtype float32, finite."""
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        result = compute_raw_nqp_residual(img, sample_id=123, replicas=5)

        res = result["residual"]
        self.assertEqual(res.shape, (3, 256, 256))
        self.assertEqual(res.dtype, torch.float32)
        self.assertTrue(torch.all(torch.isfinite(res)))
        self.assertEqual(result["residual_sign"], "I - I_hat")
        self.assertEqual(result["output_scale"], "raw_uint8_intensity")

    def test_03_controlled_noise_threshold_crossing(self):
        """Test exact threshold crossing and residual sign using injected noise."""
        img = torch.full((3, 4, 4), 100, dtype=torch.uint8)
        # R=2 replicas: replica 0 has noise +0.6 (crosses round boundary -> 101)
        # replica 1 has noise 0.0 (stays 100)
        noise = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
        noise[0, 0, 0, 0] = 0.6  # round(100.6) = 101

        res = compute_raw_nqp_from_noise(img, noise)
        # Quantized replica 0: 101, replica 1: 100 -> mean = 100.5
        # Residual = I - I_hat = 100.0 - 100.5 = -0.5
        self.assertAlmostEqual(res[0, 0, 0].item(), -0.5, places=5)
        # All other pixels had zero noise -> residual is exactly 0.0
        self.assertAlmostEqual(res[0, 1, 1].item(), 0.0, places=5)

    def test_04_controlled_noise_cancellation(self):
        """Test upward and downward replica cancellation."""
        img = torch.full((3, 4, 4), 100, dtype=torch.uint8)
        # R=2 replicas: replica 0 has +0.6 (quantizes to 101), replica 1 has -0.6 (quantizes to 99)
        noise = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
        noise[0, 0, 0, 0] = 0.6
        noise[1, 0, 0, 0] = -0.6

        res = compute_raw_nqp_from_noise(img, noise)
        # Mean quantized = (101 + 99) / 2 = 100.0 -> residual = 100 - 100 = 0.0
        self.assertAlmostEqual(res[0, 0, 0].item(), 0.0, places=5)

    def test_05_boundary_clipping_behavior(self):
        """Test boundary clipping at 0 and 255."""
        img = torch.zeros((3, 4, 4), dtype=torch.uint8)
        img[:, 0, 0] = 0
        img[:, 1, 1] = 255

        # Noise -2.0 on 0 -> clamped to 0
        # Noise +2.0 on 255 -> clamped to 255
        noise = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        noise[0, :, 0, 0] = -2.0
        noise[0, :, 1, 1] = 2.0

        res = compute_raw_nqp_from_noise(img, noise)
        # 0 + (-2) = -2 -> clamp(round(-2), 0, 255) = 0 -> res = 0 - 0 = 0
        self.assertAlmostEqual(res[0, 0, 0].item(), 0.0, places=5)
        # 255 + 2 = 257 -> clamp(round(257), 0, 255) = 255 -> res = 255 - 255 = 0
        self.assertAlmostEqual(res[0, 1, 1].item(), 0.0, places=5)

    def test_06_integer_degeneracy_empirical_property(self):
        """On exact interior integer image with sigma=0.10, residual is nearly all zero."""
        img = torch.randint(10, 240, (3, 64, 64), dtype=torch.uint8)
        res = compute_raw_nqp_residual(img, sample_id=42, replicas=50, sigma=0.10)["residual"]
        nonzero_fraction = (res.abs() > 1e-4).float().mean().item()
        # With 3*64*64 = 12,288 scalars, expected nonzero is < 0.1%
        self.assertLess(nonzero_fraction, 0.01)


if __name__ == "__main__":
    unittest.main()
