"""Unit tests for Block 2 — RGB Branch B (Noise Quantization Probe).

Verifies:
1. Probe contract, output shapes, and signed residual preservation.
2. Deterministic seed derivation from sample ID and base seed.
3. Deterministic repeatability across repeated runs.
4. Mathematical degeneracy on integer inputs under sigma=0.10.
5. Probe behavior on non-integer inputs.
6. FP16 vs FP32 precision preservation.
"""

import os
import unittest
import numpy as np
import torch

from src.rgb.branch_b_nqp import (
    compute_nqp_residual,
    derive_sample_seed,
)


class TestBranchBNQP(unittest.TestCase):
    """Test suite for RGB Branch B."""

    def test_01_seed_derivation(self):
        """Verify deterministic seed derivation from sample_id."""
        s1 = derive_sample_seed("train_000001", base_seed=42)
        s2 = derive_sample_seed("train_000001", base_seed=42)
        s3 = derive_sample_seed("train_000002", base_seed=42)
        s4 = derive_sample_seed("train_000001", base_seed=100)

        # Same input -> same seed
        self.assertEqual(s1, s2)
        self.assertIsInstance(s1, int)
        self.assertGreaterEqual(s1, 0)

        # Different sample ID -> different seed
        self.assertNotEqual(s1, s3)

        # Different base seed -> different seed
        self.assertNotEqual(s1, s4)

    def test_02_probe_contract_and_shapes(self):
        """Verify output shape [3, 256, 256] and dtype float32."""
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        delta = compute_nqp_residual(img, sample_id="test_001", sigma=0.10, replicas=10)

        self.assertEqual(delta.shape, (3, 256, 256))
        self.assertEqual(delta.dtype, torch.float32)
        self.assertEqual(torch.isnan(delta).sum().item(), 0)
        self.assertEqual(torch.isinf(delta).sum().item(), 0)

    def test_03_determinism_and_repeatability(self):
        """Repeated calls with same sample_id and base_seed yield identical residuals."""
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        delta1 = compute_nqp_residual(img, sample_id="sample_abc", sigma=0.10, replicas=20, base_seed=42)
        delta2 = compute_nqp_residual(img, sample_id="sample_abc", sigma=0.10, replicas=20, base_seed=42)

        max_diff = (delta1 - delta2).abs().max().item()
        self.assertEqual(max_diff, 0.0)

    def test_04_integer_degeneracy_verification(self):
        """On exact integer uint8 inputs with sigma=0.10, residuals are nearly zero."""
        img = torch.randint(10, 245, (3, 256, 256), dtype=torch.uint8)
        delta = compute_nqp_residual(img, sample_id="int_sample", sigma=0.10, replicas=50, base_seed=42)

        nonzero_fraction = (delta != 0.0).float().mean().item()
        # For integer pixels with N(0, 0.01) noise, P(|N| > 0.5) is ~ 5.7e-7.
        # With R=50, fraction of nonzero scalars should be extremely small (< 0.001)
        self.assertLess(nonzero_fraction, 0.001)
        self.assertLess(delta.abs().mean().item(), 1e-4)

    def test_05_fractional_input_behavior(self):
        """When input values have fractional parts (e.g. 0.5), rounding creates active residuals."""
        # Create image with exact half-integers e.g. 10.5
        img_half = torch.full((3, 64, 64), 10.5, dtype=torch.float32)
        delta = compute_nqp_residual(img_half, sample_id="half_sample", sigma=0.10, replicas=50, base_seed=42)

        # With 10.5 + N(0, 0.1^2), values cross 10.5 and round to 10 or 11 with ~50% probability
        nonzero_fraction = (delta != 0.0).float().mean().item()
        self.assertGreater(nonzero_fraction, 0.5)

    def test_06_fp16_storage_fidelity(self):
        """Verify FP16 conversion preserves nonzero residual values."""
        img_half = torch.full((3, 32, 32), 10.5, dtype=torch.float32)
        delta_fp32 = compute_nqp_residual(img_half, sample_id="fp16_test", sigma=0.10, replicas=50, base_seed=42)

        delta_fp16 = delta_fp32.half()
        delta_recovered = delta_fp16.float()

        # Nonzero positions in fp32 should match nonzero in fp16
        nonzero_fp32 = (delta_fp32 != 0.0)
        nonzero_fp16 = (delta_recovered != 0.0)
        self.assertEqual((nonzero_fp32 == nonzero_fp16).all().item(), True)

    def test_07_pid_residual_contract_and_shapes(self):
        """Verify active Branch B PiD YUV (s=0.2) output shape and contract."""
        from src.rgb.branch_b_nqp import compute_branch_b_residual, compute_pid_residual
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        delta = compute_pid_residual(img, s=0.2)

        self.assertEqual(delta.shape, (3, 256, 256))
        self.assertEqual(delta.dtype, torch.float32)
        self.assertEqual(int(torch.isnan(delta).sum().item()), 0)
        self.assertEqual(int(torch.isinf(delta).sum().item()), 0)

        # Check alias
        delta_alias = compute_branch_b_residual(img, s=0.2)
        self.assertTrue(torch.equal(delta, delta_alias))

    def test_08_pid_determinism(self):
        """Verify PiD YUV s=0.2 is strictly deterministic."""
        from src.rgb.branch_b_nqp import compute_pid_residual
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        delta1 = compute_pid_residual(img, s=0.2)
        delta2 = compute_pid_residual(img, s=0.2)

        self.assertTrue(torch.equal(delta1, delta2))

    def test_09_pid_non_degeneracy_and_density(self):
        """On integer uint8 inputs, PiD s=0.2 avoids degeneracy and yields active residuals."""
        from src.rgb.branch_b_nqp import compute_pid_residual
        img = torch.randint(10, 245, (3, 256, 256), dtype=torch.uint8)
        delta = compute_pid_residual(img, s=0.2)

        nonzero_fraction = (delta != 0.0).float().mean().item()
        mar = delta.abs().mean().item()

        # PiD s=0.2 produces ~85% nonzero scalars and MAR ~1.95 (non-degenerate)
        self.assertGreater(nonzero_fraction, 0.50)
        self.assertGreater(mar, 1.0)
        self.assertLessEqual(delta.abs().max().item(), 10.0)

    def test_10_pipeline_active_branch_b_integration(self):
        """Verify RGBPipeline extracts active PiD YUV s=0.2 for Branch B."""
        from src.rgb.pipeline import RGBPipeline
        pipeline = RGBPipeline(branches=("A", "B"))
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)

        outputs = pipeline.extract(img)
        self.assertIn("lab_maps", outputs)
        self.assertIn("branch_b_map", outputs)
        self.assertIn("pid_map", outputs)
        self.assertIn("nqp_map", outputs)

        # Confirm Branch B map matches PiD output
        from src.rgb.branch_b_nqp import compute_pid_residual
        expected_pid = compute_pid_residual(img, s=0.2)
        self.assertTrue(torch.equal(outputs["branch_b_map"], expected_pid))
        self.assertTrue(torch.equal(outputs["pid_map"], expected_pid))
        self.assertTrue(torch.equal(outputs["nqp_map"], expected_pid))


if __name__ == "__main__":
    unittest.main()

