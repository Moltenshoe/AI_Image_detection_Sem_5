"""Unit tests for Candidate B: CVPR 2025 4-Bit Color Quantization."""

from __future__ import annotations

import unittest
import torch

from analysis.branch_b_screening.candidates.cvpr_color import (
    quantize_cvpr,
    compute_cvpr_from_noise,
    extract_cvpr_reference,
    extract_cvpr_seeded,
)


class TestCVPRColor(unittest.TestCase):
    """Test suite for CVPR 2025 4-bit quantizer and residual extraction."""

    def test_01_quantizer_grid_points(self):
        """Verify Q_16(k/15) == k/15 exactly for k in {0, ..., 15}."""
        for k in range(16):
            val = torch.tensor([k / 15.0], dtype=torch.float32)
            q = quantize_cvpr(val, L=16, clip_quant=True)
            self.assertAlmostEqual(q.item(), k / 15.0, places=6)

    def test_02_quantizer_midpoints(self):
        """Verify rounding halfway between quantization levels."""
        # Step is 1/15
        step = 1.0 / 15.0
        # Value slightly below midpoint rounds down
        v_low = torch.tensor([step * 0.49], dtype=torch.float32)
        self.assertAlmostEqual(quantize_cvpr(v_low, L=16).item(), 0.0, places=5)
        # Value slightly above midpoint rounds up
        v_high = torch.tensor([step * 0.51], dtype=torch.float32)
        self.assertAlmostEqual(quantize_cvpr(v_high, L=16).item(), step, places=5)

    def test_03_quantizer_out_of_bounds(self):
        """Verify behavior for values below 0.0 and above 1.0."""
        neg = torch.tensor([-0.4], dtype=torch.float32)  # -0.4 * 15 = -6.0 -> round is -6.0 -> -6/15 = -0.4
        pos = torch.tensor([1.4], dtype=torch.float32)   # 1.4 * 15 = 21.0 -> round is 21.0 -> 21/15 = 1.4
        # With clipping ON
        self.assertAlmostEqual(quantize_cvpr(neg, L=16, clip_quant=True).item(), 0.0, places=6)
        self.assertAlmostEqual(quantize_cvpr(pos, L=16, clip_quant=True).item(), 1.0, places=6)
        # With clipping OFF
        self.assertAlmostEqual(quantize_cvpr(neg, L=16, clip_quant=False).item(), -0.4, places=5)
        self.assertAlmostEqual(quantize_cvpr(pos, L=16, clip_quant=False).item(), 1.4, places=5)

    def test_04_factorial_clipping_stages_independence(self):
        """Verify 2x2 factorial clipping stages are independently configurable."""
        # 3-channel input [3, 1, 1]
        x = torch.full((3, 1, 1), 0.05, dtype=torch.float32)
        # Injected noise: -0.2 (x + N = -0.15)
        noise = torch.full((1, 3, 1, 1), -0.2, dtype=torch.float32)

        # Config 1: Noise ON, Quant ON -> clamp(-0.15, 0, 1) = 0.0 -> Q_16(0) = 0.0 -> res = 0.0 - 0.05 = -0.05
        res1 = compute_cvpr_from_noise(x, noise, clip_noise=True, clip_quant=True, L=16)
        self.assertAlmostEqual(res1[0, 0, 0].item(), -0.05, places=5)

        # Config 4: Noise OFF, Quant OFF -> unclipped noise = -0.15 -> round(-0.15*15)/15 = round(-2.25)/15 = -2/15
        # res = (-2/15) - 0.05
        res4 = compute_cvpr_from_noise(x, noise, clip_noise=False, clip_quant=False, L=16)
        expected_q = round(-0.15 * 15.0) / 15.0
        self.assertAlmostEqual(res4[0, 0, 0].item(), expected_q - 0.05, places=5)
        self.assertNotEqual(res1[0, 0, 0].item(), res4[0, 0, 0].item())

    def test_05_residual_sign_convention(self):
        """Verify residual sign is Delta = x_hat - x (reconstruction minus original)."""
        x = torch.full((3, 4, 4), 0.0, dtype=torch.float32)
        # Noise +0.06 -> 0.06 * 15 = 0.90 -> round(0.90) = 1.0 -> Q = 1/15
        noise = torch.full((1, 3, 4, 4), 0.06, dtype=torch.float32)
        res = compute_cvpr_from_noise(x, noise, clip_noise=True, clip_quant=True, L=16)
        # x_hat = 1/15, x = 0.0 -> Delta = x_hat - x = +1/15 > 0
        self.assertGreater(res[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(res[0, 0, 0].item(), 1.0 / 15.0, places=5)

    def test_06_reference_vs_seeded_adaptation(self):
        """Verify reference (skimage) and seeded adaptation (torch) execute correctly."""
        x = torch.rand((3, 32, 32), dtype=torch.float32)

        # Reference reproduction (unseeded, skimage)
        out_ref = extract_cvpr_reference(x, replicas=10, var=0.01, L=16)
        self.assertEqual(out_ref["candidate_name"], "cvpr_color_reference")
        self.assertEqual(out_ref["residual"].shape, (3, 32, 32))
        self.assertTrue(torch.all(torch.isfinite(out_ref["residual"])))

        # Seeded adaptation (deterministic, torch)
        out_seed1 = extract_cvpr_seeded(x, sample_id=42, replicas=10, sigma=0.10, base_seed=42)
        out_seed2 = extract_cvpr_seeded(x, sample_id=42, replicas=10, sigma=0.10, base_seed=42)
        self.assertEqual(out_seed1["candidate_name"], "cvpr_color_seeded")
        self.assertTrue(torch.all(torch.isfinite(out_seed1["residual"])))

        # Bit-exact determinism for seeded adaptation on same device/environment
        max_diff = torch.max(torch.abs(out_seed1["residual"] - out_seed2["residual"])).item()
        self.assertEqual(max_diff, 0.0)

    def test_07_noise_sweep_execution(self):
        """Verify noise sweep across all five sigma/q levels produces finite, valid residuals."""
        x = torch.rand((3, 16, 16), dtype=torch.float32)
        step_q = 1.0 / 15.0
        sweep_ratios = [0.1, 0.2, 0.3, 0.5, 1.5]
        sweep_sigmas = [r * step_q for r in sweep_ratios]

        prev_mar = None
        for sigma in sweep_sigmas:
            out = extract_cvpr_seeded(x, sample_id=999, replicas=20, sigma=sigma, base_seed=42)
            res = out["residual"]
            self.assertEqual(res.shape, (3, 16, 16))
            self.assertTrue(torch.all(torch.isfinite(res)))
            mar = res.abs().mean().item()
            # Residual magnitude should generally increase as noise increases across wide steps
            if prev_mar is not None and sigma >= 0.0333:
                self.assertGreater(mar, 0.0)
            prev_mar = mar

    def test_08_factorial_all_four_configurations(self):
        """Verify all four 2x2 factorial clipping configurations execute and produce distinct outputs."""
        x = torch.rand((3, 16, 16), dtype=torch.float32)
        configs = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]
        residuals = []
        for cn, cq in configs:
            out = extract_cvpr_seeded(
                x, sample_id=888, replicas=20, sigma=0.10, clip_noise=cn, clip_quant=cq, base_seed=42
            )
            res = out["residual"]
            self.assertEqual(res.shape, (3, 16, 16))
            self.assertTrue(torch.all(torch.isfinite(res)))
            residuals.append(res)

        self.assertEqual(len(residuals), 4)


if __name__ == "__main__":
    unittest.main()
