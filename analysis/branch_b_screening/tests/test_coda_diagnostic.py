"""Unit tests for Candidate D: CoDA 2026 Analytical Diagnostic Model."""

from __future__ import annotations

import unittest
import torch

from analysis.branch_b_screening.candidates.coda_diagnostic import (
    coda_algorithm_1,
    coda_equation_12,
    compare_coda_formulations,
)


class TestCoDADiagnostic(unittest.TestCase):
    """Test suite for CoDA analytical diagnostic formulation."""

    def test_01_algorithm_1_integer_symmetry(self):
        """Verify that Algorithm 1 expected residual is zero on interior integers."""
        # Single interior integer pixel
        img = torch.tensor([128.0], dtype=torch.float32)
        # Large R to verify convergence to E[Delta] = 0
        res = coda_algorithm_1(img, sigma=0.10, replicas=2000, seed=42)
        # Expected residual on integer with sigma=0.10 is zero
        self.assertAlmostEqual(res.item(), 0.0, places=2)

    def test_02_algorithm_1_fractional_activation(self):
        """Verify that Algorithm 1 produces active residuals on fractional inputs."""
        # Midpoint fractional pixel: 128.5 + N ~ N(128.5, 0.01)
        # round(128.5 + N) flips between 128 and 129
        img = torch.tensor([128.5], dtype=torch.float32)
        res = coda_algorithm_1(img, sigma=0.10, replicas=50, seed=42)
        # At 128.5, Y is ~128 or ~129, mean ~ 128.5 -> Delta = 128.5 - 128.5 ~ 0
        # But for 128.2: round(128.2 + N) mostly rounds to 128 -> mean ~ 128.0 -> Delta = 128.2 - 128.0 = +0.2
        img_asym = torch.tensor([128.2], dtype=torch.float32)
        res_asym = coda_algorithm_1(img_asym, sigma=0.10, replicas=50, seed=42)
        self.assertAlmostEqual(res_asym.item(), 0.2, places=1)

    def test_03_equation_12_periodic_model(self):
        """Verify Equation 12 Fourier periodic bias formula."""
        # At x=0.0: sin(0) = 0 -> bias = 0
        x_zero = torch.tensor([0.0], dtype=torch.float32)
        b_zero = coda_equation_12(x_zero, sigma=0.10)
        self.assertAlmostEqual(b_zero.item(), 0.0, places=6)

        # At x=0.25: sin(pi/2) = 1.0 -> maximum positive bias
        x_quarter = torch.tensor([0.25], dtype=torch.float32)
        b_quarter = coda_equation_12(x_quarter, sigma=0.10)
        self.assertGreater(b_quarter.item(), 0.0)

    def test_04_comparative_diagnostic_output(self):
        """Verify compare_coda_formulations returns diagnostic report with expected flags."""
        diag = compare_coda_formulations()
        self.assertFalse(diag["is_materialization_candidate"])
        self.assertIn("algorithm_1_integer_mean_abs", diag)
        self.assertIn("equation_12_fractional_max_abs", diag)
        self.assertEqual(diag["diagnostic_name"], "coda_formulation_comparison")


if __name__ == "__main__":
    unittest.main()
