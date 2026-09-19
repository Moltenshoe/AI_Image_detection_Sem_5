"""Unit test suite for Block 2 — Branch A (Frequency / Periodicity) & Forensic Pipeline.

Verifies:
1. Standard FFT feature contract (4 scalar features, naming, types).
2. Synthbuster-inspired periodicity contract (30 scalar features, naming, types).
3. Unified Branch A contract (exactly 34 scalar features).
4. Cross-difference spatial size and arithmetic (256x256 -> 255x255).
5. Determinism (byte/float identical across repeated evaluations).
6. RGB channel-independence for A2 Synthbuster features.
7. Numerical safety on pathological inputs (zeros, constant white, constant gray).
8. Distinct images produce distinct forensic signatures.
9. Diagnostic map output separation (maps do NOT enter scalar features).
10. Smoke extraction on real canonical Defactify image tensor.
11. ForensicPipeline orchestrator contracts, branch validation, and equivalence.
"""

import math
import os
import sys
import unittest

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from src.forensics.branch_a_frequency.fft import (
    compute_fft_diagnostics,
    frequency_radius,
    rgb_to_gray,
    standard_fft_features,
)
from src.forensics.branch_a_frequency.synthbuster import (
    cross_difference,
    synthbuster_periodicity_features,
)
from src.forensics.branch_a_frequency.features import extract_branch_a_features
from src.forensics.pipeline import ForensicPipeline


EXPECTED_A1_KEYS = [
    "fft_low_freq_ratio",
    "fft_mid_freq_ratio",
    "fft_high_freq_ratio",
    "fft_spectral_centroid",
]

EXPECTED_A2_KEYS = []
for _c in ("r", "g", "b"):
    for _p in (2, 4, 8):
        for _d in ("x", "y", "d"):
            EXPECTED_A2_KEYS.append(f"synth_{_c}_p{_p}_{_d}_mean")
    EXPECTED_A2_KEYS.append(f"synth_{_c}_fft_highfreq_ratio")

EXPECTED_BRANCH_A_KEYS = EXPECTED_A1_KEYS + EXPECTED_A2_KEYS


class TestBranchAFrequency(unittest.TestCase):
    """Test suite for Branch A frequency and periodicity features."""

    def setUp(self):
        torch.manual_seed(42)
        # Canonical test image: [3, 256, 256], float32 in [0, 1]
        self.canonical_img = torch.rand(3, 256, 256, dtype=torch.float32)

    def test_01_a1_standard_fft_contract(self):
        """Verify A1 returns exactly 4 expected scalar float features."""
        feats = standard_fft_features(self.canonical_img)
        self.assertEqual(len(feats), 4)
        self.assertEqual(list(feats.keys()), EXPECTED_A1_KEYS)
        for k, v in feats.items():
            self.assertIsInstance(v, float, f"{k} is not float")
            self.assertTrue(math.isfinite(v), f"{k} is not finite: {v}")
            self.assertGreaterEqual(v, 0.0, f"{k} is negative: {v}")

        # Energy ratios should sum close to 1.0
        ratio_sum = (
            feats["fft_low_freq_ratio"]
            + feats["fft_mid_freq_ratio"]
            + feats["fft_high_freq_ratio"]
        )
        self.assertAlmostEqual(ratio_sum, 1.0, places=4)

    def test_02_a2_synthbuster_contract(self):
        """Verify A2 returns exactly 30 expected scalar float features."""
        feats = synthbuster_periodicity_features(self.canonical_img)
        self.assertEqual(len(feats), 30)
        self.assertEqual(list(feats.keys()), EXPECTED_A2_KEYS)
        for k, v in feats.items():
            self.assertIsInstance(v, float, f"{k} is not float")
            self.assertTrue(math.isfinite(v), f"{k} is not finite: {v}")
            self.assertGreaterEqual(v, 0.0, f"{k} is negative: {v}")

    def test_03_unified_branch_a_contract(self):
        """Verify extract_branch_a_features returns exactly 34 scalar features."""
        feats = extract_branch_a_features(self.canonical_img)
        self.assertEqual(len(feats), 34)
        self.assertEqual(list(feats.keys()), EXPECTED_BRANCH_A_KEYS)
        for k, v in feats.items():
            self.assertIsInstance(v, float, f"{k} is not float")
            self.assertTrue(math.isfinite(v), f"{k} is not finite: {v}")

    def test_04_cross_difference_geometry_and_arithmetic(self):
        """Verify cross-difference transforms [..., 256, 256] -> [..., 255, 255] and computes exact arithmetic."""
        # Shape check
        cd = cross_difference(self.canonical_img)
        self.assertEqual(cd.shape, (3, 255, 255))

        # Batched shape check
        batched = self.canonical_img.unsqueeze(0)
        cd_b = cross_difference(batched)
        self.assertEqual(cd_b.shape, (1, 3, 255, 255))

        # Arithmetic verification on known 2x2 patch
        patch1 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.assertEqual(float(cross_difference(patch1)[0, 0, 0, 0]), 0.0)

        patch2 = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        self.assertEqual(float(cross_difference(patch2)[0, 0, 0, 0]), 2.0)

    def test_05_determinism(self):
        """Verify repeated feature extraction on same tensor produces bit-exact identical values."""
        feats1 = extract_branch_a_features(self.canonical_img)
        feats2 = extract_branch_a_features(self.canonical_img)
        self.assertEqual(feats1, feats2)

    def test_06_rgb_channel_independence_in_a2(self):
        """Verify modifying one channel in A2 only affects that channel's Synthbuster features."""
        base_img = torch.rand(3, 256, 256, dtype=torch.float32)
        base_a2 = synthbuster_periodicity_features(base_img)

        # Perturb Red channel only
        img_r = base_img.clone()
        img_r[0] = torch.rand(256, 256, dtype=torch.float32)
        a2_r = synthbuster_periodicity_features(img_r)

        r_changed = any(
            a2_r[f"synth_r_{suffix}"] != base_a2[f"synth_r_{suffix}"]
            for suffix in ["p2_x_mean", "p4_x_mean", "p8_x_mean", "fft_highfreq_ratio"]
        )
        self.assertTrue(r_changed, "R channel perturbation did not change R features")

        for k in base_a2:
            if k.startswith("synth_g_") or k.startswith("synth_b_"):
                self.assertEqual(
                    a2_r[k],
                    base_a2[k],
                    f"Perturbing R affected unrelated feature {k}",
                )

        # Perturb Green channel only
        img_g = base_img.clone()
        img_g[1] = torch.rand(256, 256, dtype=torch.float32)
        a2_g = synthbuster_periodicity_features(img_g)

        for k in base_a2:
            if k.startswith("synth_r_") or k.startswith("synth_b_"):
                self.assertEqual(
                    a2_g[k],
                    base_a2[k],
                    f"Perturbing G affected unrelated feature {k}",
                )

        # Perturb Blue channel only
        img_b = base_img.clone()
        img_b[2] = torch.rand(256, 256, dtype=torch.float32)
        a2_b = synthbuster_periodicity_features(img_b)

        for k in base_a2:
            if k.startswith("synth_r_") or k.startswith("synth_g_"):
                self.assertEqual(
                    a2_b[k],
                    base_a2[k],
                    f"Perturbing B affected unrelated feature {k}",
                )

    def test_07_numerical_safety_zero_and_constant_inputs(self):
        """Verify pathological inputs (all zeros, all ones, constant gray) produce finite outputs with zero NaNs."""
        pathological_cases = [
            ("all_zeros", torch.zeros(3, 256, 256, dtype=torch.float32)),
            ("all_ones", torch.ones(3, 256, 256, dtype=torch.float32)),
            ("constant_gray", torch.full((3, 256, 256), 0.5, dtype=torch.float32)),
        ]

        for name, tensor in pathological_cases:
            feats = extract_branch_a_features(tensor)
            self.assertEqual(
                len(feats),
                34,
                f"Feature count mismatch on {name}",
            )
            for k, v in feats.items():
                self.assertTrue(
                    math.isfinite(v),
                    f"Non-finite value {v} for feature {k} on {name}",
                )
                self.assertFalse(
                    math.isnan(v),
                    f"NaN value for feature {k} on {name}",
                )

    def test_08_distinct_images_produce_distinct_features(self):
        """Verify that distinct images produce different feature vectors."""
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, 256), torch.linspace(0, 1, 256), indexing="ij"
        )
        smooth_img = (xx + yy).unsqueeze(0).repeat(3, 1, 1) / 2.0

        cb = ((torch.arange(256).view(-1, 1) + torch.arange(256).view(1, -1)) % 2).float()
        checker_img = cb.unsqueeze(0).repeat(3, 1, 1)

        feats_smooth = extract_branch_a_features(smooth_img)
        feats_checker = extract_branch_a_features(checker_img)

        self.assertGreater(
            feats_checker["fft_high_freq_ratio"],
            feats_smooth["fft_high_freq_ratio"],
        )
        self.assertGreater(
            feats_smooth["fft_low_freq_ratio"],
            feats_checker["fft_low_freq_ratio"],
        )

    def test_09_diagnostics_isolation(self):
        """Verify diagnostic maps are returned separately and do not pollute detector features."""
        diag = compute_fft_diagnostics(self.canonical_img, radial_bins=64)
        self.assertIn("fft_logmag_map", diag)
        self.assertIn("fft_phase_map", diag)
        self.assertIn("fft_radial_power", diag)
        self.assertIn("fft_radial_r", diag)

        self.assertEqual(diag["fft_logmag_map"].shape, (1, 1, 256, 256))
        self.assertEqual(diag["fft_phase_map"].shape, (1, 1, 256, 256))
        self.assertEqual(diag["fft_radial_power"].shape, (64,))
        self.assertEqual(diag["fft_radial_r"].shape, (64,))

        det_feats = extract_branch_a_features(self.canonical_img)
        for k, v in det_feats.items():
            self.assertIsInstance(v, float)
            self.assertNotIsInstance(v, torch.Tensor)

    def test_10_smoke_real_canonical_tensor(self):
        """Smoke test extracting Branch A features on a real sample from the raw Defactify loader."""
        from src.data.loader import DefactifyDataset

        dataset = DefactifyDataset(split="train")
        sample_img, label = dataset[0]

        feats = extract_branch_a_features(sample_img)
        self.assertEqual(len(feats), 34)
        for k, v in feats.items():
            self.assertTrue(math.isfinite(v), f"Real image feature {k} is not finite: {v}")
            self.assertIsInstance(v, float)

    def test_11_forensic_pipeline_orchestration(self):
        """Verify ForensicPipeline orchestrates Branch A with exact equivalence."""
        pipeline = ForensicPipeline(branches=["A"])
        self.assertEqual(pipeline.branches, ("A",))
        self.assertEqual(pipeline.get_feature_names(), EXPECTED_BRANCH_A_KEYS)

        pipeline_feats = pipeline.extract(self.canonical_img)
        direct_feats = extract_branch_a_features(self.canonical_img)
        self.assertEqual(pipeline_feats, direct_feats)

        # Test invalid branch errors
        with self.assertRaises(ValueError):
            ForensicPipeline(branches=["INVALID_BRANCH"])

        with self.assertRaises(ValueError):
            ForensicPipeline(branches=["C"])


if __name__ == "__main__":
    unittest.main()
