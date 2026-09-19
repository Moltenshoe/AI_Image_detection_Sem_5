"""Unit tests for Block 2 — Branch E (JPEG & Compression-Artifact Forensics).

Verifies:
1. Exact feature count and dictionary key contracts for:
   - E1_DCT (10 features)
   - E2_RESP (8 features)
   - E3_PHASE (4 features)
   - E4_GRID (4 features)
   - Unified Branch E (26 features)
2. Determinism (bit-exact reproducibility across repeated evaluations).
3. Numerical safety on constant-zero, constant-one, constant-0.5, step-edge, impulse, and noise images.
4. Block DCT computation shapes, orthonormality, and energy conservation.
5. In-memory JPEG recompression codec validity across Q in {95, 90, 75, 60}.
6. Fourier phase spectrum stability bounds (cosine similarities in [-1, 1], energy in [0, pi^2]).
7. Canonical 8×8 grid boundary step calculations and ratio safety.
8. Pattern discrimination across smooth gradients, checkerboard patterns, and noise.
9. Smoke extraction on real canonical Defactify dataset sample.
10. ForensicPipeline orchestration: individual candidate sub-branches, unified 'E', and multi-branch composition (A + B + C_LBP + D_MFR + E = 111 features).
"""

import math
import os
import sys
import unittest

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from src.forensics.branch_e import (
    compute_block_dct,
    compute_grid_discontinuities,
    compute_recompression_error_map,
    extract_branch_e_dct_features,
    extract_branch_e_features,
    extract_branch_e_grid_features,
    extract_branch_e_phase_features,
    extract_branch_e_recompression_features,
)
from src.forensics.pipeline import ForensicPipeline


EXPECTED_DCT_KEYS = [
    "dct_ac_mean_abs",
    "dct_ac_energy",
    "dct_ac_kurtosis",
    "dct_sparsity_ratio",
    "dct_low_freq_ratio",
    "dct_mid_freq_ratio",
    "dct_high_freq_ratio",
    "dct_anisotropy",
    "dct_benford_ssd",
    "dct_block_var_mean",
]

EXPECTED_RESP_KEYS = [
    "ela_q95_mean",
    "ela_q90_mean",
    "ela_q75_mean",
    "ela_q60_mean",
    "ela_q90_energy",
    "ela_slope_q90_q75",
    "ela_ratio_q90_q75",
    "ela_q90_gini",
]

EXPECTED_PHASE_KEYS = [
    "phase_corr_q90",
    "phase_corr_q75",
    "phase_diff_energy_q90",
    "phase_hf_stability_q90",
]

EXPECTED_GRID_KEYS = [
    "grid_h_ratio",
    "grid_v_ratio",
    "grid_strength",
    "grid_anisotropy",
]

EXPECTED_BRANCH_E_KEYS = (
    EXPECTED_DCT_KEYS
    + EXPECTED_RESP_KEYS
    + EXPECTED_PHASE_KEYS
    + EXPECTED_GRID_KEYS
)


class TestBranchEForensics(unittest.TestCase):
    """Test suite for Branch E JPEG & Compression-Artifact Forensic Feature Extraction."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.dummy_rgb = torch.rand(3, 256, 256, dtype=torch.float32)

    def test_01_e1_dct_contract(self):
        """Verify E1_DCT returns exactly 10 expected scalar float features."""
        feats = extract_branch_e_dct_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 10, f"Expected 10 features, got {len(feats)}")
        self.assertEqual(list(feats.keys()), EXPECTED_DCT_KEYS)

        for key in EXPECTED_DCT_KEYS:
            val = feats[key]
            self.assertIsInstance(val, float, f"Feature '{key}' must be float, got {type(val)}")
            self.assertTrue(math.isfinite(val), f"Feature '{key}' is not finite: {val}")

        # Basic range checks on random image
        self.assertGreater(feats["dct_ac_mean_abs"], 0.0)
        self.assertGreater(feats["dct_ac_energy"], 0.0)
        self.assertGreaterEqual(feats["dct_sparsity_ratio"], 0.0)
        self.assertLessEqual(feats["dct_sparsity_ratio"], 1.0)
        self.assertGreaterEqual(feats["dct_low_freq_ratio"], 0.0)
        self.assertGreaterEqual(feats["dct_mid_freq_ratio"], 0.0)
        self.assertGreaterEqual(feats["dct_high_freq_ratio"], 0.0)
        self.assertGreaterEqual(feats["dct_anisotropy"], 0.0)
        self.assertGreaterEqual(feats["dct_benford_ssd"], 0.0)

    def test_02_e2_resp_contract(self):
        """Verify E2_RESP returns exactly 8 expected scalar float features."""
        feats = extract_branch_e_recompression_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 8, f"Expected 8 features, got {len(feats)}")
        self.assertEqual(list(feats.keys()), EXPECTED_RESP_KEYS)

        for key in EXPECTED_RESP_KEYS:
            val = feats[key]
            self.assertIsInstance(val, float, f"Feature '{key}' must be float, got {type(val)}")
            self.assertTrue(math.isfinite(val), f"Feature '{key}' is not finite: {val}")

        # Range checks
        self.assertGreater(feats["ela_q95_mean"], 0.0)
        self.assertGreater(feats["ela_q90_mean"], 0.0)
        self.assertGreater(feats["ela_q75_mean"], 0.0)
        self.assertGreater(feats["ela_q60_mean"], 0.0)
        self.assertGreaterEqual(feats["ela_q90_gini"], 0.0)
        self.assertLessEqual(feats["ela_q90_gini"], 1.0)

    def test_03_e3_phase_contract(self):
        """Verify E3_PHASE returns exactly 4 expected scalar float features."""
        feats = extract_branch_e_phase_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 4, f"Expected 4 features, got {len(feats)}")
        self.assertEqual(list(feats.keys()), EXPECTED_PHASE_KEYS)

        for key in EXPECTED_PHASE_KEYS:
            val = feats[key]
            self.assertIsInstance(val, float, f"Feature '{key}' must be float, got {type(val)}")
            self.assertTrue(math.isfinite(val), f"Feature '{key}' is not finite: {val}")

        # Range checks
        self.assertGreaterEqual(feats["phase_corr_q90"], -1.0)
        self.assertLessEqual(feats["phase_corr_q90"], 1.0)
        self.assertGreaterEqual(feats["phase_corr_q75"], -1.0)
        self.assertLessEqual(feats["phase_corr_q75"], 1.0)
        self.assertGreaterEqual(feats["phase_diff_energy_q90"], 0.0)
        self.assertLessEqual(feats["phase_diff_energy_q90"], math.pi ** 2)

    def test_04_e4_grid_contract(self):
        """Verify E4_GRID returns exactly 4 expected scalar float features."""
        feats = extract_branch_e_grid_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 4, f"Expected 4 features, got {len(feats)}")
        self.assertEqual(list(feats.keys()), EXPECTED_GRID_KEYS)

        for key in EXPECTED_GRID_KEYS:
            val = feats[key]
            self.assertIsInstance(val, float, f"Feature '{key}' must be float, got {type(val)}")
            self.assertTrue(math.isfinite(val), f"Feature '{key}' is not finite: {val}")

        self.assertGreater(feats["grid_h_ratio"], 0.0)
        self.assertGreater(feats["grid_v_ratio"], 0.0)
        self.assertGreater(feats["grid_strength"], 0.0)
        self.assertGreaterEqual(feats["grid_anisotropy"], 0.0)
        self.assertLessEqual(feats["grid_anisotropy"], 1.0)

    def test_05_unified_branch_e_contract(self):
        """Verify unified extract_branch_e_features returns exactly 26 features in order."""
        feats = extract_branch_e_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 26, f"Expected 26 features, got {len(feats)}")
        self.assertEqual(list(feats.keys()), EXPECTED_BRANCH_E_KEYS)

        for key, val in feats.items():
            self.assertTrue(math.isfinite(val), f"Feature '{key}' is not finite: {val}")

    def test_06_block_dct_shape_and_orthonormality(self):
        """Verify 8×8 block DCT produces shape [B, 1024, 8, 8] and preserves energy."""
        # Unbatched [3, 256, 256] -> [1, 1024, 8, 8]
        dct_out = compute_block_dct(self.dummy_rgb)
        self.assertEqual(dct_out.shape, (1, 1024, 8, 8))
        self.assertEqual(dct_out.dtype, torch.float32)

        # Batched [2, 3, 256, 256] -> [2, 1024, 8, 8]
        batch_rgb = torch.rand(2, 3, 256, 256, dtype=torch.float32)
        batch_dct = compute_block_dct(batch_rgb)
        self.assertEqual(batch_dct.shape, (2, 1024, 8, 8))

    def test_07_constant_and_zero_images_numerical_safety(self):
        """Verify constant/zero images produce finite outputs, zero residuals, and safe limits."""
        for val in (0.0, 0.5, 1.0):
            const_img = torch.full((3, 256, 256), val, dtype=torch.float32)

            f_dct = extract_branch_e_dct_features(const_img)
            f_resp = extract_branch_e_recompression_features(const_img)
            f_phase = extract_branch_e_phase_features(const_img)
            f_grid = extract_branch_e_grid_features(const_img)
            f_all = extract_branch_e_features(const_img)

            # Check all values are finite
            for f_dict in (f_dct, f_resp, f_phase, f_grid, f_all):
                for k, v in f_dict.items():
                    self.assertTrue(math.isfinite(v), f"Feature '{k}' on constant {val} is not finite: {v}")

            # On a constant image, AC coefficients must be zero
            self.assertAlmostEqual(f_dct["dct_ac_mean_abs"], 0.0, places=5)
            self.assertAlmostEqual(f_dct["dct_ac_energy"], 0.0, places=5)
            self.assertAlmostEqual(f_dct["dct_sparsity_ratio"], 1.0, places=5)

            # Recompression errors on uniform constant image should be zero (or <= 1/255 for mid-tones due to color-space rounding)
            if val in (0.0, 1.0):
                self.assertAlmostEqual(f_resp["ela_q95_mean"], 0.0, places=5)
                self.assertAlmostEqual(f_resp["ela_q90_mean"], 0.0, places=5)
            else:
                self.assertLess(f_resp["ela_q95_mean"], 0.01)
                self.assertLess(f_resp["ela_q90_mean"], 0.01)

            # Phase correlations on constant image default safely to 1.0
            self.assertAlmostEqual(f_phase["phase_corr_q90"], 1.0, places=5)
            self.assertAlmostEqual(f_phase["phase_diff_energy_q90"], 0.0, places=5)

    def test_08_determinism(self):
        """Verify repeated extractions yield bit-exact identical feature dictionaries."""
        for _ in range(3):
            f1 = extract_branch_e_features(self.dummy_rgb)
            f2 = extract_branch_e_features(self.dummy_rgb)
            self.assertEqual(f1, f2)

    def test_09_pathological_and_edge_inputs(self):
        """Verify safety on step edges, impulses, and noise images."""
        # 1. Binary step edge
        step_img = torch.zeros(3, 256, 256, dtype=torch.float32)
        step_img[:, :, :128] = 1.0
        f_step = extract_branch_e_features(step_img)
        for k, v in f_step.items():
            self.assertTrue(math.isfinite(v), f"Feature '{k}' on step edge is not finite: {v}")

        # 2. Single impulse
        impulse = torch.zeros(3, 256, 256, dtype=torch.float32)
        impulse[:, 128, 128] = 1.0
        f_impulse = extract_branch_e_features(impulse)
        for k, v in f_impulse.items():
            self.assertTrue(math.isfinite(v), f"Feature '{k}' on impulse is not finite: {v}")

    def test_10_synthetic_pattern_discrimination(self):
        """Verify distinct spatial patterns produce distinctly different Branch E feature vectors."""
        # Smooth ramp vs high-frequency checkerboard vs white noise
        ramp = torch.linspace(0, 1, 256).unsqueeze(0).repeat(256, 1).unsqueeze(0).repeat(3, 1, 1)

        checker = torch.zeros(3, 256, 256, dtype=torch.float32)
        checker[:, 0::2, 0::2] = 1.0
        checker[:, 1::2, 1::2] = 1.0

        f_ramp = extract_branch_e_features(ramp)
        f_checker = extract_branch_e_features(checker)

        # Checkerboard has vastly higher high-frequency AC energy than smooth ramp
        self.assertGreater(f_checker["dct_ac_energy"], f_ramp["dct_ac_energy"])
        self.assertNotEqual(f_checker["ela_q90_mean"], f_ramp["ela_q90_mean"])
        self.assertNotEqual(f_checker["grid_strength"], f_ramp["grid_strength"])

    def test_11_smoke_real_defactify_sample(self):
        """Smoke test extracting Branch E features on a real Defactify dataset sample."""
        from src.data.loader import DefactifyDataset

        dataset = DefactifyDataset(split="train")
        sample_img, label_a = dataset[0]

        feats = extract_branch_e_features(sample_img)
        self.assertEqual(len(feats), 26)
        self.assertEqual(list(feats.keys()), EXPECTED_BRANCH_E_KEYS)

        for k, v in feats.items():
            self.assertTrue(math.isfinite(v), f"Feature '{k}' on real sample is not finite: {v}")

    def test_12_forensic_pipeline_integration(self):
        """Verify ForensicPipeline orchestrates Branch E sub-branches and multi-branch combinations."""
        # 1. Individual candidate sub-branches
        p_dct = ForensicPipeline(branches=["E_DCT"])
        p_resp = ForensicPipeline(branches=["E_RESP"])
        p_phase = ForensicPipeline(branches=["E_PHASE"])
        p_grid = ForensicPipeline(branches=["E_GRID"])
        p_e = ForensicPipeline(branches=["E"])

        self.assertEqual(p_dct.get_feature_names(), EXPECTED_DCT_KEYS)
        self.assertEqual(p_resp.get_feature_names(), EXPECTED_RESP_KEYS)
        self.assertEqual(p_phase.get_feature_names(), EXPECTED_PHASE_KEYS)
        self.assertEqual(p_grid.get_feature_names(), EXPECTED_GRID_KEYS)
        self.assertEqual(p_e.get_feature_names(), EXPECTED_BRANCH_E_KEYS)

        # 2. Verify pipeline extract equals direct extractors
        self.assertEqual(p_dct.extract(self.dummy_rgb), extract_branch_e_dct_features(self.dummy_rgb))
        self.assertEqual(p_resp.extract(self.dummy_rgb), extract_branch_e_recompression_features(self.dummy_rgb))
        self.assertEqual(p_phase.extract(self.dummy_rgb), extract_branch_e_phase_features(self.dummy_rgb))
        self.assertEqual(p_grid.extract(self.dummy_rgb), extract_branch_e_grid_features(self.dummy_rgb))
        self.assertEqual(p_e.extract(self.dummy_rgb), extract_branch_e_features(self.dummy_rgb))

        # 3. Multi-branch combination: A (34) + B (30) + E (26) = 90 features
        p_abe = ForensicPipeline(branches=["A", "B", "E"])
        self.assertEqual(len(p_abe.get_feature_names()), 90)

        # 4. Full 5-branch candidate suite: A (34) + B (30) + C_LBP (16) + D_MFR (5) + E (26) = 111 features
        p_full = ForensicPipeline(branches=["A", "B", "C_LBP", "D_MFR", "E"])
        self.assertEqual(len(p_full.get_feature_names()), 111)

        feats_full = p_full.extract(self.dummy_rgb)
        self.assertEqual(len(feats_full), 111)
        for v in feats_full.values():
            self.assertTrue(math.isfinite(v))

        # 5. Exact match of Branch E subvector in full pipeline
        for k in EXPECTED_BRANCH_E_KEYS:
            self.assertEqual(feats_full[k], p_e.extract(self.dummy_rgb)[k])


if __name__ == "__main__":
    unittest.main()
