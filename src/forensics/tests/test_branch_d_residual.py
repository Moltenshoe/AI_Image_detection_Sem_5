"""Unit tests for Block 2 — Branch D (Residual / Error / Noise Forensic Feature Extraction).

Verifies:
1. Exact feature count and dictionary key contracts for D_HIGHPASS (5), D_LAPLACIAN (5), D_MFR (5).
2. Residual map tensor dimensions and dtypes for unbatched and batched inputs.
3. Constant-image and all-zero image response (exact zero residuals, safe zero statistics, zero NaNs/Infs).
4. Exact filter / kernel behavior and replicate padding boundary handling.
5. Median filter residual reference equivalence against scipy.ndimage.median_filter(mode='nearest').
6. Determinism (bit-exact reproducibility across repeated evaluations).
7. Numerical safety on pathological and near-zero variance inputs.
8. Synthetic pattern discrimination (smooth gradient vs checkerboard vs white noise).
9. Smoke extraction on real canonical Defactify dataset sample.
10. ForensicPipeline orchestration: independent candidate selection, alias 'D' expansion, and multi-branch composition.
"""

import math
import os
import sys
import unittest

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.ndimage import median_filter as scipy_median_filter
import torch

from src.forensics.branch_d_residual import (
    compute_highpass_residual,
    compute_laplacian_residual,
    compute_median_filter_residual,
    extract_branch_d_highpass_features,
    extract_branch_d_laplacian_features,
    extract_branch_d_mfr_features,
    extract_highpass_features,
    extract_laplacian_features,
    extract_mfr_features,
)
from src.forensics.pipeline import ForensicPipeline


EXPECTED_HIGHPASS_KEYS = [
    "hp_absmean",
    "hp_std",
    "hp_energy",
    "hp_kurtosis",
    "hp_entropy",
]

EXPECTED_LAPLACIAN_KEYS = [
    "lap_absmean",
    "lap_std",
    "lap_energy",
    "lap_kurtosis",
    "lap_entropy",
]

EXPECTED_MFR_KEYS = [
    "mfr_absmean",
    "mfr_std",
    "mfr_energy",
    "mfr_kurtosis",
    "mfr_entropy",
]


class TestBranchDResidual(unittest.TestCase):
    """Test suite for Branch D Residual / Error / Noise forensic feature extraction."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.dummy_rgb = torch.rand(3, 256, 256, dtype=torch.float32)

    def test_01_d_highpass_contract(self):
        """Verify D_HIGHPASS returns exactly 5 expected scalar float features."""
        feats = extract_branch_d_highpass_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 5, f"Expected 5 features, got {len(feats)}")
        self.assertEqual(list(feats.keys()), EXPECTED_HIGHPASS_KEYS)

        for key in EXPECTED_HIGHPASS_KEYS:
            val = feats[key]
            self.assertIsInstance(val, float, f"Feature '{key}' must be float, got {type(val)}")
            self.assertTrue(math.isfinite(val), f"Feature '{key}' is not finite: {val}")

        # Basic range checks on random image
        self.assertGreater(feats["hp_absmean"], 0.0)
        self.assertGreater(feats["hp_std"], 0.0)
        self.assertGreater(feats["hp_energy"], 0.0)
        self.assertGreater(feats["hp_entropy"], 0.0)

    def test_02_d_laplacian_contract(self):
        """Verify D_LAPLACIAN returns exactly 5 expected scalar float features."""
        feats = extract_branch_d_laplacian_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 5, f"Expected 5 features, got {len(feats)}")
        self.assertEqual(list(feats.keys()), EXPECTED_LAPLACIAN_KEYS)

        for key in EXPECTED_LAPLACIAN_KEYS:
            val = feats[key]
            self.assertIsInstance(val, float, f"Feature '{key}' must be float, got {type(val)}")
            self.assertTrue(math.isfinite(val), f"Feature '{key}' is not finite: {val}")

        self.assertGreater(feats["lap_absmean"], 0.0)
        self.assertGreater(feats["lap_std"], 0.0)
        self.assertGreater(feats["lap_energy"], 0.0)
        self.assertGreater(feats["lap_entropy"], 0.0)

    def test_03_d_mfr_contract(self):
        """Verify D_MFR returns exactly 5 expected scalar float features."""
        feats = extract_branch_d_mfr_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 5, f"Expected 5 features, got {len(feats)}")
        self.assertEqual(list(feats.keys()), EXPECTED_MFR_KEYS)

        for key in EXPECTED_MFR_KEYS:
            val = feats[key]
            self.assertIsInstance(val, float, f"Feature '{key}' must be float, got {type(val)}")
            self.assertTrue(math.isfinite(val), f"Feature '{key}' is not finite: {val}")

        self.assertGreater(feats["mfr_absmean"], 0.0)
        self.assertGreater(feats["mfr_std"], 0.0)
        self.assertGreater(feats["mfr_energy"], 0.0)
        self.assertGreater(feats["mfr_entropy"], 0.0)

    def test_04_residual_maps_shapes_and_types(self):
        """Verify residual map functions produce correct tensor shapes and float32 dtype."""
        # Unbatched [3, 256, 256] -> [1, 256, 256]
        r_hp = compute_highpass_residual(self.dummy_rgb)
        r_lap = compute_laplacian_residual(self.dummy_rgb)
        r_mfr = compute_median_filter_residual(self.dummy_rgb)

        self.assertEqual(r_hp.shape, (1, 256, 256))
        self.assertEqual(r_lap.shape, (1, 256, 256))
        self.assertEqual(r_mfr.shape, (1, 256, 256))
        self.assertEqual(r_hp.dtype, torch.float32)
        self.assertEqual(r_lap.dtype, torch.float32)
        self.assertEqual(r_mfr.dtype, torch.float32)

        # Batched [2, 3, 256, 256] -> [2, 1, 256, 256]
        batch_rgb = torch.rand(2, 3, 256, 256, dtype=torch.float32)
        b_hp = compute_highpass_residual(batch_rgb)
        b_lap = compute_laplacian_residual(batch_rgb)
        b_mfr = compute_median_filter_residual(batch_rgb)

        self.assertEqual(b_hp.shape, (2, 1, 256, 256))
        self.assertEqual(b_lap.shape, (2, 1, 256, 256))
        self.assertEqual(b_mfr.shape, (2, 1, 256, 256))

    def test_05_constant_and_zero_image_behavior(self):
        """Verify constant/zero images produce zero residuals and safe zero statistics."""
        for val in (0.0, 0.5, 1.0):
            const_img = torch.full((3, 256, 256), val, dtype=torch.float32)

            r_hp = compute_highpass_residual(const_img)
            r_lap = compute_laplacian_residual(const_img)
            r_mfr = compute_median_filter_residual(const_img)

            # Max residual delta
            self.assertLess(float(r_hp.abs().max()), 1e-6)
            self.assertEqual(float(r_lap.abs().max()), 0.0)
            self.assertEqual(float(r_mfr.abs().max()), 0.0)

            # Features
            f_hp = extract_branch_d_highpass_features(const_img)
            f_lap = extract_branch_d_laplacian_features(const_img)
            f_mfr = extract_branch_d_mfr_features(const_img)

            for f_dict in (f_hp, f_lap, f_mfr):
                for k, v in f_dict.items():
                    self.assertTrue(math.isfinite(v), f"Feature {k} on constant {val} is not finite: {v}")
                    if "kurtosis" in k or "entropy" in k or "absmean" in k or "energy" in k:
                        self.assertAlmostEqual(v, 0.0, places=5)

    def test_06_exact_kernel_behavior(self):
        """Verify exact filter kernel coefficients and boundary behavior."""
        # Single impulse at center (128, 128)
        impulse = torch.zeros(3, 256, 256, dtype=torch.float32)
        impulse[:, 128, 128] = 1.0

        r_lap = compute_laplacian_residual(impulse).squeeze()
        # Grayscale weight for (1, 1, 1) = 0.299 + 0.587 + 0.114 = 1.0
        # Laplacian center should be -8.0, 8 neighbors should be +1.0
        self.assertAlmostEqual(float(r_lap[128, 128]), -8.0, places=5)
        self.assertAlmostEqual(float(r_lap[127, 128]), 1.0, places=5)
        self.assertAlmostEqual(float(r_lap[129, 128]), 1.0, places=5)
        self.assertAlmostEqual(float(r_lap[128, 127]), 1.0, places=5)
        self.assertAlmostEqual(float(r_lap[128, 129]), 1.0, places=5)
        self.assertAlmostEqual(float(r_lap[127, 127]), 1.0, places=5)
        self.assertAlmostEqual(float(r_lap[129, 129]), 1.0, places=5)
        self.assertAlmostEqual(float(r_lap[127, 129]), 1.0, places=5)
        self.assertAlmostEqual(float(r_lap[129, 127]), 1.0, places=5)

        # Highpass Gaussian center weight: 1.0 - (1D_center * 1D_center)
        # 1D kernel sigma=1.0 size=5: coords = [-2, -1, 0, 1, 2]
        # weights ~ exp(-0.5*c^2), normalized
        c1d = np.exp(-0.5 * (np.arange(5) - 2) ** 2)
        c1d = c1d / c1d.sum()
        center_blur = float(c1d[2] * c1d[2])
        r_hp = compute_highpass_residual(impulse).squeeze()
        self.assertAlmostEqual(float(r_hp[128, 128]), 1.0 - center_blur, places=5)

    def test_07_mfr_scipy_reference_equivalence(self):
        """Verify pure PyTorch median filter residual is numerically identical to scipy reference."""
        torch.manual_seed(99)
        test_img = torch.rand(3, 256, 256, dtype=torch.float32)

        # PyTorch MFR
        res_pytorch = compute_median_filter_residual(test_img).squeeze().detach().cpu().numpy()

        # Scipy reference on exact same BT.601 grayscale
        gray_np = (0.299 * test_img[0] + 0.587 * test_img[1] + 0.114 * test_img[2]).numpy()
        med_scipy = scipy_median_filter(gray_np, size=3, mode="nearest")
        res_scipy = gray_np - med_scipy

        max_delta = np.abs(res_pytorch - res_scipy).max()
        self.assertAlmostEqual(float(max_delta), 0.0, places=6,
                               msg=f"PyTorch MFR diverged from scipy reference: max delta = {max_delta}")

    def test_08_determinism(self):
        """Verify repeated feature extractions yield bit-exact identical values."""
        for _ in range(3):
            f_hp_1 = extract_branch_d_highpass_features(self.dummy_rgb)
            f_hp_2 = extract_branch_d_highpass_features(self.dummy_rgb)
            self.assertEqual(f_hp_1, f_hp_2)

            f_lap_1 = extract_branch_d_laplacian_features(self.dummy_rgb)
            f_lap_2 = extract_branch_d_laplacian_features(self.dummy_rgb)
            self.assertEqual(f_lap_1, f_lap_2)

            f_mfr_1 = extract_branch_d_mfr_features(self.dummy_rgb)
            f_mfr_2 = extract_branch_d_mfr_features(self.dummy_rgb)
            self.assertEqual(f_mfr_1, f_mfr_2)

    def test_09_numerical_safety_pathological_inputs(self):
        """Verify safety on near-zero variance inputs, step transitions, and extreme noise."""
        # 1. Near-zero variance (1e-14 perturbations)
        tiny_noise = torch.full((3, 256, 256), 0.5, dtype=torch.float32) + 1e-14 * torch.randn(3, 256, 256)
        f_hp = extract_branch_d_highpass_features(tiny_noise)
        f_lap = extract_branch_d_laplacian_features(tiny_noise)
        f_mfr = extract_branch_d_mfr_features(tiny_noise)

        for f_dict in (f_hp, f_lap, f_mfr):
            for k, v in f_dict.items():
                self.assertTrue(math.isfinite(v), f"Feature {k} on near-zero variance is not finite: {v}")

        # 2. Binary step edge
        step_img = torch.zeros(3, 256, 256, dtype=torch.float32)
        step_img[:, :, :128] = 1.0
        f_hp_step = extract_branch_d_highpass_features(step_img)
        f_lap_step = extract_branch_d_laplacian_features(step_img)
        f_mfr_step = extract_branch_d_mfr_features(step_img)

        for f_dict in (f_hp_step, f_lap_step, f_mfr_step):
            for k, v in f_dict.items():
                self.assertTrue(math.isfinite(v), f"Feature {k} on step edge is not finite: {v}")

    def test_10_pattern_discrimination(self):
        """Verify distinct spatial patterns produce distinctly different residual feature vectors."""
        # Smooth ramp vs high-frequency checkerboard vs white noise
        ramp = torch.linspace(0, 1, 256).unsqueeze(0).repeat(256, 1).unsqueeze(0).repeat(3, 1, 1)

        checker = torch.zeros(3, 256, 256, dtype=torch.float32)
        checker[:, 0::2, 0::2] = 1.0
        checker[:, 1::2, 1::2] = 1.0

        noise = torch.rand(3, 256, 256, dtype=torch.float32)

        # High-pass comparison
        hp_ramp = extract_branch_d_highpass_features(ramp)
        hp_checker = extract_branch_d_highpass_features(checker)
        hp_noise = extract_branch_d_highpass_features(noise)

        self.assertNotEqual(hp_ramp["hp_std"], hp_checker["hp_std"])
        self.assertNotEqual(hp_checker["hp_std"], hp_noise["hp_std"])

        # Laplacian comparison
        lap_ramp = extract_branch_d_laplacian_features(ramp)
        lap_checker = extract_branch_d_laplacian_features(checker)
        self.assertGreater(lap_checker["lap_energy"], lap_ramp["lap_energy"])

        # MFR comparison
        mfr_ramp = extract_branch_d_mfr_features(ramp)
        mfr_checker = extract_branch_d_mfr_features(checker)
        self.assertGreater(mfr_checker["mfr_energy"], mfr_ramp["mfr_energy"])

    def test_11_smoke_real_defactify_sample(self):
        """Smoke test extracting Branch D features on a real Defactify dataset sample."""
        from src.data.loader import DefactifyDataset

        dataset = DefactifyDataset(split="train")
        sample_img, label_a = dataset[0]

        f_hp = extract_branch_d_highpass_features(sample_img)
        f_lap = extract_branch_d_laplacian_features(sample_img)
        f_mfr = extract_branch_d_mfr_features(sample_img)

        self.assertEqual(len(f_hp), 5)
        self.assertEqual(len(f_lap), 5)
        self.assertEqual(len(f_mfr), 5)

        for f_dict in (f_hp, f_lap, f_mfr):
            for k, v in f_dict.items():
                self.assertTrue(math.isfinite(v), f"Feature {k} on real image is not finite: {v}")

    def test_12_forensic_pipeline_integration_and_isolation(self):
        """Verify ForensicPipeline orchestrates individual Branch D candidates and multi-branch mixes."""
        # 1. Individual candidate selections
        p_hp = ForensicPipeline(branches=["D_HIGHPASS"])
        p_lap = ForensicPipeline(branches=["D_LAPLACIAN"])
        p_mfr = ForensicPipeline(branches=["D_MFR"])

        self.assertEqual(p_hp.get_feature_names(), EXPECTED_HIGHPASS_KEYS)
        self.assertEqual(p_lap.get_feature_names(), EXPECTED_LAPLACIAN_KEYS)
        self.assertEqual(p_mfr.get_feature_names(), EXPECTED_MFR_KEYS)

        # 2. Verify pipeline extract equals direct extractor
        self.assertEqual(p_hp.extract(self.dummy_rgb), extract_branch_d_highpass_features(self.dummy_rgb))
        self.assertEqual(p_lap.extract(self.dummy_rgb), extract_branch_d_laplacian_features(self.dummy_rgb))
        self.assertEqual(p_mfr.extract(self.dummy_rgb), extract_branch_d_mfr_features(self.dummy_rgb))

        # 3. Alias 'D' expansion: 5 + 5 + 5 = 15 features
        p_d = ForensicPipeline(branches=["D"])
        expected_d_keys = EXPECTED_HIGHPASS_KEYS + EXPECTED_LAPLACIAN_KEYS + EXPECTED_MFR_KEYS
        self.assertEqual(len(p_d.get_feature_names()), 15)
        self.assertEqual(p_d.get_feature_names(), expected_d_keys)
        d_feats = p_d.extract(self.dummy_rgb)
        self.assertEqual(len(d_feats), 15)

        # 4. Multi-branch combination: A (34) + B (30) + C_LBP (16) + D_MFR (5) = 85 features
        p_multi = ForensicPipeline(branches=["A", "B", "C_LBP", "D_MFR"])
        self.assertEqual(len(p_multi.get_feature_names()), 85)
        multi_feats = p_multi.extract(self.dummy_rgb)
        self.assertEqual(len(multi_feats), 85)
        for v in multi_feats.values():
            self.assertTrue(math.isfinite(v))

        # 5. Candidate independence: D_MFR subvector in p_multi is bit-exact with p_mfr
        for k in EXPECTED_MFR_KEYS:
            self.assertEqual(multi_feats[k], p_mfr.extract(self.dummy_rgb)[k])


if __name__ == "__main__":
    unittest.main()
