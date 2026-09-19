"""Unit tests for Block 2 — Branch C (Local Texture: LBP, GLCM, Edge-Guided LBP).

Verifies:
1. Exact feature count and dictionary key contracts for C_LBP (16), C_GLCM (24), C_LBP_EDGE (16).
2. Uniform LBP mathematical properties (transition counting, rotation-invariant bin mapping, sum to 1.0).
3. GLCM mathematical properties (16-level quantization, symmetry, normalization, Haralick properties).
4. Canny edge detection & sparse/empty mask fallback safety.
5. Determinism (bit-exact reproducibility).
6. Numerical safety across pathological inputs (all zeros, all ones, constant gray, checkerboards).
7. Texture pattern discrimination (smooth gradient vs noise vs oriented grids).
8. Real Defactify sample smoke extraction.
9. ForensicPipeline alternative isolation (no auto-concatenation, explicit branch names).
10. Multi-branch composition (A + C_LBP, B + C_GLCM, A + B + C_LBP).
11. Strict mutual exclusion of Branch C alternatives in ForensicPipeline.
12. Empty vs very sparse vs dense edge mask behaviors in C_LBP_EDGE.
13. Determinism of C_LBP_EDGE output.
"""

import math
import unittest

import numpy as np
import torch

from src.forensics.branch_c_texture import (
    compute_canny_edges,
    compute_glcm_features,
    compute_glcm_matrix,
    compute_lbp_map,
    extract_branch_c_glcm_features,
    extract_branch_c_lbp_edge_features,
    extract_branch_c_lbp_features,
    extract_lbp_statistics,
    quantize_grayscale,
)
from src.forensics.pipeline import ForensicPipeline


class TestBranchCTexture(unittest.TestCase):
    """Test suite for Branch C Local Texture forensic feature extraction."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.dummy_rgb = torch.rand(3, 256, 256, dtype=torch.float32)

    def test_01_c_lbp_contract(self):
        """Verify C_LBP returns exactly 16 expected scalar float features."""
        feats = extract_branch_c_lbp_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 16, f"Expected 16 features, got {len(feats)}")

        expected_keys = [
            "lbp_bin_0", "lbp_bin_1", "lbp_bin_2", "lbp_bin_3",
            "lbp_bin_4", "lbp_bin_5", "lbp_bin_6", "lbp_bin_7",
            "lbp_bin_8", "lbp_bin_nonuniform",
            "lbp_entropy", "lbp_uniformity", "lbp_dominant_bin",
            "lbp_mean_code", "lbp_std_code", "lbp_nonuniform_ratio",
        ]
        for key in expected_keys:
            self.assertIn(key, feats)
            self.assertIsInstance(feats[key], float)
            self.assertTrue(math.isfinite(feats[key]), f"Feature {key} is not finite: {feats[key]}")

        # Histogram sum should equal 1.0 (within float tolerance)
        hist_sum = sum(feats[f"lbp_bin_{i}"] for i in range(9)) + feats["lbp_bin_nonuniform"]
        self.assertAlmostEqual(hist_sum, 1.0, places=5)
        self.assertAlmostEqual(feats["lbp_bin_nonuniform"], feats["lbp_nonuniform_ratio"], places=5)

    def test_02_c_glcm_contract(self):
        """Verify C_GLCM returns exactly 24 expected scalar float features."""
        feats = extract_branch_c_glcm_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 24, f"Expected 24 features, got {len(feats)}")

        expected_stats = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "entropy"]
        expected_keys = []
        for d in [1, 2, 4]:
            for stat in expected_stats:
                expected_keys.append(f"glcm_d{d}_{stat}_mean")
        for stat in expected_stats:
            expected_keys.append(f"glcm_d1_{stat}_std")

        for key in expected_keys:
            self.assertIn(key, feats)
            self.assertIsInstance(feats[key], float)
            self.assertTrue(math.isfinite(feats[key]), f"Feature {key} is not finite: {feats[key]}")

    def test_03_c_lbp_edge_contract(self):
        """Verify C_LBP_EDGE returns exactly 16 expected scalar float features."""
        feats = extract_branch_c_lbp_edge_features(self.dummy_rgb)

        self.assertIsInstance(feats, dict)
        self.assertEqual(len(feats), 16, f"Expected 16 features, got {len(feats)}")

        expected_keys = [
            "lbp_edge_bin_0", "lbp_edge_bin_1", "lbp_edge_bin_2", "lbp_edge_bin_3",
            "lbp_edge_bin_4", "lbp_edge_bin_5", "lbp_edge_bin_6", "lbp_edge_bin_7",
            "lbp_edge_bin_8", "lbp_edge_bin_nonuniform",
            "lbp_edge_entropy", "lbp_edge_uniformity", "lbp_edge_dominant_bin",
            "lbp_edge_mean_code", "lbp_edge_std_code", "lbp_edge_pixel_density",
        ]
        for key in expected_keys:
            self.assertIn(key, feats)
            self.assertIsInstance(feats[key], float)
            self.assertTrue(math.isfinite(feats[key]), f"Feature {key} is not finite: {feats[key]}")

        hist_sum = sum(feats[f"lbp_edge_bin_{i}"] for i in range(9)) + feats["lbp_edge_bin_nonuniform"]
        self.assertAlmostEqual(hist_sum, 1.0, places=5)
        self.assertGreaterEqual(feats["lbp_edge_pixel_density"], 0.0)
        self.assertLessEqual(feats["lbp_edge_pixel_density"], 1.0)

    def test_04_lbp_mathematical_properties_and_mapping(self):
        """Verify rotation-invariant uniform LBP transition counting and bin mapping."""
        # 1. Constant image: all neighbors == center -> all 1s -> uniform pattern with 8 ones -> bin 8
        const_gray = torch.full((1, 256, 256), 0.5, dtype=torch.float32)
        lbp_map = compute_lbp_map(const_gray, p=8, r=1)
        self.assertEqual(lbp_map.shape, (256, 256))
        # Since replicate padding was used, all pixels should equal 8
        self.assertTrue((lbp_map == 8).all().item())

        # 2. Synthetic isolated spot (center < all neighbors -> 8 ones = bin 8; center > all neighbors -> 0 ones = bin 0)
        spot_img = torch.zeros((1, 16, 16), dtype=torch.float32)
        spot_img[0, 8, 8] = 1.0  # bright spot on dark background
        spot_lbp = compute_lbp_map(spot_img, p=8, r=1)
        # Center (8, 8) is greater than all its 8 neighbors -> all bits are 0 -> bin 0
        self.assertEqual(spot_lbp[8, 8].item(), 0)

        # 3. Checkerboard alternating pattern (non-uniform: 8 transitions > 2 -> bin 9)
        checker = torch.zeros((1, 16, 16), dtype=torch.float32)
        checker[0, 0::2, 0::2] = 1.0
        checker[0, 1::2, 1::2] = 1.0
        checker_lbp = compute_lbp_map(checker, p=8, r=1)
        # For interior pixels of checkerboard, circular neighbors alternate 0,1,0,1,0,1,0,1 -> 8 transitions -> bin 9
        interior = checker_lbp[2:-2, 2:-2]
        self.assertTrue((interior == 9).any().item())

    def test_05_glcm_mathematical_properties_and_quantization(self):
        """Verify GLCM quantization, matrix shape, symmetry, and normalization."""
        # 1. Quantization
        gray_test = torch.linspace(0.0, 1.0, 256).unsqueeze(0).repeat(256, 1)
        quantized = quantize_grayscale(gray_test, levels=16)
        self.assertEqual(quantized.shape, (256, 256))
        self.assertEqual(quantized.min(), 0)
        self.assertEqual(quantized.max(), 15)

        # 2. GLCM Matrix
        glcm = compute_glcm_matrix(quantized, levels=16, distances=(1, 2, 4), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4))
        self.assertEqual(glcm.shape, (16, 16, 3, 4))

        # Check symmetry: P(i, j) == P(j, i)
        for d_i in range(3):
            for a_i in range(4):
                mat = glcm[:, :, d_i, a_i]
                np.testing.assert_allclose(mat, mat.T, atol=1e-10)
                # Check normalization: sum == 1.0
                self.assertAlmostEqual(float(mat.sum()), 1.0, places=6)

    def test_06_edge_detection_and_sparse_mask_safety(self):
        """Verify Canny edge detection and safety on sparse/empty edge masks."""
        # Flat image -> zero edge pixels
        flat_gray = torch.zeros((1, 256, 256), dtype=torch.float32)
        edges = compute_canny_edges(flat_gray)
        self.assertEqual(edges.shape, (256, 256))
        self.assertEqual(edges.sum(), 0)

        # Extract edge features on flat image -> should fall back safely without NaNs
        edge_feats = extract_branch_c_lbp_edge_features(flat_gray)
        self.assertEqual(len(edge_feats), 16)
        for k, v in edge_feats.items():
            self.assertTrue(math.isfinite(v), f"Feature {k} not finite on flat image: {v}")
        self.assertEqual(edge_feats["lbp_edge_pixel_density"], 0.0)

    def test_07_determinism(self):
        """Verify that repeated extractions yield bit-exact identical features."""
        torch.manual_seed(123)
        img = torch.rand(3, 256, 256, dtype=torch.float32)

        lbp_1 = extract_branch_c_lbp_features(img)
        lbp_2 = extract_branch_c_lbp_features(img)
        self.assertEqual(lbp_1, lbp_2)

        glcm_1 = extract_branch_c_glcm_features(img)
        glcm_2 = extract_branch_c_glcm_features(img)
        self.assertEqual(glcm_1, glcm_2)

        edge_1 = extract_branch_c_lbp_edge_features(img)
        edge_2 = extract_branch_c_lbp_edge_features(img)
        self.assertEqual(edge_1, edge_2)

    def test_08_numerical_safety_pathological_inputs(self):
        """Verify numerical safety on all zeros, all ones, constant gray, and noise."""
        pathological = [
            ("all_zeros", torch.zeros(3, 256, 256, dtype=torch.float32)),
            ("all_ones", torch.ones(3, 256, 256, dtype=torch.float32)),
            ("constant_gray", torch.full((3, 256, 256), 0.5, dtype=torch.float32)),
            ("checkerboard", torch.zeros(3, 256, 256, dtype=torch.float32)),
        ]
        pathological[3][1][:, 0::2, 0::2] = 1.0
        pathological[3][1][:, 1::2, 1::2] = 1.0

        for name, tensor in pathological:
            for branch_fn in [extract_branch_c_lbp_features, extract_branch_c_glcm_features, extract_branch_c_lbp_edge_features]:
                feats = branch_fn(tensor)
                for k, v in feats.items():
                    self.assertTrue(
                        math.isfinite(v),
                        f"Non-finite value {v} in {k} for {name} using {branch_fn.__name__}",
                    )

    def test_09_texture_discrimination(self):
        """Verify distinct texture patterns produce distinct feature vectors."""
        smooth = torch.linspace(0.0, 1.0, 256).unsqueeze(0).repeat(3, 256, 1)
        noise = torch.rand(3, 256, 256, dtype=torch.float32)

        lbp_smooth = extract_branch_c_lbp_features(smooth)
        lbp_noise = extract_branch_c_lbp_features(noise)
        self.assertNotEqual(lbp_smooth["lbp_entropy"], lbp_noise["lbp_entropy"])

        glcm_smooth = extract_branch_c_glcm_features(smooth)
        glcm_noise = extract_branch_c_glcm_features(noise)
        self.assertNotEqual(glcm_smooth["glcm_d1_contrast_mean"], glcm_noise["glcm_d1_contrast_mean"])

    def test_10_smoke_real_defactify_sample(self):
        """Smoke test extracting Branch C features on real Defactify dataset sample."""
        from src.data.loader import DefactifyDataset

        dataset = DefactifyDataset(split="train")
        img_tensor, label_a = dataset[0]

        lbp_feats = extract_branch_c_lbp_features(img_tensor)
        glcm_feats = extract_branch_c_glcm_features(img_tensor)
        edge_feats = extract_branch_c_lbp_edge_features(img_tensor)

        self.assertEqual(len(lbp_feats), 16)
        self.assertEqual(len(glcm_feats), 24)
        self.assertEqual(len(edge_feats), 16)

        for feats in [lbp_feats, glcm_feats, edge_feats]:
            for k, v in feats.items():
                self.assertTrue(math.isfinite(v), f"Feature {k} not finite: {v}")

    def test_11_forensic_pipeline_isolation_and_alternatives(self):
        """Verify ForensicPipeline orchestrates C_LBP, C_GLCM, C_LBP_EDGE independently."""
        # 1. Individual alternative branches
        p_lbp = ForensicPipeline(["C_LBP"])
        self.assertEqual(len(p_lbp.get_feature_names()), 16)
        feats_lbp = p_lbp.extract(self.dummy_rgb)
        self.assertEqual(feats_lbp, extract_branch_c_lbp_features(self.dummy_rgb))

        p_glcm = ForensicPipeline(["C_GLCM"])
        self.assertEqual(len(p_glcm.get_feature_names()), 24)
        feats_glcm = p_glcm.extract(self.dummy_rgb)
        self.assertEqual(feats_glcm, extract_branch_c_glcm_features(self.dummy_rgb))

        p_edge = ForensicPipeline(["C_LBP_EDGE"])
        self.assertEqual(len(p_edge.get_feature_names()), 16)
        feats_edge = p_edge.extract(self.dummy_rgb)
        self.assertEqual(feats_edge, extract_branch_c_lbp_edge_features(self.dummy_rgb))

        # 2. Generic "C" must raise ValueError requiring explicit selection
        with self.assertRaises(ValueError):
            ForensicPipeline(["C"])

    def test_12_multi_branch_combination(self):
        """Verify multi-branch combinations produce expected combined feature counts."""
        # A + C_LBP -> 34 + 16 = 50
        p_a_lbp = ForensicPipeline(["A", "C_LBP"])
        self.assertEqual(len(p_a_lbp.get_feature_names()), 50)
        feats_a_lbp = p_a_lbp.extract(self.dummy_rgb)
        self.assertEqual(len(feats_a_lbp), 50)

        # B + C_GLCM -> 30 + 24 = 54
        p_b_glcm = ForensicPipeline(["B", "C_GLCM"])
        self.assertEqual(len(p_b_glcm.get_feature_names()), 54)
        feats_b_glcm = p_b_glcm.extract(self.dummy_rgb)
        self.assertEqual(len(feats_b_glcm), 54)

        # A + B + C_LBP -> 34 + 30 + 16 = 80
        p_all = ForensicPipeline(["A", "B", "C_LBP"])
        self.assertEqual(len(p_all.get_feature_names()), 80)
        feats_all = p_all.extract(self.dummy_rgb)
        self.assertEqual(len(feats_all), 80)

    def test_13_mutual_exclusion_of_branch_c_alternatives(self):
        """Verify ForensicPipeline strictly rejects combining multiple Branch C variants."""
        invalid_combinations = [
            ["C_LBP", "C_GLCM"],
            ["C_LBP", "C_LBP_EDGE"],
            ["C_GLCM", "C_LBP_EDGE"],
            ["C_LBP", "C_GLCM", "C_LBP_EDGE"],
            ["A", "C_LBP", "C_GLCM"],
            ["B", "C_GLCM", "C_LBP_EDGE"],
            ["A", "B", "C_LBP", "C_LBP_EDGE"],
        ]
        for combo in invalid_combinations:
            with self.assertRaises(ValueError, msg=f"Failed to reject invalid combo: {combo}"):
                ForensicPipeline(branches=combo)

    def test_14_empty_and_very_sparse_edge_masks(self):
        """Verify exact behavior of C_LBP_EDGE across empty, very sparse, and dense edge masks."""
        dummy_lbp_map = torch.randint(0, 10, (256, 256), dtype=torch.int64)

        # 1. Completely empty mask (|E| = 0)
        empty_mask = np.zeros((256, 256), dtype=np.uint8)
        feats_empty = extract_lbp_statistics(dummy_lbp_map, mask=empty_mask, prefix="lbp_edge")
        self.assertEqual(feats_empty["lbp_edge_pixel_density"], 0.0)
        for i in range(9):
            self.assertAlmostEqual(feats_empty[f"lbp_edge_bin_{i}"], 0.1, places=5)
        self.assertAlmostEqual(feats_empty["lbp_edge_bin_nonuniform"], 0.1, places=5)
        self.assertTrue(math.isfinite(feats_empty["lbp_edge_entropy"]))
        self.assertGreater(feats_empty["lbp_edge_entropy"], 0.0)

        # 2. Very sparse mask (1 <= |E| < 10, e.g. exactly 3 edge pixels)
        sparse_mask = np.zeros((256, 256), dtype=np.uint8)
        sparse_mask[10, 10] = 1
        sparse_mask[20, 20] = 1
        sparse_mask[30, 30] = 1
        feats_sparse = extract_lbp_statistics(dummy_lbp_map, mask=sparse_mask, prefix="lbp_edge")
        self.assertAlmostEqual(feats_sparse["lbp_edge_pixel_density"], 3.0 / (256 * 256), places=7)
        # Verify fallback distribution is used for sparse mask
        for i in range(9):
            self.assertAlmostEqual(feats_sparse[f"lbp_edge_bin_{i}"], 0.1, places=5)
        self.assertTrue(math.isfinite(feats_sparse["lbp_edge_entropy"]))

        # 3. Dense mask (|E| >= 10, e.g. 500 edge pixels)
        dense_mask = np.zeros((256, 256), dtype=np.uint8)
        dense_mask[:50, :10] = 1  # 500 pixels
        feats_dense = extract_lbp_statistics(dummy_lbp_map, mask=dense_mask, prefix="lbp_edge")
        self.assertAlmostEqual(feats_dense["lbp_edge_pixel_density"], 500.0 / (256 * 256), places=6)
        hist_sum = sum(feats_dense[f"lbp_edge_bin_{i}"] for i in range(9)) + feats_dense["lbp_edge_bin_nonuniform"]
        self.assertAlmostEqual(hist_sum, 1.0, places=5)

    def test_15_c_lbp_edge_determinism_and_components(self):
        """Verify deterministic C_LBP_EDGE output and parameter consistency."""
        torch.manual_seed(999)
        img = torch.rand(3, 256, 256, dtype=torch.float32)

        # Bit-exact repeated run
        res_1 = extract_branch_c_lbp_edge_features(img)
        res_2 = extract_branch_c_lbp_edge_features(img)
        self.assertEqual(res_1, res_2)

        # Verify Canny edge mask generator produces consistent binary numpy array
        edges_1 = compute_canny_edges(img[0], low_threshold=50.0, high_threshold=100.0, sigma=1.0)
        edges_2 = compute_canny_edges(img[0], low_threshold=50.0, high_threshold=100.0, sigma=1.0)
        np.testing.assert_array_equal(edges_1, edges_2)
        self.assertTrue(set(np.unique(edges_1)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
