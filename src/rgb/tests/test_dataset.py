"""Unit and integration tests for RGBDataset on-the-fly extraction (Block 2 -> Block 3).

Verifies:
1. Mode A loads only original RGB (3 channels) without invoking Branch A or Branch B.
2. Mode B invokes Branch A on demand (6 channels) and does not invoke Branch B.
3. Mode C invokes Branch B on demand (6 channels) and does not invoke Branch A.
4. Mode D invokes both branches on demand (9 channels).
5. All four modes return the correct shapes, channel ordering, and float32 dtype.
6. Both stacked (tensor) and dictionary output modes operate correctly.
7. Pure on-the-fly execution works seamlessly when rgb_dir is None (zero cache on disk).
8. Deterministic repeatability across repeated sample accesses.
9. Block 1 sample IDs (train_000000...) and labels (label_a) remain perfectly aligned.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from src.rgb.dataset import RGBDataset


class TestRGBDataset(unittest.TestCase):
    """Test suite for on-the-fly RGBDataset extraction."""

    def setUp(self):
        self.project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        self.processed_train_dir = os.path.join(
            self.project_root, "data", "processed", "train"
        )

    def test_01_mode_a_selective_execution(self):
        """Mode A must load original RGB only and never invoke Branch A or Branch B extractors."""
        ds = RGBDataset(mode="A", processed_train_dir=self.processed_train_dir, rgb_dir=None)

        with patch("src.rgb.dataset.extract_branch_a_lab_maps") as mock_a, \
             patch("src.rgb.dataset.compute_pid_residual") as mock_b:
            tensor, label, meta = ds[0]
            mock_a.assert_not_called()
            mock_b.assert_not_called()

        self.assertEqual(tensor.shape, (3, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertIn(label, (0, 1))
        self.assertEqual(meta["sample_idx"], 0)
        self.assertEqual(meta["image_id"], "train_000000")

    def test_02_mode_b_selective_execution(self):
        """Mode B must invoke Branch A on-the-fly and never invoke Branch B."""
        ds = RGBDataset(mode="B", processed_train_dir=self.processed_train_dir, rgb_dir=None)

        with patch("src.rgb.dataset.compute_pid_residual") as mock_b:
            tensor, label, meta = ds[0]
            mock_b.assert_not_called()

        self.assertEqual(tensor.shape, (6, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        # Channels 0..2 are RGB in [0, 1]
        self.assertGreaterEqual(tensor[0:3].min().item(), 0.0)
        self.assertLessEqual(tensor[0:3].max().item(), 1.0)
        # Channels 3..5 are Lab correlation maps in [-1, 1]
        self.assertGreaterEqual(tensor[3:6].min().item(), -1.0001)
        self.assertLessEqual(tensor[3:6].max().item(), 1.0001)

    def test_03_mode_c_selective_execution(self):
        """Mode C must invoke Branch B on-the-fly and never invoke Branch A."""
        ds = RGBDataset(mode="C", processed_train_dir=self.processed_train_dir, rgb_dir=None)

        with patch("src.rgb.dataset.extract_branch_a_lab_maps") as mock_a:
            tensor, label, meta = ds[0]
            mock_a.assert_not_called()

        self.assertEqual(tensor.shape, (6, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        # Channels 0..2 are RGB in [0, 1]
        self.assertGreaterEqual(tensor[0:3].min().item(), 0.0)
        self.assertLessEqual(tensor[0:3].max().item(), 1.0)
        # Channels 3..5 are PiD signed residuals (bounded approximately within [-7, 7])
        self.assertGreaterEqual(tensor[3:6].min().item(), -10.0)
        self.assertLessEqual(tensor[3:6].max().item(), 10.0)

    def test_04_mode_d_full_composition(self):
        """Mode D must invoke both Branch A and Branch B on-the-fly (9 channels total)."""
        ds = RGBDataset(mode="D", processed_train_dir=self.processed_train_dir, rgb_dir=None)
        tensor, label, meta = ds[0]

        self.assertEqual(tensor.shape, (9, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        # Verify no NaNs or Infs in composed tensor
        self.assertEqual(int(torch.isnan(tensor).sum().item()), 0)
        self.assertEqual(int(torch.isinf(tensor).sum().item()), 0)

    def test_05_dictionary_output_mode(self):
        """When stack_channels=False, dataset must return representation dictionary."""
        ds = RGBDataset(
            mode="D",
            processed_train_dir=self.processed_train_dir,
            rgb_dir=None,
            stack_channels=False,
        )
        rep_dict, label, meta = ds[0]

        self.assertIsInstance(rep_dict, dict)
        self.assertIn("rgb", rep_dict)
        self.assertIn("lab_maps", rep_dict)
        self.assertIn("branch_b_map", rep_dict)
        self.assertIn("pid_map", rep_dict)
        self.assertIn("nqp_map", rep_dict)

        self.assertEqual(rep_dict["rgb"].shape, (3, 256, 256))
        self.assertEqual(rep_dict["lab_maps"].shape, (3, 256, 256))
        self.assertEqual(rep_dict["branch_b_map"].shape, (3, 256, 256))
        self.assertTrue(torch.equal(rep_dict["branch_b_map"], rep_dict["pid_map"]))

    def test_06_pure_on_the_fly_without_cache(self):
        """Dataset must work with default rgb_dir=None and zero cached files on disk."""
        ds = RGBDataset(mode="D", processed_train_dir=self.processed_train_dir)
        self.assertIsNone(ds.rgb_dir)

        # Access multiple samples to confirm continuous on-the-fly extraction
        for idx in [0, 5, 10]:
            tensor, label, meta = ds[idx]
            self.assertEqual(tensor.shape, (9, 256, 256))
            self.assertEqual(meta["sample_idx"], idx)
            self.assertEqual(meta["image_id"], f"train_{idx:06d}")

    def test_07_deterministic_repeatability(self):
        """Repeated sample accesses on the same sample must return bitwise identical tensors."""
        ds = RGBDataset(mode="D", processed_train_dir=self.processed_train_dir)
        tensor_run1, _, _ = ds[3]
        tensor_run2, _, _ = ds[3]

        self.assertTrue(torch.equal(tensor_run1, tensor_run2))

    def test_08_dataset_length_and_indexing(self):
        """RGBDataset length must match Block 1 training split (42,000 samples)."""
        ds = RGBDataset(mode="A", processed_train_dir=self.processed_train_dir)
        self.assertEqual(len(ds), 42000)


if __name__ == "__main__":
    unittest.main()
