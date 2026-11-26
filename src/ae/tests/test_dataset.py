"""Unit tests for AEDataset, RealAEDataset, and AIAEDataset (Block 2).

Verifies:
1. RealAEDataset returns only real images (label == 0).
2. AIAEDataset returns only AI-generated images (label == 1).
3. AEDataset(class_filter=None) returns full split with both real and AI images.
4. Cross-class evaluation capability (both models can evaluate on either class or mixed).
5. Zero redundant image datasets created on disk.
6. Images have the expected shape [3, 256, 256] and float32 range [0.0, 1.0].
7. Original label_a values are preserved.
8. Original sample_idx and image_id are preserved across filtered subsets.
9. Training (42k), validation (9k), and test (45k) splits are respected without leakage.
10. Real (7k) and AI (35k) training subsets have zero overlap and sum to 42,000.
11. Repeated access to the same sample is deterministic.
12. Optional transforms work symmetrically.
13. return_metadata=False mode returns exactly (tensor, label).
"""

from __future__ import annotations

import os
import unittest

import torch

from src.ae.dataset import AEDataset, AIAEDataset, RealAEDataset


class TestAEDataset(unittest.TestCase):
    """Test suite for Block 2 Autoencoder dataset implementations."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        cls.processed_train_dir = os.path.join(
            cls.project_root, "data", "processed", "train"
        )
        cls.raw_data_dir = os.path.join(
            cls.project_root, "data", "defactify", "data"
        )

    # ----------------------------------------------------------------------
    # 1. RealAEDataset filtering & properties
    # ----------------------------------------------------------------------
    def test_01_real_ae_dataset_training_split(self):
        """RealAEDataset on train split must contain exactly 7,000 real images."""
        ds = RealAEDataset(split="train")
        self.assertEqual(len(ds), 7000)
        self.assertEqual(ds.total_unfiltered_samples, 42000)

        # Inspect first 10 and last 10 samples
        indices_to_check = list(range(10)) + list(range(len(ds) - 10, len(ds)))
        for idx in indices_to_check:
            tensor, label, meta = ds[idx]
            self.assertEqual(label, 0, f"Sample at idx {idx} has label {label} != 0")
            self.assertEqual(tensor.shape, (3, 256, 256))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertGreaterEqual(tensor.min().item(), 0.0)
            self.assertLessEqual(tensor.max().item(), 1.0)
            self.assertTrue(meta["image_id"].startswith("train_"))
            self.assertEqual(meta["split"], "train")
            self.assertIsInstance(meta["sample_idx"], int)

    # ----------------------------------------------------------------------
    # 2. AIAEDataset filtering & properties
    # ----------------------------------------------------------------------
    def test_02_ai_ae_dataset_training_split(self):
        """AIAEDataset on train split must contain exactly 35,000 AI images."""
        ds = AIAEDataset(split="train")
        self.assertEqual(len(ds), 35000)
        self.assertEqual(ds.total_unfiltered_samples, 42000)

        # Inspect first 10 samples
        for idx in range(10):
            tensor, label, meta = ds[idx]
            self.assertEqual(label, 1, f"Sample at idx {idx} has label {label} != 1")
            self.assertEqual(tensor.shape, (3, 256, 256))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertGreaterEqual(tensor.min().item(), 0.0)
            self.assertLessEqual(tensor.max().item(), 1.0)
            self.assertTrue(meta["image_id"].startswith("train_"))
            self.assertEqual(meta["split"], "train")

    # ----------------------------------------------------------------------
    # 3. Disjoint partitions and full coverage
    # ----------------------------------------------------------------------
    def test_03_training_partitions_disjoint_and_complete(self):
        """Real and AI training datasets must have zero overlap and sum to 42,000."""
        real_ds = RealAEDataset(split="train")
        ai_ds = AIAEDataset(split="train")

        self.assertEqual(len(real_ds) + len(ai_ds), 42000)

        real_sample_indices = {entry.sample_idx for entry in real_ds._filtered_entries}
        ai_sample_indices = {entry.sample_idx for entry in ai_ds._filtered_entries}

        # Check zero overlap
        intersection = real_sample_indices.intersection(ai_sample_indices)
        self.assertEqual(len(intersection), 0, f"Found overlapping samples: {intersection}")

        # Check full coverage of 0..41,999
        union = real_sample_indices.union(ai_sample_indices)
        self.assertEqual(union, set(range(42000)))

    # ----------------------------------------------------------------------
    # 4. Shared evaluation dataset (class_filter=None)
    # ----------------------------------------------------------------------
    def test_04_shared_evaluation_dataset(self):
        """AEDataset with class_filter=None must load all images for shared evaluation."""
        val_ds = AEDataset(split="validation", class_filter=None)
        self.assertEqual(len(val_ds), 9000)

        # Spot check both classes are present
        sample_0 = val_ds[0]
        self.assertEqual(sample_0[0].shape, (3, 256, 256))
        self.assertEqual(sample_0[2]["split"], "validation")
        self.assertEqual(sample_0[2]["image_id"], "validation_000000")

    # ----------------------------------------------------------------------
    # 5. Cross-class evaluation capability
    # ----------------------------------------------------------------------
    def test_05_cross_class_evaluation_datasets(self):
        """Both models can be evaluated on real, AI, or mixed validation/test sets."""
        val_real = RealAEDataset(split="validation")
        val_ai = AIAEDataset(split="validation")

        self.assertEqual(len(val_real), 1500)
        self.assertEqual(len(val_ai), 7500)
        self.assertEqual(len(val_real) + len(val_ai), 9000)

        # Real validation samples have label 0
        _, label_real, meta_real = val_real[0]
        self.assertEqual(label_real, 0)
        self.assertEqual(meta_real["split"], "validation")

        # AI validation samples have label 1
        _, label_ai, meta_ai = val_ai[0]
        self.assertEqual(label_ai, 1)
        self.assertEqual(meta_ai["split"], "validation")

    # ----------------------------------------------------------------------
    # 6. Test split validation
    # ----------------------------------------------------------------------
    def test_06_test_split_datasets(self):
        """Test split must have 45,000 total (7,500 real, 37,500 AI)."""
        test_ds = AEDataset(split="test", class_filter=None)
        self.assertEqual(len(test_ds), 45000)

        test_real = RealAEDataset(split="test")
        self.assertEqual(len(test_real), 7500)

        test_ai = AIAEDataset(split="test")
        self.assertEqual(len(test_ai), 37500)

        self.assertEqual(len(test_real) + len(test_ai), 45000)

    # ----------------------------------------------------------------------
    # 7. Sample identity and image_id preservation
    # ----------------------------------------------------------------------
    def test_07_sample_identity_preservation(self):
        """Filtered datasets must preserve original sample_idx and image_id."""
        real_ds = RealAEDataset(split="train")
        # Check first real image
        _, label, meta = real_ds[0]
        orig_idx = meta["sample_idx"]
        self.assertEqual(meta["image_id"], f"train_{orig_idx:06d}")

        # Check full dataset at orig_idx has same label
        full_ds = AEDataset(split="train", class_filter=None)
        _, full_label, full_meta = full_ds[orig_idx]
        self.assertEqual(full_label, label)
        self.assertEqual(full_meta["image_id"], meta["image_id"])
        self.assertEqual(full_meta["sample_idx"], meta["sample_idx"])

    # ----------------------------------------------------------------------
    # 8. Deterministic repeatability
    # ----------------------------------------------------------------------
    def test_08_deterministic_repeatability(self):
        """Repeated accesses to the same sample must return bitwise identical tensors."""
        ds = RealAEDataset(split="train")
        tensor_run1, _, _ = ds[5]
        tensor_run2, _, _ = ds[5]
        self.assertTrue(torch.equal(tensor_run1, tensor_run2))

    # ----------------------------------------------------------------------
    # 9. Return metadata toggle
    # ----------------------------------------------------------------------
    def test_09_return_metadata_toggle(self):
        """When return_metadata=False, __getitem__ returns exactly (tensor, label)."""
        ds = RealAEDataset(split="train", return_metadata=False)
        item = ds[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)
        tensor, label = item
        self.assertEqual(tensor.shape, (3, 256, 256))
        self.assertEqual(label, 0)

    # ----------------------------------------------------------------------
    # 10. Optional transform
    # ----------------------------------------------------------------------
    def test_10_optional_transform(self):
        """Transform callable is applied symmetrically."""
        dummy_transform = lambda t: t * 0.5
        ds = RealAEDataset(split="train", transform=dummy_transform)
        tensor, _, _ = ds[0]
        self.assertLessEqual(tensor.max().item(), 0.5001)

    # ----------------------------------------------------------------------
    # 11. No derived datasets on disk
    # ----------------------------------------------------------------------
    def test_11_no_redundant_datasets_created(self):
        """Verify no data/ae_dataset/, data/real_ae/, or data/ai_ae/ directories exist."""
        forbidden_dirs = [
            os.path.join(self.project_root, "data", "ae_dataset"),
            os.path.join(self.project_root, "data", "real_ae"),
            os.path.join(self.project_root, "data", "ai_ae"),
        ]
        for fdir in forbidden_dirs:
            self.assertFalse(
                os.path.exists(fdir),
                f"Forbidden derived dataset directory found: {fdir}",
            )


if __name__ == "__main__":
    unittest.main()
