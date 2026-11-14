"""Unit tests for Stratified Sampling and Immutable Manifest Management."""

from __future__ import annotations

import os
import tempfile
import unittest
import pandas as pd
import torch

from analysis.branch_b_screening.data.sampling import (
    select_screening_samples,
    verify_screening_manifest,
    STRATIFIED_TARGETS,
    TOTAL_SCREENING_SAMPLES,
    REQUIRED_COLUMNS,
)
from analysis.branch_b_screening.data.sample_reader import (
    ScreeningSampleReader,
    DataIntegrityError,
)


class TestSampling(unittest.TestCase):
    """Test suite for sampling protocol, schema verification, and reader safety."""

    def setUp(self):
        # Create a synthetic dataset manifest matching the real Block 1 schema
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manifest_path = os.path.join(self.temp_dir.name, "synthetic_manifest.csv")

        rows = []
        sample_idx = 0
        for gen_id, count in STRATIFIED_TARGETS.items():
            # Populate 50 samples per class to allow sampling
            for _ in range(50):
                rows.append({
                    "sample_idx": sample_idx,
                    "image_id": f"img_{sample_idx:06d}",
                    "split": "train",
                    "label_a": 0 if gen_id == 0 else 1,
                    "label_b": gen_id,
                    "generator_name": f"gen_{gen_id}",
                    "caption": "test caption",
                })
                sample_idx += 1

        df = pd.DataFrame(rows)
        df.to_csv(self.manifest_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_stratified_selection_counts_and_determinism(self):
        """Verify exact 100-sample count, per-generator quotas, and repeatable selection."""
        out_csv1 = os.path.join(self.temp_dir.name, "selected_1.csv")
        out_csv2 = os.path.join(self.temp_dir.name, "selected_2.csv")

        df1 = select_screening_samples(self.manifest_path, out_csv1, seed=42)
        df2 = select_screening_samples(self.manifest_path, out_csv2, seed=42)

        self.assertEqual(len(df1), TOTAL_SCREENING_SAMPLES)
        self.assertEqual(len(df2), TOTAL_SCREENING_SAMPLES)

        # Check exact per-class counts
        counts = df1["label_b"].value_counts().to_dict()
        for gen_id, expected in STRATIFIED_TARGETS.items():
            self.assertEqual(counts[gen_id], expected)

        # Check bit-exact repeat
        self.assertTrue(df1.equals(df2))

    def test_02_parquet_indexing_columns(self):
        """Verify parquet_part, row_group, row_in_rg calculation formula."""
        out_csv = os.path.join(self.temp_dir.name, "selected.csv")
        df = select_screening_samples(self.manifest_path, out_csv, seed=42)

        for col in REQUIRED_COLUMNS:
            self.assertIn(col, df.columns)

        for _, row in df.iterrows():
            idx = row["sample_idx"]
            self.assertEqual(row["parquet_part"], f"part-{idx // 1000:05d}.parquet")
            self.assertEqual(row["row_group"], (idx % 1000) // 500)
            self.assertEqual(row["row_in_rg"], (idx % 1000) % 500)

    def test_03_corrupt_manifest_schema_failure(self):
        """Verify manifest verification raises error on missing columns or row counts."""
        bad_csv = os.path.join(self.temp_dir.name, "bad_manifest.csv")
        # Missing row_in_rg
        bad_df = pd.DataFrame({"sample_idx": [0], "image_id": ["x"]})
        bad_df.to_csv(bad_csv, index=False)

        with self.assertRaises(ValueError):
            verify_screening_manifest(bad_csv)

    def test_04_missing_sample_failure(self):
        """Verify sample reader raises DataIntegrityError on missing part file."""
        out_csv = os.path.join(self.temp_dir.name, "selected.csv")
        df = select_screening_samples(self.manifest_path, out_csv, seed=42)
        valid_sample_idx = int(df["sample_idx"].iloc[0])

        reader = ScreeningSampleReader(
            manifest_csv=out_csv,
            processed_train_dir=self.temp_dir.name,  # Empty dir, no parquet parts
        )
        with self.assertRaises(DataIntegrityError):
            reader.load_image(sample_idx=valid_sample_idx)

    def test_05_parquet_reader_successful_load_and_label_integrity(self):
        """Verify reader loads schema [image_rgb, label_a] and verifies label parity."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_csv = os.path.join(self.temp_dir.name, "selected.csv")
        df = select_screening_samples(self.manifest_path, out_csv, seed=42)
        row0 = df.iloc[0]
        s_idx = int(row0["sample_idx"])
        part_name = row0["parquet_part"]
        rg = int(row0["row_group"])
        row_in_rg = int(row0["row_in_rg"])

        # Create synthetic 256x256x3 image bytes
        fake_bytes = bytes([128] * (256 * 256 * 3))
        num_rows = row_in_rg + 1
        bytes_list = [fake_bytes] * num_rows
        labels_list = [int(row0["label_a"])] * num_rows

        # Create part file with exact Block 1 schema: ["image_rgb", "label_a"]
        part_path = os.path.join(self.temp_dir.name, part_name)
        schema = pa.schema([
            ("image_rgb", pa.binary()),
            ("label_a", pa.int32()),
        ])

        # Write matching label_a
        table = pa.Table.from_arrays(
            [pa.array(bytes_list), pa.array(labels_list)],
            schema=schema,
        )
        with pq.ParquetWriter(part_path, schema) as writer:
            writer.write_table(table)

        reader = ScreeningSampleReader(
            manifest_csv=out_csv,
            processed_train_dir=self.temp_dir.name,
        )

        # 1. Successful load
        img_uint8, info = reader.load_image(sample_idx=s_idx, as_float=False)
        self.assertEqual(img_uint8.shape, (3, 256, 256))
        self.assertEqual(img_uint8.dtype, torch.uint8)
        self.assertEqual(info["image_id"], row0["image_id"])

        img_float, _ = reader.load_image(sample_idx=s_idx, as_float=True)
        self.assertEqual(img_float.dtype, torch.float32)
        self.assertAlmostEqual(img_float[0, 0, 0].item(), 128.0 / 255.0, places=4)

        # 2. Corrupt label mismatch detection
        bad_labels_list = [1 if row0["label_a"] == 0 else 0] * num_rows
        bad_table = pa.Table.from_arrays(
            [pa.array(bytes_list), pa.array(bad_labels_list)],
            schema=schema,
        )
        with pq.ParquetWriter(part_path, schema) as writer:
            writer.write_table(bad_table)

        with self.assertRaises(DataIntegrityError):
            reader.load_image(sample_idx=s_idx)


if __name__ == "__main__":
    unittest.main()
