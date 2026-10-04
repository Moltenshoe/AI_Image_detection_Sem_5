"""Unit and Integration Tests for Persistent Forensic Dataset & Reader (Block 2 Final).

Validates:
1.  Canonical 111-feature contract and exact ordering (A=34, B=30, C_LBP=16, D_MFR=5, E=26).
2.  Subbranch partition counts (A1=4, A2=30, B=30, C_LBP=16, D_MFR=5, E1=10, E2=8, E3=4, E4=4).
3.  Metadata vs. feature column isolation and leakage controls.
4.  Row identity, indexing, and generator label integrity.
5.  Finite values across all feature dimensions (no NaN/Inf).
6.  Feature registry schema and PyArrow metadata embedding.
7.  Provenance queries and branch/subbranch filtering.
8.  Branch F budget views (111, 64, 32, 16, 8) and manifest matching.
9.  PyTorch Dataset compatibility (to_torch_dataset).
10. Materializer overwrite protection.
11. Selected views: correct row count and feature counts.
12. No duplicate feature names.
"""

from __future__ import annotations

import json
import os
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from src.forensics.branch_f.registry import (
    EXPECTED_BRANCH_COUNTS,
    EXPECTED_FEATURE_COUNT,
    FeatureRegistry,
)
from src.forensics.dataset import ForensicDataset
from src.forensics.dataset_materializer import (
    GENERATOR_NAMES_MAP,
    SUBBRANCH_COUNTS,
    build_pyarrow_schema,
    materialize_forensic_dataset,
)

_DATASET_DIR = "data/forensic_dataset"


class TestForensicDataset(unittest.TestCase):
    """Test suite for persistent forensic dataset and reader interface."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = _DATASET_DIR
        cls.registry = FeatureRegistry()
        cls.expected_features = cls.registry.feature_names
        cls.ds = ForensicDataset(dataset_dir=cls.dataset_dir)

    # ------------------------------------------------------------------
    # 1. Canonical 111-feature contract
    # ------------------------------------------------------------------
    def test_canonical_111_feature_contract(self):
        """Dataset must expose exactly 111 canonical features in registry order."""
        self.assertEqual(EXPECTED_FEATURE_COUNT, 111)
        self.assertEqual(self.ds.num_features, 111)
        self.assertEqual(self.ds.feature_names, self.expected_features)

        # Parquet-level check
        pf = pq.ParquetFile(os.path.join(self.dataset_dir, "features.parquet"))
        col_names = pf.schema.names
        meta_cols = set(ForensicDataset.METADATA_COLUMNS)
        feature_cols = [c for c in col_names if c not in meta_cols]
        self.assertEqual(len(feature_cols), 111, f"Expected 111 feature cols, got {len(feature_cols)}")
        self.assertEqual(feature_cols, self.expected_features)

        # No duplicate feature names
        self.assertEqual(len(set(col_names)), len(col_names), "Duplicate column names detected")

        # Branch partition counts
        for branch, expected_cnt in EXPECTED_BRANCH_COUNTS.items():
            branch_feats = self.ds.get_features_by_branch(branch)
            self.assertEqual(
                len(branch_feats),
                expected_cnt,
                f"Branch {branch}: expected {expected_cnt}, got {len(branch_feats)}",
            )

    # ------------------------------------------------------------------
    # 2. Subbranch partition counts
    # ------------------------------------------------------------------
    def test_subbranch_partition_counts(self):
        """All subbranches must have exact counts summing to 111."""
        total = 0
        for subbranch, expected_cnt in SUBBRANCH_COUNTS.items():
            sub_feats = self.ds.get_features_by_subbranch(subbranch)
            self.assertEqual(
                len(sub_feats),
                expected_cnt,
                f"Subbranch {subbranch}: expected {expected_cnt}, got {len(sub_feats)}",
            )
            total += len(sub_feats)
        self.assertEqual(total, 111)

    # ------------------------------------------------------------------
    # 3. Metadata / feature isolation
    # ------------------------------------------------------------------
    def test_metadata_vs_feature_separation(self):
        """Metadata columns must be strictly isolated from model feature matrix."""
        self.assertEqual(len(ForensicDataset.METADATA_COLUMNS), 11)
        meta_df = self.ds.get_metadata()
        self.assertEqual(list(meta_df.columns), list(ForensicDataset.METADATA_COLUMNS))

        for m_col in ForensicDataset.METADATA_COLUMNS:
            self.assertNotIn(m_col, self.ds.feature_names, f"Metadata col '{m_col}' leaked into features")

        X = self.ds.get_features()
        y = self.ds.get_labels()
        self.assertEqual(X.ndim, 2)
        self.assertEqual(X.shape[1], 111)
        self.assertEqual(len(X), self.ds.num_samples)
        self.assertEqual(len(y), self.ds.num_samples)
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int32)

    # ------------------------------------------------------------------
    # 4. Row identity and deterministic order
    # ------------------------------------------------------------------
    def test_row_identity_and_deterministic_order(self):
        """image_id, sample_idx, label, generator_name must all be consistent."""
        meta_df = self.ds.get_metadata()
        for idx, row in meta_df.head(200).iterrows():
            self.assertEqual(row["image_id"], f"train_{idx:06d}")
            self.assertEqual(row["sample_idx"], idx)
            self.assertEqual(row["split"], "train")
            self.assertIn(int(row["label_a"]), (0, 1))
            self.assertIn(int(row["label_b"]), (0, 1, 2, 3, 4, 5))
            expected_gen = GENERATOR_NAMES_MAP.get(int(row["label_b"]))
            self.assertEqual(row["generator_name"], expected_gen)

    # ------------------------------------------------------------------
    # 5. Finite values
    # ------------------------------------------------------------------
    def test_finite_values(self):
        """No NaN or Inf values may appear in the feature matrix."""
        X = self.ds.get_features()
        nan_count = np.sum(np.isnan(X))
        inf_count = np.sum(np.isinf(X))
        self.assertEqual(nan_count, 0, f"Found {nan_count} NaN values")
        self.assertEqual(inf_count, 0, f"Found {inf_count} Inf values")

    # ------------------------------------------------------------------
    # 6. Feature registry validation
    # ------------------------------------------------------------------
    def test_feature_registry_validation(self):
        """feature_registry.csv and .json must exist and match canonical definitions."""
        csv_path = os.path.join(self.dataset_dir, "feature_registry.csv")
        json_path = os.path.join(self.dataset_dir, "feature_registry.json")
        self.assertTrue(os.path.exists(csv_path), "feature_registry.csv missing")
        self.assertTrue(os.path.exists(json_path), "feature_registry.json missing")

        df_reg = pd.read_csv(csv_path)
        self.assertEqual(len(df_reg), 111)
        self.assertEqual(list(df_reg["name"]), self.expected_features)

        with open(json_path, "r", encoding="utf-8") as f:
            json_reg = json.load(f)
        self.assertEqual(len(json_reg), 111)

        # PyArrow schema must embed branch/sub_branch metadata per feature
        schema = build_pyarrow_schema(self.registry)
        for fname in self.expected_features:
            field = schema.field(fname)
            self.assertIsNotNone(field.metadata, f"Field {fname} missing metadata")
            self.assertIn(b"branch", field.metadata, f"Field {fname} missing 'branch' key")
            self.assertIn(b"sub_branch", field.metadata, f"Field {fname} missing 'sub_branch' key")

    # ------------------------------------------------------------------
    # 7. Provenance queries
    # ------------------------------------------------------------------
    def test_provenance_queries(self):
        """get_provenance, get_features_by_branch, get_features_by_subbranch must work."""
        prov_a = self.ds.get_provenance("fft_low_freq_ratio")
        self.assertEqual(prov_a["branch"], "A")
        self.assertEqual(prov_a["sub_branch"], "A1_FFT")
        self.assertEqual(prov_a["domain"], "frequency")

        prov_d = self.ds.get_provenance("mfr_energy")
        self.assertEqual(prov_d["branch"], "D_MFR")
        self.assertEqual(prov_d["sub_branch"], "D_MFR")

        self.assertEqual(len(self.ds.get_features_by_branch("A")), 34)
        self.assertEqual(len(self.ds.get_features_by_branch("B")), 30)
        self.assertEqual(len(self.ds.get_features_by_branch("C_LBP")), 16)
        self.assertEqual(len(self.ds.get_features_by_branch("D_MFR")), 5)
        self.assertEqual(len(self.ds.get_features_by_branch("E")), 26)

        self.assertEqual(len(self.ds.get_features_by_subbranch("E1_DCT")), 10)
        self.assertEqual(len(self.ds.get_features_by_subbranch("E2_RESP")), 8)
        self.assertEqual(len(self.ds.get_features_by_subbranch("E3_PHASE")), 4)
        self.assertEqual(len(self.ds.get_features_by_subbranch("E4_GRID")), 4)

    # ------------------------------------------------------------------
    # 8. Budget views
    # ------------------------------------------------------------------
    def test_budget_views_and_filtering(self):
        """Selected budget views (8, 16, 32, 64, 111) must have correct feature counts."""
        expected_rows = self.ds.num_samples
        for b in [8, 16, 32, 64, 111]:
            ds_b = ForensicDataset(dataset_dir=self.dataset_dir, budget=b)
            self.assertEqual(ds_b.num_features, b, f"Budget {b}: expected {b} features, got {ds_b.num_features}")
            self.assertEqual(
                ds_b.get_features().shape[1], b,
                f"Budget {b}: feature matrix has wrong column count"
            )
            self.assertEqual(
                ds_b.num_samples, expected_rows,
                f"Budget {b}: expected {expected_rows} rows, got {ds_b.num_samples}"
            )

        # Branch/subbranch slice filters
        ds_a = ForensicDataset(dataset_dir=self.dataset_dir, branches=["A"])
        self.assertEqual(ds_a.num_features, 34)
        ds_d = ForensicDataset(dataset_dir=self.dataset_dir, subbranches=["D_MFR"])
        self.assertEqual(ds_d.num_features, 5)

        # Explicit feature name list
        target = ["fft_low_freq_ratio", "mfr_energy", "LL3_energy"]
        ds_custom = ForensicDataset(dataset_dir=self.dataset_dir, feature_names=target)
        self.assertEqual(ds_custom.num_features, 3)
        self.assertEqual(ds_custom.feature_names, target)

    # ------------------------------------------------------------------
    # 9. Selected Parquet views — row count and column count
    # ------------------------------------------------------------------
    def test_selected_parquet_views(self):
        """Each selected/features_N.parquet must have correct rows and feature cols."""
        expected_rows = self.ds.num_samples
        meta_col_count = len(ForensicDataset.METADATA_COLUMNS)
        sel_dir = os.path.join(self.dataset_dir, "selected")

        for b in [8, 16, 32, 64, 111]:
            parq_path = os.path.join(sel_dir, f"features_{b}.parquet")
            self.assertTrue(os.path.exists(parq_path), f"Missing {parq_path}")
            pf = pq.ParquetFile(parq_path)
            tbl = pf.read()
            self.assertEqual(
                len(tbl), expected_rows,
                f"selected/features_{b}.parquet: expected {expected_rows} rows, got {len(tbl)}"
            )
            feat_cols = [c for c in tbl.schema.names if c not in ForensicDataset.METADATA_COLUMNS]
            self.assertEqual(
                len(feat_cols), b,
                f"selected/features_{b}.parquet: expected {b} feature cols, got {len(feat_cols)}"
            )

            # Also verify the manifest JSON lists correct feature count
            manifest_path = os.path.join(sel_dir, f"selected_features_{b}.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["feature_count"], b)
            self.assertEqual(len(manifest["features"]), b)
            # All manifest features must exist in full dataset
            for fname in manifest["features"]:
                self.assertIn(fname, self.expected_features, f"Manifest feature '{fname}' not in registry")

    # ------------------------------------------------------------------
    # 10. PyTorch Dataset compatibility
    # ------------------------------------------------------------------
    def test_torch_dataset_integration(self):
        """to_torch_dataset() must return a working PyTorch Dataset."""
        torch_ds = self.ds.to_torch_dataset()
        self.assertEqual(len(torch_ds), self.ds.num_samples)

        x0, y0 = torch_ds[0]
        self.assertIsInstance(x0, torch.Tensor)
        self.assertIsInstance(y0, torch.Tensor)
        self.assertEqual(x0.shape, (111,))
        self.assertEqual(x0.dtype, torch.float32)
        self.assertEqual(y0.dtype, torch.int64)

    # ------------------------------------------------------------------
    # 11. __getitem__ indexing
    # ------------------------------------------------------------------
    def test_dataset_indexing(self):
        """__getitem__ must return (feature_vector, label, metadata_dict)."""
        feats, label, meta = self.ds[0]
        self.assertEqual(feats.shape, (111,))
        self.assertEqual(feats.dtype, np.float32)
        self.assertIn(label, (0, 1))
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["image_id"], "train_000000")

    # ------------------------------------------------------------------
    # 12. Overwrite protection
    # ------------------------------------------------------------------
    def test_materializer_overwrite_protection(self):
        """materialize_forensic_dataset must raise FileExistsError without overwrite=True."""
        with self.assertRaises(FileExistsError):
            materialize_forensic_dataset(
                output_dir=self.dataset_dir,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
