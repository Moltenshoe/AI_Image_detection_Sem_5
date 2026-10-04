"""Persistent Forensic Dataset Reader and Interface (Block 2 / Block 3 Interface).

Provides a high-performance, strongly-typed interface for downstream Block 3 models
and Block 4 evaluators to consume the persistent 111-feature forensic dataset without
importing, orchestrating, or recomputing feature extraction branches.

KEY CAPABILITIES:
    - Programmatic feature provenance mapping: Feature -> Subbranch -> Branch.
    - Budget-based feature filtering (111, 64, 32, 16, 8) via Branch F manifests.
    - Domain/branch-wise slicing (e.g. Branch A only, Branch E1 only, A+B).
    - Strict physical separation between metadata columns and model input matrices.
    - Zero runtime dependency on feature extraction operators (FFT, Haar, LBP, MFR, DCT).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.forensics.branch_f.registry import FeatureMetadata, FeatureRegistry


class _TorchForensicDataset(Dataset):
    """PyTorch Dataset wrapper for tensor-based training."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ForensicDataset:
    """Reader interface for the persistent canonical forensic dataset."""

    DEFAULT_DATASET_DIR: str = "data/forensic_dataset"

    METADATA_COLUMNS: Tuple[str, ...] = (
        "image_id",
        "sample_idx",
        "split",
        "source_path",
        "source_file",
        "source_row_group",
        "source_row",
        "label_a",
        "label_b",
        "generator_name",
        "caption",
    )

    def __init__(
        self,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        budget: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        branches: Optional[Sequence[str]] = None,
        subbranches: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
    ) -> None:
        """Initialize ForensicDataset reader.

        Args:
            dataset_dir: Directory containing 'features.parquet', 'feature_registry.json', etc.
            budget: If specified (8, 16, 32, 64, 111), load the mRMR selected subset.
            feature_names: Explicit list of feature names to load.
            branches: Specific branches to filter by (e.g. ['A', 'B']).
            subbranches: Specific subbranches to filter by (e.g. ['E1_DCT', 'A1_FFT']).
            split: Filter rows by split ('train', 'validation', 'test').
        """
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.features_parquet_path = os.path.join(self.dataset_dir, "features.parquet")

        if not os.path.exists(self.features_parquet_path):
            raise FileNotFoundError(
                f"Forensic dataset not found at '{self.features_parquet_path}'. "
                f"Run dataset materialization first."
            )

        # 1. Load Registry & Provenance
        reg_json_path = os.path.join(self.dataset_dir, "feature_registry.json")
        if os.path.exists(reg_json_path):
            with open(reg_json_path, "r", encoding="utf-8") as f:
                reg_data = json.load(f)
            self._registry_entries = [
                FeatureMetadata(**item) for item in reg_data
            ]
            self._registry_by_name = {e.name: e for e in self._registry_entries}
            self._all_feature_names = [e.name for e in self._registry_entries]
        else:
            registry = FeatureRegistry()
            self._registry_entries = registry.entries
            self._registry_by_name = {e.name: e for e in self._registry_entries}
            self._all_feature_names = registry.feature_names

        # 2. Resolve Active Feature Subset
        self._active_feature_names = self._resolve_feature_names(
            budget=budget,
            feature_names=feature_names,
            branches=branches,
            subbranches=subbranches,
        )

        # 3. Load Parquet Table
        cols_to_load = list(self.METADATA_COLUMNS) + self._active_feature_names
        table = pq.read_table(self.features_parquet_path, columns=cols_to_load)
        self._df = table.to_pandas()

        # Optional split filter
        if split is not None:
            self._df = self._df[self._df["split"] == split].reset_index(drop=True)

        self._metadata_df = self._df[list(self.METADATA_COLUMNS)].copy()
        self._features_df = self._df[self._active_feature_names].copy()

    def _resolve_feature_names(
        self,
        budget: Optional[int],
        feature_names: Optional[Sequence[str]],
        branches: Optional[Sequence[str]],
        subbranches: Optional[Sequence[str]],
    ) -> List[str]:
        """Resolve requested features against registry and selection manifests."""
        if budget is not None:
            manifest_path = os.path.join(
                self.dataset_dir, "selected", f"selected_features_{budget}.json"
            )
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                return list(manifest["features"])
            else:
                return self._all_feature_names[:budget]

        if feature_names is not None:
            for f in feature_names:
                if f not in self._registry_by_name:
                    raise KeyError(f"Feature '{f}' is not in canonical 111-feature registry.")
            return list(feature_names)

        if subbranches is not None:
            selected = [
                e.name for e in self._registry_entries if e.sub_branch in subbranches
            ]
            if not selected:
                raise ValueError(f"No features found matching subbranches {subbranches}.")
            return selected

        if branches is not None:
            selected = [
                e.name for e in self._registry_entries if e.branch in branches
            ]
            if not selected:
                raise ValueError(f"No features found matching branches {branches}.")
            return selected

        return list(self._all_feature_names)

    @property
    def feature_names(self) -> List[str]:
        """List of currently active feature column names."""
        return list(self._active_feature_names)

    @property
    def all_feature_names(self) -> List[str]:
        """List of all 111 canonical feature names in registry order."""
        return list(self._all_feature_names)

    @property
    def metadata_columns(self) -> List[str]:
        """List of metadata column names."""
        return list(self.METADATA_COLUMNS)

    @property
    def num_samples(self) -> int:
        """Total number of loaded samples."""
        return len(self._df)

    @property
    def num_features(self) -> int:
        """Number of active feature columns."""
        return len(self._active_feature_names)

    def get_features(self) -> np.ndarray:
        """Get model input feature matrix [N, D] as float32 numpy array."""
        return self._features_df.to_numpy(dtype=np.float32)

    def get_labels(self) -> np.ndarray:
        """Get supervised target labels (Label_A) [N] as int32 numpy array."""
        return self._metadata_df["label_a"].to_numpy(dtype=np.int32)

    def get_metadata(self) -> pd.DataFrame:
        """Get DataFrame containing all source and identity metadata."""
        return self._metadata_df.copy()

    def get_provenance(self, feature_name: str) -> Dict[str, Any]:
        """Retrieve provenance information for a feature."""
        if feature_name not in self._registry_by_name:
            raise KeyError(f"Feature '{feature_name}' not recognized.")
        entry = self._registry_by_name[feature_name]
        return {
            "name": entry.name,
            "index": entry.index,
            "branch": entry.branch,
            "sub_branch": entry.sub_branch,
            "domain": entry.domain,
            "data_type": entry.data_type,
            "description": entry.description,
            "sensitivity": entry.sensitivity,
        }

    def get_features_by_branch(self, branch: str) -> List[str]:
        """Return list of feature names belonging to a branch."""
        return [e.name for e in self._registry_entries if e.branch == branch]

    def get_features_by_subbranch(self, subbranch: str) -> List[str]:
        """Return list of feature names belonging to a subbranch."""
        return [e.name for e in self._registry_entries if e.sub_branch == subbranch]

    def to_dataframe(self, include_metadata: bool = True) -> pd.DataFrame:
        """Export dataset as pandas DataFrame."""
        if include_metadata:
            return self._df.copy()
        return self._features_df.copy()

    def to_torch_dataset(self) -> Dataset:
        """Convert to PyTorch Dataset returning (X_tensor, y_tensor)."""
        return _TorchForensicDataset(self.get_features(), self.get_labels())

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        feat_vec = self._features_df.iloc[idx].to_numpy(dtype=np.float32)
        label = int(self._metadata_df["label_a"].iloc[idx])
        meta_dict = self._metadata_df.iloc[idx].to_dict()
        return feat_vec, label, meta_dict
