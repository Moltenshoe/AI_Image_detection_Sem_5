"""Block 2 Autoencoder (AE) Dataset Interface.

Provides PyTorch Dataset implementations for the two class-specific autoencoders:
1. Real Autoencoder: Trained exclusively on real images (label_a == 0).
2. AI Autoencoder: Trained exclusively on AI-generated images (label_a == 1).

Key Architectural Guarantees:
- Reads directly from canonical Block 1 images (data/processed/train/ for materialized
  training data; on-demand canonical preprocessing via src.data.preprocessing.preprocess
  from raw Defactify Parquet for validation/test per DEC-008 and DEC-009).
- No redundant persistent image datasets or derived image copies are created.
  Temporary in-memory arrays and tensors required for loading are permitted.
- Preserves original Block 1 sample_idx and image_id (e.g. "train_000000",
  "validation_000000", "test_000000") even when filtering by class.
- Supports shared evaluation across both classes via class_filter=None on validation
  and test splits so both models can be evaluated on identical samples.
- Safe across PyTorch multiprocessing worker processes (num_workers > 0).
"""

from __future__ import annotations

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from src.data.materializer import CANONICAL_SIZE, get_processed_train_dir
from src.data.preprocessing import preprocess

# ---------------------------------------------------------------------------
# Default directories
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

_DEFAULT_RAW_DATA_DIR = os.path.join(
    _PROJECT_ROOT, "data", "defactify", "data"
)

_SPLIT_PREFIXES = {
    "train": "train-",
    "validation": "validation-",
    "test": "test-",
}


class _SampleIndexEntry:
    """Internal pointer to a sample within the underlying Parquet corpus."""

    __slots__ = (
        "file_idx",
        "rg_idx",
        "row_in_rg",
        "label_a",
        "sample_idx",
        "image_id",
    )

    def __init__(
        self,
        file_idx: int,
        rg_idx: int,
        row_in_rg: int,
        label_a: int,
        sample_idx: int,
        image_id: str,
    ) -> None:
        self.file_idx = file_idx
        self.rg_idx = rg_idx
        self.row_in_rg = row_in_rg
        self.label_a = label_a
        self.sample_idx = sample_idx
        self.image_id = image_id


class AEDataset(Dataset):
    """PyTorch Dataset reading Block 1 images for Autoencoder training and evaluation.

    Operates directly on canonical Block 1 images without duplicating data on disk.

    Supports:
    - Training split from materialized Block 1 Parquet (data/processed/train/).
    - Validation and test splits on-demand from raw Defactify Parquet or materialized splits.
    - Class filtering:
        * class_filter="real" (or 0): Real images only (label_a == 0).
        * class_filter="ai" (or 1): AI images only (label_a == 1).
        * class_filter=None (or "all"): All images (for shared cross-class evaluation).
    - Preserving canonical Block 1 sample_idx and image_id in return metadata.

    Args:
        split: "train", "validation", or "test". Defaults to "train".
        class_filter: "real", "ai", 0, 1, or None for all images. Defaults to None.
        data_dir: Optional explicit directory path. If None, resolves to canonical Block 1
                  paths (data/processed/train/ for train, data/defactify/data/ for val/test).
        transform: Optional callable applied symmetrically before tensor conversion.
        return_metadata: If True (default), __getitem__ returns (tensor, label_a, metadata).
                         If False, returns (tensor, label_a).
    """

    SUPPORTED_SPLITS = ("train", "validation", "test")
    SUPPORTED_FILTERS = ("real", "ai", 0, 1, None, "all")

    def __init__(
        self,
        split: str = "train",
        class_filter: Optional[Union[str, int]] = None,
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = True,
    ) -> None:
        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of: {list(self.SUPPORTED_SPLITS)}"
            )

        if class_filter not in self.SUPPORTED_FILTERS:
            raise ValueError(
                f"Invalid class_filter '{class_filter}'. Must be one of: {list(self.SUPPORTED_FILTERS)}"
            )

        self.split = split
        self.class_filter = self._normalize_class_filter(class_filter)
        self.transform = transform
        self.return_metadata = return_metadata

        # Resolve directory and loading strategy
        self._is_materialized, self._data_dir = self._resolve_data_source(split, data_dir)

        # Locate parquet files
        if self._is_materialized:
            parquet_paths = sorted(
                glob.glob(os.path.join(self._data_dir, "part-*.parquet"))
            )
            if not parquet_paths:
                raise FileNotFoundError(
                    f"No part-*.parquet found in materialized directory: {self._data_dir}"
                )
        else:
            prefix = _SPLIT_PREFIXES[split]
            parquet_paths = sorted(
                glob.glob(os.path.join(self._data_dir, f"{prefix}*.parquet"))
            )
            if not parquet_paths:
                raise FileNotFoundError(
                    f"No Parquet files found for split '{split}' in: {self._data_dir}\n"
                    f"Expected pattern: {prefix}*.parquet"
                )

        self._parquet_paths: List[str] = parquet_paths
        self._parquet_file_handles: Dict[int, pq.ParquetFile] = {}

        # Build lightweight index and filter by class
        self._all_entries: List[_SampleIndexEntry] = []
        self._filtered_entries: List[_SampleIndexEntry] = []
        self._build_index()

    @staticmethod
    def _normalize_class_filter(class_filter: Optional[Union[str, int]]) -> Optional[int]:
        if class_filter is None or class_filter == "all":
            return None
        if class_filter in ("real", 0):
            return 0
        if class_filter in ("ai", 1):
            return 1
        raise ValueError(f"Unrecognized class_filter: {class_filter}")

    def _resolve_data_source(
        self, split: str, explicit_dir: Optional[str]
    ) -> Tuple[bool, str]:
        """Determine whether data source is materialized (uint8) or raw (JPEG bytes)."""
        if explicit_dir is not None:
            # Check if directory contains part-*.parquet (materialized) or split-*.parquet (raw)
            if glob.glob(os.path.join(explicit_dir, "part-*.parquet")):
                return True, explicit_dir
            return False, explicit_dir

        if split == "train":
            # Materialized Block 1 training data is the canonical primary source
            processed_train = get_processed_train_dir()
            if os.path.isdir(processed_train) and glob.glob(
                os.path.join(processed_train, "part-*.parquet")
            ):
                return True, processed_train

        # Check if materialized split exists for validation/test
        processed_split = os.path.join(_PROJECT_ROOT, "data", "processed", split)
        if os.path.isdir(processed_split) and glob.glob(
            os.path.join(processed_split, "part-*.parquet")
        ):
            return True, processed_split

        # Fall back to raw Defactify parquet on demand (DEC-009)
        if os.path.isdir(_DEFAULT_RAW_DATA_DIR):
            return False, _DEFAULT_RAW_DATA_DIR

        raise FileNotFoundError(
            f"Could not locate canonical data for split '{split}'. Checked:\n"
            f"  - Materialized: data/processed/{split}/\n"
            f"  - Raw Defactify: {_DEFAULT_RAW_DATA_DIR}"
        )

    def _get_parquet_file(self, file_idx: int) -> pq.ParquetFile:
        """Lazily retrieve or open a ParquetFile handle for the given file index."""
        if not hasattr(self, "_parquet_file_handles") or self._parquet_file_handles is None:
            self._parquet_file_handles = {}
        if file_idx not in self._parquet_file_handles:
            self._parquet_file_handles[file_idx] = pq.ParquetFile(
                self._parquet_paths[file_idx]
            )
        return self._parquet_file_handles[file_idx]

    def __getstate__(self) -> Dict[str, Any]:
        """Custom pickler state omitting unpicklable C++ ParquetFile handles for workers."""
        state = self.__dict__.copy()
        state["_parquet_file_handles"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state in worker process with clean handle dictionary."""
        self.__dict__.update(state)
        self._parquet_file_handles = {}

    def _build_index(self) -> None:
        """Scan Parquet footers and label columns to build lightweight index."""
        global_sample_idx = 0
        label_col = "label_a" if self._is_materialized else "Label_A"

        for f_idx in range(len(self._parquet_paths)):
            pf = self._get_parquet_file(f_idx)
            meta = pf.metadata
            for rg_idx in range(meta.num_row_groups):
                batch = pf.read_row_group(rg_idx, columns=[label_col])
                labels = batch.column(label_col).to_pylist()
                for row_in_rg, la in enumerate(labels):
                    la_int = int(la)
                    entry = _SampleIndexEntry(
                        file_idx=f_idx,
                        rg_idx=rg_idx,
                        row_in_rg=row_in_rg,
                        label_a=la_int,
                        sample_idx=global_sample_idx,
                        image_id=f"{self.split}_{global_sample_idx:06d}",
                    )
                    self._all_entries.append(entry)

                    # Apply class filter if specified
                    if self.class_filter is None or la_int == self.class_filter:
                        self._filtered_entries.append(entry)

                    global_sample_idx += 1

    def __len__(self) -> int:
        return len(self._filtered_entries)

    def _load_tensor(self, entry: _SampleIndexEntry) -> torch.Tensor:
        """Load image and convert to standardized [3, 256, 256] float32 tensor in [0, 1]."""
        pf = self._get_parquet_file(entry.file_idx)

        if self._is_materialized:
            # Materialized path: read flat uint8 RGB bytes (DEC-009, DEC-010)
            batch = pf.read_row_group(entry.rg_idx, columns=["image_rgb"])
            rgb_bytes = batch.column("image_rgb").to_pylist()[entry.row_in_rg]
            arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                CANONICAL_SIZE, CANONICAL_SIZE, 3
            ).copy()
            # [H, W, 3] uint8 -> [3, H, W] float32 in [0, 1]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        else:
            # Raw Defactify path: read raw JPEG bytes and execute canonical preprocess (DEC-008)
            batch = pf.read_row_group(entry.rg_idx, columns=["Image"])
            image_structs = batch.column("Image").to_pylist()
            image_struct = image_structs[entry.row_in_rg]
            raw_bytes: bytes = image_struct["bytes"]

            # Canonical spatial preprocessing (DEC-008)
            img = preprocess(raw_bytes)
            # PIL Image -> float32 tensor [3, 256, 256] in [0, 1]
            tensor = to_tensor(img)

        # Optional symmetric transformation (DEC-006)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int, Dict[str, Any]], Tuple[torch.Tensor, int]]:
        entry = self._filtered_entries[idx]
        tensor = self._load_tensor(entry)

        if self.return_metadata:
            meta = {
                "sample_idx": entry.sample_idx,
                "image_id": entry.image_id,
                "split": self.split,
            }
            return tensor, entry.label_a, meta

        return tensor, entry.label_a

    @property
    def total_unfiltered_samples(self) -> int:
        """Total number of samples in this split before class filtering."""
        return len(self._all_entries)

    def __repr__(self) -> str:
        filter_str = (
            "all"
            if self.class_filter is None
            else ("real" if self.class_filter == 0 else "ai")
        )
        return (
            f"AEDataset("
            f"split={self.split!r}, "
            f"class_filter={filter_str!r}, "
            f"n={len(self._filtered_entries)}/{len(self._all_entries)})"
        )


class RealAEDataset(AEDataset):
    """Convenience dataset selecting only real images (label_a == 0).

    Default split is 'train' (7,000 samples). Supports 'validation' (1,500 samples)
    and 'test' (7,500 samples) when requested.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = True,
    ) -> None:
        super().__init__(
            split=split,
            class_filter="real",
            data_dir=data_dir,
            transform=transform,
            return_metadata=return_metadata,
        )


class AIAEDataset(AEDataset):
    """Convenience dataset selecting only AI-generated images (label_a == 1).

    Default split is 'train' (35,000 samples). Supports 'validation' (7,500 samples)
    and 'test' (37,500 samples) when requested.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = True,
    ) -> None:
        super().__init__(
            split=split,
            class_filter="ai",
            data_dir=data_dir,
            transform=transform,
            return_metadata=return_metadata,
        )
