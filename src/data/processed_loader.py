"""
src/data/processed_loader.py
Block 1 — Materialized Processed Training Dataset Loader

Reads the processed training Parquet dataset produced by materializer.py.
This is the training-time reader: it reads pre-decoded RGB uint8 arrays
rather than raw JPEG bytes, so training does not pay JPEG decompression cost
on every sample access.

The processed dataset stores ONLY:
    image_rgb : binary  — 256×256×3 uint8 flat bytes
    label_a   : int32   — detector target (0=real, 1=AI)

It does NOT contain:
    label_b, caption, path, generator name, or any other metadata.

Label_B is available in the manifest.json for audit purposes. It is the
caller's responsibility to never use manifest metadata as model input.

CONTRAST WITH loader.py (on-demand path):
    loader.py          — reads raw Parquet, decodes JPEG per __getitem__
    processed_loader.py — reads processed Parquet, decodes uint8 per __getitem__

Both use the same preprocessing contract (DEC-008). The processed loader is
faster for training because RGB decoding from uint8 bytes is much cheaper
than JPEG decompression.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .materializer import PREPROCESSING_VERSION, get_processed_train_dir

# Constants — must match materializer.py
CANONICAL_SIZE = 256


class ProcessedTrainDataset(Dataset):
    """PyTorch Dataset backed by the materialized processed training Parquet files.

    Reads pre-decoded uint8 RGB arrays. No JPEG decompression at training time.

    Memory model:
        - __init__ builds a lightweight index (file_idx, rg_idx, row_in_rg,
          label_a) identical in design to DefactifyDataset. No image bytes
          are retained in memory at init.
        - __getitem__ reads one row group from the processed Parquet, extracts
          one sample's RGB bytes, converts to tensor, returns.

    Returns:
        (image_tensor, label_a)
        image_tensor : float32 [3, 256, 256] in [0.0, 1.0]
        label_a      : int, 0=real / 1=AI

    Args:
        processed_dir: Path to the materialized dataset directory
                       (e.g. data/processed/train/).
                       If None, uses default from materializer.get_processed_train_dir().
        transform:     Optional symmetric callable applied to the PIL Image
                       before tensor conversion (DEC-006).
    """

    def __init__(
        self,
        processed_dir: Optional[str] = None,
        transform=None,
    ) -> None:
        if processed_dir is None:
            processed_dir = get_processed_train_dir()

        if not os.path.isdir(processed_dir):
            raise FileNotFoundError(
                f"Processed training directory not found: {processed_dir}\n"
                "Run materialize_training_dataset() first."
            )

        self._processed_dir = processed_dir
        self.transform = transform

        parquet_paths = sorted(
            glob.glob(os.path.join(processed_dir, "part-*.parquet"))
        )
        if not parquet_paths:
            raise FileNotFoundError(
                f"No part-*.parquet files found in: {processed_dir}"
            )

        # Open ParquetFile handles (footer metadata only — kilobytes).
        self._parquet_files = [pq.ParquetFile(p) for p in parquet_paths]

        # Build flat lightweight index: read label_a column only.
        self._index = []  # list of (file_idx, rg_idx, row_in_rg, label_a)
        for file_idx, pf in enumerate(self._parquet_files):
            meta = pf.metadata
            for rg_idx in range(meta.num_row_groups):
                batch = pf.read_row_group(rg_idx, columns=["label_a"])
                labels = batch.column("label_a").to_pylist()
                for row_in_rg, la in enumerate(labels):
                    self._index.append((file_idx, rg_idx, row_in_rg, int(la)))

        self._len = len(self._index)

        # Load manifest if present (for inspection only — not used in __getitem__)
        manifest_path = os.path.join(processed_dir, "manifest.json")
        self.manifest: Optional[dict] = None
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        file_idx, rg_idx, row_in_rg, label_a = self._index[idx]

        # Read one row group from the processed Parquet.
        pf = self._parquet_files[file_idx]
        batch = pf.read_row_group(rg_idx, columns=["image_rgb"])
        rgb_bytes = batch.column("image_rgb").to_pylist()[row_in_rg]

        # Decode: flat uint8 bytes → numpy → PIL-equivalent shape [H, W, C]
        arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
            CANONICAL_SIZE, CANONICAL_SIZE, 3
        ).copy()

        # Convert numpy [H, W, C] uint8 → float32 tensor [C, H, W] in [0, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        # Optional symmetric transform (DEC-006)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label_a

    def __repr__(self) -> str:
        return (
            f"ProcessedTrainDataset("
            f"n={self._len}, "
            f"dir={os.path.basename(self._processed_dir)!r})"
        )
