"""On-Demand Canonical Sample Reader for Branch B Screening.

Reads individual preprocessed images from Block 1 Parquet parts on demand
without loading the 42,000-image dataset into memory.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


class DataIntegrityError(Exception):
    """Raised when sample data fails structural or cryptographic integrity checks."""


class ScreeningSampleReader:
    """Reads individual images on demand from Block 1 Parquet parts."""

    def __init__(
        self,
        manifest_csv: str,
        processed_train_dir: str,
    ) -> None:
        if not os.path.isfile(manifest_csv):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
        if not os.path.isdir(processed_train_dir):
            raise FileNotFoundError(f"Processed train dir not found: {processed_train_dir}")

        self.manifest_path = manifest_csv
        self.processed_train_dir = processed_train_dir
        self.manifest_df = pd.read_csv(manifest_csv)

        # Index by sample_idx for fast O(1) lookup
        self._lookup: Dict[int, pd.Series] = {
            int(row["sample_idx"]): row for _, row in self.manifest_df.iterrows()
        }

    def __len__(self) -> int:
        return len(self.manifest_df)

    def get_sample_info(self, sample_idx: int) -> Dict[str, Any]:
        """Retrieve manifest metadata for a sample index."""
        if sample_idx not in self._lookup:
            raise KeyError(f"Sample index {sample_idx} not found in screening manifest.")
        return dict(self._lookup[sample_idx])

    def load_image(
        self,
        sample_idx: int,
        as_float: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load canonical 256x256 RGB image array from its Parquet part.

        Args:
            sample_idx: Integer sample index.
            as_float: If True, returns normalized float32 tensor in [0.0, 1.0].
                     If False (default), returns uint8 tensor in [0, 255].

        Returns:
            Tuple of (image_tensor [3, 256, 256], metadata_dict).

        Raises:
            DataIntegrityError: If part file is missing or row byte length is invalid.
        """
        info = self.get_sample_info(sample_idx)
        part_name = info["parquet_part"]
        rg_idx = int(info["row_group"])
        row_in_rg = int(info["row_in_rg"])

        part_path = os.path.join(self.processed_train_dir, part_name)
        if not os.path.isfile(part_path):
            raise DataIntegrityError(
                f"Missing Parquet part file {part_path} for sample_idx={sample_idx}"
            )

        try:
            pf = pq.ParquetFile(part_path)
            # Explicitly request only columns present in Block 1 Parquet schema
            batch = pf.read_row_group(rg_idx, columns=["image_rgb", "label_a"])
            rgb_bytes_list = batch.column("image_rgb").to_pylist()
            label_a_list = batch.column("label_a").to_pylist()

            if row_in_rg >= len(rgb_bytes_list):
                raise DataIntegrityError(
                    f"row_in_rg={row_in_rg} out of range (rg has {len(rgb_bytes_list)} rows) in {part_path}"
                )

            # Integrity check: verify Parquet label_a matches manifest label_a
            parquet_label_a = int(label_a_list[row_in_rg])
            manifest_label_a = int(info["label_a"])
            if parquet_label_a != manifest_label_a:
                raise DataIntegrityError(
                    f"Label mismatch for sample {sample_idx}: Parquet label_a={parquet_label_a} "
                    f"vs manifest label_a={manifest_label_a}"
                )

            rgb_bytes = rgb_bytes_list[row_in_rg]
        except DataIntegrityError:
            raise
        except Exception as e:
            raise DataIntegrityError(
                f"Failed reading sample {sample_idx} from {part_path}: {e}"
            ) from e

        expected_bytes = 256 * 256 * 3
        if len(rgb_bytes) != expected_bytes:
            raise DataIntegrityError(
                f"Sample {sample_idx} byte length mismatch: expected {expected_bytes}, "
                f"got {len(rgb_bytes)}"
            )

        arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(256, 256, 3)
        # Transpose HWC -> CHW: [256, 256, 3] -> [3, 256, 256]
        arr_chw = np.transpose(arr, (2, 0, 1)).copy()
        tensor = torch.from_numpy(arr_chw)

        if as_float:
            tensor = tensor.to(dtype=torch.float32) / 255.0

        return tensor, info
