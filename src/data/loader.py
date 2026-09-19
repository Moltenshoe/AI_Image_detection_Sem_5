"""
src/data/loader.py
Block 1 — Defactify Dataset Loader

Wraps the Defactify Parquet files in a PyTorch Dataset using a lightweight
lazy index. Raw image bytes are NOT retained in memory; they are retrieved
on demand per __getitem__ call.

Schema (per-row, Parquet):
    Image   : struct {"bytes": binary, "path": string}  — Image["path"] DISCARDED
    Label_A : int32   (0 = real, 1 = AI)               — detector target
    Label_B : int32   (0–5, generator id)              — audit metadata ONLY
    Caption : string                                    — audit metadata ONLY

Split → Parquet files:
    train      : data/defactify/data/train-*.parquet      (7 files, 35 rg, 42,000 rows)
    validation : data/defactify/data/validation-*.parquet (2 files,  8 rg,  9,000 rows)
    test       : data/defactify/data/test-*.parquet       (8 files, 40 rg, 45,000 rows)

Memory model (LIGHTWEIGHT INDEX DESIGN):
    Initialization:
        - Opens each Parquet file with pq.ParquetFile (file handle only).
        - Reads Parquet file metadata (row counts per row group) from the
          footer — this is kilobytes, not megabytes.
        - Reads Label_A (and optionally Label_B, Caption) for ALL rows at
          init time. Label_A is an int32 column; at 4 bytes/row × 96,000
          rows this is ~375 KB total, well within acceptable memory bounds.
        - Does NOT read the Image column at init time.
        - Builds a flat index: for each global sample index stores
          (file_index, row_group_index, row_within_row_group).

    __getitem__:
        - Uses the flat index to locate the file and row group.
        - Opens the specific row group from the correct Parquet file.
        - Reads ONLY the Image column (and metadata columns if requested)
          for that row within the row group.
        - Runs canonical preprocessing on the raw bytes.
        - Returns the tensor and label. Raw bytes are discarded after use.

    What is NOT cached:
        - JPEG byte arrays for any image.
        - Decoded PIL Images.
        - Preprocessed tensors.

    What IS cached (lightweight):
        - pq.ParquetFile handles (file handles, footer metadata).
        - label_a values for all rows (~375 KB for full dataset).
        - Flat index mapping global index → (file_idx, rg_idx, row_in_rg).
        - Optionally label_b and caption for all rows when include_metadata=True.

Leakage controls:
    - Image["path"] is discarded immediately after extracting Image["bytes"].
    - Caption and Label_B are not returned unless include_metadata=True.
    - Official Defactify splits are NEVER reshuffled (DEC-001).
    - No data augmentation or randomness.
    - Preprocessing uses only the canonical pipeline from preprocessing.py.

No new dependencies required. Uses torch, pyarrow, Pillow (all installed).
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .preprocessing import preprocess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),   # src/data/
    "..", "..",                  # project root
    "data", "defactify", "data"
))

_SPLIT_PREFIXES = {
    "train":      "train-",
    "validation": "validation-",
    "test":       "test-",
}


# ---------------------------------------------------------------------------
# Index entry type (named for clarity)
# ---------------------------------------------------------------------------

class _IndexEntry:
    """Lightweight pointer to one sample within the Parquet corpus.

    Attributes:
        file_idx:    Index into self._parquet_files list.
        rg_idx:      Row group index within that Parquet file.
        row_in_rg:   Row offset within that row group.
        label_a:     Pre-loaded Label_A value (int). Avoids Parquet read for
                     label-only operations (e.g. class balancing).
        label_b:     Pre-loaded Label_B value (int) or None if not requested.
        caption:     Pre-loaded Caption string or None if not requested.
    """
    __slots__ = ("file_idx", "rg_idx", "row_in_rg", "label_a", "label_b", "caption")

    def __init__(
        self,
        file_idx: int,
        rg_idx: int,
        row_in_rg: int,
        label_a: int,
        label_b: Optional[int],
        caption: Optional[str],
    ) -> None:
        self.file_idx  = file_idx
        self.rg_idx    = rg_idx
        self.row_in_rg = row_in_rg
        self.label_a   = label_a
        self.label_b   = label_b
        self.caption   = caption


# ---------------------------------------------------------------------------
# DefactifyDataset
# ---------------------------------------------------------------------------


class DefactifyDataset(Dataset):
    """PyTorch Dataset backed by the Defactify Parquet files.

    Uses a lightweight index (file/row-group/row-offset + pre-loaded labels)
    so that initialization does NOT retain JPEG byte arrays in memory.
    Image bytes are fetched from Parquet on demand per __getitem__ call.

    Each __getitem__ returns:
        If include_metadata=False (default, detector mode):
            (image_tensor, label_a)
            image_tensor: float32 [3, 256, 256] in [0.0, 1.0]

        If include_metadata=True (audit/evaluation mode):
            (image_tensor, label_a, label_b, caption)
            label_b and caption are for auditing ONLY — must never be passed
            to the model (DEC-003, DEC-004).

    Args:
        split:            "train", "validation", or "test".
        data_dir:         Directory containing Parquet files. Defaults to
                          data/defactify/data/ relative to the project root.
        include_metadata: If True, pre-load and return Label_B + Caption.
                          These must NOT be used as model inputs.
        transform:        Optional callable applied to the PIL Image after
                          canonical preprocessing, before tensor conversion.
                          Must be applied symmetrically across classes (DEC-006).
    """

    def __init__(
        self,
        split: str,
        data_dir: Optional[str] = None,
        include_metadata: bool = False,
        transform=None,
    ) -> None:
        if split not in _SPLIT_PREFIXES:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of: "
                f"{list(_SPLIT_PREFIXES.keys())}"
            )

        self.split            = split
        self.include_metadata = include_metadata
        self.transform        = transform

        if data_dir is None:
            data_dir = _DEFAULT_DATA_DIR

        parquet_paths = sorted(
            glob.glob(os.path.join(data_dir, f"{_SPLIT_PREFIXES[split]}*.parquet"))
        )
        if not parquet_paths:
            raise FileNotFoundError(
                f"No Parquet files found for split '{split}' in: {data_dir}\n"
                f"Expected pattern: {_SPLIT_PREFIXES[split]}*.parquet"
            )

        # Open ParquetFile handles (reads footer metadata only — kilobytes).
        self._parquet_files: List[pq.ParquetFile] = [
            pq.ParquetFile(p) for p in parquet_paths
        ]

        # Columns to pre-load at init (labels only — no Image column).
        label_columns = ["Label_A"]
        if include_metadata:
            label_columns += ["Label_B", "Caption"]

        # Build flat index: read label columns row-by-row across all files.
        # Label_A is int32 → 4 bytes/row; 96k rows → ~375 KB max. Acceptable.
        self._index: List[_IndexEntry] = []

        for file_idx, pf in enumerate(self._parquet_files):
            meta = pf.metadata
            row_in_file = 0
            for rg_idx in range(meta.num_row_groups):
                rg_meta = meta.row_group(rg_idx)
                rg_size = rg_meta.num_rows

                # Read label columns for this row group only.
                batch = pf.read_row_group(rg_idx, columns=label_columns)
                label_a_col = batch.column("Label_A").to_pylist()
                label_b_col = (
                    batch.column("Label_B").to_pylist() if include_metadata else [None] * rg_size
                )
                caption_col = (
                    batch.column("Caption").to_pylist() if include_metadata else [None] * rg_size
                )

                for row_in_rg in range(rg_size):
                    self._index.append(
                        _IndexEntry(
                            file_idx=file_idx,
                            rg_idx=rg_idx,
                            row_in_rg=row_in_rg,
                            label_a=int(label_a_col[row_in_rg]),
                            label_b=(
                                int(label_b_col[row_in_rg])
                                if include_metadata else None
                            ),
                            caption=(
                                str(caption_col[row_in_rg])
                                if include_metadata else None
                            ),
                        )
                    )
                row_in_file += rg_size

        self._len = len(self._index)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        entry = self._index[idx]

        # Fetch ONLY the Image column for this single row group.
        # Raw bytes for this one row group are read, one row extracted, then
        # the Python object is discarded after preprocessing.
        pf = self._parquet_files[entry.file_idx]
        batch = pf.read_row_group(entry.rg_idx, columns=["Image"])
        image_structs = batch.column("Image").to_pylist()
        image_struct  = image_structs[entry.row_in_rg]

        # LEAKAGE CONTROL: extract bytes only; discard path.
        raw_bytes: bytes = image_struct["bytes"]

        # Canonical preprocessing: JPEG decode → RGB → center-crop → resize 256×256
        img: Image.Image = preprocess(raw_bytes)

        # Optional symmetric transform (DEC-006)
        if self.transform is not None:
            img = self.transform(img)

        # Convert PIL uint8 [H, W, C] → float32 tensor [C, H, W] in [0.0, 1.0]
        tensor = to_tensor(img)

        if self.include_metadata:
            return tensor, entry.label_a, entry.label_b, entry.caption

        return tensor, entry.label_a

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DefactifyDataset("
            f"split={self.split!r}, "
            f"n={self._len}, "
            f"include_metadata={self.include_metadata})"
        )
