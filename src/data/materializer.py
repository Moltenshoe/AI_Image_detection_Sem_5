"""
src/data/materializer.py
Block 1 — Training Dataset Materialization

Reads raw Defactify Parquet files for the TRAINING split, applies the
canonical preprocessing pipeline, and writes a versioned processed Parquet
dataset alongside a JSON manifest.

WHY MATERIALIZE ONLY TRAINING?
    Training involves many epochs; re-decoding 42,000 raw JPEG images from
    Parquet every epoch is expensive. The processed training dataset caches
    the deterministic preprocessing output so training reads pre-decoded
    RGB arrays instead of raw JPEG bytes.

WHY NOT MATERIALIZE VALIDATION / TEST?
    Validation and test sets are used once per epoch (or less). On-demand
    processing through the raw Parquet loader (DefactifyDataset) is
    sufficient and avoids storing a second copy of the data. See loader.py
    for the on-demand path.

OUTPUT FORMAT (DEC-009):
    Parquet files containing columns:
        image_rgb : binary — 256×256×3 uint8 flat bytes (row-major RGB)
        label_a   : int32  — 0=real, 1=AI  (detector target ONLY)

    Columns NOT included in the processed dataset (leakage controls):
        label_b   — stored in manifest ONLY (audit metadata)
        caption   — excluded entirely (not detector evidence)
        path      — excluded entirely (not detector evidence)

    The audit manifest (JSON) retains per-sample label_b for generator-specific
    evaluation. The manifest is separate from the tensor input.

OUTPUT DIRECTORY STRUCTURE:
    data/processed/train/
        part-00000.parquet
        part-00001.parquet
        ...
        manifest.json

The manifest records the preprocessing configuration (version v2, cv2.INTER_AREA).

REPRODUCIBILITY:
    - Deterministic: same raw data + same preprocessing version → same output.
    - Raw data is never modified.
    - All derived data is written to data/processed/, which is separate from
      data/defactify/ (the immutable raw source).

No new dependencies required. Uses pyarrow, Pillow (already installed).
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .preprocessing import CANONICAL_SIZE, INTERPOLATION, preprocess

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

# Version string encodes the preprocessing configuration.
# Bump this if any preprocessing parameter changes (DEC-008 contract).
PREPROCESSING_VERSION: str = "v2"

# Parquet write options for the processed dataset.
# 1,000 rows × 196,608 bytes/image ≈ 196.6 MB buffer per flush batch.
_ROWS_PER_FILE: int = 1000
_ROW_GROUP_SIZE: int = 500   # Row group size within each output file

# ---------------------------------------------------------------------------
# Schema for the processed training Parquet
# ---------------------------------------------------------------------------

_PROCESSED_SCHEMA = pa.schema([
    pa.field("image_rgb", pa.binary(),
             metadata={b"description": b"256x256x3 uint8 RGB bytes, row-major"}),
    pa.field("label_a",   pa.int32(),
             metadata={b"description": b"0=real, 1=AI (detector target)"}),
])

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def materialize_training_dataset(
    raw_data_dir: Optional[str] = None,
    output_base_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
) -> str:
    """Materialize the processed training dataset from raw Parquet files.

    Reads the raw train split, applies canonical preprocessing (DEC-008),
    and writes processed Parquet files + a JSON manifest.

    Args:
        raw_data_dir:    Path to raw Parquet files. Defaults to
                         data/defactify/data/ relative to project root.
        output_base_dir: Root for processed output. Defaults to
                         data/processed/ relative to project root.
        max_samples:     If set, stop after this many samples (for testing/
                         smoke tests). None = full dataset.
        overwrite:       If False (default), raises if the output directory
                         already exists to prevent silent overwrites.

    Returns:
        Path to the output directory containing the materialized dataset.

    Raises:
        FileExistsError: If the output directory already exists and
                         overwrite=False.
        FileNotFoundError: If raw Parquet files are not found.
    """
    _project_root = _get_project_root()

    if raw_data_dir is None:
        raw_data_dir = os.path.join(_project_root, "data", "defactify", "data")
    if output_base_dir is None:
        output_base_dir = os.path.join(_project_root, "data", "processed")

    out_dir = os.path.join(output_base_dir, "train")

    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(
            f"Processed training dataset already exists at: {out_dir}\n"
            f"Pass overwrite=True to regenerate, or use the existing dataset."
        )
    if os.path.exists(out_dir) and overwrite:
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(raw_data_dir, "train-*.parquet")))
    if not raw_files:
        raise FileNotFoundError(
            f"No train-*.parquet files found in: {raw_data_dir}"
        )

    # --- Process raw files → accumulated rows → flush to output Parquet ---
    image_buffer: List[bytes] = []
    label_a_buffer: List[int] = []
    generator_counts: Dict[str, int] = {}  # tracked incrementally for audit manifest

    total_processed = 0
    errors = 0
    file_index = 0
    output_paths: List[str] = []

    start_time = time.time()

    for raw_path in raw_files:
        pf = pq.ParquetFile(raw_path)

        for rg_idx in range(pf.metadata.num_row_groups):
            batch = pf.read_row_group(rg_idx, columns=["Image", "Label_A", "Label_B"])
            images  = batch.column("Image").to_pylist()
            labels_a = batch.column("Label_A").to_pylist()
            labels_b = batch.column("Label_B").to_pylist()

            for img_struct, la, lb in zip(images, labels_a, labels_b):
                if max_samples is not None and total_processed >= max_samples:
                    break

                try:
                    pil_img = preprocess(img_struct["bytes"])
                    # Convert PIL RGB Image → flat uint8 bytes (256×256×3, row-major)
                    rgb_bytes = pil_img.tobytes()  # 256*256*3 = 196,608 bytes
                except Exception as e:
                    errors += 1
                    # Record but do not silently skip — reported in manifest.
                    continue

                image_buffer.append(rgb_bytes)
                label_a_buffer.append(int(la))
                gen_key = str(int(lb))
                generator_counts[gen_key] = generator_counts.get(gen_key, 0) + 1
                total_processed += 1

                # Flush when buffer reaches _ROWS_PER_FILE
                if len(image_buffer) >= _ROWS_PER_FILE:
                    out_path = _flush_buffer(
                        image_buffer, label_a_buffer, out_dir, file_index
                    )
                    output_paths.append(out_path)
                    image_buffer = []
                    label_a_buffer = []
                    file_index += 1

            if max_samples is not None and total_processed >= max_samples:
                break
        if max_samples is not None and total_processed >= max_samples:
            break

    # Flush remaining
    if image_buffer:
        out_path = _flush_buffer(image_buffer, label_a_buffer, out_dir, file_index)
        output_paths.append(out_path)
        file_index += 1

    elapsed = time.time() - start_time

    # Count per-class from output files directly
    label_a_all = _read_labels_from_outputs(output_paths)

    n_real = sum(1 for x in label_a_all if x == 0)
    n_ai   = sum(1 for x in label_a_all if x == 1)

    manifest = {
        "dataset": "Defactify_Image_Dataset",
        "split": "train",
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_config": {
            "canonical_size": CANONICAL_SIZE,
            "interpolation": "INTER_AREA",
            "crop": "largest_centered_square",
            "normalization": "none_in_block1",
        },
        "output_format": {
            "image_column": "image_rgb",
            "image_encoding": "uint8_rgb_flat",
            "image_shape": [CANONICAL_SIZE, CANONICAL_SIZE, 3],
            "label_column": "label_a",
            "note": (
                "label_b (generator id) is in this manifest for audit only. "
                "It is NOT in the processed Parquet image files."
            ),
        },
        "counts": {
            "total": total_processed,
            "real_label_a_0": n_real,
            "ai_label_a_1":   n_ai,
            "errors_skipped":  errors,
        },
        "audit_generator_counts_label_b": generator_counts,
        "output_files": [os.path.basename(p) for p in output_paths],
        "elapsed_seconds": round(elapsed, 2),
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return out_dir


def _flush_buffer(
    image_buffer: List[bytes],
    label_a_buffer: List[int],
    out_dir: str,
    file_index: int,
) -> str:
    """Write accumulated rows to a single output Parquet file."""
    out_path = os.path.join(out_dir, f"part-{file_index:05d}.parquet")
    table = pa.table(
        {
            "image_rgb": pa.array(image_buffer, type=pa.binary()),
            "label_a":   pa.array(label_a_buffer, type=pa.int32()),
        },
        schema=_PROCESSED_SCHEMA,
    )
    pq.write_table(
        table,
        out_path,
        row_group_size=_ROW_GROUP_SIZE,
        compression="snappy",
    )
    return out_path


def _read_labels_from_outputs(output_paths: List[str]) -> List[int]:
    """Read label_a from all output files."""
    label_a_all: List[int] = []
    for p in output_paths:
        t = pq.read_table(p, columns=["label_a"])
        label_a_all.extend(t.column("label_a").to_pylist())
    return label_a_all


def _get_project_root() -> str:
    """Return the project root directory (two levels above src/data/)."""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


# ---------------------------------------------------------------------------
# Convenience: open a materialized dataset directory
# ---------------------------------------------------------------------------


def get_processed_train_dir(
    output_base_dir: Optional[str] = None,
) -> str:
    """Return the path to the canonical materialized processed training directory.

    Raises FileNotFoundError if the directory does not exist.
    Call materialize_training_dataset() first.
    """
    if output_base_dir is None:
        output_base_dir = os.path.join(_get_project_root(), "data", "processed")
    out_dir = os.path.join(output_base_dir, "train")
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"Processed training dataset not found at: {out_dir}\n"
            f"Run materialize_training_dataset() first."
        )
    return out_dir
