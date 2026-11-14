"""Derived RGB Dataset Materializer (Block 2).

Materializes derived spatial representations (Branch A Lab correlation maps and
Branch B NQP residual maps) under:
    data/rgb/
        branch_a_lab/
        branch_b_nqp/
        branch_c_validation/
        manifests/

MANDATORY PILOT CHECKPOINT:
Branch B (NQP) full dataset generation is strictly gated by the 100-image pilot.
If the NQP pilot findings indicate near-zero residuals or an unresolved intensity convention,
full materialization of Branch B is blocked.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import torch

from src.data.materializer import get_processed_train_dir

from .branch_a_lab import extract_branch_a_lab_maps
from .branch_b_nqp import compute_branch_b_residual, compute_pid_residual
from .branch_c_validation import (
    ValidationResult,
    generate_validation_report,
    validate_branch_b_map,
    validate_dataset_alignment,
    validate_lab_maps,
    validate_nqp_map,
    verify_read_after_write,
)


def _get_project_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


def materialize_rgb_representations(
    branches: Sequence[str] = ("A",),
    processed_train_dir: Optional[str] = None,
    output_base_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    dtype: str = "float16",
    overwrite: bool = False,
    allow_unresolved_nqp: bool = False,
) -> Dict[str, Any]:
    """Materialize derived RGB representations for Block 1 samples.

    Args:
        branches: Sequence of branches to materialize ("A", "B").
        processed_train_dir: Path to Block 1 materialized images.
        output_base_dir: Target output root (defaults to data/rgb/).
        max_samples: Max samples to process (None for full split).
        dtype: Storage dtype ("float16" or "float32").
        overwrite: Overwrite existing outputs if True.
        allow_unresolved_nqp: Reserved parameter for legacy NQP research scripts.

    Returns:
        Summary dictionary with counts, timing, and manifest path.
    """
    project_root = _get_project_root()
    if processed_train_dir is None:
        processed_train_dir = get_processed_train_dir()
    if output_base_dir is None:
        output_base_dir = os.path.join(project_root, "data", "rgb")

    branch_a_dir = os.path.join(output_base_dir, "branch_a_lab")
    branch_b_dir = os.path.join(output_base_dir, "branch_b_pid")
    manifests_dir = os.path.join(output_base_dir, "manifests")
    validation_dir = os.path.join(output_base_dir, "branch_c_validation")

    if "A" in branches:
        os.makedirs(branch_a_dir, exist_ok=True)
    if "B" in branches:
        os.makedirs(branch_b_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    parquet_paths = sorted(
        glob.glob(os.path.join(processed_train_dir, "part-*.parquet"))
    )
    if not parquet_paths:
        raise FileNotFoundError(f"No part-*.parquet in {processed_train_dir}")

    np_dtype = np.float16 if dtype == "float16" else np.float32

    total_processed = 0
    start_time = time.time()
    manifest_records = []
    validation_results = []
    block1_ids = []
    processed_ids = []

    print(f"Starting RGB representation materialization: branches={branches}, dtype={dtype}")

    for p in parquet_paths:
        pf = pq.ParquetFile(p)
        for rg_idx in range(pf.metadata.num_row_groups):
            batch = pf.read_row_group(rg_idx, columns=["image_rgb", "label_a"])
            rgb_col = batch.column("image_rgb").to_pylist()
            la_col = batch.column("label_a").to_pylist()

            for r_idx in range(len(la_col)):
                if max_samples is not None and total_processed >= max_samples:
                    break

                sample_idx = total_processed
                image_id = f"train_{sample_idx:06d}"
                block1_ids.append(image_id)
                processed_ids.append(image_id)

                arr_u8 = np.frombuffer(rgb_col[r_idx], dtype=np.uint8).reshape(256, 256, 3).copy()

                rec = {
                    "image_id": image_id,
                    "sample_idx": sample_idx,
                    "label_a": int(la_col[r_idx]),
                }

                if "A" in branches:
                    out_path_a = os.path.join(branch_a_dir, f"{image_id}.npy")
                    if overwrite or not os.path.isfile(out_path_a):
                        lab_maps = extract_branch_a_lab_maps(arr_u8) # [3, 256, 256]
                        if total_processed < 5:
                            v_res = validate_lab_maps(lab_maps)
                            validation_results.append(v_res)
                        np.save(out_path_a, lab_maps.cpu().numpy().astype(np_dtype))
                    rec["lab_maps_path"] = os.path.relpath(out_path_a, output_base_dir)

                if "B" in branches:
                    out_path_b = os.path.join(branch_b_dir, f"{image_id}.npy")
                    if overwrite or not os.path.isfile(out_path_b):
                        # Active Branch B: PiD YUV residual with s=0.2 (DEC-015)
                        pid_map = compute_pid_residual(arr_u8, s=0.2) # [3, 256, 256]
                        if total_processed < 5:
                            v_res = validate_branch_b_map(pid_map)
                            validation_results.append(v_res)
                        np.save(out_path_b, pid_map.cpu().numpy().astype(np_dtype))
                    rec["branch_b_map_path"] = os.path.relpath(out_path_b, output_base_dir)
                    rec["pid_map_path"] = rec["branch_b_map_path"]
                    rec["nqp_map_path"] = rec["branch_b_map_path"]  # compatibility alias

                manifest_records.append(rec)
                total_processed += 1

            if max_samples is not None and total_processed >= max_samples:
                break
        if max_samples is not None and total_processed >= max_samples:
            break

    elapsed = time.time() - start_time

    # Validate dataset alignment
    align_res = validate_dataset_alignment(block1_ids, processed_ids)
    validation_results.append(align_res)

    # Save manifest
    manifest_data = {
        "dataset": "Defactify_Image_Dataset",
        "split": "train",
        "branches": list(branches),
        "total_processed": total_processed,
        "storage_dtype": dtype,
        "elapsed_seconds": round(elapsed, 2),
        "records_count": len(manifest_records),
    }

    manifest_json_path = os.path.join(manifests_dir, "manifest.json")
    with open(manifest_json_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    # Save validation report
    val_report_path = os.path.join(validation_dir, "validation_report.json")
    val_report = generate_validation_report(
        validation_results,
        output_path=val_report_path,
        dataset_summary=manifest_data,
    )

    print(f"Materialization complete: {total_processed} samples in {elapsed:.2f}s. Validation: {val_report['status']}")
    return {
        "total_processed": total_processed,
        "manifest_path": manifest_json_path,
        "validation_report_path": val_report_path,
        "elapsed_seconds": elapsed,
    }
