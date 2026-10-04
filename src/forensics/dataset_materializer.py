"""Persistent Forensic Dataset Materialization Engine (Block 2).

Reads preprocessed images from Block 1, extracts the complete canonical 111-feature
forensic bank across all 5 branches (A, B, C_LBP, D_MFR, E), pairs each sample with its
full source metadata and identity keys, and materializes a persistent dataset under:

    data/forensic_dataset/

OUTPUT FORMAT & SCHEMA SEPARATION:
    Metadata columns (11 columns):
        image_id         : string (stable unique identifier e.g. "train_000000")
        sample_idx       : int64  (global sequential index)
        split            : string ("train", "validation", "test")
        source_path      : string (original source path from raw Defactify Image struct)
        source_file      : string (raw Parquet file name)
        source_row_group : int32  (row group index in raw Parquet file)
        source_row       : int32  (row index in row group)
        label_a          : int32  (0=real, 1=AI — detector supervisory target ONLY)
        label_b          : int32  (0=Real, 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL-E3, 5=Midjourney — audit metadata ONLY)
        generator_name   : string (human-readable generator name)
        caption          : string (raw text caption — audit metadata ONLY)

    Forensic feature columns (111 float32 columns):
        Exact 111 canonical feature names with embedded field metadata (branch, sub_branch, domain).

LEAKAGE CONTROLS:
    - Metadata columns are strictly isolated from the 111-feature matrix.
    - Generator identity and source metadata are NEVER model inputs.
    - All 111 features are preserved in the canonical dataset without premature reduction.
"""

from __future__ import annotations

import csv
import glob
import io
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from src.forensics.branch_f.registry import (
    CANONICAL_BRANCHES,
    EXPECTED_BRANCH_COUNTS,
    EXPECTED_FEATURE_COUNT,
    FeatureMetadata,
    FeatureRegistry,
)
from src.forensics.pipeline import ForensicPipeline

GENERATOR_NAMES_MAP: Dict[int, str] = {
    0: "Real",
    1: "SD2.1",
    2: "SDXL",
    3: "SD3",
    4: "DALL-E3",
    5: "Midjourney",
}

SUBBRANCH_COUNTS: Dict[str, int] = {
    "A1_FFT": 4,
    "A2_SYNTH": 30,
    "B_HAAR": 30,
    "C_LBP": 16,
    "D_MFR": 5,
    "E1_DCT": 10,
    "E2_RESP": 8,
    "E3_PHASE": 4,
    "E4_GRID": 4,
}


def _get_project_root() -> str:
    """Resolve project root directory."""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


def build_pyarrow_schema(registry: FeatureRegistry) -> pa.Schema:
    """Build strongly-typed PyArrow schema with explicit metadata and feature column definitions."""
    fields = [
        pa.field("image_id", pa.string(), metadata={b"description": b"Stable unique image identity key"}),
        pa.field("sample_idx", pa.int64(), metadata={b"description": b"Global sequential index"}),
        pa.field("split", pa.string(), metadata={b"description": b"Dataset split (train/validation/test)"}),
        pa.field("source_path", pa.string(), metadata={b"description": b"Original source image path"}),
        pa.field("source_file", pa.string(), metadata={b"description": b"Raw source Parquet file"}),
        pa.field("source_row_group", pa.int32(), metadata={b"description": b"Source row group index"}),
        pa.field("source_row", pa.int32(), metadata={b"description": b"Source row index in row group"}),
        pa.field("label_a", pa.int32(), metadata={b"description": b"0=real, 1=AI (supervisory target ONLY)"}),
        pa.field("label_b", pa.int32(), metadata={b"description": b"0..5 generator id (evaluation metadata ONLY)"}),
        pa.field("generator_name", pa.string(), metadata={b"description": b"Generator name (evaluation metadata ONLY)"}),
        pa.field("caption", pa.string(), metadata={b"description": b"Source caption text (audit metadata ONLY)"}),
    ]

    for fname in registry.feature_names:
        meta = registry.get_by_name(fname)
        field_meta = {
            b"branch": meta.branch.encode("utf-8"),
            b"sub_branch": meta.sub_branch.encode("utf-8"),
            b"domain": meta.domain.encode("utf-8"),
            b"description": meta.description.encode("utf-8"),
            b"data_type": meta.data_type.encode("utf-8"),
            b"sensitivity": meta.sensitivity.encode("utf-8"),
        }
        fields.append(pa.field(fname, pa.float32(), metadata=field_meta))

    return pa.schema(fields)


def load_raw_source_metadata(
    raw_data_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read full source metadata from raw Defactify Parquet files in deterministic order."""
    pattern = os.path.join(raw_data_dir, f"{split}-*.parquet")
    raw_files = sorted(glob.glob(pattern))
    if not raw_files:
        raise FileNotFoundError(f"No raw Parquet files found matching '{pattern}'.")

    metadata_records: List[Dict[str, Any]] = []
    total_count = 0

    for f_idx, rf in enumerate(raw_files):
        pf = pq.ParquetFile(rf)
        fname = os.path.basename(rf)
        for rg_idx in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg_idx, columns=["Image", "Label_A", "Label_B", "Caption"])
            img_col = tbl["Image"].to_pylist()
            las = tbl["Label_A"].to_pylist()
            lbs = tbl["Label_B"].to_pylist()
            caps = tbl["Caption"].to_pylist()

            for r_idx in range(len(las)):
                if max_samples is not None and total_count >= max_samples:
                    break
                img_val = img_col[r_idx]
                source_path = img_val.get("path") if isinstance(img_val, dict) else f"image_{total_count}.jpg"
                lb = int(lbs[r_idx])
                gen_name = GENERATOR_NAMES_MAP.get(lb, f"Generator_{lb}")

                metadata_records.append({
                    "image_id": f"{split}_{total_count:06d}",
                    "sample_idx": total_count,
                    "split": split,
                    "source_path": str(source_path) if source_path is not None else "",
                    "source_file": fname,
                    "source_row_group": rg_idx,
                    "source_row": r_idx,
                    "label_a": int(las[r_idx]),
                    "label_b": lb,
                    "generator_name": gen_name,
                    "caption": str(caps[r_idx]) if caps[r_idx] is not None else "",
                })
                total_count += 1

            if max_samples is not None and total_count >= max_samples:
                break
        if max_samples is not None and total_count >= max_samples:
            break

    return metadata_records


def materialize_forensic_dataset(
    processed_train_dir: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 500,
    overwrite: bool = False,
    branch_f_analysis_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Materialize the canonical 111-feature persistent forensic dataset.

    Args:
        processed_train_dir: Directory containing Block 1 materialized train Parquet files.
        raw_data_dir: Directory containing raw Defactify Parquet files (for metadata extraction).
        output_dir: Target output directory (defaults to 'data/forensic_dataset').
        max_samples: Maximum number of samples to extract (None = full dataset).
        batch_size: Number of samples per Parquet row group write batch.
        overwrite: Overwrite existing dataset if True.
        branch_f_analysis_dir: Directory with existing Branch F artifacts for manifest copying.

    Returns:
        Dictionary with materialization summary, paths, and record counts.
    """
    project_root = _get_project_root()
    if processed_train_dir is None:
        processed_train_dir = os.path.join(project_root, "data", "processed", "train")
    if raw_data_dir is None:
        raw_data_dir = os.path.join(project_root, "data", "defactify", "data")
    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "forensic_dataset")
    if branch_f_analysis_dir is None:
        branch_f_analysis_dir = os.path.join(project_root, "analysis", "forensic_feature_analysis")

    features_parquet_path = os.path.join(output_dir, "features.parquet")
    if os.path.exists(features_parquet_path) and not overwrite:
        raise FileExistsError(
            f"Forensic dataset already exists at '{features_parquet_path}'. "
            f"Pass overwrite=True to re-materialize."
        )

    os.makedirs(output_dir, exist_ok=True)
    selected_dir = os.path.join(output_dir, "selected")
    os.makedirs(selected_dir, exist_ok=True)

    print("=" * 70)
    print("Block 2 — Persistent Forensic Dataset Materialization")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Processed train source: {processed_train_dir}")
    print(f"Raw metadata source: {raw_data_dir}")

    # 1. Initialize Feature Registry & Pipeline
    registry = FeatureRegistry()
    feature_names = registry.feature_names
    num_features = len(feature_names)
    assert num_features == EXPECTED_FEATURE_COUNT, f"Expected {EXPECTED_FEATURE_COUNT} features, got {num_features}"
    schema = build_pyarrow_schema(registry)

    # 2. Export Feature Registry Artifacts
    reg_csv_path = os.path.join(output_dir, "feature_registry.csv")
    reg_json_path = os.path.join(output_dir, "feature_registry.json")
    registry.export_csv(reg_csv_path)
    registry.export_json(reg_json_path)
    print(f"Exported Feature Registry ({num_features} features) to CSV & JSON.")

    # 3. Load Raw Source Metadata for Alignment
    print("Loading source metadata from raw Parquet files...")
    metadata_list = load_raw_source_metadata(raw_data_dir, split="train", max_samples=max_samples)
    total_samples = len(metadata_list)
    print(f"Loaded metadata for {total_samples} samples.")

    # 4. Read Processed Images & Extract Features in Chunks
    proc_files = sorted(
        [os.path.join(processed_train_dir, f) for f in os.listdir(processed_train_dir) if f.endswith(".parquet")]
    )
    if not proc_files:
        raise FileNotFoundError(f"No processed Parquet files found in '{processed_train_dir}'.")

    pipeline = ForensicPipeline(branches=CANONICAL_BRANCHES)

    writer = pq.ParquetWriter(features_parquet_path, schema=schema, compression="snappy")

    t0 = time.perf_counter()
    processed_count = 0
    current_meta_idx = 0

    batch_meta: Dict[str, List[Any]] = {
        "image_id": [],
        "sample_idx": [],
        "split": [],
        "source_path": [],
        "source_file": [],
        "source_row_group": [],
        "source_row": [],
        "label_a": [],
        "label_b": [],
        "generator_name": [],
        "caption": [],
    }
    batch_features: Dict[str, List[float]] = {fname: [] for fname in feature_names}

    manifest_rows: List[Dict[str, Any]] = []

    for pf_path in proc_files:
        table = pq.read_table(pf_path, columns=["image_rgb", "label_a"])
        img_bytes_list = table["image_rgb"].to_pylist()
        label_a_list = table["label_a"].to_pylist()

        for raw_bytes, la in zip(img_bytes_list, label_a_list):
            if max_samples is not None and processed_count >= max_samples:
                break

            meta = metadata_list[current_meta_idx]
            current_meta_idx += 1

            # Validate label alignment between processed image and raw metadata
            assert meta["label_a"] == int(la), (
                f"Label mismatch at index {processed_count}: processed={la}, raw={meta['label_a']}"
            )

            # Reconstruct canonical [3, 256, 256] float32 tensor
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((256, 256, 3))
            tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).to(dtype=torch.float32) / 255.0

            # Extract full 111 forensic features
            feats = pipeline.extract(tensor)

            # Append to batch
            for k in batch_meta:
                batch_meta[k].append(meta[k])

            for fname in feature_names:
                val = float(feats[fname])
                if not np.isfinite(val):
                    raise ValueError(f"Non-finite value {val} encountered for feature '{fname}' at sample {processed_count}!")
                batch_features[fname].append(val)

            manifest_rows.append({
                "image_id": meta["image_id"],
                "sample_idx": meta["sample_idx"],
                "split": meta["split"],
                "source_path": meta["source_path"],
                "label_a": meta["label_a"],
                "label_b": meta["label_b"],
                "generator_name": meta["generator_name"],
            })

            processed_count += 1

            # Flush batch
            if len(batch_meta["image_id"]) >= batch_size:
                data_dict = {}
                data_dict.update(batch_meta)
                data_dict.update(batch_features)
                batch_table = pa.Table.from_pydict(data_dict, schema=schema)
                writer.write_table(batch_table)

                # Reset batch buffers
                for k in batch_meta:
                    batch_meta[k] = []
                for fname in feature_names:
                    batch_features[fname] = []

                elapsed = time.perf_counter() - t0
                print(f"  Processed {processed_count}/{total_samples} samples ({elapsed:.1f}s, {elapsed/processed_count*1000:.1f} ms/img)")

        if max_samples is not None and processed_count >= max_samples:
            break

    # Flush remaining records
    if len(batch_meta["image_id"]) > 0:
        data_dict = {}
        data_dict.update(batch_meta)
        data_dict.update(batch_features)
        batch_table = pa.Table.from_pydict(data_dict, schema=schema)
        writer.write_table(batch_table)

    writer.close()
    elapsed_total = time.perf_counter() - t0
    print(f"Materialized {processed_count} rows with 111 features into {features_parquet_path} in {elapsed_total:.2f}s.")

    # 5. Export Dataset Manifest CSV
    manifest_csv_path = os.path.join(output_dir, "dataset_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_csv_path, index=False)
    print(f"Exported dataset manifest to {manifest_csv_path}.")

    # 6. Export / Link Branch F Selected-Feature Manifests & Views
    budget_views_created = []
    budgets = [111, 64, 32, 16, 8]

    # Check for existing Branch F selection artifacts or derive from mRMR ranking
    mrmr_ranks_csv = os.path.join(branch_f_analysis_dir, "mrmr_rankings.csv")
    if os.path.exists(mrmr_ranks_csv):
        df_mrmr = pd.read_csv(mrmr_ranks_csv)
        ordered_features = list(df_mrmr["name"])
    else:
        ordered_features = list(feature_names)

    for b in budgets:
        subset_names = ordered_features[:b]
        sel_json_path = os.path.join(selected_dir, f"selected_features_{b}.json")
        with open(sel_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "budget": b,
                "feature_count": len(subset_names),
                "features": subset_names,
            }, f, indent=2)

        # Also materialize the selected view Parquet containing metadata + subset features
        selected_parquet_path = os.path.join(selected_dir, f"features_{b}.parquet")
        selected_cols = list(batch_meta.keys()) + subset_names
        sub_table = pq.read_table(features_parquet_path, columns=selected_cols)
        pq.write_table(sub_table, selected_parquet_path, compression="snappy")
        budget_views_created.append(selected_parquet_path)

    print(f"Generated {len(budgets)} selected budget views (111, 64, 32, 16, 8) under {selected_dir}.")

    # 7. Export Dataset Metadata JSON
    generator_dist = {}
    for r in manifest_rows:
        g = r["generator_name"]
        generator_dist[g] = generator_dist.get(g, 0) + 1

    real_count = sum(1 for r in manifest_rows if r["label_a"] == 0)
    ai_count = sum(1 for r in manifest_rows if r["label_a"] == 1)

    metadata_doc = {
        "dataset_name": "Defactify_Canonical_Forensic_Dataset",
        "stage": "Block 2 Final Persistent Forensic Dataset",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": processed_count,
        "class_distribution": {
            "real_label_0": real_count,
            "ai_label_1": ai_count,
        },
        "generator_distribution": generator_dist,
        "feature_count": num_features,
        "metadata_column_count": len(batch_meta),
        "total_column_count": len(schema),
        "branch_counts": EXPECTED_BRANCH_COUNTS,
        "subbranch_counts": SUBBRANCH_COUNTS,
        "files": {
            "canonical_features_parquet": os.path.relpath(features_parquet_path, project_root),
            "feature_registry_csv": os.path.relpath(reg_csv_path, project_root),
            "feature_registry_json": os.path.relpath(reg_json_path, project_root),
            "dataset_manifest_csv": os.path.relpath(manifest_csv_path, project_root),
            "selected_budgets_directory": os.path.relpath(selected_dir, project_root),
        },
        "selected_budgets": budgets,
        "elapsed_seconds": round(elapsed_total, 2),
    }

    metadata_json_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_doc, f, indent=2)
    print(f"Exported dataset metadata to {metadata_json_path}.")

    return metadata_doc
