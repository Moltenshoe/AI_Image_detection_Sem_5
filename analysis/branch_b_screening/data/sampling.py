"""Deterministic 100-Sample Stratified Selection for Branch B Screening.

Extracts exactly 100 representative samples from the materialized Block 1 training
dataset across Real and all 5 AI generators, exporting them to an immutable static
manifest with precise Parquet part and row-group pointers.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "sample_idx",
    "image_id",
    "split",
    "label_a",
    "label_b",
    "generator_name",
    "parquet_part",
    "row_group",
    "row_in_rg",
]

STRATIFIED_TARGETS = {
    0: 20,  # Real
    1: 16,  # SD2.1
    2: 16,  # SDXL
    3: 16,  # SD3
    4: 16,  # DALL-E3
    5: 16,  # Midjourney v6
}

TOTAL_SCREENING_SAMPLES = 100


def get_default_manifest_path() -> str:
    """Return the absolute path to the static screening manifest."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "manifests", "screening_samples.csv")


def compute_file_checksum(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_screening_manifest(manifest_csv: str) -> Dict[str, Any]:
    """Verify schema, class counts, and row counts of an existing manifest.

    Raises:
        ValueError: If manifest is corrupt, incomplete, or fails schema validation.
    """
    if not os.path.isfile(manifest_csv):
        raise FileNotFoundError(f"Screening manifest not found: {manifest_csv}")

    df = pd.read_csv(manifest_csv)

    # 1. Schema check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {missing_cols}")

    # 2. Row count check
    if len(df) != TOTAL_SCREENING_SAMPLES:
        raise ValueError(
            f"Expected exactly {TOTAL_SCREENING_SAMPLES} samples, found {len(df)}"
        )

    # 3. Class stratification check
    counts = df["label_b"].value_counts().to_dict()
    for gen_id, expected_count in STRATIFIED_TARGETS.items():
        actual_count = counts.get(gen_id, 0)
        if actual_count != expected_count:
            raise ValueError(
                f"Class {gen_id} count mismatch: expected {expected_count}, got {actual_count}"
            )

    # 4. Unique IDs check
    if df["sample_idx"].nunique() != TOTAL_SCREENING_SAMPLES:
        raise ValueError("Duplicate sample_idx values found in manifest.")
    if df["image_id"].nunique() != TOTAL_SCREENING_SAMPLES:
        raise ValueError("Duplicate image_id values found in manifest.")

    checksum = compute_file_checksum(manifest_csv)
    return {
        "status": "VALID",
        "sample_count": len(df),
        "class_counts": counts,
        "checksum_sha256": checksum,
    }


def select_screening_samples(
    source_manifest_csv: str,
    output_csv: Optional[str] = None,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Select stratified samples and export to immutable static CSV.

    Args:
        source_manifest_csv: Path to Block 1 dataset_manifest.csv.
        output_csv: Target output CSV path (default: manifests/screening_samples.csv).
        seed: Random seed for stratified sampling (default: 42).
        overwrite: If False and output file exists, verifies existing without overwriting.

    Returns:
        pd.DataFrame: Validated screening manifest dataframe.
    """
    if output_csv is None:
        output_csv = get_default_manifest_path()

    if os.path.isfile(output_csv) and not overwrite:
        # Existing manifest: verify and load
        verify_screening_manifest(output_csv)
        return pd.read_csv(output_csv)

    if not os.path.isfile(source_manifest_csv):
        raise FileNotFoundError(f"Source manifest not found: {source_manifest_csv}")

    df = pd.read_csv(source_manifest_csv)

    selected_dfs = []
    for gen_id, count in STRATIFIED_TARGETS.items():
        gen_df = df[df["label_b"] == gen_id]
        if len(gen_df) < count:
            raise ValueError(
                f"Source manifest has only {len(gen_df)} samples for class {gen_id}, "
                f"needed {count}"
            )
        sampled = gen_df.sample(n=count, random_state=seed)
        selected_dfs.append(sampled)

    pilot_df = pd.concat(selected_dfs).sort_values("sample_idx").reset_index(drop=True)

    # Add Parquet part and row group mapping columns:
    # Processed Parquet parts have 1,000 rows each, with two 500-row row groups
    pilot_df["parquet_part"] = pilot_df["sample_idx"].apply(
        lambda idx: f"part-{idx // 1000:05d}.parquet"
    )
    pilot_df["row_group"] = pilot_df["sample_idx"].apply(
        lambda idx: (idx % 1000) // 500
    )
    pilot_df["row_in_rg"] = pilot_df["sample_idx"].apply(
        lambda idx: (idx % 1000) % 500
    )

    # Filter and reorder columns
    out_df = pilot_df[REQUIRED_COLUMNS]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    verify_screening_manifest(output_csv)
    return out_df
