"""
src/data/tests/test_materializer.py
Block 1 — Materialization + On-Demand Evaluation Tests

Tests the training materialization path (materializer.py + processed_loader.py)
and verifies that both paths share the same canonical preprocessing.

ALL TESTS use max_samples=50 or similar small subset — they do NOT materialize
all 42,000 training images. The smoke test is sufficient to verify correctness.
The full materialization is a separate operational step.

Tests:
    C — materialization produces expected sample count
    D — materialized images have correct shape/dtype/channel format
    E — labels in materialized data are correct (0 or 1)
    F — materialized data does not contain forbidden detector metadata
    G — materialization is deterministic (same output for same input)
    H — on-demand eval path and materialization path produce the same canonical
        representation for the same source image
    I — malformed/corrupt image bytes are handled predictably (skipped, counted)
    J — repeated generation with overwrite=True does not change the output
    K — manifest counts match actual materialized data counts
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.materializer import (
    PREPROCESSING_VERSION,
    materialize_training_dataset,
)
from src.data.preprocessing import preprocess, CANONICAL_SIZE
from src.data.processed_loader import ProcessedTrainDataset
from src.data.loader import DefactifyDataset

# Number of samples for smoke-test materialization.
# Must be small — we do NOT materialize all 42,000 images in unit tests.
SMOKE_N = 50
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "defactify", "data")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_out_dir() -> str:
    """Return a unique temp directory path for this test run."""
    return tempfile.mkdtemp(prefix="block1_test_", dir=os.path.join(_PROJECT_ROOT, "data", "processed"))


def _materialize_smoke(out_dir: str) -> str:
    """Materialize SMOKE_N training samples into out_dir."""
    return materialize_training_dataset(
        raw_data_dir=RAW_DATA_DIR,
        output_base_dir=out_dir,
        max_samples=SMOKE_N,
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# TEST C — Materialization produces expected sample count
# ---------------------------------------------------------------------------


def test_materializer_C_sample_count():
    """
    FACT: materialization of SMOKE_N samples produces exactly SMOKE_N
    processed samples (assuming no decode errors in the first SMOKE_N images).
    """
    out_base = _tmp_out_dir()
    try:
        result_dir = _materialize_smoke(out_base)

        # Read manifest
        with open(os.path.join(result_dir, "manifest.json")) as f:
            manifest = json.load(f)

        total = manifest["counts"]["total"]
        errors = manifest["counts"]["errors_skipped"]

        assert total + errors == SMOKE_N, (
            f"Expected total+errors=={SMOKE_N}, got total={total}, errors={errors}"
        )
        assert total > 0, "Materialization produced 0 samples — something is wrong"
        assert errors == 0, (
            f"Unexpected decode errors in first {SMOKE_N} images: {errors}. "
            "These images should be valid JPEG."
        )

        # Verify actual row count in Parquet files
        parquet_files = sorted(glob.glob(os.path.join(result_dir, "part-*.parquet")))
        assert len(parquet_files) > 0, "No part-*.parquet files produced"

        actual_rows = sum(
            pq.read_metadata(p).num_rows for p in parquet_files
        )
        assert actual_rows == total, (
            f"Manifest says {total} rows but Parquet files contain {actual_rows} rows"
        )
    finally:
        shutil.rmtree(out_base, ignore_errors=True)


# ---------------------------------------------------------------------------
# TEST D — Materialized images have correct shape/dtype/channel format
# ---------------------------------------------------------------------------


def test_materializer_D_image_shape_dtype():
    """
    FACT: every materialized image_rgb column entry must decode to
    exactly 256×256×3 uint8, readable as a float32 tensor [3,256,256] in [0,1].
    """
    out_base = _tmp_out_dir()
    try:
        result_dir = _materialize_smoke(out_base)
        ds = ProcessedTrainDataset(processed_dir=result_dir)

        assert len(ds) == SMOKE_N, f"Expected {SMOKE_N} samples, got {len(ds)}"

        for i in range(min(5, len(ds))):
            tensor, label_a = ds[i]

            assert isinstance(tensor, torch.Tensor), \
                f"Expected Tensor, got {type(tensor)}"
            assert tensor.shape == torch.Size([3, CANONICAL_SIZE, CANONICAL_SIZE]), \
                f"Wrong shape: {tensor.shape}"
            assert tensor.dtype == torch.float32, \
                f"Wrong dtype: {tensor.dtype}"
            assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0, \
                f"Pixel range [{tensor.min():.4f}, {tensor.max():.4f}] out of [0,1]"
            assert label_a in (0, 1), f"label_a must be 0 or 1, got {label_a}"
    finally:
        shutil.rmtree(out_base, ignore_errors=True)


# ---------------------------------------------------------------------------
# TEST E — Labels are correct
# ---------------------------------------------------------------------------


def test_materializer_E_labels_correct():
    """
    FACT: label_a values in the materialized dataset must match those in
    the raw Parquet for the same samples.
    """
    out_base = _tmp_out_dir()
    try:
        result_dir = _materialize_smoke(out_base)

        # Read raw label_a values for the first SMOKE_N samples
        raw_labels = []
        raw_pf = pq.ParquetFile(
            sorted(glob.glob(os.path.join(RAW_DATA_DIR, "train-*.parquet")))[0]
        )
        for rg_idx in range(raw_pf.metadata.num_row_groups):
            batch = raw_pf.read_row_group(rg_idx, columns=["Label_A"])
            raw_labels.extend(batch.column("Label_A").to_pylist())
            if len(raw_labels) >= SMOKE_N:
                break
        raw_labels = raw_labels[:SMOKE_N]

        # Read materialized label_a values
        ds = ProcessedTrainDataset(processed_dir=result_dir)
        mat_labels = [ds._index[i][3] for i in range(len(ds))]

        assert mat_labels == raw_labels, (
            f"Materialized labels do not match raw labels for first {SMOKE_N} samples.\n"
            f"First 10 raw:        {raw_labels[:10]}\n"
            f"First 10 materialized: {mat_labels[:10]}"
        )
    finally:
        shutil.rmtree(out_base, ignore_errors=True)


# ---------------------------------------------------------------------------
# TEST F — No forbidden detector metadata in materialized Parquet
# ---------------------------------------------------------------------------


def test_materializer_F_no_forbidden_metadata():
    """
    LEAKAGE CONTROL (DEC-003, DEC-004):
        The materialized Parquet files must NOT contain columns:
            label_b, caption, path, generator, source, filename
        They must contain ONLY: image_rgb, label_a.
    """
    FORBIDDEN_COLUMNS = {"label_b", "caption", "path", "generator",
                         "source", "filename", "image_path"}
    REQUIRED_COLUMNS  = {"image_rgb", "label_a"}

    out_base = _tmp_out_dir()
    try:
        result_dir = _materialize_smoke(out_base)
        parquet_files = sorted(glob.glob(os.path.join(result_dir, "part-*.parquet")))

        for pf_path in parquet_files:
            schema = pq.read_schema(pf_path)
            column_names = set(schema.names)

            forbidden_found = column_names & FORBIDDEN_COLUMNS
            assert not forbidden_found, (
                f"LEAKAGE: Forbidden columns found in {os.path.basename(pf_path)}: "
                f"{forbidden_found}. These must never appear in detector input files."
            )

            missing = REQUIRED_COLUMNS - column_names
            assert not missing, (
                f"Required columns missing from {os.path.basename(pf_path)}: {missing}"
            )
    finally:
        shutil.rmtree(out_base, ignore_errors=True)


# ---------------------------------------------------------------------------
# TEST G — Materialization is deterministic
# ---------------------------------------------------------------------------


def test_materializer_G_deterministic():
    """
    FACT: running materialization twice on the same raw data with the same
    preprocessing version must produce byte-identical output Parquet files.
    """
    out_base1 = _tmp_out_dir()
    out_base2 = _tmp_out_dir()
    try:
        dir1 = _materialize_smoke(out_base1)
        dir2 = _materialize_smoke(out_base2)

        files1 = sorted(os.listdir(dir1))
        files2 = sorted(os.listdir(dir2))

        # Same file names
        assert files1 == files2, (
            f"Determinism: different file lists.\nRun1: {files1}\nRun2: {files2}"
        )

        # Compare SHA-256 of each Parquet file (exclude manifest to avoid timestamp)
        for fname in files1:
            if not fname.endswith(".parquet"):
                continue
            p1 = os.path.join(dir1, fname)
            p2 = os.path.join(dir2, fname)
            h1 = _sha256_file(p1)
            h2 = _sha256_file(p2)
            assert h1 == h2, (
                f"Determinism FAIL: {fname} differs between two runs.\n"
                f"  Run1 SHA-256: {h1}\n  Run2 SHA-256: {h2}"
            )
    finally:
        shutil.rmtree(out_base1, ignore_errors=True)
        shutil.rmtree(out_base2, ignore_errors=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# TEST H — Eval path and materialization path produce same canonical image
# ---------------------------------------------------------------------------


def test_materializer_H_same_preprocessing_both_paths():
    """
    FACT (critical for scientific validity — no train/eval preprocessing divergence):
        For the same source image, the materialization path and the on-demand
        evaluation path (DefactifyDataset) must produce identical pixel tensors.

    Method:
        1. Read raw bytes for sample[0] from the raw Parquet directly.
        2. Run preprocess() on those bytes → PIL Image → uint8 numpy array.
        3. Load sample[0] from the materialized ProcessedTrainDataset → tensor.
        4. Compare.
    """
    out_base = _tmp_out_dir()
    try:
        result_dir = _materialize_smoke(out_base)

        # Step 1: get raw bytes for sample[0] directly from raw Parquet
        raw_pf = pq.ParquetFile(
            sorted(glob.glob(os.path.join(RAW_DATA_DIR, "train-*.parquet")))[0]
        )
        batch = raw_pf.read_row_group(0, columns=["Image"])
        raw_bytes = batch.column("Image").to_pylist()[0]["bytes"]

        # Step 2: eval path preprocessing
        pil_img = preprocess(raw_bytes)
        eval_arr = np.array(pil_img)            # [H, W, 3] uint8
        eval_tensor = torch.from_numpy(eval_arr).permute(2, 0, 1).float() / 255.0

        # Step 3: materialization path
        ds = ProcessedTrainDataset(processed_dir=result_dir)
        mat_tensor, _ = ds[0]

        # Step 4: compare
        assert eval_tensor.shape == mat_tensor.shape, (
            f"Shape mismatch: eval={eval_tensor.shape}, mat={mat_tensor.shape}"
        )
        max_diff = float((eval_tensor - mat_tensor).abs().max())
        assert max_diff == 0.0, (
            f"PREPROCESSING DIVERGENCE DETECTED: eval path and materialization path "
            f"produce different pixel values for the same source image.\n"
            f"Max absolute diff: {max_diff}\n"
            f"This would mean training and evaluation see different representations."
        )
    finally:
        shutil.rmtree(out_base, ignore_errors=True)


# ---------------------------------------------------------------------------
# TEST I — Corrupt image bytes are handled predictably (skipped, counted)
# ---------------------------------------------------------------------------


def test_materializer_I_corrupt_image_handling():
    """
    FACT: if a raw image byte array is corrupt/malformed, materializer must
    skip the sample and increment the error counter — it must not crash.
    """
    # We cannot inject corrupt bytes into the real Parquet without modifying
    # raw data (forbidden by DEC-002). Instead, we test the preprocess() function
    # directly with corrupt bytes, and verify it raises an appropriate exception
    # that the materializer's try/except would catch.
    import PIL

    corrupt_bytes = b"this is not a valid JPEG file"

    raised = False
    try:
        preprocess(corrupt_bytes)
    except Exception:
        raised = True

    assert raised, (
        "preprocess() did not raise an exception on corrupt bytes. "
        "The materializer's error handler would not catch it."
    )

    # Also verify valid bytes do not raise
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    result = preprocess(buf.getvalue())
    assert result.size == (256, 256), "Valid image failed preprocessing"


# ---------------------------------------------------------------------------
# TEST J — Repeated generation with overwrite=True does not change output
# ---------------------------------------------------------------------------


def test_materializer_J_overwrite_idempotent():
    """
    FACT: running materialize_training_dataset(overwrite=True) twice must
    produce byte-identical Parquet output. Overwrite must not introduce
    non-determinism.
    """
    out_base = _tmp_out_dir()
    try:
        dir1 = _materialize_smoke(out_base)
        hashes1 = {
            f: _sha256_file(os.path.join(dir1, f))
            for f in os.listdir(dir1)
            if f.endswith(".parquet")
        }

        # Overwrite in same location
        dir2 = materialize_training_dataset(
            raw_data_dir=RAW_DATA_DIR,
            output_base_dir=out_base,
            max_samples=SMOKE_N,
            overwrite=True,
        )

        hashes2 = {
            f: _sha256_file(os.path.join(dir2, f))
            for f in os.listdir(dir2)
            if f.endswith(".parquet")
        }

        assert hashes1 == hashes2, (
            "Overwrite-idempotency FAIL: second run with overwrite=True produced "
            "different Parquet content.\n"
            f"First run hashes:  {hashes1}\n"
            f"Second run hashes: {hashes2}"
        )
    finally:
        shutil.rmtree(out_base, ignore_errors=True)


# ---------------------------------------------------------------------------
# TEST K — Manifest counts match actual materialized data counts
# ---------------------------------------------------------------------------


def test_materializer_K_manifest_counts():
    """
    FACT: the manifest.json must accurately reflect the actual number of rows
    in the materialized Parquet files.
    """
    out_base = _tmp_out_dir()
    try:
        result_dir = _materialize_smoke(out_base)

        with open(os.path.join(result_dir, "manifest.json")) as f:
            manifest = json.load(f)

        manifest_total = manifest["counts"]["total"]
        manifest_real  = manifest["counts"]["real_label_a_0"]
        manifest_ai    = manifest["counts"]["ai_label_a_1"]

        # Count actual rows in Parquet files
        parquet_files = sorted(glob.glob(os.path.join(result_dir, "part-*.parquet")))
        actual_total  = 0
        actual_real   = 0
        actual_ai     = 0
        for pf_path in parquet_files:
            t = pq.read_table(pf_path, columns=["label_a"])
            labels = t.column("label_a").to_pylist()
            actual_total += len(labels)
            actual_real  += sum(1 for x in labels if x == 0)
            actual_ai    += sum(1 for x in labels if x == 1)

        assert actual_total == manifest_total, (
            f"Total count mismatch: manifest={manifest_total}, actual={actual_total}"
        )
        assert actual_real == manifest_real, (
            f"Real count mismatch: manifest={manifest_real}, actual={actual_real}"
        )
        assert actual_ai == manifest_ai, (
            f"AI count mismatch: manifest={manifest_ai}, actual={actual_ai}"
        )
        assert manifest_real + manifest_ai == manifest_total, (
            f"Manifest internal inconsistency: {manifest_real}+{manifest_ai} != {manifest_total}"
        )
    finally:
        shutil.rmtree(out_base, ignore_errors=True)
