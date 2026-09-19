"""
src/data/tests/test_loader.py
Block 1 — DefactifyDataset Loader Tests (Tests A–G)

Tests the actual loader against the real Defactify Parquet files.
Reads only a SMALL number of images from the real data (Tests D/E/G).
Length checks (Tests A/B/C) use the lightweight index built at init time
and do NOT decode any images.

Run via:
    cd <project-root>
    .venv/bin/python src/data/tests/run_tests.py
"""

from __future__ import annotations

import os
import sys

import torch

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.loader import DefactifyDataset

# ---------------------------------------------------------------------------
# Constants — expected split sizes (FACT from Defactify metadata)
# ---------------------------------------------------------------------------

EXPECTED_TRAIN_LEN      = 42_000
EXPECTED_VALIDATION_LEN =  9_000
EXPECTED_TEST_LEN       = 45_000

# How many real images to load in the "sample" tests.
# Keep small: we must not decode 96,000 images just to run the test suite.
SAMPLE_N = 5


# ---------------------------------------------------------------------------
# Test A — Train split length
# ---------------------------------------------------------------------------


def test_loader_A_train_length():
    """
    FACT: The train split must contain exactly 42,000 samples.
    Length is determined from the lightweight index built at init
    (label columns only — no Image bytes read).
    """
    ds = DefactifyDataset("train")
    actual = len(ds)
    assert actual == EXPECTED_TRAIN_LEN, (
        f"Train split length mismatch: expected {EXPECTED_TRAIN_LEN}, got {actual}"
    )


# ---------------------------------------------------------------------------
# Test B — Validation split length
# ---------------------------------------------------------------------------


def test_loader_B_validation_length():
    """
    FACT: The validation split must contain exactly 9,000 samples.
    """
    ds = DefactifyDataset("validation")
    actual = len(ds)
    assert actual == EXPECTED_VALIDATION_LEN, (
        f"Validation split length mismatch: expected {EXPECTED_VALIDATION_LEN}, got {actual}"
    )


# ---------------------------------------------------------------------------
# Test C — Test split length
# ---------------------------------------------------------------------------


def test_loader_C_test_length():
    """
    FACT: The test split must contain exactly 45,000 samples.
    """
    ds = DefactifyDataset("test")
    actual = len(ds)
    assert actual == EXPECTED_TEST_LEN, (
        f"Test split length mismatch: expected {EXPECTED_TEST_LEN}, got {actual}"
    )


# ---------------------------------------------------------------------------
# Test D — Real Parquet sample: shape, dtype, value range, label validity
# ---------------------------------------------------------------------------


def test_loader_D_real_parquet_sample():
    """
    FACT: Loading SAMPLE_N real images from the train split must produce:
        image.shape == torch.Size([3, 256, 256])
        image.dtype == torch.float32
        image.min() >= 0.0
        image.max() <= 1.0
        label in {0, 1}

    Uses the real Parquet files — NOT synthetic PIL images.
    Decodes only SAMPLE_N images, not all 96,000.
    """
    ds = DefactifyDataset("train")
    assert len(ds) > 0, "Train dataset is empty"

    for i in range(SAMPLE_N):
        item = ds[i]

        assert len(item) == 2, (
            f"Detector mode must return (image, label_a) — got tuple of length {len(item)}"
        )
        image, label_a = item

        assert isinstance(image, torch.Tensor), \
            f"image must be a torch.Tensor, got {type(image)}"
        assert image.shape == torch.Size([3, 256, 256]), \
            f"image.shape must be [3, 256, 256], got {image.shape}"
        assert image.dtype == torch.float32, \
            f"image.dtype must be float32, got {image.dtype}"
        assert float(image.min()) >= 0.0, \
            f"image.min() = {float(image.min())} < 0.0 — pixel values out of range"
        assert float(image.max()) <= 1.0, \
            f"image.max() = {float(image.max())} > 1.0 — pixel values out of range"
        assert label_a in (0, 1), \
            f"label_a must be 0 or 1, got {label_a}"


# ---------------------------------------------------------------------------
# Test E — Both classes loadable from real data
# ---------------------------------------------------------------------------


def test_loader_E_both_classes():
    """
    FACT: The train split contains both Label_A=0 (real) and Label_A=1 (AI).
    Find one sample of each class and verify they load without error and
    produce valid tensors.

    Scans index until one of each class is found (no image decoding during scan —
    the index stores pre-loaded label_a values).
    Then loads only those two samples.
    """
    ds = DefactifyDataset("train")

    idx_real = None
    idx_ai   = None
    for i, entry in enumerate(ds._index):
        if entry.label_a == 0 and idx_real is None:
            idx_real = i
        if entry.label_a == 1 and idx_ai is None:
            idx_ai = i
        if idx_real is not None and idx_ai is not None:
            break

    assert idx_real is not None, "No Label_A=0 (real) sample found in train split"
    assert idx_ai   is not None, "No Label_A=1 (AI) sample found in train split"

    for idx, expected_label, label_name in [
        (idx_real, 0, "real (Label_A=0)"),
        (idx_ai,   1, "AI (Label_A=1)"),
    ]:
        image, label_a = ds[idx]

        assert label_a == expected_label, \
            f"Expected label {expected_label} for {label_name}, got {label_a}"
        assert image.shape == torch.Size([3, 256, 256]), \
            f"image.shape must be [3,256,256] for {label_name}, got {image.shape}"
        assert image.dtype == torch.float32, \
            f"image.dtype must be float32 for {label_name}"
        assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0, \
            f"Pixel range [0,1] violated for {label_name}"


# ---------------------------------------------------------------------------
# Test F — Detector mode: only (image, label_a) returned; no metadata
# ---------------------------------------------------------------------------


def test_loader_F_detector_mode():
    """
    LEAKAGE CONTROL (DEC-003, DEC-004):
        With include_metadata=False (the default), __getitem__ must return
        EXACTLY a 2-tuple (image_tensor, label_a).
        Caption, Label_B, and path must NOT be present in the return value.
    """
    ds = DefactifyDataset("train", include_metadata=False)
    item = ds[0]

    assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
    assert len(item) == 2, (
        f"Detector mode must return exactly (image, label_a) — "
        f"got {len(item)}-tuple. Possible metadata leak."
    )

    image, label_a = item

    assert isinstance(image, torch.Tensor), f"image must be Tensor, got {type(image)}"
    assert isinstance(label_a, int), f"label_a must be int, got {type(label_a)}"

    # Confirm no metadata sneaked into the tuple as extra elements
    # (already covered by len check above, but be explicit)
    assert label_a in (0, 1), f"label_a must be 0 or 1, got {label_a}"


# ---------------------------------------------------------------------------
# Test G — Explicit audit mode: label_b and caption available, isolated
# ---------------------------------------------------------------------------


def test_loader_G_audit_mode():
    """
    LEAKAGE CONTROL (DEC-003, DEC-004):
        With include_metadata=True, __getitem__ returns (image, label_a, label_b, caption).
        label_b and caption must be ABSENT from the default (detector) mode.
        They must only be accessible when explicitly requested for audit.
    """
    # Audit mode: all four fields present
    ds_audit = DefactifyDataset("train", include_metadata=True)
    item_audit = ds_audit[0]

    assert len(item_audit) == 4, (
        f"Audit mode must return (image, label_a, label_b, caption) — "
        f"got {len(item_audit)}-tuple"
    )
    image_a, label_a_a, label_b, caption = item_audit

    assert isinstance(image_a, torch.Tensor), f"image must be Tensor in audit mode"
    assert label_a_a in (0, 1),              f"label_a must be 0 or 1, got {label_a_a}"
    assert isinstance(label_b, int),          f"label_b must be int, got {type(label_b)}"
    assert 0 <= label_b <= 5,                 f"label_b (generator id) must be 0–5, got {label_b}"
    assert isinstance(caption, str),          f"caption must be str, got {type(caption)}"
    assert len(caption) > 0,                  "caption must not be empty"

    # Detector mode on same index: must NOT include label_b or caption
    ds_det = DefactifyDataset("train", include_metadata=False)
    item_det = ds_det[0]

    assert len(item_det) == 2, (
        f"Detector mode must return 2-tuple, not {len(item_det)}-tuple. "
        f"Metadata must not leak into detector output."
    )
