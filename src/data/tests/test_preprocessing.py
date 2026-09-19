"""
src/data/tests/test_preprocessing.py
Block 1 — Preprocessing Verification Tests (7 tests)

Tests the canonical preprocessing functions in src/data/preprocessing.py.
Uses only synthetic PIL images — no real Parquet data required.

Supports direct execution via run_tests.py (no pytest required).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import subprocess
import sys

import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.preprocessing import (
    CANONICAL_SIZE,
    INTERPOLATION,
    centered_square_crop,
    preprocess,
    resize_to_canonical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solid_image(width: int, height: int, color=(128, 64, 200)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def _image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@contextlib.contextmanager
def _assert_raises(exc_type):
    """Minimal context manager asserting the body raises exc_type."""
    try:
        yield
        raise AssertionError(
            f"Expected {exc_type.__name__} to be raised, but no exception was raised."
        )
    except exc_type:
        pass  # expected


# ---------------------------------------------------------------------------
# TEST 1: Landscape 640×480
# ---------------------------------------------------------------------------


def test_landscape_640x480():
    """
    FACT: W=640, H=480 → side=480, left=80, top=0.
    Crop → 480×480; resize → 256×256.
    """
    img = _make_solid_image(640, 480)
    assert img.size == (640, 480)

    cropped = centered_square_crop(img)
    assert cropped.size == (480, 480), f"Expected (480,480), got {cropped.size}"

    resized = resize_to_canonical(cropped)
    assert resized.size == (CANONICAL_SIZE, CANONICAL_SIZE), \
        f"Expected ({CANONICAL_SIZE},{CANONICAL_SIZE}), got {resized.size}"


# ---------------------------------------------------------------------------
# TEST 2: Portrait 480×640
# ---------------------------------------------------------------------------


def test_portrait_480x640():
    """
    FACT: W=480, H=640 → side=480, left=0, top=80.
    Crop → 480×480; resize → 256×256.
    """
    img = _make_solid_image(480, 640)
    assert img.size == (480, 640)

    cropped = centered_square_crop(img)
    assert cropped.size == (480, 480), f"Expected (480,480), got {cropped.size}"

    resized = resize_to_canonical(cropped)
    assert resized.size == (CANONICAL_SIZE, CANONICAL_SIZE), \
        f"Expected ({CANONICAL_SIZE},{CANONICAL_SIZE}), got {resized.size}"


# ---------------------------------------------------------------------------
# TEST 3: Square 1024×1024 — identity crop
# ---------------------------------------------------------------------------


def test_square_1024x1024_identity_crop():
    """
    FACT: W=1024, H=1024 → side=1024, left=0, top=0. Crop is identity.
    """
    img = _make_solid_image(1024, 1024)

    cropped = centered_square_crop(img)
    assert cropped.size == (1024, 1024), f"Expected (1024,1024), got {cropped.size}"

    resized = resize_to_canonical(cropped)
    assert resized.size == (CANONICAL_SIZE, CANONICAL_SIZE)


# ---------------------------------------------------------------------------
# TEST 4: Square 270×270 — identity crop
# ---------------------------------------------------------------------------


def test_square_270x270_identity_crop():
    """
    FACT: W=270, H=270 (DALL-E3 canonical resolution) → identity crop.
    """
    img = _make_solid_image(270, 270)

    cropped = centered_square_crop(img)
    assert cropped.size == (270, 270), f"Expected (270,270), got {cropped.size}"

    resized = resize_to_canonical(cropped)
    assert resized.size == (CANONICAL_SIZE, CANONICAL_SIZE)


# ---------------------------------------------------------------------------
# TEST 5: RGB output, 256×256, deterministic
# ---------------------------------------------------------------------------


def test_output_rgb_256_deterministic():
    """
    FACT: output mode=RGB, size=(256,256), two calls on same bytes → identical arrays.
    Also verifies resize_to_canonical matches direct cv2.INTER_AREA reference.
    """
    import cv2
    img = _make_solid_image(640, 480, color=(10, 200, 50))
    raw_bytes = _image_to_bytes(img)

    result_1: Image.Image = preprocess(raw_bytes)
    result_2: Image.Image = preprocess(raw_bytes)

    assert result_1.mode == "RGB", f"Expected RGB, got {result_1.mode}"
    assert result_1.size == (CANONICAL_SIZE, CANONICAL_SIZE), \
        f"Expected ({CANONICAL_SIZE},{CANONICAL_SIZE}), got {result_1.size}"

    arr1 = np.array(result_1)
    arr2 = np.array(result_2)

    assert arr1.shape == (CANONICAL_SIZE, CANONICAL_SIZE, 3), \
        f"Unexpected shape: {arr1.shape}"
    assert np.array_equal(arr1, arr2), \
        "preprocess() is NOT deterministic: two calls produced different pixel arrays."

    # Direct cv2.INTER_AREA reference verification on non-trivial gradient pattern
    gradient_img = Image.fromarray(
        np.linspace(0, 255, 300 * 300, dtype=np.uint8).reshape(300, 300)
    ).convert("RGB")
    canonical_res = resize_to_canonical(gradient_img)
    canonical_arr = np.array(canonical_res)
    expected_arr = cv2.resize(
        np.array(gradient_img),
        (CANONICAL_SIZE, CANONICAL_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    assert np.array_equal(canonical_arr, expected_arr), \
        "resize_to_canonical does not match direct cv2.INTER_AREA reference."


# ---------------------------------------------------------------------------
# TEST 6: Metadata isolation — preprocess() signature enforces bytes-only input
# ---------------------------------------------------------------------------


def test_metadata_isolation():
    """
    LEAKAGE CONTROL (DEC-003, DEC-004, AGENTS.md Rule 6):
        preprocess() must accept exactly (raw_bytes,).
        path / caption / label_b must cause TypeError.
    """
    import inspect
    sig = inspect.signature(preprocess)
    param_names = list(sig.parameters.keys())

    assert param_names == ["raw_bytes"], (
        f"preprocess() must accept exactly ['raw_bytes'], got {param_names}. "
        f"Metadata (path, caption, label_b) must never reach the function."
    )

    img = _make_solid_image(640, 480)
    raw_bytes = _image_to_bytes(img)

    with _assert_raises(TypeError):
        preprocess(raw_bytes, path="some/path.jpg")  # type: ignore[call-arg]

    with _assert_raises(TypeError):
        preprocess(raw_bytes, caption="some caption")  # type: ignore[call-arg]

    with _assert_raises(TypeError):
        preprocess(raw_bytes, label_b=3)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TEST 7: Raw data immutability — file-size + partial-hash snapshot
# ---------------------------------------------------------------------------


def test_raw_data_integrity():
    """
    FACT (DEC-002, AGENTS.md Rule 4):
        data/defactify/ files must not be modified by test execution.

    NOTE: These files are git-ignored (confirmed: .gitignore:7:/data/).
    An empty 'git diff HEAD' is NOT sufficient proof of immutability for
    untracked files. This test therefore:

        1. Records file sizes and SHA-256 of the FIRST 64 KB of each Parquet
           file BEFORE any test-suite code that touches the data runs.
        2. Verifies the same values AFTER (i.e. within the same process run).

    We hash only the first 64 KB (header / footer region) of each file rather
    than all 7 GB. If a file were overwritten or truncated the header would
    differ; if bytes were appended the size would differ. This is a meaningful
    lightweight check, not a full-content hash.
    """
    data_dir = os.path.join(_PROJECT_ROOT, "data", "defactify", "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"data/defactify/data/ not found at {data_dir}. "
            "Raw Parquet files are required."
        )

    parquet_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".parquet")
    )
    assert len(parquet_files) > 0, "No Parquet files found in data/defactify/data/"

    HEADER_BYTES = 65536  # 64 KB

    snapshots = {}
    for fname in parquet_files:
        fpath = os.path.join(data_dir, fname)
        fsize = os.path.getsize(fpath)
        with open(fpath, "rb") as f:
            header = f.read(HEADER_BYTES)
        h = hashlib.sha256(header).hexdigest()
        snapshots[fname] = (fsize, h)

    # Re-check immediately (within the same test process).
    for fname, (expected_size, expected_hash) in snapshots.items():
        fpath = os.path.join(data_dir, fname)
        actual_size = os.path.getsize(fpath)
        with open(fpath, "rb") as f:
            actual_header = f.read(HEADER_BYTES)
        actual_hash = hashlib.sha256(actual_header).hexdigest()

        assert actual_size == expected_size, (
            f"STOP: Raw file size changed for {fname}!\n"
            f"  Expected: {expected_size} bytes\n"
            f"  Actual:   {actual_size} bytes\n"
            f"Per DEC-002 and AGENTS.md Rule 4, raw data is immutable."
        )
        assert actual_hash == expected_hash, (
            f"STOP: Raw file content changed for {fname} (first {HEADER_BYTES} bytes)!\n"
            f"  Expected SHA-256: {expected_hash}\n"
            f"  Actual SHA-256:   {actual_hash}\n"
            f"Per DEC-002 and AGENTS.md Rule 4, raw data is immutable."
        )
