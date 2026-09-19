"""
src/data/preprocessing.py
Block 1 — Canonical Image Preprocessing Pipeline

DECISION (DEC-008, formalized in DECISIONS.md):
    Preprocessing contract:
        ORIGINAL IMAGE
            ↓
        LARGEST CENTERED SQUARE CROP
            ↓
        RESIZE TO 256 × 256 (area-based downsampling: cv2.INTER_AREA)
            ↓
        STANDARDIZED PROCESSED IMAGE (RGB, uint8)

Design rationale (from Defactify confound audit):
    - AI images are 100% square; real images are 97.46% non-square.
    - Squashing non-square real images with a naive resize would introduce
      anisotropic geometric distortion that differs structurally between classes,
      creating a resolution/aspect-ratio confound.
    - Centered square crop removes border context symmetrically; crop is identity
      for already-square images, so AI images are untouched by the crop step.
    - Area-based downsampling (cv2.INTER_AREA) is deterministic and is applied identically to every
      sample. The interpolation operation modifies image frequency content and is
      therefore part of the standardized preprocessing contract.

Leakage controls:
    - Only Image["bytes"] is consumed; Image["path"] is never passed here.
    - No ImageNet normalization is performed at this stage.
    - No random augmentation or randomness of any kind.
    - Output is RGB uint8 PIL Image — caller handles tensor conversion if needed.

Dependencies:
    - Pillow
    - OpenCV (cv2)
    - numpy
"""

from __future__ import annotations

import io
import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants — canonical preprocessing contract
# ---------------------------------------------------------------------------

CANONICAL_SIZE: int = 256
"""Target spatial dimension (both width and height) for all processed images."""

INTERPOLATION: int = cv2.INTER_AREA
"""Interpolation flag for OpenCV resize (area-based downsampling)."""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def centered_square_crop(img: Image.Image) -> Image.Image:
    """Return the largest centered square crop of *img*.

    For an image of size (W, H):
        side = min(W, H)
        left = (W - side) // 2
        top  = (H - side) // 2
        box  = (left, top, left + side, top + side)

    This is an identity operation when W == H (already square).

    Args:
        img: A PIL Image in any mode.

    Returns:
        A PIL Image cropped to a square of size (side, side).
    """
    W, H = img.size
    side = min(W, H)
    left = (W - side) // 2
    top  = (H - side) // 2
    box = (left, top, left + side, top + side)
    return img.crop(box)


def resize_to_canonical(img: Image.Image) -> Image.Image:
    """Resize *img* to CANONICAL_SIZE × CANONICAL_SIZE using area-based interpolation (cv2.INTER_AREA).

    Area-based downsampling (cv2.INTER_AREA) is deterministic and is applied identically to every
    sample. The interpolation modifies image frequency content; this is a
    documented part of the preprocessing contract (DEC-008).

    Args:
        img: A PIL Image (should already be square after centered_square_crop).

    Returns:
        A PIL Image of size (CANONICAL_SIZE, CANONICAL_SIZE) in RGB mode.
    """
    np_img = np.array(img)
    resized = cv2.resize(
        np_img,
        (CANONICAL_SIZE, CANONICAL_SIZE),
        interpolation=INTERPOLATION,
    )
    return Image.fromarray(resized)


def preprocess(raw_bytes: bytes) -> Image.Image:
    """Full canonical preprocessing pipeline.

    Steps:
        1. Decode JPEG bytes from Parquet Image["bytes"] field → PIL Image.
        2. Convert to RGB (handles any mode: RGBA, L, P, CMYK, etc.).
        3. Apply centered_square_crop.
        4. Apply resize_to_canonical (cv2.INTER_AREA, 256×256).

    No ImageNet normalization is applied here. Tensor conversion is the
    responsibility of the caller (e.g., DefactifyDataset).

    Args:
        raw_bytes: Raw image bytes as stored in the Parquet Image["bytes"] column.
                   Must be a valid JPEG byte stream.

    Returns:
        A PIL Image in RGB mode with size (CANONICAL_SIZE, CANONICAL_SIZE).

    Raises:
        PIL.UnidentifiedImageError: If raw_bytes is not a valid image.
        ValueError: If the image cannot be converted to RGB.
    """
    img = Image.open(io.BytesIO(raw_bytes))
    img = img.convert("RGB")
    img = centered_square_crop(img)
    img = resize_to_canonical(img)
    return img
