"""Controlled JPEG recompression response feature extraction (Block 2 — Branch E candidate: E2_RESP).

Mathematical & Forensic Formulation
-----------------------------------
Input:
    Canonical RGB image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

Controlled In-Memory Symmetric JPEG Recompression:
    For a canonical RGB image I in [0, 1]:
        1. Quantize float32 [0, 1] to uint8 [0, 255] and convert RGB to BGR for standard codec.
        2. Encode to in-memory JPEG byte buffer at target quality Q:
               buf = cv2.imencode('.jpg', I_uint8, [cv2.IMWRITE_JPEG_QUALITY, Q])
        3. Decode from byte buffer:
               I_recomp = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        4. Convert back to RGB float32 in [0, 1].
        5. Compute absolute per-pixel error map:
               Delta_Q(y, x) = mean_c(|I(c, y, x) - I_recomp(c, y, x)|) in [0, 1]

Approved Quality Grid:
    Q in {95, 90, 75, 60}

Candidate Features (8 scalars — prefix: ela_*):
-----------------------------------------------
    11. ela_q95_mean      : Mean error across pixels at Q=95 (near-lossless probe).
    12. ela_q90_mean      : Mean error across pixels at Q=90 (high-quality probe).
    13. ela_q75_mean      : Mean error across pixels at Q=75 (baseline quality probe).
    14. ela_q60_mean      : Mean error across pixels at Q=60 (aggressive quality probe).
    15. ela_q90_energy    : Mean squared error energy mean(Delta_90^2).
    16. ela_slope_q90_q75 : Compressibility slope: (ela_q75_mean - ela_q90_mean) / 15.0.
    17. ela_ratio_q90_q75 : Cross-quality ratio: ela_q90_mean / (ela_q75_mean + 1e-6).
    18. ela_q90_gini      : Gini coefficient measuring spatial concentration/sparsity of Delta_90.

Gini Coefficient Definition:
    For non-negative error values y sorted in ascending order y_1 <= y_2 <= ... <= y_N:
        G = (2 / N) * (sum_{i=1}^N i * y_i) / (sum_{i=1}^N y_i + 1e-12) - (N + 1) / N
    Bounded in [0, 1]. G = 0 for perfectly uniform error; G -> 1 for error concentrated in few pixels.

CRITICAL SCIENTIFIC QUALIFICATIONS
----------------------------------
1. Symmetric Execution: Every image is subjected to the identical in-memory JPEG codec.
2. Metadata Isolation: No external EXIF or file container metadata is consumed.
3. Monotonicity: While natural error typically increases with lower Q (higher compression),
   monotonicity is NOT assumed to be absolute for all synthetic patterns.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# In-Memory JPEG Codec Helpers
# ---------------------------------------------------------------------------

def _jpeg_recompress_single(image_rgb_np: np.ndarray, quality: int) -> np.ndarray:
    """Recompress a single RGB uint8 image [H, W, 3] in memory at quality Q.

    Args:
        image_rgb_np: RGB uint8 numpy array [H, W, 3].
        quality: JPEG quality factor in [1, 100].

    Returns:
        Recompressed RGB uint8 numpy array [H, W, 3].
    """
    # RGB to BGR for OpenCV encoder
    bgr = cv2.cvtColor(image_rgb_np, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buf = cv2.imencode(".jpg", bgr, encode_params)
    if not success:
        raise RuntimeError(f"In-memory JPEG encode failed at quality Q={quality}.")
    decoded_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError(f"In-memory JPEG decode failed at quality Q={quality}.")
    recomp_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return recomp_rgb


def compute_recompression_error_map(
    image: torch.Tensor, quality: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute recompression error map Delta_Q and recompressed image tensor.

    Args:
        image: Canonical RGB tensor [3, H, W] or [B, 3, H, W], float32 in [0, 1].
        quality: JPEG quality factor (e.g. 95, 90, 75, 60).

    Returns:
        Tuple of (error_map [B, 1, H, W] in [0, 1], recomp_image [B, 3, H, W] in [0, 1]).
    """
    if image.dim() == 3:
        img = image.unsqueeze(0)
    else:
        img = image

    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    # Convert to uint8 numpy for in-memory codec
    img_clamped = torch.clamp(img, 0.0, 1.0)
    img_uint8 = (img_clamped * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()

    recomp_list = []
    error_list = []
    for b in range(B):
        recomp_np = _jpeg_recompress_single(img_uint8[b], quality)
        recomp_t = torch.from_numpy(recomp_np).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        orig_t = img_clamped[b].cpu()  # [3, H, W]
        # Absolute error per channel, averaged across channels
        err_t = (orig_t - recomp_t).abs().mean(dim=0, keepdim=True)  # [1, H, W]
        recomp_list.append(recomp_t)
        error_list.append(err_t)

    recomp_tensor = torch.stack(recomp_list, dim=0).to(device=device, dtype=dtype)
    error_tensor = torch.stack(error_list, dim=0).to(device=device, dtype=dtype)
    return error_tensor, recomp_tensor


def _safe_gini_coefficient(flat: torch.Tensor) -> float:
    """Compute Gini coefficient of non-negative values safely.

    Args:
        flat: 1D float32 tensor >= 0.

    Returns:
        Scalar float in [0, 1]. Returns 0.0 if total sum < 1e-12.
    """
    flat = flat.float()
    total = flat.sum()
    if float(total) < 1e-12:
        return 0.0
    n = flat.numel()
    sorted_vals, _ = torch.sort(flat)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=flat.device)
    gini = (2.0 / n) * (index * sorted_vals).sum() / total - (n + 1.0) / n
    return float(torch.clamp(gini, min=0.0, max=1.0))


# ---------------------------------------------------------------------------
# Public Extraction Function
# ---------------------------------------------------------------------------

def extract_branch_e_recompression_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract 8 candidate E2_RESP scalar features across Q in {95, 90, 75, 60}.

    Features:
        11. ela_q95_mean
        12. ela_q90_mean
        13. ela_q75_mean
        14. ela_q60_mean
        15. ela_q90_energy
        16. ela_slope_q90_q75
        17. ela_ratio_q90_q75
        18. ela_q90_gini

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of exactly 8 scalar float features in deterministic order.
    """
    err_q95, _ = compute_recompression_error_map(image, quality=95)
    err_q90, _ = compute_recompression_error_map(image, quality=90)
    err_q75, _ = compute_recompression_error_map(image, quality=75)
    err_q60, _ = compute_recompression_error_map(image, quality=60)

    flat_q95 = err_q95.reshape(-1).cpu()
    flat_q90 = err_q90.reshape(-1).cpu()
    flat_q75 = err_q75.reshape(-1).cpu()
    flat_q60 = err_q60.reshape(-1).cpu()

    q95_mean = float(flat_q95.mean())
    q90_mean = float(flat_q90.mean())
    q75_mean = float(flat_q75.mean())
    q60_mean = float(flat_q60.mean())

    q90_energy = float(flat_q90.pow(2).mean())
    slope_q90_q75 = (q75_mean - q90_mean) / 15.0
    ratio_q90_q75 = q90_mean / (q75_mean + 1e-6)
    q90_gini = _safe_gini_coefficient(flat_q90)

    features: Dict[str, float] = {
        "ela_q95_mean": q95_mean,
        "ela_q90_mean": q90_mean,
        "ela_q75_mean": q75_mean,
        "ela_q60_mean": q60_mean,
        "ela_q90_energy": q90_energy,
        "ela_slope_q90_q75": slope_q90_q75,
        "ela_ratio_q90_q75": ratio_q90_q75,
        "ela_q90_gini": q90_gini,
    }

    return features
