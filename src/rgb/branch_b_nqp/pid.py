"""Branch B — Active Representation: PiD ICML 2025 Deterministic Color-Space Decomposition.

Implements Pixelwise Decomposition Residuals (PiD, ICML 2025) using the verified
ITU-R BT.601 forward matrix and the corroborated Figure 2 inverse matrix, with
scaling factor s=0.2 as finalized in DEC-015.

OPERATION SPECIFICATION:
    1. Forward projection to YUV under matrix scaling factor s (default s=0.2):
       yuv = s * x * M_YUV^T
    2. Quantization (nearest integer rounding in YUV space):
       yuv_quant = round(yuv)
    3. Inverse projection back to RGB under (1/s) * M_inv^T:
       rgb_rec = (1/s) * yuv_quant * (M_true^-1)^T
    4. Post-inverse rounding (nearest integer in RGB space):
       x_hat = round(rgb_rec)
    5. Signed residual map:
       Delta = x - x_hat
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch

# Standard ITU-R BT.601 forward matrix (RGB -> YUV)
M_YUV_BT601 = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float64,
)

# Verified inverse matrix corroborated by paper Figure 2 and direct numerical inversion
# Third row [1.0, 1.772, 0.0] reconstructs Blue: B = 1.0*Y + 1.772*U + 0.0*V
M_INV_VERIFIED = np.array(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=np.float64,
)

DEFAULT_SCALE_S: float = 0.2


def forward_yuv(
    rgb: torch.Tensor,
    scale: float = DEFAULT_SCALE_S,
) -> torch.Tensor:
    """Project RGB image to YUV space under scaling factor s.

    Args:
        rgb: Float64 tensor of shape [3, H, W] representing RGB channels.
        scale: Matrix scaling factor s (default: 0.2).

    Returns:
        torch.Tensor: YUV tensor of shape [3, H, W] in float64.
    """
    M_s = torch.from_numpy(scale * M_YUV_BT601).to(device=rgb.device, dtype=torch.float64)
    H, W = rgb.shape[1], rgb.shape[2]
    rgb_hwc = rgb.to(dtype=torch.float64).permute(1, 2, 0).reshape(-1, 3)
    yuv_hwc = torch.matmul(rgb_hwc, M_s.T)
    return yuv_hwc.reshape(H, W, 3).permute(2, 0, 1)


def inverse_yuv(
    yuv: torch.Tensor,
    scale: float = DEFAULT_SCALE_S,
) -> torch.Tensor:
    """Project YUV image back to RGB space under verified inverse and scale 1/s.

    Args:
        yuv: Float64 tensor of shape [3, H, W] representing YUV channels.
        scale: Matrix scaling factor s (inverted as 1/s, default: 0.2).

    Returns:
        torch.Tensor: Reconstructed RGB tensor of shape [3, H, W] in float64.
    """
    M_inv_s = torch.from_numpy((1.0 / scale) * M_INV_VERIFIED).to(
        device=yuv.device, dtype=torch.float64
    )
    H, W = yuv.shape[1], yuv.shape[2]
    yuv_hwc = yuv.to(dtype=torch.float64).permute(1, 2, 0).reshape(-1, 3)
    rgb_hwc = torch.matmul(yuv_hwc, M_inv_s.T)
    return rgb_hwc.reshape(H, W, 3).permute(2, 0, 1)


def compute_pid_residual(
    image: Union[torch.Tensor, np.ndarray],
    sample_id: Optional[Union[str, int]] = None,
    s: float = DEFAULT_SCALE_S,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute active Branch B Pixelwise Decomposition Residual (PiD YUV, s=0.2).

    Args:
        image: Canonical RGB image:
            - torch.Tensor of shape [3, H, W] in [0.0, 1.0] or [0.0, 255.0]
            - np.ndarray of shape [H, W, 3] in [0, 255] uint8 or float
        sample_id: Optional sample identifier for pipeline interface compatibility.
        s: Matrix scaling factor (default: 0.2 per DEC-015).
        device: Optional execution device.

    Returns:
        torch.Tensor of shape [3, H, W] representing signed residual Delta = x - x_hat.
        Residual values are float32 in [-255.0, 255.0].
    """
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected np.ndarray with shape [H, W, 3], got {image.shape}")
        if image.dtype == np.uint8:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        elif np.issubdtype(image.dtype, np.floating):
            arr = image.astype(np.float32)
            if arr.max() <= 1.01:
                arr = arr * 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unsupported numpy dtype: {image.dtype}")
    elif isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected torch.Tensor with shape [3, H, W], got {image.shape}")
        t = image.detach().float()
        if t.max() <= 1.01:
            t = t * 255.0
        tensor = t
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(image)}")

    if device is not None:
        tensor = tensor.to(device)

    target_device = tensor.device
    x_orig = tensor.to(dtype=torch.float64)

    # 1. Forward transform to YUV with scale factor s
    yuv = forward_yuv(x_orig, scale=s)

    # 2. Integer rounding in transformed YUV space
    yuv_quant = torch.round(yuv)

    # 3. Inverse transform back to RGB with 1/s
    rgb_reconstructed = inverse_yuv(yuv_quant, scale=s)

    # 4. Post-inverse rounding in RGB space
    x_hat = torch.round(rgb_reconstructed)

    # 5. Signed residual: continuous original minus reconstructed
    residual = x_orig - x_hat

    return residual.to(device=target_device, dtype=torch.float32)


# Canonical alias for Branch B active representation
compute_branch_b_residual = compute_pid_residual
