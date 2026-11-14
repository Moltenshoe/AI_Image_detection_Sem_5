"""Candidate C: PiD ICML 2025 Deterministic Color-Space Decomposition.

Primary-source implementation of Pixelwise Decomposition Residuals (ICML 2025)
using the verified ITU-R BT.601 forward matrix and the corroborated Figure 2
inverse matrix, supporting Table 5 matrix scaling factors.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

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
# Third row (index 2) [1.0, 1.772, 0.0] reconstructs Blue: B = 1.0*Y + 1.772*U + 0.0*V
M_INV_VERIFIED = np.array(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=np.float64,
)


def forward_yuv(
    rgb: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Project RGB image to YUV space under optional matrix scaling factor s.

    Args:
        rgb: Float tensor of shape [3, H, W] representing RGB channels.
        scale: Matrix scaling factor s (default: 1.0).

    Returns:
        torch.Tensor: YUV tensor of shape [3, H, W] in float64.
    """
    M_s = torch.from_numpy(scale * M_YUV_BT601).to(device=rgb.device, dtype=torch.float64)
    # rgb shape: [3, H, W] -> permute to [H, W, 3] for matrix multiplication
    H, W = rgb.shape[1], rgb.shape[2]
    rgb_hwc = rgb.to(dtype=torch.float64).permute(1, 2, 0).reshape(-1, 3)
    yuv_hwc = torch.matmul(rgb_hwc, M_s.T)
    return yuv_hwc.reshape(H, W, 3).permute(2, 0, 1)


def inverse_yuv(
    yuv: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Project YUV image back to RGB space under verified inverse and scale 1/s.

    Args:
        yuv: Float tensor of shape [3, H, W] representing YUV channels.
        scale: Matrix scaling factor s (inverted as 1/s, default: 1.0).

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
    s: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Execute Candidate C (PiD ICML 2025) deterministic color-space decomposition.

    Algorithm:
        x' = round(s * x * M_YUV^T)
        x_hat = round((1/s) * x' * (M_true^-1)^T)
        R_pid = x - x_hat

    Args:
        image: Exact uint8 RGB image [3, H, W] or [H, W, 3] in [0, 255].
        sample_id: Optional sample identifier for record keeping.
        s: Matrix scaling factor from Table 5 (default: 1.0 unscaled reference;
           sensitivity options: 0.2, 0.5, 0.8, 1.0, 1.5, 2.0).
        device: Execution device (CPU or CUDA).

    Returns:
        Dict containing residual tensor and reproducibility metadata.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image)
    else:
        tensor = image

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] image, got shape {list(tensor.shape)}")

    if tensor.dtype != torch.uint8 and tensor.max() <= 1.0:
        tensor = (tensor * 255.0).round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.uint8)

    if device is not None:
        tensor = tensor.to(device)

    target_device = tensor.device
    x_orig = tensor.to(dtype=torch.float64)

    # 1. Forward transform to YUV with scaling factor s
    yuv = forward_yuv(x_orig, scale=s)

    # 2. Transformed-space quantization (nearest integer in YUV)
    yuv_quant = torch.round(yuv)

    # 3. Inverse transform back to RGB with 1/s
    rgb_reconstructed = inverse_yuv(yuv_quant, scale=s)

    # 4. Post-inverse RGB rounding (nearest integer in RGB)
    x_hat = torch.round(rgb_reconstructed)

    # 5. Residual: original minus reconstructed (signed)
    residual = x_orig - x_hat

    return {
        "residual": residual.to(dtype=torch.float32),
        "candidate_name": "pid_yuv",
        "implementation_type": "primary_source_deterministic",
        "config": {
            "scale_factor_s": s,
            "forward_matrix": "ITU-R_BT.601",
            "inverse_matrix": "Figure_2_verified_inverse",
            "replicas": 1,
            "noise": None,
            "yuv_quantization": "nearest_integer",
            "rgb_post_rounding": "nearest_integer",
            "clipping": "none",
        },
        "metadata": {
            "sample_id": sample_id,
            "device": str(target_device),
            "deterministic": True,
        },
        "residual_sign": "x - x_hat",
        "output_scale": "raw_uint8_intensity",
        "dtype": "float32",
    }
