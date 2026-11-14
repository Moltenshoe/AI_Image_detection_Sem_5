"""Color space conversion for Branch A (Lab Local Correlation Maps).

Converts canonical RGB images to CIE L*a*b* color space.

CONVENTIONS & SPECIFICATION:
- Input: Canonical RGB image as either:
    1) PyTorch float32 tensor of shape [3, H, W] or [B, 3, H, W] in [0.0, 1.0].
    2) NumPy uint8 array of shape [H, W, 3] in [0, 255].
- Implementation: OpenCV `cv2.cvtColor` with `cv2.COLOR_RGB2LAB`.
- Output: Float32 PyTorch tensor of shape [3, H, W] or [B, 3, H, W] in CIE Lab space:
    - L* in [0.0, 100.0]
    - a* in [-128.0, 127.0] (typically ~[-86.2, 98.2])
    - b* in [-128.0, 127.0] (typically ~[-107.9, 94.5])
- Channel ordering:
    - Channel 0: L* (Lightness)
    - Channel 1: a* (Green - Magenta axis)
    - Channel 2: b* (Blue - Yellow axis)

Conversion operates directly in float32 without intermediate uint8 quantization
to preserve continuous sub-integer color relationships.
"""

from __future__ import annotations

from typing import Union

import cv2
import numpy as np
import torch


def rgb_to_cielab(image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert an RGB image to CIE L*a*b* color space.

    Args:
        image: RGB input image. Supported formats:
            - torch.Tensor of shape [3, H, W] with values in [0.0, 1.0].
            - torch.Tensor of shape [B, 3, H, W] with values in [0.0, 1.0].
            - np.ndarray of shape [H, W, 3] with values in [0, 255] (uint8)
              or [0.0, 1.0] (float32).

    Returns:
        torch.Tensor of shape [3, H, W] or [B, 3, H, W] containing CIE Lab channels
        (L in [0, 100], a in [-128, 127], b in [-128, 127]) as float32.

    Raises:
        ValueError: If input tensor dimensions, channel counts, or ranges are invalid.
    """
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected np.ndarray with shape [H, W, 3], got {image.shape}")
        if image.dtype == np.uint8:
            rgb_f = image.astype(np.float32) / 255.0
        elif np.issubdtype(image.dtype, np.floating):
            rgb_f = image.astype(np.float32)
            if rgb_f.max() > 1.01:
                rgb_f = rgb_f / 255.0
        else:
            raise ValueError(f"Unsupported numpy dtype: {image.dtype}")

        lab_np = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2LAB)  # [H, W, 3]
        lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).float()  # [3, H, W]
        return lab_tensor

    elif isinstance(image, torch.Tensor):
        if image.ndim == 3:
            if image.shape[0] != 3:
                raise ValueError(f"Expected 3 channels in dim 0 for [3, H, W], got {image.shape}")
            t = image.detach().float()
            if t.max() > 1.01 and t.max() <= 255.0:
                t = t / 255.0

            np_rgb = t.permute(1, 2, 0).cpu().numpy()
            lab_np = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2LAB)
            return torch.from_numpy(lab_np).permute(2, 0, 1).to(image.device).float()

        elif image.ndim == 4:
            if image.shape[1] != 3:
                raise ValueError(f"Expected 3 channels in dim 1 for [B, 3, H, W], got {image.shape}")
            b_size = image.shape[0]
            out_list = []
            for b in range(b_size):
                out_list.append(rgb_to_cielab(image[b]))
            return torch.stack(out_list, dim=0)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {image.shape}")
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(image)}")
