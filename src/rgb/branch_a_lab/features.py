"""Branch A composition layer for Lab Local Correlation Maps.

Extracts 3-channel inter-channel correlation maps (L-a, L-b, a-b) directly from
canonical RGB images.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from .color_conversion import rgb_to_cielab
from .correlation import compute_local_correlation_maps


def extract_branch_a_lab_maps(
    image: Union[torch.Tensor, np.ndarray],
    window_size: int = 7,
) -> torch.Tensor:
    """Extract Branch A Lab local correlation maps from an RGB image.

    Args:
        image: Canonical RGB image:
            - torch.Tensor of shape [3, 256, 256] in [0, 1]
            - torch.Tensor of shape [B, 3, 256, 256] in [0, 1]
            - np.ndarray of shape [256, 256, 3] in [0, 255] or [0, 1]
        window_size: Sliding window size (default: 7).

    Returns:
        torch.Tensor of shape [3, 256, 256] or [B, 3, 256, 256] as float32
        with values in [-1.0, 1.0].
        Channels:
            0: r(L, a)
            1: r(L, b)
            2: r(a, b)
    """
    lab_tensor = rgb_to_cielab(image)
    return compute_local_correlation_maps(lab_tensor, window_size=window_size)
