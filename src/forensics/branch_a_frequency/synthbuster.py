"""Synthbuster-inspired periodicity analysis module (Block 2 — Branch A2).

Provides modular, deterministic cross-difference residual and periodic frequency peak
feature extraction on canonical image tensors.
Operates independently per RGB channel at specified periods (2, 4, 8) and directional components (x, y, diagonal).

Reference implementation refactored from historical exploratory notebook analysis_800_v1.ipynb.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from src.forensics.branch_a_frequency.fft import frequency_radius


def cross_difference(image: torch.Tensor) -> torch.Tensor:
    """Compute 2D cross-difference residual tensor.

    Formula:
        | I[y, x] + I[y+1, x+1] - I[y+1, x] - I[y, x+1] |

    Produces spatial dimensions [..., C, H-1, W-1].
    For a canonical 256x256 image, the residual spatial size is exactly 255x255.

    Args:
        image: Tensor of shape [..., C, H, W].

    Returns:
        Tensor of shape [..., C, H-1, W-1].
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")
    if image.ndim < 2 or image.shape[-2] < 2 or image.shape[-1] < 2:
        raise ValueError(
            f"Image spatial dimensions must be at least 2x2, got shape {tuple(image.shape)}"
        )

    return (
        image[..., :-1, :-1]
        + image[..., 1:, 1:]
        - image[..., 1:, :-1]
        - image[..., :-1, 1:]
    ).abs()


def synthbuster_periodicity_features(
    image: torch.Tensor,
    periods: Sequence[int] = (2, 4, 8),
) -> Dict[str, float]:
    """Extract Synthbuster-inspired periodicity scalar detector features (Branch A2).

    Pipeline per RGB channel:
        1. Extract single-channel image.
        2. Compute cross-difference residual ([H-1, W-1]).
        3. 2D FFT with orthonormal normalization (`torch.fft.fft2(..., norm="ortho")`).
        4. Quadrant shift (`torch.fft.fftshift`).
        5. Measure magnitude at period-related frequency offsets (x, y, diagonal directions).
        6. Compute residual high-frequency energy ratio (radius >= 0.50).

    Args:
        image: Canonical RGB image tensor [3, H, W] or [B, 3, H, W], float32 in [0, 1].
        periods: Sequence of pixel period lengths to probe (default: (2, 4, 8)).

    Returns:
        Dictionary of exactly 30 scalar float features (10 per RGB channel):
            - synth_{channel}_p{period}_{direction}_mean for channel in (r, g, b),
              period in (2, 4, 8), direction in (x, y, d)
            - synth_{channel}_fft_highfreq_ratio for channel in (r, g, b)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")

    orig_ndim = image.ndim
    if orig_ndim == 3:
        x_in = image.unsqueeze(0)
    elif orig_ndim == 4:
        x_in = image
    else:
        raise ValueError(
            f"Expected 3D [C, H, W] or 4D [B, C, H, W] tensor, got ndim={orig_ndim}"
        )

    if x_in.shape[1] != 3:
        raise ValueError(
            f"Expected 3 RGB channels at dim 1, got shape {tuple(x_in.shape)}"
        )

    out: Dict[str, float] = {}
    channel_names = ("r", "g", "b")

    for channel_idx, name in enumerate(channel_names):
        channel_image = x_in[:, channel_idx : channel_idx + 1]
        residual = cross_difference(channel_image)

        h, w = residual.shape[-2:]
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(residual, norm="ortho"),
            dim=(-2, -1),
        )
        magnitude = spectrum.abs()
        power = magnitude.square()

        cy, cx = h // 2, w // 2

        for period in periods:
            ky = min(max(1, round(h / period)), h // 2)
            kx = min(max(1, round(w / period)), w // 2)

            directions = {
                "x": [(0, kx), (0, -kx)],
                "y": [(ky, 0), (-ky, 0)],
                "d": [
                    (ky, kx),
                    (ky, -kx),
                    (-ky, kx),
                    (-ky, -kx),
                ],
            }

            for direction, offsets in directions.items():
                values = [
                    magnitude[..., cy + dy, cx + dx]
                    for dy, dx in offsets
                    if 0 <= cy + dy < h and 0 <= cx + dx < w
                ]

                if values:
                    stacked = torch.stack(values)
                    out[f"synth_{name}_p{period}_{direction}_mean"] = float(
                        stacked.mean().detach().cpu()
                    )
                else:
                    out[f"synth_{name}_p{period}_{direction}_mean"] = 0.0

        radius = frequency_radius(h, w, device=residual.device, dtype=residual.dtype).view(
            1, 1, h, w
        )
        total_power = power.sum() + 1e-12
        high_mask = (radius >= 0.50).float()
        hf_ratio = (power * high_mask).sum() / total_power
        out[f"synth_{name}_fft_highfreq_ratio"] = float(hf_ratio.detach().cpu())

    return out
