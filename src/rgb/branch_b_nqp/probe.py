"""Branch B — Noise Quantization Probe (NQP) implementation (CoDA-inspired).

Research reference:
Zexi Jia et al., CoDA: Color Distribution Probing for Efficient and
Generalizable AI-Generated Image Detection (2026).

OPERATION SPECIFICATION:
For image I in [0, 255]^{3 x H x W}:
    I'_r = I + N_r,  N_r ~ N(0, sigma^2)
    Y_r = clip(round(I'_r), 0, 255)
    I_hat = (1 / R) * sum_{r=1}^R Y_r
    Delta = I - I_hat

TARGET SETTINGS:
    - sigma = 0.10
    - R = 50 replicas

REPRODUCIBILITY & EFFICIENCY:
    - Deterministic per-sample seeds derived from base_seed and stable sample_id.
    - Memory-efficient accumulation: accumulator sums replicas incrementally
      rather than allocating 50 full images.
    - Preserves signed residual values (Delta in [-255, 255], typically near 0).
"""

from __future__ import annotations

import hashlib
from typing import Union

import numpy as np
import torch


def derive_sample_seed(sample_id: Union[str, int], base_seed: int = 42) -> int:
    """Derive a deterministic 32-bit integer seed from base_seed and sample_id.

    Args:
        sample_id: Stable identifier for the image (e.g. "train_000042" or 42).
        base_seed: Global base seed (default: 42).

    Returns:
        Integer seed in [0, 2**31 - 1].
    """
    key = f"{base_seed}_{str(sample_id)}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:4], byteorder="big") & 0x7FFFFFFF
    return seed


def compute_nqp_residual(
    image: Union[torch.Tensor, np.ndarray],
    sample_id: Union[str, int],
    sigma: float = 0.10,
    replicas: int = 50,
    base_seed: int = 42,
) -> torch.Tensor:
    """Compute the Noise Quantization Probe (NQP) residual Delta.

    Args:
        image: Canonical RGB image:
            - torch.Tensor of shape [3, H, W] in [0.0, 1.0] or [0.0, 255.0]
            - np.ndarray of shape [H, W, 3] in [0, 255] uint8 or float
        sample_id: Unique sample identifier for deterministic seeding.
        sigma: Noise standard deviation (paper default: 0.10).
        replicas: Number of Monte Carlo quantization replicas R (paper default: 50).
        base_seed: Base seed for RNG (default: 42).

    Returns:
        torch.Tensor of shape [3, H, W] representing the signed residual Delta = I - I_hat.
        Residual values are float32 in [-255.0, 255.0].
    """
    # 1. Standardize input to float32 tensor in intensity range [0.0, 255.0] with shape [3, H, W]
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected np.ndarray with shape [H, W, 3], got {image.shape}")
        if image.dtype == np.uint8:
            I_float = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
        elif np.issubdtype(image.dtype, np.floating):
            arr = image.astype(np.float32)
            if arr.max() <= 1.01:
                arr = arr * 255.0
            I_float = torch.from_numpy(arr).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unsupported numpy dtype: {image.dtype}")
    elif isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected torch.Tensor with shape [3, H, W], got {image.shape}")
        t = image.detach().float()
        if t.max() <= 1.01:
            t = t * 255.0
        I_float = t
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(image)}")

    device = I_float.device
    shape = I_float.shape

    # 2. Derive deterministic RNG seed and generator
    seed = derive_sample_seed(sample_id, base_seed=base_seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # 3. Incremental replica accumulation
    accum = torch.zeros(shape, dtype=torch.float32)
    cpu_I = I_float.cpu()

    for _ in range(replicas):
        noise = torch.randn(shape, generator=gen, dtype=torch.float32) * sigma
        perturbed = cpu_I + noise
        quantized = torch.clamp(torch.round(perturbed), 0.0, 255.0)
        accum += quantized

    I_hat = accum / float(replicas)
    delta = cpu_I - I_hat

    return delta.to(device)
