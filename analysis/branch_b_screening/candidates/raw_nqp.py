"""Candidate A: Raw Noise Quantization Probe (NQP) Baseline.

Faithfully reproduces the mathematical behavior of production Branch B
(src/rgb/branch_b_nqp/probe.py) in an isolated research harness for benchmarking
as a reference control.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def derive_sample_seed(base_seed: int, sample_id: Union[str, int]) -> int:
    """Derive deterministic 32-bit unsigned integer seed from sample identifier.

    Uses SHA-256 hash of f"{base_seed}_{sample_id}" to guarantee sample independence
    from batch ordering and evaluation sequence.
    """
    key = f"{base_seed}_{sample_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def compute_raw_nqp_from_noise(
    image: torch.Tensor,
    noise_replicas: torch.Tensor,
) -> torch.Tensor:
    """Compute raw NQP residual given pre-generated noise replicas.

    Enables controlled deterministic mathematical tests on threshold crossing,
    cancellation, and boundary clipping without relying on random sampling.

    Args:
        image: Exact uint8 RGB tensor [3, H, W] in [0, 255].
        noise_replicas: Float tensor [R, 3, H, W] representing noise N_r.

    Returns:
        torch.Tensor: Signed float32 residual Delta = I - I_hat of shape [3, H, W].
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image of shape [3, H, W], got {list(image.shape)}")
    if noise_replicas.ndim != 4 or noise_replicas.shape[1:] != image.shape:
        raise ValueError(
            f"Expected noise of shape [R, 3, H, W] matching image, got {list(noise_replicas.shape)}"
        )

    R = noise_replicas.shape[0]
    I_float = image.to(dtype=torch.float32)

    accumulator = torch.zeros_like(I_float)
    for r in range(R):
        noisy = I_float + noise_replicas[r]
        quantized = torch.clamp(torch.round(noisy), 0.0, 255.0)
        accumulator.add_(quantized)

    I_hat = accumulator / float(R)
    residual = I_float - I_hat
    return residual.to(dtype=torch.float32)


def compute_raw_nqp_residual(
    image: Union[torch.Tensor, np.ndarray],
    sample_id: Union[str, int],
    sigma: float = 0.10,
    replicas: int = 50,
    base_seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Execute Candidate A (Raw NQP) on an input image.

    Args:
        image: Canonical RGB uint8 image [3, H, W] or [H, W, 3] in [0, 255].
        sample_id: Unique sample identifier for deterministic seeding.
        sigma: Noise standard deviation in raw intensity units (default: 0.10).
        replicas: Monte Carlo replica count (default: 50).
        base_seed: Base random seed (default: 42).
        device: Execution device (CPU or CUDA).

    Returns:
        Dict containing residual tensor and reproducibility metadata.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            # HWC -> CHW
            image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image)
    else:
        tensor = image

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] image, got shape {list(tensor.shape)}")

    if tensor.dtype != torch.uint8 and tensor.max() <= 1.0:
        # User passed normalized float; contract requires raw uint8 scale
        tensor = (tensor * 255.0).round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.uint8)

    if device is not None:
        tensor = tensor.to(device)

    target_device = tensor.device
    seed = derive_sample_seed(base_seed, sample_id)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)

    I_float = tensor.to(dtype=torch.float32)
    accumulator = torch.zeros_like(I_float)

    for _ in range(replicas):
        noise = torch.randn(
            tensor.shape,
            generator=generator,
            device=target_device,
            dtype=torch.float32,
        ) * sigma
        noisy = I_float + noise
        quantized = torch.clamp(torch.round(noisy), 0.0, 255.0)
        accumulator.add_(quantized)

    I_hat = accumulator / float(replicas)
    residual = I_float - I_hat

    return {
        "residual": residual.to(dtype=torch.float32),
        "candidate_name": "raw_nqp",
        "config": {
            "sigma": sigma,
            "replicas": replicas,
            "base_seed": base_seed,
            "rounding": "nearest_integer",
            "clipping": [0, 255],
        },
        "metadata": {
            "sample_id": sample_id,
            "seed": seed,
            "device": str(target_device),
        },
        "residual_sign": "I - I_hat",
        "output_scale": "raw_uint8_intensity",
        "dtype": "float32",
    }
