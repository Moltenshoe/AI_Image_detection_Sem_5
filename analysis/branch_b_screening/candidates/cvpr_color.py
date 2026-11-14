"""Candidate B: CVPR 2025 4-Bit Color Quantization Residual.

Faithfully implements the quantizer and perturbation pipeline from CVPR 2025
("Secret Lies in Color"), supporting both the primary-source reference reproduction
(scikit-image Gaussian noise) and the deterministic project adaptation (PyTorch SHA-256).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Union

import numpy as np
import skimage.util
import torch


def quantize_cvpr(
    x: torch.Tensor,
    L: int = 16,
    clip_quant: bool = True,
) -> torch.Tensor:
    """Apply uniform color quantizer Q_L(x).

    Formula:
        Q_L(x) = clamp(round((L - 1) * x) / (L - 1), 0.0, 1.0)
    When clip_quant is False, the outer clamp is omitted.

    Args:
        x: Continuous float tensor in normalized space [0.0, 1.0].
        L: Number of quantization levels (default: 16 -> step q = 1/15).
        clip_quant: Whether to clamp quantized values to [0.0, 1.0].

    Returns:
        torch.Tensor: Quantized tensor of same shape and dtype.
    """
    scale = float(L - 1)
    quantized = torch.round(x * scale) / scale
    if clip_quant:
        quantized = torch.clamp(quantized, 0.0, 1.0)
    return quantized


def compute_cvpr_from_noise(
    x: torch.Tensor,
    noise_replicas: torch.Tensor,
    clip_noise: bool = True,
    clip_quant: bool = True,
    L: int = 16,
) -> torch.Tensor:
    """Compute CVPR quantization residual from pre-generated noise replicas.

    Enables controlled mathematical testing of quantizer thresholds, clipping
    disentanglement, and accumulation without relying on stochastic sampling.

    Args:
        x: Normalized continuous float32 tensor [3, H, W] in [0.0, 1.0].
        noise_replicas: Float tensor [R, 3, H, W] of additive perturbations.
        clip_noise: Whether to clamp (x + N) to [0.0, 1.0].
        clip_quant: Whether to clamp quantizer output to [0.0, 1.0].
        L: Quantization levels (default: 16).

    Returns:
        torch.Tensor: Signed residual Delta = x_hat - x of shape [3, H, W].
    """
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected tensor of shape [3, H, W], got {list(x.shape)}")
    if noise_replicas.ndim != 4 or noise_replicas.shape[1:] != x.shape:
        raise ValueError(
            f"Expected noise of shape [R, 3, H, W] matching x, got {list(noise_replicas.shape)}"
        )

    R = noise_replicas.shape[0]
    accumulator = torch.zeros_like(x)

    for r in range(R):
        noisy = x + noise_replicas[r]
        if clip_noise:
            noisy = torch.clamp(noisy, 0.0, 1.0)
        q = quantize_cvpr(noisy, L=L, clip_quant=clip_quant)
        accumulator.add_(q)

    x_hat = accumulator / float(R)
    # Residual sign: reconstruction minus continuous input
    residual = x_hat - x
    return residual.to(dtype=torch.float32)


def derive_sample_seed(base_seed: int, sample_id: Union[str, int]) -> int:
    """Derive deterministic 32-bit unsigned integer seed from sample identifier."""
    key = f"{base_seed}_{sample_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def extract_cvpr_reference(
    image: Union[torch.Tensor, np.ndarray],
    sample_id: Optional[Union[str, int]] = None,
    replicas: int = 500,
    var: float = 0.01,
    L: int = 16,
    clip_noise: bool = True,
    clip_quant: bool = True,
) -> Dict[str, Any]:
    """Primary-source reference reproduction of CVPR 2025 supplementary code.

    Calls `skimage.util.random_noise(..., mode='gaussian', clip=clip_noise, var=var)`.
    Does NOT use SHA-256 seeding; matches author unseeded stochastic execution.
    Subtracts original continuous image from the average quantized replica.

    Args:
        image: Continuous float tensor [3, H, W] in [0.0, 1.0].
        sample_id: Optional sample identifier for record keeping.
        replicas: Replica count R (Table 5 best: 500, or sensitivity: 50).
        var: Gaussian noise variance (default: 0.01, sigma = 0.10).
        L: Quantization levels (default: 16 -> step q = 1/15).
        clip_noise: Noise-stage clipping (default: True).
        clip_quant: Quantizer-stage clipping (default: True).

    Returns:
        Dict containing residual tensor and execution metadata.
    """
    if isinstance(image, torch.Tensor):
        x = image.detach().cpu().to(dtype=torch.float32)
        if x.ndim == 3 and x.shape[0] == 3:
            # [3, H, W] -> [H, W, 3] for scikit-image
            x_hwc = x.permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"Expected [3, H, W] tensor, got {list(x.shape)}")
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] == 3:
            x_hwc = arr
            x = torch.from_numpy(arr).permute(2, 0, 1)
        elif arr.ndim == 3 and arr.shape[0] == 3:
            x_hwc = np.transpose(arr, (1, 2, 0))
            x = torch.from_numpy(arr)
        else:
            raise ValueError(f"Expected 3-channel image, got shape {arr.shape}")

    if x.max() > 1.0:
        raise ValueError("Candidate B contract requires normalized float inputs in [0.0, 1.0].")

    # Accumulate using scikit-image random_noise
    accumulator_hwc = np.zeros_like(x_hwc, dtype=np.float64)
    scale = float(L - 1)

    for _ in range(replicas):
        # skimage.util.random_noise handles gaussian noise with var=0.01
        noisy_hwc = skimage.util.random_noise(
            x_hwc,
            mode="gaussian",
            clip=clip_noise,
            var=var,
        )
        # Quantize
        q_hwc = np.round(noisy_hwc * scale) / scale
        if clip_quant:
            q_hwc = np.clip(q_hwc, 0.0, 1.0)
        accumulator_hwc += q_hwc

    x_hat_hwc = accumulator_hwc / float(replicas)
    x_hat = torch.from_numpy(x_hat_hwc).permute(2, 0, 1).to(dtype=torch.float32)

    # Residual sign: reconstruction minus continuous input
    residual = x_hat - x

    return {
        "residual": residual.to(dtype=torch.float32),
        "candidate_name": "cvpr_color_reference",
        "implementation_type": "reference_reproduction",
        "config": {
            "replicas": replicas,
            "variance": var,
            "sigma": float(np.sqrt(var)),
            "levels": L,
            "step_q": 1.0 / float(L - 1),
            "clip_noise": clip_noise,
            "clip_quant": clip_quant,
            "rng_library": "skimage.util.random_noise",
            "skimage_version": skimage.__version__,
        },
        "metadata": {
            "sample_id": sample_id,
            "seeded": False,
        },
        "residual_sign": "x_hat - x",
        "output_scale": "normalized_float32",
        "dtype": "float32",
    }


def extract_cvpr_seeded(
    image: Union[torch.Tensor, np.ndarray],
    sample_id: Union[str, int],
    replicas: int = 50,
    sigma: float = 0.10,
    L: int = 16,
    clip_noise: bool = True,
    clip_quant: bool = True,
    base_seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Deterministic project adaptation of CVPR 2025 4-bit color quantization.

    Uses PyTorch torch.randn with per-image deterministic seed derived from
    SHA-256 hash of (base_seed, sample_id) to enforce reproducibility.

    Args:
        image: Continuous float tensor [3, H, W] in [0.0, 1.0].
        sample_id: Unique sample identifier for deterministic seeding.
        replicas: Replica count R (default: 50 for screening budget).
        sigma: Gaussian standard deviation (default: 0.10 in [0, 1]).
        L: Quantization levels (default: 16 -> step q = 1/15).
        clip_noise: Noise-stage clipping (default: True).
        clip_quant: Quantizer-stage clipping (default: True).
        base_seed: Base seed (default: 42).
        device: Execution device (CPU or CUDA).

    Returns:
        Dict containing residual tensor and execution metadata.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        x = torch.from_numpy(image).to(dtype=torch.float32)
    else:
        x = image.to(dtype=torch.float32)

    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] tensor, got shape {list(x.shape)}")

    if x.max() > 1.0:
        raise ValueError("Candidate B contract requires normalized float inputs in [0.0, 1.0].")

    if device is not None:
        x = x.to(device)

    target_device = x.device
    seed = derive_sample_seed(base_seed, sample_id)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)

    accumulator = torch.zeros_like(x)
    scale = float(L - 1)

    for _ in range(replicas):
        noise = torch.randn(
            x.shape,
            generator=generator,
            device=target_device,
            dtype=torch.float32,
        ) * sigma

        noisy = x + noise
        if clip_noise:
            noisy = torch.clamp(noisy, 0.0, 1.0)

        quantized = torch.round(noisy * scale) / scale
        if clip_quant:
            quantized = torch.clamp(quantized, 0.0, 1.0)

        accumulator.add_(quantized)

    x_hat = accumulator / float(replicas)
    residual = x_hat - x

    return {
        "residual": residual.to(dtype=torch.float32),
        "candidate_name": "cvpr_color_seeded",
        "implementation_type": "deterministic_project_adaptation",
        "config": {
            "replicas": replicas,
            "sigma": sigma,
            "variance": sigma ** 2,
            "levels": L,
            "step_q": 1.0 / scale,
            "clip_noise": clip_noise,
            "clip_quant": clip_quant,
            "base_seed": base_seed,
            "rng_library": "torch.randn",
            "torch_version": torch.__version__,
        },
        "metadata": {
            "sample_id": sample_id,
            "seed": seed,
            "device": str(target_device),
        },
        "residual_sign": "x_hat - x",
        "output_scale": "normalized_float32",
        "dtype": "float32",
    }
