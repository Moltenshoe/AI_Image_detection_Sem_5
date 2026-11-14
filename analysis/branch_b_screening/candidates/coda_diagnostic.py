"""Candidate D: CoDA 2026 Analytical Diagnostic Model.

Retained strictly as a mathematical diagnostic comparing Algorithm 1
(raw 8-bit integer probe) against Section IV-C Equation 12 (normalized unit-step
Fourier periodic bias model).

IMPORTANT: Strictly diagnostic only; NOT a candidate for Block 2 dataset materialization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch


def coda_algorithm_1(
    image: torch.Tensor,
    sigma: float = 0.10,
    replicas: int = 50,
    seed: int = 42,
) -> torch.Tensor:
    """Evaluate CoDA Algorithm 1 formulation.

    Formula:
        Y_r = clip(round(I + N_r), 0, 255),  N_r ~ N(0, sigma^2)
        I_hat = (1/R) * sum(Y_r)
        Delta = I - I_hat

    On exact interior integers with sigma = 0.10, symmetry dictates E[Delta] = 0.

    Args:
        image: Input tensor in raw scale [0, 255] (can be integer or fractional).
        sigma: Noise standard deviation in raw intensity units (default: 0.10).
        replicas: Monte Carlo replica count (default: 50).
        seed: Random seed for repeatability.

    Returns:
        torch.Tensor: Residual tensor Delta = I - I_hat of same shape.
    """
    gen = torch.Generator(device=image.device)
    gen.manual_seed(seed)

    I_float = image.to(dtype=torch.float32)
    acc = torch.zeros_like(I_float)

    for _ in range(replicas):
        noise = torch.randn(I_float.shape, generator=gen, device=image.device) * sigma
        y = torch.clamp(torch.round(I_float + noise), 0.0, 255.0)
        acc.add_(y)

    I_hat = acc / float(replicas)
    return (I_float - I_hat).to(dtype=torch.float32)


def coda_equation_12(
    x: torch.Tensor,
    sigma: float = 0.10,
) -> torch.Tensor:
    """Evaluate CoDA Section IV-C Equation 12 periodic Fourier bias model.

    Formula:
        B(x; sigma) = (1 / pi) * exp(-2 * pi^2 * sigma^2) * sin(2 * pi * x)

    This models the expected periodic quantization bias of a unit-step rounder
    on a normalized continuous variable x in [0, 1).

    Args:
        x: Normalized coordinate tensor in [0, 1).
        sigma: Noise standard deviation.

    Returns:
        torch.Tensor: Expected theoretical periodic bias B(x; sigma).
    """
    x_float = x.to(dtype=torch.float32)
    attenuation = math.exp(-2.0 * (math.pi ** 2) * (sigma ** 2))
    bias = (1.0 / math.pi) * attenuation * torch.sin(2.0 * math.pi * x_float)
    return bias.to(dtype=torch.float32)


def compare_coda_formulations(
    integer_input: Optional[torch.Tensor] = None,
    fractional_input: Optional[torch.Tensor] = None,
    sigma: float = 0.10,
    replicas: int = 50,
) -> Dict[str, Any]:
    """Execute analytical diagnostic comparing Algorithm 1 and Equation 12.

    Demonstrates that:
    1. On integer interior inputs, Algorithm 1 expected residual is identically zero.
    2. On fractional inputs (e.g. k + 0.5), Algorithm 1 produces active residuals.
    3. Equation 12 is a separate theoretical unit-step model on [0, 1), not the
       normalized equivalent of Algorithm 1.
    """
    if integer_input is None:
        # Default diagnostic integer tensor: interior values and boundaries
        integer_input = torch.tensor([0.0, 10.0, 128.0, 200.0, 255.0], dtype=torch.float32)
    if fractional_input is None:
        # Default diagnostic fractional tensor: half-integer midpoints
        fractional_input = torch.tensor([0.5, 10.5, 128.5, 200.5, 254.5], dtype=torch.float32)

    # 1. Algorithm 1 on integer inputs
    res_int = coda_algorithm_1(integer_input, sigma=sigma, replicas=replicas)

    # 2. Algorithm 1 on fractional inputs
    res_frac = coda_algorithm_1(fractional_input, sigma=sigma, replicas=replicas)

    # 3. Equation 12 on normalized inputs x = I / 256.0
    x_int_norm = integer_input / 256.0
    x_frac_norm = fractional_input / 256.0
    eq12_int = coda_equation_12(x_int_norm, sigma=sigma)
    eq12_frac = coda_equation_12(x_frac_norm, sigma=sigma)

    return {
        "diagnostic_name": "coda_formulation_comparison",
        "is_materialization_candidate": False,
        "config": {
            "sigma": sigma,
            "replicas": replicas,
        },
        "algorithm_1_integer_mean_abs": float(res_int.abs().mean().item()),
        "algorithm_1_fractional_mean_abs": float(res_frac.abs().mean().item()),
        "equation_12_integer_max_abs": float(eq12_int.abs().max().item()),
        "equation_12_fractional_max_abs": float(eq12_frac.abs().max().item()),
        "conclusion": (
            "Algorithm 1 expected residual is zero on interior integers away from boundaries, "
            "whereas Equation 12 models a unit-step rounder on [0, 1) and is not equivalent to "
            "evaluating raw 8-bit integers."
        ),
    }
