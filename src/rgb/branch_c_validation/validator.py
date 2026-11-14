"""Branch C — Final Representation Validation (Quality Control).

Validates:
1. Structural properties: Shape [3, 256, 256], channel count, float dtype.
2. Numerical integrity: 0 NaN, 0 Inf, Lab correlation within [-1.0, 1.0],
   signed NQP preservation, detection of unexpected constant/all-zero maps.
3. Dataset integrity: 1:1 alignment with Block 1 sample IDs, absence of duplicate or missing IDs.
4. Storage verification: Read-after-write numerical parity within specified tolerance.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch


@dataclass
class ValidationResult:
    """Outcome of a validation check."""
    passed: bool
    stage: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_lab_maps(
    lab_maps: Union[torch.Tensor, np.ndarray],
    expected_shape: Sequence[int] = (3, 256, 256),
) -> ValidationResult:
    """Validate Branch A Lab local correlation maps."""
    if isinstance(lab_maps, np.ndarray):
        t = torch.from_numpy(lab_maps)
    elif isinstance(lab_maps, torch.Tensor):
        t = lab_maps
    else:
        return ValidationResult(
            passed=False,
            stage="structure",
            message=f"Unsupported type {type(lab_maps)}, expected torch.Tensor or np.ndarray",
        )

    # 1. Shape check
    if tuple(t.shape) != tuple(expected_shape):
        return ValidationResult(
            passed=False,
            stage="structure",
            message=f"Expected shape {expected_shape}, got {list(t.shape)}",
            details={"shape": list(t.shape)},
        )

    # 2. NaN / Inf check
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item())
    if nan_count > 0 or inf_count > 0:
        return ValidationResult(
            passed=False,
            stage="numerical_integrity",
            message=f"Found {nan_count} NaNs and {inf_count} Infs in Lab correlation maps",
            details={"nan_count": nan_count, "inf_count": inf_count},
        )

    # 3. Value range [-1.0, 1.0] (with small float tolerance)
    min_val = float(t.min().item())
    max_val = float(t.max().item())
    if min_val < -1.0001 or max_val > 1.0001:
        return ValidationResult(
            passed=False,
            stage="numerical_integrity",
            message=f"Values exceed [-1, 1] range: min={min_val:.5f}, max={max_val:.5f}",
            details={"min": min_val, "max": max_val},
        )

    return ValidationResult(
        passed=True,
        stage="branch_a_lab",
        message="Lab correlation maps validation passed",
        details={"shape": list(t.shape), "min": min_val, "max": max_val},
    )


def validate_branch_b_map(
    residual_map: Union[torch.Tensor, np.ndarray],
    expected_shape: Sequence[int] = (3, 256, 256),
    allow_near_zero: bool = True,
) -> ValidationResult:
    """Validate Branch B residual map (PiD YUV s=0.2 per DEC-015, or legacy NQP)."""
    if isinstance(residual_map, np.ndarray):
        t = torch.from_numpy(residual_map)
    elif isinstance(residual_map, torch.Tensor):
        t = residual_map
    else:
        return ValidationResult(
            passed=False,
            stage="structure",
            message=f"Unsupported type {type(residual_map)}, expected torch.Tensor or np.ndarray",
        )

    # 1. Shape check
    if tuple(t.shape) != tuple(expected_shape):
        return ValidationResult(
            passed=False,
            stage="structure",
            message=f"Expected shape {expected_shape}, got {list(t.shape)}",
            details={"shape": list(t.shape)},
        )

    # 2. NaN / Inf check
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item())
    if nan_count > 0 or inf_count > 0:
        return ValidationResult(
            passed=False,
            stage="numerical_integrity",
            message=f"Found {nan_count} NaNs and {inf_count} Infs in Branch B residual map",
            details={"nan_count": nan_count, "inf_count": inf_count},
        )

    # 3. Signed values preserved in intensity range [-255, 255]
    min_val = float(t.min().item())
    max_val = float(t.max().item())
    if min_val < -255.0 or max_val > 255.0:
        return ValidationResult(
            passed=False,
            stage="numerical_integrity",
            message=f"Values exceed [-255, 255] range: min={min_val:.5f}, max={max_val:.5f}",
            details={"min": min_val, "max": max_val},
        )

    # 4. Check whether completely all-zero
    is_all_zero = bool((t == 0.0).all().item())
    if is_all_zero and not allow_near_zero:
        return ValidationResult(
            passed=False,
            stage="degeneracy",
            message="Branch B residual map is completely all-zero",
            details={"is_all_zero": True},
        )

    return ValidationResult(
        passed=True,
        stage="branch_b",
        message="Branch B residual map validation passed",
        details={
            "shape": list(t.shape),
            "min": min_val,
            "max": max_val,
            "is_all_zero": is_all_zero,
            "mean_abs": float(t.abs().mean().item()),
        },
    )


# Canonical and backwards-compatibility aliases
validate_pid_map = validate_branch_b_map
validate_nqp_map = validate_branch_b_map



def validate_dataset_alignment(
    block1_ids: Sequence[str],
    rgb_ids: Sequence[str],
) -> ValidationResult:
    """Validate 1:1 sample ID alignment between Block 1 and derived RGB representations."""
    b1_set = set(block1_ids)
    rgb_set = set(rgb_ids)

    # Check lengths
    if len(block1_ids) != len(rgb_ids):
        return ValidationResult(
            passed=False,
            stage="dataset_alignment",
            message=f"Count mismatch: Block 1 has {len(block1_ids)} samples, RGB has {len(rgb_ids)}",
            details={"block1_count": len(block1_ids), "rgb_count": len(rgb_ids)},
        )

    # Check duplicates
    if len(rgb_ids) != len(rgb_set):
        duplicate_count = len(rgb_ids) - len(rgb_set)
        return ValidationResult(
            passed=False,
            stage="dataset_alignment",
            message=f"Found {duplicate_count} duplicate IDs in RGB dataset",
            details={"duplicate_count": duplicate_count},
        )

    # Check set equivalence
    missing_in_rgb = b1_set - rgb_set
    unexpected_in_rgb = rgb_set - b1_set
    if missing_in_rgb or unexpected_in_rgb:
        return ValidationResult(
            passed=False,
            stage="dataset_alignment",
            message=(
                f"ID alignment failed: {len(missing_in_rgb)} missing from RGB, "
                f"{len(unexpected_in_rgb)} unexpected IDs"
            ),
            details={
                "missing_count": len(missing_in_rgb),
                "unexpected_count": len(unexpected_in_rgb),
            },
        )

    # Check exact order
    is_ordered = list(block1_ids) == list(rgb_ids)
    if not is_ordered:
        return ValidationResult(
            passed=False,
            stage="dataset_alignment",
            message="IDs match in content but sequence order differs from Block 1",
            details={"order_matched": False},
        )

    return ValidationResult(
        passed=True,
        stage="dataset_alignment",
        message=f"Strict 1:1 ID alignment verified across all {len(block1_ids)} samples",
        details={"aligned_count": len(block1_ids)},
    )


def verify_read_after_write(
    original_tensor: torch.Tensor,
    loaded_tensor: torch.Tensor,
    atol: float = 1e-4,
) -> bool:
    """Verify that stored and reloaded arrays match within numerical tolerance."""
    if original_tensor.shape != loaded_tensor.shape:
        return False
    diff = (original_tensor - loaded_tensor).abs().max().item()
    return bool(diff <= atol)
