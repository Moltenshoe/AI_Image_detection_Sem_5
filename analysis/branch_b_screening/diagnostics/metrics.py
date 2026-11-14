"""Comprehensive Scale-Aware Numerical Screening Metrics.

Computes multi-faceted dispersion, sparsity, dynamic range, and stability metrics
across candidate residual representations, enforcing anti-collapse evaluation
per individual image as well as across classes and generators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

TOLERANCE_NORMALIZED = 1.0e-6
TOLERANCE_RAW = 1.0e-4


def get_domain_tolerance(is_normalized: bool) -> float:
    """Return numerical noise tolerance based on representation domain."""
    return TOLERANCE_NORMALIZED if is_normalized else TOLERANCE_RAW


def compute_scalar_metrics(
    residual: Union[torch.Tensor, np.ndarray],
    is_normalized: bool = False,
    rounding_decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute comprehensive scalar metrics on a residual tensor or array.

    Args:
        residual: Residual array of any shape (typically [3, H, W] or flat).
        is_normalized: True if residual is in normalized float scale [-1, 1];
                      False if in raw intensity scale [-255, 255].
        rounding_decimals: Decimals for unique-value cardinality count.
                          Default: 6 for normalized, 4 for raw.

    Returns:
        Dict containing all required Level A and Level B scalar metrics.
    """
    if isinstance(residual, torch.Tensor):
        arr = residual.detach().cpu().numpy()
    else:
        arr = np.asarray(residual)

    tau = get_domain_tolerance(is_normalized)
    if rounding_decimals is None:
        rounding_decimals = 6 if is_normalized else 4

    flat = arr.flatten().astype(np.float64)
    total_scalars = len(flat)

    # 1. Finite value counts
    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())
    finite_count = int(np.isfinite(flat).sum())

    if finite_count == 0:
        return {
            "total_scalars": total_scalars,
            "finite_count": 0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "is_degenerate": True,
            "degeneracy_reason": "all_non_finite",
        }

    valid = flat[np.isfinite(flat)]
    abs_valid = np.abs(valid)

    # 2. Zero and sparsity fractions
    exact_zero_count = int((valid == 0.0).sum())
    exact_zero_fraction = float(exact_zero_count) / float(total_scalars)

    tolerance_zero_count = int((abs_valid <= tau).sum())
    tolerance_zero_fraction = float(tolerance_zero_count) / float(total_scalars)

    # 3. Magnitude and dispersion
    mar = float(np.mean(abs_valid))
    rms = float(np.sqrt(np.mean(valid ** 2)))
    max_ar = float(np.max(abs_valid))
    min_val = float(np.min(valid))
    max_val = float(np.max(valid))
    dynamic_range = max_val - min_val
    variance = float(np.var(valid))

    # 4. Quantiles (p50, p95, p99 of absolute residual)
    p50 = float(np.percentile(abs_valid, 50))
    p95 = float(np.percentile(abs_valid, 95))
    p99 = float(np.percentile(abs_valid, 99))

    # 5. Sign proportions
    pos_count = int((valid > tau).sum())
    neg_count = int((valid < -tau).sum())
    pos_proportion = float(pos_count) / float(total_scalars)
    neg_proportion = float(neg_count) / float(total_scalars)
    zero_proportion = tolerance_zero_fraction

    # 6. Unique values count
    rounded_valid = np.round(valid, decimals=rounding_decimals)
    unique_value_count = int(len(np.unique(rounded_valid)))

    # 7. Scale-Aware Degeneracy Check
    is_degenerate = (
        (dynamic_range <= tau)
        or (max_ar <= tau)
        or (tolerance_zero_fraction >= 1.0)
    )
    degeneracy_reasons = []
    if dynamic_range <= tau:
        degeneracy_reasons.append(f"collapsed_dynamic_range(DR={dynamic_range:.2e}<=tau)")
    if max_ar <= tau:
        degeneracy_reasons.append(f"insignificant_peak(MaxAR={max_ar:.2e}<=tau)")
    if tolerance_zero_fraction >= 1.0:
        degeneracy_reasons.append("100%_within_numerical_tolerance")

    return {
        "total_scalars": total_scalars,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "exact_zero_count": exact_zero_count,
        "exact_zero_fraction": exact_zero_fraction,
        "tolerance_zero_count": tolerance_zero_count,
        "tolerance_zero_fraction": tolerance_zero_fraction,
        "nonzero_scalar_fraction": 1.0 - tolerance_zero_fraction,
        "mar": mar,
        "rms": rms,
        "max_ar": max_ar,
        "min_val": min_val,
        "max_val": max_val,
        "dynamic_range": dynamic_range,
        "variance": variance,
        "p50_abs": p50,
        "p95_abs": p95,
        "p99_abs": p99,
        "pos_proportion": pos_proportion,
        "neg_proportion": neg_proportion,
        "zero_proportion": zero_proportion,
        "unique_value_count": unique_value_count,
        "rounding_decimals": rounding_decimals,
        "tolerance": tau,
        "is_degenerate": is_degenerate,
        "degeneracy_reasons": degeneracy_reasons,
    }


def compute_per_image_metrics(
    residual: Union[torch.Tensor, np.ndarray],
    is_normalized: bool = False,
) -> Dict[str, Any]:
    """Compute per-image metrics including channel-wise and pixel-wise nonzero rates.

    Args:
        residual: 3D residual tensor [3, H, W].
        is_normalized: True if normalized float; False if raw integer scale.

    Returns:
        Dict with overall image metrics, per-channel metrics, and pixel nonzero rate.
    """
    if isinstance(residual, torch.Tensor):
        arr = residual.detach().cpu().numpy()
    else:
        arr = np.asarray(residual)

    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected residual shape [3, H, W], got {arr.shape}")

    tau = get_domain_tolerance(is_normalized)
    H, W = arr.shape[1], arr.shape[2]
    total_pixels = H * W

    # Pixel nonzero rate: location (y, x) is nonzero if ANY channel has |Delta| > tau
    max_channel_abs = np.max(np.abs(arr), axis=0)  # [H, W]
    nonzero_pixels = int((max_channel_abs > tau).sum())
    pixel_nonzero_fraction = float(nonzero_pixels) / float(total_pixels)

    # Combined scalar metrics across all channels
    combined_scalar = compute_scalar_metrics(arr, is_normalized=is_normalized)

    # Per-channel scalar metrics
    channel_names = ["R", "G", "B"]
    per_channel = {}
    for c_idx, c_name in enumerate(channel_names):
        per_channel[c_name] = compute_scalar_metrics(
            arr[c_idx], is_normalized=is_normalized
        )

    return {
        "pixel_nonzero_count": nonzero_pixels,
        "total_pixels": total_pixels,
        "pixel_nonzero_fraction": pixel_nonzero_fraction,
        "combined": combined_scalar,
        "per_channel": per_channel,
        "is_degenerate": combined_scalar["is_degenerate"],
        "max_ar": combined_scalar["max_ar"],
        "dynamic_range": combined_scalar["dynamic_range"],
        "rms": combined_scalar["rms"],
        "variance": combined_scalar["variance"],
        "tolerance_zero_fraction": combined_scalar["tolerance_zero_fraction"],
    }


def assess_numerical_degeneracy(
    per_image_results: List[Dict[str, Any]],
    tau: float,
) -> Dict[str, Any]:
    """Assess whether a representation has collapsed across the sample set.

    Guarantees anti-collapse: A representation is NOT declared non-degenerate
    merely because a single image or outlier produced a nonzero value.
    """
    num_images = len(per_image_results)
    if num_images == 0:
        return {"status": "NO_DATA", "is_acceptable": False}

    degenerate_images = 0
    drs = []
    max_ars = []
    rms_vals = []
    zero_fractions = []

    for img_res in per_image_results:
        comb = img_res["combined"]
        if comb["is_degenerate"]:
            degenerate_images += 1
        drs.append(comb["dynamic_range"])
        max_ars.append(comb["max_ar"])
        rms_vals.append(comb["rms"])
        zero_fractions.append(comb["tolerance_zero_fraction"])

    pct_degenerate = float(degenerate_images) / float(num_images)

    # A representation is collapsed if more than 50% of images are individually degenerate
    is_acceptable = pct_degenerate < 0.05  # At least 95% of images must be active

    return {
        "total_images_evaluated": num_images,
        "degenerate_image_count": degenerate_images,
        "degenerate_image_percentage": pct_degenerate * 100.0,
        "median_dynamic_range": float(np.median(drs)),
        "median_max_ar": float(np.median(max_ars)),
        "median_rms": float(np.median(rms_vals)),
        "median_zero_fraction": float(np.median(zero_fractions)),
        "is_acceptable": is_acceptable,
        "tolerance": tau,
    }


def aggregate_screening_metrics(
    image_records: List[Dict[str, Any]],
    is_normalized: bool = False,
) -> Dict[str, Any]:
    """Aggregate per-image screening metrics by class, generator, and overall.

    Args:
        image_records: List of dicts, each containing:
            'sample_idx', 'label_a', 'label_b', 'generator_name',
            'metrics' (output of compute_per_image_metrics),
            'morans_i', 'gini', 'runtime_ms'

    Returns:
        Structured dictionary of overall and grouped screening statistics.
    """
    tau = get_domain_tolerance(is_normalized)
    total_imgs = len(image_records)

    overall_per_image = [r["metrics"] for r in image_records]
    degeneracy_assessment = assess_numerical_degeneracy(overall_per_image, tau=tau)

    # Group by generator_name / label_b
    generators: Dict[str, List[Dict[str, Any]]] = {}
    for r in image_records:
        gen = r.get("generator_name", f"gen_{r.get('label_b')}")
        generators.setdefault(gen, []).append(r)

    gen_summaries = {}
    for gen, recs in generators.items():
        mar_list = [r["metrics"]["combined"]["mar"] for r in recs]
        rms_list = [r["metrics"]["combined"]["rms"] for r in recs]
        max_ar_list = [r["metrics"]["combined"]["max_ar"] for r in recs]
        zero_list = [r["metrics"]["combined"]["tolerance_zero_fraction"] for r in recs]
        pixel_nz_list = [r["metrics"]["pixel_nonzero_fraction"] for r in recs]

        morans_vals = [
            r["morans_i"]["morans_i"]
            for r in recs
            if r.get("morans_i") and r["morans_i"].get("morans_i") is not None
        ]
        gini_vals = [
            r["gini"]["gini"]
            for r in recs
            if r.get("gini") and r["gini"].get("gini") is not None
        ]

        gen_summaries[gen] = {
            "sample_count": len(recs),
            "mean_mar": float(np.mean(mar_list)),
            "mean_rms": float(np.mean(rms_list)),
            "mean_max_ar": float(np.mean(max_ar_list)),
            "mean_zero_scalar_pct": float(np.mean(zero_list)) * 100.0,
            "mean_nonzero_pixel_pct": float(np.mean(pixel_nz_list)) * 100.0,
            "mean_morans_i": float(np.mean(morans_vals)) if morans_vals else None,
            "mean_gini": float(np.mean(gini_vals)) if gini_vals else None,
        }

    # Overall metrics across all images
    all_mars = [r["metrics"]["combined"]["mar"] for r in image_records]
    all_rms = [r["metrics"]["combined"]["rms"] for r in image_records]
    all_max_ars = [r["metrics"]["combined"]["max_ar"] for r in image_records]
    all_pixel_nz = [r["metrics"]["pixel_nonzero_fraction"] for r in image_records]
    all_zero_scalars = [r["metrics"]["combined"]["tolerance_zero_fraction"] for r in image_records]
    all_runtimes = [r["runtime_ms"] for r in image_records if "runtime_ms" in r]

    all_morans = [
        r["morans_i"]["morans_i"]
        for r in image_records
        if r.get("morans_i") and r["morans_i"].get("morans_i") is not None
    ]
    all_ginis = [
        r["gini"]["gini"]
        for r in image_records
        if r.get("gini") and r["gini"].get("gini") is not None
    ]

    return {
        "total_images": total_imgs,
        "is_normalized": is_normalized,
        "tolerance": tau,
        "degeneracy_assessment": degeneracy_assessment,
        "aggregate_summary": {
            "mean_mar": float(np.mean(all_mars)),
            "mean_rms": float(np.mean(all_rms)),
            "mean_max_ar": float(np.mean(all_max_ars)),
            "mean_zero_scalar_percentage": float(np.mean(all_zero_scalars)) * 100.0,
            "mean_nonzero_pixel_percentage": float(np.mean(all_pixel_nz)) * 100.0,
            "mean_morans_i": float(np.mean(all_morans)) if all_morans else None,
            "mean_gini": float(np.mean(all_ginis)) if all_ginis else None,
            "mean_runtime_ms_per_image": float(np.mean(all_runtimes)) if all_runtimes else None,
        },
        "generator_summaries": gen_summaries,
    }
