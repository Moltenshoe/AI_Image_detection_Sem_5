"""Numerically Stable Spatial Metrics for Residual Maps.

Implements direct neighbour-based Moran's I and O(N log N) sorted Gini coefficient
without constructing dense N x N adjacency matrices.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch


def compute_morans_i(
    residual_map: Union[torch.Tensor, np.ndarray],
) -> Dict[str, Any]:
    """Compute Moran's I spatial autocorrelation using row-standardized 4-neighbour weights.

    Evaluated directly in O(HW) time and memory via shifted spatial lag accumulation,
    avoiding the creation of a dense HW x HW weight matrix.

    Formula:
        w_ij* = w_ij / k_i  (row-standardized weights where k_i is neighbor count)
        S_0 = sum_{i,j} w_ij* = N
        I = (N / S_0) * [ sum_i z_i * (sum_{j in N(i)} z_j / k_i) ] / sum_i z_i^2
          = [ sum_i z_i * z_lag,i ] / sum_i z_i^2
    where z_i = x_i - x_bar and z_lag,i = (1 / k_i) * sum_{j in N(i)} z_j.

    Natural boundary conditions are respected:
        k_i = 4 for interior pixels
        k_i = 3 for edge pixels (non-corner)
        k_i = 2 for corner pixels

    Args:
        residual_map: 2D array [H, W] or 3D array [C, H, W] representing spatial residual.

    Returns:
        Dict with 'morans_i' (float or None), 'status', and spatial metadata.
    """
    if isinstance(residual_map, torch.Tensor):
        arr = residual_map.detach().cpu().numpy()
    else:
        arr = np.asarray(residual_map)

    if arr.ndim == 3:
        # If [C, H, W], compute across channel average
        arr = np.mean(arr, axis=0)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D map [H, W], got shape {arr.shape}")

    H, W = arr.shape
    N = float(H * W)

    # Check finite
    if not np.all(np.isfinite(arr)):
        return {
            "morans_i": None,
            "status": "undefined_non_finite_values",
        }

    # Variance check (constant or all-zero map)
    mean_val = float(np.mean(arr))
    z = (arr - mean_val).astype(np.float64)
    ss = float(np.sum(z ** 2))

    if ss <= 1e-12:
        return {
            "morans_i": None,
            "status": "undefined_constant_or_zero_variance",
            "variance": ss / N,
        }

    # Direct 4-connectivity neighbor accumulation and degree count in O(HW)
    neighbor_sum = np.zeros((H, W), dtype=np.float64)
    k = np.zeros((H, W), dtype=np.float64)

    if H > 1:
        # Vertical neighbors
        neighbor_sum[:-1, :] += z[1:, :]
        k[:-1, :] += 1.0
        neighbor_sum[1:, :] += z[:-1, :]
        k[1:, :] += 1.0

    if W > 1:
        # Horizontal neighbors
        neighbor_sum[:, :-1] += z[:, 1:]
        k[:, :-1] += 1.0
        neighbor_sum[:, 1:] += z[:, :-1]
        k[:, 1:] += 1.0

    # Spatial lag z_lag = (1 / k_i) * sum_{j in N(i)} z_j
    spatial_lag = np.zeros_like(z)
    mask = k > 0
    spatial_lag[mask] = neighbor_sum[mask] / k[mask]

    # Row-standardized Moran's I
    numerator = float(np.sum(z * spatial_lag))
    morans_i = numerator / ss

    return {
        "morans_i": float(morans_i),
        "status": "valid",
        "spatial_shape": [H, W],
        "weight_type": "row_standardized_4_nearest_neighbors",
    }


def compute_gini_coefficient(
    residual_map: Union[torch.Tensor, np.ndarray],
) -> Dict[str, Any]:
    """Compute Gini spatial concentration coefficient via O(N log N) sorting.

    Measures spatial energy inequality across pixel locations:
        G = [ 2 * sum_{i=1}^N (i * |z|_(i)) ] / [ N * sum_{i=1}^N |z|_(i) ] - (N + 1) / N
    where |z|_(i) are the sorted absolute values.

    Args:
        residual_map: 2D or 3D residual tensor.

    Returns:
        Dict with 'gini' (float or None) and status.
    """
    if isinstance(residual_map, torch.Tensor):
        arr = residual_map.detach().cpu().numpy()
    else:
        arr = np.asarray(residual_map)

    vals = np.abs(arr.flatten())

    if not np.all(np.isfinite(vals)):
        return {"gini": None, "status": "undefined_non_finite_values"}

    total_sum = float(np.sum(vals))
    if total_sum <= 1e-12:
        return {"gini": 0.0, "status": "all_zero_residual"}

    N = len(vals)
    sorted_vals = np.sort(vals)
    ranks = np.arange(1, N + 1, dtype=np.float64)

    weighted_sum = float(np.sum(ranks * sorted_vals))
    gini = (2.0 * weighted_sum) / (float(N) * total_sum) - float(N + 1) / float(N)

    return {
        "gini": float(np.clip(gini, 0.0, 1.0)),
        "status": "valid",
        "total_elements": N,
    }
