"""Intensity-Conditioned Residual Statistics.

Computes the conditional expectation and variance of residual values across
16 input intensity bins to disentangle boundary clipping from interior quantization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def compute_intensity_conditioned_profile(
    image: Union[torch.Tensor, np.ndarray],
    residual: Union[torch.Tensor, np.ndarray],
    is_normalized: bool = False,
    num_bins: int = 16,
) -> Dict[str, Any]:
    """Compute E[Delta | I in B_k] and Var(Delta | I in B_k) across 16 intensity bins.

    Args:
        image: Original input image [3, H, W] (normalized [0, 1] or raw [0, 255]).
        residual: Residual map [3, H, W] of matching dimensions.
        is_normalized: True if input is in [0, 1]; False if in [0, 255].
        num_bins: Number of intensity intervals (default: 16).

    Returns:
        Dict containing per-bin statistics per channel and combined.
        Empty bins are represented explicitly as None.
    """
    if isinstance(image, torch.Tensor):
        img_arr = image.detach().cpu().numpy()
    else:
        img_arr = np.asarray(image)

    if isinstance(residual, torch.Tensor):
        res_arr = residual.detach().cpu().numpy()
    else:
        res_arr = np.asarray(residual)

    if img_arr.shape != res_arr.shape:
        raise ValueError(
            f"Shape mismatch between image {img_arr.shape} and residual {res_arr.shape}"
        )

    # Define bin edges
    if is_normalized:
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    else:
        bin_edges = np.linspace(0.0, 256.0, num_bins + 1)

    bins_list = []
    for k in range(num_bins):
        low = float(bin_edges[k])
        high = float(bin_edges[k + 1])
        bins_list.append((low, high))

    channel_names = ["R", "G", "B"]
    profile: Dict[str, Any] = {
        "is_normalized": is_normalized,
        "num_bins": num_bins,
        "bins": [],
    }

    # Combined channels and per-channel
    for k, (low, high) in enumerate(bins_list):
        bin_data: Dict[str, Any] = {
            "bin_index": k,
            "range_low": low,
            "range_high": high,
            "combined": {},
            "per_channel": {},
        }

        # Combined across all 3 channels
        if k == num_bins - 1:
            # Include upper boundary in last bin
            mask_comb = (img_arr >= low) & (img_arr <= high)
        else:
            mask_comb = (img_arr >= low) & (img_arr < high)

        count_comb = int(np.sum(mask_comb))
        bin_data["combined"]["count"] = count_comb
        if count_comb > 0:
            res_in_bin = res_arr[mask_comb]
            bin_data["combined"]["mean"] = float(np.mean(res_in_bin))
            bin_data["combined"]["var"] = float(np.var(res_in_bin))
            bin_data["combined"]["mar"] = float(np.mean(np.abs(res_in_bin)))
            bin_data["combined"]["available"] = True
        else:
            bin_data["combined"]["mean"] = None
            bin_data["combined"]["var"] = None
            bin_data["combined"]["mar"] = None
            bin_data["combined"]["available"] = False

        # Per-channel
        for c_idx, c_name in enumerate(channel_names):
            img_c = img_arr[c_idx]
            res_c = res_arr[c_idx]

            if k == num_bins - 1:
                mask_c = (img_c >= low) & (img_c <= high)
            else:
                mask_c = (img_c >= low) & (img_c < high)

            count_c = int(np.sum(mask_c))
            c_data: Dict[str, Any] = {"count": count_c}
            if count_c > 0:
                res_c_bin = res_c[mask_c]
                c_data["mean"] = float(np.mean(res_c_bin))
                c_data["var"] = float(np.var(res_c_bin))
                c_data["mar"] = float(np.mean(np.abs(res_c_bin)))
                c_data["available"] = True
            else:
                c_data["mean"] = None
                c_data["var"] = None
                c_data["mar"] = None
                c_data["available"] = False

            bin_data["per_channel"][c_name] = c_data

        profile["bins"].append(bin_data)

    return profile


def format_intensity_profile_table(profile: Dict[str, Any]) -> str:
    """Format intensity profile as a readable Markdown table."""
    lines = [
        "| Bin | Range | Count | Mean Residual | Variance | Mean Abs Residual |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for b in profile["bins"]:
        k = b["bin_index"]
        rng = f"[{b['range_low']:.2f}, {b['range_high']:.2f})"
        cnt = b["combined"]["count"]
        if b["combined"]["available"]:
            m = f"{b['combined']['mean']:+.6f}"
            v = f"{b['combined']['var']:.6e}"
            mar = f"{b['combined']['mar']:.6f}"
        else:
            m, v, mar = "N/A", "N/A", "N/A"
        lines.append(f"| B{k:02d} | {rng} | {cnt:,} | {m} | {v} | {mar} |")
    return "\n".join(lines)
