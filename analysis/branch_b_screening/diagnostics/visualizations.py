"""Visualization Utilities for Residual Maps and Profiles.

Generates diagnostic plots including residual spatial heatmaps and
intensity-conditioned expectation curves for reporting.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_residual_spatial_plot(
    original_rgb: Union[torch.Tensor, np.ndarray],
    residual: Union[torch.Tensor, np.ndarray],
    output_png: str,
    title: str = "Residual Spatial Map",
) -> str:
    """Save a side-by-side visualization of original image and residual channels.

    Args:
        original_rgb: RGB image [3, H, W] in [0, 255] or [0, 1].
        residual: Residual map [3, H, W].
        output_png: Path to save the PNG file.
        title: Plot title.

    Returns:
        str: Absolute path of saved plot.
    """
    if isinstance(original_rgb, torch.Tensor):
        orig = original_rgb.detach().cpu().numpy()
    else:
        orig = np.asarray(original_rgb)

    if isinstance(residual, torch.Tensor):
        res = residual.detach().cpu().numpy()
    else:
        res = np.asarray(residual)

    # Normalize original for display
    if orig.max() > 1.0:
        orig_disp = np.clip(orig / 255.0, 0.0, 1.0)
    else:
        orig_disp = np.clip(orig, 0.0, 1.0)
    orig_disp = np.transpose(orig_disp, (1, 2, 0))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, fontsize=14)

    # 1. Original image
    axes[0].imshow(orig_disp)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")

    # 2. Residual channels
    channels = ["R Residual", "G Residual", "B Residual"]
    cmaps = ["Reds", "Greens", "Blues"]

    for i in range(3):
        v = res[i]
        vmax = max(float(np.max(np.abs(v))), 1e-4)
        im = axes[i + 1].imshow(v, cmap=cmaps[i], vmin=-vmax, vmax=vmax)
        axes[i + 1].set_title(channels[i])
        axes[i + 1].axis("off")
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    return os.path.abspath(output_png)


def save_intensity_profile_plot(
    profile: Dict[str, Any],
    output_png: str,
    title: str = "Intensity-Conditioned Residual Profile",
) -> str:
    """Plot E[Delta | I in B_k] across intensity intervals."""
    bins = profile["bins"]
    bin_centers = [0.5 * (b["range_low"] + b["range_high"]) for b in bins]

    comb_means = [b["combined"]["mean"] if b["combined"]["available"] else np.nan for b in bins]
    r_means = [b["per_channel"]["R"]["mean"] if b["per_channel"]["R"]["available"] else np.nan for b in bins]
    g_means = [b["per_channel"]["G"]["mean"] if b["per_channel"]["G"]["available"] else np.nan for b in bins]
    b_means = [b["per_channel"]["B"]["mean"] if b["per_channel"]["B"]["available"] else np.nan for b in bins]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bin_centers, comb_means, "k-o", label="Combined", linewidth=2)
    ax.plot(bin_centers, r_means, "r--s", label="Red", alpha=0.7)
    ax.plot(bin_centers, g_means, "g--^", label="Green", alpha=0.7)
    ax.plot(bin_centers, b_means, "b--d", label="Blue", alpha=0.7)

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Input Intensity (Bin Center)")
    ax.set_ylabel("Expected Residual E[Delta | I in B_k]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    return os.path.abspath(output_png)
