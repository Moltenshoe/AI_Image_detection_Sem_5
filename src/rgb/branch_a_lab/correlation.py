"""Local Pearson correlation map computation for Branch A (Lab Local Correlation Maps).

Research reference:
Juan Pablo Sotelo et al., CHROMA: Detecting AI-Generated Images through
Inter-Channel Color-Space Correlations (2026).

Computes inter-channel Pearson correlation maps across sliding 7x7 windows
for the channel pairs:
    1. L* versus a*
    2. L* versus b*
    3. a* versus b*

NUMERICAL PROPERTIES & SAFETY CONTRACT:
- Output dimensions match input spatial dimensions [3, H, W] via reflect padding (p=3).
- Stride is 1.
- Pearson correlation r_XY is strictly clamped to [-1.0, 1.0].
- In windows with zero or negligible variance (< min_var), correlation is undefined;
  the output is explicitly set to 0.0. Epsilon is used solely for numerical division
  protection and does not create synthetic correlations.
- Output contains 0 NaN and 0 Inf values.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _compute_pair_correlation(
    w1: torch.Tensor,
    w2: torch.Tensor,
    min_var: float = 1e-8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute Pearson correlation between two window batches.

    Args:
        w1: Windows of shape [B, K*K, H*W]
        w2: Windows of shape [B, K*K, H*W]
        min_var: Variance floor below which a window is considered constant/zero-variance.
        eps: Small epsilon for division stability.

    Returns:
        Correlation tensor of shape [B, 1, H*W] in [-1.0, 1.0].
    """
    # Two-pass mean subtraction for numerical stability
    mean1 = w1.mean(dim=1, keepdim=True)
    mean2 = w2.mean(dim=1, keepdim=True)
    w1_c = w1 - mean1
    w2_c = w2 - mean2

    cov = (w1_c * w2_c).sum(dim=1, keepdim=True)
    var1 = (w1_c * w1_c).sum(dim=1, keepdim=True)
    var2 = (w2_c * w2_c).sum(dim=1, keepdim=True)

    denom = torch.sqrt(var1 * var2)

    # Where variance is too small, correlation is undefined -> output exactly 0.0
    zero_mask = (var1 < min_var) | (var2 < min_var) | (denom < eps)

    r = cov / (denom + eps)
    r = torch.clamp(r, -1.0, 1.0)
    r[zero_mask] = 0.0
    return r


def compute_local_correlation_maps(
    lab_tensor: torch.Tensor,
    window_size: int = 7,
    min_var: float = 1e-8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute 7x7 sliding window Pearson correlation maps for CIE Lab channels.

    Args:
        lab_tensor: CIE Lab image tensor with 3 channels (L, a, b):
            - Shape [3, H, W]
            - Or shape [B, 3, H, W]
        window_size: Size of square sliding window (default: 7).
        min_var: Minimum window variance threshold for valid correlation (default: 1e-8).
        eps: Division protection epsilon (default: 1e-12).

    Returns:
        Tensor of shape [3, H, W] or [B, 3, H, W] with channels:
            Channel 0: Pearson correlation r(L, a)
            Channel 1: Pearson correlation r(L, b)
            Channel 2: Pearson correlation r(a, b)
        Values are bounded in [-1.0, 1.0].

    Raises:
        ValueError: If input tensor does not have 3 channels or window_size is even.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    pad = window_size // 2
    is_3d = False
    if lab_tensor.ndim == 3:
        if lab_tensor.shape[0] != 3:
            raise ValueError(f"Expected 3 channels in dim 0 for [3, H, W], got {lab_tensor.shape}")
        lab = lab_tensor.unsqueeze(0)
        is_3d = True
    elif lab_tensor.ndim == 4:
        if lab_tensor.shape[1] != 3:
            raise ValueError(f"Expected 3 channels in dim 1 for [B, 3, H, W], got {lab_tensor.shape}")
        lab = lab_tensor
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {lab_tensor.shape}")

    b, c, h, w = lab.shape
    device = lab.device
    dtype = lab.dtype

    # Reflect padding preserves border statistics gracefully
    padded = F.pad(lab, (pad, pad, pad, pad), mode="reflect")

    # Unfold each channel independently: [B, K*K, H*W]
    l_win = F.unfold(padded[:, 0:1], kernel_size=window_size)
    a_win = F.unfold(padded[:, 1:2], kernel_size=window_size)
    b_win = F.unfold(padded[:, 2:3], kernel_size=window_size)

    # Compute pairwise Pearson correlations
    r_La = _compute_pair_correlation(l_win, a_win, min_var=min_var, eps=eps)
    r_Lb = _compute_pair_correlation(l_win, b_win, min_var=min_var, eps=eps)
    r_ab = _compute_pair_correlation(a_win, b_win, min_var=min_var, eps=eps)

    # Concatenate along channel dimension: [B, 3, H*W] -> view [B, 3, H, W]
    corr_maps = torch.cat([r_La, r_Lb, r_ab], dim=1).view(b, 3, h, w)

    # Safety assertion: zero NaNs or Infs
    corr_maps = torch.nan_to_num(corr_maps, nan=0.0, posinf=0.0, neginf=0.0)

    if is_3d:
        return corr_maps.squeeze(0)
    return corr_maps
