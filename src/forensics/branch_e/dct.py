"""8×8 Block DCT coefficient fingerprint feature extraction (Block 2 — Branch E candidate: E1_DCT).

Mathematical & Forensic Formulation
-----------------------------------
Input:
    Canonical RGB image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

Grayscale conversion (ITU-R BT.601):
    Y = 0.299 * R + 0.587 * G + 0.114 * B,  Y in [0, 1]

Block Partitioning:
    Non-overlapping 8×8 blocks. For a 256×256 canonical image, this yields
    exactly (256/8) × (256/8) = 32 × 32 = 1,024 blocks.

2D Orthonormal DCT-II:
    For an 8×8 block X, D = T @ X @ T.T, where T is the 8×8 orthonormal DCT-II matrix:
        T[0, j] = 1 / sqrt(8)
        T[i, j] = sqrt(2/8) * cos(pi * (2j + 1) * i / 16) for i > 0

    D[0, 0] is the DC coefficient (mean block illumination).
    The remaining 63 coefficients D[u, v] for (u, v) != (0, 0) are the AC coefficients.

Zonal AC Frequency Partition:
    - Low-frequency AC  : 1 <= u + v <= 3 (9 basis frequencies)
    - Mid-frequency AC  : 4 <= u + v <= 7 (24 basis frequencies)
    - High-frequency AC : u + v >= 8      (30 basis frequencies)

Directional AC Partition:
    - Horizontal AC : u < v, (u, v) != (0, 0) (higher horizontal frequency variation)
    - Vertical AC   : u > v, (u, v) != (0, 0) (higher vertical frequency variation)

Candidate Features (10 scalars — prefix: dct_*):
------------------------------------------------
    1. dct_ac_mean_abs    : Mean absolute magnitude across all 1024 × 63 AC coefficients.
    2. dct_ac_energy      : Mean squared energy across all 1024 × 63 AC coefficients.
    3. dct_ac_kurtosis    : Fisher excess kurtosis of the AC coefficient distribution.
    4. dct_sparsity_ratio : Fraction of canonical float AC coefficients with |D| < 1e-3.
    5. dct_low_freq_ratio : Ratio of low-frequency AC energy to total AC energy.
    6. dct_mid_freq_ratio : Ratio of mid-frequency AC energy to total AC energy.
    7. dct_high_freq_ratio: Ratio of high-frequency AC energy to total AC energy.
    8. dct_anisotropy     : Directional energy asymmetry |E_horiz - E_vert| / (E_horiz + E_vert + eps).
    9. dct_benford_ssd    : Sum of squared deviations between empirical first-digit distribution
                            of rounded AC coefficients and Benford's Law P(d) = log10(1 + 1/d).
    10. dct_block_var_mean: Mean across all 1024 blocks of intra-block AC coefficient variance.

CRITICAL SCIENTIFIC QUALIFICATIONS
----------------------------------
1. Canonical Data Reality: The input is already decoded and resized (cv2.INTER_AREA to 256×256).
   The 8×8 DCT computed here is a basis projection of the canonical image, NOT a recovery of the
   camera's original JPEG DCT stream.
2. Sparsity Meaning: 'dct_sparsity_ratio' measures near-zero canonical floating-point DCT values,
   NOT original integer JPEG quantization dead-zones.
3. Benford SSD: Treated strictly as an experimental candidate descriptor, not a guaranteed property.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Orthonormal 8×8 DCT-II basis matrix construction
# ---------------------------------------------------------------------------

def _build_dct8_matrix() -> torch.Tensor:
    """Build the 8×8 orthonormal DCT-II transform matrix T.

    Returns:
        Tensor of shape [8, 8], float32.
    """
    t = torch.zeros((8, 8), dtype=torch.float32)
    c0 = 1.0 / math.sqrt(8.0)
    c1 = math.sqrt(2.0 / 8.0)
    for i in range(8):
        for j in range(8):
            if i == 0:
                t[i, j] = c0
            else:
                t[i, j] = c1 * math.cos(math.pi * (2 * j + 1) * i / 16.0)
    return t


# Precomputed static constant matrix
_DCT8_MATRIX: torch.Tensor = _build_dct8_matrix()

# Theoretical Benford's First Digit Law probabilities P(d) = log10(1 + 1/d) for d in 1..9
_BENFORD_PROBS: torch.Tensor = torch.tensor(
    [math.log10(1.0 + 1.0 / d) for d in range(1, 10)],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Zonal Masks for 8×8 AC bases
# ---------------------------------------------------------------------------

def _build_ac_masks() -> Dict[str, torch.Tensor]:
    """Build binary masks for zonal and directional AC basis frequency partitions.

    Returns:
        Dictionary of boolean tensors of shape [8, 8].
    """
    masks = {}
    ac_mask = torch.ones((8, 8), dtype=torch.bool)
    ac_mask[0, 0] = False
    masks["ac"] = ac_mask

    low = torch.zeros((8, 8), dtype=torch.bool)
    mid = torch.zeros((8, 8), dtype=torch.bool)
    high = torch.zeros((8, 8), dtype=torch.bool)
    horiz = torch.zeros((8, 8), dtype=torch.bool)
    vert = torch.zeros((8, 8), dtype=torch.bool)

    for u in range(8):
        for v in range(8):
            if u == 0 and v == 0:
                continue
            diag = u + v
            if 1 <= diag <= 3:
                low[u, v] = True
            elif 4 <= diag <= 7:
                mid[u, v] = True
            else:
                high[u, v] = True

            if u < v:
                horiz[u, v] = True
            elif u > v:
                vert[u, v] = True

    masks["low"] = low
    masks["mid"] = mid
    masks["high"] = high
    masks["horiz"] = horiz
    masks["vert"] = vert
    return masks


_AC_MASKS: Dict[str, torch.Tensor] = _build_ac_masks()


# ---------------------------------------------------------------------------
# Numerical Safety Helpers
# ---------------------------------------------------------------------------

def _safe_kurtosis(flat: torch.Tensor) -> float:
    """Compute Fisher excess kurtosis safely (returns 0.0 for near-constant inputs)."""
    flat = flat.float()
    sigma = flat.std()
    if float(sigma) < 1e-12:
        return 0.0
    mu = flat.mean()
    z = (flat - mu) / sigma
    return float(z.pow(4).mean()) - 3.0


def _safe_benford_ssd(ac_coeffs: torch.Tensor) -> float:
    """Compute sum of squared differences from Benford's First Digit Law.

    Scales AC coefficients to non-zero significant digits and computes
    deviation against P(d) = log10(1 + 1/d) for d in 1..9.

    Args:
        ac_coeffs: 1D float32 tensor of AC coefficients.

    Returns:
        Scalar float >= 0.0. Returns 0.0 if no valid non-zero digits exist.
    """
    flat = ac_coeffs.abs()
    valid = flat[flat >= 1e-5]
    if len(valid) == 0:
        return 0.0

    # Extract first significant digit via base-10 mantissa
    log_val = torch.log10(valid)
    mantissa = 10.0 ** (log_val - torch.floor(log_val))
    first_digits = torch.clamp(torch.floor(mantissa).long(), min=1, max=9)

    # Compute empirical digit histogram (bins 1..9)
    counts = torch.bincount(first_digits, minlength=10)[1:10].float()
    total = counts.sum()
    if float(total) < 1e-12:
        return 0.0
    p_emp = counts / total

    benford = _BENFORD_PROBS.to(device=p_emp.device)
    ssd = (p_emp - benford).pow(2).sum()
    return float(ssd)


# ---------------------------------------------------------------------------
# Public Extraction Functions
# ---------------------------------------------------------------------------

def compute_block_dct(image: torch.Tensor) -> torch.Tensor:
    """Compute 2D orthonormal 8×8 block DCT coefficients from canonical image tensor.

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Tensor of shape [B, 1024, 8, 8] containing the 2D DCT coefficients of all 8×8 blocks.
    """
    if image.dim() == 3:
        img = image.unsqueeze(0)
    else:
        img = image

    # Grayscale conversion (BT.601)
    gray = (
        0.299 * img[:, 0:1, :, :]
        + 0.587 * img[:, 1:2, :, :]
        + 0.114 * img[:, 2:3, :, :]
    )  # [B, 1, 256, 256]

    B, _, H, W = gray.shape
    # Partition into non-overlapping 8×8 blocks via unfold
    # Unfold produces [B, 1*8*8, L] where L = (H/8) * (W/8) = 1024
    blocks = F.unfold(gray, kernel_size=8, stride=8)  # [B, 64, 1024]
    blocks = blocks.transpose(1, 2).reshape(B, -1, 8, 8)  # [B, 1024, 8, 8]

    # Apply 2D DCT: D = T @ X @ T.T
    t = _DCT8_MATRIX.to(device=gray.device, dtype=gray.dtype)
    t_t = t.t()
    dct_blocks = torch.matmul(t, torch.matmul(blocks, t_t))  # [B, 1024, 8, 8]

    return dct_blocks


def extract_branch_e_dct_features(image: torch.Tensor) -> Dict[str, float]:
    """Extract 10 candidate E1_DCT scalar features from canonical image tensor.

    Features:
        1. dct_ac_mean_abs
        2. dct_ac_energy
        3. dct_ac_kurtosis
        4. dct_sparsity_ratio
        5. dct_low_freq_ratio
        6. dct_mid_freq_ratio
        7. dct_high_freq_ratio
        8. dct_anisotropy
        9. dct_benford_ssd
        10. dct_block_var_mean

    Args:
        image: Canonical RGB tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

    Returns:
        Dictionary of exactly 10 scalar float features in deterministic order.
    """
    dct_blocks = compute_block_dct(image)  # [B, 1024, 8, 8]
    ac_mask = _AC_MASKS["ac"].to(device=dct_blocks.device)
    low_mask = _AC_MASKS["low"].to(device=dct_blocks.device)
    mid_mask = _AC_MASKS["mid"].to(device=dct_blocks.device)
    high_mask = _AC_MASKS["high"].to(device=dct_blocks.device)
    horiz_mask = _AC_MASKS["horiz"].to(device=dct_blocks.device)
    vert_mask = _AC_MASKS["vert"].to(device=dct_blocks.device)

    # Flattened AC coefficients across all blocks
    ac_coeffs = dct_blocks[:, :, ac_mask].reshape(-1).detach().cpu()
    eps = 1e-12

    # 1. AC mean absolute magnitude
    ac_mean_abs = float(ac_coeffs.abs().mean())

    # 2. AC mean squared energy
    ac_energy = float(ac_coeffs.pow(2).mean())

    # 3. AC excess kurtosis
    ac_kurtosis = _safe_kurtosis(ac_coeffs)

    # 4. Canonical float sparsity ratio (|D| < 1e-3)
    sparsity_ratio = float((ac_coeffs.abs() < 1e-3).float().mean())

    # Zonal energies
    low_coeffs = dct_blocks[:, :, low_mask].reshape(-1).detach().cpu()
    mid_coeffs = dct_blocks[:, :, mid_mask].reshape(-1).detach().cpu()
    high_coeffs = dct_blocks[:, :, high_mask].reshape(-1).detach().cpu()

    low_energy = float(low_coeffs.pow(2).mean()) if len(low_coeffs) > 0 else 0.0
    mid_energy = float(mid_coeffs.pow(2).mean()) if len(mid_coeffs) > 0 else 0.0
    high_energy = float(high_coeffs.pow(2).mean()) if len(high_coeffs) > 0 else 0.0
    total_zonal_energy = low_energy + mid_energy + high_energy + eps

    # 5. Low frequency AC ratio
    low_freq_ratio = low_energy / total_zonal_energy

    # 6. Mid frequency AC ratio
    mid_freq_ratio = mid_energy / total_zonal_energy

    # 7. High frequency AC ratio
    high_freq_ratio = high_energy / total_zonal_energy

    # Directional energy anisotropy
    horiz_coeffs = dct_blocks[:, :, horiz_mask].reshape(-1).detach().cpu()
    vert_coeffs = dct_blocks[:, :, vert_mask].reshape(-1).detach().cpu()
    horiz_energy = float(horiz_coeffs.pow(2).mean()) if len(horiz_coeffs) > 0 else 0.0
    vert_energy = float(vert_coeffs.pow(2).mean()) if len(vert_coeffs) > 0 else 0.0

    # 8. Anisotropy
    anisotropy = abs(horiz_energy - vert_energy) / (horiz_energy + vert_energy + eps)

    # 9. Benford First Digit Law SSD
    benford_ssd = _safe_benford_ssd(ac_coeffs)

    # 10. Intra-block AC variance mean
    # Per-block AC coefficients: [B*1024, 63]
    block_ac = dct_blocks[:, :, ac_mask].reshape(-1, 63).detach().cpu()
    block_vars = block_ac.var(dim=-1, unbiased=False)
    block_var_mean = float(block_vars.mean())

    features: Dict[str, float] = {
        "dct_ac_mean_abs": ac_mean_abs,
        "dct_ac_energy": ac_energy,
        "dct_ac_kurtosis": ac_kurtosis,
        "dct_sparsity_ratio": sparsity_ratio,
        "dct_low_freq_ratio": low_freq_ratio,
        "dct_mid_freq_ratio": mid_freq_ratio,
        "dct_high_freq_ratio": high_freq_ratio,
        "dct_anisotropy": anisotropy,
        "dct_benford_ssd": benford_ssd,
        "dct_block_var_mean": block_var_mean,
    }

    return features
