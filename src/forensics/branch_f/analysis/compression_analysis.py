"""Symmetric JPEG Compression Sensitivity & Degradation Analysis Module.

Evaluates forensic feature behavior across controlled in-memory JPEG compression levels:
Clean, Q95, Q80, Q60, Q40, Q20.

SCIENTIFIC CONTROLS (DEC-006):
Compression transformations are applied 100% SYMMETRICALLY to both Real and AI images.
No asymmetric class-conditioned compression is permitted.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from src.forensics.pipeline import ForensicPipeline


@dataclass(frozen=True)
class FeatureCompressionStats:
    """Compression sensitivity statistics for a single forensic feature across qualities."""

    index: int
    name: str
    branch: str
    sub_branch: str
    auc_clean: float
    auc_q95: float
    auc_q80: float
    auc_q60: float
    auc_q40: float
    auc_q20: float
    auc_mean_compressed: float
    auc_worst_compressed: float
    auc_retention_q60: float  # AUC(Q60) / AUC(Clean)
    auc_retention_q20: float  # AUC(Q20) / AUC(Clean)
    is_compression_robust: bool  # AUC(Q60) >= 0.85 * AUC(Clean)


def apply_symmetric_jpeg_compression(
    image_tensor: torch.Tensor,
    quality: int,
) -> torch.Tensor:
    """Apply in-memory JPEG compression symmetrically to a canonical image tensor.

    Args:
        image_tensor: Tensor of shape [3, 256, 256] or [1, 3, 256, 256] in [0, 1].
        quality: JPEG compression quality integer (1..100).

    Returns:
        Recompressed canonical image tensor of same shape [3, 256, 256], float32 in [0, 1].
    """
    if quality >= 100:
        return image_tensor.squeeze(0) if image_tensor.ndim == 4 else image_tensor

    t = image_tensor.squeeze(0) if image_tensor.ndim == 4 else image_tensor
    # Convert [3, 256, 256] RGB [0,1] -> [256, 256, 3] BGR uint8
    arr = t.permute(1, 2, 0).detach().cpu().numpy()
    arr_uint8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR)

    # Encode with specified JPEG quality
    success, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        raise RuntimeError(f"JPEG encoding failed at quality {quality}.")

    # Decode back
    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec_bgr is None:
        raise RuntimeError(f"JPEG decoding failed at quality {quality}.")

    dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
    out_tensor = torch.from_numpy(dec_rgb).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    return out_tensor


class CompressionSensitivityAnalyzer:
    """Analyzes forensic feature degradation and retention under symmetric JPEG compression."""

    QUALITIES: Tuple[int, ...] = (95, 80, 60, 40, 20)

    def __init__(
        self,
        pipeline: Optional[ForensicPipeline] = None,
        branch_map: Optional[Dict[str, str]] = None,
        sub_branch_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize analyzer."""
        self.pipeline = pipeline or ForensicPipeline(["A", "B", "C_LBP", "D_MFR", "E"])
        self.feature_names = self.pipeline.get_feature_names()
        self.branch_map = branch_map or {}
        self.sub_branch_map = sub_branch_map or {}

    def extract_features_under_compression(
        self,
        images: Sequence[torch.Tensor],
        qualities: Sequence[int] = QUALITIES,
    ) -> Dict[str, np.ndarray]:
        """Extract candidate feature bank on clean and compressed versions of sample images.

        Args:
            images: List of canonical image tensors [3, 256, 256].
            qualities: Sequence of JPEG qualities to evaluate.

        Returns:
            Dictionary mapping condition name ("clean", "q95", "q80", etc.) to feature matrix [N, 111].
        """
        N = len(images)
        D = len(self.feature_names)
        results: Dict[str, np.ndarray] = {
            "clean": np.zeros((N, D), dtype=np.float32),
        }
        for q in qualities:
            results[f"q{q}"] = np.zeros((N, D), dtype=np.float32)

        for i, img in enumerate(images):
            # Clean extraction
            clean_feats = self.pipeline.extract(img)
            for j, f in enumerate(self.feature_names):
                results["clean"][i, j] = clean_feats[f]

            # Compressed extractions
            for q in qualities:
                compressed_img = apply_symmetric_jpeg_compression(img, q)
                q_feats = self.pipeline.extract(compressed_img)
                for j, f in enumerate(self.feature_names):
                    results[f"q{q}"][i, j] = q_feats[f]

        return results

    def compute_compression_stability(
        self,
        compressed_feature_matrices: Dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> List[FeatureCompressionStats]:
        """Compute discriminative AUC retention across compression qualities."""
        y = np.asarray(labels, dtype=np.int32)
        D = len(self.feature_names)

        def _calc_eff_auc(matrix: np.ndarray, col_idx: int) -> float:
            vals = matrix[:, col_idx]
            mask = np.isfinite(vals)
            if np.sum(mask & (y == 0)) < 2 or np.sum(mask & (y == 1)) < 2:
                return 0.5
            try:
                raw_auc = float(roc_auc_score(y[mask], vals[mask]))
                return float(max(raw_auc, 1.0 - raw_auc))
            except Exception:
                return 0.5

        stats_list: List[FeatureCompressionStats] = []
        for j, name in enumerate(self.feature_names):
            auc_clean = _calc_eff_auc(compressed_feature_matrices["clean"], j)
            auc_q95 = _calc_eff_auc(compressed_feature_matrices.get("q95", compressed_feature_matrices["clean"]), j)
            auc_q80 = _calc_eff_auc(compressed_feature_matrices.get("q80", compressed_feature_matrices["clean"]), j)
            auc_q60 = _calc_eff_auc(compressed_feature_matrices.get("q60", compressed_feature_matrices["clean"]), j)
            auc_q40 = _calc_eff_auc(compressed_feature_matrices.get("q40", compressed_feature_matrices["clean"]), j)
            auc_q20 = _calc_eff_auc(compressed_feature_matrices.get("q20", compressed_feature_matrices["clean"]), j)

            comp_aucs = [auc_q95, auc_q80, auc_q60, auc_q40, auc_q20]
            mean_comp = float(np.mean(comp_aucs))
            worst_comp = float(np.min(comp_aucs))

            denom = auc_clean if auc_clean > 0.5 else 0.5
            ret_q60 = auc_q60 / denom
            ret_q20 = auc_q20 / denom
            is_robust = bool(auc_q60 >= 0.85 * auc_clean and auc_q60 >= 0.55)

            stats_list.append(
                FeatureCompressionStats(
                    index=j,
                    name=name,
                    branch=self.branch_map.get(name, "UNKNOWN"),
                    sub_branch=self.sub_branch_map.get(name, "UNKNOWN"),
                    auc_clean=auc_clean,
                    auc_q95=auc_q95,
                    auc_q80=auc_q80,
                    auc_q60=auc_q60,
                    auc_q40=auc_q40,
                    auc_q20=auc_q20,
                    auc_mean_compressed=mean_comp,
                    auc_worst_compressed=worst_comp,
                    auc_retention_q60=ret_q60,
                    auc_retention_q20=ret_q20,
                    is_compression_robust=is_robust,
                )
            )

        return stats_list

    @staticmethod
    def to_dataframe(stats_list: Sequence[FeatureCompressionStats]) -> pd.DataFrame:
        """Convert list of FeatureCompressionStats to a pandas DataFrame."""
        df = pd.DataFrame([asdict(s) for s in stats_list])
        return df.sort_values(by="auc_mean_compressed", ascending=False).reset_index(drop=True)
