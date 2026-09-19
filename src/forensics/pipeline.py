"""Forensic Feature Extraction Pipeline (Block 2 Orchestrator).

Orchestrates multi-branch forensic feature extraction on canonical Block 1 image representations.
Combines enabled forensic branches while preserving deterministic feature ordering, modularity,
and strict isolation between alternative descriptors (e.g. C_LBP vs C_GLCM vs C_LBP_EDGE).

Branch D Notes
--------------
Branch D provides three independent, alternative pixel-domain residual candidates:
    D_HIGHPASS  : Gaussian high-pass residual (linear, 5 candidate features)
    D_LAPLACIAN : Discrete Laplacian residual (linear, 5 candidate features)
    D_MFR       : Median filter residual (nonlinear, 5 candidate features)

The convenience alias "D" expands to D_HIGHPASS + D_LAPLACIAN + D_MFR.
This is NOT a claim of complementarity. Block 4 determines whether all three
are needed or whether some should be dropped as redundant with each other or
with Branches A/B/C.

D_HIGHPASS and D_LAPLACIAN are both linear residuals and may be redundant.
D_MFR is nonlinear and is the most theoretically distinct from A/B.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.forensics.branch_a_frequency import extract_branch_a_features
from src.forensics.branch_b_wavelet import extract_branch_b_features
from src.forensics.branch_c_texture import (
    extract_branch_c_glcm_features,
    extract_branch_c_lbp_edge_features,
    extract_branch_c_lbp_features,
)
from src.forensics.branch_d_residual import (
    extract_branch_d_highpass_features,
    extract_branch_d_laplacian_features,
    extract_branch_d_mfr_features,
)
from src.forensics.branch_e import (
    extract_branch_e_dct_features,
    extract_branch_e_features,
    extract_branch_e_grid_features,
    extract_branch_e_phase_features,
    extract_branch_e_recompression_features,
)


# Branch registry mapping branch identifier to its extractor function
_BRANCH_REGISTRY: Dict[str, Callable[[torch.Tensor], Dict[str, float]]] = {
    "A": extract_branch_a_features,
    "B": extract_branch_b_features,
    "C_LBP": extract_branch_c_lbp_features,
    "C_GLCM": extract_branch_c_glcm_features,
    "C_LBP_EDGE": extract_branch_c_lbp_edge_features,
    "D_HIGHPASS": extract_branch_d_highpass_features,
    "D_LAPLACIAN": extract_branch_d_laplacian_features,
    "D_MFR": extract_branch_d_mfr_features,
    "E": extract_branch_e_features,
    "E_DCT": extract_branch_e_dct_features,
    "E_RESP": extract_branch_e_recompression_features,
    "E_PHASE": extract_branch_e_phase_features,
    "E_GRID": extract_branch_e_grid_features,
}

# Known/planned branch descriptions for validation & documentation
_KNOWN_BRANCHES: Dict[str, str] = {
    "A": "Branch A — Frequency / Periodicity (FFT & Synthbuster)",
    "B": "Branch B — Wavelet (3-level Haar DWT)",
    "C_LBP": "Branch C — Local Texture (Standard Uniform LBP)",
    "C_GLCM": "Branch C — Local Texture (Multi-distance GLCM)",
    "C_LBP_EDGE": "Branch C — Local Texture (Edge-guided LBP candidate)",
    "D": "Branch D — Residual / Noise (convenience alias: D_HIGHPASS + D_LAPLACIAN + D_MFR)",
    "D_HIGHPASS": "Branch D — Residual / Noise (Gaussian high-pass candidate)",
    "D_LAPLACIAN": "Branch D — Residual / Noise (Discrete Laplacian candidate)",
    "D_MFR": "Branch D — Residual / Noise (Median filter residual candidate)",
    "E": "Branch E — JPEG / Compression Forensics (unified 26 candidate features)",
    "E_DCT": "Branch E — JPEG / Compression Forensics (8×8 Block DCT Fingerprints, 10 features)",
    "E_RESP": "Branch E — JPEG / Compression Forensics (Controlled Multi-Q Response, 8 features)",
    "E_PHASE": "Branch E — JPEG / Compression Forensics (Compression-Stable Phase, 4 features)",
    "E_GRID": "Branch E — JPEG / Compression Forensics (Canonical 8×8 Grid Discontinuities, 4 features)",
}

# Default active branches for default instantiation
DEFAULT_BRANCHES: Tuple[str, ...] = ("A", "B")

# Branch D convenience alias: expands to all three D candidates in deterministic order.
# WARNING: This alias is NOT a claim that the three candidates are complementary.
# It enables running the full candidate set; Block 4 determines what to keep.
_D_ALIAS_EXPANSION: Tuple[str, ...] = ("D_HIGHPASS", "D_LAPLACIAN", "D_MFR")


class ForensicPipeline:
    """Orchestrator for extracting forensic feature vectors across selected branches."""

    def __init__(self, branches: Optional[Sequence[str]] = None) -> None:
        """Initialize pipeline with specified forensic branches.

        Args:
            branches: Sequence of branch identifiers. Examples:
                      ["A"], ["B"], ["C_LBP"], ["C_GLCM"], ["C_LBP_EDGE"],
                      ["D_HIGHPASS"], ["D_LAPLACIAN"], ["D_MFR"], ["D"],
                      ["A", "B", "C_LBP", "D_MFR"], etc.
                      Defaults to ("A", "B").

        Branch D notes:
            - D_HIGHPASS, D_LAPLACIAN, D_MFR may each be selected independently.
            - "D" is a convenience alias that expands to D_HIGHPASS + D_LAPLACIAN + D_MFR.
              This does NOT imply the three candidates are complementary.
            - D_HIGHPASS and D_LAPLACIAN are alternative linear candidates that may
              be redundant with each other and with Branch B wavelet features.
            - Block 4 determines final candidate inclusion.
        """
        if branches is None:
            self._branches: Tuple[str, ...] = DEFAULT_BRANCHES
        else:
            norm_branches = []
            for b in branches:
                b_up = b.upper().strip()

                # Branch C must select an explicit alternative
                if b_up == "C":
                    raise ValueError(
                        "Branch 'C' requires selecting an explicit alternative variant: "
                        "'C_LBP', 'C_GLCM', or 'C_LBP_EDGE'. Alternative texture descriptors "
                        "are not concatenated automatically."
                    )

                # Branch D convenience alias — expand to three individual candidates
                if b_up == "D":
                    for d_candidate in _D_ALIAS_EXPANSION:
                        if d_candidate not in norm_branches:
                            norm_branches.append(d_candidate)
                    continue

                if b_up not in _KNOWN_BRANCHES:
                    raise ValueError(
                        f"Unknown forensic branch: '{b}'. Valid branches are: {list(_KNOWN_BRANCHES.keys())}"
                    )
                if b_up not in _BRANCH_REGISTRY:
                    raise NotImplementedError(
                        f"Branch '{b_up}' ({_KNOWN_BRANCHES[b_up]}) is planned but not yet implemented."
                    )
                if b_up not in norm_branches:
                    norm_branches.append(b_up)

            if not norm_branches:
                raise ValueError("At least one forensic branch must be specified.")

            # Enforce mutual exclusivity among Branch C alternative texture descriptors
            c_variants = [b for b in norm_branches if b in ("C_LBP", "C_GLCM", "C_LBP_EDGE")]
            if len(c_variants) > 1:
                raise ValueError(
                    f"Branch C texture variants are mutually exclusive alternatives and cannot be combined. "
                    f"Found multiple Branch C variants: {c_variants}. "
                    f"Select exactly one of 'C_LBP', 'C_GLCM', or 'C_LBP_EDGE'."
                )

            self._branches = tuple(norm_branches)

    @property
    def branches(self) -> Tuple[str, ...]:
        """Tuple of enabled branch identifiers."""
        return self._branches

    def extract(self, image: torch.Tensor) -> Dict[str, float]:
        """Extract combined scalar forensic features from all enabled branches.

        Args:
            image: Canonical image tensor [3, 256, 256] or [B, 3, 256, 256], float32 in [0, 1].

        Returns:
            Dictionary of scalar float features with deterministic ordering.
        """
        features: Dict[str, float] = {}
        for branch in self._branches:
            extractor = _BRANCH_REGISTRY[branch]
            branch_feats = extractor(image)
            features.update(branch_feats)
        return features

    def get_feature_names(self, dummy_shape: Tuple[int, ...] = (3, 256, 256)) -> List[str]:
        """Get list of feature names for currently enabled branches.

        Args:
            dummy_shape: Image tensor shape to probe feature keys (default: (3, 256, 256)).

        Returns:
            List of feature name strings in deterministic order.
        """
        dummy = torch.zeros(dummy_shape, dtype=torch.float32)
        feats = self.extract(dummy)
        return list(feats.keys())
