"""RGB Pipeline Orchestrator (Block 2).

Dispatches canonical images to Branch A (Lab Local Correlation Maps) and
Branch B (PiD YUV Residuals, s=0.2 per DEC-015), unifying derived spatial representations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from .branch_a_lab import extract_branch_a_lab_maps
from .branch_b_nqp import compute_branch_b_residual, compute_pid_residual

SUPPORTED_BRANCHES = ("A", "B")


class RGBPipeline:
    """Orchestrator for Block 2 RGB representation extraction.

    Extracts spatial representations:
        Branch A: 3-channel Lab local correlation maps [3, 256, 256]
        Branch B: 3-channel PiD YUV residual map (s=0.2) [3, 256, 256]
    """

    def __init__(self, branches: Optional[Sequence[str]] = None) -> None:
        if branches is None:
            self.branches = ("A", "B")
        else:
            invalid = [b for b in branches if b not in SUPPORTED_BRANCHES]
            if invalid:
                raise ValueError(
                    f"Unsupported RGB branch(es): {invalid}. "
                    f"Supported branches: {SUPPORTED_BRANCHES}"
                )
            self.branches = tuple(branches)

    def extract(
        self,
        image: Union[torch.Tensor, np.ndarray],
        sample_id: Optional[Union[str, int]] = None,
        window_size: int = 7,
        s: float = 0.2,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Extract requested RGB representations from a canonical image.

        Args:
            image: Canonical RGB image (float32 [3, 256, 256] in [0, 1] or uint8 [256, 256, 3]).
            sample_id: Optional sample identifier.
            window_size: Sliding window size for Branch A (default: 7).
            s: PiD YUV scaling factor for Branch B (default: 0.2 per DEC-015).
            **kwargs: Additional compatibility keyword arguments.

        Returns:
            Dictionary containing:
                "lab_maps":      torch.Tensor [3, 256, 256] (if "A" in branches)
                "branch_b_map":  torch.Tensor [3, 256, 256] (if "B" in branches)
                "pid_map":       torch.Tensor [3, 256, 256] (alias if "B" in branches)
                "nqp_map":       torch.Tensor [3, 256, 256] (legacy alias if "B" in branches)
        """
        results: Dict[str, torch.Tensor] = {}

        if "A" in self.branches:
            results["lab_maps"] = extract_branch_a_lab_maps(image, window_size=window_size)

        if "B" in self.branches:
            # Active Branch B: PiD YUV residuals with s=0.2 (DEC-015)
            branch_b_map = compute_pid_residual(image, s=s, sample_id=sample_id)
            results["branch_b_map"] = branch_b_map
            results["pid_map"] = branch_b_map
            results["nqp_map"] = branch_b_map  # backward compatibility alias

        return results

