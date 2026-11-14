"""Block 3 Integration Interface for RGB Pipeline.

Provides the dataset reading interface for downstream Block 3 model training
and Block 4 evaluation, supporting controlled ablations across:
    Experiment A: Original RGB only
    Experiment B: Original RGB + Lab correlation maps
    Experiment C: Original RGB + NQP residual map
    Experiment D: Original RGB + Lab correlation maps + NQP residual map
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.data.materializer import get_processed_train_dir


class RGBDataset(Dataset):
    """PyTorch Dataset joining Block 1 images with derived RGB spatial representations.

    Retrieves canonical images from Block 1 and joins them with optional Lab maps
    and optional NQP maps using the stable Block 1 sample ID.

    Args:
        mode: Ablation configuration:
            "A" -> Original RGB only ([3, 256, 256])
            "B" -> Original RGB + Lab maps ([6, 256, 256] or tuple)
            "C" -> Original RGB + NQP map ([6, 256, 256] or tuple)
            "D" -> Original RGB + Lab maps + NQP map ([9, 256, 256] or tuple)
        processed_train_dir: Path to Block 1 materialized images.
        rgb_dir: Path to derived RGB directory (data/rgb/).
        transform: Optional joint spatial transform (applied identically to RGB and maps).
        stack_channels: If True (default), stacks all channels into a single tensor
            [C_total, 256, 256]; if False, returns a dictionary of representations.
    """

    SUPPORTED_MODES = ("A", "B", "C", "D")

    def __init__(
        self,
        mode: str = "A",
        processed_train_dir: Optional[str] = None,
        rgb_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        stack_channels: bool = True,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from {self.SUPPORTED_MODES}")

        self.mode = mode
        self.transform = transform
        self.stack_channels = stack_channels

        project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        if processed_train_dir is None:
            processed_train_dir = get_processed_train_dir()
        if rgb_dir is None:
            rgb_dir = os.path.join(project_root, "data", "rgb")

        self.processed_train_dir = processed_train_dir
        self.rgb_dir = rgb_dir

        # Open Block 1 parquet files
        parquet_paths = sorted(
            glob.glob(os.path.join(processed_train_dir, "part-*.parquet"))
        )
        if not parquet_paths:
            raise FileNotFoundError(f"No part-*.parquet found in {processed_train_dir}")

        self._parquet_files = [pq.ParquetFile(p) for p in parquet_paths]

        # Build lightweight index: (file_idx, rg_idx, row_in_rg, label_a, sample_idx, image_id)
        self._index: List[Tuple[int, int, int, int, int, str]] = []
        global_idx = 0
        for f_idx, pf in enumerate(self._parquet_files):
            for rg_idx in range(pf.metadata.num_row_groups):
                batch = pf.read_row_group(rg_idx, columns=["label_a"])
                labels = batch.column("label_a").to_pylist()
                for row_in_rg, la in enumerate(labels):
                    image_id = f"train_{global_idx:06d}"
                    self._index.append((f_idx, rg_idx, row_in_rg, int(la), global_idx, image_id))
                    global_idx += 1

        self._len = len(self._index)

    def __len__(self) -> int:
        return self._len

    def _load_original_rgb(self, file_idx: int, rg_idx: int, row_in_rg: int) -> torch.Tensor:
        pf = self._parquet_files[file_idx]
        batch = pf.read_row_group(rg_idx, columns=["image_rgb"])
        rgb_bytes = batch.column("image_rgb").to_pylist()[row_in_rg]
        arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(256, 256, 3).copy()
        # [H, W, 3] -> [3, H, W] float32 in [0, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return tensor

    def _load_derived_map(self, branch_subdir: str, image_id: str) -> Optional[torch.Tensor]:
        map_path = os.path.join(self.rgb_dir, branch_subdir, f"{image_id}.npy")
        if not os.path.isfile(map_path):
            return None
        arr = np.load(map_path)
        return torch.from_numpy(arr).float()

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], int, Dict[str, Any]]:
        file_idx, rg_idx, row_in_rg, label_a, sample_idx, image_id = self._index[idx]

        rgb = self._load_original_rgb(file_idx, rg_idx, row_in_rg)
        tensors = [rgb]
        rep_dict: Dict[str, torch.Tensor] = {"rgb": rgb}

        if self.mode in ("B", "D"):
            lab_maps = self._load_derived_map("branch_a_lab", image_id)
            if lab_maps is None:
                # If derived map not yet materialized on disk, extract on-the-fly
                from .branch_a_lab import extract_branch_a_lab_maps
                lab_maps = extract_branch_a_lab_maps(rgb)
            tensors.append(lab_maps)
            rep_dict["lab_maps"] = lab_maps

        if self.mode in ("C", "D"):
            # Load active Branch B representation (branch_b_pid, fallback to legacy branch_b_nqp)
            branch_b_map = self._load_derived_map("branch_b_pid", image_id)
            if branch_b_map is None:
                branch_b_map = self._load_derived_map("branch_b_nqp", image_id)
            if branch_b_map is None:
                # If derived map not yet materialized on disk, extract on-the-fly using active PiD s=0.2
                from .branch_b_nqp import compute_pid_residual
                branch_b_map = compute_pid_residual(rgb, s=0.2)
            tensors.append(branch_b_map)
            rep_dict["branch_b_map"] = branch_b_map
            rep_dict["pid_map"] = branch_b_map
            rep_dict["nqp_map"] = branch_b_map  # compatibility alias


        # Apply geometric transform identically across all representations if present (DEC-006)
        if self.transform is not None:
            if self.stack_channels:
                stacked = torch.cat(tensors, dim=0)
                transformed = self.transform(stacked)
                return transformed, label_a, {"sample_idx": sample_idx, "image_id": image_id}
            else:
                for k in rep_dict:
                    rep_dict[k] = self.transform(rep_dict[k])
                return rep_dict, label_a, {"sample_idx": sample_idx, "image_id": image_id}

        if self.stack_channels:
            stacked = torch.cat(tensors, dim=0)
            return stacked, label_a, {"sample_idx": sample_idx, "image_id": image_id}
        else:
            return rep_dict, label_a, {"sample_idx": sample_idx, "image_id": image_id}
