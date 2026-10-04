"""Executable CLI runner for Forensic Feature Analysis (Block 2).

Extracts the 111 candidate forensic features across training and validation sets,
and runs the complete Feature Analysis pipeline (Stages A through I).

Usage:
    .venv/bin/python src/forensics/run_feature_analysis.py --train-samples 2000 --val-samples 1000 --comp-samples 100
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from typing import List, Tuple

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from src.forensics.branch_f.runner import FeatureAnalysisRunner
from src.forensics.pipeline import ForensicPipeline


def load_materialized_train_features(
    train_dir: str,
    max_samples: int,
    feature_names: List[str],
    n_workers: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed samples from Block 1 materialized Parquet files and extract 111 features.

    Note: n_workers is accepted for CLI compatibility but extraction runs sequentially to avoid
    ProcessPoolExecutor hangs with PyTorch + cv2 in multiprocessing environments.
    """
    parquet_files = sorted(
        [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".parquet")]
    )
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under '{train_dir}'.")

    print(f"Reading up to {max_samples} samples from materialized train Parquet files...")
    all_bytes: List[bytes] = []
    all_labels: List[int] = []

    # Read evenly across parts
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["image_rgb", "label_a"])
        for i in range(len(table)):
            all_bytes.append(table["image_rgb"][i].as_py())
            all_labels.append(table["label_a"][i].as_py())
            if len(all_bytes) >= max_samples:
                break
        if len(all_bytes) >= max_samples:
            break

    total = len(all_bytes)
    print(f"Loaded {total} materialized images ({np.sum(np.array(all_labels)==0)} real, {np.sum(np.array(all_labels)==1)} AI).")
    print(f"Extracting 111 forensic features across {total} images sequentially...")

    pipeline = ForensicPipeline(["A", "B", "C_LBP", "D_MFR", "E"])
    feature_matrix = np.zeros((total, len(feature_names)), dtype=np.float32)

    t0 = time.perf_counter()
    for idx, raw_bytes in enumerate(all_bytes):
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((256, 256, 3))
        # arr.copy() avoids UserWarning: non-writable array passed to torch.from_numpy
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        feats = pipeline.extract(tensor)
        feature_matrix[idx, :] = [feats[f] for f in feature_names]
        if (idx + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {idx + 1}/{total} ({elapsed:.1f}s elapsed, {elapsed / (idx + 1) * 1000:.1f} ms/img)")

    dt = time.perf_counter() - t0
    print(f"Extraction completed in {dt:.2f}s ({dt / total * 1000:.2f} ms/image).")

    return feature_matrix, np.array(all_labels, dtype=np.int32)


def load_raw_validation_samples(
    val_data_dir: str,
    max_samples: int,
    feature_names: List[str],
    n_workers: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[torch.Tensor]]:
    """Load raw validation Parquet files, apply canonical crop+resize, and extract features and metadata."""
    parquet_files = sorted(
        [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if "validation" in f and f.endswith(".parquet")]
    )
    if not parquet_files:
        parquet_files = sorted(
            [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith(".parquet")]
        )

    print(f"Reading up to {max_samples} validation samples from {len(parquet_files)} Parquet files...")
    raw_image_bytes: List[bytes] = []
    val_labels_a: List[int] = []
    val_labels_b: List[int] = []

    for pf in parquet_files:
        table = pq.read_table(pf, columns=["Image", "Label_A", "Label_B"])
        n_rows = len(table)
        for i in range(n_rows):
            img_dict = table["Image"][i].as_py()
            raw_image_bytes.append(img_dict["bytes"])
            val_labels_a.append(table["Label_A"][i].as_py())
            val_labels_b.append(table["Label_B"][i].as_py())
            if len(raw_image_bytes) >= max_samples:
                break
        if len(raw_image_bytes) >= max_samples:
            break

    total = len(raw_image_bytes)
    print(f"Loaded {total} validation samples. Preprocessing and extracting features...")

    # Canonical preprocessing (crop + resize 256x256 INTER_AREA)
    tensors: List[torch.Tensor] = []
    for b in raw_image_bytes:
        pil_img = Image.open(io.BytesIO(b)).convert("RGB")
        w, h = pil_img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        cropped = pil_img.crop((left, top, left + side, top + side))
        arr = np.array(cropped, dtype=np.uint8)
        resized = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        tensors.append(t)

    pipeline = ForensicPipeline(["A", "B", "C_LBP", "D_MFR", "E"])
    feature_matrix = np.zeros((total, len(feature_names)), dtype=np.float32)

    for idx, t in enumerate(tensors):
        feats = pipeline.extract(t)
        for j, f in enumerate(feature_names):
            feature_matrix[idx, j] = feats[f]

    return feature_matrix, np.array(val_labels_a, dtype=np.int32), np.array(val_labels_b, dtype=np.int32), tensors


def parse_args():
    parser = argparse.ArgumentParser(description="Run Block 2 Forensic Feature Analysis.")
    parser.add_argument("--train-dir", type=str, default="data/processed/train")
    parser.add_argument("--val-dir", type=str, default="data/defactify/data")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--comp-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="analysis/forensic_feature_analysis")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    runner = FeatureAnalysisRunner(
        output_dir=args.output_dir,
        random_state=args.seed,
    )
    feature_names = runner.feature_names

    # 1. Load training features
    X_train, y_train = load_materialized_train_features(
        train_dir=args.train_dir,
        max_samples=args.train_samples,
        feature_names=feature_names,
        n_workers=args.n_workers,
        seed=args.seed,
    )

    # 2. Load validation features & generator metadata
    X_val, y_val, val_gen_labels, val_tensors = load_raw_validation_samples(
        val_data_dir=args.val_dir,
        max_samples=args.val_samples,
        feature_names=feature_names,
        n_workers=args.n_workers,
        seed=args.seed,
    )

    gen_names_map = {
        0: "Real",
        1: "SD2.1",
        2: "SDXL",
        3: "SD3",
        4: "DALL-E3",
        5: "Midjourney",
    }

    # 3. Compression sample slice
    comp_tensors = val_tensors[:args.comp_samples]
    comp_labels = y_val[:args.comp_samples]

    # 4. Run full analysis pipeline
    artifacts = runner.run_analysis(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        generator_labels_val=val_gen_labels,
        generator_names_map=gen_names_map,
        compression_sample_tensors=comp_tensors,
        compression_sample_labels=comp_labels,
    )

    print("\nFeature Analysis Execution Finished Successfully.")


if __name__ == "__main__":
    main()
