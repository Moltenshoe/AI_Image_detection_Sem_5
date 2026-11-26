"""DataLoader Performance Benchmark for Block 2 RGB On-The-Fly Pipeline.

Measures:
1. Micro-benchmarks of Branch A (Lab local correlation maps) vs Branch B (PiD residuals).
2. DataLoader throughput (img/s, ms/batch, memory RSS) across worker counts (0, 2, 4)
   for Mode D (9 channels), Mode C (6 channels), and Mode A (3 channels).
"""

from __future__ import annotations

import os
import sys
import psutil
import time

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from src.rgb.branch_a_lab import extract_branch_a_lab_maps
from src.rgb.branch_b_nqp import compute_pid_residual
from src.rgb.dataset import RGBDataset



def benchmark_rgb_pipeline() -> None:
    print("=" * 70)
    print("Block 2 RGB Pipeline — Performance Benchmark")
    print("=" * 70)

    # 1. Micro-benchmarks of individual branches on canonical 256x256 image
    sample_rgb = torch.rand(3, 256, 256, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        _ = extract_branch_a_lab_maps(sample_rgb)
        _ = compute_pid_residual(sample_rgb, s=0.2)

    n_micro = 50

    # Branch A
    t0 = time.perf_counter()
    for _ in range(n_micro):
        _ = extract_branch_a_lab_maps(sample_rgb)
    t1 = time.perf_counter()
    ms_branch_a = (t1 - t0) / n_micro * 1000

    # Branch B
    t0 = time.perf_counter()
    for _ in range(n_micro):
        _ = compute_pid_residual(sample_rgb, s=0.2)
    t1 = time.perf_counter()
    ms_branch_b = (t1 - t0) / n_micro * 1000

    print(f"Branch A (Lab local correlation maps 7x7): {ms_branch_a:6.2f} ms/image")
    print(f"Branch B (PiD YCbCr residuals s=0.2):       {ms_branch_b:6.2f} ms/image")
    bottleneck = "Branch A (Lab Local Correlation)" if ms_branch_a > ms_branch_b else "Branch B (PiD)"
    ratio = ms_branch_a / ms_branch_b if ms_branch_b > 0 else 0
    print(f"Primary Operator Bottleneck: {bottleneck} ({ratio:.1f}x execution cost of Branch B)")
    print("-" * 70)

    # 2. DataLoader Benchmarks across Modes & Worker Counts
    batch_size = 32
    num_batches_measure = 5  # 160 images per test setting

    process = psutil.Process(os.getpid())
    initial_rss_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Initial Process RSS Memory: {initial_rss_mb:.1f} MB")

    for mode, name, channels in [
        ("D", "Mode D (RGB + Branch A + Branch B)", 9),
        ("C", "Mode C (RGB + Branch B PiD)", 6),
        ("A", "Mode A (Original RGB only)", 3),
    ]:
        print(f"\n--- DataLoader: {name} [{channels} channels] ---")
        print(f"Batch Size: {batch_size}, Evaluation Samples per Run: {num_batches_measure * batch_size}")

        for nw in [0, 2, 4]:
            try:
                ds = RGBDataset(mode=mode, rgb_dir=None)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)

                # Warmup 1 batch
                iterator = iter(loader)
                _ = next(iterator)

                t0 = time.perf_counter()
                for _ in range(num_batches_measure):
                    _ = next(iterator)
                t1 = time.perf_counter()

                total_time = t1 - t0
                total_imgs = num_batches_measure * batch_size
                fps = total_imgs / total_time
                ms_per_batch = (total_time / num_batches_measure) * 1000
                current_rss_mb = process.memory_info().rss / (1024 * 1024)

                print(f"num_workers={nw}: {fps:6.1f} img/s | {ms_per_batch:6.1f} ms/batch | Memory RSS: {current_rss_mb:5.1f} MB")
            except Exception as e:
                print(f"num_workers={nw}: Failed with {type(e).__name__}: {e}")

    print("=" * 70)


if __name__ == "__main__":
    benchmark_rgb_pipeline()
