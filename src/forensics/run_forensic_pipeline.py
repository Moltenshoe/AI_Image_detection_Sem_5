"""Executable CLI runner for the forensic feature extraction pipeline (Block 2).

Executes multi-branch feature extraction on canonical Block 1 image datasets or synthetic test tensors.

Usage:
    .venv/bin/python src/forensics/run_forensic_pipeline.py --synthetic --branches C_LBP C_GLCM
    .venv/bin/python src/forensics/run_forensic_pipeline.py --split train --branches A B C_LBP --max-samples 5
"""

import argparse
import math
import os
import sys
import time

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from src.forensics.pipeline import ForensicPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run forensic feature extraction pipeline."
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["A", "B", "C_LBP"],
        help=(
            "Forensic branches to enable: A, B, C_LBP, C_GLCM, C_LBP_EDGE, "
            "D_HIGHPASS, D_LAPLACIAN, D_MFR, D (default: A B C_LBP)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate (default: train).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Number of samples to process (default: 3).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run smoke extraction on synthetic tensors instead of loading dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("Forensic Pipeline Runner (Block 2)")
    print("=" * 70)
    print(f"Enabled branches: {args.branches}")

    pipeline = ForensicPipeline(branches=args.branches)
    feature_names = pipeline.get_feature_names()
    print(f"Total features per image: {len(feature_names)}")

    if args.synthetic:
        print("\n--- Running Synthetic Tensor Smoke Extraction ---")
        torch.manual_seed(42)
        sample_tensor = torch.rand(3, 256, 256, dtype=torch.float32)
        t0 = time.perf_counter()
        feats = pipeline.extract(sample_tensor)
        dt = time.perf_counter() - t0

        print(f"Extracted {len(feats)} features in {dt*1000:.2f} ms")
        for k, v in list(feats.items())[:12]:
            print(f"  {k}: {v:.6f}")
        if len(feats) > 12:
            print(f"  ... ({len(feats) - 12} more features)")

        # Verify finiteness
        all_finite = all(math.isfinite(v) for v in feats.values())
        print(f"All features finite: {all_finite}")
        if not all_finite:
            sys.exit(1)
    else:
        print(f"\n--- Loading {args.max_samples} Samples from {args.split} Split ---")
        from src.data.loader import DefactifyDataset

        dataset = DefactifyDataset(split=args.split)
        num_samples = min(args.max_samples, len(dataset))

        total_time = 0.0
        for idx in range(num_samples):
            img_tensor, label_a = dataset[idx]
            t0 = time.perf_counter()
            feats = pipeline.extract(img_tensor)
            dt = time.perf_counter() - t0
            total_time += dt

            all_finite = all(math.isfinite(v) for v in feats.values())
            print(
                f"Sample {idx:02d} | Label_A={label_a} | Features={len(feats)} | "
                f"Time={dt*1000:.2f} ms | Finite={all_finite}"
            )
            if not all_finite:
                print(f"ERROR: Non-finite feature detected in sample {idx}!")
                sys.exit(1)

        print(f"\nAverage extraction time: {total_time / num_samples * 1000:.2f} ms/image")

    print("\n" + "=" * 70)
    print("Pipeline Execution Succeeded.")
    print("=" * 70)


if __name__ == "__main__":
    main()
