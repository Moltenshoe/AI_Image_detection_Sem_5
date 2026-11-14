"""Mandatory 100-Image NQP Pilot Runner (Block 2 — Branch B).

Executes the Noise Quantization Probe (NQP) pilot on representative Block 1 images
to empirically evaluate the mathematical degeneracy risk (nearly zero residuals)
on integer uint8-derived pixel data under paper settings:
    - sigma = 0.10
    - replicas = 50
    - rounding to nearest integer
    - clipping to [0, 255]

REPORTING REQUIREMENTS (per docs/RGB_PIPELINE.md):
1. Fraction of nonzero residual scalar values.
2. Fraction of pixels with any nonzero channel.
3. Mean absolute residual.
4. Maximum absolute residual.
5. Number of unique residual values.
6. Per-channel nonzero fraction and mean absolute residual.
7. Summary grouped by class and generator.
8. Save visual original/residual examples.
9. Verification of determinism and repeatability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .probe import compute_nqp_residual, derive_sample_seed


def _get_project_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )


def select_pilot_samples(
    manifest_csv: str,
    total_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Select stratified representative samples across Real and 5 AI generators.

    Stratification targets ~100 samples:
        Real (0): 20
        SD2.1 (1): 16
        SDXL (2): 16
        SD3 (3): 16
        DALL-E3 (4): 16
        Midjourney (5): 16
        Total: 100
    """
    df = pd.read_csv(manifest_csv)

    stratified_targets = {
        0: 20,
        1: 16,
        2: 16,
        3: 16,
        4: 16,
        5: 16,
    }

    selected_dfs = []
    for gen_id, count in stratified_targets.items():
        gen_df = df[df["label_b"] == gen_id]
        if len(gen_df) < count:
            sampled = gen_df
        else:
            sampled = gen_df.sample(n=count, random_state=seed)
        selected_dfs.append(sampled)

    pilot_df = pd.concat(selected_dfs).sort_values("sample_idx").reset_index(drop=True)
    return pilot_df


def load_image_by_index(
    processed_train_dir: str,
    sample_idx: int,
) -> np.ndarray:
    """Load canonical preprocessed RGB image array from Block 1 Parquet parts."""
    # Processed Parquet files have 1000 rows each
    file_idx = sample_idx // 1000
    row_in_file = sample_idx % 1000

    part_path = os.path.join(processed_train_dir, f"part-{file_idx:05d}.parquet")
    if not os.path.isfile(part_path):
        raise FileNotFoundError(f"Part file not found: {part_path}")

    # Row groups have 500 rows each
    rg_idx = row_in_file // 500
    row_in_rg = row_in_file % 500

    pf = pq.ParquetFile(part_path)
    batch = pf.read_row_group(rg_idx, columns=["image_rgb", "label_a"])
    rgb_bytes = batch.column("image_rgb").to_pylist()[row_in_rg]
    arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(256, 256, 3).copy()
    return arr


def run_nqp_pilot(
    processed_train_dir: Optional[str] = None,
    manifest_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    total_samples: int = 100,
    sigma: float = 0.10,
    replicas: int = 50,
    base_seed: int = 42,
    save_visual_examples: int = 6,
) -> Dict[str, Any]:
    """Execute the mandatory NQP 100-image pilot and return complete diagnostics."""
    project_root = _get_project_root()
    if processed_train_dir is None:
        processed_train_dir = os.path.join(project_root, "data", "processed", "train")
    if manifest_csv is None:
        manifest_csv = os.path.join(project_root, "data", "forensic_dataset", "dataset_manifest.csv")
    if output_dir is None:
        output_dir = os.path.join(project_root, "analysis", "rgb_nqp_pilot")

    os.makedirs(output_dir, exist_ok=True)
    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    pilot_df = select_pilot_samples(manifest_csv, total_samples=total_samples, seed=base_seed)

    print(f"======================================================================")
    print(f"Block 2 — Branch B: Noise Quantization Probe (NQP) 100-Image Pilot")
    print(f"======================================================================")
    print(f"Selected samples: {len(pilot_df)}")
    print(f"Configuration: sigma={sigma}, replicas={replicas}, base_seed={base_seed}")
    print(f"Output directory: {output_dir}")

    # Global tracking accumulators
    total_scalars = 0
    total_nonzero_scalars = 0
    total_pixels = 0
    total_pixels_nonzero_any_channel = 0

    abs_residuals_sum = 0.0
    global_max_abs_residual = 0.0
    unique_residual_values_set = set()

    channel_nonzero_counts = [0, 0, 0]  # R, G, B
    channel_abs_sums = [0.0, 0.0, 0.0]

    # Generator-level tracking: gen_id -> stats
    generator_stats: Dict[str, Dict[str, Any]] = {}
    class_stats: Dict[str, Dict[str, Any]] = {
        "Real": {"scalars": 0, "nonzero": 0, "abs_sum": 0.0, "max_abs": 0.0},
        "AI":   {"scalars": 0, "nonzero": 0, "abs_sum": 0.0, "max_abs": 0.0},
    }

    per_sample_records = []
    saved_examples_count = 0

    for idx, row in pilot_df.iterrows():
        sample_idx = int(row["sample_idx"])
        image_id = str(row["image_id"])
        label_a = int(row["label_a"])
        label_b = int(row["label_b"])
        gen_name = str(row["generator_name"])

        if gen_name not in generator_stats:
            generator_stats[gen_name] = {
                "generator_id": label_b,
                "label_a": label_a,
                "sample_count": 0,
                "scalars": 0,
                "nonzero_scalars": 0,
                "pixels": 0,
                "pixels_nonzero_any": 0,
                "abs_residual_sum": 0.0,
                "max_abs_residual": 0.0,
            }

        # Load image
        img_arr = load_image_by_index(processed_train_dir, sample_idx)

        # Compute NQP residual
        delta = compute_nqp_residual(
            img_arr,
            sample_id=image_id,
            sigma=sigma,
            replicas=replicas,
            base_seed=base_seed,
        ) # [3, 256, 256]

        delta_np = delta.cpu().numpy()  # [3, 256, 256]
        abs_delta = np.abs(delta_np)

        n_scalars = delta_np.size
        nonzero_mask = (delta_np != 0.0)
        n_nonzero = int(np.sum(nonzero_mask))

        pixel_nonzero_mask = np.any(nonzero_mask, axis=0) # [256, 256]
        n_pixels = pixel_nonzero_mask.size
        n_pixels_nonzero = int(np.sum(pixel_nonzero_mask))

        sample_abs_sum = float(np.sum(abs_delta))
        sample_max_abs = float(np.max(abs_delta))

        # Update unique values
        for v in np.unique(delta_np):
            unique_residual_values_set.add(round(float(v), 6))

        # Update global counts
        total_scalars += n_scalars
        total_nonzero_scalars += n_nonzero
        total_pixels += n_pixels
        total_pixels_nonzero_any_channel += n_pixels_nonzero

        abs_residuals_sum += sample_abs_sum
        if sample_max_abs > global_max_abs_residual:
            global_max_abs_residual = sample_max_abs

        # Per channel
        for c in range(3):
            c_nonzero = int(np.sum(nonzero_mask[c]))
            channel_nonzero_counts[c] += c_nonzero
            channel_abs_sums[c] += float(np.sum(abs_delta[c]))

        # Update generator stats
        g_stat = generator_stats[gen_name]
        g_stat["sample_count"] += 1
        g_stat["scalars"] += n_scalars
        g_stat["nonzero_scalars"] += n_nonzero
        g_stat["pixels"] += n_pixels
        g_stat["pixels_nonzero_any"] += n_pixels_nonzero
        g_stat["abs_residual_sum"] += sample_abs_sum
        if sample_max_abs > g_stat["max_abs_residual"]:
            g_stat["max_abs_residual"] = sample_max_abs

        # Update class stats
        c_key = "Real" if label_a == 0 else "AI"
        class_stats[c_key]["scalars"] += n_scalars
        class_stats[c_key]["nonzero"] += n_nonzero
        class_stats[c_key]["abs_sum"] += sample_abs_sum
        if sample_max_abs > class_stats[c_key]["max_abs"]:
            class_stats[c_key]["max_abs"] = sample_max_abs

        per_sample_records.append({
            "image_id": image_id,
            "sample_idx": sample_idx,
            "label_a": label_a,
            "label_b": label_b,
            "generator_name": gen_name,
            "nonzero_scalar_fraction": float(n_nonzero / n_scalars),
            "nonzero_pixel_fraction": float(n_pixels_nonzero / n_pixels),
            "mean_abs_residual": float(sample_abs_sum / n_scalars),
            "max_abs_residual": sample_max_abs,
        })

        # Save visual examples for a few samples
        if saved_examples_count < save_visual_examples:
            # Amplified residual for visualization (residual is tiny: multiply by 255 for visibility)
            # Permute to [H, W, 3] and convert RGB to BGR for cv2
            vis_orig = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            vis_delta = np.clip(np.abs(delta_np).transpose(1, 2, 0) * 2550.0, 0, 255).astype(np.uint8)
            vis_delta_bgr = cv2.cvtColor(vis_delta, cv2.COLOR_RGB2BGR)

            combined_vis = np.hstack([vis_orig, vis_delta_bgr])
            vis_path = os.path.join(examples_dir, f"example_{saved_examples_count:02d}_{image_id}_{gen_name}.png")
            cv2.imwrite(vis_path, combined_vis)
            saved_examples_count += 1

    # Determinism verification on first sample
    first_row = pilot_df.iloc[0]
    first_idx = int(first_row["sample_idx"])
    first_id = str(first_row["image_id"])
    first_arr = load_image_by_index(processed_train_dir, first_idx)

    re_delta_1 = compute_nqp_residual(first_arr, first_id, sigma=sigma, replicas=replicas, base_seed=base_seed)
    re_delta_2 = compute_nqp_residual(first_arr, first_id, sigma=sigma, replicas=replicas, base_seed=base_seed)
    repeatability_error = float((re_delta_1 - re_delta_2).abs().max().item())
    is_repeatable = (repeatability_error == 0.0)

    # Compute aggregate metrics
    overall_nonzero_scalar_fraction = float(total_nonzero_scalars / total_scalars)
    overall_nonzero_pixel_fraction = float(total_pixels_nonzero_any_channel / total_pixels)
    overall_mean_abs_residual = float(abs_residuals_sum / total_scalars)

    channel_names = ["R", "G", "B"]
    scalars_per_channel = total_scalars // 3
    per_channel_metrics = {}
    for c, c_name in enumerate(channel_names):
        per_channel_metrics[c_name] = {
            "nonzero_scalar_fraction": float(channel_nonzero_counts[c] / scalars_per_channel),
            "mean_abs_residual": float(channel_abs_sums[c] / scalars_per_channel),
        }

    # Finalize generator breakdown
    generator_summary = {}
    for g_name, g_data in generator_stats.items():
        n_sc = g_data["scalars"]
        n_px = g_data["pixels"]
        generator_summary[g_name] = {
            "generator_id": g_data["generator_id"],
            "label_a": g_data["label_a"],
            "samples": g_data["sample_count"],
            "nonzero_scalar_fraction": float(g_data["nonzero_scalars"] / n_sc) if n_sc > 0 else 0.0,
            "nonzero_pixel_fraction": float(g_data["pixels_nonzero_any"] / n_px) if n_px > 0 else 0.0,
            "mean_abs_residual": float(g_data["abs_residual_sum"] / n_sc) if n_sc > 0 else 0.0,
            "max_abs_residual": float(g_data["max_abs_residual"]),
        }

    # Finalize class breakdown
    class_summary = {}
    for c_name, c_data in class_stats.items():
        sc = c_data["scalars"]
        class_summary[c_name] = {
            "nonzero_scalar_fraction": float(c_data["nonzero"] / sc) if sc > 0 else 0.0,
            "mean_abs_residual": float(c_data["abs_sum"] / sc) if sc > 0 else 0.0,
            "max_abs_residual": float(c_data["max_abs"]),
        }

    # Degeneracy assessment
    # If fraction of nonzero scalars < 0.01% (0.0001) or mean abs residual < 1e-4:
    is_degenerate = (overall_nonzero_scalar_fraction < 0.001) or (overall_mean_abs_residual < 1e-4)

    report: Dict[str, Any] = {
        "pilot_samples_count": len(pilot_df),
        "configuration": {
            "sigma": sigma,
            "replicas": replicas,
            "base_seed": base_seed,
            "input_intensity_convention": "uint8_integer_[0,255]",
        },
        "aggregate_metrics": {
            "total_scalars": total_scalars,
            "total_nonzero_scalars": total_nonzero_scalars,
            "nonzero_scalar_fraction": overall_nonzero_scalar_fraction,
            "total_pixels": total_pixels,
            "total_pixels_nonzero_any_channel": total_pixels_nonzero_any_channel,
            "nonzero_pixel_fraction": overall_nonzero_pixel_fraction,
            "mean_abs_residual": overall_mean_abs_residual,
            "max_abs_residual": float(global_max_abs_residual),
            "num_unique_residual_values": len(unique_residual_values_set),
            "unique_residual_values_sample": sorted(list(unique_residual_values_set))[:20],
        },
        "per_channel_metrics": per_channel_metrics,
        "class_summary": class_summary,
        "generator_summary": generator_summary,
        "repeatability": {
            "is_repeatable": is_repeatable,
            "max_discrepancy": repeatability_error,
        },
        "degeneracy_evaluation": {
            "is_nearly_zero": is_degenerate,
            "finding": (
                "DEGENERATE (NEARLY ZERO): With sigma=0.10 on integer-valued [0, 255] pixels, "
                "adding N(0, 0.01) noise and rounding to nearest integer flips pixel values with "
                "probability ~ 5.7e-7. Residuals collapse to 0 almost everywhere."
                if is_degenerate else "NON-DEGENERATE: Meaningful residual variance observed."
            ),
            "recommendation": (
                "STOP full generation. The paper convention requires float values with non-integer "
                "coordinates, or noise scale sigma relative to [0, 1] normalization, or probe before integer quantization."
                if is_degenerate else "PROCEED with full generation."
            ),
        },
        "visual_examples_dir": examples_dir,
    }

    report_path = os.path.join(output_dir, "pilot_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("\n----------------------------------------------------------------------")
    print("PILOT AGGREGATE RESULTS:")
    print("----------------------------------------------------------------------")
    print(f"Nonzero scalar fraction: {overall_nonzero_scalar_fraction * 100:.6f}% ({total_nonzero_scalars} / {total_scalars})")
    print(f"Nonzero pixel fraction:  {overall_nonzero_pixel_fraction * 100:.6f}% ({total_pixels_nonzero_any_channel} / {total_pixels})")
    print(f"Mean absolute residual:  {overall_mean_abs_residual:.8f}")
    print(f"Max absolute residual:   {global_max_abs_residual:.6f}")
    print(f"Unique residual values:  {len(unique_residual_values_set)}")
    print(f"Repeatability verified:  {is_repeatable} (delta={repeatability_error})")
    print(f"Status:                  {'DEGENERATE / NEARLY ZERO' if is_degenerate else 'NON-DEGENERATE'}")
    print(f"Report written to:       {report_path}")
    print("======================================================================\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Branch B NQP 100-image pilot.")
    parser.add_argument("--samples", type=int, default=100, help="Number of pilot samples (default: 100)")
    parser.add_argument("--sigma", type=float, default=0.10, help="Noise sigma (default: 0.10)")
    parser.add_argument("--replicas", type=int, default=50, help="Replicas count (default: 50)")
    args = parser.parse_args()

    run_nqp_pilot(total_samples=args.samples, sigma=args.sigma, replicas=args.replicas)
