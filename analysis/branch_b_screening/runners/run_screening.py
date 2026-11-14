"""Master Command-Line Runner for Branch B Screening Experiments.

Supports executing individual candidate experiments, ablations, or the complete
first-stage screening matrix on the 100-sample stratified manifest.
Includes a non-destructive dry-run mode for configuration and pipeline verification.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from analysis.branch_b_screening.candidates.raw_nqp import compute_raw_nqp_residual
from analysis.branch_b_screening.candidates.cvpr_color import (
    extract_cvpr_reference,
    extract_cvpr_seeded,
)
from analysis.branch_b_screening.candidates.pid_yuv import compute_pid_residual
from analysis.branch_b_screening.candidates.coda_diagnostic import compare_coda_formulations
from analysis.branch_b_screening.data.sample_reader import ScreeningSampleReader
from analysis.branch_b_screening.data.sampling import (
    get_default_manifest_path,
    verify_screening_manifest,
)
from analysis.branch_b_screening.diagnostics.metrics import (
    compute_per_image_metrics,
    aggregate_screening_metrics,
)
from analysis.branch_b_screening.diagnostics.intensity_profiles import (
    compute_intensity_conditioned_profile,
    format_intensity_profile_table,
)
from analysis.branch_b_screening.diagnostics.spatial_metrics import (
    compute_morans_i,
    compute_gini_coefficient,
)


def get_environment_metadata() -> Dict[str, Any]:
    """Capture environment and hardware configuration for change-control provenance."""
    meta: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["cuda_device_count"] = torch.cuda.device_count()
        meta["cuda_device_name"] = torch.cuda.get_device_name(0)

    # PyArrow version
    try:
        import pyarrow
        meta["pyarrow_version"] = pyarrow.__version__
    except Exception:
        meta["pyarrow_version"] = "UNKNOWN"

    # Git commit hash
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        meta["git_commit_hash"] = commit
    except Exception:
        meta["git_commit_hash"] = "UNKNOWN"

    return meta


class ScreeningRunner:
    """Manages execution, metrics collection, and reporting for candidate screening."""

    def __init__(
        self,
        manifest_csv: Optional[str] = None,
        processed_train_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        base_seed: int = 42,
    ) -> None:
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        self.project_root = project_root

        if manifest_csv is None:
            manifest_csv = get_default_manifest_path()
        if processed_train_dir is None:
            processed_train_dir = os.path.join(project_root, "data", "processed", "train")
        if output_dir is None:
            output_dir = os.path.join(project_root, "analysis", "branch_b_screening")

        self.manifest_path = manifest_csv
        self.processed_train_dir = processed_train_dir
        self.output_dir = output_dir
        self.base_seed = base_seed

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Verify manifest
        self.manifest_info = verify_screening_manifest(self.manifest_path)
        self.reader = ScreeningSampleReader(
            manifest_csv=self.manifest_path,
            processed_train_dir=self.processed_train_dir,
        )

    def run_sweep(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute noise-to-quantization step sweep (Exp B1)."""
        params = config.get("parameters", {})
        sigmas = params.get("sweep_sigmas", [0.0067, 0.0133, 0.02, 0.0333, 0.10])
        ratios = params.get("sweep_sigma_over_q", [0.1, 0.2, 0.3, 0.5, 1.5])

        sweep_results = {}
        for sigma, ratio in zip(sigmas, ratios):
            sub_config = dict(config)
            sub_params = dict(params)
            sub_params.pop("sweep_sigmas", None)
            sub_params.pop("sweep_sigma_over_q", None)
            sub_params["sigma"] = sigma
            sub_config["parameters"] = sub_params
            sub_exp_id = f"{experiment_id}_ratio_{ratio:.1f}"
            res = self.run_candidate(sub_exp_id, sub_config, max_samples=max_samples)
            sweep_results[f"sigma_over_q_{ratio}"] = res

        return {
            "experiment_id": experiment_id,
            "sweep_type": "sigma_over_q",
            "sweep_results": sweep_results,
        }

    def run_factorial(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute 2x2 factorial clipping disentanglement (Exp B2)."""
        params = config.get("parameters", {})
        matrix = params.get("factorial_matrix", [])

        factorial_results = {}
        for entry in matrix:
            sub_config = dict(config)
            sub_params = dict(params)
            sub_params.pop("factorial_matrix", None)
            sub_params["clip_noise"] = entry["clip_noise"]
            sub_params["clip_quant"] = entry["clip_quant"]
            sub_config["parameters"] = sub_params
            sub_exp_id = f"{experiment_id}_{entry['label']}"
            res = self.run_candidate(sub_exp_id, sub_config, max_samples=max_samples)
            factorial_results[entry["label"]] = res

        return {
            "experiment_id": experiment_id,
            "factorial_results": factorial_results,
        }

    def run_pid_scaling(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute PiD color-space scaling sweep across Table 5 values (Exp C)."""
        params = config.get("parameters", {})
        s_values = params.get("table_5_ablation_s_values", [0.2, 0.5, 0.8, 1.0, 1.5, 2.0])

        scaling_results = {}
        for s in s_values:
            sub_config = dict(config)
            sub_params = dict(params)
            sub_params.pop("table_5_ablation_s_values", None)
            sub_params["unscaled_reference_s"] = s
            sub_params["s"] = s
            sub_config["parameters"] = sub_params
            sub_exp_id = f"{experiment_id}_scale_{s}"
            res = self.run_candidate(sub_exp_id, sub_config, max_samples=max_samples)
            scaling_results[f"scale_{s}"] = res

        return {
            "experiment_id": experiment_id,
            "sweep_type": "pid_matrix_scaling",
            "scaling_results": scaling_results,
        }

    def run_candidate(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        max_samples: Optional[int] = None,
        save_visualizations: bool = False,
    ) -> Dict[str, Any]:
        """Execute a single candidate experiment across the screening sample set.

        Args:
            experiment_id: Experiment identifier (e.g. 'exp_a_baseline').
            config: Configuration dictionary.
            max_samples: Optional cap on sample count.
            save_visualizations: Whether to export PNG visual maps.

        Returns:
            Dict containing per-image records and aggregate statistics.
        """
        candidate_name = config.get("candidate_name")
        is_norm = config.get("is_normalized", False)
        params = config.get("parameters", {})

        # Special case: Candidate D (analytical diagnostic only)
        if candidate_name == "coda_diagnostic":
            return compare_coda_formulations(
                sigma=params.get("sigma", 0.10),
                replicas=params.get("replicas", 50),
            )

        # Dispatch sweep or factorial if specified
        if "sweep_sigmas" in params and experiment_id.startswith("exp_b1"):
            return self.run_sweep(experiment_id, config, max_samples=max_samples)
        if "factorial_matrix" in params and experiment_id.startswith("exp_b2"):
            return self.run_factorial(experiment_id, config, max_samples=max_samples)
        if "table_5_ablation_s_values" in params and experiment_id.startswith("exp_c"):
            return self.run_pid_scaling(experiment_id, config, max_samples=max_samples)

        sample_indices = [
            int(row["sample_idx"]) for _, row in self.reader.manifest_df.iterrows()
        ]
        if max_samples is not None:
            sample_indices = sample_indices[:max_samples]

        image_records: List[Dict[str, Any]] = []
        timings: List[float] = []

        print(f"[{experiment_id}] Running {len(sample_indices)} samples on device={self.device}...")

        for idx in sample_indices:
            info = self.reader.get_sample_info(idx)
            # Load raw uint8 image [3, 256, 256] or normalized float32
            img_tensor, _ = self.reader.load_image(idx, as_float=is_norm)

            t0 = time.perf_counter()

            # Execute candidate
            if candidate_name == "raw_nqp":
                res_dict = compute_raw_nqp_residual(
                    img_tensor,
                    sample_id=info["image_id"],
                    sigma=params.get("sigma", 0.10),
                    replicas=params.get("replicas", 50),
                    base_seed=self.base_seed,
                    device=self.device,
                )
            elif candidate_name == "cvpr_color_reference":
                res_dict = extract_cvpr_reference(
                    img_tensor,
                    sample_id=info["image_id"],
                    replicas=params.get("replicas", 500),
                    var=params.get("variance", 0.01),
                    L=params.get("levels", 16),
                    clip_noise=params.get("clip_noise", True),
                    clip_quant=params.get("clip_quant", True),
                )
            elif candidate_name == "cvpr_color_seeded":
                res_dict = extract_cvpr_seeded(
                    img_tensor,
                    sample_id=info["image_id"],
                    replicas=params.get("replicas", 50),
                    sigma=params.get("sigma", 0.10),
                    L=params.get("levels", 16),
                    clip_noise=params.get("clip_noise", True),
                    clip_quant=params.get("clip_quant", True),
                    base_seed=self.base_seed,
                    device=self.device,
                )
            elif candidate_name == "pid_yuv":
                s_val = params.get("s", params.get("unscaled_reference_s", 1.0))
                res_dict = compute_pid_residual(
                    img_tensor,
                    sample_id=info["image_id"],
                    s=s_val,
                    device=self.device,
                )
            else:
                raise ValueError(f"Unknown candidate name: {candidate_name}")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)

            residual = res_dict["residual"]  # [3, 256, 256]

            # Diagnostic calculations
            metrics = compute_per_image_metrics(residual, is_normalized=is_norm)
            morans = compute_morans_i(residual)
            gini = compute_gini_coefficient(residual)

            image_records.append({
                "sample_idx": idx,
                "image_id": info["image_id"],
                "label_a": int(info["label_a"]),
                "label_b": int(info["label_b"]),
                "generator_name": str(info["generator_name"]),
                "runtime_ms": float(elapsed_ms),
                "metrics": metrics,
                "morans_i": morans,
                "gini": gini,
            })

        # Aggregation
        aggregated = aggregate_screening_metrics(image_records, is_normalized=is_norm)

        # Intensity profile on sample 0
        img_0, _ = self.reader.load_image(sample_indices[0], as_float=is_norm)
        # Compute candidate residual for sample 0
        if candidate_name == "raw_nqp":
            res_0 = compute_raw_nqp_residual(img_0, sample_id="0", replicas=10)["residual"]
        elif candidate_name in ("cvpr_color_reference", "cvpr_color_seeded"):
            res_0 = extract_cvpr_seeded(img_0, sample_id="0", replicas=10)["residual"]
        elif candidate_name == "pid_yuv":
            s_val = params.get("s", params.get("unscaled_reference_s", 1.0))
            res_0 = compute_pid_residual(img_0, s=s_val)["residual"]
        else:
            res_0 = torch.zeros_like(img_0)

        intensity_profile = compute_intensity_conditioned_profile(
            img_0, res_0, is_normalized=is_norm, num_bins=16
        )

        return {
            "experiment_id": experiment_id,
            "candidate_name": candidate_name,
            "config": config,
            "manifest_checksum_sha256": self.manifest_info["checksum_sha256"],
            "total_samples_evaluated": len(sample_indices),
            "device": self.device,
            "timing_summary": {
                "mean_ms_per_image": float(np.mean(timings)),
                "p50_ms_per_image": float(np.median(timings)),
                "p95_ms_per_image": float(np.percentile(timings, 95)),
            },
            "aggregated_metrics": aggregated,
            "intensity_profile_sample_0": intensity_profile,
            "per_image_records": image_records,
        }


def main():
    parser = argparse.ArgumentParser(
        description="RGB Branch B First-Stage Numerical Screening Runner."
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="dry_run",
        choices=[
            "dry_run",
            "all",
            "exp_a",
            "exp_b_ref_500",
            "exp_b_ref_50",
            "exp_b_seeded_50",
            "exp_b1_sweep",
            "exp_b2_clipping",
            "exp_c_pid",
            "exp_d_coda",
        ],
        help="Candidate experiment to execute (default: dry_run).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to custom experiment JSON config.",
    )
    parser.add_argument(
        "--sample-manifest",
        type=str,
        default=None,
        help="Path to 100-sample screening manifest CSV.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory for screening results.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum sample count (useful for fast verification).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run configuration and pipeline verification without full workflow execution.",
    )

    args = parser.parse_args()

    runner = ScreeningRunner(
        manifest_csv=args.sample_manifest,
        output_dir=args.output_dir,
        device=args.device,
        base_seed=args.seed,
    )

    env_meta = get_environment_metadata()
    print("=" * 70)
    print("RGB Branch B Numerical Screening Runner")
    print(f"Timestamp: {env_meta['timestamp_utc']}")
    print(f"Git Commit: {env_meta['git_commit_hash']}")
    print(f"Device: {runner.device}")
    print(f"Screening Manifest: {runner.manifest_path}")
    print(f"Manifest Checksum: {runner.manifest_info['checksum_sha256']}")
    print("=" * 70)

    if args.dry_run or args.candidate == "dry_run":
        print("[DRY-RUN] Verifying configurations, manifest, and on-demand sample reader...")
        # Load sample 0 to verify read pipeline
        sample_0_idx = int(runner.reader.manifest_df["sample_idx"].iloc[0])
        img_raw, info = runner.reader.load_image(sample_0_idx, as_float=False)
        img_norm, _ = runner.reader.load_image(sample_0_idx, as_float=True)
        print(f"Successfully loaded sample_idx={sample_0_idx} ({info['image_id']}, {info['generator_name']}):")
        print(f"  Raw image shape: {list(img_raw.shape)}, dtype: {img_raw.dtype}, range: [{img_raw.min()}, {img_raw.max()}]")
        print(f"  Norm image shape: {list(img_norm.shape)}, dtype: {img_norm.dtype}, range: [{img_norm.min():.4f}, {img_norm.max():.4f}]")

        # Test smoke execution of candidates on sample 0
        res_a = compute_raw_nqp_residual(img_raw, sample_id=info["image_id"], replicas=2)
        res_b = extract_cvpr_seeded(img_norm, sample_id=info["image_id"], replicas=2)
        res_c = compute_pid_residual(img_raw, s=1.0)
        res_d = compare_coda_formulations()

        # Smoke test Exp B1 (noise sweep on sample 0 with 2 replicas)
        sweep_sigmas = [0.0067, 0.02, 0.10]
        res_b1 = [
            extract_cvpr_seeded(img_norm, sample_id=info["image_id"], replicas=2, sigma=sig)["residual"].abs().mean().item()
            for sig in sweep_sigmas
        ]

        # Smoke test Exp B2 (2x2 clipping factorial on sample 0 with 2 replicas)
        res_b2 = [
            extract_cvpr_seeded(img_norm, sample_id=info["image_id"], replicas=2, clip_noise=cn, clip_quant=cq)["residual"].abs().mean().item()
            for cn, cq in [(True, True), (True, False), (False, True), (False, False)]
        ]

        print("[DRY-RUN] Smoke execution on sample 0 passed:")
        print(f"  Exp A (Raw NQP): shape={list(res_a['residual'].shape)}, range=[{res_a['residual'].min():.4f}, {res_a['residual'].max():.4f}]")
        print(f"  Exp B (CVPR Seeded): shape={list(res_b['residual'].shape)}, range=[{res_b['residual'].min():.4f}, {res_b['residual'].max():.4f}]")
        print(f"  Exp B1 (Sweep smoke MARs): {res_b1}")
        print(f"  Exp B2 (2x2 Clipping smoke MARs): {res_b2}")
        print(f"  Exp C (PiD): shape={list(res_c['residual'].shape)}, range=[{res_c['residual'].min():.4f}, {res_c['residual'].max():.4f}]")
        print(f"  Exp D (CoDA Diag): status={res_d['diagnostic_name']}")

        # Save dry-run environment metadata
        meta_dir = os.path.join(runner.output_dir, "metadata")
        os.makedirs(meta_dir, exist_ok=True)
        env_file = os.path.join(meta_dir, "environment.json")
        with open(env_file, "w") as f:
            json.dump(env_meta, f, indent=2)
        print(f"[DRY-RUN] Saved environment metadata to: {env_file}")
        print("Dry run complete. All pipelines and configs verified successfully.")
        return 0

    # Full execution
    diagnostics_dir = os.path.join(runner.output_dir, "raw_diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    configs_dir = os.path.join(runner.output_dir, "configs")

    def make_serializable(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, torch.Tensor):
            return make_serializable(obj.detach().cpu().tolist())
        return obj

    # Determine which configurations to execute
    if args.candidate == "all":
        experiments = [
            ("exp_a_baseline", "exp_a_baseline.json"),
            ("exp_b_cvpr_seeded_50", "exp_b_cvpr_seeded_50.json"),
            ("exp_b_cvpr_reference_50", "exp_b_cvpr_reference_50.json"),
            ("exp_b_cvpr_reference_500", "exp_b_cvpr_reference_500.json"),
            ("exp_b1_sweep", "exp_b1_sweep.json"),
            ("exp_b2_clipping", "exp_b2_clipping.json"),
            ("exp_c_pid", "exp_c_pid.json"),
            ("exp_d_coda_diagnostic", "exp_d_coda_diagnostic.json"),
        ]
    elif args.config is not None:
        cfg_name = os.path.basename(args.config).replace(".json", "")
        experiments = [(cfg_name, args.config)]
    else:
        name_map = {
            "exp_a": ("exp_a_baseline", "exp_a_baseline.json"),
            "exp_b_seeded_50": ("exp_b_cvpr_seeded_50", "exp_b_cvpr_seeded_50.json"),
            "exp_b_ref_50": ("exp_b_cvpr_reference_50", "exp_b_cvpr_reference_50.json"),
            "exp_b_ref_500": ("exp_b_cvpr_reference_500", "exp_b_cvpr_reference_500.json"),
            "exp_b1_sweep": ("exp_b1_sweep", "exp_b1_sweep.json"),
            "exp_b2_clipping": ("exp_b2_clipping", "exp_b2_clipping.json"),
            "exp_c_pid": ("exp_c_pid", "exp_c_pid.json"),
            "exp_d_coda": ("exp_d_coda_diagnostic", "exp_d_coda_diagnostic.json"),
        }
        if args.candidate in name_map:
            experiments = [name_map[args.candidate]]
        else:
            raise ValueError(f"Unknown candidate argument: {args.candidate}")

    execution_start = time.time()
    results_summary = {}

    for exp_id, cfg_file in experiments:
        cfg_path = cfg_file if os.path.isabs(cfg_file) else os.path.join(configs_dir, cfg_file)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)

        print(f"\n>>> Starting execution: {exp_id} ({cfg_file})")
        t_start = time.time()
        res = runner.run_candidate(exp_id, cfg_dict, max_samples=args.max_samples)
        elapsed = time.time() - t_start
        print(f">>> Completed {exp_id} in {elapsed:.2f}s")

        # Save per-experiment raw diagnostics
        out_json_path = os.path.join(diagnostics_dir, f"{exp_id}.json")
        with open(out_json_path, "w") as f:
            json.dump(make_serializable(res), f, indent=2)
        print(f">>> Saved raw diagnostics: {out_json_path}")
        results_summary[exp_id] = {
            "elapsed_seconds": elapsed,
            "output_path": out_json_path,
        }

    total_execution_time = time.time() - execution_start

    # Save full environment and execution metadata
    meta_dir = os.path.join(runner.output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    env_meta["total_execution_time_seconds"] = total_execution_time
    env_meta["completed_experiments"] = list(results_summary.keys())
    env_file = os.path.join(meta_dir, "environment.json")
    with open(env_file, "w") as f:
        json.dump(env_meta, f, indent=2)
    print(f"\n[EXECUTION COMPLETE] Saved environment metadata to: {env_file}")
    print(f"Total screening time: {total_execution_time:.2f}s across {len(experiments)} experiment configurations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
