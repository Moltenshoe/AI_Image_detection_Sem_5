"""CLI Tool for Materializing the Persistent Forensic Dataset (Block 2).

Usage:
    .venv/bin/python src/forensics/materialize_forensic_dataset.py [--max-samples N] [--output-dir data/forensic_dataset] [--overwrite]
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.forensics.dataset_materializer import materialize_forensic_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize persistent canonical 111-feature forensic dataset (Block 2)."
    )
    parser.add_argument(
        "--processed-train-dir",
        type=str,
        default="data/processed/train",
        help="Path to Block 1 materialized train Parquet directory.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/defactify/data",
        help="Path to raw Defactify Parquet directory (for source metadata).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/forensic_dataset",
        help="Output directory for persistent forensic dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to extract (None = full dataset).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Row group write batch size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing forensic dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = materialize_forensic_dataset(
        processed_train_dir=args.processed_train_dir,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print("\nMaterialization summary:")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Features: {summary['feature_count']}")
    print(f"  Metadata columns: {summary['metadata_column_count']}")
    print(f"  Real samples: {summary['class_distribution']['real_label_0']}")
    print(f"  AI samples: {summary['class_distribution']['ai_label_1']}")
    print("Forensic dataset materialization complete.")


if __name__ == "__main__":
    main()
