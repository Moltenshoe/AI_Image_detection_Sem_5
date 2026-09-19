#!/usr/bin/env python3
"""
Audit the locally downloaded Defactify Image Dataset.

Expected local layout:
    Minor/
    ├── data/
    │   └── defactify/
    │       └── data/
    │           ├── train-*.parquet
    │           ├── validation-*.parquet
    │           └── test-*.parquet
    └── src/
        └── analysis/
            └── audit_defactify.py

The script is intentionally read-only:
- It never modifies the dataset.
- It does not extract images to disk.
- It never loads the entire Image column into RAM.
- It reads metadata columns in batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyarrow\n"
        "Install with: pip install pyarrow"
    ) from exc

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: Pillow\n"
        "Install with: pip install Pillow"
    ) from exc


EXPECTED_TOTALS = {
    "train": 42_000,
    "validation": 9_000,
    "test": 45_000,
}

EXPECTED_LABEL_A = {0, 1}
EXPECTED_LABEL_B = {0, 1, 2, 3, 4, 5}

LABEL_A_NAMES = {
    0: "real",
    1: "ai",
}

LABEL_B_NAMES = {
    0: "real",
    1: "sd21",
    2: "sdxl",
    3: "sd3",
    4: "dalle3",
    5: "midjourney_v6",
}

EXPECTED_COLUMNS = {"Caption", "Image", "Label_A", "Label_B"}

DEFAULT_SAMPLE_PER_LABEL = 12
DEFAULT_BATCH_SIZE = 64
DEFAULT_DUPLICATE_SAMPLE = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the local Defactify Image Dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/defactify"),
        help="Defactify dataset directory (default: data/defactify)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/analysis/defactify_audit_report.json"),
        help="JSON report path.",
    )
    parser.add_argument(
        "--sample-per-label",
        type=int,
        default=DEFAULT_SAMPLE_PER_LABEL,
        help="Images to inspect per Label_B value (default: 12).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Parquet batch size while scanning (default: 64).",
    )
    parser.add_argument(
        "--duplicate-sample",
        type=int,
        default=DEFAULT_DUPLICATE_SAMPLE,
        help=(
            "Number of images to hash for exact-duplicate sampling. "
            "Set 0 to disable. Default: 2000."
        ),
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def discover_split_files(data_dir: Path) -> dict[str, list[Path]]:
    """
    Supports both:
        data/defactify/*.parquet
    and:
        data/defactify/data/*.parquet
    """
    if not data_dir.exists():
        fail(f"Dataset directory does not exist: {data_dir}")

    candidates = sorted(data_dir.rglob("*.parquet"))
    if not candidates:
        fail(f"No Parquet files found under: {data_dir}")

    split_files: dict[str, list[Path]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for path in candidates:
        name = path.name.lower()
        matched = False
        for split in split_files:
            if name.startswith(f"{split}-") or name.startswith(f"{split}_"):
                split_files[split].append(path)
                matched = True
                break

        if not matched:
            print(f"WARNING: ignoring unrecognized Parquet file: {path}")

    missing = [s for s, files in split_files.items() if not files]
    if missing:
        fail(f"Missing expected split(s): {', '.join(missing)}")

    return split_files


def schema_to_json(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


def parquet_file_info(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_gib": path.stat().st_size / (1024**3),
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "schema": schema_to_json(pf.schema_arrow),
    }


def extract_image_bytes(value: Any) -> bytes | None:
    """
    Defactify's Image column is normally an Arrow struct such as:
        {"bytes": b"...", "path": "..."}
    but this also handles raw binary columns.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is None:
            raw = value.get("data")
        if raw is None:
            return None
        if isinstance(raw, memoryview):
            return raw.tobytes()
        if isinstance(raw, bytearray):
            return bytes(raw)
        if isinstance(raw, bytes):
            return raw
        return bytes(raw)

    if isinstance(value, memoryview):
        return value.tobytes()

    if isinstance(value, bytearray):
        return bytes(value)

    if isinstance(value, bytes):
        return value

    return None


def image_info(raw: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(raw)) as img:
        # verify() checks the file structure without keeping the decoded image.
        img.verify()

    with Image.open(BytesIO(raw)) as img:
        return {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format,
            "encoded_bytes": len(raw),
        }


def stable_caption_key(caption: Any) -> str | None:
    if caption is None:
        return None
    if isinstance(caption, str):
        return caption
    return str(caption)


def update_counter(counter: Counter, value: Any) -> None:
    if value is None:
        counter["NULL"] += 1
    else:
        try:
            counter[int(value)] += 1
        except (TypeError, ValueError):
            counter[str(value)] += 1


def scan_metadata(
    split_files: dict[str, list[Path]],
    batch_size: int,
    sample_per_label: int,
    duplicate_sample: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Scan only Caption/Label_A/Label_B for the complete dataset.

    Images are not read in this pass.
    """
    metadata = {
        "total_rows": 0,
        "split_rows": {},
        "label_a": Counter(),
        "label_b": Counter(),
        "label_a_x_label_b": Counter(),
        "nulls": {
            "Caption": 0,
            "Image": 0,
            "Label_A": 0,
            "Label_B": 0,
        },
        "caption_counter": Counter(),
        "caption_label_a": defaultdict(set),
        "caption_label_b": defaultdict(set),
        "caption_image_counts": Counter(),
        "sample_image_rows": defaultdict(list),
        "files": [],
    }

    # Deterministic first-N sample per generator. This avoids storing image data.
    sample_counts = Counter()

    # For exact-duplicate checking, select deterministic row positions.
    duplicate_candidates: list[tuple[str, Path, int, int]] = []
    global_sample_counter = 0

    for split, files in split_files.items():
        split_rows = 0

        for path in files:
            pf = pq.ParquetFile(path)

            available = set(pf.schema_arrow.names)
            missing = EXPECTED_COLUMNS - available
            if missing:
                fail(
                    f"{path} is missing expected column(s): "
                    f"{', '.join(sorted(missing))}"
                )

            metadata["files"].append(parquet_file_info(path))

            # Image is included only to count nulls. It is not materialized as
            # a full table; batches are small and immediately discarded.
            columns = ["Caption", "Image", "Label_A", "Label_B"]

            file_row = 0
            for batch in pf.iter_batches(
                batch_size=batch_size,
                columns=columns,
                use_threads=True,
            ):
                rows = batch.to_pylist()

                for row in rows:
                    caption = row.get("Caption")
                    image = row.get("Image")
                    label_a = row.get("Label_A")
                    label_b = row.get("Label_B")

                    split_rows += 1
                    metadata["total_rows"] += 1

                    if caption is None:
                        metadata["nulls"]["Caption"] += 1
                    else:
                        caption_key = stable_caption_key(caption)
                        metadata["caption_counter"][caption_key] += 1
                        metadata["caption_image_counts"][caption_key] += 1
                        if label_a is not None:
                            try:
                                metadata["caption_label_a"][caption_key].add(
                                    int(label_a)
                                )
                            except (TypeError, ValueError):
                                pass
                        if label_b is not None:
                            try:
                                metadata["caption_label_b"][caption_key].add(
                                    int(label_b)
                                )
                            except (TypeError, ValueError):
                                pass

                    if image is None:
                        metadata["nulls"]["Image"] += 1
                    if label_a is None:
                        metadata["nulls"]["Label_A"] += 1
                    if label_b is None:
                        metadata["nulls"]["Label_B"] += 1

                    update_counter(metadata["label_a"], label_a)
                    update_counter(metadata["label_b"], label_b)

                    if label_a is not None and label_b is not None:
                        try:
                            metadata["label_a_x_label_b"][
                                (int(label_a), int(label_b))
                            ] += 1
                        except (TypeError, ValueError):
                            pass

                    if label_b is not None:
                        try:
                            lb = int(label_b)
                            if sample_counts[lb] < sample_per_label:
                                metadata["sample_image_rows"][lb].append(
                                    {
                                        "split": split,
                                        "file": str(path),
                                        "row_in_file": file_row,
                                        "label_a": label_a,
                                        "label_b": label_b,
                                    }
                                )
                                sample_counts[lb] += 1
                        except (TypeError, ValueError):
                            pass

                    # Deterministic evenly spaced sample for duplicate hashing.
                    if duplicate_sample > 0:
                        # Approximate uniformity without knowing total rows.
                        # Every Nth row is selected, with a fixed interval.
                        interval = max(
                            1,
                            math.ceil(96_000 / duplicate_sample),
                        )
                        if global_sample_counter % interval == 0:
                            duplicate_candidates.append(
                                (
                                    split,
                                    path,
                                    file_row,
                                    int(label_b)
                                    if label_b is not None
                                    else -1,
                                )
                            )

                    file_row += 1
                    global_sample_counter += 1

        metadata["split_rows"][split] = split_rows

    return metadata, {
        "duplicate_candidates": duplicate_candidates[:duplicate_sample]
    }


def read_specific_image(
    path: Path,
    row_index: int,
    batch_size: int = 64,
) -> tuple[Any, Any, Any, Any]:
    """
    Read one row without loading the complete Parquet file.

    We stop as soon as the batch containing the requested row is reached.
    """
    pf = pq.ParquetFile(path)

    rows_seen = 0
    for batch in pf.iter_batches(
        batch_size=batch_size,
        columns=["Caption", "Image", "Label_A", "Label_B"],
        use_threads=True,
    ):
        batch_rows = batch.to_pylist()
        end = rows_seen + len(batch_rows)

        if rows_seen <= row_index < end:
            row = batch_rows[row_index - rows_seen]
            return (
                row.get("Caption"),
                row.get("Image"),
                row.get("Label_A"),
                row.get("Label_B"),
            )

        rows_seen = end

    raise IndexError(f"Row {row_index} not found in {path}")


def inspect_image_samples(
    sample_rows: dict[int, list[dict[str, Any]]],
    batch_size: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    for label_b, rows in sorted(sample_rows.items()):
        entries = []

        for item in rows:
            path = Path(item["file"])
            try:
                _, image_value, label_a, actual_label_b = read_specific_image(
                    path,
                    item["row_in_file"],
                    batch_size=batch_size,
                )
                raw = extract_image_bytes(image_value)

                if raw is None:
                    entries.append(
                        {
                            **item,
                            "error": "Could not extract image bytes",
                        }
                    )
                    continue

                info = image_info(raw)
                entries.append(
                    {
                        **item,
                        "actual_label_a": label_a,
                        "actual_label_b": actual_label_b,
                        **info,
                    }
                )
            except Exception as exc:
                entries.append(
                    {
                        **item,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        results[str(label_b)] = entries

    return results


def hash_duplicate_sample(
    candidates: list[tuple[str, Path, int, int]],
    batch_size: int,
) -> dict[str, Any]:
    """
    Hash a bounded sample of encoded image bytes.

    This is NOT a full-dataset duplicate proof. It is explicitly reported
    as a sample-based check.
    """
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors = []

    for split, path, row_index, label_b in candidates:
        try:
            _, image_value, label_a, actual_label_b = read_specific_image(
                path,
                row_index,
                batch_size=batch_size,
            )
            raw = extract_image_bytes(image_value)

            if raw is None:
                errors.append(
                    {
                        "split": split,
                        "file": str(path),
                        "row_in_file": row_index,
                        "error": "Could not extract image bytes",
                    }
                )
                continue

            digest = hashlib.sha256(raw).hexdigest()
            hashes[digest].append(
                {
                    "split": split,
                    "file": str(path),
                    "row_in_file": row_index,
                    "label_a": label_a,
                    "label_b": actual_label_b,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "split": split,
                    "file": str(path),
                    "row_in_file": row_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    duplicate_groups = [
        {"sha256": digest, "rows": rows}
        for digest, rows in hashes.items()
        if len(rows) > 1
    ]

    return {
        "sample_size": len(candidates),
        "unique_hashes": len(hashes),
        "duplicate_groups": duplicate_groups,
        "errors": errors,
        "note": (
            "This is a bounded encoded-byte sample, not a proof that the "
            "entire dataset contains no duplicates."
        ),
    }


def counter_to_json(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def validate(
    metadata: dict[str, Any],
    split_files: dict[str, list[Path]],
    sample_results: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    for split, expected in EXPECTED_TOTALS.items():
        actual = metadata["split_rows"].get(split, 0)
        if actual != expected:
            errors.append(
                f"{split}: expected {expected:,} rows, found {actual:,}"
            )

    total_expected = sum(EXPECTED_TOTALS.values())
    if metadata["total_rows"] != total_expected:
        errors.append(
            f"total: expected {total_expected:,} rows, "
            f"found {metadata['total_rows']:,}"
        )

    for value in metadata["label_a"]:
        if value != "NULL":
            try:
                if int(value) not in EXPECTED_LABEL_A:
                    errors.append(f"Unexpected Label_A value: {value}")
            except ValueError:
                errors.append(f"Non-integer Label_A value: {value}")

    for value in metadata["label_b"]:
        if value != "NULL":
            try:
                if int(value) not in EXPECTED_LABEL_B:
                    errors.append(f"Unexpected Label_B value: {value}")
            except ValueError:
                errors.append(f"Non-integer Label_B value: {value}")

    # Label_B=0 should correspond to real Label_A=0.
    for (label_a, label_b), count in metadata["label_a_x_label_b"].items():
        if label_b == 0 and label_a != 0:
            errors.append(
                f"Inconsistent labels: Label_A={label_a}, Label_B=0, "
                f"count={count}"
            )
        if label_b in {1, 2, 3, 4, 5} and label_a != 1:
            errors.append(
                f"Inconsistent labels: Label_A={label_a}, "
                f"Label_B={label_b}, count={count}"
            )

    for split, files in split_files.items():
        if not files:
            errors.append(f"No files found for split: {split}")

    # We expect at least one successful image sample for each Label_B.
    for label_b in sorted(EXPECTED_LABEL_B):
        entries = sample_results.get(str(label_b), [])
        successful = [x for x in entries if "error" not in x]
        if not successful:
            errors.append(
                f"No successfully decoded image sample for Label_B={label_b}"
            )

    return errors


def make_caption_analysis(metadata: dict[str, Any]) -> dict[str, Any]:
    caption_counter: Counter = metadata["caption_counter"]
    caption_label_a = metadata["caption_label_a"]
    caption_label_b = metadata["caption_label_b"]

    duplicate_captions = [
        (caption, count)
        for caption, count in caption_counter.items()
        if count > 1
    ]

    cross_class = 0
    cross_generator = 0

    for caption, labels in caption_label_a.items():
        if len(labels) > 1:
            cross_class += 1

    for caption, labels in caption_label_b.items():
        if len(labels) > 1:
            cross_generator += 1

    top_duplicates = sorted(
        duplicate_captions,
        key=lambda x: (-x[1], x[0]),
    )[:50]

    return {
        "total_unique_captions": len(caption_counter),
        "duplicated_caption_count": len(duplicate_captions),
        "cross_real_fake_caption_count": cross_class,
        "cross_generator_caption_count": cross_generator,
        "maximum_images_per_caption": (
            max(caption_counter.values()) if caption_counter else 0
        ),
        "top_50_duplicated_captions": [
            {"caption": caption, "count": count}
            for caption, count in top_duplicates
        ],
        "interpretation": (
            "Caption duplication is expected in this dataset because the "
            "same source prompt/caption can be paired with real and generated "
            "images. Captions must therefore not be classifier inputs."
        ),
    }


def main() -> None:
    args = parse_args()

    if args.sample_per_label <= 0:
        fail("--sample-per-label must be > 0")
    if args.batch_size <= 0:
        fail("--batch-size must be > 0")
    if args.duplicate_sample < 0:
        fail("--duplicate-sample must be >= 0")

    print("=" * 72)
    print("Defactify Dataset Audit")
    print("=" * 72)
    print(f"Dataset: {args.data_dir.resolve()}")
    print(f"Report:  {args.output.resolve()}")
    print()

    split_files = discover_split_files(args.data_dir)

    print("Discovered Parquet files:")
    for split, files in split_files.items():
        print(f"  {split}: {len(files)}")
        for path in files:
            print(f"    - {path}")
    print()

    metadata, duplicate_info = scan_metadata(
        split_files=split_files,
        batch_size=args.batch_size,
        sample_per_label=args.sample_per_label,
        duplicate_sample=args.duplicate_sample,
    )

    print("Row counts:")
    for split in ("train", "validation", "test"):
        actual = metadata["split_rows"][split]
        expected = EXPECTED_TOTALS[split]
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {split:10s}: {actual:>8,} / {expected:>8,}  [{status}]")
    print(f"  {'TOTAL':10s}: {metadata['total_rows']:>8,} / 96,000")
    print()

    print("Label_A:")
    for key, value in sorted(
        metadata["label_a"].items(),
        key=lambda x: str(x[0]),
    ):
        name = LABEL_A_NAMES.get(int(key), "") if key != "NULL" else ""
        print(f"  {key} ({name}): {value:,}")
    print()

    print("Label_B:")
    for key, value in sorted(
        metadata["label_b"].items(),
        key=lambda x: str(x[0]),
    ):
        name = LABEL_B_NAMES.get(int(key), "") if key != "NULL" else ""
        print(f"  {key} ({name}): {value:,}")
    print()

    print("Label_A x Label_B:")
    for (label_a, label_b), count in sorted(
        metadata["label_a_x_label_b"].items()
    ):
        print(
            f"  Label_A={label_a}, Label_B={label_b} "
            f"({LABEL_B_NAMES.get(label_b, 'unknown')}): {count:,}"
        )
    print()

    print("Null counts:")
    for key, value in metadata["nulls"].items():
        print(f"  {key:8s}: {value:,}")
    print()

    print("Caption analysis:")
    caption_analysis = make_caption_analysis(metadata)
    print(
        f"  Unique captions: {caption_analysis['total_unique_captions']:,}"
    )
    print(
        f"  Duplicated captions: "
        f"{caption_analysis['duplicated_caption_count']:,}"
    )
    print(
        f"  Captions shared across real/fake: "
        f"{caption_analysis['cross_real_fake_caption_count']:,}"
    )
    print(
        f"  Captions shared across generators: "
        f"{caption_analysis['cross_generator_caption_count']:,}"
    )
    print(
        f"  Maximum images per caption: "
        f"{caption_analysis['maximum_images_per_caption']:,}"
    )
    print()

    print("Inspecting image samples...")
    sample_results = inspect_image_samples(
        metadata["sample_image_rows"],
        batch_size=args.batch_size,
    )

    for label_key, entries in sample_results.items():
        successful = [x for x in entries if "error" not in x]
        print(
            f"  Label_B={label_key} "
            f"({LABEL_B_NAMES.get(int(label_key), 'unknown')}): "
            f"{len(successful)}/{len(entries)} decoded"
        )

    print()
    print("Checking exact duplicate hashes on bounded sample...")
    duplicate_results = hash_duplicate_sample(
        duplicate_info["duplicate_candidates"],
        batch_size=args.batch_size,
    )
    print(
        f"  Sampled: {duplicate_results['sample_size']:,} images"
    )
    print(
        f"  Duplicate hash groups: "
        f"{len(duplicate_results['duplicate_groups']):,}"
    )
    if duplicate_results["duplicate_groups"]:
        print(
            "  WARNING: duplicate encoded image bytes were found in the "
            "bounded sample."
        )
    print()

    errors = validate(
        metadata=metadata,
        split_files=split_files,
        sample_results=sample_results,
    )

    report = {
        "dataset": "Rajarshi-Roy-research/Defactify_Image_Dataset",
        "local_data_dir": str(args.data_dir.resolve()),
        "audit_status": "PASS" if not errors else "FAIL",
        "expected_totals": EXPECTED_TOTALS,
        "actual_totals": {
            **metadata["split_rows"],
            "total": metadata["total_rows"],
        },
        "label_a": counter_to_json(metadata["label_a"]),
        "label_b": counter_to_json(metadata["label_b"]),
        "label_a_x_label_b": {
            f"{a},{b}": count
            for (a, b), count in metadata["label_a_x_label_b"].items()
        },
        "nulls": metadata["nulls"],
        "files": metadata["files"],
        "caption_analysis": caption_analysis,
        "image_samples": sample_results,
        "duplicate_sample": duplicate_results,
        "validation_errors": errors,
        "expected_label_a": LABEL_A_NAMES,
        "expected_label_b": LABEL_B_NAMES,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    if errors:
        print("AUDIT FAILED")
        for error in errors:
            print(f"  - {error}")
        print(f"\nReport written to: {args.output}")
        raise SystemExit(1)

    print("AUDIT PASSED")
    print(f"Report written to: {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
