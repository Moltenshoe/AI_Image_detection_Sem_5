# Block 1 — Data Loading and Preprocessing

## Purpose

Block 1 converts raw Defactify images into the canonical representation used by every downstream detector component.

## Dataset
 
Primary dataset: Defactify (`Rajarshi-Roy-research/Defactify_Image_Dataset`).

Complete dataset composition (96,000 images total):

- Total: 96,000 images
- Train: 42,000 images (materialized at `data/processed/train/`)
- Validation: 9,000 images (processed on-demand or materialized in later phase)
- Test: 45,000 images (processed on-demand or materialized in later phase)
- Class balance: 16,000 real (7k train, 1.5k val, 7.5k test), 80,000 AI-generated (35k train, 7.5k val, 37.5k test)
- Five AI generators (16,000 images each across all splits):
  - 0: Real
  - 1: Stable Diffusion 2.1
  - 2: Stable Diffusion XL
  - 3: Stable Diffusion 3
  - 4: DALL-E 3
  - 5: Midjourney v6

`Label_A` is the binary real/fake target (0=real, 1=AI).

`Label_B` is evaluation metadata (0..5 generator ID) and must not be passed into detector features.

## Canonical transformation

```text
Raw image bytes
  ↓
largest centered square crop (box = (left, top, left+side, top+side), side = min(W, H))
  ↓
resize to 256×256 (cv2.INTER_AREA downsampling)
  ↓
RGB uint8 (shape: 256×256×3)
  ↓
stored processed data (data/processed/train/)
```

Rules:

- deterministic centered crop
- crop before resize
- `cv2.INTER_AREA` area-based downsampling
- no padding / letterboxing
- no random cropping or random augmentation
- no aspect-ratio distortion
- no ImageNet normalization for common forensic extraction
- all metadata (`path`, `caption`, `label_b`) excluded from image Parquet columns

## Forbidden detector inputs

The detector must not derive information from:

- caption
- file path
- filename
- generator identity
- train/validation/test split
- EXIF
- JPEG container metadata
- original JPEG quantization tables
- file size
- source IDs

## Data integrity

Raw Defactify files (`data/defactify/`) are immutable project inputs.

Do not alter the raw dataset during experiments.

Large raw and processed data are excluded from Git commits via `.gitignore`.

## Block output & current status

- **Materialized Training Split:** `data/processed/train/` contains 42,000 processed samples across 42 Parquet files + `manifest.json` (`image_rgb: binary`, `label_a: int32`).
- **Validation / Test Splits:** Currently processed on demand via `DefactifyDataset` (`src/data/loader.py`). Deterministic materialization for validation (9,000) and test (45,000) will be generated later using the same frozen pipeline after Block 2 is finalized.
- **Verification:** 23/23 tests PASS in `src/data/tests/run_tests.py`.

