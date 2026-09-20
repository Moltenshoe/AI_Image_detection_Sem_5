# Block 1 — Data Loading and Preprocessing

## Purpose

Block 1 converts raw Defactify images into the canonical representation used by every downstream detector component.

## Dataset

Primary dataset: Defactify.

Specified dataset composition:

- 96,000 images
- 16,000 real
- 80,000 AI-generated
- five AI generators:
  - Stable Diffusion 2.1
  - Stable Diffusion XL
  - Stable Diffusion 3
  - DALL-E 3
  - Midjourney v6

`Label_A` is the binary real/fake target.

`Label_B` is evaluation metadata and must not be passed into detector features.

## Canonical transformation

```text
Raw image
  ↓
largest centered square crop
  ↓
resize to 256×256
  ↓
RGB uint8
```

Rules:

- deterministic centered crop
- crop before resize
- `cv2.INTER_AREA`
- no padding
- no letterboxing
- no random cropping
- no random augmentation
- no aspect-ratio distortion
- no ImageNet normalization for common forensic extraction

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

Raw Defactify files are immutable project inputs.

Do not alter the raw dataset during experiments.

Large raw and processed data should remain outside the normal Git source commit unless a deliberate artifact/data-storage strategy is established.

## Block output

The output is the stable processed image dataset consumed by Block 2.

No downstream branch should need to repeat raw-image loading and canonical preprocessing independently.
