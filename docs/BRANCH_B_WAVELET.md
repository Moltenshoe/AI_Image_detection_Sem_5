# Branch B — Haar Wavelet

## Purpose

Branch B captures multi-scale, localized and directional frequency information using a separable 2D Haar transform.

## Decomposition

Three levels are used:

```text
256 → 128 → 64 → 32
```

## Haar convention

For each 2×2 block:

```text
LL = (x00 + x01 + x10 + x11) / 2
LH = (x00 - x01 + x10 - x11) / 2
HL = (x00 + x01 - x10 - x11) / 2
HH = (x00 - x01 - x10 + x11) / 2
```

## Candidate features

Total = 30.

Components:

- LL3 energy
- LL3 entropy
- nine detail subbands:
  - LH3
  - HL3
  - HH3
  - LH2
  - HL2
  - HH2
  - LH1
  - HL1
  - HH1
- for every detail subband:
  - energy
  - absolute mean
  - entropy
- detail-energy ratio

## Implementation

The architecture specifies a pure PyTorch implementation.

## Scientific role

B differs from global FFT analysis because the wavelet representation preserves localized spatial information while separating scale and orientation.

Potential overlap with:

- A: frequency-domain evidence
- C: local structure/texture
- D: high-frequency residual behavior

This overlap is intentional enough to be tested, not assumed away.

## Canonical count

B = 30.
