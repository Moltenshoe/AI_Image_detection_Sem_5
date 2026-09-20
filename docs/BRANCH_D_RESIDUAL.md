# Branch D — Residual / Noise

## Purpose

Branch D attempts to expose image information that is less visible in the low-frequency/intensity image by constructing residual representations.

Three candidates exist.

---

# D_HIGHPASS

Count: 5.

```text
R = Y - GaussianBlur(Y)
```

Configuration:

- BT.601 grayscale
- Gaussian 5×5
- σ = 1
- replicate padding

Features:

- absolute mean
- standard deviation
- energy
- kurtosis
- entropy

---

# D_LAPLACIAN

Count: 5.

Uses the 3×3 8-neighbor Laplacian:

```text
 1   1   1
 1  -8   1
 1   1   1
```

Uses the same five summary statistics.

---

# D_MFR — Canonical

Count: 5.

```text
R = Y - MedianFilter3×3(Y)
```

Uses the same five summary statistics.

D_MFR is the canonical D branch in the main forensic configuration.

---

## Scientific role

Residual representations can expose high-frequency/noise-like structure that is not directly represented by raw RGB statistics.

Potential overlap exists with:

- B high-frequency wavelet bands
- C local texture
- E compression-response features

The project therefore treats residual extraction as a candidate evidence family rather than automatically assuming unique information.

## Canonical choice

D_MFR = 5.
