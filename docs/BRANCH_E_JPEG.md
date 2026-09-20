# Branch E — JPEG / Compression-Aware Forensics

## Status

**Pending implementation and code-level verification.**

This document defines the intended Branch E design. It must not be interpreted as evidence that the branch has already been implemented.

## Purpose

Branch E investigates image behavior in DCT, controlled recompression, Fourier phase, and canonical 8×8 grid-aligned structure.

It is explicitly **not** intended to be:

- a generic JPEG detector;
- an original JPEG-history detector;
- a JPEG quality estimator;
- a source encoder detector;
- a recovery system for the original JPEG block grid;
- a claim that compression artifacts themselves are proof of AI generation.

---

# E1 — DCT

Count: 10.

Features:

- `dct_ac_mean_abs`
- `dct_ac_energy`
- `dct_ac_kurtosis`
- `dct_sparsity_ratio`
- `dct_low_freq_ratio`
- `dct_mid_freq_ratio`
- `dct_high_freq_ratio`
- `dct_anisotropy`
- `dct_benford_ssd`
- `dct_block_var_mean`

Uses canonical 8×8 block DCTs.

Important interpretation:

`dct_sparsity_ratio` describes floating-point canonical DCT coefficients. It does not count original JPEG quantization zeros.

The DCT low/mid/high bands are project-defined engineering bands, not claims that the cited DCT literature established those exact bands.

The DCT feature family is motivated by DCT-trace evidence but is not a reproduction of DCT-Traces.

---

# E2 — Controlled JPEG response

Count: 8.

Controlled in-memory recompression qualities:

```text
Q95
Q90
Q75
Q60
```

Features:

- `ela_q95_mean`
- `ela_q90_mean`
- `ela_q75_mean`
- `ela_q60_mean`
- `ela_q90_energy`
- `ela_slope_q90_q75`
- `ela_ratio_q90_q75`
- `ela_q90_gini`

The same controlled procedure must be applied symmetrically to real and AI images.

The branch measures response to controlled recompression rather than trying to infer original JPEG history.

---

# E3 — Phase stability

Count: 4.

Features:

- `phase_corr_q90`
- `phase_corr_q75`
- `phase_diff_energy_q90`
- `phase_hf_stability_q90`

The design compares Fourier phase before and after controlled JPEG recompression.

Degenerate images such as constant/zero-valued images require explicit fallback behavior so that outputs remain finite and deterministic.

The implementation must document the actual Fourier-bin validity mask; wording must match code.

This is motivated by compression-phase robustness literature and is not a reproduction of the CVPR 2026 detector.

---

# E4 — Canonical 8×8 grid

Count: 4.

Features:

- `grid_h_ratio`
- `grid_v_ratio`
- `grid_strength`
- `grid_anisotropy`

The descriptors measure discontinuity aligned with the canonical analysis grid.

They do not recover the original JPEG encoder's block alignment.

---

# Branch E total

```text
E1 DCT          10
E2 response      8
E3 phase         4
E4 grid          4
-------------------
                 26
```

## Required verification before Branch E is closed

At minimum:

- exact feature count and ordering
- direct extractor vs pipeline numerical equivalence
- pathological-input numerical robustness
- no metadata/label/generator/split leakage
- symmetric real/fake processing
- full forensic regression tests
- raw Defactify integrity
- scientific wording cross-check

See `BRANCH_E_AUDIT_STATE.md` and the project source under `src/forensics/branch_e/`.
