# Branch A — Frequency

## Purpose

Branch A captures frequency-domain and periodicity-related evidence.

It contains two candidate families.

---

## A1 — FFT

### Count

4 features:

- `fft_low_freq_ratio`
- `fft_mid_freq_ratio`
- `fft_high_freq_ratio`
- `fft_spectral_centroid`

### Processing

The architecture specifies:

1. convert to BT.601 grayscale;
2. subtract grayscale mean;
3. apply orthonormal 2D FFT;
4. FFT shift;
5. compute normalized radial frequency;
6. calculate low/mid/high radial energy;
7. calculate spectral centroid.

These are aggregate spectral descriptors.

---

## A2 — Synthbuster-style periodicity candidates

### Count

30 features.

The candidate family uses the cross-difference:

```text
I[x,y] + I[x+1,y+1]
- I[x+1,y] - I[x,y+1]
```

followed by FFT analysis.

Specified periods:

- 2
- 4
- 8

Specified offsets:

- x
- y
- diagonal

The analysis is performed across RGB channels and includes channel high-frequency ratios.

## Scope

This is **Synthbuster-inspired**, not a reproduction of the complete Synthbuster feature set.

## Role in the pipeline

A provides:

- global spectral summaries
- periodicity-oriented evidence
- channel-specific frequency information

It may overlap with B and E1, so feature analysis must determine whether those overlaps are useful redundancy or removable redundancy.

## Canonical count

A = 34.
