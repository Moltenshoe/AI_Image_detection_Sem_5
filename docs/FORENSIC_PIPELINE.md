# Forensic Pipeline

## Objective

The forensic pipeline attempts to capture low-level evidence that may distinguish real and AI-generated images without using semantic metadata or generator identifiers.

The candidate design consists of five evidence domains:

```text
A Frequency
B Wavelet
C Texture
D Residual
E JPEG / Compression-aware
```

## Canonical pool
 
```text
A Frequency:                34 (A1 FFT=4, A2 Synthbuster=30)
B Haar Wavelet:             30 (3-level 2D Haar DWT subband statistics)
C_LBP Texture:              16 (Rotation-invariant uniform LBP)
D_MFR Residual:              5 (3×3 Median filter residual statistics)
E Compression-aware:        26 (E1 DCT=10, E2 Response=8, E3 Phase=4, E4 Grid=4)
─────────────────────────────────────────────────────────────────────────────
Total Canonical Pool:      111 features
```

The canonical 111-feature pool is fully implemented, verified, and materialized into `data/forensic_dataset/` across all 42,000 Block 1 training images.

## Scientific interpretation

The branches are **candidate evidence families**, not claims of independent information.

Possible overlap is expected:

- A and E1 both use frequency-domain information.
- B and C can both respond to local structure.
- C and D can both respond to high-frequency/local texture.
- D and E can both respond to post-processing/compression effects.

Therefore the project should measure complementarity rather than assume it.

## Feature-analysis objectives

For every candidate:

1. verify numerical validity;
2. measure class discrimination on training data;
3. inspect feature distribution;
4. measure feature redundancy;
5. inspect generator-specific behavior;
6. inspect compression sensitivity;
7. measure extraction cost;
8. assess contribution during selection/ablation.

## Selection rule

Feature selection must be fitted using training data only.

Validation and test sets are reserved for unbiased evaluation.

## Branch documentation

- A → `BRANCH_A_FREQUENCY.md`
- B → `BRANCH_B_WAVELET.md`
- C → `BRANCH_C_TEXTURE.md`
- D → `BRANCH_D_RESIDUAL.md`
- E → `BRANCH_E_JPEG.md`
