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
A = 34
B = 30
C_LBP = 16
D_MFR = 5
E = 26
------------
    111
```

Before E is implemented, the canonical A–D implementation contains 85 candidates.

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
