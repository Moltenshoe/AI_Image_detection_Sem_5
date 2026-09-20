# Block 2 — Dual Image Analysis Pipeline

## Purpose

Block 2 transforms the canonical images produced by Block 1 into the two representations consumed by Block 3:

1. forensic dataset
2. RGB dataset

```text
                Block 1
                   ↓
             Canonical RGB
                   │
          ┌────────┴────────┐
          ↓                 ↓
     Forensic              RGB
     pipeline             pipeline
          ↓                 ↓
  analysis/selection  analysis/selection
          ↓                 ↓
   Forensic Dataset     RGB Dataset
```

---

# Part A — Forensic pipeline

The current canonical forensic design contains five branches:

| Branch | Domain | Planned count |
|---|---|---:|
| A | Frequency / periodicity | 34 |
| B | Haar wavelet | 30 |
| C | LBP texture | 16 |
| D | MFR residual | 5 |
| E | JPEG/compression-aware | 26 |
| | **Total** | **111** |

A–D currently account for 85 implemented canonical candidates.

Branch E is still pending.

## Candidate versus selected feature

A feature being extracted does not mean that it will enter the final Block 2 forensic dataset.

The planned sequence is:

```text
candidate extraction
       ↓
quality / validity checks
       ↓
feature analysis
       ↓
redundancy analysis
       ↓
branch analysis / ablation
       ↓
feature selection
       ↓
selected forensic dataset
```

## Branch alternatives

C and D have alternative implementations for later ablation.

Canonical:

- C_LBP = 16
- D_MFR = 5

Alternatives:

- C_GLCM = 24
- C_LBP_EDGE = 16
- D_HIGHPASS = 5
- D_LAPLACIAN = 5

These alternatives should not automatically be concatenated with the canonical branch.

---

# Part B — RGB pipeline

The RGB pipeline is intentionally separate from handcrafted forensic features.

Planned lightweight model families include:

- MobileNetV3-Small
- ShuffleNetV2

The exact representation extraction and feature-analysis/selection workflow remains to be implemented.

The final design should be documented after implementation rather than retroactively inventing details.

---

# Block 2 completion

Block 2 is complete only after:

- Branch E is implemented and verified
- forensic candidate extraction is stable
- forensic analysis is complete
- forensic feature selection is complete
- RGB representation extraction is complete
- RGB analysis is complete
- RGB selection/reduction is complete
- both final datasets are stored
- both datasets are reproducible
- Block 3 can consume them without rerunning Block 1/2
