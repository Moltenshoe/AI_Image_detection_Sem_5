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

The canonical forensic pipeline contains exactly five branches producing 111 features:

| Branch | Subbranch | Domain / Description | Canonical Count |
|---|---|---|---:|
| A | `A1_FFT` (4) + `A2_SYNTH` (30) | Frequency / periodicity | 34 |
| B | `B_HAAR` | Haar wavelet 3-level subband statistics | 30 |
| C_LBP | `C_LBP` | Rotation-invariant uniform LBP texture | 16 |
| D_MFR | `D_MFR` | 3×3 median filter residual statistics | 5 |
| E | `E1_DCT` (10) + `E2_RESP` (8) + `E3_PHASE` (4) + `E4_GRID` (4) | JPEG / compression-aware forensics | 26 |
| | | **Total Canonical Bank** | **111** |

All five canonical branches (A–E) and the orchestration pipeline are fully implemented and verified.

## Candidate versus selected feature

The feature lifecycle follows a strict sequence:

```text
candidate extraction (111 features)
       ↓
quality / validity checks (0 NaN, 0 Inf)
       ↓
univariate relevance analysis (effective ROC-AUC, MI)
       ↓
redundancy analysis (111×111 Pearson & Spearman matrices)
       ↓
branch complementarity & generator stability diagnostics
       ↓
train-only mRMR feature selection (budgets: 111, 64, 32, 16, 8)
       ↓
persistent forensic dataset materialization (data/forensic_dataset/)
```

## Branch alternatives

Branches C and D have isolated alternative candidate implementations for ablation experiments:

Canonical:
- C_LBP = 16
- D_MFR = 5

Alternatives:
- C_GLCM = 24
- C_LBP_EDGE = 16
- D_HIGHPASS = 5
- D_LAPLACIAN = 5

These alternatives are evaluated independently in ablation studies and are not concatenated into the canonical 111-feature dataset.

---

# Part B — RGB pipeline

The RGB pipeline is intentionally separate from handcrafted forensic features.

Planned lightweight model families include:
- MobileNetV3-Small
- ShuffleNetV2

The exact representation extraction and feature-analysis/selection workflow remains to be implemented.

The final design will be documented after implementation.

---

# Block 2 Completion Status

Block 2 status is **in progress**:

- [x] Branch A implemented and verified (34 features)
- [x] Branch B implemented and verified (30 features)
- [x] Branch C implemented and verified (16 canonical features)
- [x] Branch D implemented and verified (5 canonical features)
- [x] Branch E implemented and verified (26 features)
- [x] Forensic feature analysis & mRMR selection (Branch F) complete
- [x] Persistent forensic dataset materialized (`data/forensic_dataset/`, 42,000 rows × 111 features)
- [x] `ForensicDataset` reader interface verified
- [ ] RGB representation pipeline implemented
- [ ] RGB analysis & selection complete
- [ ] Final persistent RGB dataset materialized

Block 2 is fully complete only after both the Forensic and RGB persistent datasets are stored and ready for Block 3 consumption.

