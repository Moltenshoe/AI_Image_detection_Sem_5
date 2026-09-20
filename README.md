# AI Image Detection — Semester 5 Minor Project

A modular research project for detecting AI-generated images using a **data-efficient, low-complexity combination of RGB and low-level forensic evidence**.

The project is deliberately organized into four sequential blocks. Each block consumes a defined output from the previous block and stores its own output so that later development and experiments do not require repeatedly rerunning earlier stages.

> **Current project status (2026-09-20):** Block 1 is complete. The project is currently finishing Block 2. Forensic branches A–D are implemented; Branch E, forensic feature analysis/selection, and the RGB analysis/selection pipeline remain to be completed. Block 3 model training and Block 4 evaluation have not yet started as the main experimental phase.

---

## 1. Research Objective

The central research question is:

> **Can carefully selected low-level forensic evidence provide useful AI-image detection while minimizing training image count, feature count, model complexity, and inference cost, while remaining useful on unseen generators and compressed images?**

The project is not primarily attempting to build the largest or most accurate detector possible. Its focus is the **performance–efficiency trade-off**:

- How much training data is actually needed?
- How many features are actually useful?
- How much redundant information exists across forensic representations?
- How much model complexity is necessary?
- How well do the resulting representations generalize to generators not used for training?
- How does controlled JPEG compression affect the evidence?

The project therefore treats feature extraction, feature analysis, and feature selection as central research components rather than merely preprocessing.

---

# 2. Four-Block Architecture

```text
                           Defactify
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│ BLOCK 1 — DATA LOADING + PREPROCESSING                        │
│                                                               │
│ Raw images → deterministic crop → resize → RGB uint8          │
└──────────────────────────────┬────────────────────────────────┘
                               │
                        Stored processed data
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ BLOCK 2 — IMAGE ANALYSIS                                      │
│                                                               │
│                  ┌────────────────────────┐                   │
│                  │                        │                   │
│                  ▼                        ▼                   │
│           RGB PIPELINE             FORENSIC PIPELINE          │
│                  │                        │                   │
│           representation          A Frequency                 │
│           extraction               B Wavelet                   │
│           analysis                 C Texture                   │
│           selection                D Residual                  │
│                  │                  E JPEG / Compression       │
│                  │                        │                   │
│                  ▼                        ▼                   │
│             RGB DATASET             FORENSIC DATASET          │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ BLOCK 3 — MODELS                                              │
│                                                               │
│ RGB models + forensic models + optional controlled fusion     │
└──────────────────────────────┬────────────────────────────────┘
                               │
                         Predictions/results
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ BLOCK 4 — EVALUATION + EXPERIMENTS                            │
│                                                               │
│ Data budget × feature budget × model type                     │
│ + generator-disjoint evaluation + compression robustness     │
│ + efficiency measurements                                     │
└───────────────────────────────────────────────────────────────┘
```

The important architectural property is that **Block 2 produces two persistent datasets**:

1. a selected **forensic dataset**
2. a selected **RGB dataset**

Block 3 should consume these outputs rather than rebuilding Block 2.

See:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/BLOCK_1_DATA_PIPELINE.md`](docs/BLOCK_1_DATA_PIPELINE.md)
- [`docs/BLOCK_2_DUAL_PIPELINE.md`](docs/BLOCK_2_DUAL_PIPELINE.md)

---

# 3. Dataset

## Primary dataset

**Defactify**

The project architecture document specifies:

- 96,000 images
- 16,000 real images
- 80,000 AI-generated images
- five AI generators:
  - Stable Diffusion 2.1
  - Stable Diffusion XL
  - Stable Diffusion 3
  - DALL-E 3
  - Midjourney v6

`Label_A` is the binary real/fake target.

`Label_B` is evaluation metadata and is **not detector input**.

The raw dataset is treated as immutable project input and is not intended to be committed to Git.

See [`docs/BLOCK_1_DATA_PIPELINE.md`](docs/BLOCK_1_DATA_PIPELINE.md).

---

# 4. Block 1 — Data Loading and Preprocessing

Block 1 converts the raw dataset into the canonical image representation used by downstream analysis.

```text
Raw image
   ↓
largest centered square crop
   ↓
resize to 256 × 256
   ↓
RGB uint8
   ↓
stored processed data
```

Canonical rules:

- deterministic centered crop
- crop before resize
- `cv2.INTER_AREA`
- no padding/letterboxing
- no random crop
- no random augmentation
- no aspect-ratio squashing
- no common ImageNet normalization for forensic extraction

The detector must not use:

- caption
- path
- filename
- generator identity
- split
- EXIF
- JPEG container metadata
- original JPEG quantization tables
- file size
- source IDs

See [`docs/BLOCK_1_DATA_PIPELINE.md`](docs/BLOCK_1_DATA_PIPELINE.md).

---

# 5. Block 2 — Dual Image-Analysis Pipeline

Block 2 is the main feature/representation engineering stage.

It has two deliberately different paths.

## 5.1 Forensic pipeline

The forensic pipeline is designed around multiple low-level evidence domains:

| Branch | Evidence domain | Planned candidates |
|---|---|---:|
| A | Frequency / periodicity | 34 |
| B | Haar wavelet | 30 |
| C | Local texture | 16 canonical LBP |
| D | Residual / noise | 5 canonical MFR |
| E | JPEG / compression-aware | 26 |
| **Total** | **Canonical forensic pool** | **111** |

The **111-feature pool is the planned canonical configuration after Branch E is implemented**. The currently implemented canonical A–D portion is 85 features.

Branch C and Branch D contain alternatives for ablation:

- C_LBP: 16
- C_GLCM: 24
- C_LBP_EDGE: 16
- D_HIGHPASS: 5
- D_LAPLACIAN: 5
- D_MFR: 5

The alternatives are not automatically concatenated into the canonical pool.

See:

- [`docs/FORENSIC_PIPELINE.md`](docs/FORENSIC_PIPELINE.md)
- [`docs/BRANCH_A_FREQUENCY.md`](docs/BRANCH_A_FREQUENCY.md)
- [`docs/BRANCH_B_WAVELET.md`](docs/BRANCH_B_WAVELET.md)
- [`docs/BRANCH_C_TEXTURE.md`](docs/BRANCH_C_TEXTURE.md)
- [`docs/BRANCH_D_RESIDUAL.md`](docs/BRANCH_D_RESIDUAL.md)
- [`docs/BRANCH_E_JPEG.md`](docs/BRANCH_E_JPEG.md)

## 5.2 RGB pipeline

The RGB path retains conventional spatial/image information and is intentionally kept separate from the handcrafted forensic representation.

Its planned model candidates include:

- MobileNetV3-Small
- ShuffleNetV2

The exact RGB feature/representation extraction, analysis, and selection implementation is **not yet finalized** and must be documented after it is designed and implemented.

See [`docs/RGB_PIPELINE.md`](docs/RGB_PIPELINE.md).

---

# 6. Why the Forensic Branches Are Separate

The five forensic branches are **designed to interrogate different types of low-level evidence**, but the project does not assume that they are statistically independent.

This distinction is important.

For example:

- FFT and DCT are both frequency-domain representations.
- Haar high-frequency bands and LBP can both respond to local detail.
- LBP and residual statistics can respond to related local/high-frequency structure.
- DCT/compression response and residual behavior can be correlated.

Therefore:

> **Complementarity is a research hypothesis, not a predetermined result.**

The feature-analysis stage is expected to determine which features are discriminative, redundant, unstable, generator-dependent, compression-sensitive, or computationally expensive.

This allows the project to test whether combining branches actually provides information that cannot be obtained from a smaller subset.

See [`docs/FEATURE_ANALYSIS_SELECTION.md`](docs/FEATURE_ANALYSIS_SELECTION.md).

---

# 7. Branch E — JPEG / Compression-Aware Forensics

Branch E is divided into four candidate families:

```text
E1 DCT                  10
E2 Controlled JPEG      8
E3 Phase stability      4
E4 Canonical grid       4
                       ───
                        26
```

### E1 — DCT

Candidate features:

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

The DCT features use canonical 8×8 blocks.

`dct_sparsity_ratio` refers to floating-point canonical DCT coefficients, not original JPEG quantization zeros.

### E2 — Controlled JPEG response

Controlled in-memory recompression is performed at:

```text
Q95 / Q90 / Q75 / Q60
```

Candidate features:

- `ela_q95_mean`
- `ela_q90_mean`
- `ela_q75_mean`
- `ela_q60_mean`
- `ela_q90_energy`
- `ela_slope_q90_q75`
- `ela_ratio_q90_q75`
- `ela_q90_gini`

This is a compression-response representation, not a JPEG-history detector.

### E3 — Phase stability

Candidate features:

- `phase_corr_q90`
- `phase_corr_q75`
- `phase_diff_energy_q90`
- `phase_hf_stability_q90`

These measure Fourier-phase behavior before/after controlled recompression.

### E4 — Canonical grid

Candidate features:

- `grid_h_ratio`
- `grid_v_ratio`
- `grid_strength`
- `grid_anisotropy`

The grid is the **canonical 8×8 analysis grid**. It does not recover the original JPEG encoder's block grid.

See [`docs/BRANCH_E_JPEG.md`](docs/BRANCH_E_JPEG.md).

---

# 8. Feature Analysis and Selection

After candidate extraction, the project does not assume that every feature is useful.

The intended analysis considers:

- per-feature discrimination
- feature correlation/redundancy
- mutual information
- feature importance
- per-generator behavior
- compression sensitivity
- branch ablation
- computational cost
- numerical stability

Selection must be fitted using **training data only**.

Validation and test information must not influence which features are selected.

The output is a reduced feature representation suitable for Block 3.

The project is particularly interested in the relationship:

```text
candidate feature count
        ↓
selected feature count
        ↓
model performance
        ↓
computational cost
```

See [`docs/FEATURE_ANALYSIS_SELECTION.md`](docs/FEATURE_ANALYSIS_SELECTION.md).

---

# 9. Block 2 Completion Criterion

Block 2 is considered complete only when both paths produce stable, reproducible outputs:

```text
                 BLOCK 2
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Forensic pipeline      RGB pipeline
          │                   │
   analysis + selection analysis + selection
          │                   │
          ▼                   ▼
   FORENSIC DATASET       RGB DATASET
```

These two datasets become the inputs to Block 3.

Implementing feature extraction alone does **not** complete Block 2.

---

# 10. Block 3 — Models

Block 3 consumes the outputs of Block 2.

### Forensic models

Primary:

- LightGBM

Secondary:

- Tiny MLP

### RGB models

Candidate lightweight CNNs:

- MobileNetV3-Small
- ShuffleNetV2

Optional fusion is considered only after the independent RGB and forensic pipelines are understood.

Fusion is not assumed to improve performance.

See [`docs/BLOCK_3_MODELS.md`](docs/BLOCK_3_MODELS.md).

---

# 11. Block 4 — Evaluation and Experiments

The experimental design is organized around three main experimental factors.

## Factor 1 — Training-data budget

Example levels:

```text
1k / 5k / 10k / 20k
```

Question:

> How does detection performance change as available training data decreases?

## Factor 2 — Feature budget

Example levels:

```text
111 / 64 / 32 / 16 / 8 / ...
```

The exact final feature-budget grid can be selected after the feature-analysis stage.

Question:

> How much can the representation be reduced while retaining useful detection performance?

## Factor 3 — Model type

Examples:

- LightGBM
- Tiny MLP
- MobileNetV3-Small
- ShuffleNetV2

Question:

> How does model complexity affect performance and computational cost when the available information is controlled?

These are **experimental factors**, not themselves metrics.

Actual evaluation metrics include:

- ROC-AUC
- PR-AUC
- F1
- TPR at a fixed FPR
- per-generator AUC
- mean generator AUC
- worst-generator AUC
- parameter count
- model size
- FLOPs
- inference runtime
- RAM usage

See:

- [`docs/BLOCK_4_EVALUATION.md`](docs/BLOCK_4_EVALUATION.md)
- [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md)

---

# 12. Generator-Disjoint Evaluation

Generator generalization is an important control.

With five AI generators, the project can rotate the held-out generator:

```text
Train: generators A + B + C + D
Test:  generator E
```

then repeat for each generator.

Generator identity is never supplied to the detector as an input feature.

This prevents a random image-level split from being the only measure of generalization.

---

# 13. Compression Robustness

Compression robustness is evaluated separately from Branch E feature extraction.

Example test qualities:

```text
Clean
Q95
Q80
Q60
Q40
Q20
```

Possible protocols include:

### Clean → compressed

```text
Train: clean
Test:  compressed
```

### Compression augmentation

```text
Train: clean + compressed
Test:  compressed
```

### Cross-quality generalization

```text
Train: clean + Q95 + Q80
Test:  Q60 + Q40 + Q20
```

The objective is to determine whether the detector retains evidence of synthetic origin rather than merely learning JPEG artifacts.

See [`docs/BLOCK_4_EVALUATION.md`](docs/BLOCK_4_EVALUATION.md).

---

# 14. Efficiency Objective

The project targets constrained hardware.

Efficiency should therefore be reported alongside detection performance:

- number of selected features
- model parameter count
- model size
- FLOPs where applicable
- CPU/GPU inference time
- RAM usage

The desired result is not necessarily the absolute smallest detector.

The research target is a useful **performance–efficiency trade-off**.

---

# 15. Scientific Controls

The project does not claim novelty from the existence of:

- FFT
- DWT
- LBP
- GLCM
- DCT
- residual filters
- LightGBM
- lightweight CNNs
- real-only training
- multi-branch architecture

The research contribution is instead centered on the systematic combination and evaluation of low-level evidence under controlled data, feature, model, generalization, compression, and efficiency conditions.

Core controls:

- image-only detector input
- immutable raw dataset
- deterministic preprocessing
- generator-disjoint evaluation
- symmetric transformations
- no random augmentation during forensic extraction
- training-only feature selection
- explicit compression experiments
- no metadata/container shortcuts
- branch ablations
- feature-budget experiments
- data-budget experiments

See [`docs/SCIENTIFIC_CONTROLS.md`](docs/SCIENTIFIC_CONTROLS.md).

---

# 16. Current Project Status

| Component | Status |
|---|---|
| Block 1 data loading | Complete |
| Block 1 canonical preprocessing | Complete |
| Forensic Branch A | Implemented |
| Forensic Branch B | Implemented |
| Forensic Branch C | Implemented |
| Forensic Branch D | Implemented |
| Forensic Branch E | **Pending** |
| Forensic feature analysis | **Pending** |
| Forensic feature selection | **Pending** |
| RGB pipeline | **Pending** |
| RGB feature analysis | **Pending** |
| RGB feature selection | **Pending** |
| Block 2 final datasets | **Pending** |
| Block 3 training | Pending |
| Block 4 evaluation | Pending |

The planned canonical forensic pool becomes 111 features after E is completed:

```text
A 34
B 30
C 16
D  5
E 26
────
111
```

Current implemented canonical A–D pool:

```text
34 + 30 + 16 + 5 = 85
```

No final detection-performance claim should be made until Block 4 experiments have been run.

---

# 17. Repository Structure

```text
.
├── .agents/
├── .venv/
├── analysis_800x800/
├── data/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BLOCK_1_DATA_PIPELINE.md
│   ├── BLOCK_2_DUAL_PIPELINE.md
│   ├── FORENSIC_PIPELINE.md
│   ├── BRANCH_A_FREQUENCY.md
│   ├── BRANCH_B_WAVELET.md
│   ├── BRANCH_C_TEXTURE.md
│   ├── BRANCH_D_RESIDUAL.md
│   ├── BRANCH_E_JPEG.md
│   ├── RGB_PIPELINE.md
│   ├── FEATURE_ANALYSIS_SELECTION.md
│   ├── BLOCK_3_MODELS.md
│   ├── BLOCK_4_EVALUATION.md
│   ├── EXPERIMENTAL_PROTOCOL.md
│   ├── SCIENTIFIC_CONTROLS.md
│   ├── RESEARCH_REFERENCES.md
│   ├── PROJECT_STATUS.md
│   ├── BRANCH_E_AUDIT_STATE.md
│   └── MODULE_MAP.md
├── first_analysis_own/
├── src/
│   ├── analysis/
│   ├── data/
│   └── forensics/
│       ├── branch_a_frequency/
│       ├── branch_b_wavelet/
│       ├── branch_c_texture/
│       ├── branch_d_residual/
│       ├── branch_e/
│       ├── tests/
│       ├── pipeline.py
│       └── run_forensic_pipeline.py
├── .gitignore
├── AGENTS.md
├── CHANGELOG.md
├── DECISIONS.md
├── PROJECT.md
└── README.md
```

Large/raw datasets and virtual environments should remain excluded through `.gitignore`.

---

# 18. Research References

The research rationale and source list are maintained in:

[`docs/RESEARCH_REFERENCES.md`](docs/RESEARCH_REFERENCES.md)

Important literature includes work on:

- frequency-domain AI-image detection
- DCT traces
- spectral learning
- generator generalization
- compression robustness
- phase-spectrum robustness
- JPEG/compression-response analysis
- feature relevance/redundancy selection

---

# 19. Documentation Policy

The README intentionally describes the **system architecture, scientific objective, current status, and experimental structure**.

Detailed implementation information belongs in `docs/`.

Code-level truth should ultimately be verified against the source files under `src/`.

When documentation and implementation disagree:

1. inspect the implementation;
2. update the documentation;
3. do not silently claim that an unimplemented component is complete.

The current status in this README reflects the project's latest stated implementation state, not the older status lines in earlier architecture notes.
