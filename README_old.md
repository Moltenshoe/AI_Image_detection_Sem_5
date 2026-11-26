# AI Image Detection --- Semester 5 Minor Project

A modular research project for detecting AI-generated images using a
**data-efficient, low-complexity combination of RGB and low-level
forensic evidence**.

The project is deliberately organized into four sequential blocks. Each
block consumes a defined output from the previous block and stores its
own output so that later development and experiments do not require
repeatedly rerunning earlier stages.

> **Current project status (2026-11-14):** Block 1 and the forensic side
> of Block 2 are complete. The RGB representation implementation and
> screening comparison are complete: Branch A uses Lab local-correlation
> maps, Branch B uses PiD BT.601 YCbCr residuals at `s=0.2`, and Branch
> C provides validation. NQP screening code and results remain
> research-only. The RGB implementation passed its reported tests, but a
> final review identified three important corrections still to apply
> (PiD reconstruction clipping and missing typing imports). Full-scale
> RGB dataset materialization remains pending. Block 3 training and
> Block 4 evaluation have not started.

------------------------------------------------------------------------

## 1. Research Objective

The central research question is:

> **Can carefully selected low-level forensic evidence provide useful
> AI-image detection while minimizing training image count, feature
> count, model complexity, and inference cost, while remaining useful on
> unseen generators and compressed images?**

The project is not primarily attempting to build the largest or most
accurate detector possible. Its focus is the **performance--efficiency
trade-off**:

-   How much training data is actually needed?
-   How many features are actually useful?
-   How much redundant information exists across forensic
    representations?
-   How much model complexity is necessary?
-   How well do the resulting representations generalize to generators
    not used for training?
-   How does controlled JPEG compression affect the evidence?

The project therefore treats feature extraction, feature analysis, and
feature selection as central research components rather than merely
preprocessing.

------------------------------------------------------------------------

# 2. Four-Block Architecture

``` text
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

The important architectural property is that **Block 2 produces two
persistent datasets**:

1.  a selected **forensic dataset**
2.  a selected **RGB dataset**

Block 3 should consume these outputs rather than rebuilding Block 2.

See:

-   [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
-   [`docs/BLOCK_1_DATA_PIPELINE.md`](docs/BLOCK_1_DATA_PIPELINE.md)
-   [`docs/BLOCK_2_DUAL_PIPELINE.md`](docs/BLOCK_2_DUAL_PIPELINE.md)

------------------------------------------------------------------------

# 3. Dataset

## Primary dataset

**Defactify**

The project architecture document specifies:

-   96,000 images
-   16,000 real images
-   80,000 AI-generated images
-   five AI generators:
    -   Stable Diffusion 2.1
    -   Stable Diffusion XL
    -   Stable Diffusion 3
    -   DALL-E 3
    -   Midjourney v6

`Label_A` is the binary real/fake target.

`Label_B` is evaluation metadata and is **not detector input**.

The raw dataset is treated as immutable project input and is not
intended to be committed to Git.

See [`docs/BLOCK_1_DATA_PIPELINE.md`](docs/BLOCK_1_DATA_PIPELINE.md).

------------------------------------------------------------------------

# 4. Block 1 --- Data Loading and Preprocessing

Block 1 converts the raw dataset into the canonical image representation
used by downstream analysis.

``` text
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

-   deterministic centered crop
-   crop before resize
-   `cv2.INTER_AREA`
-   no padding/letterboxing
-   no random crop
-   no random augmentation
-   no aspect-ratio squashing
-   no common ImageNet normalization for forensic extraction

The detector must not use:

-   caption
-   path
-   filename
-   generator identity
-   split
-   EXIF
-   JPEG container metadata
-   original JPEG quantization tables
-   file size
-   source IDs

See [`docs/BLOCK_1_DATA_PIPELINE.md`](docs/BLOCK_1_DATA_PIPELINE.md).

------------------------------------------------------------------------

# 5. Block 2 --- Dual Image-Analysis Pipeline

Block 2 is the main feature/representation engineering stage.

It has two deliberately different paths.

## 5.1 Forensic pipeline

The forensic pipeline is designed around multiple low-level evidence
domains:

  ------------------------------------------------------------------------
  Branch                Evidence domain                 Canonical features
  --------------------- --------------------- ----------------------------
  A                     Frequency /                                     34
                        periodicity (A1       
                        FFT=4, A2             
                        Synthbuster=30)       

  B                     Haar wavelet (3-level                           30
                        2D DWT subbands)      

  C                     Local texture                                   16
                        (rotation-invariant   
                        uniform LBP)          

  D                     Residual / noise (3×3                            5
                        median filter         
                        residual)             

  E                     JPEG /                                          26
                        compression-aware (E1 
                        DCT=10, E2            
                        Response=8, E3        
                        Phase=4, E4 Grid=4)   

  **Total**             **Canonical forensic                       **111**
                        pool**                
  ------------------------------------------------------------------------

The **111-feature pool is fully implemented and verified**. All 111
features are materialized in the persistent dataset
`data/forensic_dataset/features.parquet` across all 42,000 training
images.

Branch C and Branch D contain alternative candidate pipelines reserved
for ablation studies:

-   C_LBP: 16 (canonical)
-   C_GLCM: 24 (alternative)
-   C_LBP_EDGE: 16 (alternative)
-   D_HIGHPASS: 5 (alternative)
-   D_LAPLACIAN: 5 (alternative)
-   D_MFR: 5 (canonical)

These alternatives are evaluated independently and not concatenated into
the canonical 111-feature dataset.

See:

-   [`docs/FORENSIC_PIPELINE.md`](docs/FORENSIC_PIPELINE.md)
-   [`docs/BRANCH_A_FREQUENCY.md`](docs/BRANCH_A_FREQUENCY.md)
-   [`docs/BRANCH_B_WAVELET.md`](docs/BRANCH_B_WAVELET.md)
-   [`docs/BRANCH_C_TEXTURE.md`](docs/BRANCH_C_TEXTURE.md)
-   [`docs/BRANCH_D_RESIDUAL.md`](docs/BRANCH_D_RESIDUAL.md)
-   [`docs/BRANCH_E_JPEG.md`](docs/BRANCH_E_JPEG.md)

## 5.2 RGB pipeline

The RGB path retains conventional spatial/image information and remains
separate from the handcrafted forensic representation.

The accepted active RGB pipeline is:

-   **Branch A --- Lab local-correlation maps:** retained as the first
    RGB representation.
-   **Branch B --- PiD BT.601 YCbCr residuals:** uses the selected scale
    `s=0.2`.
-   **Branch C --- validation:** checks the generated Branch B
    representation; it is not a separate feature-generation branch.
-   **NQP:** its implementation and screening outputs are preserved for
    reproducibility but are excluded from the active RGB pipeline.

The Branch B screening experiments and comparative analysis are
complete. The selected representations are implementation decisions for
downstream model evaluation; they are not claims of proven
detection-accuracy superiority. The implementation review reported
134/134 repository tests passing before the review corrections. Three
important fixes remain pending: explicitly clip the reconstructed image
to `[0, 255]` as specified, import `Any` in `pipeline.py`, and import
`Sequence` in `materializer.py`. Minor documentation clean-up is also
pending.

Full-scale RGB dataset materialization and output verification are still
required before Block 2 can be closed. Lightweight model candidates for
Block 3 include MobileNetV3-Small and ShuffleNetV2.

See [`docs/RGB_PIPELINE.md`](docs/RGB_PIPELINE.md) and
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

------------------------------------------------------------------------

# 6. Why the Forensic Branches Are Separate

The five forensic branches are **designed to interrogate different types
of low-level evidence**, but the project does not assume that they are
statistically independent.

This distinction is important.

For example:

-   FFT and DCT are both frequency-domain representations.
-   Haar high-frequency bands and LBP can both respond to local detail.
-   LBP and residual statistics can respond to related
    local/high-frequency structure.
-   DCT/compression response and residual behavior can be correlated.

Therefore:

> **Complementarity is a research hypothesis, not a predetermined
> result.**

The feature-analysis stage is expected to determine which features are
discriminative, redundant, unstable, generator-dependent,
compression-sensitive, or computationally expensive.

This allows the project to test whether combining branches actually
provides information that cannot be obtained from a smaller subset.

See
[`docs/FEATURE_ANALYSIS_SELECTION.md`](docs/FEATURE_ANALYSIS_SELECTION.md).

------------------------------------------------------------------------

# 7. Branch E --- JPEG / Compression-Aware Forensics

Branch E is divided into four candidate families:

``` text
E1 DCT                  10
E2 Controlled JPEG      8
E3 Phase stability      4
E4 Canonical grid       4
                       ───
                        26
```

### E1 --- DCT

Candidate features:

-   `dct_ac_mean_abs`
-   `dct_ac_energy`
-   `dct_ac_kurtosis`
-   `dct_sparsity_ratio`
-   `dct_low_freq_ratio`
-   `dct_mid_freq_ratio`
-   `dct_high_freq_ratio`
-   `dct_anisotropy`
-   `dct_benford_ssd`
-   `dct_block_var_mean`

The DCT features use canonical 8×8 blocks.

`dct_sparsity_ratio` refers to floating-point canonical DCT
coefficients, not original JPEG quantization zeros.

### E2 --- Controlled JPEG response

Controlled in-memory recompression is performed at:

``` text
Q95 / Q90 / Q75 / Q60
```

Candidate features:

-   `ela_q95_mean`
-   `ela_q90_mean`
-   `ela_q75_mean`
-   `ela_q60_mean`
-   `ela_q90_energy`
-   `ela_slope_q90_q75`
-   `ela_ratio_q90_q75`
-   `ela_q90_gini`

This is a compression-response representation, not a JPEG-history
detector.

### E3 --- Phase stability

Candidate features:

-   `phase_corr_q90`
-   `phase_corr_q75`
-   `phase_diff_energy_q90`
-   `phase_hf_stability_q90`

These measure Fourier-phase behavior before/after controlled
recompression.

### E4 --- Canonical grid

Candidate features:

-   `grid_h_ratio`
-   `grid_v_ratio`
-   `grid_strength`
-   `grid_anisotropy`

The grid is the **canonical 8×8 analysis grid**. It does not recover the
original JPEG encoder's block grid.

See [`docs/BRANCH_E_JPEG.md`](docs/BRANCH_E_JPEG.md).

------------------------------------------------------------------------

# 8. Feature Analysis and Selection

After candidate extraction, the project does not assume that every
feature is useful.

The intended analysis considers:

-   per-feature discrimination
-   feature correlation/redundancy
-   mutual information
-   feature importance
-   per-generator behavior
-   compression sensitivity
-   branch ablation
-   computational cost
-   numerical stability

Selection must be fitted using **training data only**.

Validation and test information must not influence which features are
selected.

The output is a reduced feature representation suitable for Block 3.

The project is particularly interested in the relationship:

``` text
candidate feature count
        ↓
selected feature count
        ↓
model performance
        ↓
computational cost
```

See
[`docs/FEATURE_ANALYSIS_SELECTION.md`](docs/FEATURE_ANALYSIS_SELECTION.md).

------------------------------------------------------------------------

# 9. Block 2 Completion Criterion

Block 2 is considered complete only when both paths produce stable,
reproducible, persistent outputs:

``` text
                 BLOCK 2
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Forensic pipeline      RGB pipeline
          │                   │
   analysis + selection  selected representations
          │                   │
          ▼                   ▼
   FORENSIC DATASET       RGB DATASET
```

The forensic dataset is already materialized. The RGB implementation and
screening-based representation decision are complete, but the final RGB
dataset has not yet been materialized and verified across the required
splits.

These two datasets become the inputs to Block 3. Implementing feature
extraction alone does **not** complete Block 2.

------------------------------------------------------------------------

# 10. Block 3 --- Models

Block 3 consumes the outputs of Block 2.

### Forensic models

Primary:

-   LightGBM

Secondary:

-   Tiny MLP

### RGB models

Candidate lightweight CNNs:

-   MobileNetV3-Small
-   ShuffleNetV2

Optional fusion is considered only after the independent RGB and
forensic pipelines are understood.

Fusion is not assumed to improve performance.

See [`docs/BLOCK_3_MODELS.md`](docs/BLOCK_3_MODELS.md).

------------------------------------------------------------------------

# 11. Block 4 --- Evaluation and Experiments

The experimental design is organized around three main experimental
factors.

## Factor 1 --- Training-data budget

Example levels:

``` text
1k / 5k / 10k / 20k
```

Question:

> How does detection performance change as available training data
> decreases?

## Factor 2 --- Feature budget

Example levels:

``` text
111 / 64 / 32 / 16 / 8 / ...
```

The exact final feature-budget grid can be selected after the
feature-analysis stage.

Question:

> How much can the representation be reduced while retaining useful
> detection performance?

## Factor 3 --- Model type

Examples:

-   LightGBM
-   Tiny MLP
-   MobileNetV3-Small
-   ShuffleNetV2

Question:

> How does model complexity affect performance and computational cost
> when the available information is controlled?

These are **experimental factors**, not themselves metrics.

Actual evaluation metrics include:

-   ROC-AUC
-   PR-AUC
-   F1
-   TPR at a fixed FPR
-   per-generator AUC
-   mean generator AUC
-   worst-generator AUC
-   parameter count
-   model size
-   FLOPs
-   inference runtime
-   RAM usage

See:

-   [`docs/BLOCK_4_EVALUATION.md`](docs/BLOCK_4_EVALUATION.md)
-   [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md)

------------------------------------------------------------------------

# 12. Generator-Disjoint Evaluation

Generator generalization is an important control.

With five AI generators, the project can rotate the held-out generator:

``` text
Train: generators A + B + C + D
Test:  generator E
```

then repeat for each generator.

Generator identity is never supplied to the detector as an input
feature.

This prevents a random image-level split from being the only measure of
generalization.

------------------------------------------------------------------------

# 13. Compression Robustness

Compression robustness is evaluated separately from Branch E feature
extraction.

Example test qualities:

``` text
Clean
Q95
Q80
Q60
Q40
Q20
```

Possible protocols include:

### Clean → compressed

``` text
Train: clean
Test:  compressed
```

### Compression augmentation

``` text
Train: clean + compressed
Test:  compressed
```

### Cross-quality generalization

``` text
Train: clean + Q95 + Q80
Test:  Q60 + Q40 + Q20
```

The objective is to determine whether the detector retains evidence of
synthetic origin rather than merely learning JPEG artifacts.

See [`docs/BLOCK_4_EVALUATION.md`](docs/BLOCK_4_EVALUATION.md).

------------------------------------------------------------------------

# 14. Efficiency Objective

The project targets constrained hardware.

Efficiency should therefore be reported alongside detection performance:

-   number of selected features
-   model parameter count
-   model size
-   FLOPs where applicable
-   CPU/GPU inference time
-   RAM usage

The desired result is not necessarily the absolute smallest detector.

The research target is a useful **performance--efficiency trade-off**.

------------------------------------------------------------------------

# 15. Scientific Controls

The project does not claim novelty from the existence of:

-   FFT
-   DWT
-   LBP
-   GLCM
-   DCT
-   residual filters
-   LightGBM
-   lightweight CNNs
-   real-only training
-   multi-branch architecture

The research contribution is instead centered on the systematic
combination and evaluation of low-level evidence under controlled data,
feature, model, generalization, compression, and efficiency conditions.

Core controls:

-   image-only detector input
-   immutable raw dataset
-   deterministic preprocessing
-   generator-disjoint evaluation
-   symmetric transformations
-   no random augmentation during forensic extraction
-   training-only feature selection
-   explicit compression experiments
-   no metadata/container shortcuts
-   branch ablations
-   feature-budget experiments
-   data-budget experiments

See [`docs/SCIENTIFIC_CONTROLS.md`](docs/SCIENTIFIC_CONTROLS.md).

------------------------------------------------------------------------

# 16. Current Project Status

Last updated: **2026-11-14**

  ---------------------------------------------------------------------------
  Component                   Status                  Verified / Passing
  --------------------------- ----------------------- -----------------------
  Block 1 data loading &      **Complete**            23/23 tests PASS
  preprocessing                                       

  Block 1 training dataset    **Complete**            42,000 samples
  (`data/processed/train/`)                           

  Forensic Branches A--E (111 **Complete**            Existing branch tests
  features)                                           PASS

  Branch F feature analysis & **Complete**            16/16 tests PASS
  selection                                           

  Persistent forensic dataset **Complete**            42,000 rows × 111
                                                      features

  RGB Branch A --- Lab        **Implemented**         Included in reported
  local-correlation maps                              RGB suite

  RGB Branch B --- PiD BT.601 **Implemented; small    Review found one
  YCbCr residuals (`s=0.2`)   corrections pending**   specification mismatch

  RGB Branch C --- validation **Implemented**         Included in reported
                                                      RGB suite

  Branch B screening and      **Complete**            Stage 1 and Stage 2
  comparative analysis                                analyses accepted

  NQP experiments and         **Research-only**       Preserved; excluded
  implementation                                      from active RGB path

  RGB full-dataset            **Pending**             Must verify
  materialization                                     train/validation/test
                                                      outputs

  Block 2 final closeout      **Pending**             Requires RGB
                                                      materialization and
                                                      output checks

  Block 3 model training      **Not started**         \-

  Block 4 evaluation          **Not started**         \-
  ---------------------------------------------------------------------------

## Final RGB code-review actions

The implementation review found the following items. These are targeted
corrections, not a reason to repeat the experiments or redesign the
pipeline.

**Important --- fix before Block 2 closeout** 1. In
`src/rgb/branch_b_nqp/pid.py`, clip the rounded reconstructed RGB image
to `[0, 255]` before calculating the residual, matching the documented
specification. 2. Import `Any` in `src/rgb/pipeline.py` so runtime
type-hint introspection works. 3. Import `Sequence` in
`src/rgb/materializer.py` for the same reason.

**Minor --- clean up documentation** - Replace stale NQP descriptions in
RGB materializer, dataset, and validation package docstrings. - Clarify
that the BT.601 coefficients used are technically YCbCr, although
existing `YUV` identifiers may be retained where renaming would break
compatibility. - Document the input-scale heuristic edge case: a very
dark float image expressed in `[0, 255]` with maximum value near 1 can
be mistaken for normalized `[0, 1]` input. Current materializer and
dataset paths use unambiguous input conventions.

**Reviewed and accepted** - Active Branch B execution uses PiD; NQP
remains research-only. - NQP compatibility aliases currently resolve to
the same Branch B representation and are retained for compatibility. -
The review found no unrelated changes in the reported change set. - The
reported PiD CPU benchmark was approximately 1.36 ms/image on the
project laptop; treat this as a local measurement, not a universal
performance guarantee.

## Test status

The implementation task reported **134/134 tests passing** before the
final review corrections: - RGB pipeline: 22/22 - Forensic pipeline:
89/89 - Block 1 data pipeline: 23/23

The three important corrections above have not yet been applied and
retested. Therefore, 134/134 is the last reported pre-correction result,
not a post-fix verification result.

## Next steps

1.  Apply the three important code corrections and the small
    documentation clean-up.
2.  Run the RGB test suite; then run the relevant repository suites.
3.  Materialize the RGB representations for the required dataset splits
    and verify sample identity, labels, shapes, finite values, and split
    coverage.
4.  Once both persistent Block 2 datasets are ready, freeze Block 2 and
    begin Block 3 model development.

No further Branch B screening or broad paper-verification cycle is
planned unless a concrete implementation defect is discovered.

The canonical forensic pool remains **111 features**:

``` text
A Frequency:                34
B Haar Wavelet:             30
C_LBP Texture:              16
D_MFR Residual:              5
E Compression-aware:        26
──────────────────────────────
Total Canonical Pool:      111
```

No final detection-performance claim should be made until Block 4
experiments have been run.

------------------------------------------------------------------------

# 17. Repository Structure

``` text
.
├── .agents/
├── .venv/
├── analysis/
│   └── forensic_feature_analysis/
├── data/
│   ├── defactify/
│   ├── processed/
│   │   └── train/
│   └── forensic_dataset/
│       ├── features.parquet
│       ├── dataset_manifest.csv
│       ├── dataset_metadata.json
│       ├── feature_registry.csv
│       ├── feature_registry.json
│       └── selected/
│           ├── features_{8,16,32,64,111}.parquet
│           └── selected_features_{8,16,32,64,111}.json
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
│   ├── BRANCH_E_AUDIT_STATE.MD
│   ├── MODULE_MAP.md
│   └── research/
│       ├── BRANCH_E_JPEG_RESEARCH.md
│       ├── BRANCH_F_FEATURE_ANALYSIS_AND_SELECTION.md
│       └── BRANCH_F_VERIFICATION.md
├── src/
│   ├── analysis/
│   ├── data/
│   └── forensics/
│       ├── branch_a_frequency/
│       ├── branch_b_wavelet/
│       ├── branch_c_texture/
│       ├── branch_d_residual/
│       ├── branch_e/
│       ├── branch_f/
│       ├── dataset.py
│       ├── dataset_materializer.py
│       ├── materialize_forensic_dataset.py
│       ├── pipeline.py
│       ├── run_forensic_pipeline.py
│       ├── run_feature_analysis.py
│       └── tests/
├── .gitignore
├── AGENTS.md
├── CHANGELOG.md
├── DECISIONS.md
├── PROJECT.md
└── README.md
```

Large/raw datasets and virtual environments should remain excluded
through `.gitignore`.

------------------------------------------------------------------------

# 18. Research References

The research rationale and source list are maintained in:

[`docs/RESEARCH_REFERENCES.md`](docs/RESEARCH_REFERENCES.md)

Important literature includes work on:

-   frequency-domain AI-image detection
-   DCT traces
-   spectral learning
-   generator generalization
-   compression robustness
-   phase-spectrum robustness
-   JPEG/compression-response analysis
-   feature relevance/redundancy selection

------------------------------------------------------------------------

# 19. Documentation Policy

The README intentionally describes the **system architecture, scientific
objective, current status, and experimental structure**.

Detailed implementation information belongs in `docs/`.

Code-level truth should ultimately be verified against the source files
under `src/`.

When documentation and implementation disagree:

1.  inspect the implementation;
2.  update the documentation;
3.  do not silently claim that an unimplemented component is complete.

The current status in this README reflects the project's latest stated
implementation state, not the older status lines in earlier architecture
notes.
