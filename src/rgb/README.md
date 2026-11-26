# Block 2 — RGB Representation Pipeline (`src/rgb/`)

## 1. Overview

The RGB pipeline is the spatial representation component of Block 2 (alongside the 111-feature forensic feature bank). It derives complementary spatial representations from Block 1 images to support planned Block 3 ablation experiments.

## 2. Architecture & Modules

```text
src/rgb/
├── branch_a_lab/              # Branch A: Lab Local Correlation Maps (CHROMA-inspired)
│   ├── color_conversion.py    # Canonical RGB -> CIE Lab conversion (OpenCV float32)
│   ├── correlation.py         # Vectorized 7x7 sliding window Pearson correlation
│   ├── features.py            # Composition API (extract_branch_a_lab_maps)
│   └── __init__.py
├── branch_b_nqp/              # Branch B: Spatial Color Inversion Residuals
│   ├── pid.py                 # ACTIVE: PiD YUV residual extraction (s=0.2, DEC-015)
│   ├── probe.py               # RESEARCH: NQP core probe (stochastic baseline, DEC-014)
│   ├── pilot.py               # RESEARCH: NQP 100-image screening runner & pilot
│   └── __init__.py
├── branch_c_validation/       # Branch C: Final Representation Validation (Quality Control)
│   ├── validator.py           # Integrity, shape, range, and ID alignment validation
│   ├── reporter.py            # Summary report generation
│   └── __init__.py
├── pipeline.py                # RGBPipeline unified orchestrator (active branches A & B)
├── dataset.py                 # RGBDataset on-the-fly loader for Block 3 ablations (A, B, C, D)
├── materializer.py            # OPTIONAL: offline cache utility (data/rgb_cache/)
├── README.md                  # This documentation
└── tests/                     # Unit, integration, and benchmark test suites
    ├── test_branch_a_lab.py
    ├── test_branch_b_nqp.py
    ├── test_branch_c_validation.py
    ├── test_dataset.py
    ├── benchmark_dataloader.py
    └── run_tests.py
```

## 3. Data Contracts & On-The-Fly Workflow

- **Block 1 as Single Canonical Image Source:** All RGB models read directly from Block 1 Parquet (`data/processed/train/part-*.parquet`).
- **On-the-Fly Feature Extraction:** No redundant image dataset or pre-computed feature files are required or created on disk. Representations are extracted on demand during data loading via `RGBDataset`.
- **Zero Image Duplication:** Block 1 images are never duplicated, copied, or re-exported.
- **Sample ID Alignment:** Each sample preserves Block 1's stable `image_id` (e.g. `train_000000`) and ground-truth `label_a`.
- **Ablation Configurations (Block 3):**
  - **Experiment A:** Original RGB only ($[3, 256, 256]$)
  - **Experiment B:** Original RGB + Branch A Lab correlation maps ($[6, 256, 256]$)
  - **Experiment C:** Original RGB + Branch B PiD YCbCr residuals ($[6, 256, 256]$, $s=0.2$)
  - **Experiment D:** Original RGB + Branch A Lab + Branch B PiD ($[9, 256, 256]$)

## 4. Branch B Representation Decision (DEC-015)

Following the isolated 100-image numerical screening across 8 configurations and the Stage 2 comparative analysis:
- **Active Representation:** Projected Inversion Dynamics (PiD) with ITU-R BT.601 YCbCr conversion, scaling $s=0.2$, integer rounding, inverse projection ($1/s$), clipped reconstruction $\hat{x} \in [0, 255]$, and signed residual $\Delta = x - \hat{x}$ was selected as the active Branch B representation (**DEC-015**). It is deterministic, non-degenerate (~15% zero fraction), and computationally fast (~0.36 ms/sample).
- **Research Retention:** The Noise Quantization Probe (NQP) implementation (`probe.py`) and screening artifacts (`analysis/branch_b_screening/`) are retained for historical auditability and reproducibility.

## 5. Running Tests and Benchmarks

```bash
# Run unit and integration tests (30 tests)
.venv/bin/python src/rgb/tests/run_tests.py

# Run DataLoader performance benchmark across worker settings
.venv/bin/python src/rgb/tests/benchmark_dataloader.py
```

