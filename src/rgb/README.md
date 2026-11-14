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
├── dataset.py                 # RGBDataset reader interface for Block 3 ablations (A, B, C, D)
├── materializer.py            # Storage materialization engine under data/rgb/
├── README.md                  # This documentation
└── tests/                     # Unit and integration test suites
    ├── test_branch_a_lab.py
    ├── test_branch_b_nqp.py
    ├── test_branch_c_validation.py
    └── run_tests.py
```

## 3. Data Contracts

- **Block 1 Source of Truth:** Images are read from Block 1 (`data/processed/train/` or raw Defactify on-demand loader).
- **Sample ID Alignment:** Each derived map is indexed by Block 1's stable `image_id` (e.g. `train_000000`).
- **Zero Image Duplication:** Only derived spatial maps and manifest metadata are saved under `data/rgb/`.
- **Ablation Configurations (Block 3):**
  - **Experiment A:** Original RGB only ($[3, 256, 256]$)
  - **Experiment B:** Original RGB + Lab maps ($[6, 256, 256]$)
  - **Experiment C:** Original RGB + PiD map ($[6, 256, 256]$, Branch B $s=0.2$)
  - **Experiment D:** Original RGB + Lab maps + PiD map ($[9, 256, 256]$)

## 4. Branch B Representation Decision (DEC-015)

Following the isolated 100-image numerical screening across 8 configurations and the Stage 2 comparative analysis:
- **Active Representation:** Projected Inversion Dynamics (PiD) with ITU-R BT.601 YUV conversion, scaling $s=0.2$, integer rounding, inverse projection ($1/s$), and signed residual $\Delta = x - \hat{x} \in [-255, 255]$ was selected as the active Branch B representation (**DEC-015**). It is deterministic, non-degenerate (~15% zero fraction), and computationally fast (~1.3 ms/sample).
- **Research Retention:** The Noise Quantization Probe (NQP) implementation (`probe.py`) and screening artifacts (`analysis/branch_b_screening/`) are retained for historical auditability and reproducibility.

## 5. Running Tests

```bash
.venv/bin/python src/rgb/tests/run_tests.py
```
