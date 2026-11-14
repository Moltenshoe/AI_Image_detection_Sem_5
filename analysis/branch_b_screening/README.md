# Isolated Research Harness: RGB Branch B Numerical Screening

This directory contains the isolated research harness for evaluating candidate representations for Block 2 RGB Branch B under strict change-control rules.

## Directory Structure
- `candidates/`: Independent implementations of candidate representations (Raw NQP, CVPR 2025, PiD 2025, CoDA diagnostic).
- `configs/`: JSON configuration files for all candidate screening experiments.
- `data/`: Immutable stratified sample selection and on-demand Parquet reader.
- `diagnostics/`: Scale-aware metrics, intensity profiles, spatial statistics, and visualizers.
- `tests/`: Comprehensive unit test suites verifying algorithms with synthetic controlled inputs.
- `runners/`: Command-line runners for candidate execution and dry-run verification.
- `manifests/`: Immutable 100-sample selection manifest (`screening_samples.csv`).
- `raw_diagnostics/`: Machine-readable per-image diagnostic metrics (JSON).
- `reports/`: Screening summary reports.
- `metadata/`: Execution environment and hardware timing metadata.
