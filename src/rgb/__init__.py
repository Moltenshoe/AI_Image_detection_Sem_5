"""Block 2 — RGB Representation Pipeline.

Modules:
    - branch_a_lab: Lab Local Correlation Maps (CHROMA-inspired).
    - branch_b_nqp: Noise Quantization Probe (CoDA-inspired).
    - branch_c_validation: Final Representation Validation (Quality Control).
    - pipeline: RGBPipeline orchestrator.
    - dataset: RGBDataset Block 3 integration reader.
    - materializer: Materialization engine for derived representations.
"""

from .branch_a_lab import extract_branch_a_lab_maps, rgb_to_cielab
from .branch_b_nqp import (
    compute_branch_b_residual,
    compute_nqp_residual,
    compute_pid_residual,
    run_nqp_pilot,
)
from .branch_c_validation import (
    ValidationResult,
    generate_validation_report,
    validate_branch_b_map,
    validate_dataset_alignment,
    validate_lab_maps,
    validate_nqp_map,
    validate_pid_map,
)
from .dataset import RGBDataset
from .materializer import materialize_rgb_representations
from .pipeline import RGBPipeline

__all__ = [
    "RGBPipeline",
    "RGBDataset",
    "extract_branch_a_lab_maps",
    "compute_branch_b_residual",
    "compute_pid_residual",
    "compute_nqp_residual",
    "run_nqp_pilot",
    "materialize_rgb_representations",
    "validate_lab_maps",
    "validate_branch_b_map",
    "validate_pid_map",
    "validate_nqp_map",
    "validate_dataset_alignment",
    "generate_validation_report",
    "ValidationResult",
]

