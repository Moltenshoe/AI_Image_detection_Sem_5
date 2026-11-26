"""Branch C — Final Representation Validation (Quality Control).

Public API:
    - validate_lab_maps: Structural and range validation for Branch A.
    - validate_branch_b_map: Structural and range validation for Branch B (PiD / NQP).
    - validate_pid_map: Canonical alias for Branch B PiD representation validation.
    - validate_nqp_map: Legacy compatibility alias for Branch B validation.
    - validate_dataset_alignment: 1:1 ID alignment with Block 1.
    - verify_read_after_write: Precision and storage verification.
    - generate_validation_report: Report generation.
    - ValidationResult: Result dataclass.
"""

from .reporter import generate_validation_report
from .validator import (
    ValidationResult,
    validate_branch_b_map,
    validate_dataset_alignment,
    validate_lab_maps,
    validate_nqp_map,
    validate_pid_map,
    verify_read_after_write,
)

__all__ = [
    "ValidationResult",
    "validate_lab_maps",
    "validate_branch_b_map",
    "validate_pid_map",
    "validate_nqp_map",
    "validate_dataset_alignment",
    "verify_read_after_write",
    "generate_validation_report",
]
