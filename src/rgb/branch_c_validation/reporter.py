"""Validation reporting for Branch C.

Compiles validation checks into structured reports (JSON and Markdown).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .validator import ValidationResult


def generate_validation_report(
    results: List[ValidationResult],
    output_path: Optional[str] = None,
    dataset_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate structured summary report from validation results."""
    all_passed = all(r.passed for r in results)

    report = {
        "status": "PASS" if all_passed else "FAIL",
        "total_checks": len(results),
        "passed_checks": sum(1 for r in results if r.passed),
        "failed_checks": sum(1 for r in results if not r.passed),
        "dataset_summary": dataset_summary or {},
        "check_details": [r.to_dict() for r in results],
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report
