"""Unit tests for Block 2 — RGB Branch C (Final Representation Validation).

Verifies:
1. Structural, NaN/Inf, and range validation for Lab correlation maps.
2. Structural, NaN/Inf, and signed residual validation for NQP maps.
3. Dataset alignment (1:1 IDs, order, duplicate detection, missing ID detection).
4. Read-after-write verification.
5. Validation report generation.
"""

import os
import tempfile
import unittest
import torch

from src.rgb.branch_c_validation import (
    ValidationResult,
    generate_validation_report,
    validate_dataset_alignment,
    validate_lab_maps,
    validate_nqp_map,
    verify_read_after_write,
)


class TestBranchCValidation(unittest.TestCase):
    """Test suite for RGB Branch C."""

    def test_01_validate_lab_maps(self):
        """Verify Lab correlation map validation catches invalid shapes, NaNs, and bounds."""
        # 1. Valid map
        valid = torch.zeros(3, 256, 256)
        res = validate_lab_maps(valid)
        self.assertTrue(res.passed)

        # 2. Invalid shape
        bad_shape = torch.zeros(4, 256, 256)
        res_shape = validate_lab_maps(bad_shape)
        self.assertFalse(res_shape.passed)

        # 3. Contains NaN
        has_nan = torch.zeros(3, 256, 256)
        has_nan[0, 0, 0] = float("nan")
        res_nan = validate_lab_maps(has_nan)
        self.assertFalse(res_nan.passed)

        # 4. Out of bounds (> 1.0)
        out_bounds = torch.zeros(3, 256, 256)
        out_bounds[1, 10, 10] = 1.5
        res_bounds = validate_lab_maps(out_bounds)
        self.assertFalse(res_bounds.passed)

    def test_02_validate_nqp_map(self):
        """Verify Branch B residual map validation catches invalid shapes, NaNs, and bounds."""
        from src.rgb.branch_c_validation import validate_branch_b_map, validate_pid_map

        # 1. Valid signed map
        valid = torch.full((3, 256, 256), -0.02)
        res = validate_nqp_map(valid)
        self.assertTrue(res.passed)

        # Also verify through validate_branch_b_map and validate_pid_map
        self.assertTrue(validate_branch_b_map(valid).passed)
        self.assertTrue(validate_pid_map(valid).passed)

        # 2. Invalid shape
        bad_shape = torch.zeros(3, 128, 128)
        res_shape = validate_branch_b_map(bad_shape)
        self.assertFalse(res_shape.passed)

        # 3. Contains Inf
        has_inf = torch.zeros(3, 256, 256)
        has_inf[2, 5, 5] = float("inf")
        res_inf = validate_branch_b_map(has_inf)
        self.assertFalse(res_inf.passed)

        # 4. Out of bounds (> 255.0)
        out_bounds = torch.full((3, 256, 256), 300.0)
        res_bounds = validate_branch_b_map(out_bounds)
        self.assertFalse(res_bounds.passed)


    def test_03_validate_dataset_alignment(self):
        """Verify ID alignment checks detect count, duplicate, missing, and ordering mismatches."""
        b1_ids = [f"train_{i:06d}" for i in range(10)]

        # 1. Exact match
        res_ok = validate_dataset_alignment(b1_ids, b1_ids)
        self.assertTrue(res_ok.passed)

        # 2. Length mismatch
        res_len = validate_dataset_alignment(b1_ids, b1_ids[:8])
        self.assertFalse(res_len.passed)

        # 3. Duplicates
        dup_ids = b1_ids[:9] + [b1_ids[0]]
        res_dup = validate_dataset_alignment(b1_ids, dup_ids)
        self.assertFalse(res_dup.passed)

        # 4. Missing / unexpected ID
        alt_ids = b1_ids[:9] + ["train_999999"]
        res_alt = validate_dataset_alignment(b1_ids, alt_ids)
        self.assertFalse(res_alt.passed)

        # 5. Out of order
        rev_ids = list(reversed(b1_ids))
        res_order = validate_dataset_alignment(b1_ids, rev_ids)
        self.assertFalse(res_order.passed)

    def test_04_verify_read_after_write(self):
        """Verify read-after-write numerical precision check."""
        t1 = torch.rand(3, 256, 256)
        t2 = t1.clone()
        self.assertTrue(verify_read_after_write(t1, t2, atol=1e-5))

        # Small error within tolerance
        t3 = t1 + 1e-6
        self.assertTrue(verify_read_after_write(t1, t3, atol=1e-4))

        # Large error exceeding tolerance
        t4 = t1 + 1e-2
        self.assertFalse(verify_read_after_write(t1, t4, atol=1e-4))

    def test_05_generate_validation_report(self):
        """Verify validation report generation and file persistence."""
        results = [
            ValidationResult(passed=True, stage="branch_a", message="OK"),
            ValidationResult(passed=True, stage="branch_b", message="OK"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "report.json")
            report = generate_validation_report(results, output_path=out_file)

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["passed_checks"], 2)
            self.assertEqual(report["failed_checks"], 0)
            self.assertTrue(os.path.isfile(out_file))


if __name__ == "__main__":
    unittest.main()
