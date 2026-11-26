"""Master Test Runner for Block 2 — RGB Pipeline.

Executes all unit and integration test suites across:
    - Branch A: Lab Local Correlation Maps (test_branch_a_lab.py)
    - Branch B: Noise Quantization Probe (test_branch_b_nqp.py)
    - Branch C: Representation Validation (test_branch_c_validation.py)

Usage:
    .venv/bin/python src/rgb/tests/run_tests.py
"""

from __future__ import annotations

import os
import sys
import time

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import unittest


def run_rgb_test_suite() -> int:
    """Run all RGB pipeline unit tests and return exit code."""
    print("=" * 70)
    print("Block 2 — RGB Pipeline Test Suite")
    print("=" * 70)

    start_time = time.time()
    loader = unittest.TestLoader()

    suite = unittest.TestSuite()

    # Import test cases explicitly
    from src.rgb.tests.test_branch_a_lab import TestBranchALab
    from src.rgb.tests.test_branch_b_nqp import TestBranchBNQP
    from src.rgb.tests.test_branch_c_validation import TestBranchCValidation
    from src.rgb.tests.test_dataset import TestRGBDataset

    suite.addTests(loader.loadTestsFromTestCase(TestBranchALab))
    suite.addTests(loader.loadTestsFromTestCase(TestBranchBNQP))
    suite.addTests(loader.loadTestsFromTestCase(TestBranchCValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestRGBDataset))


    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    elapsed = time.time() - start_time
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print("\n" + "=" * 70)
    print(f"RGB Pipeline Test Results: {passed} passed, {failures} failed, {errors} errors out of {total} tests ({elapsed:.2f}s).")
    print("=" * 70 + "\n")

    return 0 if (failures == 0 and errors == 0) else 1


if __name__ == "__main__":
    sys.exit(run_rgb_test_suite())
