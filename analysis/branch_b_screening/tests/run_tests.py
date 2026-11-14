"""Master test runner for Branch B research harness unit tests."""

from __future__ import annotations

import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_screening_tests() -> int:
    """Discover and execute all unit tests under analysis/branch_b_screening/tests/."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    print("=" * 70)
    print("Branch B Research Screening — Automated Unit Test Suite")
    print("=" * 70)

    result = runner.run(suite)
    print("=" * 70)
    total = result.testsRun
    failed = len(result.failures)
    errored = len(result.errors)
    passed = total - failed - errored
    print(f"Results: {passed} passed, {failed} failed, {errored} errors out of {total} tests.")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_screening_tests())
