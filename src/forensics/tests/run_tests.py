"""Standalone test runner for Block 2 Forensic Feature Extraction Suites.

Executes:
- Branch A (Frequency / Periodicity) test suite
- Branch B (Wavelet / Haar DWT) test suite
- Branch C (Local Texture: C_LBP) test suite
- Branch D (MFR Residual) test suite
- Branch E (JPEG/Compression-aware) test suite
- Feature Analysis (Branch F) test suite
- Persistent Forensic Dataset (Block 2 final output) test suite
"""

import os
import sys

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import unittest

from src.forensics.tests.test_branch_a_frequency import TestBranchAFrequency
from src.forensics.tests.test_branch_b_wavelet import TestBranchBWavelet
from src.forensics.tests.test_branch_c_texture import TestBranchCTexture
from src.forensics.tests.test_branch_d_residual import TestBranchDResidual
from src.forensics.tests.test_branch_e_forensics import TestBranchEForensics
from src.forensics.tests.test_feature_analysis import TestFeatureAnalysis
from src.forensics.tests.test_forensic_dataset import TestForensicDataset


def main():
    print("======================================================================")
    print("Block 2 — Forensic Feature Extraction & Persistent Dataset Test Suite")
    print("======================================================================")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(TestBranchAFrequency),
        loader.loadTestsFromTestCase(TestBranchBWavelet),
        loader.loadTestsFromTestCase(TestBranchCTexture),
        loader.loadTestsFromTestCase(TestBranchDResidual),
        loader.loadTestsFromTestCase(TestBranchEForensics),
        loader.loadTestsFromTestCase(TestFeatureAnalysis),
        loader.loadTestsFromTestCase(TestForensicDataset),
    ])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n======================================================================")
    print(
        f"Result: {result.testsRun - len(result.failures) - len(result.errors)} passed, "
        f"{len(result.failures)} failed, {len(result.errors)} errors "
        f"out of {result.testsRun} tests."
    )
    print("======================================================================")

    if not result.wasSuccessful():
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
