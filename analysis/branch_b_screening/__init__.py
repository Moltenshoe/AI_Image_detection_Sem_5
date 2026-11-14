"""Isolated Research Harness for RGB Branch B Screening (Block 2).

This package contains candidate representations, deterministic sampling,
diagnostic metrics, test suites, and runners for the Branch B investigation:
- Candidate A: Raw NQP baseline (reference control)
- Candidate B: CVPR 2025 4-Bit Color Quantization (reference & seeded adaptation)
- Candidate B1: CVPR Noise-to-Quantization Sweep
- Candidate B2: CVPR 2x2 Factorial Clipping Disentanglement
- Candidate C: PiD ICML 2025 Deterministic Color-Space Decomposition
- Candidate D: CoDA 2026 Analytical Diagnostic Model
"""

from __future__ import annotations

__version__ = "1.0.0"
