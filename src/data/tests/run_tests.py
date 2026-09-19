"""
Standalone Block 1 test runner.
Runs:
    Suite 1 — Preprocessing tests (1–7)
    Suite 2 — Loader tests (A–G)
    Suite 3 — Materializer tests (C–K)

No pytest required. Uses only stdlib runner.

Usage:
    cd <project-root>
    .venv/bin/python src/data/tests/run_tests.py
"""
import os
import sys

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Suite 1: Preprocessing ---
from src.data.tests.test_preprocessing import (
    test_landscape_640x480,
    test_portrait_480x640,
    test_square_1024x1024_identity_crop,
    test_square_270x270_identity_crop,
    test_output_rgb_256_deterministic,
    test_metadata_isolation,
    test_raw_data_integrity,
)

# --- Suite 2: Loader ---
from src.data.tests.test_loader import (
    test_loader_A_train_length,
    test_loader_B_validation_length,
    test_loader_C_test_length,
    test_loader_D_real_parquet_sample,
    test_loader_E_both_classes,
    test_loader_F_detector_mode,
    test_loader_G_audit_mode,
)

# --- Suite 3: Materializer ---
from src.data.tests.test_materializer import (
    test_materializer_C_sample_count,
    test_materializer_D_image_shape_dtype,
    test_materializer_E_labels_correct,
    test_materializer_F_no_forbidden_metadata,
    test_materializer_G_deterministic,
    test_materializer_H_same_preprocessing_both_paths,
    test_materializer_I_corrupt_image_handling,
    test_materializer_J_overwrite_idempotent,
    test_materializer_K_manifest_counts,
)

SUITE_1 = [
    ("Test 1 — Landscape 640×480 → crop 480×480 → resize 256×256",  test_landscape_640x480),
    ("Test 2 — Portrait 480×640 → crop 480×480 → resize 256×256",   test_portrait_480x640),
    ("Test 3 — Square 1024×1024 → identity crop → 256×256",         test_square_1024x1024_identity_crop),
    ("Test 4 — Square 270×270 → identity crop → 256×256",           test_square_270x270_identity_crop),
    ("Test 5 — RGB output, 256×256, deterministic",                  test_output_rgb_256_deterministic),
    ("Test 6 — Metadata isolation (path/caption/label_b blocked)",   test_metadata_isolation),
    ("Test 7 — Raw data immutability (file-size + 64KB SHA-256)",    test_raw_data_integrity),
]

SUITE_2 = [
    ("Test A — Train split length == 42,000",                        test_loader_A_train_length),
    ("Test B — Validation split length == 9,000",                    test_loader_B_validation_length),
    ("Test C — Test split length == 45,000",                         test_loader_C_test_length),
    ("Test D — Real Parquet sample: shape [3,256,256] float32 [0,1]",test_loader_D_real_parquet_sample),
    ("Test E — Both classes (Label_A=0 and =1) loadable",            test_loader_E_both_classes),
    ("Test F — Detector mode returns exactly (image, label_a)",      test_loader_F_detector_mode),
    ("Test G — Audit mode returns (image, label_a, label_b, caption)",test_loader_G_audit_mode),
]

SUITE_3 = [
    ("Test C — Materialization: sample count matches SMOKE_N",       test_materializer_C_sample_count),
    ("Test D — Materialization: shape [3,256,256] float32 [0,1]",    test_materializer_D_image_shape_dtype),
    ("Test E — Materialization: labels match raw Parquet",           test_materializer_E_labels_correct),
    ("Test F — Materialization: no forbidden metadata columns",       test_materializer_F_no_forbidden_metadata),
    ("Test G — Materialization: deterministic (byte-identical runs)", test_materializer_G_deterministic),
    ("Test H — Both paths: same canonical image for same source",    test_materializer_H_same_preprocessing_both_paths),
    ("Test I — Corrupt image bytes: raised exception (caught)",      test_materializer_I_corrupt_image_handling),
    ("Test J — Overwrite idempotent (same output on repeat)",        test_materializer_J_overwrite_idempotent),
    ("Test K — Manifest counts match actual Parquet rows",           test_materializer_K_manifest_counts),
]


def run_suite(title, tests):
    print(f"\n{'=' * 70}")
    print(title)
    print('=' * 70)
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        {type(e).__name__}: {e}")
            failed += 1
    return passed, failed


p1, f1 = run_suite("Suite 1 — Preprocessing Tests (1–7)", SUITE_1)
p2, f2 = run_suite("Suite 2 — Loader Tests (A–G)",        SUITE_2)
p3, f3 = run_suite("Suite 3 — Materializer Tests (C–K)",  SUITE_3)

total_passed = p1 + p2 + p3
total_failed = f1 + f2 + f3
total        = total_passed + total_failed

print(f"\n{'=' * 70}")
print(f"TOTAL: {total_passed} passed, {total_failed} failed out of {total} tests.")
print('=' * 70)

sys.exit(0 if total_failed == 0 else 1)
