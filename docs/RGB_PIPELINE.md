# RGB Pipeline

## Status

**Not yet implemented as the final Block 2 pipeline.**

This document defines the intended role and constraints without inventing implementation details that have not yet been finalized.

## Purpose

The RGB pipeline provides a learned representation of the canonical RGB image.

It exists alongside the handcrafted forensic pipeline so the project can compare:

```text
conventional learned RGB evidence
             vs
handcrafted low-level forensic evidence
```

and later test whether their information is complementary.

## Input

The RGB pipeline receives the canonical Block 1 image:

```text
256×256 RGB uint8
```

No generator metadata, labels, filenames, paths, EXIF, JPEG metadata, or other side information should be supplied.

## Candidate models

The architecture identifies lightweight candidates:

- MobileNetV3-Small
- ShuffleNetV2

The final selection is an experimental decision.

## Required Block 2 stages

The RGB path must eventually include:

1. representation extraction;
2. representation/data validation;
3. analysis of useful information;
4. controlled reduction/selection if applicable;
5. persistent RGB dataset/representation.

## Important design constraint

The RGB pipeline should not be artificially forced into the exact handcrafted-feature workflow used by the forensic pipeline.

For example, a CNN representation may be a learned embedding rather than a list of handcrafted scalar features.

The final implementation must define:

- exact representation;
- dimensionality;
- extraction location;
- storage format;
- reduction method if used;
- training-only fitting rules;
- reproducibility controls.

## Final output

Block 2 must produce an RGB dataset that Block 3 can consume independently.

## Fusion

Fusion is intentionally deferred.

The project should first understand:

- RGB alone;
- forensic alone;

before claiming that fusion adds value.
