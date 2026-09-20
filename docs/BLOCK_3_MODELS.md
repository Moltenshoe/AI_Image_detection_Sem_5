# Block 3 — Models

## Purpose

Block 3 is deliberately separated from image analysis.

Its input is the persistent output of Block 2.

```text
Forensic Dataset ──→ Forensic Models
RGB Dataset ────────→ RGB Models
```

## Forensic models

### Primary

LightGBM.

Input:

```text
selected forensic feature vector
```

### Secondary

Tiny MLP.

Purpose:

- determine whether nonlinear feature interactions provide additional benefit;
- compare a tree-based model with a compact neural model.

## RGB models

Candidate lightweight CNNs:

- MobileNetV3-Small
- ShuffleNetV2

## Experimental control

Model comparisons should be made under controlled information budgets where possible.

For example:

```text
same training images
same evaluation split
same selected feature budget
different model
```

Otherwise differences may be caused by both model and data/feature changes.

## Fusion

Possible later configurations:

- feature-level fusion;
- decision-level fusion.

Fusion should only be evaluated after the independent pipelines are understood.

Fusion is not assumed to improve performance.

## What Block 3 does not decide

Block 3 should not silently modify:

- preprocessing;
- feature definitions;
- feature selection;
- generator labels;
- evaluation splits.

Those belong to earlier blocks or to explicitly documented experimental configuration.
