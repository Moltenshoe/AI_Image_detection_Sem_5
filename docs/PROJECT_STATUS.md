# Project Status

Last documented status: 2026-09-20

## Completed

### Block 1

- raw-data loading
- canonical preprocessing
- persistent processed-data concept

### Forensic Block 2

- Branch A implemented
- Branch B implemented
- Branch C implemented
- Branch D implemented

Canonical A–D candidate pool:

```text
34 + 30 + 16 + 5 = 85
```

## Pending

### Forensic

- Branch E implementation
- Branch E verification
- complete candidate pool = 111
- feature analysis
- redundancy analysis
- feature selection
- final forensic dataset

### RGB

- representation pipeline
- representation analysis
- reduction/selection where applicable
- final RGB dataset

### Block 3

- model training
- model comparison
- optional fusion

### Block 4

- data-budget experiments
- feature-budget experiments
- model experiments
- generator-disjoint evaluation
- compression robustness
- efficiency measurements
- final analysis

---

# Block 2 definition of done

```text
A–E candidate extraction
        +
forensic analysis
        +
forensic selection
        +
RGB pipeline
        +
RGB analysis
        +
RGB selection
        ↓
two reproducible Block-2 datasets
```

Only then is Block 2 considered complete.

---

# Branch E audit

The historical audit state is maintained separately in:

`docs/BRANCH_E_AUDIT_STATE.md`

That file may contain implementation-audit checkpoints and must be reconciled with the actual current source before claiming Branch E completion.
