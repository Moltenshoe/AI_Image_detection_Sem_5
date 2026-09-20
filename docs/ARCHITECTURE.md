# System Architecture

## 1. Purpose

This document defines the authoritative high-level architecture of the AI-generated image detection project.

The system is divided into four sequential blocks:

```text
Block 1
Data Loading + Preprocessing
        ↓
Stored Processed Data
        ↓
Block 2
Image Analysis
   ├── RGB
   └── Forensic
        ↓
Two Stored Block-2 Datasets
        ↓
Block 3
Models
        ↓
Predictions
        ↓
Block 4
Evaluation + Experiments
```

The architecture is intentionally modular so that later experiments can operate on persistent Block 1 and Block 2 artifacts rather than repeatedly recomputing earlier stages.

---

## 2. Block boundaries

### Block 1

Input:

- raw Defactify dataset

Output:

- canonical processed image representation

### Block 2

Input:

- Block 1 processed images

Outputs:

- forensic selected-feature dataset
- RGB selected representation/dataset

### Block 3

Input:

- Block 2 datasets

Output:

- trained models
- predictions

### Block 4

Input:

- Block 3 predictions and experiment logs

Output:

- performance analysis
- robustness analysis
- efficiency analysis
- final comparisons

---

## 3. Block 2 completion definition

Block 2 is not complete when individual feature functions exist.

It is complete when:

```text
Block 1 processed data
        ↓
 ┌──────┴──────┐
 ↓             ↓
Forensic      RGB
analysis      analysis
 ↓             ↓
selection     selection
 ↓             ↓
FOR_DATA      RGB_DATA
```

Both datasets must be reproducible and consumable independently by Block 3.

---

## 4. Forensic branch philosophy

The forensic branches are intended to cover different low-level evidence domains:

- frequency and periodicity
- localized multi-scale frequency structure
- local texture
- residual/noise structure
- compression/DCT/phase/grid response

They are not assumed to be statistically independent.

Complementarity must be tested through:

- feature correlation
- mutual information
- feature selection
- branch ablation
- model performance under controlled feature budgets

---

## 5. RGB branch philosophy

The RGB branch represents conventional spatial/image information.

It exists as a separate path so that the project can answer:

- how much useful information is available from RGB learned representations?
- how does RGB compare with handcrafted forensic evidence?
- does combining RGB and forensic information provide measurable benefit?

Fusion is a later experiment, not a premise.

---

## 6. Persistent data principle

Each block should have a defined input and output artifact.

```text
Block N input
      ↓
processing
      ↓
Block N output
      ↓
persistent artifact
```

The goal is reproducibility, modular debugging, and controlled experiments.
