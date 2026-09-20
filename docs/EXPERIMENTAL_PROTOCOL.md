# Experimental Protocol

## Research factors

The main experimental design has three independent factors:

### Factor 1 — Data budget

How many training images are supplied?

Example:

```text
1k / 5k / 10k / 20k
```

### Factor 2 — Feature budget

How much information is retained?

Example:

```text
111 / 64 / 32 / 16 / 8
```

### Factor 3 — Model type

Which model consumes the representation?

Examples:

- LightGBM
- Tiny MLP
- MobileNetV3-Small
- ShuffleNetV2

These are factors, not metrics.

---

# Evaluation dimensions

The factor grid is evaluated using:

- ROC-AUC
- PR-AUC
- F1
- TPR at fixed FPR
- per-generator AUC
- mean generator AUC
- worst-generator AUC
- runtime
- RAM
- parameter count
- model size
- FLOPs where applicable

---

# Recommended experiment organization

A run should be uniquely described by:

```text
pipeline
data_budget
feature_budget
model
generator_split
compression_condition
seed
```

For example:

```text
forensic/
data=5000/
features=32/
model=lightgbm/
holdout=SDXL/
compression=Q60/
seed=42
```

---

# Leakage controls

The following must never be used to fit a detector:

- test labels;
- test feature distributions for selection;
- generator identity as an input;
- filename;
- path;
- source ID;
- metadata.

Feature selection is fitted on training data only.

---

# Reproducibility

Store:

- configuration;
- feature list;
- selected feature list;
- model hyperparameters;
- seed;
- split definition;
- metrics;
- runtime;
- resource measurements.

---

# Interpretation

The central output should not be a single accuracy number.

The project is trying to characterize a trade-off:

```text
less data
less feature information
less model complexity
        ↓
how much useful detection performance remains?
```

A useful result may be a compact configuration that retains most of the performance of a larger baseline.

That is a stronger efficiency statement than simply reporting the highest score.
