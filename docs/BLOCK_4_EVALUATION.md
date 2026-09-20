# Block 4 — Evaluation and Experiments

## Purpose

Block 4 determines whether the representations and models built in Blocks 1–3 actually work and under what conditions.

No final performance claim should be made before Block 4.

---

# 1. Core classification metrics

Recommended metrics:

- ROC-AUC
- PR-AUC
- F1
- TPR at a fixed FPR

Accuracy may be reported when useful, but should not be the sole metric.

---

# 2. Generator-disjoint evaluation

Defactify contains five AI generators.

A generator-disjoint protocol rotates the held-out generator.

Example:

```text
Train: generators A+B+C+D
Test:  generator E
```

Repeat so each generator is held out.

Report:

- per-generator AUC;
- mean generator AUC;
- worst-generator AUC.

Generator identity is not detector input.

---

# 3. Data-efficiency experiments

Example training budgets:

```text
1k
5k
10k
20k
```

Keep evaluation data fixed.

The question is:

> How much training data is necessary to retain useful performance?

Sampling methodology must be documented so comparisons are reproducible.

---

# 4. Feature-budget experiments

Example:

```text
111
64
32
16
8
```

For each budget:

- use the same underlying candidate pool;
- fit selection on training data only;
- evaluate on held-out data;
- record performance and computational cost.

---

# 5. Model experiments

Compare model families while controlling the information available to them.

For example:

```text
same data budget
same feature budget
different model
```

This isolates model complexity more effectively.

---

# 6. Compression robustness

Separate feature extraction from robustness testing.

Suggested test qualities:

```text
Clean
Q95
Q80
Q60
Q40
Q20
```

Possible protocols:

### Clean → compressed

Train on clean; test on compressed.

### Compression augmentation

Train on clean + compressed; test on compressed.

### Cross-quality

Train on higher-quality compression; test on lower-quality compression.

---

# 7. Efficiency metrics

Measure:

- selected feature count;
- model parameter count;
- model file size;
- FLOPs where applicable;
- inference runtime;
- RAM usage.

The target is a useful performance/efficiency trade-off.

---

# 8. Branch ablation

Forensic branches should be evaluated individually and in controlled combinations.

The purpose is to determine:

- which branches carry useful information;
- which branches are redundant;
- which branches improve generalization;
- which branches improve compression robustness;
- which branches add cost without useful benefit.

---

# 9. Reporting

Every experiment should record:

- dataset version;
- data budget;
- train/validation/test split;
- held-out generator;
- compression condition;
- feature budget;
- feature-selection procedure;
- model;
- random seed(s);
- metric values;
- runtime;
- resource usage.

This prevents results from becoming irreproducible configuration snapshots.
