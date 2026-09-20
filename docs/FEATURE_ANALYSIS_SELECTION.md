# Feature Analysis and Feature Selection

## Purpose

Feature analysis and selection are central to the research objective.

The project starts from candidate representations and asks:

> Which information is useful, which information is redundant, and how much can be removed without materially degrading detection?

## Candidate pool

Planned canonical forensic pool:

```text
A = 34
B = 30
C = 16
D = 5
E = 26
------------
111
```

The final selected feature count is expected to be substantially smaller in at least some experimental configurations, but the final number must be determined empirically.

---

# 1. Validity analysis

Before statistical selection:

- verify shape;
- verify dtype;
- verify finite values;
- identify NaN/Inf;
- identify constant features;
- inspect missing values;
- verify deterministic extraction;
- verify feature ordering.

Invalid numerical behavior should be fixed at the extractor rather than hidden during model training.

---

# 2. Relevance / discrimination

Candidate features can be examined for their relationship to the real/fake target.

Possible analyses include:

- univariate discrimination;
- distribution separation;
- mutual information with the target;
- model-derived feature importance.

The exact statistical test should be selected and documented before final experimentation.

---

# 3. Redundancy

The project explicitly wants branches to be complementary rather than unnecessarily redundant.

Therefore measure relationships among candidate features.

Useful tools include:

- Pearson correlation for linear relationships;
- Spearman correlation for monotonic relationships;
- mutual information for nonlinear dependency.

A feature can be individually useful while still being redundant with another selected feature.

Feature selection should therefore consider both:

```text
high relevance to target
        +
low redundancy with selected features
```

This maximum-relevance/minimum-redundancy principle is well established in feature-selection literature.

---

# 4. Branch-level complementarity

Feature-level correlation is not sufficient to establish branch complementarity.

Use branch ablations such as:

```text
A
B
C
D
E

A+B
A+C
...
```

and controlled larger combinations where practical.

The key question is:

> Does adding a branch improve the representation after the information already present in the existing branches is considered?

A branch can be highly correlated with another branch and still provide useful nonlinear or conditional information.

Therefore ablation and model-level comparison are necessary in addition to correlation matrices.

---

# 5. Generator behavior

A feature can appear highly discriminative because it identifies a generator rather than synthetic origin.

For this reason inspect feature behavior across the five Defactify generators.

A desirable feature should not simply be a shortcut for:

```text
generator identity
```

The feature-analysis stage should therefore retain generator-aware diagnostics without ever giving generator identity to the detector.

---

# 6. Compression behavior

Analyze how candidate features change under controlled JPEG compression.

This is particularly important for Branch E, but should also be examined for other branches because compression can alter:

- high-frequency information;
- residuals;
- texture;
- spectral energy;
- phase;
- grid discontinuities.

A feature that collapses immediately under mild compression should be identified before making robustness claims.

---

# 7. Computational cost

Record feature extraction cost where practical:

- CPU time;
- memory;
- storage per image;
- implementation complexity.

A feature that contributes negligible discrimination but significant computation is a candidate for removal.

---

# 8. Training-only fitting

Any learned/statistical selection operation must be fitted using training data only.

Correct:

```text
TRAIN
 ↓
fit selector
 ↓
selected feature list
 ↓
TRAIN / VALIDATION / TEST
```

Incorrect:

```text
TRAIN + VALIDATION + TEST
 ↓
feature selection
```

The latter leaks evaluation information into representation design.

---

# 9. Feature-budget experiments

The final experiment should evaluate multiple feature budgets.

Example:

```text
111
64
32
16
8
```

The exact levels can be changed after inspecting the candidate distributions.

The objective is not simply to minimize feature count.

The objective is to measure the **performance–feature-budget trade-off**.

---

# 10. Recommended interpretation

Results should distinguish:

### Relevance

Does a feature contain information about the target?

### Redundancy

Does another feature already contain similar information?

### Complementarity

Does adding the feature/branch improve the detector beyond existing information?

### Robustness

Does the information survive generator changes and compression?

### Efficiency

How much computation and storage does the feature require?

These are separate properties.
