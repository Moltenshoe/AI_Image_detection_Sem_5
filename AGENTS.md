# AGENTS.md

# AI Agent Operating Rules

This file defines how an AI coding agent must operate inside this repository.

---

## 1. Source of Truth

Before making project-level changes, read:

1. `PROJECT.md`
2. `DECISIONS.md`
3. The specific source files, reports, and configuration files relevant to the requested task.

Use explicit file references such as `@PROJECT.md`, `@DECISIONS.md`, and `@path/to/file` when the agent supports file references.

Do not reconstruct project context from assumptions when the relevant information already exists in the repository.

---

## 2. Work Only on the Requested Scope

For each task:

1. Understand the requested objective.
2. Inspect the relevant existing implementation.
3. Make the smallest necessary changes.
4. Run the relevant checks/tests.
5. Verify the result.
6. Report what changed.

Do not perform unrelated refactoring, cleanup, renaming, dependency changes, architecture changes, or optimization.

---

## 3. Never Silently Change Research Methodology

The agent must not silently change:

- the research question
- dataset selection
- data splits
- preprocessing methodology
- feature definitions
- feature selection
- model selection
- evaluation methodology
- leakage controls
- robustness tests
- experimental variables
- finalized project decisions

If a requested implementation conflicts with a finalized decision, stop and report the conflict.

Do not resolve methodological conflicts by guessing.

---

## 4. Raw Data Is Immutable

Raw source data must never be modified in place.

Do not:

- overwrite raw files
- rename raw files
- delete raw files
- alter raw files
- preprocess raw files in place

All derived data must be written to a separate location.

---

## 5. Prevent Data Leakage

The agent must actively consider:

- train/validation/test contamination
- duplicate images
- near-duplicate or content leakage where relevant
- metadata leakage
- filename/path leakage
- generator-label leakage
- resolution leakage
- format leakage
- compression leakage
- caption/text leakage
- preprocessing leakage
- feature leakage
- target leakage

If a potential leakage source is discovered, stop the affected experiment and report it before continuing.

Do not hide, ignore, or silently work around leakage.

---

## 6. Image-Only Detector Principle

Unless a future finalized decision explicitly changes this:

The detector itself must use image information as its input.

Dataset metadata may be retained for:

- auditing
- split construction
- analysis
- generator-specific evaluation
- reporting

Metadata must not accidentally enter the model through filenames, paths, labels, captions, directory names, or preprocessing logic.

---

## 7. Distinguish Evidence Types

When documenting or reporting work, distinguish:

- **FACT** — directly verified from data, source code, documentation, or an executed experiment.
- **DECISION** — explicitly finalized project choice.
- **HYPOTHESIS** — a claim being tested.
- **RESULT** — measured experimental outcome.
- **ASSUMPTION** — not yet verified.
- **PROBLEM** — a discovered issue.
- **FIX** — a verified correction to a problem.

Never present a hypothesis or assumption as an experimental result.

---

## 8. Verification Is Mandatory

After implementing a meaningful change:

1. Execute the relevant code.
2. Check exit status.
3. Inspect important output.
4. Check for errors and warnings.
5. Verify that generated files contain the expected data.
6. Run relevant tests or validation checks.
7. Report the actual result.

Do not say that something "works" merely because code was written.

---

## 9. Reproducibility

When an experiment is performed, preserve enough information to reproduce it.

Record relevant:

- random seed
- dataset version/state
- split definition
- preprocessing settings
- feature settings
- model settings
- hyperparameters
- transformations
- compression settings
- evaluation settings
- software/dependency versions when relevant

Do not overwrite previous experiment results.

---

## 10. Dependencies

Do not install or add a dependency unless it is necessary.

Before adding a dependency:

1. Check the existing environment.
2. Check whether the task can be completed with existing dependencies.
3. If a new dependency is required, document why.
4. Avoid replacing existing libraries without a concrete reason.

---

## 11. Documentation

After verified work, update the appropriate documentation.

Documentation must describe what actually happened.

Do not document:

- untested claims as results
- planned work as completed work
- speculative explanations as facts

Important failures and fixes should be preserved because they are part of the project's history.

---

## 12. Stop Conditions

Stop and report instead of continuing when:

- a data leak is discovered
- a split is invalid
- raw data was unexpectedly modified
- a test fails for an unexplained reason
- an implementation contradicts a finalized decision
- a required input is missing
- an important assumption has not been verified
- the requested task would invalidate an existing experiment

Do not silently improvise a scientific workaround.

---

## 13. Agent Workflow

The expected workflow is:

```text
READ
  ↓
UNDERSTAND
  ↓
INSPECT
  ↓
PLAN
  ↓
IMPLEMENT
  ↓
RUN
  ↓
VERIFY
  ↓
DOCUMENT
  ↓
REPORT
```

Do not skip verification.

Do not proceed to unrelated work after completing the requested task.
