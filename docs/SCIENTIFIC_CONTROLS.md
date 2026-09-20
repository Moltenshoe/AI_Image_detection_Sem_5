# Scientific Controls

## 1. Image-only detector input

The detector must operate on the image representation produced by Block 1.

Forbidden shortcuts:

- filename
- path
- caption
- generator identity
- split
- EXIF
- container metadata
- original JPEG quantization tables
- file size
- source identifiers

---

## 2. Immutable raw data

Raw Defactify data must not be modified during preprocessing or experimentation.

---

## 3. Deterministic preprocessing

Common preprocessing must be deterministic:

```text
centered square crop
→ 256×256
→ RGB uint8
```

---

## 4. Symmetric processing

Real and AI images must receive the same preprocessing and controlled transformations.

Branch E recompression must use the same quality levels and procedure for both classes.

---

## 5. No augmentation during forensic extraction

The forensic candidate extractor should operate on the canonical image without random augmentation.

---

## 6. Generator identity

Generator identity may be used to define evaluation splits and diagnostics.

It must not be supplied as a detector input.

---

## 7. Generator-disjoint evaluation

Where generalization is being measured, the held-out generator must not contribute training examples to the detector.

---

## 8. Training-only feature selection

Feature selection must not inspect validation/test outcomes.

---

## 9. Branch complementarity

Do not claim that branches are complementary merely because they use different mathematical transforms.

Complementarity requires empirical evidence from:

- redundancy analysis;
- branch ablation;
- controlled model comparisons.

---

## 10. Compression

Compression is both:

1. a source of candidate forensic evidence in Branch E;
2. a separate robustness condition in Block 4.

These must not be conflated.

---

## 11. Novelty claims

The project does not claim that it invented:

- FFT analysis;
- DWT;
- LBP;
- GLCM;
- residual analysis;
- DCT;
- ELA/JPEG response;
- phase analysis;
- LightGBM;
- lightweight CNNs.

The project should describe its contribution as the design and controlled empirical analysis of a compact, multi-domain detection pipeline and its performance/efficiency trade-offs, subject to experimental validation.
