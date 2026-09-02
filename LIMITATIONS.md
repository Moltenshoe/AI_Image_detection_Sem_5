# Limitations — v3 AI Image Forensic Analysis

## 1. Sample size (biggest limitation)
Only **8 matched pairs**. With n=8, the best possible exact sign-flip p-value is 0.0078 — every "significant" result in this notebook sits at that ceiling. This means "100% consistent across 8 examples," not statistical proof. No claim here generalizes without a much larger dataset.

## 2. Single, controlled generation source
All AI images come from one img2img pipeline (one model, one sampler, one setting). Findings may be specific to that generator/config, not AI images in general. A different model, prompt, or denoising strength could shift or erase the same signals.

## 3. No text-to-image, only img2img
Img2img starts from a real photo, so some real-image statistics may partially survive the transformation. A pure text-to-image generator (no real starting image) could behave very differently. Conclusions shouldn't be extended to that case.

## 4. Image content differs per pair
The 8 scenes are different subjects (forest, fabric, etc.), each with naturally different texture/frequency baselines. Pairing reduces this problem but doesn't remove it — some features may reflect scene type as much as AI-vs-real.

## 5. Resizing / center-crop
All images forced to 512×512 center crop. This discards edges/content and can change frequency and texture statistics independent of AI-vs-real, especially for images with important detail outside the crop.

## 6. Compression not controlled or varied
JPEG compression affects FFT, DCT, residuals, edges, ELA, and wavelet coefficients directly. This notebook doesn't test whether results hold across different compression levels or hold at all if real/AI images had different original formats/quality going in.

## 7. Multiple-comparisons problem
~539 features tested per image. BH-FDR correction controls the *false discovery rate*, but it cannot manufacture more independent data — it only adjusts confidence given the same 8 pairs. Many "surviving" features may still be sample-size artifacts.

## 8. Quaternion wavelet (QWT) is not a validated method
The `qwt_*` module is a custom, notebook-authored quaternion-Haar implementation "inspired by" published QWT/CQWT forensic papers — explicitly **not** a reproduction of those algorithms. The strongest result in the whole notebook (`qwt_HL2_magnitude_entropy`, dz = -3.06) comes from this unvalidated method. Should be treated as a hypothesis, not a citation-backed finding.

## 9. No held-out / unseen data test
All 539 features were computed and ranked on the same 8 pairs used to "discover" them. There's no separate validation set — classic overfitting risk if this were to become a detector. Nothing here has been checked against images the pipeline hasn't already seen.

## 10. Whole-image, global statistics only
Every feature is computed over the full 512×512 image. Modern forensic research increasingly finds signal in **local/patch-level** anomalies (small regions that look wrong) rather than global averages — this notebook can miss localized artifacts that get diluted in a whole-image mean.

## 11. No learned/deep-feature detector included
This is a pure hand-crafted feature study (frequency, texture, residual, wavelet). It does not include CNN classifiers, ViT/CLIP embedding checks, or any trained model — which make up a large and increasingly dominant share of real-world AI-detection methods. Results here can't be compared directly to that class of detector's accuracy.

## 12. No semantic / physical-plausibility checks
Nothing here looks at content-level tells (anatomical errors, impossible shadows/reflections, inconsistent lighting) — a category of detection that's often more visible to humans than any of the statistical features used here.

## 13. No sensor-fingerprint (PRNU) analysis
Real camera sensors leave a unique, camera-specific noise fingerprint. This notebook never tests for the presence/absence of that fingerprint, which is a standard, strong signal in classical photo forensics.

## 14. Metadata stripped before analysis
Images are decoded straight to RGB tensors early in the pipeline — EXIF/metadata (which can be one of the simplest, strongest real-world AI-image signals) is not used at all.

## 15. GAN vs diffusion mismatch risk
Some of the periodic/grid-artifact features (cross-difference, Synthbuster) are historically strongest on **GAN**-generated images; diffusion models are known to produce fewer of these grid artifacts. If the AI images here are diffusion-based, this whole feature family may be structurally low-yield regardless of dataset size — worth checking generator type before trusting/dismissing these results.

## 16. Effect size ≠ practical detector performance
Cohen's dz tells you how consistent a difference is across 8 pairs — it says nothing about false-positive/false-negative rates, accuracy, or how the feature would behave on a real, large, mixed dataset. None of these numbers should be read as "% accuracy."

---

**One-line summary:** this is a well-built, internally consistent exploratory feature study on a very small, single-source dataset. Every result is a lead worth testing further, not a finding that's been proven.