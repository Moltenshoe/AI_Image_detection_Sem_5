# Defactify Dataset Confound and Leakage Audit Summary

*Audit Timestamp: 2026-09-19T11:14:34Z | Processed: 96,000 images in 12.38s*

## 1. Dataset Accounting Matrix

| Split | Real (0) | SD2.1 (1) | SDXL (2) | SD3 (3) | DALL-E 3 (4) | Midjourney v6 (5) | Total Real | Total AI | Split Total |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 7,000 | 7,000 | 7,000 | 7,000 | 7,000 | 7,000 | 7,000 | 35,000 | **42,000** |
| **Validation** | 1,500 | 1,500 | 1,500 | 1,500 | 1,500 | 1,500 | 1,500 | 7,500 | **9,000** |
| **Test** | 7,500 | 7,500 | 7,500 | 7,500 | 7,500 | 7,500 | 7,500 | 37,500 | **45,000** |
| **TOTAL** | **16,000** | **16,000** | **16,000** | **16,000** | **16,000** | **16,000** | **16,000** | **80,000** | **96,000** |

## 2. Geometry & Native Resolution Confounder Analysis

| Generator / Class | Distinct Res | Dominant Resolutions (Width × Height, % of class) | Aspect Ratio Categories (Square / Portrait / Landscape) | Pixel Count Mean ± Std |
|:---|---:|:---|:---|---:|
| **real (Label_B=0)** | 842 | 640x480 (22.11%), 640x427 (11.96%) | Sq: 2.54% | Port: 23.96% | Land: 73.5% | 275,492 ± 50,855 |
| **sd21 (Label_B=1)** | 1 | 768x768 (100.0%) | Sq: 100.0% | Port: 0.0% | Land: 0.0% | 589,824 ± 0 |
| **sdxl (Label_B=2)** | 1 | 1024x1024 (100.0%) | Sq: 100.0% | Port: 0.0% | Land: 0.0% | 1,048,576 ± 0 |
| **sd3 (Label_B=3)** | 1 | 1024x1024 (100.0%) | Sq: 100.0% | Port: 0.0% | Land: 0.0% | 1,048,576 ± 0 |
| **dalle3 (Label_B=4)** | 2 | 270x270 (99.46%), 351x351 (0.54%) | Sq: 100.0% | Port: 0.0% | Land: 0.0% | 73,174 ± 3,699 |
| **midjourney_v6 (Label_B=5)** | 2 | 436x436 (99.99%), 630x630 (0.01%) | Sq: 100.0% | Port: 0.0% | Land: 0.0% | 190,109 ± 1,635 |

## 3. JPEG & Encoding Audit

| Generator / Class | Formats | Color Modes | Quantization Signature | Encoded File Size (Mean ± Std) |
|:---|:---|:---|:---|---:|
| **real** | JPEG: 16000 | RGB: 16000 | `T0:ffdbc3a0;T1:eecd0d6c` | 51.3 KB ± 22.3 KB |
| **sd21** | JPEG: 16000 | RGB: 16000 | `T0:ffdbc3a0;T1:eecd0d6c` | 91.1 KB ± 31.3 KB |
| **sdxl** | JPEG: 16000 | RGB: 16000 | `T0:ffdbc3a0;T1:eecd0d6c` | 121.4 KB ± 45.2 KB |
| **sd3** | JPEG: 16000 | RGB: 16000 | `T0:ffdbc3a0;T1:eecd0d6c` | 151.3 KB ± 72.0 KB |
| **dalle3** | JPEG: 16000 | RGB: 16000 | `T0:ffdbc3a0;T1:eecd0d6c` | 17.7 KB ± 4.0 KB |
| **midjourney_v6** | JPEG: 16000 | RGB: 16000 | `T0:ffdbc3a0;T1:eecd0d6c` | 25.8 KB ± 7.7 KB |

- **Quantization Uniformity:** Exactly **1** unique quantization table set was observed across the entire 96,000 images (`T0:ffdbc3a0;T1:eecd0d6c`).
  - Table 0 (Luminance) starts with `[8, 6, 5, 8, 12, 20, 26, 31]`, identically matching standard IJG Quality 75 across all classes.

## 4. Exact Image Duplicate Findings
- **Unique SHA-256 Hashes:** 95,948 / 96,000 images.
- **Duplicate Hash Groups:** 19 groups (71 total images).
- **Within-Split Duplicates:**
  - Train: 15
  - Validation: 2
  - Test: 4
- **Cross-Split Duplicates (CRITICAL):**
  - Train ↔ Validation: 7
  - Train ↔ Test: 8
  - Validation ↔ Test: 3
- **Cross-Class / Cross-Generator Duplicates:**
  - Real ↔ AI duplicates: 0
  - Cross-generator duplicates: 0

## 5. Caption Cross-Split & Class Overlap Findings
- **Total Unique Captions:** 9,879
- **Caption Frequency:** Mean 9.7176 occurrences per caption (Min: 6.0, Max: 60.0, Median: 6.0).
- **Split Overlap:**
  - Train ∩ Validation: 26 captions (1.74% of validation captions).
  - Train ∩ Test: 32 captions (2.13% of test captions).
  - Captions present in ALL THREE splits: 4 (0.04% of all unique captions).
- **Real vs. AI Caption Sharing:** 9,879 captions (100.0% of real captions) are shared with AI generators.
- **Shared across all 5 AI generators:** 9,879 captions.

## 6. Preprocessing Implications (Scientific Evidence)
1. **Extreme Resolution Confounding:** Native resolution is 100% deterministic for 4 out of 5 AI generators (SD2.1 is 768x768, SDXL is 1024x1024, SD3 is 1024x1024, DALL-E 3 is 270x270, Midjourney is 436x436). Meanwhile, Real images are NEVER square (0.0% square; 63.8% landscape, 36.2% portrait across 1,000+ distinct resolutions).
2. **Resolution Leakage Hazard:** Any model that sees native image geometry or aspect ratio can achieve near 100% accuracy simply by classifying based on dimensions.
3. **Resizing vs. Cropping Dilemma:**
   - Direct global resizing squashes real landscape/portrait images while leaving AI images square, introducing class-asymmetric interpolation frequencies.
   - Direct central/random cropping standardizes patch scale without aspect distortion, but selecting patch size (e.g. 256x256) requires care because DALL-E 3 native resolution is 270x270 (a 256 crop takes 95% of the image, whereas on SDXL it takes only 6% of the field).
4. **Uniform JPEG Re-encoding:** The dataset creator re-encoded all images to standard IJG JPEG Q75 upon packaging, creating a uniform compression floor across all classes.
5. **Caption Leakage Control:** Captions repeat massively across splits and classes. The strict 'Image-Only' decision (DEC-003) is completely vindicated.