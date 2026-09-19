# Image Forensic Analysis --- Concepts and Results

## 1. Overview

This study compares **8 real photographs** with their corresponding
**AI-generated img2img versions**.

The purpose was not to build a final AI detector. Instead, the study
investigates whether AI processing creates measurable changes in
different parts of an image, such as:

-   frequency patterns
-   fine details
-   textures
-   edges
-   brightness and colour
-   compression behaviour
-   image residuals

Because the experiment contains only 8 image pairs, the results should
be treated as **observations from a controlled experiment**, not as
universal rules for all AI-generated images.

------------------------------------------------------------------------

## 2. FFT / Fourier Analysis

### Concept

FFT looks at an image in terms of its **frequencies** instead of its
normal pixels.

In simple terms, it helps us see how much of the image is made up of:

-   broad, slowly changing structures
-   medium-scale details
-   very fine details and rapid changes

This can reveal differences that are difficult to notice by simply
looking at the image.

### Result

The frequency analysis showed measurable differences between the real
photographs and their AI counterparts.

The changes were not identical for every image, meaning that the
frequency characteristics depend on the particular image and the img2img
transformation.

**Conclusion:** FFT is useful as a supporting forensic analysis because
it can reveal changes in the frequency structure of an image, but it is
not by itself proof that an image is AI-generated.

------------------------------------------------------------------------

## 3. DWT --- Wavelet Analysis

### Concept

DWT separates an image into different levels of detail.

It allows us to look at both:

-   **where** details occur in the image
-   **how much** detail exists at different scales

This makes it useful for studying fine textures and small image
structures.

### Result

Several wavelet-detail measurements showed a strong and consistent
decrease in the AI images.

For example, the entropy of several detail components decreased for all
8 pairs. The strongest examples included:

-   second-level LH detail entropy
-   second-level HL detail entropy
-   second-level HH detail entropy
-   first-level HL detail entropy

### Conclusion

The AI-generated images showed changes in the distribution and
complexity of fine-scale image details.

This suggests that wavelet analysis can capture changes introduced by
the img2img process, particularly in fine image structure.

------------------------------------------------------------------------

## 4. DCT --- Discrete Cosine Analysis

### Concept

DCT describes an image using different levels of visual detail, from
broad smooth regions to fine changes.

It is closely related to how image compression systems represent images.

### Result

The AI images generally showed a reduction in the proportion of
high-frequency DCT information.

The average change in the high-frequency ratio was approximately
**-0.08**, meaning that the AI counterparts tended to contain less
relative high-frequency DCT energy than their real counterparts in this
experiment.

The change was consistent in direction for most pairs.

### Conclusion

The img2img process appears to alter the distribution of fine image
information.

DCT therefore provides another useful way of observing changes in image
structure, although it should not be treated as a standalone AI
detector.

------------------------------------------------------------------------

## 5. Residual / Fingerprint Analysis

### Concept

A residual is what remains after removing part of the normal image
information.

The study used residuals to emphasize subtle details that are normally
hidden by the main visual content.

One of the residual approaches is particularly useful for detecting
repeated or structured patterns that may be introduced during image
generation.

### Result

The residual-based analysis produced noticeable differences between the
real and AI images.

The differences became particularly useful when the residual information
was examined in the frequency domain.

### Conclusion

Residual analysis is one of the more promising approaches in this
experiment because it attempts to remove normal image content and expose
subtle processing patterns.

However, the observed patterns should still be treated as
characteristics of this particular img2img process rather than universal
AI fingerprints.

------------------------------------------------------------------------

## 6. Cross-Difference and Periodic Frequency Analysis

### Concept

Cross-difference analysis removes much of the slowly changing image
content and emphasizes very local changes.

The resulting information can then be examined for repeated frequency
patterns.

This type of analysis is related to forensic research on fingerprints
left by diffusion-based image generation.

### Result

The cross-difference approach revealed structured frequency information
that was useful for comparing the two groups.

It provides a different view from a normal FFT because much of the
ordinary image content has already been reduced.

### Conclusion

This is an important supporting analysis in the project because it
specifically focuses on subtle processing patterns rather than obvious
visual differences.

It is still an **exploratory implementation** and should not be
described as a complete reproduction of a published AI detector.

------------------------------------------------------------------------

## 7. Edge, Gradient and Contour Analysis

### Concept

Edges are locations where image brightness or colour changes quickly.

For example:

-   object boundaries
-   text
-   fabric patterns
-   branches
-   fine structures

The analysis measures how strong and how common these changes are.

### Result

The average gradient magnitude decreased in the AI images by
approximately **0.057** on average.

The direction was consistent across all 8 pairs: the AI version had a
lower average gradient value than the corresponding real image.

The high-percentile gradient measurement also decreased.

### Conclusion

The AI transformations tended to produce smoother or less locally varied
image structures in this dataset.

This does not mean that all AI images are smoother than real
photographs, but it is a measurable characteristic of the images used in
this experiment.

------------------------------------------------------------------------

## 8. LBP --- Local Texture Analysis

### Concept

LBP studies very small neighbourhoods of pixels to describe
**micro-texture**.

In simple terms, it asks how the pixels around a point are arranged
compared with that point.

This is useful for surfaces such as:

-   fabric
-   fur
-   wood
-   repeated patterns
-   small textures

### Result

LBP produced one of the strongest differences in the experiment.

The average LBP code increased by approximately **0.71** in the AI
images, and this increase occurred in **all 8 pairs**.

LBP histogram entropy decreased by approximately **0.61**, with the
decrease occurring in 7 of the 8 pairs.

### Conclusion

The AI versions showed a noticeable change in their local texture
patterns.

Among the analyses performed, LBP provided a particularly clear and
consistent difference in this controlled dataset.

------------------------------------------------------------------------

## 9. GLCM --- Texture Statistics

### Concept

GLCM studies how often different brightness levels occur next to each
other.

It provides information about whether a texture is:

-   smooth
-   varied
-   repetitive
-   strongly structured

### Result

GLCM homogeneity increased by approximately **0.088**, and this increase
occurred in all 8 pairs.

GLCM dissimilarity decreased by approximately **0.356**, also
consistently across the 8 pairs.

### Conclusion

The AI images tended to show more homogeneous local texture and less
local variation according to these measurements.

This agrees with the changes observed using LBP and supports the idea
that the img2img process altered the fine texture structure of the
photographs.

------------------------------------------------------------------------

## 10. Intensity, Colour and Entropy Analysis

### Concept

These are basic image statistics used as a baseline.

They examine changes in:

-   brightness
-   contrast
-   colour channels
-   saturation
-   overall information/complexity

### Result

Differences were observed between the real and AI images, but these
measurements were less useful as direct forensic evidence.

Changes in brightness or colour can happen simply because the img2img
model changes the appearance of the photograph.

### Conclusion

These features are useful as supporting information and controls, but
they should not be considered strong evidence of AI generation by
themselves.

------------------------------------------------------------------------

## 11. Error Level Analysis (ELA)

### Concept

ELA compares an image with a newly compressed version of itself.

It is commonly used as a **compression and editing diagnostic**.

It can highlight regions that respond differently to JPEG compression.

### Result

ELA measurements showed consistent differences between the two groups in
this experiment.

However, the differences were small and ELA is strongly affected by the
compression history of an image.

### Conclusion

ELA should be treated as a supporting forensic technique rather than an
AI detector.

In particular, an ELA difference does not automatically mean that an
image was AI-generated.

------------------------------------------------------------------------

## 12. Overall Statistical Analysis

### Concept

Instead of simply comparing the average real image with the average AI
image, each real photograph was compared directly with its own AI
counterpart.

This is important because the photographs contain different subjects and
textures.

The analysis considered:

-   average change
-   median change
-   consistency across the 8 pairs
-   effect size
-   an exact paired sign-flip test
-   correction for multiple comparisons

### Result

Several features showed large and consistent changes.

Some of the strongest observations were:

  Feature                          Average AI − Real change   Consistency
  ------------------------------ -------------------------- -------------
  LBP mean code                                      +0.709           8/8
  GLCM homogeneity                                   +0.088           8/8
  GLCM dissimilarity                                 -0.356           8/8
  Gradient mean                                      -0.057           8/8
  DCT high-frequency ratio                           -0.080           7/8
  LBP histogram entropy                              -0.610           7/8
  Several DWT detail entropies                    Decreased           8/8

The largest standardized paired effect in the reported feature table was
for **LBP mean code**, with an effect size of approximately **2.45**.

Several other wavelet, gradient, texture and ELA features also showed
large paired effects.

### Important interpretation

A small p-value does not automatically make a feature a good AI
detector.

The most interesting features are those that show:

1.  a meaningful difference,
2.  a reasonably consistent direction across image pairs, and
3.  a plausible relationship to image processing.

------------------------------------------------------------------------

## 13. Overall Findings

The different analyses point toward several broad observations.

### 1. Fine texture changed

LBP and GLCM showed particularly clear changes in local texture.

The AI images generally became more homogeneous and showed different
micro-texture patterns.

### 2. Fine-scale image information changed

DWT and DCT both showed changes in the representation of fine details.

Several DWT detail-entropy measurements decreased consistently.

### 3. Edge behaviour changed

Gradient measurements decreased consistently across the 8 pairs,
suggesting that the AI transformations altered local image variations.

### 4. Frequency structure changed

FFT-based analysis showed measurable changes in the frequency
distribution of the images.

The residual-based frequency analysis provided an additional way of
examining these differences.

### 5. Basic image statistics were less conclusive

Brightness, colour and other basic statistics can change during img2img
generation, but they are not reliable indicators of AI generation by
themselves.

### 6. ELA has limited forensic value here

ELA can reveal compression differences, but it cannot distinguish AI
generation from other causes of different compression histories.

------------------------------------------------------------------------

## 14. Final Conclusion

The experiment demonstrates that an img2img transformation can produce
**measurable changes in the visual, texture, spatial and frequency
characteristics of images**.

Among the techniques tested, **LBP and GLCM provided particularly
consistent texture differences**, while **DWT, DCT, FFT and residual
analysis revealed changes in fine-scale and frequency information**.
Gradient analysis also showed a consistent reduction across the eight
pairs.

The important conclusion is not that any single transformation can prove
that an image is AI-generated. Instead, the results suggest that
**multiple complementary image characteristics can be combined to
investigate whether an image has undergone AI-based generation or
transformation**.

Because this experiment contains only 8 controlled real/img2img pairs,
the findings are preliminary. A larger dataset containing images from
multiple AI generators, different prompts, different image types,
different cameras and different compression conditions would be required
to determine whether these observations generalize.

------------------------------------------------------------------------

## 15. Limitations

The main limitations of this study are:

-   Only 8 real/AI pairs were analyzed.
-   The AI images came from a controlled img2img setup rather than many
    independent generators.
-   Different image content can naturally produce different texture and
    frequency characteristics.
-   Compression and resizing can affect several measurements.
-   The experiment does not train or validate a general-purpose AI-image
    classifier.
-   The results therefore should not be interpreted as proof that the
    measured features will work equally well on arbitrary AI-generated
    images.

------------------------------------------------------------------------

## 16. Future Work

The next stage would be to expand the dataset and test whether the
observed differences remain consistent.

Possible future directions include:

-   increasing the number of real/AI pairs;
-   using multiple AI image generators;
-   testing different image resolutions and compression levels;
-   adding images generated directly from text as well as img2img
    images;
-   selecting the most useful features;
-   training a machine-learning classifier using the extracted features;
-   evaluating the classifier on images from generators that were not
    used during training.

The ultimate goal would be to determine whether the forensic patterns
observed in this controlled experiment can be turned into a reliable and
generalizable AI-image detection system.
