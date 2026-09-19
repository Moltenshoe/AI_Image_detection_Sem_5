# docs/MODULE_MAP.md

# Module Inventory & Architectural Registry

This document defines the responsibility, inputs, outputs, public interfaces, dependencies, and test coverage for every module in this codebase.

---

## Block 1: Data Ingestion & Preprocessing

### `src/data/preprocessing.py`
- **Responsibility:** Canonical deterministic image preprocessing pipeline (DEC-008).
- **Block:** Block 1 (Data Preprocessing).
- **Input:** Raw image bytes (`Image["bytes"]`).
- **Output:** Standardized RGB PIL Image of size $256 \times 256$ (uint8).
- **Public Interface:**
  - `preprocess(raw_bytes: bytes) -> Image.Image`
  - `centered_square_crop(img: Image.Image) -> Image.Image`
  - `resize_to_canonical(img: Image.Image) -> Image.Image`
  - `CANONICAL_SIZE: int = 256`
  - `INTERPOLATION: int = cv2.INTER_AREA`
- **Dependencies:** Standard library `io`, `PIL.Image`, `cv2`, `numpy`.
- **Used by:**
  - `src/data/loader.py` (on-demand evaluation path)
  - `src/data/materializer.py` (training materialization path)
- **Tests:** `src/data/tests/test_preprocessing.py` (Tests 1–7).
- **Modification Notes:** Preprocessing contract is finalized by DEC-008 (largest centered square crop followed by 256×256 area-based resizing: `cv2.INTER_AREA`).

---

### `src/data/loader.py`
- **Responsibility:** On-demand lazy dataset loader wrapping raw Defactify Parquet files for evaluation (`validation` and `test` splits).
- **Block:** Block 1 (Data Preprocessing / On-Demand Evaluation).
- **Input:** Raw Defactify Parquet files (`data/defactify/data/*.parquet`).
- **Output:**
  - Detector mode: `(image_tensor: torch.FloatTensor [3, 256, 256] in [0, 1], label_a: int)`
  - Audit mode (`include_metadata=True`): `(image_tensor, label_a, label_b, caption)`
- **Public Interface:**
  - `DefactifyDataset(split: str, data_dir: Optional[str], include_metadata: bool, transform: Optional[Callable])`
- **Dependencies:** `pyarrow.parquet`, `torch`, `torchvision.transforms.functional.to_tensor`, `PIL`, `src.data.preprocessing`.
- **Used by:** Validation, testing, and error analysis pipelines.
- **Tests:** `src/data/tests/test_loader.py` (Tests A–G).
- **Modification Notes:** Uses a lightweight index at `__init__`. Does NOT retain raw image bytes in memory.

---

### `src/data/materializer.py`
- **Responsibility:** Materialization engine for transforming raw training data into versioned, pre-decoded processed Parquet datasets + metadata manifests.
- **Block:** Block 1 (Training Materialization).
- **Input:** Raw train split Parquet files (`data/defactify/data/train-*.parquet`).
- **Output:**
  - Materialized Parquet files (`data/processed/train/part-*.parquet`) with columns: `image_rgb: binary (256*256*3 flat uint8)` and `label_a: int32`.
  - JSON metadata manifest (`manifest.json`) containing split statistics, generator audit counts, preprocessing version (v2, INTER_AREA), and file lists.
- **Public Interface:**
  - `materialize_training_dataset(raw_data_dir: Optional[str], output_base_dir: Optional[str], max_samples: Optional[int], overwrite: bool) -> str`
  - `get_processed_train_dir(output_base_dir: Optional[str]) -> str`
  - `PREPROCESSING_VERSION: str = "v2"`
- **Dependencies:** `pyarrow`, `pyarrow.parquet`, `numpy`, `PIL`, `src.data.preprocessing`.
- **Used by:** Training data preparation, future Block 2 training feature extraction, future Block 3 model training.
- **Tests:** `src/data/tests/test_materializer.py` (Tests C–K).
- **Modification Notes:** Buffers images in batches of 1,000 (~196 MB buffer) before flushing to disk. Excludes all metadata (`label_b`, `caption`, `path`) from output Parquet files.

---

### `src/data/processed_loader.py`
- **Responsibility:** High-throughput PyTorch Dataset reader for materialized training Parquet datasets.
- **Block:** Block 1 (Materialized Training Reader).
- **Input:** Materialized training Parquet files (`data/processed/train/part-*.parquet`).
- **Output:** `(image_tensor: torch.FloatTensor [3, 256, 256] in [0, 1], label_a: int)`.
- **Public Interface:**
  - `ProcessedTrainDataset(processed_dir: Optional[str], transform: Optional[Callable])`
- **Dependencies:** `pyarrow.parquet`, `numpy`, `torch`, `src.data.materializer`.
- **Used by:** Future Block 2 training extraction and future Block 3 RGB baseline model training.
- **Tests:** `src/data/tests/test_materializer.py` (Tests D, E, H).
- **Modification Notes:** Decodes flat uint8 bytes directly to tensors without JPEG decompression overhead.

---

### `src/data/__init__.py`
- **Responsibility:** Package entrypoint exposing canonical preprocessing, raw dataset loader, materializer, and processed loader.
- **Block:** Block 1.
- **Public Interface:** Exposes `preprocess`, `centered_square_crop`, `resize_to_canonical`, `CANONICAL_SIZE`, `INTERPOLATION`, `DefactifyDataset`, `materialize_training_dataset`, `get_processed_train_dir`, `PREPROCESSING_VERSION`, `ProcessedTrainDataset`.

---

## Block 1 Tests

- `src/data/tests/test_preprocessing.py`: 7 tests verifying geometry, RGB conversion, determinism, metadata isolation, and raw data integrity snapshot.
- `src/data/tests/test_loader.py`: 7 tests verifying raw loader split lengths (train: 42k, val: 9k, test: 45k), tensor shapes/dtypes/ranges, both classes, detector mode, and audit mode.
- `src/data/tests/test_materializer.py`: 9 tests verifying sample counts, shape/dtype, label alignment, absence of forbidden metadata, determinism, exact parity between on-demand and materialized paths, corrupt input handling, idempotency, and manifest counts.
- `src/data/tests/run_tests.py`: Master test runner executing all 23 Block 1 tests across all 3 suites.

---

## Analysis / Audit Scripts

### `src/analysis/defactify_confound_audit.py`
- **Responsibility:** Comprehensive 96,000-image dataset audit establishing resolution distributions, aspect-ratio confounds, quantization tables, exact duplicates, and caption overlaps.
- **Output:** `src/analysis/defactify_confound_audit_report.json` and `src/analysis/defactify_confound_audit_summary.md`.

---

## Block 2: Forensic Feature Extraction & Pipeline

### `src/forensics/branch_a_frequency/fft.py`
- **Responsibility:** Standard Fourier / frequency-domain feature extraction on canonical grayscale image tensors (Branch A1).
- **Block:** Block 2 (Branch A1).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 4 scalar float features:
  - `fft_low_freq_ratio`
  - `fft_mid_freq_ratio`
  - `fft_high_freq_ratio`
  - `fft_spectral_centroid`
- **Public Interface:**
  - `standard_fft_features(image: torch.Tensor) -> Dict[str, float]`
  - `frequency_radius(h: int, w: int, device: Optional[torch.device], dtype: torch.dtype) -> torch.Tensor`
  - `rgb_to_gray(image: torch.Tensor) -> torch.Tensor`
  - `compute_fft_diagnostics(image: torch.Tensor, radial_bins: int = 64) -> Dict[str, torch.Tensor]`
- **Dependencies:** Standard library `math`, `typing`, `torch`.
- **Used by:** `src/forensics/branch_a_frequency/features.py`, `src/forensics/branch_a_frequency/synthbuster.py`.
- **Tests:** `src/forensics/tests/test_branch_a_frequency.py` (Tests 1, 4, 5, 7, 8, 9, 10).
- **Modification Notes:** DC component is removed prior to FFT via mean subtraction. Uses orthonormal FFT normalization (`norm="ortho"`) and `torch.fft.fftshift`. Diagnostic maps (phase, log-magnitude, radial power profile) are separated in `compute_fft_diagnostics` and excluded from detector features.

---

### `src/forensics/branch_a_frequency/synthbuster.py`
- **Responsibility:** Synthbuster-inspired periodicity analysis via per-channel cross-difference residual FFT and directional frequency peak sampling (Branch A2).
- **Block:** Block 2 (Branch A2).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 30 scalar float features (10 per channel for R, G, B):
  - `synth_{channel}_p{2,4,8}_{x,y,d}_mean` (27 directional peak means)
  - `synth_{channel}_fft_highfreq_ratio` (3 residual high-frequency energy ratios)
- **Public Interface:**
  - `synthbuster_periodicity_features(image: torch.Tensor, periods: Sequence[int] = (2, 4, 8)) -> Dict[str, float]`
  - `cross_difference(image: torch.Tensor) -> torch.Tensor`
- **Dependencies:** `torch`, `typing`, `src.forensics.branch_a_frequency.fft.frequency_radius`.
- **Used by:** `src/forensics/branch_a_frequency/features.py`.
- **Tests:** `src/forensics/tests/test_branch_a_frequency.py` (Tests 2, 3, 4, 5, 6, 7, 8, 10).
- **Modification Notes:** Cross-difference operator reduces spatial dimensions from $256 \times 256$ to $255 \times 255$. Coordinate indexing is dynamically bounded by residual dimensions. R, G, and B channels are processed strictly independently.

---

### `src/forensics/branch_a_frequency/features.py`
- **Responsibility:** Branch A composition layer combining A1 and A2 into unified 34-feature dictionary.
- **Block:** Block 2 (Branch A Composition).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 34 scalar float features.
- **Public Interface:**
  - `extract_branch_a_features(image: torch.Tensor, periods: Sequence[int] = (2, 4, 8)) -> Dict[str, float]`
- **Dependencies:** `torch`, `typing`, `src.forensics.branch_a_frequency.fft`, `src.forensics.branch_a_frequency.synthbuster`.
- **Used by:** `src/forensics/pipeline.py`, `src/forensics/__init__.py`.
- **Tests:** `src/forensics/tests/test_branch_a_frequency.py` (Tests 3, 5, 7, 8, 10, 11).

---

### `src/forensics/branch_a_frequency/__init__.py`
- **Responsibility:** Package entrypoint exposing clean Branch A API (`extract_branch_a_features`, `standard_fft_features`, `synthbuster_periodicity_features`, etc.).
- **Block:** Block 2.

---

### `src/forensics/branch_b_wavelet/haar.py`
- **Responsibility:** Pure PyTorch 2D Haar Discrete Wavelet Transform (DWT) decomposition.
- **Block:** Block 2 (Branch B Haar DWT).
- **Input:** Grayscale image tensor `[..., 1, 256, 256]` or `[..., 256, 256]`, float32.
- **Output:** Dictionary of 10 subband coefficient tensors across 3 levels:
  - `LL3`, `LH3`, `HL3`, `HH3` (each $[..., 32, 32]$)
  - `LH2`, `HL2`, `HH2` (each $[..., 64, 64]$)
  - `LH1`, `HL1`, `HH1` (each $[..., 128, 128]$)
- **Public Interface:**
  - `haar_2d_level(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`
  - `haar_dwt_3level(x: torch.Tensor) -> Dict[str, torch.Tensor]`
- **Dependencies:** Standard library `typing`, `torch`. Pure PyTorch runtime with no third-party wavelet dependencies.
- **Used by:** `src/forensics/branch_b_wavelet/features.py`.
- **Tests:** `src/forensics/tests/test_branch_b_wavelet.py` (Tests 2, 3, 6).
- **Modification Notes:** Uses separable 2D orthonormal Haar basis scaled by 0.5 ($1/\sqrt{2} \times 1/\sqrt{2} = 0.5$). Downstream statistics (energy, absmean, entropy) are strictly sign-invariant.

---

### `src/forensics/branch_b_wavelet/features.py`
- **Responsibility:** Branch B wavelet feature extraction producing exactly 30 multiscale scalar features.
- **Block:** Block 2 (Branch B Wavelet Features).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 30 scalar float features:
  - `LL3_energy`, `LL3_entropy` (2 scalars)
  - For each of 9 detail subbands ($LH_3, HL_3, HH_3, LH_2, HL_2, HH_2, LH_1, HL_1, HH_1$):
    - `{band}_energy`, `{band}_absmean`, `{band}_entropy` (27 scalars)
  - `detail_energy_ratio` (1 scalar)
- **Public Interface:**
  - `extract_branch_b_features(image: torch.Tensor) -> Dict[str, float]`
  - `safe_entropy(x: torch.Tensor, bins: int = 256) -> float`
  - `rgb_to_gray(image: torch.Tensor) -> torch.Tensor`
- **Dependencies:** `torch`, `typing`, `src.forensics.branch_b_wavelet.haar`.
- **Used by:** `src/forensics/pipeline.py`, `src/forensics/__init__.py`.
- **Tests:** `src/forensics/tests/test_branch_b_wavelet.py` (Tests 1, 4, 5, 6, 7, 8, 9, 10, 11).
- **Modification Notes:** Independent of Branch A. `detail_energy_ratio` is defined as $\frac{\sum \text{detail energy}}{\text{total energy} + 10^{-12}}$ where $\text{total energy} = \text{energy}(LL_3) + \sum \text{detail energy}$.

---

### `src/forensics/branch_b_wavelet/__init__.py`
- **Responsibility:** Package entrypoint exposing clean Branch B API (`extract_branch_b_features`, `haar_dwt_3level`, `haar_2d_level`, `safe_entropy`, `rgb_to_gray`).
- **Block:** Block 2.

---

### `src/forensics/branch_c_texture/lbp.py`
- **Responsibility:** Standard rotation-invariant uniform LBP computation ($P=8, R=1$), deterministic Canny edge extraction, and masked LBP histogram and summary statistics.
- **Block:** Block 2 (Branch C LBP / Edge-Guided LBP).
- **Input:** Grayscale image tensor `[..., 1, 256, 256]` or `[..., 256, 256]`, float32 in $[0, 1]$.
- **Output:** Tensor of integer uniform LBP bin codes $[..., 256, 256]$ in $[0, 9]$, binary Canny edge mask $[256, 256]$, or dictionary of 16 scalar features.
- **Public Interface:**
  - `compute_lbp_map(gray: torch.Tensor, p: int = 8, r: int = 1) -> torch.Tensor`
  - `compute_canny_edges(gray: torch.Tensor, low_threshold: float = 50.0, high_threshold: float = 100.0, sigma: float = 1.0) -> np.ndarray`
  - `extract_lbp_statistics(lbp_map: torch.Tensor, mask: Optional[np.ndarray] = None, prefix: str = "lbp") -> Dict[str, float]`
- **Dependencies:** `cv2`, `numpy`, `torch`, `torch.nn.functional`.
- **Used by:** `src/forensics/branch_c_texture/features.py`.
- **Tests:** `src/forensics/tests/test_branch_c_texture.py` (Tests 1, 3, 4, 6, 7, 8, 9, 10).

---

### `src/forensics/branch_c_texture/glcm.py`
- **Responsibility:** Gray-Level Co-occurrence Matrix (GLCM) computation and Haralick texture feature extraction.
- **Block:** Block 2 (Branch C GLCM).
- **Input:** Grayscale image tensor `[..., 1, 256, 256]` or `[..., 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 24 scalar float features (18 directional means across $d \in \{1, 2, 4\}$ + 6 directional standard deviations across 4 angles at $d=1$).
- **Public Interface:**
  - `quantize_grayscale(gray: torch.Tensor, levels: int = 16) -> np.ndarray`
  - `compute_glcm_matrix(image_quantized: np.ndarray, levels: int = 16, distances: Sequence[int] = (1, 2, 4), angles: Sequence[float] = (0, np.pi/4, np.pi/2, 3*np.pi/4)) -> np.ndarray`
  - `compute_glcm_features(gray: torch.Tensor, levels: int = 16, distances: Sequence[int] = (1, 2, 4), angles: Sequence[float] = (0, np.pi/4, np.pi/2, 3*np.pi/4)) -> Dict[str, float]`
- **Dependencies:** `numpy`, `skimage.feature.graycomatrix`, `skimage.feature.graycoprops`, `torch`.
- **Used by:** `src/forensics/branch_c_texture/features.py`.
- **Tests:** `src/forensics/tests/test_branch_c_texture.py` (Tests 2, 5, 7, 8, 9, 10).
- **Modification Notes:** Linear quantization to 16 levels eliminates matrix sparsity and boosts efficiency. Directional std at $d=1$ captures micro-scale anisotropy.

---

### `src/forensics/branch_c_texture/features.py`
- **Responsibility:** Branch C composition layer exposing three independent, alternative texture extraction pipelines.
- **Block:** Block 2 (Branch C Features).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of scalar float features:
  - `extract_branch_c_lbp_features`: Exactly 16 features (`lbp_*`).
  - `extract_branch_c_glcm_features`: Exactly 24 features (`glcm_*`).
  - `extract_branch_c_lbp_edge_features`: Exactly 16 features (`lbp_edge_*`).
- **Public Interface:**
  - `extract_branch_c_lbp_features(image: torch.Tensor) -> Dict[str, float]`
  - `extract_branch_c_glcm_features(image: torch.Tensor) -> Dict[str, float]`
  - `extract_branch_c_lbp_edge_features(image: torch.Tensor) -> Dict[str, float]`
  - `rgb_to_gray(image: torch.Tensor) -> torch.Tensor`
- **Dependencies:** `torch`, `src.forensics.branch_c_texture.lbp`, `src.forensics.branch_c_texture.glcm`.
- **Used by:** `src/forensics/pipeline.py`, `src/forensics/__init__.py`.
- **Tests:** `src/forensics/tests/test_branch_c_texture.py` (Tests 1, 2, 3, 7, 8, 9, 10, 11, 12).

---

### `src/forensics/branch_c_texture/__init__.py`
- **Responsibility:** Package entrypoint exposing clean Branch C API (`extract_branch_c_lbp_features`, `extract_branch_c_glcm_features`, `extract_branch_c_lbp_edge_features`, `compute_lbp_map`, `compute_canny_edges`, `compute_glcm_matrix`, `compute_glcm_features`, etc.).
- **Block:** Block 2.

---

### `src/forensics/branch_d_residual/highpass.py`
- **Responsibility:** Gaussian high-pass residual computation ($5 \times 5, \sigma=1.0$) and candidate scalar statistics extraction (Branch D candidate `D_HIGHPASS`).
- **Block:** Block 2 (Branch D High-Pass Residual).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of 5 candidate scalar float features (`hp_absmean`, `hp_std`, `hp_energy`, `hp_kurtosis`, `hp_entropy`).
- **Public Interface:**
  - `compute_highpass_residual(image: torch.Tensor) -> torch.Tensor`
  - `extract_highpass_features(image: torch.Tensor) -> Dict[str, float]`
- **Dependencies:** `torch`, `torch.nn.functional`.
- **Used by:** `src/forensics/branch_d_residual/features.py`.
- **Tests:** `src/forensics/tests/test_branch_d_residual.py` (Tests 1, 4, 5, 6, 8, 9, 10, 11, 12).
- **Modification Notes:** Linear subtractive low-pass operator. Replicate padding (2 pixels) prevents boundary ringing. Candidate representation; final subset determined in Block 4.

---

### `src/forensics/branch_d_residual/laplacian.py`
- **Responsibility:** Discrete 8-neighbor Laplacian 2nd-order curvature residual computation and candidate scalar statistics extraction (Branch D candidate `D_LAPLACIAN`).
- **Block:** Block 2 (Branch D Laplacian Residual).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of 5 candidate scalar float features (`lap_absmean`, `lap_std`, `lap_energy`, `lap_kurtosis`, `lap_entropy`).
- **Public Interface:**
  - `compute_laplacian_residual(image: torch.Tensor) -> torch.Tensor`
  - `extract_laplacian_features(image: torch.Tensor) -> Dict[str, float]`
- **Dependencies:** `torch`, `torch.nn.functional`, `src.forensics.branch_d_residual.highpass._safe_kurtosis`, `src.forensics.branch_d_residual.highpass._safe_entropy`.
- **Used by:** `src/forensics/branch_d_residual/features.py`.
- **Tests:** `src/forensics/tests/test_branch_d_residual.py` (Tests 2, 4, 5, 6, 8, 9, 10, 11, 12).
- **Modification Notes:** Alternative linear residual candidate. Evaluates spatial curvature and gradient discontinuity.

---

### `src/forensics/branch_d_residual/median_filter.py`
- **Responsibility:** Non-linear $3 \times 3$ median filter residual computation and candidate scalar statistics extraction (Branch D candidate `D_MFR`).
- **Block:** Block 2 (Branch D Median Filter Residual).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of 5 candidate scalar float features (`mfr_absmean`, `mfr_std`, `mfr_energy`, `mfr_kurtosis`, `mfr_entropy`).
- **Public Interface:**
  - `compute_median_filter_residual(image: torch.Tensor) -> torch.Tensor`
  - `extract_mfr_features(image: torch.Tensor) -> Dict[str, float]`
- **Dependencies:** `torch`, `torch.nn.functional`, `src.forensics.branch_d_residual.highpass._safe_kurtosis`, `src.forensics.branch_d_residual.highpass._safe_entropy`.
- **Used by:** `src/forensics/branch_d_residual/features.py`.
- **Tests:** `src/forensics/tests/test_branch_d_residual.py` (Tests 3, 4, 5, 7, 8, 9, 10, 11, 12).
- **Modification Notes:** Pure PyTorch implementation via `F.unfold` and `.median()`. Non-linear rank-order operator; numerically identical to `scipy.ndimage.median_filter(mode='nearest')`.

---

### `src/forensics/branch_d_residual/features.py`
- **Responsibility:** Branch D composition layer exposing three independent, alternative residual extraction pipelines.
- **Block:** Block 2 (Branch D Composition).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of scalar float features:
  - `extract_branch_d_highpass_features`: 5 features (`hp_*`).
  - `extract_branch_d_laplacian_features`: 5 features (`lap_*`).
  - `extract_branch_d_mfr_features`: 5 features (`mfr_*`).
- **Public Interface:**
  - `extract_branch_d_highpass_features(image: torch.Tensor) -> Dict[str, float]`
  - `extract_branch_d_laplacian_features(image: torch.Tensor) -> Dict[str, float]`
  - `extract_branch_d_mfr_features(image: torch.Tensor) -> Dict[str, float]`
- **Dependencies:** `torch`, `src.forensics.branch_d_residual.highpass`, `src.forensics.branch_d_residual.laplacian`, `src.forensics.branch_d_residual.median_filter`.
- **Used by:** `src/forensics/pipeline.py`, `src/forensics/__init__.py`.
- **Tests:** `src/forensics/tests/test_branch_d_residual.py` (Tests 1, 2, 3, 8, 11, 12).

---

### `src/forensics/branch_d_residual/__init__.py`
- **Responsibility:** Package entrypoint exposing clean Branch D API (`extract_branch_d_highpass_features`, `extract_branch_d_laplacian_features`, `extract_branch_d_mfr_features`, `compute_highpass_residual`, `compute_laplacian_residual`, `compute_median_filter_residual`, etc.).
- **Block:** Block 2.

---

### `src/forensics/branch_e/dct.py`
- **Responsibility:** 8×8 block DCT coefficient fingerprint and zonal statistics extraction (Branch E candidate `E1_DCT`).
- **Block:** Block 2 (Branch E1 DCT).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 10 candidate scalar float features (`dct_ac_mean_abs`, `dct_ac_energy`, `dct_ac_kurtosis`, `dct_sparsity_ratio`, `dct_low_freq_ratio`, `dct_mid_freq_ratio`, `dct_high_freq_ratio`, `dct_anisotropy`, `dct_benford_ssd`, `dct_block_var_mean`).
- **Public Interface:**
  - `compute_block_dct(image: torch.Tensor) -> torch.Tensor`
  - `extract_branch_e_dct_features(image: torch.Tensor) -> Dict[str, float]`
- **Dependencies:** `torch`, `torch.nn.functional`, `math`.
- **Used by:** `src/forensics/branch_e/features.py`, `src/forensics/pipeline.py`.
- **Tests:** `src/forensics/tests/test_branch_e_forensics.py` (Tests 1, 6, 7, 8, 9, 10, 11, 12).

---

### `src/forensics/branch_e/recompression.py`
- **Responsibility:** Controlled multi-quality in-memory JPEG recompression response extraction across $Q \in \{95, 90, 75, 60\}$ (Branch E candidate `E2_RESP`).
- **Block:** Block 2 (Branch E2 Recompression Response).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 8 candidate scalar float features (`ela_q95_mean`, `ela_q90_mean`, `ela_q75_mean`, `ela_q60_mean`, `ela_q90_energy`, `ela_slope_q90_q75`, `ela_ratio_q90_q75`, `ela_q90_gini`).
- **Public Interface:**
  - `compute_recompression_error_map(image: torch.Tensor, quality: int) -> Tuple[torch.Tensor, torch.Tensor]`
  - `extract_branch_e_recompression_features(image: torch.Tensor) -> Dict[str, float]`
- **Dependencies:** `cv2`, `numpy`, `torch`.
- **Used by:** `src/forensics/branch_e/features.py`, `src/forensics/branch_e/phase_stability.py`, `src/forensics/pipeline.py`.
- **Tests:** `src/forensics/tests/test_branch_e_forensics.py` (Tests 2, 7, 8, 9, 10, 11, 12).

---

### `src/forensics/branch_e/phase_stability.py`
- **Responsibility:** Compression-stable Fourier phase spectrum response extraction under controlled JPEG recompression (Branch E candidate `E3_PHASE`).
- **Block:** Block 2 (Branch E3 Phase Stability).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 4 candidate scalar float features (`phase_corr_q90`, `phase_corr_q75`, `phase_diff_energy_q90`, `phase_hf_stability_q90`).
- **Public Interface:**
  - `extract_branch_e_phase_features(image: torch.Tensor) -> Dict[str, float]`
- **Dependencies:** `torch`, `math`, `src.forensics.branch_e.recompression`.
- **Used by:** `src/forensics/branch_e/features.py`, `src/forensics/pipeline.py`.
- **Tests:** `src/forensics/tests/test_branch_e_forensics.py` (Tests 3, 7, 8, 9, 10, 11, 12).

---

### `src/forensics/branch_e/grid.py`
- **Responsibility:** Canonical $8 \times 8$ grid boundary step discontinuity and blocking factor extraction (Branch E candidate `E4_GRID`).
- **Block:** Block 2 (Branch E4 Grid).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 4 candidate scalar float features (`grid_h_ratio`, `grid_v_ratio`, `grid_strength`, `grid_anisotropy`).
- **Public Interface:**
  - `compute_grid_discontinuities(image: torch.Tensor) -> Dict[str, float]`
  - `extract_branch_e_grid_features(image: torch.Tensor) -> Dict[str, float]`
- **Dependencies:** `torch`.
- **Used by:** `src/forensics/branch_e/features.py`, `src/forensics/pipeline.py`.
- **Tests:** `src/forensics/tests/test_branch_e_forensics.py` (Tests 4, 7, 8, 9, 10, 11, 12).

---

### `src/forensics/branch_e/features.py`
- **Responsibility:** Branch E composition layer unifying E1, E2, E3, and E4 into unified 26-feature dictionary.
- **Block:** Block 2 (Branch E Composition).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`, float32 in $[0, 1]$.
- **Output:** Dictionary of exactly 26 candidate scalar float features.
- **Public Interface:**
  - `extract_branch_e_features(image: torch.Tensor) -> Dict[str, float]`
- **Dependencies:** `torch`, `src.forensics.branch_e.dct`, `src.forensics.branch_e.recompression`, `src.forensics.branch_e.phase_stability`, `src.forensics.branch_e.grid`.
- **Used by:** `src/forensics/pipeline.py`, `src/forensics/__init__.py`.
- **Tests:** `src/forensics/tests/test_branch_e_forensics.py` (Tests 5, 7, 8, 9, 10, 11, 12).

---

### `src/forensics/branch_e/__init__.py`
- **Responsibility:** Package entrypoint exposing clean Branch E API (`extract_branch_e_features`, `extract_branch_e_dct_features`, `extract_branch_e_recompression_features`, `extract_branch_e_phase_features`, `extract_branch_e_grid_features`, `compute_block_dct`, `compute_recompression_error_map`, `compute_grid_discontinuities`).
- **Block:** Block 2.

---

### `src/forensics/pipeline.py`
- **Responsibility:** Multi-branch forensic orchestration pipeline. Dispatches canonical image representations to enabled branches and merges scalar feature vectors.
- **Block:** Block 2 (Orchestration).
- **Input:** Canonical RGB image tensor `[3, 256, 256]` or `[B, 3, 256, 256]`.
- **Output:** Combined scalar feature dictionary:
  - `branches=["A"]` -> 34 features
  - `branches=["B"]` -> 30 features
  - `branches=["C_LBP"]` -> 16 features
  - `branches=["C_GLCM"]` -> 24 features
  - `branches=["C_LBP_EDGE"]` -> 16 features
  - `branches=["D_HIGHPASS"]` -> 5 features
  - `branches=["D_LAPLACIAN"]` -> 5 features
  - `branches=["D_MFR"]` -> 5 features
  - `branches=["D"]` -> 15 features (alias for D_HIGHPASS + D_LAPLACIAN + D_MFR)
  - `branches=["E_DCT"]` -> 10 features
  - `branches=["E_RESP"]` -> 8 features
  - `branches=["E_PHASE"]` -> 4 features
  - `branches=["E_GRID"]` -> 4 features
  - `branches=["E"]` -> 26 features (unified Branch E)
  - Multi-branch combinations (e.g. `["A", "B", "C_LBP", "D_MFR", "E"]` -> 111 features)
- **Public Interface:**
  - `ForensicPipeline(branches: Optional[Sequence[str]] = None)`
  - `pipeline.extract(image: torch.Tensor) -> Dict[str, float]`
  - `pipeline.get_feature_names() -> List[str]`
  - `pipeline.branches -> Tuple[str, ...]`
  - `DEFAULT_BRANCHES: Tuple[str, ...] = ("A", "B")`
- **Dependencies:** `torch`, `typing`, `src.forensics.branch_a_frequency`, `src.forensics.branch_b_wavelet`, `src.forensics.branch_c_texture`, `src.forensics.branch_d_residual`, `src.forensics.branch_e`.
- **Used by:** `src/forensics/run_forensic_pipeline.py`, future Block 3/4 feature extraction pipelines.
- **Tests:** `src/forensics/tests/test_branch_a_frequency.py` (Test 11), `src/forensics/tests/test_branch_b_wavelet.py` (Test 11), `src/forensics/tests/test_branch_c_texture.py` (Tests 11, 12, 13), `src/forensics/tests/test_branch_d_residual.py` (Test 12), `src/forensics/tests/test_branch_e_forensics.py` (Test 12).

---

### `src/forensics/run_forensic_pipeline.py`
- **Responsibility:** CLI entrypoint for running forensic feature extraction across splits or synthetic tensors.
- **Block:** Block 2 (CLI Execution).
- **Usage:** `.venv/bin/python src/forensics/run_forensic_pipeline.py [--branches A B C_LBP D_MFR E] [--synthetic] [--split train] [--max-samples N]`
- **Dependencies:** `argparse`, `torch`, `src.forensics.pipeline`, `src.data.loader`.

---

### `src/forensics/__init__.py`
- **Responsibility:** Package root exposing `ForensicPipeline`, Branch A, Branch B, Branch C, Branch D, and Branch E feature extractors and core public interfaces.
- **Block:** Block 2.

---

## Block 2 Tests

- `src/forensics/tests/test_branch_a_frequency.py`: 11 comprehensive tests verifying A1 contract (4 features), A2 contract (30 features), unified Branch A contract (34 features), cross-difference geometry ($256 \times 256 \to 255 \times 255$), determinism, RGB channel independence in A2, numerical safety on zero/constant inputs, pattern discrimination, diagnostic map isolation, real Defactify sample smoke extraction, and ForensicPipeline branch validation and numerical equivalence.
- `src/forensics/tests/test_branch_b_wavelet.py`: 11 comprehensive tests verifying Branch B contract (30 features), multiscale subband shapes across all 3 levels, Parseval energy conservation, detail_energy_ratio arithmetic, safe_entropy bounds/invariance, reference equivalence and PyWavelets comparison, determinism, numerical safety on constant/pathological inputs, pattern orientation discrimination, real Defactify sample smoke extraction, and ForensicPipeline multi-branch orchestration.
- `src/forensics/tests/test_branch_c_texture.py`: 15 comprehensive tests verifying C_LBP contract (16 features), C_GLCM contract (24 features), C_LBP_EDGE contract (16 features), uniform LBP bit transitions & rotation-invariant bin mapping, GLCM 16-level quantization/symmetry/normalization, Canny edge detection & sparse mask fallback safety, determinism, numerical safety on pathological inputs, texture discrimination, real Defactify sample smoke extraction, ForensicPipeline alternative isolation, multi-branch composition, strict mutual exclusion among Branch C variants, empty/sparse/dense edge mask behaviors, and edge LBP determinism.
- `src/forensics/tests/test_branch_d_residual.py`: 12 comprehensive tests verifying D_HIGHPASS contract (5 features), D_LAPLACIAN contract (5 features), D_MFR contract (5 features), residual map shapes/dtypes, constant/zero image exact zero response & numerical safety, exact filter/kernel weights, Scipy reference equivalence for MFR, determinism, pathological input safety, pattern discrimination, real Defactify sample smoke extraction, and ForensicPipeline individual candidate isolation & multi-branch orchestration.
- `src/forensics/tests/test_branch_e_forensics.py`: 12 comprehensive tests verifying E1_DCT contract (10 features), E2_RESP contract (8 features), E3_PHASE contract (4 features), E4_GRID contract (4 features), unified Branch E contract (26 features), block DCT shape/orthonormality, constant/zero image safety, determinism, pathological input safety, pattern discrimination, real Defactify sample smoke extraction, and ForensicPipeline candidate sub-branch isolation and multi-branch orchestration.
- `src/forensics/tests/run_tests.py`: Standalone master test runner executing all 61 forensic tests across Branch A (11), Branch B (11), Branch C (15), Branch D (12), and Branch E (12) with full reporting.



