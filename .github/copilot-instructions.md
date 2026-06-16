# XRD AI Denoising Project - AI Agent Instructions

## Project Overview
This is an **X-Ray Diffraction (XRD) data denoising system** that provides two complementary approaches:
1. **Classical Methods** (proven, immediate results) - Savitzky-Golay, wavelet, median, and hybrid filtering
2. **Deep Learning Methods** (AI-based) - 1D/2D U-Net architectures with Anscombe transform preprocessing

**Data Format**: XRD patterns with 2θ (angle) vs Intensity; can be 1D (linear) or 2D (heatmaps).

## Architecture & Key Components

### Classical Methods Pipeline
- **File**: `1DXRDDenoiser-ClassicalMethods(Proven).py`
- **Input**: XRD 1D data (2-column format: 2θ, intensity or single intensity column)
- **Processing Steps**:
  1. Baseline correction (rolling minimum + smoothing)
  2. Denoising filter (user selects: `savgol`, `wavelet`, `median`, `hybrid`)
  3. Peak detection with prominence-based thresholding
  4. Quality metrics (SNR, peak count, noise %)
- **Outputs**: Denoised data, peak list, comparison plot
- **Key Config**: `METODE_DENOISING` (string) and `DENOISING_STRENGTH` (1-10 scale)

### Deep Learning Pipeline (1D)
- **File**: `denoising1DXRD-versiClaude.py`
- **Architecture**: U-Net1D or SimpleCNN1D with configurable base channels
- **Variance Stabilization**: Anscombe Transform (Poisson noise → Gaussian)
- **Classes**:
  - `AnscombeTrans`: Forward/inverse transform for noise stabilization
  - `UNet1D`: 4-level encoder-decoder with skip connections
  - `SimpleCNN1D`: Lightweight alternative (faster training)
  - `XRD1DPreprocessor`: Handles Anscombe + min-max normalization
  - `XRD1DQualityEvaluator`: Computes SNR, peak detection, baseline noise
  - `XRD1DDenoiser`: Main inference pipeline
- **Inference Flow**: Raw data → Anscombe → Normalize → Model → Denorm → Inverse Anscombe

### Deep Learning Pipeline (2D)
- **File**: `denoising2DXRD-versiClaude.py`
- **Architecture**: Tunable U-Net with configurable depth/channels
- **Key Features**:
  - **TunableUNet**: Adjustable `cb` (base channels), `r` (growth rate), `depth` (layers)
  - **QuantileUNet**: Outputs q0.05, q0.50, q0.95 for prediction intervals
  - **PinballLoss**: Quantile regression loss function
  - **ConformalPredictor**: Calibrates confidence intervals
- **Channel Calculation**: `ch = int(cb * (r ** i))` for each level

## Data Pipeline & Conventions

### Data Format Standards
- **XRD Files**: `.xy`, `.txt`, `.dat`, `.csv` (2-column: 2θ and intensity)
- **Input Range**: 2θ typically 5°-85°; intensity values Poisson-distributed
- **Preprocessing**: Always apply Anscombe transform before deep learning (not for classical methods)
- **Normalization**: Min-max to [0, 1] after Anscombe; preserve original scale info

### Quality Evaluation
All pipelines evaluate using three metrics:
1. **SNR** (Signal-to-Noise Ratio): `signal_mean / noise_std`
2. **Peak Detection**: Prominence-based with configurable thresholds (default: 2-5% of max)
3. **Quality Score**: `(num_peaks * snr) / (baseline_noise + 1)`

## Key Patterns & Best Practices

### When to Use Classical vs Deep Learning
- **Classical Methods**: Use when training data is unavailable or results need immediate validation
- **Deep Learning**: Use when model is pre-trained; requires significant labeled dataset for training

### Skip Connection Handling (2D U-Net)
Critical fix documented in `TunableUNet.forward()`:
```python
# Decoder input = upsampled features + skip connection
x = torch.cat([skip, x], dim=1)  # Concatenate along channel dimension
# Handle spatial mismatches from odd-sized inputs
if x.shape[2:] != skip.shape[2:]:
    diff_h = skip.shape[2] - x.shape[2]
    diff_w = skip.shape[3] - x.shape[3]
    x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
```

### Size Matching (1D)
Use `_match_size()` helper to handle variable input lengths:
```python
def _match_size(self, x, target):
    if x.shape[2] != target.shape[2]:
        diff = target.shape[2] - x.shape[2]
        x = F.pad(x, [diff // 2, diff - diff // 2])
    return x
```

### Preprocessing/Postprocessing Symmetry
- **Forward**: Raw intensity → Anscombe → Min-max normalize
- **Inverse**: Denormalize → Inverse Anscombe → Clamp to [0, ∞)
- **Critical**: Store `data_min/data_max` during forward pass for exact reconstruction

## File Organization

```
project_root/
├── 1DXRDDenoiser-ClassicalMethods(Proven).py     # Classical baseline
├── denoising1DXRD-versiClaude.py                 # 1D deep learning
├── denoising2DXRD-versiClaude.py                 # 2D deep learning
├── test-denoising1DXRD-versiClaude.py            # 1D tests
├── data/tests/
│   ├── amorf/                                     # Amorphous XRD samples
│   ├── background/                                # Background noise references
│   ├── dynamic range/                             # High dynamic range samples
│   └── extreme noise/                             # Challenging noisy samples
├── data/train/clean/                              # Ground truth training data
└── [output files: denoised_*.xy, peaks_*.txt, *.png]
```

## Development Workflow

### Running Classical Methods
```python
# 1. Edit config at top of file
NAMA_FILE = 'Data XRD TiO2(1).xy'
METODE_DENOISING = 'hybrid'       # or 'savgol', 'wavelet', 'median'
DENOISING_STRENGTH = 5             # 1-10 scale
# 2. Execute: python 1DXRDDenoiser-ClassicalMethods(Proven).py
# 3. Check outputs: denoised_*.xy, peaks_*.txt, comparison_plot_classicalmethods.png
```

### Running Deep Learning (1D)
```python
denoiser = XRD1DDenoiser(model_type='simple_cnn', model_path='model.pth', device='cuda')
results, denoised = denoiser.evaluate(two_theta, noisy_intensity)
# Batch processing: batch_denoise_folder('data/tests/amorf', denoiser)
```

### Running Deep Learning (2D)
```python
model = TunableUNet(cb=32, r=1.8, depth=5)
# or with quantile regression
model = QuantileUNet(cb=32, r=1.8, depth=5)
q_05, q_50, q_95 = model(input_2d)  # prediction intervals
```

## Common Tasks & Solutions

**Task: Adjust denoising aggressiveness**
- Classical: Modify `DENOISING_STRENGTH` (lower = gentler)
- Deep Learning: Retrain with different learning rate or data augmentation

**Task: Detect if peaks are being over-smoothed**
- Check peak count before/after in output metrics
- If `len(peaks_denoised) < 0.8 * len(peaks_original)`: reduce strength or use median filter

**Task: Handle odd-sized 2D inputs**
- `TunableUNet` automatically pads; ensure bias correction in postprocessing

**Task: Use pre-trained model for new dataset**
- Load with `denoiser.load_model('path.pth')`; reuse preprocessor fitted on training data to ensure consistent normalization

## External Dependencies
- **PyTorch**: Deep learning models, tensor operations
- **NumPy**: Array processing
- **SciPy**: Signal processing (savgol_filter, find_peaks, medfilt, uniform_filter1d)
- **Matplotlib**: Visualization (3-panel plots with linear/log scales and residuals)

## Testing & Validation
- **Synthetic data**: Generated in example usage (Gaussian peaks + Poisson noise)
- **Real data path**: `data/tests/amorf/LBNL_B_pattern_group_0_*.xy` (130+ samples)
- **Quality assertion**: SNR should improve; peak count should maintain ≥80% baseline

---
**Last Updated**: January 2026 | **Scope**: 1D/2D XRD denoising with classical and AI methods
