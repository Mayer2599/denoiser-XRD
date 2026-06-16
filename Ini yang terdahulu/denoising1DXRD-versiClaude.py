"""
XRD 1D Profile Deep Learning Denoiser
Khusus untuk data XRD 1D: 2theta vs Intensity

Berdasarkan paper: Milan de Mooij (2024)
Modified untuk 1D signal processing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
import json

# ======================================================
# VARIANCE STABILIZING TRANSFORMATION (VST)
# ======================================================

class AnscombeTrans:
    """Anscombe Transform untuk Poisson noise → Gaussian noise"""
    
    @staticmethod
    def forward(x):
        """A(x) = 2 * sqrt(x + 3/8)"""
        return 2 * np.sqrt(np.maximum(x, 0) + 3/8)
    
    @staticmethod
    def inverse(y):
        """Exact unbiased inverse Anscombe transform"""
        z = (y / 2) ** 2 - 3/8
        return np.maximum(z, 0)


# ======================================================
# 1D CONVOLUTIONAL NEURAL NETWORK
# ======================================================

class Conv1DBlock(nn.Module):
    """1D Convolution Block dengan BatchNorm dan ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet1D(nn.Module):
    """
    1D U-Net untuk XRD profile denoising
    Input: [Batch, 1, Length]
    Output: [Batch, 1, Length]
    """
    
    def __init__(self, base_channels=32):
        super().__init__()
        
        # Encoder
        self.enc1 = Conv1DBlock(1, base_channels)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = Conv1DBlock(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = Conv1DBlock(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool1d(2)
        
        self.enc4 = Conv1DBlock(base_channels*4, base_channels*8)
        self.pool4 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = Conv1DBlock(base_channels*8, base_channels*16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose1d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec4 = Conv1DBlock(base_channels*16, base_channels*8)
        
        self.upconv3 = nn.ConvTranspose1d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = Conv1DBlock(base_channels*8, base_channels*4)
        
        self.upconv2 = nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = Conv1DBlock(base_channels*4, base_channels*2)
        
        self.upconv1 = nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = Conv1DBlock(base_channels*2, base_channels)
        
        # Output
        self.out = nn.Conv1d(base_channels, 1, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder dengan skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = self._match_size(dec4, enc4)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = self._match_size(dec3, enc3)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = self._match_size(dec2, enc2)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self._match_size(dec1, enc1)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)
    
    def _match_size(self, x, target):
        """Match x size to target size dengan padding"""
        if x.shape[2] != target.shape[2]:
            diff = target.shape[2] - x.shape[2]
            x = F.pad(x, [diff // 2, diff - diff // 2])
        return x


# ======================================================
# ALTERNATIVE: SIMPLE 1D CNN
# ======================================================

class SimpleCNN1D(nn.Module):
    """
    Simple 1D CNN untuk denoising
    Lebih cepat dan ringan dari U-Net
    """
    
    def __init__(self, channels=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            
            nn.Conv1d(channels, channels*2, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels*2),
            nn.ReLU(),
            
            nn.Conv1d(channels*2, channels*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels*2),
            nn.ReLU(),
            
            nn.Conv1d(channels*2, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            
            nn.Conv1d(channels, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.encoder(x)


# ======================================================
# XRD 1D PREPROCESSOR
# ======================================================

class XRD1DPreprocessor:
    """Preprocessing untuk data XRD 1D"""
    
    def __init__(self):
        self.anscombe = AnscombeTrans()
        self.data_min = None
        self.data_max = None
    
    def preprocess(self, intensity, fit=False):
        """
        Pipeline:
        1. Anscombe transform
        2. Min-Max normalization
        """
        # Pastikan 1D
        intensity = np.array(intensity).flatten()
        
        # Anscombe transform
        intensity_anscombe = self.anscombe.forward(intensity)
        
        # Normalization
        if fit:
            self.data_min = np.min(intensity_anscombe)
            self.data_max = np.max(intensity_anscombe)
        
        intensity_normalized = (intensity_anscombe - self.data_min) / (self.data_max - self.data_min + 1e-8)
        
        return intensity_normalized
    
    def postprocess(self, intensity):
        """
        Inverse pipeline:
        1. Inverse normalization
        2. Inverse Anscombe
        """
        # Inverse normalization
        intensity_denorm = intensity * (self.data_max - self.data_min) + self.data_min
        
        # Inverse Anscombe
        intensity_original = self.anscombe.inverse(intensity_denorm)
        
        return intensity_original


# ======================================================
# QUALITY EVALUATOR
# ======================================================

class XRD1DQualityEvaluator:
    """Evaluasi kualitas XRD 1D profile"""
    
    @staticmethod
    def calculate_snr(intensity):
        """Signal-to-Noise Ratio"""
        signal = np.mean(intensity)
        noise = np.std(intensity)
        return signal / (noise + 1e-8)
    
    @staticmethod
    def detect_peaks(intensity, prominence=None, width=3, distance=10):
        """Deteksi peaks"""
        if prominence is None:
            prominence = np.max(intensity) * 0.05
        
        peaks, properties = find_peaks(
            intensity,
            prominence=prominence,
            width=width,
            distance=distance
        )
        return peaks, properties
    
    @staticmethod
    def calculate_baseline_noise(intensity, window_size=50):
        """Estimasi noise di baseline"""
        baseline = uniform_filter1d(intensity, size=window_size)
        noise = intensity - baseline
        return np.std(noise)
    
    @staticmethod
    def evaluate_quality(intensity):
        """Comprehensive quality evaluation"""
        snr = XRD1DQualityEvaluator.calculate_snr(intensity)
        peaks, props = XRD1DQualityEvaluator.detect_peaks(intensity)
        baseline_noise = XRD1DQualityEvaluator.calculate_baseline_noise(intensity)
        
        num_peaks = len(peaks)
        
        # Quality score (higher is better)
        quality_score = (num_peaks * snr) / (baseline_noise + 1)
        
        return {
            'snr': float(snr),
            'num_peaks': int(num_peaks),
            'baseline_noise': float(baseline_noise),
            'quality_score': float(quality_score)
        }


# ======================================================
# MAIN 1D DENOISER
# ======================================================

class XRD1DDenoiser:
    """
    Main pipeline untuk XRD 1D denoising
    """
    
    def __init__(self, model_type='unet', model_path=None, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model_type: 'unet' atau 'simple_cnn'
            model_path: path ke pre-trained model (optional)
            device: 'cuda' atau 'cpu'
        """
        self.device = device
        self.preprocessor = XRD1DPreprocessor()
        self.evaluator = XRD1DQualityEvaluator()
        
        print(f"🖥️  Device: {device}")
        print(f"📦 Model type: {model_type}")
        
        # Initialize model
        if model_type == 'unet':
            self.model = UNet1D(base_channels=32).to(device)
        elif model_type == 'simple_cnn':
            self.model = SimpleCNN1D(channels=64).to(device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Load pre-trained jika ada
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️  No pre-trained model loaded. Model uses random weights.")
            print("   For best results, train the model with your data first.")
        
        # Test forward pass
        self._test_forward()
    
    def _test_forward(self):
        """Test forward pass"""
        try:
            dummy_input = torch.randn(1, 1, 1000).to(self.device)
            self.model.eval()
            with torch.no_grad():
                _ = self.model(dummy_input)
            print("✅ Model forward pass test: OK")
        except Exception as e:
            print(f"⚠️  Model forward pass test failed: {e}")
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
    
    def save_model(self, save_path):
        """Save model"""
        checkpoint = {
            'model': self.model.state_dict(),
            'date': datetime.now().isoformat()
        }
        torch.save(checkpoint, save_path)
        print(f"💾 Model saved to {save_path}")
    
    def denoise(self, two_theta, intensity):
        """
        Denoise XRD 1D profile
        
        Args:
            two_theta: array 1D, nilai 2theta
            intensity: array 1D, intensity values
        
        Returns:
            denoised_intensity: array 1D yang sudah dibersihkan
        """
        # Convert to numpy
        two_theta = np.array(two_theta).flatten()
        intensity = np.array(intensity).flatten()
        
        original_length = len(intensity)
        
        # Preprocessing
        intensity_prep = self.preprocessor.preprocess(intensity, fit=True)
        
        # Convert to torch tensor
        x = torch.from_numpy(intensity_prep).float()
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, Length]
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x)
            y_pred = y_pred.squeeze().cpu().numpy()
        
        # Ensure same length
        if len(y_pred) != original_length:
            y_pred = np.interp(
                np.linspace(0, 1, original_length),
                np.linspace(0, 1, len(y_pred)),
                y_pred
            )
        
        # Postprocessing
        denoised_intensity = self.preprocessor.postprocess(y_pred)
        
        return denoised_intensity
    
    def evaluate(self, two_theta, noisy_intensity, clean_intensity=None):
        """
        Evaluate quality before and after denoising
        
        Returns:
            results: dict dengan metrics
            denoised: denoised intensity
        """
        # Quality before
        quality_before = self.evaluator.evaluate_quality(noisy_intensity)
        
        # Denoise
        denoised = self.denoise(two_theta, noisy_intensity)
        
        # Quality after
        quality_after = self.evaluator.evaluate_quality(denoised)
        
        results = {
            'before': quality_before,
            'after': quality_after,
            'improvement': {
                'snr': quality_after['snr'] - quality_before['snr'],
                'quality_score': quality_after['quality_score'] - quality_before['quality_score']
            }
        }
        
        # Jika ground truth ada
        if clean_intensity is not None:
            mae_before = np.mean(np.abs(noisy_intensity - clean_intensity))
            mae_after = np.mean(np.abs(denoised - clean_intensity))
            
            results['mae'] = {
                'before': float(mae_before),
                'after': float(mae_after),
                'improvement': float(mae_before - mae_after)
            }
        
        return results, denoised


# ======================================================
# VISUALIZATION
# ======================================================

def visualize_1d_denoising(two_theta, noisy, denoised, clean=None, 
                           peaks_before=None, peaks_after=None, save_path=None):
    """
    Visualisasi hasil denoising untuk data 1D
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Comparison (Linear scale)
    ax1 = axes[0]
    ax1.plot(two_theta, noisy, 'gray', alpha=0.5, linewidth=1, label='Noisy Input')
    ax1.plot(two_theta, denoised, 'b-', linewidth=2, label='Denoised', zorder=3)
    if clean is not None:
        ax1.plot(two_theta, clean, 'r--', linewidth=1.5, label='Ground Truth', alpha=0.7)
    
    # Mark peaks
    if peaks_after is not None:
        ax1.plot(two_theta[peaks_after], denoised[peaks_after], 'go', 
                markersize=8, label=f'Detected Peaks ({len(peaks_after)})')
    
    ax1.set_xlabel('2θ (degrees)', fontsize=12)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax1.set_title('XRD Pattern Denoising Result', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale (untuk lihat detail di low intensity)
    ax2 = axes[1]
    ax2.semilogy(two_theta, np.maximum(noisy, 1), 'gray', alpha=0.5, linewidth=1, label='Noisy Input')
    ax2.semilogy(two_theta, np.maximum(denoised, 1), 'b-', linewidth=2, label='Denoised')
    if clean is not None:
        ax2.semilogy(two_theta, np.maximum(clean, 1), 'r--', linewidth=1.5, label='Ground Truth', alpha=0.7)
    
    ax2.set_xlabel('2θ (degrees)', fontsize=12)
    ax2.set_ylabel('Intensity (log scale)', fontsize=12)
    ax2.set_title('XRD Pattern (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Residual / Difference
    ax3 = axes[2]
    residual = noisy - denoised
    ax3.plot(two_theta, residual, 'purple', linewidth=1, label='Removed Noise')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax3.fill_between(two_theta, 0, residual, alpha=0.3, color='purple')
    
    ax3.set_xlabel('2θ (degrees)', fontsize=12)
    ax3.set_ylabel('Residual Intensity', fontsize=12)
    ax3.set_title('Removed Noise (Noisy - Denoised)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    
    plt.show()


# ======================================================
# BATCH PROCESSING
# ======================================================

def batch_denoise_folder(folder_path, denoiser, output_folder='denoised_results'):
    """
    Proses banyak file XRD sekaligus
    
    Args:
        folder_path: folder berisi file XRD
        denoiser: XRD1DDenoiser object
        output_folder: folder untuk simpan hasil
    """
    import glob
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all txt/csv files
    patterns = ['*.txt', '*.dat', '*.csv', '*.xy']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder_path, pattern)))
    
    print(f"\n🔍 Found {len(files)} files to process")
    
    results_summary = []
    
    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{i}/{len(files)}] Processing: {filename}")
        
        try:
            # Load data
            data = np.loadtxt(file_path)
            
            if data.shape[1] >= 2:
                two_theta = data[:, 0]
                intensity = data[:, 1]
            else:
                two_theta = np.arange(len(data))
                intensity = data
            
            # Denoise
            denoised = denoiser.denoise(two_theta, intensity)
            
            # Evaluate
            results, _ = denoiser.evaluate(two_theta, intensity)
            
            # Save
            output_path = os.path.join(output_folder, f"denoised_{filename}")
            output_data = np.column_stack([two_theta, denoised])
            np.savetxt(output_path, output_data, fmt='%.6f', 
                      header='2theta intensity_denoised')
            
            # Save plot
            plot_path = os.path.join(output_folder, f"plot_{filename.replace('.txt', '.png')}")
            visualize_1d_denoising(two_theta, intensity, denoised, save_path=plot_path)
            plt.close()
            
            results_summary.append({
                'file': filename,
                'improvement': results['improvement']['quality_score'],
                'status': 'success'
            })
            
            print(f"  ✅ Quality improvement: +{results['improvement']['quality_score']:.2f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results_summary.append({
                'file': filename,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save summary
    summary_path = os.path.join(output_folder, 'batch_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ Batch processing complete!")
    print(f"📊 Results saved to: {output_folder}")
    print(f"📄 Summary saved to: {summary_path}")


# ======================================================
# EXAMPLE USAGE
# ======================================================

if __name__ == "__main__":
    print("="*70)
    print("XRD 1D PROFILE DEEP LEARNING DENOISER")
    print("="*70)
    
    # Contoh: Generate synthetic XRD 1D data
    print("\n📊 Generating synthetic XRD 1D data...")
    
    # Buat data sintetik
    two_theta = np.linspace(10, 80, 2000)
    
    # Simulasi beberapa peaks (Gaussian)
    clean_intensity = np.zeros_like(two_theta)
    peak_positions = [20, 30, 45, 60]
    peak_heights = [1000, 500, 800, 300]
    peak_widths = [0.5, 0.8, 0.6, 1.0]
    
    for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
        clean_intensity += height * np.exp(-((two_theta - pos) / width) ** 2)
    
    # Tambah background
    clean_intensity += 50
    
    # Tambah Poisson noise
    noisy_intensity = np.random.poisson(clean_intensity * 0.1).astype(float)
    
    print(f"  ✅ Data points: {len(two_theta)}")
    print(f"  ✅ 2θ range: {two_theta.min():.1f}° - {two_theta.max():.1f}°")
    print(f"  ✅ Intensity range: {noisy_intensity.min():.0f} - {noisy_intensity.max():.0f}")
    
    # Initialize denoiser
    print("\n🤖 Initializing 1D Denoiser...")
    denoiser = XRD1DDenoiser(
        model_type='simple_cnn',  # atau 'unet'
        device='cpu'
    )
    
    # Evaluate and denoise
    print("\n🧹 Denoising...")
    results, denoised = denoiser.evaluate(two_theta, noisy_intensity, clean_intensity)
    
    print("\n📈 Quality Metrics:")
    print(f"  Before denoising:")
    print(f"    - SNR: {results['before']['snr']:.2f}")
    print(f"    - Peaks detected: {results['before']['num_peaks']}")
    print(f"    - Quality score: {results['before']['quality_score']:.2f}")
    print(f"  After denoising:")
    print(f"    - SNR: {results['after']['snr']:.2f}")
    print(f"    - Peaks detected: {results['after']['num_peaks']}")
    print(f"    - Quality score: {results['after']['quality_score']:.2f}")
    print(f"  Improvement:")
    print(f"    - SNR: +{results['improvement']['snr']:.2f}")
    print(f"    - Quality score: +{results['improvement']['quality_score']:.2f}")
    
    if 'mae' in results:
        print(f"    - MAE improvement: {results['mae']['improvement']:.2f}")
    
    # Detect peaks
    peaks_before, _ = denoiser.evaluator.detect_peaks(noisy_intensity)
    peaks_after, _ = denoiser.evaluator.detect_peaks(denoised)
    
    # Visualize
    print("\n📊 Creating visualization...")
    visualize_1d_denoising(
        two_theta, 
        noisy_intensity, 
        denoised,
        clean=clean_intensity,
        peaks_before=peaks_before,
        peaks_after=peaks_after,
        save_path='xrd_1d_denoising_result_vClaude.png'
    )
    
    print("\n" + "="*70)
    print("✅ DONE! Check 'xrd_1d_denoising_result_vClaude.png'")
    print("="*70)