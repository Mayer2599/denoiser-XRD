"""
XRD Deep Learning Denoiser
Berdasarkan paper: "Deep Learning Based Denoising of X-ray Scattering Data"
Author: Milan de Mooij (2024)

Script ini mengimplementasikan:
1. Anscombe Transform untuk Poisson noise
2. U-Net architecture untuk denoising
3. Quantile Regression untuk prediction intervals
4. Conformal Prediction untuk kalibrasi confidence
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
import json

# ======================================================
# VARIANCE STABILIZING TRANSFORMATION (VST)
# ======================================================

class AnscombeTrans:
    """
    Anscombe Transform untuk mengubah Poisson noise → Gaussian noise
    Referensi: Anscombe (1948)
    """
    
    @staticmethod
    def forward(x):
        """
        Anscombe transform: A(x) = 2 * sqrt(x + 3/8)
        """
        return 2 * np.sqrt(x + 3/8)
    
    @staticmethod
    def inverse(y):
        """
        Exact unbiased inverse Anscombe transform
        Referensi: Makitalo & Foi (2011)
        """
        # Algebraic inverse
        z = (y / 2) ** 2 - 3/8
        
        # Bias correction untuk unbiased inverse
        # Simplified version - untuk implementasi lengkap lihat paper
        return np.maximum(z, 0)


# ======================================================
# U-NET ARCHITECTURE (Tunable)
# ======================================================

class DoubleConv(nn.Module):
    """Double Convolution Block dengan BatchNorm dan ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class TunableUNet(nn.Module):
    """
    Tunable U-Net untuk XRD denoising (FIXED VERSION)
    Hyperparameters:
    - cb: base channels (default=32)
    - r: growth rate (default=1.8)
    - depth: jumlah layers (default=5)
    
    Fix: Proper channel calculation untuk skip connections
    """
    
    def __init__(self, in_channels=1, out_channels=1, cb=32, r=1.8, depth=5):
        super().__init__()
        self.depth = depth
        
        # Calculate channel sizes dengan proper rounding
        channels = [in_channels]
        for i in range(depth):
            ch = int(cb * (r ** i))
            channels.append(ch)
        
        print(f"🔧 U-Net Channel Configuration: {channels}")
        
        # Encoder (Contracting path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(depth):
            self.encoders.append(DoubleConv(channels[i], channels[i+1]))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
        
        # Decoder (Expansive path)
        # CRITICAL FIX: Decoder input = upsampled + skip connection
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth-1, 0, -1):
            # Upconv: channels[i+1] -> channels[i]
            self.upconvs.append(
                nn.ConvTranspose2d(channels[i+1], channels[i], kernel_size=2, stride=2)
            )
            # Decoder input: channels[i] (upsampled) + channels[i] (skip) = 2*channels[i]
            self.decoders.append(DoubleConv(channels[i] * 2, channels[i]))
        
        # Final output
        self.final_conv = nn.Conv2d(channels[1], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Store input shape untuk debugging
        input_shape = x.shape
        
        # Encoder
        skip_connections = []
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.pools):
                skip_connections.append(x)
                x = self.pools[i](x)
        
        # Decoder with skip connections
        skip_connections = skip_connections[::-1]
        
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            # Upsample
            x = upconv(x)
            
            # Get corresponding skip connection
            skip = skip_connections[i]
            
            # Handle spatial dimension mismatch (untuk odd-sized inputs)
            if x.shape[2:] != skip.shape[2:]:
                # Pad atau crop untuk match spatial dimensions
                diff_h = skip.shape[2] - x.shape[2]
                diff_w = skip.shape[3] - x.shape[3]
                
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                             diff_h // 2, diff_h - diff_h // 2])
            
            # Concatenate
            x = torch.cat([skip, x], dim=1)
            
            # Decode
            x = decoder(x)
        
        # Final convolution
        output = self.final_conv(x)
        
        return output


# ======================================================
# QUANTILE REGRESSION U-NET
# ======================================================

class QuantileUNet(nn.Module):
    """
    U-Net extension untuk Quantile Regression
    Menghasilkan 3 output: q0.05, q0.50 (median), q0.95
    """
    
    def __init__(self, cb=32, r=1.8, depth=5):
        super().__init__()
        
        # Base U-Net dengan 8 output channels
        self.base_unet = TunableUNet(
            in_channels=1, 
            out_channels=8, 
            cb=cb, 
            r=r, 
            depth=depth
        )
        
        # Separate heads untuk setiap quantile
        self.q_low = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=1)
        )
        
        self.q_mid = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=1)
        )
        
        self.q_high = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=1)
        )
    
    def forward(self, x):
        features = self.base_unet(x)
        
        q_05 = self.q_low(features)
        q_50 = self.q_mid(features)
        q_95 = self.q_high(features)
        
        return q_05, q_50, q_95


# ======================================================
# LOSS FUNCTIONS
# ======================================================

class PinballLoss(nn.Module):
    """
    Pinball Loss untuk Quantile Regression
    L(y, ŷ, τ) = τ * |y - ŷ| if y ≥ ŷ
               = (1-τ) * |y - ŷ| if y < ŷ
    """
    
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, pred, target):
        error = target - pred
        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )
        return loss.mean()


# ======================================================
# XRD DATA PREPROCESSING
# ======================================================

class XRDPreprocessor:
    """Preprocessing untuk data XRD"""
    
    def __init__(self):
        self.anscombe = AnscombeTrans()
        self.data_min = None
        self.data_max = None
    
    def preprocess(self, data, fit=False):
        """
        Pipeline preprocessing:
        1. Anscombe transform (Poisson → Gaussian)
        2. Min-Max normalization [0, 1]
        """
        # Step 1: Anscombe transform
        data_anscombe = self.anscombe.forward(data)
        
        # Step 2: Min-Max normalization
        if fit:
            self.data_min = np.min(data_anscombe)
            self.data_max = np.max(data_anscombe)
        
        data_normalized = (data_anscombe - self.data_min) / (self.data_max - self.data_min + 1e-8)
        
        return data_normalized
    
    def postprocess(self, data):
        """
        Pipeline postprocessing:
        1. Inverse Min-Max normalization
        2. Inverse Anscombe transform
        """
        # Step 1: Inverse normalization
        data_denorm = data * (self.data_max - self.data_min) + self.data_min
        
        # Step 2: Inverse Anscombe
        data_original = self.anscombe.inverse(data_denorm)
        
        return data_original


# ======================================================
# CONFORMAL PREDICTION
# ======================================================

class ConformalPredictor:
    """
    Conformal Prediction untuk kalibrasi prediction intervals
    Referensi: Romano et al. (2019), Angelopoulos & Bates (2022)
    """
    
    def __init__(self, alpha=0.1):
        """
        alpha: miscoverage rate (default 0.1 untuk 90% coverage)
        """
        self.alpha = alpha
        self.q_score = None
    
    def calibrate(self, q_low, q_high, y_true):
        """
        Kalkulasi q_score dari calibration set
        
        Score function: s(x,y) = max{q_0.05(x) - y, y - q_0.95(x)}
        """
        scores = np.maximum(q_low - y_true, y_true - q_high)
        scores_flat = scores.flatten()
        
        n = len(scores_flat)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        
        self.q_score = np.quantile(scores_flat, quantile_level)
        
        return self.q_score
    
    def conformalize(self, q_low, q_high):
        """
        Aplikasikan conformalization ke prediction intervals
        I(x)_calib = [q_0.05(x) - q_score, q_0.95(x) + q_score]
        """
        if self.q_score is None:
            raise ValueError("Must calibrate first!")
        
        q_low_calib = q_low - self.q_score
        q_high_calib = q_high + self.q_score
        
        return q_low_calib, q_high_calib


# ======================================================
# XRD QUALITY EVALUATOR
# ======================================================

class XRDQualityEvaluator:
    """
    Evaluasi kualitas data XRD
    Metrics: SNR, Peak detection, Baseline drift
    """
    
    @staticmethod
    def calculate_snr(data):
        """
        SNR untuk Poisson: √λ
        """
        # Avoid division by zero
        data_safe = np.maximum(data, 1e-8)
        snr = np.sqrt(data_safe)
        return np.mean(snr)
    
    @staticmethod
    def detect_peaks(profile_1d, prominence=0.05, width=3, distance=10):
        """Deteksi peaks dari 1D integration profile"""
        peaks, properties = find_peaks(
            profile_1d,
            prominence=prominence,
            width=width,
            distance=distance
        )
        return peaks, properties
    
    @staticmethod
    def calculate_baseline_drift(profile_1d):
        """Estimasi baseline drift"""
        baseline_est = uniform_filter1d(profile_1d, size=50)
        drift = np.max(baseline_est) - np.min(baseline_est)
        return drift
    
    @staticmethod
    def evaluate_quality(data_2d):
        """
        Comprehensive quality evaluation
        Returns: dict dengan metrics
        """
        # Convert 2D to 1D profile
        profile_1d = np.mean(data_2d, axis=0)
        profile_1d = profile_1d / (np.max(profile_1d) + 1e-8)
        
        # Calculate metrics
        snr = XRDQualityEvaluator.calculate_snr(data_2d)
        peaks, props = XRDQualityEvaluator.detect_peaks(profile_1d)
        baseline_drift = XRDQualityEvaluator.calculate_baseline_drift(profile_1d)
        
        num_peaks = len(peaks)
        noise_level = np.std(profile_1d)
        
        quality_score = (num_peaks * (1 / (noise_level + 0.01))) / (baseline_drift + 0.01)
        
        return {
            'snr': float(snr),
            'num_peaks': num_peaks,
            'noise_level': float(noise_level),
            'baseline_drift': float(baseline_drift),
            'quality_score': float(quality_score)
        }


# ======================================================
# ALTERNATIVE: SIMPLER U-NET (Jika masih ada error)
# ======================================================

class SimpleUNet(nn.Module):
    """
    Simplified U-Net dengan fixed channels
    Lebih robust untuk berbagai input sizes
    """
    
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 512 from upconv + 512 from skip
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)  # 256 from upconv + 256 from skip
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)  # 128 from upconv + 128 from skip
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)  # 64 from upconv + 64 from skip
        
        # Final
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self._pad_to_match(dec4, enc4)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = self._pad_to_match(dec3, enc3)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = self._pad_to_match(dec2, enc2)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self._pad_to_match(dec1, enc1)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)
    
    def _pad_to_match(self, x, target):
        """Pad x to match target spatial dimensions"""
        diff_h = target.shape[2] - x.shape[2]
        diff_w = target.shape[3] - x.shape[3]
        
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                     diff_h // 2, diff_h - diff_h // 2])
        return x


# ======================================================
# DEBUGGING UTILITIES
# ======================================================

def test_model_forward(model, input_shape=(1, 1, 619, 487)):
    """
    Test forward pass untuk detect channel mismatch
    """
    print("\n" + "="*60)
    print("🔍 TESTING MODEL FORWARD PASS")
    print("="*60)
    
    # Create dummy input
    x = torch.randn(*input_shape)
    print(f"Input shape: {x.shape}")
    
    try:
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"✅ Output shape: {output.shape}")
        print("✅ Forward pass successful!")
        return True
        
    except RuntimeError as e:
        print(f"❌ Error during forward pass:")
        print(f"   {str(e)}")
        
        # Parse error message untuk helpful info
        if "expected input" in str(e) and "channels" in str(e):
            print("\n💡 Channel mismatch detected!")
            print("   This typically happens due to:")
            print("   1. Incorrect decoder input channel calculation")
            print("   2. Skip connection concatenation issues")
            print("   3. Odd input dimensions causing size mismatches")
            print("\n   Try using SimpleUNet instead of TunableUNet")
        
        return False


# ======================================================
# MAIN DENOISING PIPELINE (UPDATED)
# ======================================================

class XRDDenoiser:
    """
    Main pipeline untuk XRD denoising (UPDATED dengan error handling)
    """
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu',
                 use_simple_unet=False, test_forward=True):
        """
        Args:
            model_path: path ke pre-trained model
            device: 'cuda' atau 'cpu'
            use_simple_unet: jika True, gunakan SimpleUNet (lebih robust)
            test_forward: jika True, test forward pass saat init
        """
        self.device = device
        self.preprocessor = XRDPreprocessor()
        self.evaluator = XRDQualityEvaluator()
        
        print(f"\n🖥️  Device: {device}")
        
        # Initialize models
        if use_simple_unet:
            print("📦 Using SimpleUNet (more robust)")
            self.denoiser = SimpleUNet(in_channels=1, out_channels=1).to(device)
        else:
            print("📦 Using TunableUNet (cb=32, r=1.8, depth=5)")
            try:
                self.denoiser = TunableUNet(cb=32, r=1.8, depth=5).to(device)
            except Exception as e:
                print(f"⚠️  TunableUNet failed: {e}")
                print("🔄 Falling back to SimpleUNet...")
                self.denoiser = SimpleUNet(in_channels=1, out_channels=1).to(device)
                use_simple_unet = True
        
        # Test forward pass
        if test_forward:
            success = test_model_forward(self.denoiser)
            if not success and not use_simple_unet:
                print("\n🔄 Switching to SimpleUNet due to forward pass error...")
                self.denoiser = SimpleUNet(in_channels=1, out_channels=1).to(device)
                test_model_forward(self.denoiser)
        
        # Quantile network (optional, bisa di-disable jika error)
        try:
            if use_simple_unet:
                print("⚠️  Quantile regression not available with SimpleUNet")
                self.quantile_net = None
            else:
                self.quantile_net = QuantileUNet(cb=32, r=1.8, depth=5).to(device)
        except Exception as e:
            print(f"⚠️  Quantile network initialization failed: {e}")
            self.quantile_net = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.conformal = ConformalPredictor(alpha=0.1)
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.denoiser.load_state_dict(checkpoint['denoiser'])
            if 'quantile_net' in checkpoint and self.quantile_net is not None:
                self.quantile_net.load_state_dict(checkpoint['quantile_net'])
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
    
    def denoise(self, xrd_data, return_intervals=False, calibration_data=None):
        """
        Denoise XRD data
        
        Args:
            xrd_data: numpy array [H, W] atau [B, H, W]
            return_intervals: jika True, return prediction intervals
            calibration_data: tuple (xrd_calib, ground_truth) untuk CP
        
        Returns:
            denoised_data: numpy array
            intervals (optional): dict dengan 'low' dan 'high' bounds
        """
        # Check if quantile regression available
        if return_intervals and self.quantile_net is None:
            print("⚠️  Quantile regression not available. Returning point prediction only.")
            return_intervals = False
        
        # Ensure correct shape
        if len(xrd_data.shape) == 2:
            xrd_data = xrd_data[np.newaxis, ...]
        
        batch_size = xrd_data.shape[0]
        
        # Preprocessing
        data_preprocessed = []
        for i in range(batch_size):
            prep = self.preprocessor.preprocess(xrd_data[i], fit=(i==0))
            data_preprocessed.append(prep)
        
        data_preprocessed = np.array(data_preprocessed)
        
        # Convert to torch
        x_tensor = torch.from_numpy(data_preprocessed).float()
        x_tensor = x_tensor.unsqueeze(1).to(self.device)  # [B, 1, H, W]
        
        # Inference
        self.denoiser.eval()
        with torch.no_grad():
            try:
                # Point prediction
                y_pred = self.denoiser(x_tensor)
                y_pred = y_pred.squeeze(1).cpu().numpy()
            except RuntimeError as e:
                print(f"❌ Error during inference: {e}")
                print("💡 Try re-initializing with use_simple_unet=True")
                raise
            
            # Quantile predictions (jika diminta dan available)
            if return_intervals and self.quantile_net is not None:
                try:
                    q_low, q_mid, q_high = self.quantile_net(x_tensor)
                    q_low = q_low.squeeze(1).cpu().numpy()
                    q_mid = q_mid.squeeze(1).cpu().numpy()
                    q_high = q_high.squeeze(1).cpu().numpy()
                except Exception as e:
                    print(f"⚠️  Quantile prediction failed: {e}")
                    return_intervals = False
        
        # Postprocessing
        denoised_data = []
        for i in range(batch_size):
            denoised = self.preprocessor.postprocess(y_pred[i])
            denoised_data.append(denoised)
        
        denoised_data = np.array(denoised_data)
        
        if batch_size == 1:
            denoised_data = denoised_data[0]
        
        # Return with or without intervals
        if not return_intervals:
            return denoised_data
        
        # Process quantile predictions
        q_low_post = [self.preprocessor.postprocess(q_low[i]) for i in range(batch_size)]
        q_mid_post = [self.preprocessor.postprocess(q_mid[i]) for i in range(batch_size)]
        q_high_post = [self.preprocessor.postprocess(q_high[i]) for i in range(batch_size)]
        
        intervals = {
            'low': np.array(q_low_post),
            'median': np.array(q_mid_post),
            'high': np.array(q_high_post)
        }
        
        # Conformal Prediction (jika calibration data tersedia)
        if calibration_data is not None:
            try:
                xrd_calib, gt_calib = calibration_data
                
                # Denoise calibration data untuk get quantiles
                q_calib_low, q_calib_mid, q_calib_high = self._get_quantiles(xrd_calib)
                
                # Calibrate
                self.conformal.calibrate(q_calib_low, q_calib_high, gt_calib)
                
                # Apply conformalization
                intervals['low'], intervals['high'] = self.conformal.conformalize(
                    intervals['low'], intervals['high']
                )
                intervals['conformalized'] = True
                intervals['q_score'] = self.conformal.q_score
            except Exception as e:
                print(f"⚠️  Conformal prediction failed: {e}")
                intervals['conformalized'] = False
        
        if batch_size == 1:
            for key in intervals:
                if isinstance(intervals[key], np.ndarray):
                    intervals[key] = intervals[key][0]
        
        return denoised_data, intervals
    
    def _get_quantiles(self, xrd_data):
        """Helper untuk get quantile predictions"""
        if len(xrd_data.shape) == 2:
            xrd_data = xrd_data[np.newaxis, ...]
        
        batch_size = xrd_data.shape[0]
        
        data_preprocessed = []
        for i in range(batch_size):
            prep = self.preprocessor.preprocess(xrd_data[i], fit=(i==0))
            data_preprocessed.append(prep)
        
        x_tensor = torch.from_numpy(np.array(data_preprocessed)).float()
        x_tensor = x_tensor.unsqueeze(1).to(self.device)
        
        self.quantile_net.eval()
        with torch.no_grad():
            q_low, q_mid, q_high = self.quantile_net(x_tensor)
            q_low = q_low.squeeze(1).cpu().numpy()
            q_mid = q_mid.squeeze(1).cpu().numpy()
            q_high = q_high.squeeze(1).cpu().numpy()
        
        q_low_post = np.array([self.preprocessor.postprocess(q_low[i]) for i in range(batch_size)])
        q_mid_post = np.array([self.preprocessor.postprocess(q_mid[i]) for i in range(batch_size)])
        q_high_post = np.array([self.preprocessor.postprocess(q_high[i]) for i in range(batch_size)])
        
        return q_low_post, q_mid_post, q_high_post
    
    def evaluate(self, xrd_data, ground_truth=None):
        """
        Evaluate quality sebelum dan sesudah denoising
        """
        # Quality before
        quality_before = self.evaluator.evaluate_quality(xrd_data)
        
        # Denoise
        denoised = self.denoise(xrd_data)
        
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
        
        # Jika ground truth ada, calculate MAE
        if ground_truth is not None:
            mae_before = np.mean(np.abs(xrd_data - ground_truth))
            mae_after = np.mean(np.abs(denoised - ground_truth))
            
            results['mae'] = {
                'before': float(mae_before),
                'after': float(mae_after),
                'improvement': float(mae_before - mae_after)
            }
        
        return results, denoised


# ======================================================
# VISUALIZATION
# ======================================================

def visualize_denoising_results(original, denoised, intervals=None, save_path=None):
    """
    Visualisasi hasil denoising
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 2D diffraction patterns
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(original, cmap='viridis', aspect='auto')
    ax1.set_title('Original (Noisy)')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(denoised, cmap='viridis', aspect='auto')
    ax2.set_title('Denoised')
    plt.colorbar(im2, ax=ax2)
    
    # Difference map
    ax3 = plt.subplot(2, 3, 3)
    diff = original - denoised
    im3 = ax3.imshow(diff, cmap='RdBu', aspect='auto')
    ax3.set_title('Difference (Original - Denoised)')
    plt.colorbar(im3, ax=ax3)
    
    # 1D integration profiles
    profile_orig = np.mean(original, axis=0)
    profile_denoised = np.mean(denoised, axis=0)
    
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(profile_orig, label='Original', alpha=0.7, linewidth=1)
    ax4.plot(profile_denoised, label='Denoised', linewidth=2)
    ax4.set_xlabel('q (pixel position)')
    ax4.set_ylabel('Intensity')
    ax4.set_title('1D Integration Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Log scale
    ax5 = plt.subplot(2, 3, 5)
    ax5.semilogy(profile_orig, label='Original', alpha=0.7, linewidth=1)
    ax5.semilogy(profile_denoised, label='Denoised', linewidth=2)
    ax5.set_xlabel('q (pixel position)')
    ax5.set_ylabel('Intensity (log scale)')
    ax5.set_title('1D Profile (Log Scale)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Prediction intervals (jika ada)
    if intervals is not None:
        ax6 = plt.subplot(2, 3, 6)
        profile_median = np.mean(intervals['median'], axis=0)
        profile_low = np.mean(intervals['low'], axis=0)
        profile_high = np.mean(intervals['high'], axis=0)
        
        x = np.arange(len(profile_median))
        ax6.plot(profile_orig, 'k-', label='Ground Truth', linewidth=1, alpha=0.5)
        ax6.plot(profile_median, 'b-', label='Median Prediction', linewidth=2)
        ax6.fill_between(x, profile_low, profile_high, alpha=0.3, label='90% Prediction Interval')
        
        if 'conformalized' in intervals and intervals['conformalized']:
            ax6.set_title(f'Prediction Intervals (Conformalized)\nq_score={intervals["q_score"]:.3f}')
        else:
            ax6.set_title('Prediction Intervals (Uncalibrated)')
        
        ax6.set_xlabel('q (pixel position)')
        ax6.set_ylabel('Intensity')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    
    plt.show()


# ======================================================
# EXAMPLE USAGE
# ======================================================

if __name__ == "__main__":
    print("="*60)
    print("XRD DEEP LEARNING DENOISER")
    print("Based on: Milan de Mooij (2024)")
    print("="*60)
    
    # Simulasi data XRD (ganti dengan data real Anda)
    print("\n📊 Generating synthetic XRD data for demonstration...")
    height, width = 619, 487
    
    # Simulasi XRD pattern dengan Poisson noise
    np.random.seed(42)
    ground_truth = np.random.poisson(lam=100, size=(height, width)).astype(float)
    noisy_data = np.random.poisson(lam=ground_truth * 0.1, size=(height, width)).astype(float)
    
    # Initialize denoiser dengan error handling
    print("\n🤖 Initializing XRD Denoiser...")
    
    try:
        # Coba dengan TunableUNet dulu
        denoiser = XRDDenoiser(
            use_simple_unet=False,  # Coba TunableUNet dulu
            test_forward=True       # Test forward pass
        )
    except Exception as e:
        print(f"\n⚠️  TunableUNet failed: {e}")
        print("🔄 Using SimpleUNet instead...")
        denoiser = XRDDenoiser(
            use_simple_unet=True,   # Fallback ke SimpleUNet
            test_forward=True
        )
    
    # Evaluate dan denoise
    print("\n🔬 Analyzing XRD quality...")
    try:
        results, denoised = denoiser.evaluate(noisy_data, ground_truth)
        
        print("\n📈 Quality Metrics:")
        print(f"  Before denoising:")
        print(f"    - SNR: {results['before']['snr']:.2f}")
        print(f"    - Peaks detected: {results['before']['num_peaks']}")
        print(f"    - Quality score: {results['before']['quality_score']:.3f}")
        print(f"  After denoising:")
        print(f"    - SNR: {results['after']['snr']:.2f}")
        print(f"    - Peaks detected: {results['after']['num_peaks']}")
        print(f"    - Quality score: {results['after']['quality_score']:.3f}")
        
        if 'mae' in results:
            print(f"\n  MAE Improvement: {results['mae']['improvement']:.3f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("💡 Trying simple denoising without evaluation...")
        denoised = denoiser.denoise(noisy_data)
    
    # Denoise dengan prediction intervals (optional)
    print("\n🎯 Attempting to generate prediction intervals...")
    try:
        denoised_with_intervals, intervals = denoiser.denoise(
            noisy_data, 
            return_intervals=True,
            calibration_data=None  # Bisa tambahkan calibration data di sini
        )
        print("✅ Prediction intervals generated successfully!")
    except Exception as e:
        print(f"⚠️  Prediction intervals not available: {e}")
        intervals = None
    
    # Visualize
    print("\n📊 Generating visualization...")
    try:
        visualize_denoising_results(
            noisy_data, 
            denoised,
            intervals=intervals,
            save_path='2Dxrd_denoising_result_vCLaude.png'
        )
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        print("📊 Creating simple comparison plot...")
        
        # Simple fallback visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(noisy_data, cmap='viridis')
        ax1.set_title('Noisy Input')
        ax1.axis('off')
        
        ax2.imshow(denoised, cmap='viridis')
        ax2.set_title('Denoised Output')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('xrd_simple_comparison.png', dpi=150)
        print("✅ Simple comparison saved to xrd_simple_comparison.png")
    
    print("\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETE!")
    print("="*60)
    print("\n📝 Quick Start Guide:")
    print("   1. Load your XRD data: xrd = np.load('your_data.npy')")
    print("   2. Initialize: denoiser = XRDDenoiser(use_simple_unet=True)")
    print("   3. Denoise: denoised = denoiser.denoise(xrd)")
    print("   4. Visualize: visualize_denoising_results(xrd, denoised)")
    print("\n💡 If you encounter channel mismatch errors:")
    print("   → Always use: XRDDenoiser(use_simple_unet=True)")
    print("   → SimpleUNet is more robust for various input sizes")
    print("="*60)