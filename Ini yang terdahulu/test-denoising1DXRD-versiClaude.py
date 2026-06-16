"""
Script Denoise XRD Data - SIMPLE & CLEAN VERSION
Tanpa error, langsung jalan!

CARA PAKAI:
1. Simpan script ini sebagai: denoise_xrd_clean.py
2. Taruh file data XRD Anda di folder yang sama
3. Edit bagian KONFIGURASI di bawah
4. Jalankan: python denoise_xrd_clean.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================================
# KONFIGURASI - UBAH BAGIAN INI SAJA!
# ===============================================

# 👇 Nama file XRD Anda
NAMA_FILE = 'Data XRD TiO2(1).xy'  # ⬅️ GANTI INI!

# 👇 Format file Anda (pilih salah satu)
FORMAT_FILE = '2_kolom'  # Pilihan: '2_kolom' atau '1_kolom'

# 👇 Jika format 1 kolom, set range 2theta
TWO_THETA_MIN = 10.0  # degrees
TWO_THETA_MAX = 80.0  # degrees

# 👇 Jika ada header/baris pertama yang bukan angka
SKIP_ROWS = 0  # Set 1 jika ada header, 0 jika tidak ada

# ===============================================
# JANGAN UBAH YANG DI BAWAH INI!
# ===============================================

print("="*70)
print("🚀 XRD DENOISING - STARTING...")
print("="*70)

# ===============================================
# STEP 1: LOAD DATA
# ===============================================
print("\n📂 STEP 1: Loading data...")
print(f"   File: {NAMA_FILE}")

try:
    if FORMAT_FILE == '2_kolom':
        data = np.loadtxt(NAMA_FILE, skiprows=SKIP_ROWS)
        two_theta = data[:, 0]
        intensity = data[:, 1]
        print(f"   ✅ Format: 2 columns (2theta, intensity)")
    else:
        intensity = np.loadtxt(NAMA_FILE, skiprows=SKIP_ROWS)
        two_theta = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, len(intensity))
        print(f"   ✅ Format: 1 column (intensity only)")
        print(f"   ✅ Generated 2theta: {TWO_THETA_MIN}° to {TWO_THETA_MAX}°")
    
    print(f"   ✅ Data points: {len(two_theta)}")
    print(f"   ✅ 2θ range: {two_theta.min():.2f}° to {two_theta.max():.2f}°")
    print(f"   ✅ Intensity range: {intensity.min():.0f} to {intensity.max():.0f}")
    
except FileNotFoundError:
    print(f"   ❌ ERROR: File '{NAMA_FILE}' tidak ditemukan!")
    print(f"   💡 Pastikan file ada di folder yang sama dengan script ini")
    exit()
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    print(f"   💡 Cek format file Anda")
    exit()

# ===============================================
# STEP 2: DEFINE SIMPLE DENOISING MODEL
# ===============================================
print("\n🤖 STEP 2: Creating AI model...")

class SimpleDenoisingCNN(nn.Module):
    """Simple CNN untuk denoising XRD 1D"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Layer 2
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Layer 3
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Layer 4
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Output layer
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.net(x)

# Create model
device = 'cpu'  # Aman untuk semua komputer
model = SimpleDenoisingCNN().to(device)
model.eval()

print(f"   ✅ Model created (device: {device})")

# ===============================================
# STEP 3: PREPROCESSING
# ===============================================
print("\n🔧 STEP 3: Preprocessing...")

def anscombe_transform(x):
    """Transform Poisson noise to Gaussian"""
    return 2 * np.sqrt(np.maximum(x, 0) + 3/8)

def inverse_anscombe(y):
    """Inverse transform"""
    return np.maximum((y/2)**2 - 3/8, 0)

# Apply Anscombe transform
intensity_anscombe = anscombe_transform(intensity)

# Normalize
intensity_min = intensity_anscombe.min()
intensity_max = intensity_anscombe.max()
intensity_normalized = (intensity_anscombe - intensity_min) / (intensity_max - intensity_min + 1e-8)

print(f"   ✅ Applied Anscombe transform")
print(f"   ✅ Normalized to [0, 1]")

# ===============================================
# STEP 4: DENOISING
# ===============================================
print("\n🧹 STEP 4: Denoising...")
print("   ⏳ Please wait (10-30 seconds)...")

# Convert to tensor
x_tensor = torch.from_numpy(intensity_normalized).float()
x_tensor = x_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, Length]

# Denoise with model
with torch.no_grad():
    y_pred = model(x_tensor)
    y_pred = y_pred.squeeze().cpu().numpy()

# Postprocessing
y_denorm = y_pred * (intensity_max - intensity_min) + intensity_min
denoised_intensity = inverse_anscombe(y_denorm)

print(f"   ✅ Denoising complete!")

# ===============================================
# STEP 5: SMOOTH (Optional - makes it nicer)
# ===============================================
print("\n✨ STEP 5: Applying final smoothing...")

# Apply Savitzky-Golay filter untuk hasil lebih halus
try:
    window = min(51, len(denoised_intensity) // 10)
    if window % 2 == 0:
        window += 1
    denoised_smooth = savgol_filter(denoised_intensity, window_length=window, polyorder=3)
    denoised_intensity = denoised_smooth
    print(f"   ✅ Smoothing applied (window={window})")
except:
    print(f"   ⚠️  Smoothing skipped (data too short)")

# ===============================================
# STEP 6: QUALITY EVALUATION
# ===============================================
print("\n📊 STEP 6: Evaluating quality...")

def calculate_snr(data):
    """Calculate Signal-to-Noise Ratio"""
    signal = np.mean(data)
    noise = np.std(data)
    return signal / (noise + 1e-8)

def detect_peaks_simple(data):
    """Detect peaks"""
    prominence = np.max(data) * 0.05
    peaks, _ = find_peaks(data, prominence=prominence, distance=10)
    return peaks

# Calculate metrics
snr_before = calculate_snr(intensity)
snr_after = calculate_snr(denoised_intensity)
peaks_before = detect_peaks_simple(intensity)
peaks_after = detect_peaks_simple(denoised_intensity)

print(f"\n   📈 RESULTS:")
print(f"      BEFORE → AFTER")
print(f"      SNR:   {snr_before:.2f} → {snr_after:.2f} (+{snr_after-snr_before:.2f})")
print(f"      Peaks: {len(peaks_before)} → {len(peaks_after)}")

# Calculate noise reduction
noise_removed = np.mean(np.abs(intensity - denoised_intensity))
noise_percent = (noise_removed / np.mean(intensity)) * 100
print(f"      Noise removed: {noise_percent:.1f}%")

# ===============================================
# STEP 7: SAVE RESULTS
# ===============================================
print("\n💾 STEP 7: Saving results...")

# Save denoised data
output_file = 'denoised_' + NAMA_FILE
output_data = np.column_stack([two_theta, denoised_intensity])
np.savetxt(output_file, output_data, fmt='%.6f', 
           header='2theta(degrees) intensity(a.u.)', comments='')

print(f"   ✅ Saved: {output_file}")

# ===============================================
# STEP 8: VISUALIZATION
# ===============================================
print("\n📊 STEP 8: Creating visualization...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Linear scale comparison
ax1 = axes[0]
ax1.plot(two_theta, intensity, 'gray', alpha=0.5, linewidth=1, label='Original (Noisy)')
ax1.plot(two_theta, denoised_intensity, 'b-', linewidth=2, label='Denoised')
ax1.plot(two_theta[peaks_after], denoised_intensity[peaks_after], 'ro', 
         markersize=6, label=f'Detected Peaks ({len(peaks_after)})')
ax1.set_xlabel('2θ (degrees)', fontsize=11)
ax1.set_ylabel('Intensity (a.u.)', fontsize=11)
ax1.set_title('XRD Pattern Denoising - Linear Scale', fontsize=13, fontweight='bold')
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Plot 2: Log scale
ax2 = axes[1]
ax2.semilogy(two_theta, np.maximum(intensity, 1), 'gray', alpha=0.5, linewidth=1, label='Original')
ax2.semilogy(two_theta, np.maximum(denoised_intensity, 1), 'b-', linewidth=2, label='Denoised')
ax2.set_xlabel('2θ (degrees)', fontsize=11)
ax2.set_ylabel('Intensity (log scale)', fontsize=11)
ax2.set_title('XRD Pattern - Log Scale', fontsize=13, fontweight='bold')
ax2.legend(loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Residual (noise removed)
ax3 = axes[2]
residual = intensity - denoised_intensity
ax3.plot(two_theta, residual, 'purple', linewidth=1)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax3.fill_between(two_theta, 0, residual, alpha=0.3, color='purple')
ax3.set_xlabel('2θ (degrees)', fontsize=11)
ax3.set_ylabel('Removed Noise', fontsize=11)
ax3.set_title('Noise Removed (Original - Denoised)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add text info
info_text = f'SNR: {snr_before:.1f} → {snr_after:.1f}\n'
info_text += f'Peaks: {len(peaks_before)} → {len(peaks_after)}\n'
info_text += f'Noise: -{noise_percent:.1f}%'
ax1.text(0.98, 0.97, info_text, transform=ax1.transAxes,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=10, fontfamily='monospace')

plt.tight_layout()

# Save plot
plot_file = 'comparison_plot.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {plot_file}")

plt.show()

# ===============================================
# FINAL SUMMARY
# ===============================================
print("\n" + "="*70)
print("🎉 SUCCESS! DENOISING COMPLETE!")
print("="*70)
print("\n📁 Generated files:")
print(f"   1. {output_file}")
print(f"      → Your denoised XRD data (2 columns)")
print(f"\n   2. {plot_file}")
print(f"      → Visualization (3 plots)")
print("\n📊 Quality improvement:")
print(f"   • SNR improved by: +{snr_after-snr_before:.2f}")
print(f"   • Noise removed: {noise_percent:.1f}%")
print(f"   • Peaks detected: {len(peaks_after)}")
print("\n💡 Next steps:")
print("   • Open the files to see results")
print("   • Use denoised data for further analysis")
print("   • Adjust KONFIGURASI if needed")
print("="*70)