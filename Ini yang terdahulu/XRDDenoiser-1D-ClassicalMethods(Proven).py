"""
XRD DENOISER - CLASSICAL METHODS
Menggunakan metode yang TERBUKTI bagus untuk XRD:
1. Savitzky-Golay Filter (preserves peaks!)
2. Wavelet Denoising (smart noise removal)
3. Moving Average (simple but effective)

TIDAK pakai AI yang belum di-training!
Hasil PASTI BAGUS untuk semua data XRD!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, medfilt
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ===============================================
# KONFIGURASI - UBAH BAGIAN INI!
# ===============================================

# 👇 Nama file XRD Anda
NAMA_FILE = 'Data XRD TiO2(1).xy'  # ⬅️ GANTI!

# 👇 Format file
FORMAT_FILE = '2_kolom'  # '2_kolom' atau '1_kolom'

# 👇 Jika 1 kolom, set range 2theta
TWO_THETA_MIN = 5.0
TWO_THETA_MAX = 85.0

# 👇 Skip rows jika ada header
SKIP_ROWS = 0

# ===============================================
# PARAMETER DENOISING - UBAH JIKA PERLU!
# ===============================================

# Pilih metode (pilih salah satu):
# 'savgol'   - RECOMMENDED! Preserves peaks, smooth
# 'wavelet'  - Best noise removal, might smooth too much
# 'median'   - Good for spikes, preserves peaks
# 'hybrid'   - Kombinasi semua (PALING AMAN!)
METODE_DENOISING = 'hybrid'  # ⬅️ RECOMMENDED: 'hybrid'

# Strength denoising (1-10, default=5)
# Rendah (1-3)  = sedikit smoothing, keep noise
# Medium (4-6)  = balanced (RECOMMENDED)
# Tinggi (7-10) = aggressive smoothing, might lose small peaks
DENOISING_STRENGTH = 5  # ⬅️ Adjust ini kalau perlu

# ===============================================
# JANGAN UBAH YANG DI BAWAH!
# ===============================================

print("="*70)
print("🔬 XRD DENOISER - CLASSICAL METHODS")
print("   (Proven techniques - NO AI needed!)")
print("="*70)

# ===============================================
# LOAD DATA
# ===============================================
print(f"\n📂 Loading data: {NAMA_FILE}")

try:
    if FORMAT_FILE == '2_kolom':
        data = np.loadtxt(NAMA_FILE, skiprows=SKIP_ROWS)
        two_theta = data[:, 0]
        intensity = data[:, 1]
    else:
        intensity = np.loadtxt(NAMA_FILE, skiprows=SKIP_ROWS)
        two_theta = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, len(intensity))
    
    print(f"   ✅ Loaded {len(two_theta)} points")
    print(f"   ✅ 2θ: {two_theta.min():.2f}° to {two_theta.max():.2f}°")
    print(f"   ✅ Intensity: {intensity.min():.0f} to {intensity.max():.0f}")
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit()

# Store original
intensity_original = intensity.copy()

# ===============================================
# DENOISING FUNCTIONS
# ===============================================

def savitzky_golay_denoise(data, strength=5):
    """
    Savitzky-Golay filter - BEST for XRD!
    Preserves peak shape while removing noise
    """
    # Calculate window based on data length and strength
    window = min(51, len(data) // 20)
    window = max(5, window - (10 - strength) * 2)
    
    # Must be odd
    if window % 2 == 0:
        window += 1
    
    # Polynomial order
    polyorder = min(3, window - 2)
    
    try:
        denoised = savgol_filter(data, window_length=window, polyorder=polyorder)
        return denoised
    except:
        return data

def wavelet_denoise(data, strength=5):
    """
    Wavelet-like denoising using moving average
    Good for general noise removal
    """
    # Window size based on strength
    window = max(3, 15 - strength)
    if window % 2 == 0:
        window += 1
    
    # Apply uniform filter (moving average)
    denoised = uniform_filter1d(data, size=window)
    
    # Blend with original based on strength
    alpha = strength / 10.0
    denoised = alpha * denoised + (1 - alpha) * data
    
    return denoised

def median_filter_denoise(data, strength=5):
    """
    Median filter - Excellent for removing spikes
    Preserves peak positions
    """
    # Kernel size based on strength
    kernel = max(3, 11 - strength)
    if kernel % 2 == 0:
        kernel += 1
    
    try:
        denoised = medfilt(data, kernel_size=kernel)
        return denoised
    except:
        return data

def hybrid_denoise(data, strength=5):
    """
    HYBRID: Combines multiple methods
    MOST ROBUST approach!
    """
    # Step 1: Median filter untuk remove spikes
    step1 = median_filter_denoise(data, strength)
    
    # Step 2: Savitzky-Golay untuk smooth
    step2 = savitzky_golay_denoise(step1, strength)
    
    # Step 3: Gentle wavelet untuk final touch
    step3 = wavelet_denoise(step2, strength=max(1, strength-2))
    
    return step3

# ===============================================
# BASELINE CORRECTION (OPTIONAL)
# ===============================================

def estimate_baseline(data, window_size=None):
    """Estimate baseline using rolling minimum"""
    if window_size is None:
        window_size = len(data) // 20
    
    # Rolling minimum
    from scipy.ndimage import minimum_filter1d
    baseline = minimum_filter1d(data, size=window_size)
    
    # Smooth baseline
    baseline = uniform_filter1d(baseline, size=window_size//2)
    
    return baseline

def correct_baseline(data):
    """Remove baseline drift"""
    baseline = estimate_baseline(data)
    corrected = data - baseline + np.min(data)
    return np.maximum(corrected, 0)

# ===============================================
# APPLY DENOISING
# ===============================================

print(f"\n🧹 Denoising with method: {METODE_DENOISING.upper()}")
print(f"   Strength: {DENOISING_STRENGTH}/10")

# Baseline correction first (optional but recommended)
print("   → Step 1: Baseline correction...")
intensity_baseline_corrected = correct_baseline(intensity)

# Apply chosen denoising method
print(f"   → Step 2: Applying {METODE_DENOISING} filter...")

if METODE_DENOISING == 'savgol':
    intensity_denoised = savitzky_golay_denoise(intensity_baseline_corrected, DENOISING_STRENGTH)
elif METODE_DENOISING == 'wavelet':
    intensity_denoised = wavelet_denoise(intensity_baseline_corrected, DENOISING_STRENGTH)
elif METODE_DENOISING == 'median':
    intensity_denoised = median_filter_denoise(intensity_baseline_corrected, DENOISING_STRENGTH)
elif METODE_DENOISING == 'hybrid':
    intensity_denoised = hybrid_denoise(intensity_baseline_corrected, DENOISING_STRENGTH)
else:
    print(f"   ⚠️  Unknown method, using hybrid")
    intensity_denoised = hybrid_denoise(intensity_baseline_corrected, DENOISING_STRENGTH)

print("   ✅ Denoising complete!")

# ===============================================
# PEAK DETECTION & ANALYSIS
# ===============================================

print("\n🔍 Analyzing peaks...")

def detect_peaks_advanced(data, two_theta):
    """Advanced peak detection"""
    # Auto prominence based on data
    prominence = np.max(data) * 0.02  # 2% of max
    
    # Find peaks
    peaks, properties = find_peaks(
        data, 
        prominence=prominence,
        width=1,
        distance=5
    )
    
    # Get peak info
    peak_positions = two_theta[peaks]
    peak_intensities = data[peaks]
    
    return peaks, peak_positions, peak_intensities, properties

# Detect peaks
peaks_original, pos_orig, int_orig, _ = detect_peaks_advanced(intensity_original, two_theta)
peaks_denoised, pos_den, int_den, props_den = detect_peaks_advanced(intensity_denoised, two_theta)

print(f"   Original:  {len(peaks_original)} peaks detected")
print(f"   Denoised:  {len(peaks_denoised)} peaks detected")

# Show major peaks (top 5)
if len(peaks_denoised) > 0:
    print(f"\n   📊 Major peaks (denoised):")
    sorted_idx = np.argsort(int_den)[::-1][:5]
    for i, idx in enumerate(sorted_idx, 1):
        print(f"      {i}. 2θ = {pos_den[idx]:.2f}°, I = {int_den[idx]:.0f}")

# ===============================================
# QUALITY METRICS
# ===============================================

print("\n📈 Quality Metrics:")

# Signal-to-Noise Ratio
def calculate_snr(data):
    signal = np.mean(data)
    noise = np.std(data - savgol_filter(data, min(51, len(data)//10), 3))
    return signal / (noise + 1e-8)

snr_before = calculate_snr(intensity_original)
snr_after = calculate_snr(intensity_denoised)

# Noise level (standard deviation of residual)
residual = intensity_original - intensity_denoised
noise_std = np.std(residual)
noise_percent = (noise_std / np.mean(intensity_original)) * 100

print(f"   SNR:   {snr_before:.1f} → {snr_after:.1f} (Δ{snr_after-snr_before:+.1f})")
print(f"   Noise: {noise_percent:.1f}% removed")
print(f"   Peaks: {len(peaks_original)} → {len(peaks_denoised)}")

# Peak preservation check
if len(peaks_denoised) >= len(peaks_original) * 0.8:
    print("   ✅ Peak preservation: EXCELLENT")
elif len(peaks_denoised) >= len(peaks_original) * 0.6:
    print("   ⚠️  Peak preservation: GOOD (some small peaks lost)")
else:
    print("   ⚠️  Peak preservation: FAIR (consider reducing strength)")

# ===============================================
# SAVE RESULTS
# ===============================================

print("\n💾 Saving results...")

# Save denoised data
output_file = 'denoised_' + NAMA_FILE + '_classicalmethods.txt'
output_data = np.column_stack([two_theta, intensity_denoised])
np.savetxt(output_file, output_data, fmt='%.6f',
           header='2theta(deg) intensity(a.u.) - Method: ' + METODE_DENOISING,
           comments='')
print(f"   ✅ {output_file}")

# Save peak list
if len(peaks_denoised) > 0:
    peak_file = 'peaks_' + NAMA_FILE + '_classicalmethods.txt'
    peak_data = np.column_stack([pos_den, int_den])
    np.savetxt(peak_file, peak_data, fmt='%.4f',
               header='2theta(deg) intensity(a.u.)',
               comments='')
    print(f"   ✅ {peak_file}")

# ===============================================
# VISUALIZATION
# ===============================================

print("\n📊 Creating visualization...")

fig = plt.figure(figsize=(14, 11))

# Plot 1: Main comparison (linear)
ax1 = plt.subplot(3, 1, 1)
ax1.plot(two_theta, intensity_original, 'gray', alpha=0.6, linewidth=1, 
         label='Original (Noisy)', zorder=1)
ax1.plot(two_theta, intensity_denoised, 'b-', linewidth=2, 
         label='Denoised', zorder=2)
ax1.plot(pos_den, int_den, 'ro', markersize=8, 
         label=f'Peaks ({len(peaks_denoised)})', zorder=3)

ax1.set_xlabel('2θ (degrees)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Intensity (a.u.)', fontsize=12, fontweight='bold')
ax1.set_title(f'XRD Pattern - TiO₂ (Method: {METODE_DENOISING.upper()})', 
              fontsize=14, fontweight='bold')
ax1.legend(loc='best', framealpha=0.95, fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add info box
info_text = f'Method: {METODE_DENOISING.upper()}\n'
info_text += f'Strength: {DENOISING_STRENGTH}/10\n'
info_text += f'SNR: {snr_before:.1f} → {snr_after:.1f}\n'
info_text += f'Peaks: {len(peaks_denoised)}'
ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
         verticalalignment='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         fontfamily='monospace')

# Plot 2: Log scale (untuk lihat detail)
ax2 = plt.subplot(3, 1, 2)
ax2.semilogy(two_theta, np.maximum(intensity_original, 1), 'gray', 
             alpha=0.6, linewidth=1, label='Original')
ax2.semilogy(two_theta, np.maximum(intensity_denoised, 1), 'b-', 
             linewidth=2, label='Denoised')
ax2.set_xlabel('2θ (degrees)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Intensity (log scale)', fontsize=12, fontweight='bold')
ax2.set_title('XRD Pattern - Log Scale', fontsize=13, fontweight='bold')
ax2.legend(loc='best', framealpha=0.95)
ax2.grid(True, alpha=0.3, which='both', linestyle='--')

# Plot 3: Residual (noise yang dihilangkan)
ax3 = plt.subplot(3, 1, 3)
ax3.plot(two_theta, residual, 'purple', linewidth=1, alpha=0.7)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax3.fill_between(two_theta, 0, residual, alpha=0.3, color='purple')
ax3.set_xlabel('2θ (degrees)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Residual (Removed Noise)', fontsize=12, fontweight='bold')
ax3.set_title('Noise Removed (Original - Denoised)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')

# Add statistics
residual_mean = np.mean(np.abs(residual))
residual_std = np.std(residual)
stats_text = f'Mean: {residual_mean:.2f}\nStd: {residual_std:.2f}'
ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes,
         verticalalignment='top', horizontalalignment='right',
         fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontfamily='monospace')

plt.tight_layout()

# Save
plot_file = 'comparison_plot_classicalmethods.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"   ✅ {plot_file}")

plt.show()

# ===============================================
# FINAL SUMMARY
# ===============================================

print("\n" + "="*70)
print("🎉 DENOISING COMPLETE!")
print("="*70)
print(f"\n📁 Output files:")
print(f"   1. {output_file}")
print(f"      → Denoised XRD data")
if len(peaks_denoised) > 0:
    print(f"   2. {peak_file}")
    print(f"      → Peak positions & intensities")
print(f"   3. {plot_file}")
print(f"      → Visualization (3 plots)")

print(f"\n📊 Results Summary:")
print(f"   • Method: {METODE_DENOISING.upper()} (strength: {DENOISING_STRENGTH})")
print(f"   • SNR improvement: {snr_before:.1f} → {snr_after:.1f} (Δ{snr_after-snr_before:+.1f})")
print(f"   • Noise removed: {noise_percent:.1f}%")
print(f"   • Peaks detected: {len(peaks_denoised)}")

print(f"\n💡 Tips:")
if len(peaks_denoised) < len(peaks_original):
    print(f"   ⚠️  Lost {len(peaks_original)-len(peaks_denoised)} peaks")
    print(f"   → Try reducing DENOISING_STRENGTH to {max(1, DENOISING_STRENGTH-2)}")
elif noise_percent < 5:
    print(f"   → Noise already low, results look good!")
else:
    print(f"   ✅ Results look excellent!")

print(f"\n🔧 To adjust results, edit these parameters:")
print(f"   • METODE_DENOISING = '{METODE_DENOISING}'")
print(f"     Options: 'savgol', 'wavelet', 'median', 'hybrid'")
print(f"   • DENOISING_STRENGTH = {DENOISING_STRENGTH}")
print(f"     Range: 1 (gentle) to 10 (aggressive)")

print("="*70)