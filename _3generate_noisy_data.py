"""
3_generate_noisy_data.py
Generate Noisy XRD Data dari Clean Data

Fungsi:
- Load clean preprocessed data (format: clean_XXXXXX.npy)
- Generate noisy versions dengan realistic noise
- Simpan sebagai noisy_XXXXXX.npy (ID sama dengan clean)
- Variasi noise levels (low, medium, high)
- Variasi noise types (Poisson, Gaussian, background)
- Save noisy-clean pairs untuk training

Author: XRD AI Project
Date: 2026-01-29
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================
DATA_DIR = r"C:\Users\COMPUTER\Documents\xrdAI_withoutmatch3_v2\data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "processed", "clean")
NOISY_OUTPUT_DIR = os.path.join(DATA_DIR, "processed", "noisy")

# Noise generation parameters
NOISE_LEVELS = {
    'low': {
        'weight': 0.30,  # 30% of data
        'poisson_lambda': 1.0,
        'gaussian_sigma': 0.02,
        'background_intensity': (0, 50)
    },
    'medium': {
        'weight': 0.40,  # 40% of data
        'poisson_lambda': 0.5,
        'gaussian_sigma': 0.05,
        'background_intensity': (0, 100)
    },
    'high': {
        'weight': 0.30,  # 30% of data
        'poisson_lambda': 0.2,
        'gaussian_sigma': 0.1,
        'background_intensity': (0, 200)
    }
}

# Create output directory
os.makedirs(NOISY_OUTPUT_DIR, exist_ok=True)


# ========================================
# NOISE GENERATION FUNCTIONS
# ========================================
def add_poisson_noise(intensity, lambda_scale=1.0):
    scaled = intensity * lambda_scale
    noisy = np.random.poisson(np.maximum(scaled, 0))
    noisy = noisy / lambda_scale
    return noisy.astype(float)

def add_gaussian_noise(intensity, sigma=0.05):
    intensity_max = np.max(intensity)
    noise_magnitude = intensity_max * sigma
    noise = np.random.normal(0, noise_magnitude, size=intensity.shape)
    return intensity + noise

def add_background_variation(intensity, background_type='flat', background_intensity=(0, 100)):
    n_points = len(intensity)
    bg_min, bg_max = background_intensity

    if background_type == 'flat':
        background = np.random.uniform(bg_min, bg_max)
        background_array = np.full(n_points, background)
    elif background_type == 'sloping':
        bg_start = np.random.uniform(bg_min, bg_max)
        bg_end = np.random.uniform(bg_min, bg_max)
        background_array = np.linspace(bg_start, bg_end, n_points)
    elif background_type == 'curved':
        bg_start = np.random.uniform(bg_min, bg_max)
        bg_mid = np.random.uniform(bg_min, bg_max)
        bg_end = np.random.uniform(bg_min, bg_max)
        x = np.linspace(0, 1, n_points)
        a = 2 * bg_mid - bg_start - bg_end
        b = 4 * bg_end - 3 * bg_mid + bg_start
        c = bg_start
        background_array = a * x**2 + b * x + c
    else:
        background_array = np.zeros(n_points)

    return intensity + background_array

def add_baseline_drift(intensity, drift_amplitude=0.05):
    n_points = len(intensity)
    intensity_max = np.max(intensity)
    freq = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, 2*np.pi)
    x = np.linspace(0, 2*np.pi*freq, n_points)
    drift = drift_amplitude * intensity_max * np.sin(x + phase)
    return intensity + drift

def random_intensity_scaling(intensity, scale_range=(0.8, 1.2)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return intensity * scale_factor


# ========================================
# COMPLETE NOISE GENERATION PIPELINE
# ========================================
def generate_noisy_xrd(clean_intensity, noise_level='medium'):
    params = NOISE_LEVELS[noise_level]
    noisy = clean_intensity.copy()

    # Step 1: Random intensity scaling
    if noise_level == 'low':
        scale_range = (0.9, 1.1)
    elif noise_level == 'medium':
        scale_range = (0.8, 1.2)
    else:  # high
        scale_range = (0.7, 1.3)
    noisy = random_intensity_scaling(noisy, scale_range)

    # Step 2: Background variation
    bg_types = ['flat', 'sloping', 'curved']
    bg_type = np.random.choice(bg_types, p=[0.5, 0.3, 0.2])
    noisy = add_background_variation(
        noisy,
        background_type=bg_type,
        background_intensity=params['background_intensity']
    )

    # Step 3: Baseline drift (30% chance)
    if np.random.random() < 0.3:
        drift_amplitude = params['gaussian_sigma']
        noisy = add_baseline_drift(noisy, drift_amplitude)

    # Step 4: Poisson noise
    noisy = add_poisson_noise(noisy, lambda_scale=params['poisson_lambda'])

    # Step 5: Gaussian noise
    noisy = add_gaussian_noise(noisy, sigma=params['gaussian_sigma'])

    # Step 6: Non-negative
    noisy = np.maximum(noisy, 0)

    metadata = {
        'noise_level': noise_level,
        'poisson_lambda': params['poisson_lambda'],
        'gaussian_sigma': params['gaussian_sigma'],
        'background_type': bg_type,
        'background_range': params['background_intensity']
    }

    return noisy, metadata

def determine_noise_level():
    levels = list(NOISE_LEVELS.keys())
    weights = [NOISE_LEVELS[level]['weight'] for level in levels]
    return np.random.choice(levels, p=weights)


# ========================================
# BATCH GENERATION
# ========================================
def generate_all_noisy_data():
    print("\n" + "=" * 70)
    print("GENERATING NOISY XRD DATA")
    print("=" * 70)

    clean_files = sorted([f for f in os.listdir(CLEAN_DATA_DIR) if f.endswith('.npy')])
    print(f"\n📊 Total clean files: {len(clean_files)}")
    print(f"📁 Output directory: {NOISY_OUTPUT_DIR}")

    print(f"\n⚙️  Noise generation settings:")
    for level, params in NOISE_LEVELS.items():
        print(f"   {level.upper():8s}: {params['weight']*100:.0f}% of data")
        print(f"            λ={params['poisson_lambda']}, σ={params['gaussian_sigma']}, "
              f"bg={params['background_intensity']}")

    stats = {
        'success_count': 0,
        'error_count': 0,
        'noise_level_counts': {'low': 0, 'medium': 0, 'high': 0},
        'metadata_list': []
    }

    print(f"\n🔄 Generating noisy data...")

    for clean_filename in tqdm(clean_files, desc="Generating"):
        try:
            # Validasi format nama file
            if not clean_filename.startswith("clean_") or not clean_filename.endswith(".npy"):
                raise ValueError(f"Unexpected filename format: {clean_filename}")

            # Ekstrak ID: clean_001234.npy → 001234
            id_part = clean_filename[6:-4]

            # Load clean data
            clean_path = os.path.join(CLEAN_DATA_DIR, clean_filename)
            clean_intensity = np.load(clean_path)

            # Generate noisy
            noise_level = determine_noise_level()
            noisy_intensity, noise_metadata = generate_noisy_xrd(clean_intensity, noise_level)

            # Simpan sebagai noisy_{id}.npy
            output_filename = f"noisy_{id_part}.npy"
            output_path = os.path.join(NOISY_OUTPUT_DIR, output_filename)
            np.save(output_path, noisy_intensity)

            # Update stats
            stats['success_count'] += 1
            stats['noise_level_counts'][noise_level] += 1
            stats['metadata_list'].append({
                'clean_filename': clean_filename,
                'noisy_filename': output_filename,
                'noise_metadata': noise_metadata
            })

        except Exception as e:
            stats['error_count'] += 1
            print(f"\n❌ Error processing {clean_filename}: {e}")

    return stats


# ========================================
# VISUALIZATION & SUMMARY
# ========================================
def visualize_noise_examples(n_examples=9):
    print("\n📊 Creating noise visualization...")
    clean_files = sorted([f for f in os.listdir(CLEAN_DATA_DIR) if f.endswith('.npy')])
    np.random.seed(42)
    sample_indices = np.random.choice(len(clean_files), n_examples, replace=False)
    fig, axes = plt.subplots(n_examples, 3, figsize=(18, 3*n_examples))
    fig.suptitle('Noise Generation Examples', fontsize=16, fontweight='bold')
    two_theta = np.linspace(10, 80, 2048)

    for idx, (sample_idx, ax_row) in enumerate(zip(sample_indices, axes)):
        clean_filename = clean_files[sample_idx]
        clean_path = os.path.join(CLEAN_DATA_DIR, clean_filename)
        clean_intensity = np.load(clean_path)
        noise_levels = ['low', 'medium', 'high']
        for noise_level, ax in zip(noise_levels, ax_row):
            noisy_intensity, metadata = generate_noisy_xrd(clean_intensity, noise_level)
            ax.plot(two_theta, clean_intensity, 'g-', linewidth=1.5, alpha=0.7, label='Clean')
            ax.plot(two_theta, noisy_intensity, 'b-', linewidth=0.8, label='Noisy')
            ax.set_xlabel('2θ (degrees)', fontsize=9)
            ax.set_ylabel('Intensity (a.u.)', fontsize=9)
            ax.set_title(f'{noise_level.upper()} noise | {clean_filename[:20]}...', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            signal = np.mean(clean_intensity)
            noise_val = np.std(noisy_intensity - clean_intensity)
            snr = 20 * np.log10(signal / (noise_val + 1e-10))
            ax.text(0.02, 0.98,
                   f'SNR: {snr:.1f} dB\n'
                   f'BG: {metadata["background_type"]}\n'
                   f'λ: {metadata["poisson_lambda"]}',
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(NOISY_OUTPUT_DIR, 'noise_generation_examples.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_noise_comparison_plot():
    print("\n📊 Creating noise distribution comparison...")
    clean_files = sorted([f for f in os.listdir(CLEAN_DATA_DIR) if f.endswith('.npy')])
    clean_path = os.path.join(CLEAN_DATA_DIR, clean_files[0])
    clean_intensity = np.load(clean_path)
    n_samples = 50
    noise_levels = ['low', 'medium', 'high']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Noise Level Comparison (50 samples each)', fontsize=14, fontweight='bold')
    two_theta = np.linspace(10, 80, 2048)

    for noise_level, ax in zip(noise_levels, axes):
        ax.plot(two_theta, clean_intensity, 'r-', linewidth=2, alpha=0.8, label='Clean', zorder=100)
        for i in range(n_samples):
            noisy, _ = generate_noisy_xrd(clean_intensity, noise_level)
            ax.plot(two_theta, noisy, 'b-', linewidth=0.3, alpha=0.1)
        ax.set_xlabel('2θ (degrees)', fontsize=11)
        ax.set_ylabel('Intensity (a.u.)', fontsize=11)
        ax.set_title(f'{noise_level.upper()} Noise', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(NOISY_OUTPUT_DIR, 'noise_level_comparison.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def save_generation_summary(stats):
    print("\n📄 Saving generation summary...")
    summary = {
        'noise_generation_config': {
            'noise_levels': NOISE_LEVELS
        },
        'results': {
            'total_generated': stats['success_count'],
            'error_count': stats['error_count'],
            'noise_level_distribution': stats['noise_level_counts']
        },
        'output_directory': NOISY_OUTPUT_DIR
    }

    summary_path = os.path.join(NOISY_OUTPUT_DIR, 'noise_generation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved: {summary_path}")

    metadata_sample = stats['metadata_list'][:1000]
    metadata_path = os.path.join(NOISY_OUTPUT_DIR, 'noise_metadata_sample.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_sample, f, indent=2)
    print(f"✅ Saved: {metadata_path}")


# ========================================
# MAIN
# ========================================
def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  " * 15 + "NOISY XRD DATA GENERATION" + "  " * 27 + "║")
    print("╚" + "═" * 68 + "╝")

    if not os.path.exists(CLEAN_DATA_DIR):
        print(f"\n❌ ERROR: Directory not found: {CLEAN_DATA_DIR}")
        print("Please run 2_preprocess_clean_data.py first!")
        return

    clean_files = [f for f in os.listdir(CLEAN_DATA_DIR) if f.endswith('.npy')]
    print(f"\n⚠️  About to generate noisy versions for {len(clean_files)} clean files")
    print(f"📁 Output will be saved to: {NOISY_OUTPUT_DIR}")

    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    stats = generate_all_noisy_data()

    print("\n" + "= " * 70)
    print("NOISE GENERATION RESULTS")
    print("= " * 70)
    print(f"\n✅ Successfully generated: {stats['success_count']} noisy files")
    print(f"❌ Failed to generate: {stats['error_count']} files")

    print(f"\n📊 Noise level distribution:")
    for level, count in stats['noise_level_counts'].items():
        percentage = (count / stats['success_count']) * 100 if stats['success_count'] > 0 else 0
        print(f"   {level.upper():8s}: {count:6d} files ({percentage:5.1f}%)")

    visualize_noise_examples(n_examples=9)
    create_noise_comparison_plot()
    save_generation_summary(stats)

    print("\n" + "= " * 70)
    print("✅ NOISE GENERATION COMPLETE!")
    print("= " * 70)
    print(f"\n📁 Generated data saved in:")
    print(f"   Clean:  {CLEAN_DATA_DIR}/")
    print(f"   Noisy:  {NOISY_OUTPUT_DIR}/")
    print(f"\n📊 Files:")
    print(f"   - {stats['success_count']} clean-noisy pairs")
    print(f"   - noise_generation_summary.json")
    print(f"   - noise_metadata_sample.json")
    print(f"   - noise_generation_examples.png")
    print(f"   - noise_level_comparison.png")
    print("\n💡 Next step: Split data into train/val and start training!")
    print("   - Run 4_split_dataset.py (pastikan sudah diperbarui untuk .npy dan pairing ID)")

if __name__ == "__main__":
    main()