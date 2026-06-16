"""
2_preprocess_clean_data.py
===========================
Preprocessing Clean XRD Data

Fungsi:
- Load semua clean data (24,168 files)
- Standardisasi format & range
- Resample ke jumlah titik uniform
- Save dalam format .npy untuk training

Author: XRD AI Project
Date: 2026-01-29
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================

DATA_DIR = r"C:\Users\COMPUTER\Documents\xrdAI_withoutmatch3_v2\data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "train", "clean")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed", "clean")

# Preprocessing parameters
TARGET_NUM_POINTS = 2048  # Uniform length for all data
TARGET_TWO_THETA_MIN = 3.0  # degrees (diperluas dari 10.0)
TARGET_TWO_THETA_MAX = 150.0  # degrees (diperluas dari 80.0)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# UTILITY FUNCTIONS
# ========================================

def load_xrd_file(filepath):
    """
    Load XRD file dengan berbagai format
    Improvements:
    - Skip header comments (#)
    - Skip text headers (non-numeric first line)
    - Auto-detect delimiter (tab, space, comma)
    - Handle Windows line endings
    """
    try:
        # Method 1: Try numpy loadtxt with comments
        try:
            data = np.loadtxt(filepath, comments='#', delimiter=None)
            if data.size > 0:
                if data.ndim == 1:
                    two_theta = np.arange(len(data))
                    intensity = data
                elif data.shape[1] >= 2:
                    two_theta = data[:, 0]
                    intensity = data[:, 1]
                else:
                    raise ValueError("Invalid shape")
                return two_theta, intensity, True
        except:
            pass
        
        # Method 2: Manual parsing for problematic files
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Skip comments and find first numeric line
        data_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip comment lines
            if line.startswith('#'):
                continue
            # Skip text header (check if first char is digit or minus)
            if line[0].isdigit() or line[0] == '-' or line[0] == '.':
                data_lines.append(line)
        
        if len(data_lines) == 0:
            return None, None, False
        
        # Parse data lines
        two_theta_list = []
        intensity_list = []
        
        for line in data_lines:
            # Try different delimiters
            for delimiter in ['\t', ' ', ',', ';']:
                parts = [p.strip() for p in line.split(delimiter) if p.strip()]
                if len(parts) >= 2:
                    try:
                        tt = float(parts[0])
                        it = float(parts[1])
                        two_theta_list.append(tt)
                        intensity_list.append(it)
                        break
                    except:
                        continue
        
        if len(two_theta_list) == 0:
            return None, None, False
        
        two_theta = np.array(two_theta_list)
        intensity = np.array(intensity_list)
        
        return two_theta, intensity, True
        
    except Exception as e:
        return None, None, False


def resample_xrd(two_theta, intensity, target_num_points=2048):
    """
    Resample XRD data ke jumlah titik uniform
    
    Args:
        two_theta: original 2theta values
        intensity: original intensity values
        target_num_points: target number of points
    
    Returns:
        resampled_two_theta: uniform 2theta array
        resampled_intensity: interpolated intensity
    """
    # Remove duplicates and sort
    sorted_indices = np.argsort(two_theta)
    two_theta_sorted = two_theta[sorted_indices]
    intensity_sorted = intensity[sorted_indices]
    
    # Remove exact duplicates
    unique_indices = np.concatenate([[True], np.diff(two_theta_sorted) != 0])
    two_theta_unique = two_theta_sorted[unique_indices]
    intensity_unique = intensity_sorted[unique_indices]
    
    # Create interpolation function
    if len(two_theta_unique) < 2:
        return None, None
    
    interp_func = interpolate.interp1d(
        two_theta_unique, 
        intensity_unique,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    
    # Create uniform 2theta grid
    resampled_two_theta = np.linspace(
        two_theta_unique.min(),
        two_theta_unique.max(),
        target_num_points
    )
    
    # Interpolate
    resampled_intensity = interp_func(resampled_two_theta)
    
    return resampled_two_theta, resampled_intensity


def clip_to_range(two_theta, intensity, min_angle=3.0, max_angle=150.0):
    """
    Clip data ke 2theta range tertentu
    
    Args:
        two_theta: 2theta values
        intensity: intensity values
        min_angle: minimum 2theta
        max_angle: maximum 2theta
    
    Returns:
        clipped_two_theta, clipped_intensity
    """
    # Find indices in range
    mask = (two_theta >= min_angle) & (two_theta <= max_angle)
    
    if np.sum(mask) < 5:  # Dikurangi dari 10 menjadi 5 (lebih toleran)
        return None, None
    
    return two_theta[mask], intensity[mask]


def remove_outliers(intensity, percentile=99.9):
    """
    Remove extreme outliers dari intensity
    
    Args:
        intensity: intensity values
        percentile: percentile cutoff
    
    Returns:
        cleaned_intensity
    """
    threshold = np.percentile(intensity, percentile)
    intensity_cleaned = np.clip(intensity, 0, threshold)
    return intensity_cleaned


def smooth_data(intensity, window=3):
    """
    Smooth data dengan uniform filter (optional)
    
    Args:
        intensity: intensity values
        window: smoothing window size
    
    Returns:
        smoothed_intensity
    """
    if window > 1:
        return uniform_filter1d(intensity, size=window)
    return intensity


# ========================================
# PREPROCESSING PIPELINE
# ========================================

def preprocess_single_file(filepath, 
                          target_num_points=2048,
                          target_min=3.0,
                          target_max=150.0,
                          remove_outlier=True,
                          smooth_window=1):
    """
    Preprocess single XRD file
    
    Returns:
        preprocessed_intensity: array [target_num_points]
        success: bool
        metadata: dict
    """
    # Load file
    two_theta, intensity, success = load_xrd_file(filepath)
    if not success:
        return None, False, None
    
    # Metadata
    metadata = {
        'original_num_points': len(intensity),
        'original_two_theta_range': [float(two_theta.min()), float(two_theta.max())],
        'original_intensity_range': [float(intensity.min()), float(intensity.max())]
    }
    
    # Step 1: Remove negative values
    intensity = np.maximum(intensity, 0)
    
    # Step 2: Clip to target 2theta range
    two_theta_clipped, intensity_clipped = clip_to_range(
        two_theta, intensity, target_min, target_max
    )
    
    if two_theta_clipped is None:
        return None, False, None
    
    # Step 3: Remove outliers (optional)
    if remove_outlier:
        intensity_clipped = remove_outliers(intensity_clipped)
    
    # Step 4: Resample to uniform grid
    two_theta_resampled, intensity_resampled = resample_xrd(
        two_theta_clipped, intensity_clipped, target_num_points
    )
    
    if intensity_resampled is None:
        return None, False, None
    
    # Step 5: Smooth (optional)
    if smooth_window > 1:
        intensity_resampled = smooth_data(intensity_resampled, smooth_window)
    
    # Final check
    if len(intensity_resampled) != target_num_points:
        return None, False, None
    
    # Update metadata
    metadata['preprocessed_num_points'] = len(intensity_resampled)
    metadata['preprocessed_two_theta_range'] = [float(two_theta_resampled.min()), 
                                                 float(two_theta_resampled.max())]
    metadata['preprocessed_intensity_range'] = [float(intensity_resampled.min()), 
                                                 float(intensity_resampled.max())]
    
    return intensity_resampled, True, metadata


def preprocess_all_files():
    """
    Preprocess semua clean files
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING ALL CLEAN FILES")
    print("=" * 70)
    
    # Get all files
    all_files = [f for f in os.listdir(CLEAN_DATA_DIR) 
                 if os.path.isfile(os.path.join(CLEAN_DATA_DIR, f))]
    
    print(f"\n📊 Total files to process: {len(all_files)}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"\n⚙️  Preprocessing settings:")
    print(f"   - Target points: {TARGET_NUM_POINTS}")
    print(f"   - Target 2θ range: {TARGET_TWO_THETA_MIN}° - {TARGET_TWO_THETA_MAX}°")
    
    # Statistics
    stats = {
        'success_count': 0,
        'error_count': 0,
        'error_files': [],
        'metadata_list': []
    }
    
    # Progress bar
    print(f"\n🔄 Processing files...")
    
    for i, filename in enumerate(tqdm(all_files, desc="Preprocessing")):
        filepath = os.path.join(CLEAN_DATA_DIR, filename)
        
        # Preprocess
        intensity, success, metadata = preprocess_single_file(
            filepath,
            target_num_points=TARGET_NUM_POINTS,
            target_min=TARGET_TWO_THETA_MIN,
            target_max=TARGET_TWO_THETA_MAX,
            remove_outlier=True,
            smooth_window=1
        )
        
        if success:
            # Save as .npy
            output_filename = f"clean_{i:06d}.npy"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            np.save(output_path, intensity)
            
            stats['success_count'] += 1
            stats['metadata_list'].append({
                'original_filename': filename,
                'output_filename': output_filename,
                'metadata': metadata
            })
        else:
            stats['error_count'] += 1
            stats['error_files'].append(filename)
    
    return stats


def visualize_preprocessing_examples(n_examples=6):
    """
    Visualisasi contoh before-after preprocessing
    """
    print("\n📊 Creating preprocessing visualization...")
    
    # Get random samples
    all_files = [f for f in os.listdir(CLEAN_DATA_DIR) 
                 if os.path.isfile(os.path.join(CLEAN_DATA_DIR, f))]
    
    np.random.seed(42)
    sample_files = np.random.choice(all_files, min(n_examples, len(all_files)), replace=False)
    
    fig, axes = plt.subplots(n_examples, 2, figsize=(16, 4*n_examples))
    fig.suptitle('Preprocessing Examples: Before vs After', fontsize=16, fontweight='bold')
    
    for idx, (filename, ax_row) in enumerate(zip(sample_files, axes)):
        # Load original
        filepath = os.path.join(CLEAN_DATA_DIR, filename)
        two_theta_orig, intensity_orig, success = load_xrd_file(filepath)
        
        if not success:
            continue
        
        # Preprocess
        intensity_preprocessed, success, metadata = preprocess_single_file(filepath)
        
        if not success:
            continue
        
        # Create uniform 2theta for preprocessed
        two_theta_preprocessed = np.linspace(TARGET_TWO_THETA_MIN, 
                                            TARGET_TWO_THETA_MAX, 
                                            TARGET_NUM_POINTS)
        
        # Plot original
        ax_row[0].plot(two_theta_orig, intensity_orig, 'b-', linewidth=1)
        ax_row[0].set_xlabel('2θ (degrees)', fontsize=10)
        ax_row[0].set_ylabel('Intensity (a.u.)', fontsize=10)
        ax_row[0].set_title(f'Before: {filename[:40]}...', fontsize=9)
        ax_row[0].grid(True, alpha=0.3)
        ax_row[0].text(0.02, 0.98, 
                      f'Points: {len(intensity_orig)}\n' + 
                      f'Range: {two_theta_orig.min():.1f}°-{two_theta_orig.max():.1f}°\n' +
                      f'Max: {intensity_orig.max():.0f}',
                      transform=ax_row[0].transAxes, fontsize=8, 
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot preprocessed
        ax_row[1].plot(two_theta_preprocessed, intensity_preprocessed, 'g-', linewidth=1)
        ax_row[1].set_xlabel('2θ (degrees)', fontsize=10)
        ax_row[1].set_ylabel('Intensity (a.u.)', fontsize=10)
        ax_row[1].set_title(f'After: clean_{idx:06d}.npy', fontsize=9)
        ax_row[1].grid(True, alpha=0.3)
        ax_row[1].text(0.02, 0.98, 
                      f'Points: {len(intensity_preprocessed)}\n' + 
                      f'Range: {TARGET_TWO_THETA_MIN:.1f}°-{TARGET_TWO_THETA_MAX:.1f}°\n' +
                      f'Max: {intensity_preprocessed.max():.0f}',
                      transform=ax_row[1].transAxes, fontsize=8, 
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'preprocessing_examples.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def save_preprocessing_summary(stats):
    """
    Save preprocessing summary
    """
    print("\n📄 Saving preprocessing summary...")
    
    summary = {
        'preprocessing_config': {
            'target_num_points': TARGET_NUM_POINTS,
            'target_two_theta_range': [TARGET_TWO_THETA_MIN, TARGET_TWO_THETA_MAX],
            'outlier_removal': True,
            'smooth_window': 1
        },
        'results': {
            'total_processed': stats['success_count'] + stats['error_count'],
            'success_count': stats['success_count'],
            'error_count': stats['error_count'],
            'success_rate': f"{stats['success_count'] / (stats['success_count'] + stats['error_count']) * 100:.2f}%"
        },
        'error_files': stats['error_files'][:100],  # Save first 100 errors
        'output_directory': OUTPUT_DIR
    }
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'preprocessing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved: {summary_path}")
    
    # Save metadata (sample of 1000 files to keep file size reasonable)
    if len(stats['metadata_list']) > 1000:
        metadata_sample = np.random.choice(stats['metadata_list'], 1000, replace=False).tolist()
    else:
        metadata_sample = stats['metadata_list']
    
    metadata_path = os.path.join(OUTPUT_DIR, 'preprocessing_metadata_sample.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_sample, f, indent=2)
    print(f"✅ Saved: {metadata_path}")
    
    # Save simple mapping file for reference
    mapping = {item['original_filename']: item['output_filename'] 
               for item in stats['metadata_list']}
    mapping_path = os.path.join(OUTPUT_DIR, 'filename_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Saved: {mapping_path}")


# ========================================
# MAIN
# ========================================

def main():
    """
    Main preprocessing pipeline
    """
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "XRD DATA PREPROCESSING" + " " * 30 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Check directory exists
    if not os.path.exists(CLEAN_DATA_DIR):
        print(f"\n❌ ERROR: Directory not found: {CLEAN_DATA_DIR}")
        print("Please check your DATA_DIR path in the script!")
        return
    
    # Confirm before processing
    all_files = [f for f in os.listdir(CLEAN_DATA_DIR) 
                 if os.path.isfile(os.path.join(CLEAN_DATA_DIR, f))]
    
    print(f"\n⚠️  About to preprocess {len(all_files)} files")
    print(f"📁 Output will be saved to: {OUTPUT_DIR}")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Step 1: Preprocess all files
    stats = preprocess_all_files()
    
    # Step 2: Print results
    print("\n" + "=" * 70)
    print("PREPROCESSING RESULTS")
    print("=" * 70)
    print(f"\n✅ Successfully preprocessed: {stats['success_count']} files")
    print(f"❌ Failed to preprocess: {stats['error_count']} files")
    
    if stats['error_count'] > 0:
        print(f"\n❌ Error files ({min(10, stats['error_count'])} shown):")
        for filename in stats['error_files'][:10]:
            print(f"   - {filename}")
        if stats['error_count'] > 10:
            print(f"   ... and {stats['error_count'] - 10} more")
    
    # Step 3: Create visualizations
    visualize_preprocessing_examples(n_examples=6)
    
    # Step 4: Save summary
    save_preprocessing_summary(stats)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Preprocessed data saved in: {OUTPUT_DIR}/")
    print(f"   - {stats['success_count']} .npy files")
    print(f"   - preprocessing_summary.json")
    print(f"   - preprocessing_metadata_sample.json")
    print(f"   - filename_mapping.json")
    print(f"   - preprocessing_examples.png")
    print("\n💡 Next step: Run 3_generate_noisy_data.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()