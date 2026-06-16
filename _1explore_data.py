"""
1_explore_data.py
=================
Exploratory Data Analysis untuk XRD Dataset

Fungsi:
- Analisis struktur file (.xy, .asc, .txt)
- Cek format data
- Statistik deskriptif
- Visualisasi sampel
- Identifikasi file bermasalah

Author: XRD AI Project
Date: 2026-01-29
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================

DATA_DIR = r"C:\Users\COMPUTER\Documents\xrdAI_withoutmatch3_v2\data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "train", "clean")
OUTPUT_DIR = "eda_results"

# Buat output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# UTILITY FUNCTIONS
# ========================================

def load_xrd_file(filepath):
    """
    Load XRD file dengan berbagai format
    
    Returns:
        two_theta: array 2theta values
        intensity: array intensity values
        success: bool
        error_msg: str
    """
    try:
        # Coba baca sebagai text file
        data = np.loadtxt(filepath)
        
        if data.ndim == 1:
            # Single column (hanya intensity)
            two_theta = np.arange(len(data))
            intensity = data
        elif data.shape[1] == 2:
            # Two columns (2theta, intensity)
            two_theta = data[:, 0]
            intensity = data[:, 1]
        elif data.shape[1] > 2:
            # Multiple columns, ambil 2 pertama
            two_theta = data[:, 0]
            intensity = data[:, 1]
        else:
            return None, None, False, "Invalid data shape"
        
        return two_theta, intensity, True, ""
        
    except Exception as e:
        return None, None, False, str(e)


def get_file_info(filepath):
    """
    Extract info dari file
    """
    extension = Path(filepath).suffix
    size_kb = os.path.getsize(filepath) / 1024
    return extension, size_kb


# ========================================
# ANALYSIS FUNCTIONS
# ========================================

def analyze_dataset_structure():
    """
    Analisis struktur dataset
    """
    print("=" * 70)
    print("DATASET STRUCTURE ANALYSIS")
    print("=" * 70)
    
    # Count files by extension
    file_stats = defaultdict(int)
    total_files = 0
    
    for filename in os.listdir(CLEAN_DATA_DIR):
        filepath = os.path.join(CLEAN_DATA_DIR, filename)
        if os.path.isfile(filepath):
            ext = Path(filename).suffix
            file_stats[ext] += 1
            total_files += 1
    
    print(f"\n📊 Total files: {total_files}")
    print(f"\n📁 Files by extension:")
    for ext, count in sorted(file_stats.items()):
        percentage = (count / total_files) * 100
        print(f"  {ext:8s}: {count:6d} files ({percentage:5.2f}%)")
    
    return file_stats, total_files


def sample_files_analysis(n_samples=100):
    """
    Analisis detail dari sample files
    """
    print("\n" + "=" * 70)
    print(f"DETAILED ANALYSIS ({n_samples} samples)")
    print("=" * 70)
    
    # Get sample files
    all_files = [f for f in os.listdir(CLEAN_DATA_DIR) 
                 if os.path.isfile(os.path.join(CLEAN_DATA_DIR, f))]
    
    # Random sampling
    np.random.seed(42)
    sample_files = np.random.choice(all_files, 
                                    min(n_samples, len(all_files)), 
                                    replace=False)
    
    # Statistics
    stats = {
        'num_points': [],
        'two_theta_min': [],
        'two_theta_max': [],
        'two_theta_range': [],
        'intensity_min': [],
        'intensity_max': [],
        'intensity_mean': [],
        'intensity_std': [],
        'file_size_kb': [],
        'success_count': 0,
        'error_count': 0,
        'error_files': []
    }
    
    print(f"\n🔍 Processing {len(sample_files)} files...")
    
    for i, filename in enumerate(sample_files):
        if (i + 1) % 20 == 0:
            print(f"  Processed: {i+1}/{len(sample_files)}")
        
        filepath = os.path.join(CLEAN_DATA_DIR, filename)
        two_theta, intensity, success, error = load_xrd_file(filepath)
        
        if success:
            stats['success_count'] += 1
            stats['num_points'].append(len(intensity))
            stats['two_theta_min'].append(np.min(two_theta))
            stats['two_theta_max'].append(np.max(two_theta))
            stats['two_theta_range'].append(np.max(two_theta) - np.min(two_theta))
            stats['intensity_min'].append(np.min(intensity))
            stats['intensity_max'].append(np.max(intensity))
            stats['intensity_mean'].append(np.mean(intensity))
            stats['intensity_std'].append(np.std(intensity))
            
            ext, size = get_file_info(filepath)
            stats['file_size_kb'].append(size)
        else:
            stats['error_count'] += 1
            stats['error_files'].append((filename, error))
    
    print(f"\n✅ Successfully loaded: {stats['success_count']}/{len(sample_files)}")
    print(f"❌ Failed to load: {stats['error_count']}/{len(sample_files)}")
    
    return stats


def print_statistics(stats):
    """
    Print statistik deskriptif
    """
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    
    # Number of points
    print(f"\n📏 Number of data points per file:")
    print(f"  Min:    {np.min(stats['num_points']):8d}")
    print(f"  Max:    {np.max(stats['num_points']):8d}")
    print(f"  Mean:   {np.mean(stats['num_points']):8.1f}")
    print(f"  Median: {np.median(stats['num_points']):8.1f}")
    print(f"  Std:    {np.std(stats['num_points']):8.1f}")
    
    # 2theta range
    print(f"\n📐 2θ Range (degrees):")
    print(f"  Min start:  {np.min(stats['two_theta_min']):8.2f}°")
    print(f"  Max end:    {np.max(stats['two_theta_max']):8.2f}°")
    print(f"  Avg range:  {np.mean(stats['two_theta_range']):8.2f}°")
    
    # Intensity statistics
    print(f"\n💡 Intensity Statistics:")
    print(f"  Min intensity:  {np.min(stats['intensity_min']):12.2f}")
    print(f"  Max intensity:  {np.max(stats['intensity_max']):12.2f}")
    print(f"  Avg mean:       {np.mean(stats['intensity_mean']):12.2f}")
    print(f"  Avg std:        {np.mean(stats['intensity_std']):12.2f}")
    
    # File size
    print(f"\n💾 File Size (KB):")
    print(f"  Min:    {np.min(stats['file_size_kb']):8.2f} KB")
    print(f"  Max:    {np.max(stats['file_size_kb']):8.2f} KB")
    print(f"  Mean:   {np.mean(stats['file_size_kb']):8.2f} KB")
    
    # Errors
    if stats['error_count'] > 0:
        print(f"\n❌ Error Files ({stats['error_count']}):")
        for filename, error in stats['error_files'][:10]:  # Show first 10
            print(f"  - {filename}: {error}")
        if len(stats['error_files']) > 10:
            print(f"  ... and {len(stats['error_files']) - 10} more")


def create_visualizations(stats):
    """
    Buat visualisasi
    """
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('XRD Dataset Statistics', fontsize=16, fontweight='bold')
    
    # 1. Number of points distribution
    ax = axes[0, 0]
    ax.hist(stats['num_points'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Data Points per File')
    ax.grid(True, alpha=0.3)
    
    # 2. 2theta range distribution
    ax = axes[0, 1]
    ax.hist(stats['two_theta_range'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('2θ Range (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of 2θ Range')
    ax.grid(True, alpha=0.3)
    
    # 3. 2theta start/end scatter
    ax = axes[0, 2]
    ax.scatter(stats['two_theta_min'], stats['two_theta_max'], alpha=0.5, s=10)
    ax.set_xlabel('2θ Start (degrees)')
    ax.set_ylabel('2θ End (degrees)')
    ax.set_title('2θ Range Coverage')
    ax.grid(True, alpha=0.3)
    
    # 4. Intensity mean distribution
    ax = axes[1, 0]
    ax.hist(stats['intensity_mean'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Mean Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Mean Intensity')
    ax.grid(True, alpha=0.3)
    
    # 5. Intensity max distribution (log scale)
    ax = axes[1, 1]
    ax.hist(np.log10(np.array(stats['intensity_max']) + 1), bins=30, 
            edgecolor='black', alpha=0.7, color='red')
    ax.set_xlabel('Log10(Max Intensity)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Max Intensity (log scale)')
    ax.grid(True, alpha=0.3)
    
    # 6. File size distribution
    ax = axes[1, 2]
    ax.hist(stats['file_size_kb'], bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('File Size (KB)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of File Size')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'dataset_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def visualize_sample_xrd_patterns():
    """
    Visualisasi beberapa sampel XRD patterns
    """
    print("\n📊 Creating sample XRD pattern visualizations...")
    
    # Get random samples
    all_files = [f for f in os.listdir(CLEAN_DATA_DIR) 
                 if os.path.isfile(os.path.join(CLEAN_DATA_DIR, f))]
    
    np.random.seed(42)
    sample_files = np.random.choice(all_files, min(9, len(all_files)), replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Sample XRD Patterns', fontsize=16, fontweight='bold')
    
    for idx, (ax, filename) in enumerate(zip(axes.flatten(), sample_files)):
        filepath = os.path.join(CLEAN_DATA_DIR, filename)
        two_theta, intensity, success, _ = load_xrd_file(filepath)
        
        if success:
            ax.plot(two_theta, intensity, 'b-', linewidth=1)
            ax.set_xlabel('2θ (degrees)', fontsize=9)
            ax.set_ylabel('Intensity (a.u.)', fontsize=9)
            ax.set_title(f'{filename[:30]}...', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            ax.text(0.02, 0.98, f'Points: {len(intensity)}\nMax: {np.max(intensity):.0f}',
                   transform=ax.transAxes, fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center')
            ax.set_title(f'{filename[:30]}...', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'sample_xrd_patterns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def save_summary_report(file_stats, total_files, stats):
    """
    Save summary report sebagai JSON
    """
    print("\n📄 Saving summary report...")
    
    summary = {
        'dataset_info': {
            'total_files': total_files,
            'file_extensions': dict(file_stats),
            'data_directory': CLEAN_DATA_DIR
        },
        'statistics': {
            'num_points': {
                'min': int(np.min(stats['num_points'])),
                'max': int(np.max(stats['num_points'])),
                'mean': float(np.mean(stats['num_points'])),
                'median': float(np.median(stats['num_points'])),
                'std': float(np.std(stats['num_points']))
            },
            'two_theta_range': {
                'min_start': float(np.min(stats['two_theta_min'])),
                'max_end': float(np.max(stats['two_theta_max'])),
                'avg_range': float(np.mean(stats['two_theta_range']))
            },
            'intensity': {
                'min': float(np.min(stats['intensity_min'])),
                'max': float(np.max(stats['intensity_max'])),
                'avg_mean': float(np.mean(stats['intensity_mean'])),
                'avg_std': float(np.mean(stats['intensity_std']))
            },
            'file_size_kb': {
                'min': float(np.min(stats['file_size_kb'])),
                'max': float(np.max(stats['file_size_kb'])),
                'mean': float(np.mean(stats['file_size_kb']))
            }
        },
        'quality_check': {
            'success_rate': f"{(stats['success_count'] / (stats['success_count'] + stats['error_count']) * 100):.2f}%",
            'error_count': stats['error_count'],
            'error_files': [{'filename': f, 'error': e} for f, e in stats['error_files'][:100]]
        }
    }
    
    output_path = os.path.join(OUTPUT_DIR, 'eda_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved: {output_path}")


def generate_recommendations(stats):
    """
    Generate rekomendasi preprocessing
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING RECOMMENDATIONS")
    print("=" * 70)
    
    # Analyze data points variability
    num_points_std = np.std(stats['num_points'])
    num_points_mean = np.mean(stats['num_points'])
    cv_points = num_points_std / num_points_mean * 100  # Coefficient of variation
    
    print(f"\n1️⃣ RESAMPLING:")
    if cv_points > 20:
        print(f"   ⚠️  High variability in data points (CV={cv_points:.1f}%)")
        print(f"   ✅ RECOMMENDATION: Resample all files to uniform length")
        recommended_points = int(np.median(stats['num_points']))
        print(f"   📌 Suggested target: {recommended_points} points")
    else:
        print(f"   ✅ Data points relatively uniform (CV={cv_points:.1f}%)")
        print(f"   📌 Minimal resampling needed")
    
    # Analyze 2theta range
    two_theta_range_std = np.std(stats['two_theta_range'])
    
    print(f"\n2️⃣ 2θ RANGE STANDARDIZATION:")
    if two_theta_range_std > 5:
        print(f"   ⚠️  Variable 2θ ranges detected (std={two_theta_range_std:.1f}°)")
        print(f"   ✅ RECOMMENDATION: Standardize to common range")
        common_min = np.percentile(stats['two_theta_min'], 25)
        common_max = np.percentile(stats['two_theta_max'], 75)
        print(f"   📌 Suggested range: {common_min:.1f}° - {common_max:.1f}°")
    else:
        print(f"   ✅ 2θ ranges relatively consistent")
    
    # Analyze intensity range
    intensity_max_range = np.max(stats['intensity_max']) / np.min(stats['intensity_max'])
    
    print(f"\n3️⃣ INTENSITY NORMALIZATION:")
    print(f"   Max intensity ratio: {intensity_max_range:.1f}x")
    if intensity_max_range > 100:
        print(f"   ⚠️  Large intensity variations!")
        print(f"   ✅ RECOMMENDATION: Apply robust normalization")
        print(f"   📌 Suggested: Anscombe transform + min-max scaling")
    else:
        print(f"   ✅ Standard normalization should work")
    
    print(f"\n4️⃣ DATA QUALITY:")
    error_rate = stats['error_count'] / (stats['success_count'] + stats['error_count']) * 100
    if error_rate > 5:
        print(f"   ⚠️  High error rate: {error_rate:.1f}%")
        print(f"   ✅ RECOMMENDATION: Manual review of error files")
    else:
        print(f"   ✅ Good data quality (error rate: {error_rate:.2f}%)")


# ========================================
# MAIN
# ========================================

def main():
    """
    Main EDA pipeline
    """
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "XRD DATASET EDA" + " " * 33 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Check directory exists
    if not os.path.exists(CLEAN_DATA_DIR):
        print(f"\n❌ ERROR: Directory not found: {CLEAN_DATA_DIR}")
        print("Please check your DATA_DIR path in the script!")
        return
    
    # Step 1: Dataset structure
    file_stats, total_files = analyze_dataset_structure()
    
    # Step 2: Sample analysis
    stats = sample_files_analysis(n_samples=200)  # Analyze 200 random files
    
    # Step 3: Print statistics
    print_statistics(stats)
    
    # Step 4: Create visualizations
    create_visualizations(stats)
    visualize_sample_xrd_patterns()
    
    # Step 5: Save summary
    save_summary_report(file_stats, total_files, stats)
    
    # Step 6: Recommendations
    generate_recommendations(stats)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ EDA COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved in: {OUTPUT_DIR}/")
    print(f"   - dataset_statistics.png")
    print(f"   - sample_xrd_patterns.png")
    print(f"   - eda_summary.json")
    print("\n💡 Next step: Run 2_preprocess_clean_data.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
