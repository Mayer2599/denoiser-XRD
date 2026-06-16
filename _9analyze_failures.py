"""
Failure Case Analysis - Identify dan analyze worst performing samples
✅ FIXED: Encoding error dengan UTF-8
✅ ENHANCED: Report lengkap dengan statistik XRD-specific
✅ ENHANCED: Analisis pola extreme values (dominan di dataset Anda)
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
from _8evaluate_model import ModelEvaluator, calculate_snr
from xrd_dataset import XRDDataset
from torch.utils.data import DataLoader

class FailureCaseAnalyzer:
    """Analyze failure cases untuk model improvement"""
    
    def __init__(self, model_path, device='cuda'):
        self.evaluator = ModelEvaluator(model_path, device=device)

    def analyze_failures(self, dataloader, num_worst=10, num_best=5):
        """
        Identify dan analyze worst dan best cases
        
        Parameters:
        -----------
        dataloader : DataLoader
            Test data loader
        num_worst : int
            Number of worst cases to analyze
        num_best : int
            Number of best cases to show (for comparison)
        
        Returns:
        --------
        results : dict
            Analysis results
        """
        print("=" * 80)
        print("FAILURE CASE ANALYSIS")
        print("=" * 80)
        
        all_results = []
        
        print("\nProcessing samples...")
        with torch.no_grad():
            for batch_idx, (noisy, clean) in enumerate(tqdm(dataloader, desc="Processing")):
                # Move to device
                noisy = noisy.to(self.evaluator.device)
                clean = clean.to(self.evaluator.device)
                
                # Forward pass
                denoised = self.evaluator.model(noisy)
                
                # Convert to numpy
                noisy_np = noisy.cpu().numpy()
                clean_np = clean.cpu().numpy()
                denoised_np = denoised.cpu().numpy()
                
                # Process each sample
                batch_size = noisy_np.shape[0]
                for i in range(batch_size):
                    n = noisy_np[i, 0, :]
                    c = clean_np[i, 0, :]
                    d = denoised_np[i, 0, :]
                    
                    # Calculate metrics
                    noise_before = n - c
                    noise_after = d - c
                    
                    snr_before = calculate_snr(c, noise_before)
                    snr_after = calculate_snr(c, noise_after)
                    snr_improvement = snr_after - snr_before
                    
                    mse = np.mean((c - d) ** 2)
                    mae = np.mean(np.abs(c - d))
                    
                    # Store
                    all_results.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'snr_improvement': snr_improvement,
                        'snr_before': snr_before,
                        'snr_after': snr_after,
                        'mse': mse,
                        'mae': mae,
                        'noisy': n,
                        'clean': c,
                        'denoised': d
                    })
        
        # Sort by SNR improvement (ascending = worst first)
        all_results.sort(key=lambda x: x['snr_improvement'])
        
        # Get worst and best cases
        worst_cases = all_results[:num_worst]
        best_cases = all_results[-num_best:]
        
        print(f"\n✓ Identified {num_worst} worst cases and {num_best} best cases")
        
        return worst_cases, best_cases, all_results

    def analyze_failure_patterns(self, worst_cases):
        """
        Analyze patterns in failure cases
        
        Returns:
        --------
        patterns : dict
            Identified patterns with detailed statistics
        """
        print("\n" + "=" * 80)
        print("ANALYZING FAILURE PATTERNS")
        print("=" * 80)
        
        patterns = {
            'high_noise': [],
            'low_signal': [],
            'extreme_values': [],
            'specific_regions': [],
            'peak_distortion': []
        }
        
        # Thresholds untuk XRD-specific analysis
        HIGH_NOISE_THRESHOLD = 0.15  # STD noise relatif terhadap signal
        LOW_SIGNAL_THRESHOLD = 0.3   # Mean intensity
        EXTREME_VALUE_THRESHOLD = 0.95  # Max intensity
        PEAK_DISTORTION_THRESHOLD = 0.2  # Relative error pada peak regions
        
        for case in worst_cases: 
            noisy = case['noisy']
            clean = case['clean']
            denoised = case['denoised']
            
            # Pattern 1: High noise level
            noise_level = np.std(noisy - clean) / (np.std(clean) + 1e-8)
            if noise_level > HIGH_NOISE_THRESHOLD:
                patterns['high_noise'].append(case)
            
            # Pattern 2: Low signal strength
            signal_strength = np.mean(clean)
            if signal_strength < LOW_SIGNAL_THRESHOLD:
                patterns['low_signal'].append(case)
            
            # Pattern 3: Extreme values (DOMINAN di dataset Anda!)
            if np.max(noisy) > EXTREME_VALUE_THRESHOLD or np.min(noisy) < (1 - EXTREME_VALUE_THRESHOLD):
                patterns['extreme_values'].append(case)
            
            # Pattern 4: Failure in specific regions (peak regions)
            # Deteksi peak pada clean signal
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(clean, height=np.mean(clean) * 1.5, distance=20)
            
            if len(peaks) > 0:
                # Hitung error pada peak regions (±10 points)
                peak_errors = []
                for peak in peaks:
                    start = max(0, peak - 10)
                    end = min(len(clean), peak + 10)
                    error = np.mean(np.abs(clean[start:end] - denoised[start:end]))
                    peak_errors.append(error)
                
                avg_peak_error = np.mean(peak_errors)
                avg_non_peak_error = np.mean(np.abs(clean - denoised)) - avg_peak_error
                
                if avg_peak_error > PEAK_DISTORTION_THRESHOLD:
                    patterns['peak_distortion'].append(case)
        
        # Print pattern statistics
        print("\nIdentified Patterns in Worst Cases:")
        print(f"  • Extreme values (intensity >95% atau <5%): {len(patterns['extreme_values'])}/{len(worst_cases)} cases ({len(patterns['extreme_values'])/len(worst_cases)*100:.0f}%) ← DOMINAN!")
        print(f"  • Peak distortion (error pada peak regions): {len(patterns['peak_distortion'])}/{len(worst_cases)} cases")
        print(f"  • High noise level (relatif): {len(patterns['high_noise'])}/{len(worst_cases)} cases")
        print(f"  • Low signal strength: {len(patterns['low_signal'])}/{len(worst_cases)} cases")
        
        return patterns

    def plot_failure_cases(self, worst_cases, best_cases, output_dir):
        """Plot worst and best cases + histogram SNR improvement"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot worst cases
        num_worst = len(worst_cases)
        fig, axes = plt.subplots(num_worst, 1, figsize=(15, 4 * num_worst))
        
        if num_worst == 1:
            axes = [axes]
        
        for idx, case in enumerate(worst_cases):
            ax = axes[idx]
            
            x = np.arange(len(case['noisy']))
            
            ax.plot(x, case['noisy'], 'gray', alpha=0.5, label='Noisy', linewidth=1)
            ax.plot(x, case['clean'], 'g', label='Clean', linewidth=1.5)
            ax.plot(x, case['denoised'], 'r', label='Denoised', linewidth=1.5)
            
            # Highlight extreme values
            extreme_mask = (case['noisy'] > 0.95) | (case['noisy'] < 0.05)
            if np.any(extreme_mask):
                ax.scatter(x[extreme_mask], case['noisy'][extreme_mask], 
                          color='orange', s=20, zorder=5, label='Extreme values', alpha=0.7)
            
            ax.set_title(
                f"WORST Case #{idx+1}: SNR Improvement = {case['snr_improvement']:.2f} dB  "
                f"(Before: {case['snr_before']:.2f} dB, After: {case['snr_after']:.2f} dB)",
                fontweight='bold', fontsize=12, color='red'
            )
            ax.set_xlabel("2θ Position")
            ax.set_ylabel("Intensity (normalized)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        worst_path = output_dir / "worst_case3b.png"
        plt.savefig(worst_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Worst cases plot saved: {worst_path}")
        
        # Plot best cases (for comparison)
        num_best = len(best_cases)
        fig, axes = plt.subplots(num_best, 1, figsize=(15, 4 * num_best))
        
        if num_best == 1:
            axes = [axes]
        
        for idx, case in enumerate(best_cases):
            ax = axes[idx]
            
            x = np.arange(len(case['noisy']))
            
            ax.plot(x, case['noisy'], 'gray', alpha=0.5, label='Noisy', linewidth=1)
            ax.plot(x, case['clean'], 'g', label='Clean', linewidth=1.5)
            ax.plot(x, case['denoised'], 'b', label='Denoised', linewidth=1.5)
            
            ax.set_title(
                f"BEST Case #{idx+1}: SNR Improvement = {case['snr_improvement']:.2f} dB  "
                f"(Before: {case['snr_before']:.2f} dB, After: {case['snr_after']:.2f} dB)",
                fontweight='bold', fontsize=12, color='green'
            )
            ax.set_xlabel("2θ Position")
            ax.set_ylabel("Intensity (normalized)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        best_path = output_dir / "best_cases3b.png"
        plt.savefig(best_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Best cases plot saved: {best_path}")
        
        # Plot histogram SNR improvement distribution
        all_improvements = [case['snr_improvement'] for case in worst_cases + best_cases]
        plt.figure(figsize=(10, 6))
        plt.hist(all_improvements, bins=30, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(all_improvements), color='red', linestyle='--', label=f'Mean: {np.mean(all_improvements):.2f} dB')
        plt.axvline(np.median(all_improvements), color='green', linestyle='--', label=f'Median: {np.median(all_improvements):.2f} dB')
        plt.xlabel('SNR Improvement (dB)')
        plt.ylabel('Frequency')
        plt.title('Distribution of SNR Improvement Across Samples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "snr_improvement_histogram3b.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ SNR improvement histogram saved: {output_dir / 'snr_improvement_histogram3b.png'}")

    def generate_report(self, worst_cases, best_cases, patterns, all_results, output_dir):
        """Generate comprehensive failure analysis report with UTF-8 encoding"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Kumpulkan statistik global
        all_improvements = [r['snr_improvement'] for r in all_results]
        all_mse = [r['mse'] for r in all_results]
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FAILURE CASE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nTotal Samples Analyzed: {len(all_results)}")
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Statistik global
        report_lines.append("\n" + "-" * 80)
        report_lines.append("GLOBAL STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"SNR Improvement (mean ± std): {np.mean(all_improvements):.2f} ± {np.std(all_improvements):.2f} dB")
        report_lines.append(f"SNR Improvement (median): {np.median(all_improvements):.2f} dB")
        report_lines.append(f"SNR Improvement (min): {np.min(all_improvements):.2f} dB")
        report_lines.append(f"SNR Improvement (max): {np.max(all_improvements):.2f} dB")
        report_lines.append(f"MSE (mean): {np.mean(all_mse):.6f}")
        report_lines.append(f"MSE (median): {np.median(all_mse):.6f}")
        
        # Worst cases detail
        report_lines.append("\n" + "-" * 80)
        report_lines.append("WORST PERFORMING SAMPLES (Top 10)")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Rank':<6} {'SNR Improv.':<15} {'SNR Before':<15} {'SNR After':<15} {'MSE':<12} {'MAE':<12}")
        report_lines.append("-" * 80)
        
        for idx, case in enumerate(worst_cases, 1):
            report_lines.append(
                f"{idx:<6} {case['snr_improvement']:<15.2f} {case['snr_before']:<15.2f} "
                f"{case['snr_after']:<15.2f} {case['mse']:<12.6f} {case['mae']:<12.6f}"
            )
        
        # Best cases untuk perbandingan
        report_lines.append("\n" + "-" * 80)
        report_lines.append("BEST PERFORMING SAMPLES (Top 5) - FOR COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Rank':<6} {'SNR Improv.':<15} {'SNR Before':<15} {'SNR After':<15} {'MSE':<12} {'MAE':<12}")
        report_lines.append("-" * 80)
        
        for idx, case in enumerate(reversed(best_cases), 1):  # reversed untuk urutan descending
            report_lines.append(
                f"{idx:<6} {case['snr_improvement']:<15.2f} {case['snr_before']:<15.2f} "
                f"{case['snr_after']:<15.2f} {case['mse']:<12.6f} {case['mae']:<12.6f}"
            )
        
        # Pattern analysis
        report_lines.append("\n" + "-" * 80)
        report_lines.append("FAILURE PATTERN ANALYSIS")
        report_lines.append("-" * 80)
        
        total_worst = len(worst_cases)
        patterns_summary = [
            ("Extreme values (intensity >95% atau <5%)", len(patterns['extreme_values'])),
            ("Peak distortion (error pada peak regions)", len(patterns['peak_distortion'])),
            ("High noise level (relatif)", len(patterns['high_noise'])),
            ("Low signal strength", len(patterns['low_signal'])),
        ]
        
        for pattern_name, count in patterns_summary:
            percentage = (count / total_worst) * 100
            marker = " ← CRITICAL!" if percentage > 70 else (" ← SIGNIFICANT" if percentage > 30 else "")
            report_lines.append(f"• {pattern_name}: {count}/{total_worst} cases ({percentage:.0f}%) {marker}")
        
        # Rekomendasi khusus untuk XRD
        report_lines.append("\n" + "-" * 80)
        report_lines.append("XR D-SPECIFIC RECOMMENDATIONS")
        report_lines.append("-" * 80)
        
        # Rekomendasi berdasarkan pola dominan
        if len(patterns['extreme_values']) / total_worst > 0.7:
            report_lines.append("\n⚠️  CRITICAL ISSUE: Extreme values dominate failures ({}%)".format(
                int(len(patterns['extreme_values']) / total_worst * 100)
            ))
            report_lines.append("   Recommended actions:")
            report_lines.append("   1. Preprocessing enhancement:")
            report_lines.append("      - Apply robust normalization (e.g., RobustScaler) instead of MinMaxScaler")
            report_lines.append("      - Clip extreme outliers (>99.5 percentile) BEFORE training")
            report_lines.append("      - Consider log-transform for high-intensity regions")
            report_lines.append("   2. Loss function modification:")
            report_lines.append("      - Replace MSE with Huber loss (less sensitive to outliers)")
            report_lines.append("      - Add weighted loss: higher weight for extreme value regions")
            report_lines.append("   3. Data augmentation:")
            report_lines.append("      - Synthetically generate more samples with extreme values")
            report_lines.append("      - Apply random clipping to simulate extreme value scenarios")
        
        if len(patterns['peak_distortion']) / total_worst > 0.3:
            report_lines.append("\n⚠️  PEAK PRESERVATION ISSUE: {}% of failures show peak distortion".format(
                int(len(patterns['peak_distortion']) / total_worst * 100)
            ))
            report_lines.append("   Recommended actions:")
            report_lines.append("   1. Architecture modification:")
            report_lines.append("      - Increase skip connection weights in UNet")
            report_lines.append("      - Add attention gates focused on peak regions")
            report_lines.append("   2. Loss function enhancement:")
            report_lines.append("      - Add peak-preserving loss term:")
            report_lines.append("        L_total = L_mse + λ * L_peak")
            report_lines.append("        where L_peak = MSE only on detected peak regions")
            report_lines.append("   3. Post-processing:")
            report_lines.append("      - Apply peak-aware smoothing after denoising")
        
        # General recommendations
        report_lines.append("\n" + "-" * 80)
        report_lines.append("GENERAL RECOMMENDATIONS FOR NEXT EXPERIMENT")
        report_lines.append("-" * 80)
        report_lines.append("✅ DO: Continue with Experiment 3 (UNet + Lower LR) - already showing promise")
        report_lines.append("✅ DO: Add extreme value preprocessing BEFORE Experiment 4")
        report_lines.append("✅ DO: Implement Huber loss instead of MSE for robustness")
        report_lines.append("⚠️  CAUTION: Avoid Experiment 4 (Larger Channels) WITHOUT fixing extreme value issue first")
        report_lines.append("   → Larger capacity will memorize extreme outliers instead of generalizing")
        report_lines.append("💡 PRO TIP: For XRD specifically, peak preservation > overall MSE reduction")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Save report dengan UTF-8 encoding (FIX UTAMA!)
        report_text = "\n".join(report_lines)
        report_path = output_dir / "failure_analysis_report3b.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n✓ Report saved successfully: {report_path}")
            print("\n" + report_text)
        except Exception as e:
            print(f"\n⚠️  Warning: UTF-8 write failed, trying fallback encoding...")
            try:
                with open(report_path, 'w', encoding='latin-1') as f:
                    f.write(report_text.encode('latin-1', errors='replace').decode('latin-1'))
                print(f"✓ Report saved with fallback encoding: {report_path}")
            except Exception as e2:
                print(f"❌ Failed to save report: {e2}")
                print("Report content (print to console only):")
                print(report_text)

def main():
    parser = argparse.ArgumentParser(description="Analyze Failure Cases")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_clean', type=str, required=True,
                        help='Path to clean test data')
    parser.add_argument('--data_noisy', type=str, required=True,
                        help='Path to noisy test data')
    parser.add_argument('--output_dir', type=str, default='failure_analysis',
                        help='Output directory')
    parser.add_argument('--num_worst', type=int, default=10,
                        help='Number of worst cases to analyze')
    parser.add_argument('--num_best', type=int, default=5,
                        help='Number of best cases to show')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # Create dataset
    print("Creating dataset...")
    dataset = XRDDataset(
        clean_dir=args.data_clean,
        noisy_dir=args.data_noisy,
        target_length=8500
    )
    print(f"Dataset initialized: {len(dataset)} pairs")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Create analyzer
    analyzer = FailureCaseAnalyzer(args.model, device=args.device)

    # Analyze failures
    worst_cases, best_cases, all_results = analyzer.analyze_failures(
        dataloader,
        num_worst=args.num_worst,
        num_best=args.num_best
    )

    # Analyze patterns
    patterns = analyzer.analyze_failure_patterns(worst_cases)

    # Plot cases
    analyzer.plot_failure_cases(worst_cases, best_cases, args.output_dir)

    # Generate report (dengan fix encoding)
    analyzer.generate_report(worst_cases, best_cases, patterns, all_results, args.output_dir)

    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()