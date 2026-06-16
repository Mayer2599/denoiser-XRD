"""
Batch Processing Script untuk Denoise Multiple XRD Files
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import argparse

from denoise_xrd import XRDDenoiser


class BatchDenoiser:
    """Class untuk batch denoising"""
    
    def __init__(self, model_path, device='cuda'):
        self.denoiser = XRDDenoiser(model_path, device=device)
        self.results = []
    
    def process_folder(self, input_folder, output_folder, pattern="*.txt", 
                      create_plots=False, max_files=None):
        """
        Process all files in a folder
        
        Parameters:
        -----------
        input_folder : str
            Path to folder containing noisy files
        output_folder : str
            Path to save denoised files
        pattern : str
            File pattern to match (default: "*.txt")
        create_plots : bool
            Whether to create comparison plots
        max_files : int, optional
            Maximum number of files to process
        
        Returns:
        --------
        results_df : pd.DataFrame
            DataFrame containing processing results
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Get list of files
        files = sorted(input_folder.glob(pattern))
        
        if max_files is not None:
            files = files[:max_files]
        
        print("="*80)
        print("BATCH DENOISING")
        print("="*80)
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Total files: {len(files)}")
        print(f"Create plots: {create_plots}")
        print("="*80)
        
        if len(files) == 0:
            print(f"No files found matching pattern: {pattern}")
            return None
        
        # Process files
        total_start_time = time.time()
        
        for idx, input_path in enumerate(tqdm(files, desc="Processing files"), 1):
            try:
                # Generate output path
                output_path = output_folder / input_path.name
                
                # Process file
                start_time = time.time()
                
                angles, noisy, denoised = self.denoiser.denoise_file(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    plot=create_plots
                )
                
                processing_time = time.time() - start_time
                
                # Calculate metrics
                noise_std_before = np.std(noisy)
                noise_std_after = np.std(noisy - denoised)
                noise_reduction = (noise_std_before - noise_std_after) / noise_std_before * 100
                
                # Calculate SNR improvement (approximate)
                signal_power = np.mean(denoised ** 2)
                noise_power_before = noise_std_before ** 2
                noise_power_after = noise_std_after ** 2
                
                if noise_power_before > 0 and noise_power_after > 0:
                    snr_before = 10 * np.log10(signal_power / noise_power_before)
                    snr_after = 10 * np.log10(signal_power / noise_power_after)
                    snr_improvement = snr_after - snr_before
                else:
                    snr_before = snr_after = snr_improvement = 0
                
                # Store results
                result = {
                    'file': input_path.name,
                    'data_points': len(noisy),
                    'processing_time': processing_time,
                    'noise_reduction_percent': noise_reduction,
                    'snr_before': snr_before,
                    'snr_after': snr_after,
                    'snr_improvement': snr_improvement,
                    'output_path': str(output_path),
                    'status': 'success'
                }
                
                self.results.append(result)
                
            except Exception as e:
                print(f"\nError processing {input_path.name}: {e}")
                
                result = {
                    'file': input_path.name,
                    'status': 'failed',
                    'error': str(e)
                }
                
                self.results.append(result)
        
        # Calculate total time
        total_time = time.time() - total_start_time
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Print summary
        self._print_summary(results_df, total_time)
        
        # Save results
        self._save_results(results_df, output_folder)
        
        return results_df
    
    def _print_summary(self, results_df, total_time):
        """Print processing summary"""
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        
        total_files = len(results_df)
        successful = len(results_df[results_df['status'] == 'success'])
        failed = total_files - successful
        
        print(f"\nTotal files: {total_files}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        if successful > 0:
            success_df = results_df[results_df['status'] == 'success']
            
            avg_time = success_df['processing_time'].mean()
            avg_noise_reduction = success_df['noise_reduction_percent'].mean()
            avg_snr_improvement = success_df['snr_improvement'].mean()
            
            print(f"\nAverage processing time: {avg_time:.2f}s per file")
            print(f"Average noise reduction: {avg_noise_reduction:.2f}%")
            print(f"Average SNR improvement: {avg_snr_improvement:.2f} dB")
            
            print(f"\nSNR Statistics:")
            print(f"  Before:  {success_df['snr_before'].mean():.2f} ± {success_df['snr_before'].std():.2f} dB")
            print(f"  After:   {success_df['snr_after'].mean():.2f} ± {success_df['snr_after'].std():.2f} dB")
            print(f"  Improvement: {avg_snr_improvement:.2f} ± {success_df['snr_improvement'].std():.2f} dB")
        
        if failed > 0:
            print(f"\nFailed files:")
            failed_df = results_df[results_df['status'] == 'failed']
            for _, row in failed_df.iterrows():
                print(f"  - {row['file']}: {row.get('error', 'Unknown error')}")
        
        print("="*80)
    
    def _save_results(self, results_df, output_folder):
        """Save processing results to CSV"""
        results_path = output_folder / "batch_processing_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to: {results_path}")
        
        # Save summary statistics
        if len(results_df[results_df['status'] == 'success']) > 0:
            success_df = results_df[results_df['status'] == 'success']
            
            summary = {
                'Total Files': len(results_df),
                'Successful': len(success_df),
                'Failed': len(results_df) - len(success_df),
                'Avg Processing Time (s)': success_df['processing_time'].mean(),
                'Avg Noise Reduction (%)': success_df['noise_reduction_percent'].mean(),
                'Avg SNR Before (dB)': success_df['snr_before'].mean(),
                'Avg SNR After (dB)': success_df['snr_after'].mean(),
                'Avg SNR Improvement (dB)': success_df['snr_improvement'].mean(),
                'Min SNR Improvement (dB)': success_df['snr_improvement'].min(),
                'Max SNR Improvement (dB)': success_df['snr_improvement'].max(),
            }
            
            summary_df = pd.DataFrame([summary]).T
            summary_df.columns = ['Value']
            
            summary_path = output_folder / "summary_statistics.csv"
            summary_df.to_csv(summary_path)
            print(f"✓ Summary statistics saved to: {summary_path}")
    
    def plot_results_distribution(self, results_df, output_folder):
        """Plot distribution of results"""
        import matplotlib.pyplot as plt
        
        success_df = results_df[results_df['status'] == 'success']
        
        if len(success_df) == 0:
            print("No successful results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: SNR Improvement distribution
        ax1 = axes[0, 0]
        ax1.hist(success_df['snr_improvement'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(success_df['snr_improvement'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {success_df['snr_improvement'].mean():.2f} dB")
        ax1.set_xlabel('SNR Improvement (dB)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of SNR Improvement', fontweight='bold', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Processing time distribution
        ax2 = axes[0, 1]
        ax2.hist(success_df['processing_time'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(success_df['processing_time'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {success_df['processing_time'].mean():.2f}s")
        ax2.set_xlabel('Processing Time (seconds)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Processing Time', fontweight='bold', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Noise reduction distribution
        ax3 = axes[1, 0]
        ax3.hist(success_df['noise_reduction_percent'], bins=30, alpha=0.7, 
                color='orange', edgecolor='black')
        ax3.axvline(success_df['noise_reduction_percent'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {success_df['noise_reduction_percent'].mean():.2f}%")
        ax3.set_xlabel('Noise Reduction (%)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Distribution of Noise Reduction', fontweight='bold', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: SNR Before vs After
        ax4 = axes[1, 1]
        ax4.scatter(success_df['snr_before'], success_df['snr_after'], alpha=0.6, s=30)
        
        # Add diagonal line (y = x)
        min_val = min(success_df['snr_before'].min(), success_df['snr_after'].min())
        max_val = max(success_df['snr_before'].max(), success_df['snr_after'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='No change')
        
        ax4.set_xlabel('SNR Before (dB)', fontsize=12)
        ax4.set_ylabel('SNR After (dB)', fontsize=12)
        ax4.set_title('SNR: Before vs After Denoising', fontweight='bold', fontsize=13)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_folder / "batch_results_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Distribution plots saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch Denoise XRD Files")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Path to folder containing noisy files')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Path to save denoised files')
    parser.add_argument('--pattern', type=str, default='*.txt',
                       help='File pattern to match (default: *.txt)')
    parser.add_argument('--plots', action='store_true',
                       help='Create comparison plots for each file')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Create batch denoiser
    batch_denoiser = BatchDenoiser(args.model, device=args.device)
    
    # Process folder
    results_df = batch_denoiser.process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        pattern=args.pattern,
        create_plots=args.plots,
        max_files=args.max_files
    )
    
    # Plot results distribution
    if results_df is not None:
        batch_denoiser.plot_results_distribution(results_df, Path(args.output_folder))
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
