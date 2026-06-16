"""
Evaluation Script untuk XRD Denoising Model
Menghitung metrics: SNR, PSNR, MSE, MAE, SSIM
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from models import get_model
from xrd_dataset import XRDDataset
from torch.utils.data import DataLoader


def calculate_snr(signal, noise):
    """
    Calculate Signal-to-Noise Ratio (SNR)
    SNR = 10 * log10(signal_power / noise_power)
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_psnr(original, denoised, max_value=None):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    PSNR = 10 * log10(MAX^2 / MSE)
    """
    mse = np.mean((original - denoised) ** 2)
    
    if mse == 0:
        return float('inf')
    
    if max_value is None:
        max_value = np.max(original)
    
    psnr = 10 * np.log10(max_value ** 2 / mse)
    return psnr


def calculate_ssim(signal1, signal2, data_range=None):
    """
    Calculate Structural Similarity Index (SSIM) for 1D signals
    Simplified version untuk 1D data
    """
    if data_range is None:
        data_range = max(signal1.max(), signal2.max()) - min(signal1.min(), signal2.min())
    
    # Constants
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    # Means
    mu1 = signal1.mean()
    mu2 = signal2.mean()
    
    # Variances and covariance
    sigma1_sq = signal1.var()
    sigma2_sq = signal2.var()
    sigma12 = np.cov(signal1, signal2)[0, 1]
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim = numerator / denominator
    return ssim


def inverse_preprocess(data, original_clean):
    """
    Inverse preprocessing untuk mendapatkan data asli
    Reverse: normalize -> anscombe
    """
    # Denormalize (asumsi normalized ke [0, 1])
    data_min = original_clean.min()
    data_max = original_clean.max()
    
    if data_max - data_min > 0:
        data_denorm = data * (data_max - data_min) + data_min
    else:
        data_denorm = data
    
    # Inverse Anscombe: x = (y/2)^2 - 3/8
    data_inv = (data_denorm / 2) ** 2 - 3/8
    data_inv = np.maximum(data_inv, 0)  # Ensure non-negative
    
    return data_inv


class ModelEvaluator:
    """Class untuk evaluasi model"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model file
        print(f"Loading model from {model_path}...")
        loaded = torch.load(model_path, map_location=self.device)
        
        # Inisialisasi variabel
        checkpoint = None
        config = {}
        
        # Cek apakah ini full checkpoint atau state dict langsung
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            # Full checkpoint
            checkpoint = loaded
            model_state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {}) or {}
            print("✓ Loaded as full checkpoint")
        else:
            # State dict langsung
            model_state_dict = loaded
            print("✓ Loaded as state dict directly")

        # Ambil konfigurasi model (gunakan default jika tidak ada)
        model_type = config.get('model_type', 'unet')
        base_channels = config.get('base_channels', 32)
        input_length = config.get('input_length', 8500)

        # Create model
        self.model = get_model(
            model_type=model_type,
            base_channels=base_channels,
            input_length=input_length
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        if checkpoint is not None:
            print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
            print(f"  Trained epochs: {checkpoint.get('epoch', 'N/A')}")
        else:
            print("  No checkpoint metadata available (state dict only)")
    
    def evaluate_dataset(self, dataloader, num_samples=None, save_examples=True):
        """
        Evaluate model pada dataset
        
        Returns:
        --------
        results : dict
            Dictionary berisi semua metrics
        """
        results = {
            'snr_before': [],
            'snr_after': [],
            'snr_improvement': [],
            'psnr': [],
            'mse': [],
            'mae': [],
            'ssim': []
        }
        
        examples = []
        
        print("\nEvaluating model...")
        with torch.no_grad():
            for idx, (noisy, clean) in enumerate(tqdm(dataloader, desc="Processing")):
                if num_samples is not None and idx >= num_samples:
                    break
                
                # Move to device
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Forward pass
                denoised = self.model(noisy)
                
                # Convert to numpy
                noisy_np = noisy.cpu().numpy()
                clean_np = clean.cpu().numpy()
                denoised_np = denoised.cpu().numpy()
                
                # Process each sample in batch
                batch_size = noisy_np.shape[0]
                for i in range(batch_size):
                    n = noisy_np[i, 0, :]
                    c = clean_np[i, 0, :]
                    d = denoised_np[i, 0, :]
                    
                    # Calculate noise
                    noise_before = n - c
                    noise_after = d - c
                    
                    # SNR
                    snr_before = calculate_snr(c, noise_before)
                    snr_after = calculate_snr(c, noise_after)
                    snr_improvement = snr_after - snr_before
                    
                    # PSNR
                    psnr = calculate_psnr(c, d)
                    
                    # MSE and MAE
                    mse = np.mean((c - d) ** 2)
                    mae = np.mean(np.abs(c - d))
                    
                    # SSIM
                    ssim = calculate_ssim(c, d)
                    
                    # Store results
                    results['snr_before'].append(snr_before)
                    results['snr_after'].append(snr_after)
                    results['snr_improvement'].append(snr_improvement)
                    results['psnr'].append(psnr)
                    results['mse'].append(mse)
                    results['mae'].append(mae)
                    results['ssim'].append(ssim)
                    
                    # Save examples
                    if save_examples and len(examples) < 20:
                        examples.append({
                            'noisy': n,
                            'clean': c,
                            'denoised': d,
                            'snr_improvement': snr_improvement
                        })
        
        # Calculate statistics
        stats = {}
        for key in results:
            values = np.array(results[key])
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return results, stats, examples
    
    def print_statistics(self, stats):
        """Print evaluation statistics"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print("\nSNR (Signal-to-Noise Ratio):")
        print(f"  Before:  {stats['snr_before']['mean']:.2f} ± {stats['snr_before']['std']:.2f} dB")
        print(f"  After:   {stats['snr_after']['mean']:.2f} ± {stats['snr_after']['std']:.2f} dB")
        print(f"  Improvement: {stats['snr_improvement']['mean']:.2f} ± {stats['snr_improvement']['std']:.2f} dB")
        
        print("\nPSNR (Peak Signal-to-Noise Ratio):")
        print(f"  Mean: {stats['psnr']['mean']:.2f} ± {stats['psnr']['std']:.2f} dB")
        
        print("\nMSE (Mean Squared Error):")
        print(f"  Mean: {stats['mse']['mean']:.6f} ± {stats['mse']['std']:.6f}")
        
        print("\nMAE (Mean Absolute Error):")
        print(f"  Mean: {stats['mae']['mean']:.6f} ± {stats['mae']['std']:.6f}")
        
        print("\nSSIM (Structural Similarity):")
        print(f"  Mean: {stats['ssim']['mean']:.4f} ± {stats['ssim']['std']:.4f}")
        
        print("\n" + "="*80)
    
    def save_results(self, results, stats, output_dir):
        """Save results to CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        df = pd.DataFrame(results)
        results_path = output_dir / "detailed_results3b.csv"
        df.to_csv(results_path, index=False)
        print(f"\n✓ Detailed results saved: {results_path}")
        
        # Save statistics
        stats_df = pd.DataFrame(stats).T
        stats_path = output_dir / "statistics3b.csv"
        stats_df.to_csv(stats_path)
        print(f"✓ Statistics saved: {stats_path}")
    
    def plot_examples(self, examples, output_dir, num_examples=10):
        """Plot example denoising results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sort by SNR improvement (best and worst)
        examples_sorted = sorted(examples, key=lambda x: x['snr_improvement'], reverse=True)
        
        # Select best and worst
        best_examples = examples_sorted[:num_examples//2]
        worst_examples = examples_sorted[-num_examples//2:]
        selected_examples = best_examples + worst_examples
        
        # Plot
        fig, axes = plt.subplots(num_examples, 1, figsize=(15, 3*num_examples))
        
        if num_examples == 1:
            axes = [axes]
        
        for idx, example in enumerate(selected_examples):
            ax = axes[idx]
            
            x = np.arange(len(example['noisy']))
            
            ax.plot(x, example['noisy'], 'gray', alpha=0.5, label='Noisy', linewidth=1)
            ax.plot(x, example['clean'], 'g', label='Clean (Ground Truth)', linewidth=1.5)
            ax.plot(x, example['denoised'], 'r', label='Denoised (Model)', linewidth=1.5)
            
            ax.set_title(f"SNR Improvement: {example['snr_improvement']:.2f} dB", 
                        fontweight='bold', fontsize=12)
            ax.set_xlabel("Position")
            ax.set_ylabel("Intensity (normalized)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_dir / "denoising_examples3b.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Example plots saved: {plot_path}")
    
    def plot_distributions(self, results, output_dir):
        """Plot distributions of metrics"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics = [
            ('snr_improvement', 'SNR Improvement (dB)'),
            ('psnr', 'PSNR (dB)'),
            ('mse', 'MSE'),
            ('mae', 'MAE'),
            ('ssim', 'SSIM'),
            ('snr_after', 'SNR After Denoising (dB)')
        ]
        
        for idx, (key, label) in enumerate(metrics):
            ax = axes[idx]
            data = results[key]
            
            ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(data):.2f}')
            ax.axvline(np.median(data), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(data):.2f}')
            
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Distribution of {label}', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_dir / "metric_distributions3b.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Distribution plots saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate XRD Denoising Model")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_clean', type=str, required=True, help='Path to clean data')
    parser.add_argument('--data_noisy', type=str, required=True, help='Path to noisy data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Create dataset
    print("Creating dataset...")
    dataset = XRDDataset(
        clean_dir=args.data_clean,
        noisy_dir=args.data_noisy,
        target_length=8500
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model, device=args.device)
    
    # Evaluate
    results, stats, examples = evaluator.evaluate_dataset(
        dataloader,
        num_samples=args.num_samples,
        save_examples=True
    )
    
    # Print statistics
    evaluator.print_statistics(stats)
    
    # Save results
    evaluator.save_results(results, stats, args.output_dir)
    
    # Plot examples and distributions
    evaluator.plot_examples(examples, args.output_dir, num_examples=10)
    evaluator.plot_distributions(results, args.output_dir)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
