"""
Inference Script untuk Denoise Single XRD File dengan GUI File Picker
Load model dan denoise input file eksperimen
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys
from scipy.interpolate import interp1d
from models import get_model

# Import tkinter untuk GUI file picker (built-in Python)
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available. GUI mode disabled. Install Python with Tk support.")

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio (SNR)"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def select_file_via_gui(mode='input'):
    """
    Buka dialog file explorer untuk memilih file/folder
    
    Parameters:
    -----------
    mode : str
        'input' = pilih file XRD, 'output_dir' = pilih folder output
    
    Returns:
    --------
    str or None : Path file/folder yang dipilih, atau None jika dibatalkan
    """
    if not TKINTER_AVAILABLE:
        print("Error: tkinter not available. Cannot open GUI file picker.")
        return None
    
    root = tk.Tk()
    root.withdraw()  # Sembunyikan window utama
    root.attributes('-topmost', True)  # Pastikan dialog di atas aplikasi lain
    
    try:
        if mode == 'input':
            file_path = filedialog.askopenfilename(
                title="Pilih File XRD untuk Denoising",
                filetypes=[
                    ("XRD Files", "*.txt *.xy *.ASC *.asc *.dat"),
                    ("Text Files", "*.txt *.xy"),
                    ("All Files", "*.*")
                ]
            )
        elif mode == 'output_dir':
            file_path = filedialog.askdirectory(
                title="Pilih Folder untuk Hasil Denoising"
            )
        else:
            file_path = None
        
        return file_path if file_path else None
    except Exception as e:
        print(f"Error saat membuka dialog file: {e}")
        return None
    finally:
        root.destroy()

class XRDDenoiser:
    """Class untuk denoise XRD data eksperimen"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        loaded = torch.load(model_path, map_location=self.device)
        
        checkpoint = None
        config = {}
        model_state_dict = None
        
        # Handle multiple checkpoint formats
        if isinstance(loaded, dict):
            # Format 1: PyTorch Lightning / default trainer (state_dict)
            if 'state_dict' in loaded:
                checkpoint = loaded
                model_state_dict = checkpoint['state_dict']
                config = checkpoint.get('config', {}) or {}
                print("✓ Loaded as PyTorch Lightning/default trainer checkpoint format")
            
            # Format 2: Custom checkpoint (model_state_dict)
            elif 'model_state_dict' in loaded:
                checkpoint = loaded
                model_state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('config', {}) or {}
                print("✓ Loaded as custom checkpoint format")
            
            # Format 3: Raw state dict
            else:
                model_state_dict = loaded
                print("✓ Loaded as raw state dict")
        else:
            model_state_dict = loaded
            print("✓ Loaded as raw state dict")

        if model_state_dict is None:
            raise RuntimeError("Could not extract model state dict from checkpoint")

        # Handle 'model.' prefix (common in Lightning)
        if isinstance(model_state_dict, dict):
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v  # Remove 'model.' prefix
                else:
                    new_state_dict[k] = v
            model_state_dict = new_state_dict
            print("✓ Removed 'model.' prefix from state dict keys")

        # Ambil konfigurasi model
        self.model_type = config.get('model_type', 'unet')
        base_channels = config.get('base_channels', 32)
        self.input_length = config.get('input_length', 8500)
        
        # Create model
        self.model = get_model(
            model_type=self.model_type,
            base_channels=base_channels,
            input_length=self.input_length
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        
        print(f"✓ Model loaded successfully: {self.model.__class__.__name__}")
        print(f"  Input length: {self.input_length} points")
        print(f"  Device: {self.device}")
        if checkpoint is not None:
            if 'best_val_loss' in checkpoint:
                print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")
            if 'epoch' in checkpoint:
                print(f"  Trained epochs: {checkpoint['epoch']}")

    def load_xrd_file(self, filepath):
        """Load XRD file (supports .txt, .xy, .ASC with 2 columns: angle, intensity)"""
        filepath = Path(filepath)
        try:
            # Coba load dengan delimiter spasi/tab/koma
            data = np.loadtxt(filepath, comments=['#', ';', '@', '/*', 'Peak'])
        except Exception as e1:
            try:
                # Fallback: load dengan delimiter fleksibel
                with open(filepath, 'r') as f:
                    lines = [line.strip() for line in f 
                            if line.strip() 
                            and not line.startswith(('#', ';', '@', '/*')) 
                            and 'Peak' not in line
                            and 'Position' not in line
                            and 'Intensity' not in line]
                data = np.array([list(map(float, line.split())) for line in lines])
            except Exception as e2:
                raise ValueError(f"Failed to load XRD file:\n  Method 1: {e1}\n  Method 2: {e2}")
        
        if data.ndim == 2:
            if data.shape[1] == 2:
                angles = data[:, 0]
                intensity = data[:, 1]
            elif data.shape[1] > 2:
                # Ambil kolom pertama sebagai angle, terakhir sebagai intensity
                angles = data[:, 0]
                intensity = data[:, -1]
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
        else:
            # Single column (intensity only)
            intensity = data
            angles = np.arange(len(intensity))
        
        print(f"  Loaded {len(intensity)} data points")
        return angles, intensity

    def resample(self, data, target_length):
        """Resample data to target length using linear interpolation"""
        if len(data) == target_length:
            return data
        
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, target_length)
        
        # Handle NaN/Inf
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        f = interp1d(x_old, data_clean, kind='linear', fill_value='extrapolate')
        data_resampled = f(x_new)
        
        return data_resampled

    def preprocess(self, data):
        """
        Preprocessing pipeline:
        1. Anscombe transform (stabilisasi variance untuk Poisson noise)
        2. Normalization ke [0, 1]
        """
        # Simpan statistik asli untuk inverse transform
        data_min_orig = data.min()
        data_max_orig = data.max()
        
        # Anscombe transform: y = 2 * sqrt(x + 3/8)
        data_safe = np.maximum(data, 0)
        data_anscombe = 2 * np.sqrt(data_safe + 3/8)
        
        # Normalization
        data_min = data_anscombe.min()
        data_max = data_anscombe.max()
        
        if data_max - data_min > 0:
            data_normalized = (data_anscombe - data_min) / (data_max - data_min)
        else:
            data_normalized = data_anscombe
        
        # Simpan parameter untuk inverse transform
        preprocessing_params = {
            'anscombe_min': data_min,
            'anscombe_max': data_max,
            'orig_min': data_min_orig,
            'orig_max': data_max_orig
        }
        
        return data_normalized, preprocessing_params

    def inverse_preprocess(self, data, params):
        """
        Inverse preprocessing:
        1. Denormalization
        2. Inverse Anscombe: x = (y/2)^2 - 3/8
        """
        # Denormalization
        data_min = params['anscombe_min']
        data_max = params['anscombe_max']
        
        if data_max - data_min > 0:
            data_denorm = data * (data_max - data_min) + data_min
        else:
            data_denorm = data
        
        # Inverse Anscombe
        data_inv = (data_denorm / 2) ** 2 - 3/8
        data_inv = np.maximum(data_inv, 0)  # Ensure non-negative
        
        return data_inv

    def denoise(self, noisy_data):
        """
        Denoise XRD data
        
        Parameters:
        -----------
        noisy_data : np.ndarray
            Noisy XRD intensity data
        
        Returns:
        --------
        denoised_data : np.ndarray
            Denoised intensity data
        preprocessing_params : dict
            Parameters used for preprocessing
        """
        # Simpan panjang asli
        original_length = len(noisy_data)
        
        # Resample ke panjang input model
        if original_length != self.input_length:
            noisy_resampled = self.resample(noisy_data, self.input_length)
        else:
            noisy_resampled = noisy_data.copy()
        
        # Preprocess
        noisy_preprocessed, preprocessing_params = self.preprocess(noisy_resampled)
        
        # Convert ke tensor [batch=1, channel=1, length]
        noisy_tensor = torch.FloatTensor(noisy_preprocessed).unsqueeze(0).unsqueeze(0)
        noisy_tensor = noisy_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            denoised_tensor = self.model(noisy_tensor)
        
        # Convert kembali ke numpy
        denoised_preprocessed = denoised_tensor.squeeze().cpu().numpy()
        
        # Inverse preprocessing
        denoised_data = self.inverse_preprocess(denoised_preprocessed, preprocessing_params)
        
        # Resample kembali ke panjang asli
        if original_length != self.input_length:
            denoised_data = self.resample(denoised_data, original_length)
        
        return denoised_data, preprocessing_params

    def denoise_file(self, input_path, output_path=None, plot=True):
        """
        Denoise XRD file eksperimen
        
        Parameters:
        -----------
        input_path : str
            Path ke file noisy (.txt, .xy, .ASC)
        output_path : str, optional
            Path untuk menyimpan hasil denoised
        plot : bool
            Buat plot perbandingan
        
        Returns:
        --------
        angles : np.ndarray
            Sudut 2θ
        noisy_intensity : np.ndarray
            Intensitas asli (noisy)
        denoised_intensity : np.ndarray
            Intensitas setelah denoising
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING XRD FILE: {Path(input_path).name}")
        print(f"{'='*80}")
        
        # Load file
        angles, noisy_intensity = self.load_xrd_file(input_path)
        
        # Denoise
        print("  Denoising with UNet1D model...")
        denoised_intensity, _ = self.denoise(noisy_intensity)
        
        # Hitung metrik kualitas
        noise_estimate = noisy_intensity - denoised_intensity
        snr_improvement = calculate_snr(denoised_intensity, noise_estimate)
        noise_before = np.std(noisy_intensity)
        noise_after = np.std(noise_estimate)
        noise_reduction_pct = (noise_before - noise_after) / noise_before * 100 if noise_before > 0 else 0
        
        print(f"  ✓ Denoising completed")
        print(f"  SNR improvement: {snr_improvement:.2f} dB")
        print(f"  Noise reduction: {noise_reduction_pct:.2f}%")
        
        # Simpan output
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Format: 2 kolom (2θ, Intensity)
            output_data = np.column_stack([angles, denoised_intensity])
            np.savetxt(output_path, output_data, fmt='%.6f', delimiter='\t',
                      header='2Theta\tIntensity', comments='')
            
            print(f"  ✓ Saved denoised file: {output_path}")
        
        # Buat plot
        if plot:
            plot_path = self._create_comparison_plot(
                angles, noisy_intensity, denoised_intensity, 
                input_path, output_path
            )
            print(f"  ✓ Saved comparison plot: {plot_path}")
        
        return angles, noisy_intensity, denoised_intensity

    def _create_comparison_plot(self, angles, noisy, denoised, input_path, output_path):
        """Buat plot perbandingan before/after + residual noise"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: Perbandingan spektrum
        ax1 = axes[0]
        ax1.plot(angles, noisy, 'gray', alpha=0.6, label='Original (Noisy)', linewidth=1.2)
        ax1.plot(angles, denoised, 'r', label='Denoised (AI)', linewidth=1.8)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=12, fontweight='bold')
        ax1.set_title(f'XRD Denoising Result: {Path(input_path).name}', 
                     fontweight='bold', fontsize=14)
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Zoom pada area dengan peak
        ax2 = axes[1]
        # Cari area dengan intensitas tinggi (peak)
        peak_mask = denoised > np.percentile(denoised, 70)
        if np.any(peak_mask):
            peak_indices = np.where(peak_mask)[0]
            start_idx = max(0, peak_indices[0] - 50)
            end_idx = min(len(angles), peak_indices[-1] + 50)
            ax2.plot(angles[start_idx:end_idx], noisy[start_idx:end_idx], 
                    'gray', alpha=0.6, label='Original', linewidth=1.2)
            ax2.plot(angles[start_idx:end_idx], denoised[start_idx:end_idx], 
                    'r', label='Denoised', linewidth=1.8)
            ax2.set_ylabel('Intensity (a.u.)', fontsize=12, fontweight='bold')
            ax2.set_title('Zoomed View (Peak Region)', fontweight='bold', fontsize=13)
            ax2.legend(loc='upper right', fontsize=11)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No significant peaks detected for zoom view', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_axis_off()
        
        # Plot 3: Residual noise (yang dihapus oleh AI)
        ax3 = axes[2]
        residual = noisy - denoised
        ax3.plot(angles, residual, 'b', alpha=0.7, linewidth=1)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.7)
        ax3.fill_between(angles, residual, 0, where=(residual > 0), 
                         alpha=0.3, color='green', label='Positive residual')
        ax3.fill_between(angles, residual, 0, where=(residual < 0), 
                         alpha=0.3, color='red', label='Negative residual')
        ax3.set_xlabel('2θ (degrees)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Residual (Noisy - Denoised)', fontsize=12, fontweight='bold')
        ax3.set_title('Removed Noise Components', fontweight='bold', fontsize=13)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Simpan plot
        if output_path is not None:
            plot_path = Path(output_path).parent / (Path(output_path).stem + '_comparison.png')
        else:
            plot_path = Path(input_path).parent / (Path(input_path).stem + '_denoised_comparison.png')
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Denoise Experimental XRD Data using Trained AI Model",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (supports .pth, .ckpt)')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to experimental XRD file (.txt, .xy, .ASC)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save denoised file (default: <input>_denoised.txt)')
    parser.add_argument('--gui', action='store_true',
                        help='Open Windows Explorer to select input file interactively')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating comparison plot')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Mode GUI: buka dialog pemilihan file
    if args.gui:
        if not TKINTER_AVAILABLE:
            print("Error: tkinter not available. Install Python with Tk support to use --gui mode.")
            sys.exit(1)
        
        print("Opening file picker dialog...")
        input_path = select_file_via_gui(mode='input')
        
        if not input_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        args.input = input_path
        print(f"Selected file: {args.input}")
        
        # Jika output tidak ditentukan, generate otomatis di folder yang sama
        if args.output is None:
            input_path_obj = Path(args.input)
            args.output = str(input_path_obj.parent / f"{input_path_obj.stem}_denoised.txt")
            print(f"Output will be saved to: {args.output}")
    
    # Validasi input wajib
    if not args.input:
        print("Error: --input file path is required (or use --gui to select interactively)")
        parser.print_help()
        sys.exit(1)
    
    # Default output path jika tidak ditentukan (non-GUI mode)
    if args.output is None and not args.gui:
        input_path = Path(args.input)
        args.output = str(input_path.parent / (input_path.stem + '_denoised.txt'))

    # Create denoiser
    try:
        denoiser = XRDDenoiser(args.model, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Denoise file
    try:
        angles, noisy, denoised = denoiser.denoise_file(
            input_path=args.input,
            output_path=args.output,
            plot=not args.no_plot
        )
    except Exception as e:
        print(f"Error during denoising: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print("DENOISING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Input file : {args.input}")
    print(f"Output file: {args.output}")
    print(f"Device     : {denoiser.device}")
    print("="*80)


if __name__ == "__main__":
    main()