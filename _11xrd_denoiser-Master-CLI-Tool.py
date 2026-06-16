"""
XRD Denoiser - Master CLI Tool
Unified command-line interface untuk semua operasi
"""

import argparse
import sys
from pathlib import Path


def print_header():
    """Print header"""
    print("\n" + "="*80)
    print(" "*25 + "XRD DENOISING AI")
    print(" "*20 + "Master Command Line Tool")
    print("="*80 + "\n")


def main():
    print_header()
    
    parser = argparse.ArgumentParser(
        description="XRD Denoising AI - Master Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split dataset
  python _7c_xrd_denoiser.py split --clean_dir data/processed/clean --noisy_dir data/processed/noisy
  
  # Train model
  python _7c_xrd_denoiser.py train --model unet --epochs 100 --batch_size 16
  
  # Evaluate model
  python _7c_xrd_denoiser.py evaluate --model models/saved/best_model.pth --data_clean data/processed/val/clean --data_noisy data/processed/val/noisy
  
  # Denoise single file
  python _7c_xrd_denoiser.py denoise --model models/saved/best_model.pth --input file.txt --output denoised.txt
  
  # Batch processing
  python _7c_xrd_denoiser.py batch --model models/saved/best_model.pth --input_folder data/tests --output_folder results
  
  # Analyze failures
  python _7c_xrd_denoiser.py analyze --model models/saved/best_model.pth --data_clean data/processed/val/clean --data_noisy data/processed/val/noisy
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ============================================================================
    # COMMAND: split
    # ============================================================================
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val')
    split_parser.add_argument('--clean_dir', type=str, required=True,
                             help='Path to clean data directory')
    split_parser.add_argument('--noisy_dir', type=str, required=True,
                             help='Path to noisy data directory')
    split_parser.add_argument('--output_dir', type=str, default='data/processed',
                             help='Output directory')
    split_parser.add_argument('--train_ratio', type=float, default=0.8,
                             help='Training data ratio (default: 0.8)')
    split_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    # ============================================================================
    # COMMAND: train
    # ============================================================================
    train_parser = subparsers.add_parser('train', help='Train denoising model')
    train_parser.add_argument('--model', type=str, default='unet',
                             choices=['unet', 'simple_cnn'],
                             help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=16,
                             help='Batch size')
    train_parser.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--device', type=str, default='cuda',
                             choices=['cuda', 'cpu'],
                             help='Device to use')
    train_parser.add_argument('--test_mode', action='store_true',
                             help='Enable test mode (small dataset, 5 epochs)')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Resume from checkpoint')
    
    # ============================================================================
    # COMMAND: evaluate
    # ============================================================================
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--data_clean', type=str, required=True,
                            help='Path to clean test data')
    eval_parser.add_argument('--data_noisy', type=str, required=True,
                            help='Path to noisy test data')
    eval_parser.add_argument('--output_dir', type=str, default='evaluation_results',
                            help='Output directory')
    eval_parser.add_argument('--batch_size', type=int, default=16,
                            help='Batch size')
    eval_parser.add_argument('--num_samples', type=int, default=None,
                            help='Number of samples to evaluate (None = all)')
    eval_parser.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'])
    
    # ============================================================================
    # COMMAND: denoise
    # ============================================================================
    denoise_parser = subparsers.add_parser('denoise', help='Denoise single XRD file')
    denoise_parser.add_argument('--model', type=str, required=True,
                               help='Path to model checkpoint')
    denoise_parser.add_argument('--input', type=str, required=True,
                               help='Path to input noisy file')
    denoise_parser.add_argument('--output', type=str, default=None,
                               help='Path to save denoised file')
    denoise_parser.add_argument('--plot', action='store_true', default=True,
                               help='Create comparison plot')
    denoise_parser.add_argument('--device', type=str, default='cuda',
                               choices=['cuda', 'cpu'])
    
    # ============================================================================
    # COMMAND: batch
    # ============================================================================
    batch_parser = subparsers.add_parser('batch', help='Batch denoise multiple files')
    batch_parser.add_argument('--model', type=str, required=True,
                             help='Path to model checkpoint')
    batch_parser.add_argument('--input_folder', type=str, required=True,
                             help='Path to folder containing noisy files')
    batch_parser.add_argument('--output_folder', type=str, required=True,
                             help='Path to save denoised files')
    batch_parser.add_argument('--pattern', type=str, default='*.txt',
                             help='File pattern to match')
    batch_parser.add_argument('--plots', action='store_true',
                             help='Create comparison plots')
    batch_parser.add_argument('--max_files', type=int, default=None,
                             help='Maximum number of files to process')
    batch_parser.add_argument('--device', type=str, default='cuda',
                             choices=['cuda', 'cpu'])
    
    # ============================================================================
    # COMMAND: analyze
    # ============================================================================
    analyze_parser = subparsers.add_parser('analyze', help='Analyze failure cases')
    analyze_parser.add_argument('--model', type=str, required=True,
                               help='Path to model checkpoint')
    analyze_parser.add_argument('--data_clean', type=str, required=True,
                               help='Path to clean test data')
    analyze_parser.add_argument('--data_noisy', type=str, required=True,
                               help='Path to noisy test data')
    analyze_parser.add_argument('--output_dir', type=str, default='failure_analysis',
                               help='Output directory')
    analyze_parser.add_argument('--num_worst', type=int, default=10,
                               help='Number of worst cases')
    analyze_parser.add_argument('--batch_size', type=int, default=16)
    analyze_parser.add_argument('--device', type=str, default='cuda',
                               choices=['cuda', 'cpu'])
    
    # ============================================================================
    # COMMAND: info
    # ============================================================================
    info_parser = subparsers.add_parser('info', help='Show system and model info')
    info_parser.add_argument('--model', type=str, default=None,
                            help='Path to model checkpoint (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # ============================================================================
    # Execute commands
    # ============================================================================
    
    if args.command == 'split':
        print("Running dataset split...")
        from _4split_dataset import split_dataset
        split_dataset(
            clean_dir=args.clean_dir,
            noisy_dir=args.noisy_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            verbose=True
        )
    
    elif args.command == 'train':
        print("Starting training...")
        import _7a_train_denoiser
        sys.argv = ['train_denoiser.py']
        if args.model:
            sys.argv.extend(['--model', args.model])
        if args.epochs:
            sys.argv.extend(['--epochs', str(args.epochs)])
        if args.batch_size:
            sys.argv.extend(['--batch_size', str(args.batch_size)])
        if args.lr:
            sys.argv.extend(['--lr', str(args.lr)])
        if args.device:
            sys.argv.extend(['--device', args.device])
        if args.test_mode:
            sys.argv.append('--test_mode')
        if args.resume:
            sys.argv.extend(['--resume', args.resume])
        
        _7a_train_denoiser.main()
    
    elif args.command == 'evaluate':
        print("Starting evaluation...")
        import _8evaluate_model
        sys.argv = [
            '_8evaluate_model.py',
            '--model', args.model,
            '--data_clean', args.data_clean,
            '--data_noisy', args.data_noisy,
            '--output_dir', args.output_dir,
            '--batch_size', str(args.batch_size),
            '--device', args.device
        ]
        if args.num_samples:
            sys.argv.extend(['--num_samples', str(args.num_samples)])
        
        _8evaluate_model.main()
    
    elif args.command == 'denoise':
        print("Denoising file...")
        import _10a_denoise_xrd
        sys.argv = [
            '_10a_denoise_xrd.py',
            '--model', args.model,
            '--input', args.input,
            '--device', args.device
        ]
        if args.output:
            sys.argv.extend(['--output', args.output])
        if args.plot:
            sys.argv.append('--plot')
        
        _10a_denoise_xrd.main()
    
    elif args.command == 'batch':
        print("Starting batch processing...")
        import _10b_batch_denoise
        sys.argv = [
            '_10b_batch_denoise.py',
            '--model', args.model,
            '--input_folder', args.input_folder,
            '--output_folder', args.output_folder,
            '--pattern', args.pattern,
            '--device', args.device
        ]
        if args.plots:
            sys.argv.append('--plots')
        if args.max_files:
            sys.argv.extend(['--max_files', str(args.max_files)])
        
        _10b_batch_denoise.main()
    
    elif args.command == 'analyze':
        print("Analyzing failure cases...")
        import analyze_failures
        sys.argv = [
            'analyze_failures.py',
            '--model', args.model,
            '--data_clean', args.data_clean,
            '--data_noisy', args.data_noisy,
            '--output_dir', args.output_dir,
            '--num_worst', str(args.num_worst),
            '--batch_size', str(args.batch_size),
            '--device', args.device
        ]
        
        analyze_failures.main()
    
    elif args.command == 'info':
        print("System Information")
        print("="*80)
        
        import torch
        import platform
        
        print(f"\nPython: {platform.python_version()}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        if args.model:
            print(f"\nModel Information")
            print("-"*80)
            try:
                checkpoint = torch.load(args.model, map_location='cpu')
                print(f"Model type: {checkpoint.get('config', {}).get('model_type', 'N/A')}")
                print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
                print(f"Trained epochs: {checkpoint.get('epoch', 'N/A')}")
                print(f"Training loss: {checkpoint.get('train_losses', ['N/A'])[-1]}")
                print(f"Validation loss: {checkpoint.get('val_losses', ['N/A'])[-1]}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()
