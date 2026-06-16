"""
QUICK START GUIDE - XRD Denoising AI Training
==============================================

Panduan singkat untuk memulai training dengan cepat.
"""

# ============================================================================
# STEP 1: VERIFY ENVIRONMENT
# ============================================================================

print("STEP 1: Verify Environment")
print("-" * 80)

import torch
import os
import numpy as np
import scipy
import sklearn
import pandas as pd
import matplotlib

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Pandas version: {pd.__version__}")

print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("  WARNING: No GPU detected. Training akan sangat lambat!")
    print("  Recommended: Use Google Colab or cloud GPU")

# ============================================================================
# STEP 2: CHECK DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Check Data")
print("-" * 80)

from pathlib import Path

base_dir = Path(r"C:\Users\COMPUTER\Documents\xrdAI_withoutmatch3_v2")
clean_dir = base_dir / "data" /  "processed" / "clean"
noisy_dir = base_dir / "data" / "processed" / "noisy"

if clean_dir.exists() and noisy_dir.exists():
    clean_files = list(clean_dir.glob("*.npy"))
    noisy_files = list(noisy_dir.glob("*.npy"))
    
    print(f"✓ Clean data: {len(clean_files)} files")
    print(f"✓ Noisy data: {len(noisy_files)} files")
    
    if len(clean_files) == len(noisy_files):
        print("✓ File counts match!")
    else:
        print("✗ WARNING: File counts don't match!")
else:
    print("✗ ERROR: Data directories not found!")
    print(f"  Clean dir: {clean_dir}")
    print(f"  Noisy dir: {noisy_dir}")
    print("\n  Please complete data preparation (Steps 1-3) first!")
    exit(1)

# ============================================================================
# STEP 3: SPLIT DATASET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Split Dataset (Training/Validation)")
print("-" * 80)

train_dir = base_dir / "data/processed/train"

if not train_dir.exists():
    print("Splitting dataset...")
    print("Run: python 4_split_dataset.py")
    print("\nThis will take 5-10 minutes...")
else:
    train_clean = list((train_dir / "clean").glob("*.txt"))
    train_noisy = list((train_dir / "noisy").glob("*.txt"))
    val_clean = list((base_dir / "data/processed/val/clean").glob("*.txt"))
    val_noisy = list((base_dir / "data/processed/val/noisy").glob("*.txt"))
    
    print(f"✓ Training: {len(train_clean)} clean + {len(train_noisy)} noisy")
    print(f"✓ Validation: {len(val_clean)} clean + {len(val_noisy)} noisy")
    print("✓ Dataset split completed!")

# ============================================================================
# STEP 4: TEST DATALOADER
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Test DataLoader")
print("-" * 80)
print("Run: python xrd_dataset.py")
print("This will verify data loading works correctly...")

# ============================================================================
# STEP 5: TEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Test Model")
print("-" * 80)
print("Run: python models.py")
print("This will verify model architectures work correctly...")

# ============================================================================
# STEP 6: TEST TRAINING (5 epochs)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Test Training (Quick Test - 5 epochs)")
print("-" * 80)
print("\nCommand:")
print("  python train_denoiser.py --test_mode --epochs 5 --batch_size 8")
print("\nThis will:")
print("  - Use small subset of data")
print("  - Train for only 5 epochs")
print("  - Verify everything works")
print("\nEstimated time: 15-30 minutes")

# ============================================================================
# STEP 7: FULL TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Full Training")
print("-" * 80)



if torch.cuda.is_available():
    print("\n✓ GPU DETECTED - Recommended command:")
    print("\n  python train_denoiser.py \\")
    print("      --model unet \\")
    print("      --epochs 100 \\")
    print("      --batch_size 16 \\")
    print("      --lr 0.001 \\")
    print("      --device cuda")
    print("\nEstimated time: 8-24 hours")
else:
    print("\n✗ NO GPU DETECTED - Options:")
    print("\n  Option 1: Use SimpleCNN (faster but less accurate)")
    print("    python train_denoiser.py \\")
    print("        --model simple_cnn \\")
    print("        --epochs 50 \\")
    print("        --batch_size 4 \\")
    print("        --device cpu")
    print("    Estimated time: 3-7 DAYS!")
    
    print("\n  Option 2: Use Google Colab (FREE GPU)")
    print("    1. Go to: https://colab.research.google.com/")
    print("    2. Upload your scripts")
    print("    3. Enable GPU: Runtime > Change runtime type > GPU")
    print("    4. Run training")

# ============================================================================
# STEP 8: MONITORING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Monitor Training")
print("-" * 80)
print("\nWhile training, check:")
print("  - logs/training_curves.png (updated every epoch)")
print("  - logs/training_history.json (loss values)")
print("  - checkpoints/ (saved models)")

# ============================================================================
# STEP 9: EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Evaluate Model (After Training)")
print("-" * 80)
print("\nCommand:")
print("  python evaluate_model.py \\")
print("      --model models/saved/best_model.pth \\")
print("      --data_clean data/processed/val/clean \\")
print("      --data_noisy data/processed/val/noisy \\")
print("      --output_dir evaluation_results")

# ============================================================================
# STEP 10: INFERENCE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Use Model for Inference")
print("-" * 80)
print("\nSingle file:")
print("  python denoise_xrd.py \\")
print("      --model models/saved/best_model.pth \\")
print("      --input path/to/noisy.txt \\")
print("      --output path/to/denoised.txt")

print("\nBatch processing:")
print("  python batch_denoise.py \\")
print("      --model models/saved/best_model.pth \\")
print("      --input_folder data/tests/my_data \\")
print("      --output_folder results/denoised")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("QUICK START SUMMARY")
print("=" * 80)

print("\nTo start training NOW:")
print("\n1. Verify data is ready (steps 1-3 completed)")
print("2. Run test training:")
print("     python train_denoiser.py --test_mode --epochs 5")
print("3. If successful, run full training:")
if torch.cuda.is_available():
    print("     python train_denoiser.py --model unet --epochs 100")
else:
    print("     Use Google Colab for GPU training!")
print("4. Wait for training to complete (8-24 hours with GPU)")
print("5. Evaluate model")
print("6. Use for inference!")

print("\n" + "=" * 80)
print("GOOD LUCK! 🚀")
print("=" * 80)

# ============================================================================
# SAVE THIS INFO
# ============================================================================

output = f"""
SYSTEM INFO
-----------
PyTorch: {torch.__version__}
CUDA: {torch.cuda.is_available()}
Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

DATA INFO
---------
Clean files: {len(clean_files) if clean_dir.exists() else 'Not found'}
Noisy files: {len(noisy_files) if noisy_dir.exists() else 'Not found'}
Training ready: {train_dir.exists()}

NEXT STEPS
----------
1. python 4_split_dataset.py (if not done)
2. python train_denoiser.py --test_mode --epochs 5
3. python train_denoiser.py --model unet --epochs 100
4. python evaluate_model.py --model models/saved/best_model.pth
5. python denoise_xrd.py --model models/saved/best_model.pth --input file.txt
"""

with open('system_check.txt', 'w') as f:
    f.write(output)

print("\n✓ System info saved to: system_check.txt")
