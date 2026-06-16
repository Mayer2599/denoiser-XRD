"""
7_train_denoiser.py
Training script untuk XRD denoising model (UNet-based)
Menggunakan data .npy dengan struktur:
- train/clean/clean_XXXXXX.npy
- train/noisy/noisy_XXXXXX.npy
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import shutil
import json
import time

# Import modul lokal (pastikan models.py dan xrd_dataset.py ada di folder yang sama)
try:
    from models import get_model
    from xrd_dataset import get_dataloaders
except ImportError as e:
    print(f"❌ ERROR: Gagal mengimpor modul lokal: {e}")
    print("Pastikan file berikut ada di folder yang sama:")
    print("  - models.py")
    print("  - xrd_dataset.py")
    sys.exit(1)


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth.tar"):
    """Simpan checkpoint model"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    if is_best:
        best_path = checkpoint_dir / "model_best.pth.tar"
        shutil.copyfile(filepath, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Muat checkpoint model"""
    if not Path(checkpoint_path).exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def validate(model, val_loader, criterion, device):
    """Validasi model"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Training satu epoch"""
    model.train()
    total_loss = 0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser(description='Train XRD Denoising Model')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode (5 epochs, small dataset)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'],
                        help='Device to use')
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'resunet'],
                        help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base channels for UNet')
    parser.add_argument('--input_length', type=int, default=8500,
                        help='Input sequence length')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use learning rate scheduler')
    parser.add_argument('--scheduler_step', type=int, default=20,
                        help='Step size for scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                        help='Gamma for scheduler')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin memory in DataLoader')
    parser.add_argument('--save_dir', type=str,
                        default='models/saved',
                        help='Directory to save final model')
    parser.add_argument('--log_dir', type=str,
                        default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Atur jumlah thread untuk CPU (sesuaikan dengan core fisik i5-6500 = 4 core)
    torch.set_num_threads(4)
    
    # Tentukan device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    
    base_dir = Path.cwd()
    print(f"\n[DATA PATHS]")
    print(f"  Base dir: {base_dir}")
    train_clean = base_dir / "data/processed/train/clean"
    train_noisy = base_dir / "data/processed/train/noisy"
    val_clean = base_dir / "data/processed/val/clean"
    val_noisy = base_dir / "data/processed/val/noisy"
    print(f"  Train clean: {train_clean}")
    print(f"  Train noisy: {train_noisy}")
    print(f"  Val clean: {val_clean}")
    print(f"  Val noisy: {val_noisy}")

    print(f"\n[MODEL]")
    print(f"  Type: {args.model_type}")
    print(f"  Base channels: {args.base_channels}")
    print(f"  Input length: {args.input_length}")

    print(f"\n[TRAINING]")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Early stopping: {args.early_stopping} (patience: {args.patience})")

    print(f"\n[SCHEDULER]")
    print(f"  Use scheduler: {args.use_scheduler}")
    if args.use_scheduler:
        print(f"  Step size: {args.scheduler_step}")
        print(f"  Gamma: {args.scheduler_gamma}")

    print(f"\n[DATA LOADING]")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Pin memory: {args.pin_memory}")

    print(f"\n[DEVICE]")
    print(f"  Device: {device}")

    print(f"\n[OUTPUT]")
    save_dir = base_dir / args.save_dir
    log_dir = base_dir / args.log_dir
    checkpoint_dir = base_dir / args.checkpoint_dir
    print(f"  Save dir: {save_dir}")
    print(f"  Log dir: {log_dir}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print("=" * 80)

    # Buat direktori output
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Buat dataloaders
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)
    
    batch_size = args.batch_size
    if args.test_mode:
        print("🧪 RUNNING IN TEST MODE (reduced epochs and batch size)")
        args.epochs = min(5, args.epochs)
        batch_size = min(8, batch_size)
    
    try:
        train_loader, val_loader = get_dataloaders(
            train_clean_dir=train_clean,
            train_noisy_dir=train_noisy,
            val_clean_dir=val_clean,
            val_noisy_dir=val_noisy,
            batch_size=batch_size,
            num_workers=args.num_workers,
            target_length=args.input_length,
            pin_memory=args.pin_memory
        )
    except Exception as e:
        print(f"\n❌ ERROR saat membuat dataloader: {e}")
        print("Pastikan:")
        print("  1. Folder train/val berisi file .npy")
        print("  2. Nama file: clean_XXXXXX.npy dan noisy_XXXXXX.npy (ID harus match)")
        print("  3. Anda sudah menjalankan 4_split_dataset.py versi terbaru")
        sys.exit(1)

    # Inisialisasi model
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    model = get_model(
    model_type=args.model_type,
    base_channels=args.base_channels,
    input_length=args.input_length
    ).to(device)
    
    print(f"Model initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer dan loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.scheduler_step, 
            gamma=args.scheduler_gamma
        )

    # Early stopping
    early_stopper = EarlyStopping(patience=args.patience) if args.early_stopping else None

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validasi
        val_loss = validate(model, val_loader, criterion, device)
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Train Loss: {train_loss:.6f} "
                  f"Val Loss: {val_loss:.6f} "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Simpan checkpoint terbaik
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, is_best, checkpoint_dir)

        # Early stopping
        if early_stopper and early_stopper(val_loss):
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break

        # Update scheduler
        if scheduler:
            scheduler.step()

    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Simpan model final
    final_model_path = save_dir / "xrd_denoiser_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n💾 Model saved to: {final_model_path}")

    # Simpan konfigurasi
    config = vars(args)
    config['device'] = str(device)
    config['training_time'] = training_time
    config['best_val_loss'] = best_val_loss
    
    config_path = save_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"📝 Configuration saved to: {config_path}")

    writer.close()
    print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()