"""
kaggle_train_denoiser.py
Versi FINAL untuk Kaggle — tanpa argparse, progress bar rapi, import aman.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ============================================================
# IMPORT LOCAL MODULES DENGAN PATH EXPLICIT
# ============================================================
# Tambahkan working directory ke sys.path secara eksplisit
sys.path.append("/kaggle/working")

try:
    from models import get_model
    from xrd_dataset import get_dataloaders
except Exception as e:
    print("❌ Gagal mengimpor modul lokal!")
    print(f"Error: {e}")
    print("\nPastikan file berikut ada di /kaggle/working/:")
    print("  - models.py")
    print("  - xrd_dataset.py")
    print("\nJalankan cell ini terlebih dahulu:")
    print('!cp /kaggle/input/xrd-denoising-dataset/*.py /kaggle/working/')
    sys.exit(1)

# ============================================================
# KONFIGURASI
# ============================================================
class Config:
    DATASET_NAME = "xrd-denoising-dataset"  # ← Ganti jika nama dataset Anda berbeda
    EPOCHS = 100          # Ubah ke 100 untuk training penuh
    BATCH_SIZE = 2
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    MODEL_TYPE = "unet"
    BASE_CHANNELS = 32
    INPUT_LENGTH = 8500
    NUM_WORKERS = 2
    PIN_MEMORY = False
    USE_SCHEDULER = False
    SCHEDULER_STEP = 20
    SCHEDULER_GAMMA = 0.5
    EARLY_STOPPING = False
    PATIENCE = 10
    SEED = 42

# ============================================================
# UTILITIES
# ============================================================
def save_checkpoint(state, is_best, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "checkpoint_last.pth"
    torch.save(state, last_path)
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pth"
        shutil.copyfile(last_path, best_path)

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.stop

# ============================================================
# MAIN
# ============================================================
def main():
    cfg = Config()

    # Seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    BASE_INPUT = Path("/kaggle/input") / cfg.DATASET_NAME / "processed"
    train_clean = BASE_INPUT / "train" / "clean"
    train_noisy = BASE_INPUT / "train" / "noisy"
    val_clean = BASE_INPUT / "val" / "clean"
    val_noisy = BASE_INPUT / "val" / "noisy"

    for p in [train_clean, train_noisy, val_clean, val_noisy]:
        if not p.exists():
            raise FileNotFoundError(f"❌ Dataset path not found: {p}")

    OUTPUT_ROOT = Path("/kaggle/working/output")
    SAVE_DIR = OUTPUT_ROOT / "models"
    LOG_DIR = OUTPUT_ROOT / "logs"
    CKPT_DIR = OUTPUT_ROOT / "checkpoints"

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Dataloader
    train_loader, val_loader = get_dataloaders(
        train_clean_dir=train_clean,
        train_noisy_dir=train_noisy,
        val_clean_dir=val_clean,
        val_noisy_dir=val_noisy,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        target_length=cfg.INPUT_LENGTH,
        pin_memory=cfg.PIN_MEMORY
    )

    # Model
    model = get_model(
        model_type=cfg.MODEL_TYPE,
        base_channels=cfg.BASE_CHANNELS,
        input_length=cfg.INPUT_LENGTH
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & Loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    scheduler = None
    if cfg.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.SCHEDULER_STEP,
            gamma=cfg.SCHEDULER_GAMMA
        )

    early_stopper = EarlyStopping(cfg.PATIENCE) if cfg.EARLY_STOPPING else None
    writer = SummaryWriter(LOG_DIR)

    # Training loop dengan progress bar rapi
    best_val = float("inf")
    start_time = time.time()

    # Progress bar utama (per epoch)
    epoch_pbar = tqdm(range(cfg.EPOCHS), desc="Epochs", position=0, leave=True)

    for epoch in epoch_pbar:
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Train Ep {epoch+1}", leave=False, position=1)
        for noisy, clean in train_pbar:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Ep {epoch+1}", leave=False, position=2)
            for noisy, clean in val_pbar:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                output = model(noisy)
                loss = criterion(output, clean)
                val_loss += loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        val_loss /= len(val_loader)

        # Update progress bar utama
        epoch_pbar.set_postfix({
            "Train": f"{train_loss:.6f}",
            "Val": f"{val_loss:.6f}"
        })

        # Logging TensorBoard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Simpan checkpoint
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val,
        }, is_best, CKPT_DIR)

        if scheduler:
            scheduler.step()

        if early_stopper and early_stopper(val_loss):
            print("\n🛑 Early stopping triggered")
            break

    # Simpan model akhir
    final_model = SAVE_DIR / "xrd_denoiser_final.pth"
    torch.save(model.state_dict(), final_model)

    elapsed = time.time() - start_time
    print(f"\n✅ Training finished in {elapsed/60:.2f} minutes")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Model saved to: {final_model}")

    config_path = SAVE_DIR / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    writer.close()
    print("🎉 DONE")

if __name__ == "__main__":
    main()