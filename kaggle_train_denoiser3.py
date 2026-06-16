"""
kaggle_train_denoiser3_final.py
✅ Auto-save setiap 5 menit ke /kaggle/working
✅ Auto-resume dari checkpoint terakhir
✅ Generate download link setiap save (backup manual mudah)
✅ TANPA Kaggle commit yang error/gagal
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
from tqdm import tqdm
from IPython.display import FileLink, display

# ============================================================================
# IMPORT MODUL XRD
# ============================================================================
sys.path.append("/kaggle/working")
try:
    from models import get_model
    from xrd_dataset import get_dataloaders
except Exception as e:
    print("❌ ERROR: Modul XRD tidak ditemukan!")
    print("   Jalankan dulu: !cp /kaggle/input/xrd-denoising-dataset/*.py /kaggle/working/")
    sys.exit(1)

# ============================================================================
# KONFIGURASI
# ============================================================================
class Config:
    DATASET_NAME = "xrd-denoising-dataset"
    EPOCHS = 100
    BATCH_SIZE = 2
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    MODEL_TYPE = "unet"
    BASE_CHANNELS = 32
    INPUT_LENGTH = 8500
    NUM_WORKERS = 2
    SEED = 42
    CHECKPOINT_INTERVAL_MIN = 5  # Auto-save setiap 5 menit

# ============================================================================
# UTILITIES ANTI-SNAP
# ============================================================================
def save_checkpoint(state, is_best, checkpoint_dir):
    """Simpan checkpoint + generate download link"""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Filename dengan timestamp
    timestamp = int(time.time())
    filename = f"ckpt_epoch_{state['epoch']:03d}_time_{timestamp}.pth"
    filepath = ckpt_dir / filename
    
    # Simpan checkpoint
    torch.save(state, filepath)
    
    # Update symlink ke checkpoint terakhir
    last_path = ckpt_dir / "checkpoint_last.pth"
    if last_path.exists():
        last_path.unlink()
    if os.name != 'nt':
        last_path.symlink_to(filepath.name)
    else:
        shutil.copyfile(filepath, last_path)
    
    # Simpan best model
    if is_best:
        best_path = ckpt_dir / "checkpoint_best.pth"
        shutil.copyfile(filepath, best_path)
    
    # Tampilkan info + download link
    progress_pct = (state['epoch'] / Config.EPOCHS) * 100
    print(f"\n💾 [{time.strftime('%H:%M:%S')}] Checkpoint tersimpan!")
    print(f"   Epoch: {state['epoch']}/{Config.EPOCHS} ({progress_pct:.1f}%)")
    print(f"   Val Loss: {state['best_val_loss']:.6f}")
    print(f"   File: {filename}")
    print("\n⬇️  KLIK LINK UNTUK BACKUP KE LAPTOP (disarankan setiap 5 epoch):")
    display(FileLink(filepath))
    print()

def find_latest_checkpoint(checkpoint_dir):
    """Cari checkpoint TERAKHIR berdasarkan file modification time"""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None, 0, float("inf")
    
    # Cari semua checkpoint
    ckpt_files = list(ckpt_dir.glob("ckpt_epoch_*.pth"))
    if not ckpt_files:
        last_path = ckpt_dir / "checkpoint_last.pth"
        if last_path.exists():
            ckpt_files = [last_path]
    
    if not ckpt_files:
        return None, 0, float("inf")
    
    # Pilih file termuda
    latest = max(ckpt_files, key=lambda f: f.stat().st_mtime)
    try:
        ckpt = torch.load(latest, map_location="cpu")
        return latest, ckpt["epoch"], ckpt["best_val_loss"]
    except Exception as e:
        print(f"⚠️ Gagal load {latest.name}: {e}")
        return None, 0, float("inf")

def load_checkpoint_for_resume(checkpoint_dir, model, optimizer):
    """Resume training dari checkpoint terakhir"""
    ckpt_file, start_epoch, best_val = find_latest_checkpoint(checkpoint_dir)
    
    if ckpt_file is None:
        print("\n🆕 Tidak ada checkpoint ditemukan. Memulai training baru dari epoch 1...")
        return 0, float("inf")
    
    try:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        
        progress = (start_epoch / Config.EPOCHS) * 100
        print("\n" + "="*80)
        print("🔄 RESUME TRAINING BERHASIL!")
        print("="*80)
        print(f"   Dilanjutkan dari   : Epoch {start_epoch + 1}")
        print(f"   Progress           : {progress:.1f}% ({start_epoch}/{Config.EPOCHS})")
        print(f"   Best validation    : {best_val:.6f}")
        print(f"   File checkpoint    : {ckpt_file.name}")
        print("="*80 + "\n")
        return start_epoch, best_val
    except Exception as e:
        print(f"⚠️ Gagal resume dari {ckpt_file.name}: {e}")
        print("🆕 Memulai training baru dari epoch 1...")
        return 0, float("inf")

# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    cfg = Config()
    
    # Seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    BASE_INPUT = Path("/kaggle/input") / cfg.DATASET_NAME / "processed"
    CKPT_DIR = Path("/kaggle/working/output/checkpoints")
    
    # Validasi dataset
    for p in ["train/clean", "train/noisy", "val/clean", "val/noisy"]:
        full_path = BASE_INPUT / p
        if not full_path.exists():
            raise FileNotFoundError(f"❌ Dataset tidak ditemukan: {full_path}")

    # Dataloader
    train_loader, val_loader = get_dataloaders(
        train_clean_dir=BASE_INPUT / "train" / "clean",
        train_noisy_dir=BASE_INPUT / "train" / "noisy",
        val_clean_dir=BASE_INPUT / "val" / "clean",
        val_noisy_dir=BASE_INPUT / "val" / "noisy",
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        target_length=cfg.INPUT_LENGTH,
        pin_memory=False
    )

    # Model & Optimizer
    model = get_model(
        model_type=cfg.MODEL_TYPE,
        base_channels=cfg.BASE_CHANNELS,
        input_length=cfg.INPUT_LENGTH
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # === RESUME DARI CHECKPOINT ===
    start_epoch, best_val = load_checkpoint_for_resume(CKPT_DIR, model, optimizer)
    if start_epoch >= cfg.EPOCHS:
        print(f"✅ Training sudah selesai ({start_epoch} epoch)")
        return

    # Training loop
    last_save_time = time.time()
    total_epochs = cfg.EPOCHS - start_epoch
    print(f"\n🚀 Training: epoch {start_epoch+1} → {cfg.EPOCHS} ({total_epochs} epoch tersisa)")
    print(f"   Auto-save setiap: {cfg.CHECKPOINT_INTERVAL_MIN} menit")
    print("="*80 + "\n")

    try:
        for epoch in range(start_epoch, cfg.EPOCHS):
            epoch_start = time.time()
            
            # --- Training ---
            model.train()
            train_loss = 0.0
            for noisy, clean in tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg.EPOCHS}", leave=False):
                noisy, clean = noisy.to(device), clean.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(noisy), clean)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    val_loss += criterion(model(noisy), clean).item()
            val_loss /= len(val_loader)

            # Progress
            progress = ((epoch + 1) / cfg.EPOCHS) * 100
            duration = (time.time() - epoch_start) / 60
            print(f"\n✅ Ep {epoch+1}/{cfg.EPOCHS} ({progress:.1f}%) | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"{duration:.1f} menit")

            # === AUTO-SAVE SETIAP AKHIR EPOCH ===
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val,
            }, is_best, CKPT_DIR)

            # === AUTO-SAVE TIME-BASED (setiap 5 menit) ===
            if (time.time() - last_save_time) / 60 >= cfg.CHECKPOINT_INTERVAL_MIN:
                print(f"\n⏱️  [AUTO-SAVE TIME-BASED] Interval {cfg.CHECKPOINT_INTERVAL_MIN} menit tercapai")
                save_checkpoint({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val,
                }, is_best, CKPT_DIR)
                last_save_time = time.time()

        # Simpan model final
        final_path = Path("/kaggle/working/output/models/final_model.pth")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_path)
        print(f"\n🎉 Training selesai! Model: {final_path}")

    except KeyboardInterrupt:
        print("\n🛑 Dihentikan manual — progress TERSIMPAN di checkpoint terakhir!")
        print("   Jalankan ulang script ini untuk RESUME training.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Checkpoint terakhir tetap aman di folder checkpoints/")

if __name__ == "__main__":
    main()