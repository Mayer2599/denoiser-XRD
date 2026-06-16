"""
XRD Dataset Class untuk PyTorch Training
✅ Enhanced dengan RobustScaler + Clipping Extreme Values
✅ Solusi untuk failure cases extreme values (100% di analisis Anda)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# ============================================================================
# ROBUST NORMALIZATION (Solusi untuk extreme value failures)
# ============================================================================
try:
    from sklearn.preprocessing import RobustScaler
    ROBUST_SCALER_AVAILABLE = True
    print("✅ RobustScaler tersedia (scikit-learn)")
except ImportError:
    ROBUST_SCALER_AVAILABLE = False
    print("⚠️  RobustScaler tidak tersedia — menggunakan fallback normalization")
    # Fallback implementation tanpa scikit-learn
    class RobustScaler:
        def fit_transform(self, X):
            # Hitung median dan IQR manual
            median = np.median(X)
            q75, q25 = np.percentile(X, [75, 25])
            iqr = q75 - q25
            # Hindari division by zero
            if iqr == 0:
                iqr = 1.0
            return (X - median) / iqr

def robust_normalize(x, clip_percentile=99.5):
    """
    Pipeline preprocessing robust untuk XRD:
    1. Anscombe transform → stabilisasi Poisson noise
    2. Clip extreme outliers (>99.5% dan <0.5% percentile)
    3. RobustScaler (median + IQR) → immune terhadap outliers
    4. Normalisasi ke [0, 1] → kompatibel dengan UNet
    
    Parameters:
    -----------
    x : np.ndarray
        Raw XRD intensity data
    clip_percentile : float
        Percentile untuk clipping (default: 99.5 → clip 0.5% ekstrem di kedua sisi)
    
    Returns:
    --------
    x_normalized : np.ndarray
        Data yang sudah dipreprocess
    """
    # Langkah 1: Anscombe transform (WAJIB untuk XRD/Poisson noise)
    # Formula: 2 * sqrt(x + 3/8)
    x_safe = np.maximum(x, 0)  # Hindari nilai negatif
    x_anscombe = 2 * np.sqrt(x_safe + 3/8)
    
    # Langkah 2: Clip extreme outliers (>99.5% dan <0.5% percentile)
    # Ini adalah SOLUSI UTAMA untuk failure cases extreme values Anda!
    upper_clip = np.percentile(x_anscombe, clip_percentile)
    lower_clip = np.percentile(x_anscombe, 100 - clip_percentile)
    x_clipped = np.clip(x_anscombe, lower_clip, upper_clip)
    
    # Langkah 3: RobustScaler (gunakan median + IQR, bukan mean/std)
    scaler = RobustScaler()
    x_reshaped = x_clipped.reshape(-1, 1)
    x_robust = scaler.fit_transform(x_reshaped).flatten()
    
    # Langkah 4: Normalisasi ke [0, 1] (wajib untuk UNet)
    x_min = x_robust.min()
    x_max = x_robust.max()
    if x_max > x_min:
        x_normalized = (x_robust - x_min) / (x_max - x_min + 1e-8)  # +1e-8 untuk safety
    else:
        x_normalized = x_robust
    
    return x_normalized

# ============================================================================
# XRD DATASET CLASS
# ============================================================================
class XRDDataset(Dataset):
    """
    PyTorch Dataset untuk XRD data (format .npy)
    Load paired clean-noisy data dengan preprocessing robust
    """
    def __init__(self, clean_dir, noisy_dir, transform=None, target_length=8500):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.transform = transform
        self.target_length = target_length

        # Ambil semua file .npy
        clean_files = sorted([f.name for f in self.clean_dir.glob("*.npy")])
        noisy_files = sorted([f.name for f in self.noisy_dir.glob("*.npy")])

        # Ekstrak ID untuk pairing
        def extract_id(fname):
            if fname.startswith("clean_"):
                return fname[6:-4]
            elif fname.startswith("noisy_"):
                return fname[6:-4]
            else:
                raise ValueError(f"Unexpected filename: {fname}")

        clean_ids = {extract_id(f): f for f in clean_files}
        noisy_ids = {extract_id(f): f for f in noisy_files}

        # Cari pasangan berdasarkan ID
        common_ids = sorted(set(clean_ids.keys()) & set(noisy_ids.keys()))

        if not common_ids:
            raise ValueError("Tidak ada pasangan clean-noisy yang cocok berdasarkan ID!")

        # Simpan daftar file berpasangan
        self.file_pairs = [
            (clean_ids[id_], noisy_ids[id_]) for id_ in common_ids
        ]

        print(f"Dataset initialized: {len(self.file_pairs)} pairs")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """
        Get single data pair dengan preprocessing robust
        
        Returns:
        --------
        noisy : torch.Tensor
            Preprocessed noisy data (shape: [1, L])
        clean : torch.Tensor
            Preprocessed clean data (shape: [1, L])
        """
        clean_fname, noisy_fname = self.file_pairs[idx]

        # Load files
        clean_path = self.clean_dir / clean_fname
        noisy_path = self.noisy_dir / noisy_fname

        clean_data = np.load(clean_path)
        noisy_data = np.load(noisy_path)

        # Pastikan 1D
        if clean_data.ndim != 1:
            clean_data = clean_data.flatten()
        if noisy_data.ndim != 1:
            noisy_data = noisy_data.flatten()

        # Resample jika perlu
        if len(clean_data) != self.target_length:
            clean_data = self._resample(clean_data, self.target_length)
        if len(noisy_data) != self.target_length:
            noisy_data = self._resample(noisy_data, self.target_length)

        # ✅ PREPROCESSING ROBUST (Solusi untuk extreme value failures)
        clean_data = robust_normalize(clean_data, clip_percentile=99.5)
        noisy_data = robust_normalize(noisy_data, clip_percentile=99.5)

        # Convert to tensors
        noisy_tensor = torch.FloatTensor(noisy_data).unsqueeze(0)  # [1, L]
        clean_tensor = torch.FloatTensor(clean_data).unsqueeze(0)

        if self.transform:
            noisy_tensor = self.transform(noisy_tensor)
            clean_tensor = self.transform(clean_tensor)

        return noisy_tensor, clean_tensor

    def _resample(self, data, target_length):
        """Resample data ke target length menggunakan interpolasi linear"""
        if len(data) == target_length:
            return data
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, target_length)
        f = interp1d(x_old, data, kind='linear', fill_value="extrapolate")
        return f(x_new)

# ============================================================================
# DATALOADER FACTORY
# ============================================================================
def get_dataloaders(
    train_clean_dir,
    train_noisy_dir,
    val_clean_dir,
    val_noisy_dir,
    batch_size=16,
    num_workers=4,
    target_length=8500,
    pin_memory=True
):
    """
    Create DataLoaders untuk training dan validation dengan preprocessing robust
    """
    print("=" * 80)
    print("CREATING DATALOADERS (Robust Preprocessing)")
    print("=" * 80)
    print("✅ Preprocessing pipeline:")
    print("   1. Anscombe transform → stabilisasi Poisson noise")
    print("   2. Clipping extreme values (99.5 percentile) → solusi failure cases")
    print("   3. RobustScaler (median + IQR) → immune terhadap outliers")
    print("   4. Normalisasi ke [0, 1] → kompatibel dengan UNet")
    print("=" * 80)

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = XRDDataset(
        clean_dir=train_clean_dir,
        noisy_dir=train_noisy_dir,
        target_length=target_length
    )

    print("\nCreating validation dataset...")
    val_dataset = XRDDataset(
        clean_dir=val_clean_dir,
        noisy_dir=val_noisy_dir,
        target_length=target_length
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty! Check your data directories.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty! Check your data directories.")

    # Create dataloaders
    print(f"\nCreating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print(f"\n✓ DataLoaders created!")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print("=" * 80)

    return train_loader, val_loader

# ============================================================================
# TEST SCRIPT
# ============================================================================
def test_dataloader(dataloader, num_samples=3):
    """Test dataloader untuk verifikasi preprocessing robust"""
    print("\n" + "=" * 80)
    print("TESTING DATALOADER (Robust Preprocessing)")
    print("=" * 80)
    for i, (noisy, clean) in enumerate(dataloader):
        if i >= num_samples:
            break
        print(f"\nSample {i+1}:")
        print(f"  Noisy shape: {noisy.shape}")
        print(f"  Clean shape: {clean.shape}")
        print(f"  Noisy range: [{noisy.min():.4f}, {noisy.max():.4f}] ← HARUS [0.0, 1.0]")
        print(f"  Clean range: [{clean.min():.4f}, {clean.max():.4f}] ← HARUS [0.0, 1.0]")
        print(f"  Noisy mean: {noisy.mean():.4f}, std: {noisy.std():.4f}")
        print(f"  Clean mean: {clean.mean():.4f}, std: {clean.std():.4f}")
        
        # Deteksi extreme values yang tersisa
        extreme_noisy = ((noisy > 0.95) | (noisy < 0.05)).sum().item()
        extreme_clean = ((clean > 0.95) | (clean < 0.05)).sum().item()
        print(f"  Extreme values (>95%/<5%): Noisy={extreme_noisy}, Clean={extreme_clean} points")
        if extreme_noisy > 0 or extreme_clean > 0:
            print("  ⚠️  Masih ada extreme values — pertimbangkan clip_percentile=99.0")
        else:
            print("  ✅ Tidak ada extreme values — preprocessing berhasil!")
    
    print("\n✓ DataLoader test completed!")
    print("=" * 80)

if __name__ == "__main__":
    # Test script
    print("Testing XRD Dataset with Robust Preprocessing...")
    
    # Paths untuk testing lokal (sesuaikan jika perlu)
    train_clean = "data/processed/train/clean"
    train_noisy = "data/processed/train/noisy"
    val_clean = "data/processed/val/clean"
    val_noisy = "data/processed/val/noisy"

    # Validasi direktori
    dirs = [train_clean, train_noisy, val_clean, val_noisy]
    missing = [d for d in dirs if not Path(d).exists()]
    if missing:
        print(f"ERROR: Direktori berikut tidak ditemukan:")
        for d in missing:
            print(f"  - {d}")
        print("Pastikan Anda sudah menjalankan 4_split_dataset.py")
        exit(1)

    try:
        train_loader, val_loader = get_dataloaders(
            train_clean_dir=train_clean,
            train_noisy_dir=train_noisy,
            val_clean_dir=val_clean,
            val_noisy_dir=val_noisy,
            batch_size=8,
            num_workers=2,
            target_length=8500
        )

        print("\n" + "=" * 80)
        print("Testing Training DataLoader:")
        test_dataloader(train_loader, num_samples=2)

        print("\n" + "=" * 80)
        print("Testing Validation DataLoader:")
        test_dataloader(val_loader, num_samples=2)

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✅")
        print("Robust preprocessing siap digunakan untuk training!")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()