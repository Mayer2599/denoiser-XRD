"""
XRD Dataset Class untuk PyTorch Training
Includes: Anscombe transform, normalization, dan data loading
Supports .npy files (NumPy binary format)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d


class XRDDataset(Dataset):
    """
    PyTorch Dataset untuk XRD data (format .npy)
    Load paired clean-noisy data dan apply preprocessing
    """

    def __init__(self, clean_dir, noisy_dir, transform=None, target_length=8500):
        """
        Parameters:
        -----------
        clean_dir : str or Path
            Path ke clean data directory (berisi clean_XXXXXX.npy)
        noisy_dir : str or Path
            Path ke noisy data directory (berisi noisy_XXXXXX.npy)
        transform : callable, optional
            Additional transformations
        target_length : int
            Target length untuk data (default: 8500)
        """
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
        Get single data pair

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

        # Preprocessing
        clean_data = self._preprocess(clean_data)
        noisy_data = self._preprocess(noisy_data)

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

    def _preprocess(self, data):
        """
        Preprocessing pipeline:
        1. Anscombe transform (stabilize variance)
        2. Normalization ke [0, 1]
        """
        # Handle negative values
        data_safe = np.maximum(data, 0)
        # Anscombe transform: 2 * sqrt(x + 3/8)
        data_anscombe = 2 * np.sqrt(data_safe + 3/8)

        # Normalisasi ke [0, 1]
        data_min = data_anscombe.min()
        data_max = data_anscombe.max()
        if data_max > data_min:
            data_normalized = (data_anscombe - data_min) / (data_max - data_min)
        else:
            data_normalized = data_anscombe

        return data_normalized


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
    Create DataLoaders untuk training dan validation
    """
    print("=" * 80)
    print("CREATING DATALOADERS")
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

    return train_loader, val_loader


def test_dataloader(dataloader, num_samples=3):
    """Test dataloader untuk verifikasi"""
    print("\n" + "=" * 80)
    print("TESTING DATALOADER")
    print("=" * 80)
    for i, (noisy, clean) in enumerate(dataloader):
        if i >= num_samples:
            break
        print(f"\nBatch {i+1}:")
        print(f"  Noisy shape: {noisy.shape}")
        print(f"  Clean shape: {clean.shape}")
        print(f"  Noisy range: [{noisy.min():.4f}, {noisy.max():.4f}]")
        print(f"  Clean range: [{clean.min():.4f}, {clean.max():.4f}]")
        print(f"  Noisy mean: {noisy.mean():.4f}, std: {noisy.std():.4f}")
        print(f"  Clean mean: {clean.mean():.4f}, std: {clean.std():.4f}")
    print("\n✓ DataLoader test completed!")


if __name__ == "__main__":
    # Test script
    print("Testing XRD Dataset and DataLoader...")

    # Paths (sesuaikan jika perlu)
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
        print("ALL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()