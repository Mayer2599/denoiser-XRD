"""
4_split_dataset.py
Split XRD dataset menjadi training dan validation berdasarkan ID pairing.

Format file:
- clean/clean_001234.npy
- noisy/noisy_001234.npy

Pairing dilakukan berdasarkan ID numerik (001234), bukan nama file literal.
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np


# Set random seed untuk reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def extract_id_from_filename(filename):
    """
    Ekstrak ID numerik dari nama file.
    Contoh:
        'clean_001234.npy' → '001234'
        'noisy_999999.npy' → '999999'
    """
    if filename.startswith("clean_") and filename.endswith(".npy"):
        return filename[6:-4]
    elif filename.startswith("noisy_") and filename.endswith(".npy"):
        return filename[6:-4]
    else:
        raise ValueError(f"Unexpected filename format: {filename}")


def split_dataset(
    clean_dir="data/processed/clean",
    noisy_dir="data/processed/noisy",
    output_dir="data/processed",
    train_ratio=0.8,
    verbose=True
):
    """
    Split dataset menjadi training dan validation berdasarkan ID pairing.

    Parameters:
    -----------
    clean_dir : str
        Path ke folder clean data (berisi clean_XXXXXX.npy)
    noisy_dir : str
        Path ke folder noisy data (berisi noisy_XXXXXX.npy)
    output_dir : str
        Path output untuk struktur train/val
    train_ratio : float
        Rasio data training (default: 0.8)
    verbose : bool
        Tampilkan progres
    """
    print("=" * 80)
    print("SPLITTING DATASET INTO TRAINING AND VALIDATION (ID-BASED PAIRING)")
    print("=" * 80)

    # Konversi ke Path objects
    clean_dir = Path(clean_dir)
    noisy_dir = Path(noisy_dir)
    output_dir = Path(output_dir)

    # Validasi direktori
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean directory tidak ditemukan: {clean_dir}")
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy directory tidak ditemukan: {noisy_dir}")

    # Ambil semua file .npy
    clean_files = sorted([f.name for f in clean_dir.glob("*.npy")])
    noisy_files = sorted([f.name for f in noisy_dir.glob("*.npy")])

    print(f"\nJumlah clean files: {len(clean_files)}")
    print(f"Jumlah noisy files: {len(noisy_files)}")

    # Ekstrak ID
    try:
        clean_ids = {extract_id_from_filename(f) for f in clean_files}
        noisy_ids = {extract_id_from_filename(f) for f in noisy_files}
    except ValueError as e:
        raise ValueError(f"Error parsing filenames: {e}")

    # Cari ID yang ada di kedua folder
    common_ids = sorted(clean_ids & noisy_ids)
    missing_in_noisy = clean_ids - noisy_ids
    missing_in_clean = noisy_ids - clean_ids

    if missing_in_noisy:
        print(f"\n⚠️ WARNING: {len(missing_in_noisy)} file di clean tapi tidak di noisy (contoh 5):")
        print(f"   {sorted(list(missing_in_noisy))[:5]}")
    if missing_in_clean:
        print(f"\n⚠️ WARNING: {len(missing_in_clean)} file di noisy tapi tidak di clean (contoh 5):")
        print(f"   {sorted(list(missing_in_clean))[:5]}")

    if not common_ids:
        raise ValueError("Tidak ada pasangan clean-noisy yang cocok berdasarkan ID!")

    print(f"\n✓ {len(common_ids)} pasangan valid ditemukan.")
    print("✓ Clean-noisy pairing verified by ID!")

    # Split ID
    train_ids, val_ids = train_test_split(
        common_ids,
        train_size=train_ratio,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print(f"\nSplit ratio: {train_ratio:.0%} training, {1 - train_ratio:.0%} validation")
    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")

    # Buat path output
    train_clean_dir = output_dir / "train" / "clean"
    train_noisy_dir = output_dir / "train" / "noisy"
    val_clean_dir = output_dir / "val" / "clean"
    val_noisy_dir = output_dir / "val" / "noisy"

    for d in [train_clean_dir, train_noisy_dir, val_clean_dir, val_noisy_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("COPYING FILES...")
    print("=" * 80)

    # Fungsi bantu untuk copy file berdasarkan ID
    def copy_pairs(id_list, clean_dest, noisy_dest, desc=""):
        for i, id_val in enumerate(id_list, 1):
            clean_src = clean_dir / f"clean_{id_val}.npy"
            noisy_src = noisy_dir / f"noisy_{id_val}.npy"

            shutil.copy2(clean_src, clean_dest / f"clean_{id_val}.npy")
            shutil.copy2(noisy_src, noisy_dest / f"noisy_{id_val}.npy")

            if verbose and i % 1000 == 0:
                print(f"  Processed {i}/{len(id_list)} {desc} files...")

    # Copy training
    print("\nCopying training files...")
    copy_pairs(train_ids, train_clean_dir, train_noisy_dir, "training")
    print(f"✓ Training files copied: {len(train_ids)} pairs")

    # Copy validation
    print("\nCopying validation files...")
    copy_pairs(val_ids, val_clean_dir, val_noisy_dir, "validation")
    print(f"✓ Validation files copied: {len(val_ids)} pairs")

    # Ringkasan akhir
    print("\n" + "=" * 80)
    print("SPLIT DATASET SUMMARY")
    print("=" * 80)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── clean/      ({len(train_ids)} files)")
    print(f"    │   └── noisy/      ({len(train_ids)} files)")
    print(f"    └── val/")
    print(f"        ├── clean/      ({len(val_ids)} files)")
    print(f"        └── noisy/      ({len(val_ids)} files)")

    # Verifikasi akhir
    final_check = (
        len(list(train_clean_dir.glob("*.npy"))) == len(train_ids) and
        len(list(train_noisy_dir.glob("*.npy"))) == len(train_ids) and
        len(list(val_clean_dir.glob("*.npy"))) == len(val_ids) and
        len(list(val_noisy_dir.glob("*.npy"))) == len(val_ids)
    )

    if final_check:
        print("\n✅ ALL FILES COPIED SUCCESSFULLY!")
        print("✅ PAIRING MAINTAINED VIA ID!")
        return True
    else:
        print("\n❌ ERROR: File count mismatch after copying!")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split XRD dataset into train/val (ID-based pairing)")
    parser.add_argument("--clean_dir", type=str, default="data/processed/clean",
                        help="Path to clean data directory")
    parser.add_argument("--noisy_dir", type=str, default="data/processed/noisy",
                        help="Path to noisy data directory")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for train/val split")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training data ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")

    args = parser.parse_args()

    # Atur seed untuk reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Using random seed: {args.seed}")

    # Jalankan split
    success = split_dataset(
        clean_dir=args.clean_dir,
        noisy_dir=args.noisy_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        verbose=True
    )

    if success:
        print("\n" + "=" * 80)
        print("DATASET SPLIT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext step: Train your denoising model with train_denoiser.py")
    else:
        print("\n" + "=" * 80)
        print("ERROR: Dataset split failed!")
        print("=" * 80)