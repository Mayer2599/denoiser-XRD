# XRD Denoiser

Repository ini berisi pipeline deep learning untuk denoising pola XRD 1D. Kode awal dilatih di Kaggle, lalu disiapkan agar bisa dipakai, dilacak, dan dibagikan melalui GitHub.

## Isi utama

- `kaggle_train_denoiser.py`, `kaggle_train_denoiser2.py`, `kaggle_train_denoiser3.py`: script training versi Kaggle.
- `models.py`: definisi arsitektur model.
- `xrd_dataset.py` dan `xrd_dataset_RSnCEV.py`: loader dataset XRD.
- `_10a_denoise_xrd.py`: inference untuk satu file XRD.
- `_10b_batch_denoise.py`: inference batch.
- `_8evaluate_model.py`: evaluasi model.
- `models/experiment*/`: bobot model hasil training.
- `evaluation_results/` dan `analyze_failures/`: hasil evaluasi dan analisis.

## Model weights

File model berukuran besar disimpan dengan Git LFS. Setelah clone repo, pastikan Git LFS aktif:

```bash
git lfs install
git lfs pull
```

Weight denoiser yang dilacak di repo:

- `models/experiment2/best_model1a.pth`
- `models/experiment2/best_model1b.pth`
- `models/experiment2/final_model1a.pth`
- `models/experiment2/final_model1b.pth`
- `models/experiment3/checkpoint_best.pth`
- `models/experiment3/checkpoint_best3b.pth`
- `models/experiment3/final_model.pth`
- `models/experiment3/final_model3b.pth`

File lokal `decifer_v1_ckpt.pt` dan `checkpoints/` tidak ikut commit pertama karena ukurannya besar dan bukan artefak utama untuk inference denoiser. Jika benar-benar dibutuhkan, unggah sebagai GitHub Release asset atau tempatkan di penyimpanan model terpisah.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Untuk Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Contoh penggunaan

Cek GPU:

```bash
python cekcuda+gpu.py
```

### Denoising satu file XRD

Command utama untuk inference adalah `denoise.py`. Secara default script memakai model `models/experiment3/final_model3b.pth`.

```bash
python denoise.py "path/to/sample.xy"
```

Tentukan model dan output secara eksplisit:

```bash
python denoise.py "path/to/sample.xy" --model models/experiment3/final_model3b.pth --output results/sample_denoised.xy
```

Simpan plot perbandingan:

```bash
python denoise.py "path/to/sample.xy" --plot
```

Pakai CPU:

```bash
python denoise.py "path/to/sample.xy" --device cpu
```

### Denoising satu folder

```bash
python denoise.py "path/to/folder" --output denoised_outputs --pattern "*.xy"
```

Untuk mencari file di subfolder:

```bash
python denoise.py "path/to/folder" --output denoised_outputs --pattern "*.xy" --recursive
```

Evaluasi model:

```bash
python _8evaluate_model.py
```

## Catatan GitHub

Repo ini sengaja mengabaikan `.venv/`, `__pycache__/`, `logs/`, dan `.1deCIFer_repo/` agar GitHub hanya berisi kode, konfigurasi, hasil penting, dan model weights yang dilacak oleh Git LFS.

Jika file model terlalu besar untuk kuota Git LFS akun GitHub, opsi yang lebih hemat adalah menghapus weight dari Git dan mengunggahnya sebagai GitHub Release asset, Hugging Face model repo, Kaggle Model, atau Google Drive/Zenodo.
