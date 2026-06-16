"""
Command-line inference for the XRD denoiser.

Examples:
    python denoise.py sample.xy
    python denoise.py sample.xy --model models/experiment3/final_model3b.pth --plot
    python denoise.py data/raw --output results/denoised --pattern "*.xy"
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm

from models import get_model


DEFAULT_MODEL = Path("models/experiment3/final_model3b.pth")


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    signal_power = float(np.mean(signal ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


class XRDDenoiser:
    """Load a trained denoiser and run inference on XRD files."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        device: str = "auto",
        model_type: str = "unet",
        base_channels: int = 32,
        input_length: int = 8500,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable; using CPU instead.")
            device = "cpu"
        self.device = torch.device(device)

        loaded = torch.load(self.model_path, map_location=self.device)
        state_dict, checkpoint_config, metadata = self._extract_state_dict(loaded)

        self.model_type = checkpoint_config.get("model_type", model_type)
        self.base_channels = int(checkpoint_config.get("base_channels", base_channels))
        self.input_length = int(checkpoint_config.get("input_length", input_length))

        self.model = get_model(
            model_type=self.model_type,
            base_channels=self.base_channels,
            input_length=self.input_length,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"Model       : {self.model_path}")
        print(f"Architecture: {self.model_type}, base_channels={self.base_channels}")
        print(f"Input length: {self.input_length}")
        print(f"Device      : {self.device}")
        if metadata:
            print("Checkpoint  : " + ", ".join(metadata))

    @staticmethod
    def _extract_state_dict(loaded: object) -> tuple[dict, dict, list[str]]:
        config: dict = {}
        metadata: list[str] = []

        if isinstance(loaded, dict):
            if "state_dict" in loaded:
                state_dict = loaded["state_dict"]
                config = loaded.get("config", {}) or {}
            elif "model_state_dict" in loaded:
                state_dict = loaded["model_state_dict"]
                config = loaded.get("config", {}) or {}
            else:
                state_dict = loaded

            if "epoch" in loaded:
                metadata.append(f"epoch={loaded['epoch']}")
            if "best_val_loss" in loaded:
                metadata.append(f"best_val_loss={loaded['best_val_loss']:.6f}")
        else:
            state_dict = loaded

        if not isinstance(state_dict, dict):
            raise RuntimeError("Checkpoint does not contain a valid PyTorch state dict.")

        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key[6:] if key.startswith("model.") else key] = value
        return cleaned, config, metadata

    @staticmethod
    def load_xrd_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        try:
            data = np.loadtxt(path, comments=("#", ";", "@"))
        except Exception:
            rows = []
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip().replace(",", " ")
                    if not line or line.startswith(("#", ";", "@", "/*")):
                        continue
                    parts = line.split()
                    try:
                        rows.append([float(part) for part in parts])
                    except ValueError:
                        continue
            if not rows:
                raise ValueError(f"No numeric XRD data found in: {path}")
            data = np.array(rows, dtype=float)

        if data.ndim == 1:
            intensity = data.astype(float)
            angles = np.arange(len(intensity), dtype=float)
        elif data.shape[1] >= 2:
            angles = data[:, 0].astype(float)
            intensity = data[:, -1].astype(float)
        else:
            raise ValueError(f"Unsupported data shape in {path}: {data.shape}")

        return angles, intensity

    @staticmethod
    def resample(values: np.ndarray, target_length: int) -> np.ndarray:
        if len(values) == target_length:
            return values.copy()

        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        x_old = np.linspace(0, 1, len(clean))
        x_new = np.linspace(0, 1, target_length)
        return interp1d(x_old, clean, kind="linear", fill_value="extrapolate")(x_new)

    @staticmethod
    def preprocess(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        safe = np.maximum(values, 0)
        transformed = 2 * np.sqrt(safe + 3 / 8)
        data_min = float(transformed.min())
        data_max = float(transformed.max())

        if data_max > data_min:
            normalized = (transformed - data_min) / (data_max - data_min)
        else:
            normalized = transformed

        return normalized, {"anscombe_min": data_min, "anscombe_max": data_max}

    @staticmethod
    def inverse_preprocess(values: np.ndarray, params: dict[str, float]) -> np.ndarray:
        data_min = params["anscombe_min"]
        data_max = params["anscombe_max"]

        if data_max > data_min:
            denormalized = values * (data_max - data_min) + data_min
        else:
            denormalized = values

        restored = (denormalized / 2) ** 2 - 3 / 8
        return np.maximum(restored, 0)

    def denoise_intensity(self, intensity: np.ndarray) -> np.ndarray:
        original_length = len(intensity)
        model_input = self.resample(intensity, self.input_length)
        normalized, params = self.preprocess(model_input)

        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            predicted = self.model(tensor).squeeze().cpu().numpy()

        denoised = self.inverse_preprocess(predicted, params)
        if original_length != self.input_length:
            denoised = self.resample(denoised, original_length)
        return denoised

    def denoise_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        plot: bool = False,
        delimiter: str = "\t",
    ) -> dict[str, object]:
        input_path = Path(input_path)
        output_path = make_output_path(input_path, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        angles, noisy = self.load_xrd_file(input_path)
        start = time.time()
        denoised = self.denoise_intensity(noisy)
        elapsed = time.time() - start

        output_data = np.column_stack([angles, denoised])
        np.savetxt(
            output_path,
            output_data,
            fmt="%.8f",
            delimiter=delimiter,
            header=f"2Theta{delimiter}Intensity",
            comments="",
        )

        residual = noisy - denoised
        noise_before = float(np.std(noisy))
        noise_after = float(np.std(residual))
        noise_reduction = (
            (noise_before - noise_after) / noise_before * 100 if noise_before > 0 else 0.0
        )
        snr = calculate_snr(denoised, residual)

        plot_path = None
        if plot:
            plot_path = output_path.with_name(f"{output_path.stem}_comparison.png")
            save_plot(angles, noisy, denoised, input_path, plot_path)

        return {
            "input": str(input_path),
            "output": str(output_path),
            "points": len(noisy),
            "processing_time_sec": elapsed,
            "snr_db": snr,
            "noise_reduction_percent": noise_reduction,
            "plot": str(plot_path) if plot_path else "",
        }


def make_output_path(input_path: Path, output: str | Path | None) -> Path:
    if output is None:
        return input_path.with_name(f"{input_path.stem}_denoised{input_path.suffix or '.txt'}")

    output_path = Path(output)
    if output_path.exists() and output_path.is_dir():
        return output_path / f"{input_path.stem}_denoised{input_path.suffix or '.txt'}"
    if str(output).endswith(("/", "\\")):
        return output_path / f"{input_path.stem}_denoised{input_path.suffix or '.txt'}"
    return output_path


def save_plot(
    angles: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    input_path: Path,
    plot_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    axes[0].plot(angles, noisy, color="0.45", linewidth=1.0, label="Original")
    axes[0].plot(angles, denoised, color="tab:red", linewidth=1.4, label="Denoised")
    axes[0].set_title(f"XRD Denoising: {input_path.name}")
    axes[0].set_ylabel("Intensity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    peak_mask = denoised > np.percentile(denoised, 70)
    if np.any(peak_mask):
        peak_indices = np.where(peak_mask)[0]
        start = max(0, int(peak_indices[0]) - 50)
        end = min(len(angles), int(peak_indices[-1]) + 50)
        axes[1].plot(angles[start:end], noisy[start:end], color="0.45", linewidth=1.0)
        axes[1].plot(angles[start:end], denoised[start:end], color="tab:red", linewidth=1.4)
        axes[1].set_title("Peak Region")
        axes[1].set_ylabel("Intensity")
        axes[1].grid(True, alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "No peak region detected", ha="center", va="center")
        axes[1].set_axis_off()

    residual = noisy - denoised
    axes[2].plot(angles, residual, color="tab:blue", linewidth=0.9)
    axes[2].axhline(0, color="0.2", linestyle="--", linewidth=0.8)
    axes[2].set_title("Residual")
    axes[2].set_xlabel("2Theta")
    axes[2].set_ylabel("Original - Denoised")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def iter_input_files(path: Path, pattern: str, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    iterator = path.rglob(pattern) if recursive else path.glob(pattern)
    return sorted(file for file in iterator if file.is_file())


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> Path:
    summary_path = output_dir / "denoise_summary.csv"
    fieldnames = [
        "input",
        "output",
        "points",
        "processing_time_sec",
        "snr_db",
        "noise_reduction_percent",
        "plot",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise one XRD file or a folder of XRD files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input XRD file or folder.")
    parser.add_argument(
        "-m",
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file for single input, or output folder for folder input.",
    )
    parser.add_argument("--pattern", default="*.xy", help="File pattern for folder input.")
    parser.add_argument("--recursive", action="store_true", help="Search folders recursively.")
    parser.add_argument("--plot", action="store_true", help="Save comparison plot beside output.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument("--model-type", default="unet", choices=["unet", "simple_cnn"])
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--input-length", type=int, default=8500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_arg = Path(args.output) if args.output else None

    try:
        denoiser = XRDDenoiser(
            model_path=args.model,
            device=args.device,
            model_type=args.model_type,
            base_channels=args.base_channels,
            input_length=args.input_length,
        )

        files = iter_input_files(input_path, args.pattern, args.recursive)
        if not files:
            print(f"No files matched {args.pattern!r} in {input_path}")
            return 1

        if input_path.is_dir():
            output_dir = output_arg or Path("denoised_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for file_path in tqdm(files, desc="Denoising"):
                output_path = output_dir / f"{file_path.stem}_denoised{file_path.suffix or '.txt'}"
                rows.append(denoiser.denoise_file(file_path, output_path, plot=args.plot))
            summary_path = write_summary(rows, output_dir)
            print(f"Processed {len(rows)} files.")
            print(f"Summary: {summary_path}")
        else:
            result = denoiser.denoise_file(input_path, output_arg, plot=args.plot)
            print(f"Output: {result['output']}")
            if result["plot"]:
                print(f"Plot  : {result['plot']}")
            print(f"SNR   : {result['snr_db']:.2f} dB")
            print(f"Noise : {result['noise_reduction_percent']:.2f}% reduction")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

