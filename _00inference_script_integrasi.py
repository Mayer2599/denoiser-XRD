"""
Inference Script Terintegrasi: Denoise Single XRD File + CIF Generation
Load model AI Denoiser untuk membersihkan noise, lalu lanjutkan ke model deCIFer untuk prediksi struktur.
"""
import sys
import os
import pickle
import re
import glob

# Daftarkan folder deCIFer-main ke dalam sistem agar Python bisa membacanya
DECIFER_DIR = r"C:\Users\COMPUTER\Documents\xrdAI_withoutmatch3_v2\.1deCIFer_repo\deCIFer-main"
if DECIFER_DIR not in sys.path:
    sys.path.insert(0, DECIFER_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import traceback
from scipy.interpolate import interp1d
from models import get_model

# Output directories
DENOISING_OUTPUT_DIR = Path(r"C:\Users\COMPUTER\Documents\data_xrd\1b.Denoising")
CIF_OUTPUT_DIR = Path(r"C:\Users\COMPUTER\Documents\data_xrd\1c.cif-file")

# Import tkinter untuk GUI file picker
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available. GUI mode disabled.")


class Struct:
    """Simple recursive object wrapper for dict-like config data."""
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v

    def __getattr__(self, name):
        if name == 'condition_size' and hasattr(self, 'cond_size'):
            return self.cond_size
        if "size" in name or "dim" in name or "length" in name:
            return 8500
        if "drop" in name or "rate" in name:
            return 0.0
        return None


class TrainConfig(Struct):
    """Compatibility shim for legacy checkpoints pickled from __main__."""
    def __setstate__(self, state):
        self.__dict__.update(state)


class _CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__' and name == 'TrainConfig':
            return TrainConfig
        return super().find_class(module, name)


class _CompatibilityPickleModule:
    Unpickler = _CompatibilityUnpickler

    @staticmethod
    def load(file_obj, **kwargs):
        return _CompatibilityUnpickler(file_obj, **kwargs).load()

    @staticmethod
    def loads(data, **kwargs):
        from io import BytesIO
        return _CompatibilityUnpickler(BytesIO(data), **kwargs).load()


def load_checkpoint_compat(checkpoint_path, map_location):
    """Load checkpoints while remapping legacy __main__.TrainConfig references."""
    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
        pickle_module=_CompatibilityPickleModule,
    )


def infer_composition_from_path(path_like):
    """Infer a chemical formula like Fe3O4 from an input filename."""
    stem = Path(path_like).stem
    for suffix in ('_denoised', '-denoised', '_clean', '-clean', '_output', '-output'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    matches = re.findall(r'(?:[A-Z][a-z]?\d*)+', stem)
    if not matches:
        return None

    matches.sort(key=len, reverse=True)
    return matches[0]


def resolve_wavelength(wavelength):
    """Resolve common XRD wavelength aliases to Angstrom values."""
    if wavelength is None:
        return None
    if isinstance(wavelength, (int, float)):
        return float(wavelength)

    aliases = {
        'cuka': 1.5406,
        'cu': 1.5406,
        'cuka1': 1.5406,
        'moka': 0.7093,
        'mo': 0.7093,
        'feka': 1.9360,
        'co': 1.78897,
        'coka': 1.78897,
    }
    key = str(wavelength).strip().lower()
    return aliases.get(key)

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio (SNR)"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def select_file_via_gui(mode='input'):
    """Buka dialog file explorer untuk memilih file/folder"""
    if not TKINTER_AVAILABLE:
        return None
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    try:
        if mode == 'input':
            file_path = filedialog.askopenfilename(
                title="Pilih File XRD untuk Denoising",
                filetypes=[("XRD Files", "*.txt *.xy *.ASC *.asc *.dat"), ("All Files", "*.*")]
            )
        else:
            file_path = None
        return file_path if file_path else None
    finally:
        root.destroy()

class XRDDenoiser:
    """Class untuk denoise XRD data eksperimen"""
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading denoiser model from {model_path}...")
        loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        
        checkpoint = None
        config = {}
        model_state_dict = None
        
        if isinstance(loaded, dict):
            if 'state_dict' in loaded:
                checkpoint = loaded
                model_state_dict = checkpoint['state_dict']
                config = checkpoint.get('config', {}) or {}
            elif 'model_state_dict' in loaded:
                checkpoint = loaded
                model_state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('config', {}) or {}
            else:
                model_state_dict = loaded
        else:
            model_state_dict = loaded

        if isinstance(model_state_dict, dict):
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            model_state_dict = new_state_dict

        self.model_type = config.get('model_type', 'unet')
        base_channels = config.get('base_channels', 32)
        self.input_length = config.get('input_length', 8500)
        
        self.model = get_model(
            model_type=self.model_type,
            base_channels=base_channels,
            input_length=self.input_length
        ).to(self.device)
        
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        print("OK Denoiser loaded successfully")

    def load_xrd_file(self, filepath):
        filepath = Path(filepath)
        try:
            data = np.loadtxt(filepath, comments=['#', ';', '@', '/*', 'Peak'])
        except Exception as e1:
            try:
                with open(filepath, 'r') as f:
                    lines = [line.strip() for line in f 
                            if line.strip() 
                            and not line.startswith(('#', ';', '@', '/*')) 
                            and 'Peak' not in line
                            and 'Position' not in line
                            and 'Intensity' not in line]
                data = np.array([list(map(float, line.split())) for line in lines])
            except Exception as e2:
                raise ValueError(f"Failed to load XRD file")
        
        if data.ndim == 2:
            if data.shape[1] == 2:
                angles, intensity = data[:, 0], data[:, 1]
            else:
                angles, intensity = data[:, 0], data[:, -1]
        else:
            intensity = data
            angles = np.arange(len(intensity))
        
        return angles, intensity

    def resample(self, data, target_length):
        if len(data) == target_length: return data
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, target_length)
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        f = interp1d(x_old, data_clean, kind='linear', fill_value='extrapolate')
        return f(x_new)

    def preprocess(self, data):
        data_min_orig, data_max_orig = data.min(), data.max()
        data_safe = np.maximum(data, 0)
        data_anscombe = 2 * np.sqrt(data_safe + 3/8)
        data_min, data_max = data_anscombe.min(), data_anscombe.max()
        
        if data_max - data_min > 0:
            data_normalized = (data_anscombe - data_min) / (data_max - data_min)
        else:
            data_normalized = data_anscombe
            
        params = {'anscombe_min': data_min, 'anscombe_max': data_max}
        return data_normalized, params

    def inverse_preprocess(self, data, params):
        data_min, data_max = params['anscombe_min'], params['anscombe_max']
        if data_max - data_min > 0:
            data_denorm = data * (data_max - data_min) + data_min
        else:
            data_denorm = data
            
        data_inv = (data_denorm / 2) ** 2 - 3/8
        return np.maximum(data_inv, 0)

    def denoise(self, noisy_data):
        original_length = len(noisy_data)
        noisy_resampled = self.resample(noisy_data, self.input_length) if original_length != self.input_length else noisy_data.copy()
        
        noisy_preprocessed, params = self.preprocess(noisy_resampled)
        noisy_tensor = torch.FloatTensor(noisy_preprocessed).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            denoised_tensor = self.model(noisy_tensor)
            
        denoised_preprocessed = denoised_tensor.squeeze().cpu().numpy()
        denoised_data = self.inverse_preprocess(denoised_preprocessed, params)
        
        if original_length != self.input_length:
            denoised_data = self.resample(denoised_data, original_length)
            
        return denoised_data, params

    def denoise_file(self, input_path, output_path=None, plot=True):
        print(f"\n{'='*80}\nPROCESSING XRD FILE: {Path(input_path).name}\n{'='*80}")
        angles, noisy_intensity = self.load_xrd_file(input_path)
        print("  Denoising with UNet1D model...")
        denoised_intensity, _ = self.denoise(noisy_intensity)
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_data = np.column_stack([angles, denoised_intensity])
            np.savetxt(output_path, output_data, fmt='%.6f', delimiter='\t', header='2Theta\tIntensity', comments='')
            print(f"  OK Saved denoised file: {output_path}")
            
        return angles, noisy_intensity, denoised_intensity

# ======================================================================================
# CLASS BARU: WRAPPER UNTUK deCIFer (BERSIH TOTAL)
# ======================================================================================
class DeCIFerWrapper:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading deCIFer model from {checkpoint_path}...")
        
        try:
            from decifer.decifer_model import Decifer
            from decifer.decifer_model import DeciferConfig
            from decifer.tokenizer import Tokenizer
            try:
                from decifer.utility import (
                    bond_length_reasonableness_score,
                    evaluate_syntax_validity,
                    generate_continuous_xrd_from_cif,
                    is_formula_consistent,
                    is_sensible,
                    is_space_group_consistent,
                    replace_symmetry_loop_with_P1,
                    reinstate_symmetry_loop,
                    extract_space_group_symbol,
                )
                self.has_advanced_scoring = True
                self._replace_symmetry_loop_with_P1 = replace_symmetry_loop_with_P1
                self._reinstate_symmetry_loop = reinstate_symmetry_loop
                self._extract_space_group_symbol = extract_space_group_symbol
                self._has_symmetry_fix = True
            except Exception as scoring_import_error:
                bond_length_reasonableness_score = None
                evaluate_syntax_validity = None
                generate_continuous_xrd_from_cif = None
                is_formula_consistent = None
                is_sensible = None
                is_space_group_consistent = None
                self.has_advanced_scoring = False
                self._has_symmetry_fix = False
                print(f"  Warning: advanced CIF scoring unavailable ({scoring_import_error})")

            # Import pymatgen Structure for validation
            try:
                from pymatgen.core.structure import Structure
                self._Structure = Structure
                self._has_pymatgen = True
            except ImportError:
                self._Structure = None
                self._has_pymatgen = False
                print("  Warning: pymatgen tidak tersedia, validasi CIF dinonaktifkan")
            
            # 1. BACA FILE CHECKPOINT (BOBOT AI)
            checkpoint = load_checkpoint_compat(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get(
                'best_model_state',
                checkpoint.get('best_model', checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint)))
            )
            
            # 2. GUNAKAN MODEL_ARGS RESMI DARI CHECKPOINT
            model_args = checkpoint.get('model_args')
            if not isinstance(model_args, dict):
                raise ValueError("Checkpoint deCIFer tidak memiliki 'model_args' yang valid")

            self.expected_xrd_length = int(model_args.get('condition_size', 1000))
            self.qmin = float(checkpoint.get('config', {}).get('qmin', 0.0))
            self.qmax = float(checkpoint.get('config', {}).get('qmax', 10.0))
            self.qstep = float(checkpoint.get('config', {}).get('qstep', 0.01))
            self.wavelength = resolve_wavelength(checkpoint.get('config', {}).get('wavelength', 'CuKa'))
            model_config = DeciferConfig(**model_args)

            # 3. LOAD MODEL DENGAN KONFIGURASI ASLI
            self.model = Decifer(model_config)
            self.model.device = self.device
            self.tokenizer = Tokenizer()
            self._bond_length_reasonableness_score = bond_length_reasonableness_score
            self._evaluate_syntax_validity = evaluate_syntax_validity
            self._generate_continuous_xrd_from_cif = generate_continuous_xrd_from_cif
            self._is_formula_consistent = is_formula_consistent
            self._is_sensible = is_sensible
            self._is_space_group_consistent = is_space_group_consistent

            # 4. MASUKKAN BOBOT
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"  OK deCIFer model loaded successfully! (Pintu XRD ukuran {self.expected_xrd_length})")
            self.is_loaded = True
            
        except Exception as e:
            print(f"  Error meload model deCIFer: {e}")
            traceback.print_exc()
            self.is_loaded = False

    def _get_spacegroup_token_ids(self):
        """Get all space group token IDs from the tokenizer."""
        sg_ids = set()
        for token, tid in self.tokenizer.token_to_id.items():
            if token.endswith('_sg'):
                sg_ids.add(tid)
        return sg_ids

    def _get_allowed_sg_ids(self, spacegroup=None, crystal_system=None):
        """Get allowed space group token IDs based on constraints."""
        if spacegroup is None and crystal_system is None:
            return None  # No constraint

        allowed = set()
        token_to_id = self.tokenizer.token_to_id

        if spacegroup:
            # Direct space group constraint
            sg_token = spacegroup if spacegroup.endswith('_sg') else spacegroup + '_sg'
            # Also try with spaces replaced
            variants = [sg_token, sg_token.replace(' ', '')]
            for v in variants:
                if v in token_to_id:
                    allowed.add(token_to_id[v])
            if not allowed:
                print(f"  Warning: Space group '{spacegroup}' tidak ditemukan di tokenizer")
                return None

        if crystal_system:
            # Map crystal system name to space group number ranges
            cs_ranges = {
                'triclinic': (1, 2), 'monoclinic': (3, 15), 'orthorhombic': (16, 74),
                'tetragonal': (75, 142), 'trigonal': (143, 167),
                'hexagonal': (168, 194), 'cubic': (195, 230),
            }
            cs_key = crystal_system.lower().strip()
            if cs_key not in cs_ranges:
                print(f"  Warning: Crystal system '{crystal_system}' tidak dikenali")
                print(f"  Pilihan: {', '.join(cs_ranges.keys())}")
                return None
            sg_min, sg_max = cs_ranges[cs_key]

            try:
                from pymatgen.symmetry.groups import SpaceGroup as SG
                for num in range(sg_min, sg_max + 1):
                    try:
                        sg = SG.from_int_number(num)
                        sg_token = sg.symbol + '_sg'
                        if sg_token in token_to_id:
                            allowed.add(token_to_id[sg_token])
                    except Exception:
                        continue
            except ImportError:
                # Fallback: scan tokenizer for all _sg tokens
                from decifer.utility import space_group_to_crystal_system
                for token, tid in token_to_id.items():
                    if token.endswith('_sg'):
                        sg_symbol = token[:-3]  # Remove _sg
                        try:
                            # Try to determine crystal system
                            cs = space_group_to_crystal_system(sg_symbol)
                            if cs and cs.lower() == cs_key:
                                allowed.add(tid)
                        except Exception:
                            continue

        if not allowed:
            print(f"  Warning: Tidak ada space group yang cocok dengan constraint")
            return None

        print(f"  Constraint: {len(allowed)} space group token(s) diizinkan")
        return allowed

    def _generate_constrained(self, idx, max_new_tokens, cond_vec, temperature=0.7,
                               allowed_sg_ids=None, sg_logit_bias=5.0):
        """Constrained autoregressive generation with SOFT space group biasing.
        Instead of hard masking (which breaks generation), we ADD a logit bonus
        to the allowed space group tokens when the model is about to generate
        the space group symbol. This gently steers the model toward the target SG
        while keeping generation coherent."""
        NEWLINE_ID = self.tokenizer.token_to_id["\n"]
        PADDING_ID = self.tokenizer.padding_id
        SG_KEYWORD_ID = self.tokenizer.token_to_id["_symmetry_space_group_name_H-M"]
        SPACE_ID = self.tokenizer.token_to_id.get(" ", None)
        all_sg_ids = self._get_spacegroup_token_ids()

        prev_id = None
        sg_keyword_seen = False

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model.config.block_size else idx[:, -self.model.config.block_size:]
            logits, _ = self.model(idx_cond, cond_vec=cond_vec, start_indices_batch=[[0]])
            logits = logits[:, -1, :] / temperature

            # === SOFT CONSTRAINT: Space Group Biasing ===
            # Boost allowed SG tokens instead of masking disallowed ones
            if allowed_sg_ids is not None and sg_keyword_seen:
                for sg_id in allowed_sg_ids:
                    logits[:, sg_id] += sg_logit_bias

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            current_id = idx_next.item()

            # Track SG keyword position
            if current_id == SG_KEYWORD_ID:
                sg_keyword_seen = True
            elif sg_keyword_seen and current_id == SPACE_ID:
                sg_keyword_seen = True
            elif sg_keyword_seen and current_id in all_sg_ids:
                sg_keyword_seen = False
            elif sg_keyword_seen and current_id == NEWLINE_ID:
                sg_keyword_seen = False

            # Stop conditions
            if prev_id is not None and prev_id == NEWLINE_ID and current_id == NEWLINE_ID:
                break
            if current_id == PADDING_ID:
                idx = idx[:, :-1]
                break
            prev_id = current_id

        return idx

    def _build_prompt(self, composition=None):
        start_id = self.tokenizer.token_to_id["data_"]
        if composition:
            prompt_text = f"data_{composition}\n"
            prompt_tokens = self.tokenizer.tokenize_cif(prompt_text)
            prompt_ids = self.tokenizer.encode(prompt_tokens)
            return torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return torch.tensor([[start_id]], dtype=torch.long, device=self.device)

    def _decode_tokens_to_cif(self, token_ids):
        start_id = self.tokenizer.token_to_id["data_"]
        filtered_ids = [tok for tok in token_ids if tok != self.tokenizer.padding_id]
        if start_id in filtered_ids:
            filtered_ids = filtered_ids[filtered_ids.index(start_id):]

        cif_text = self.tokenizer.decode(filtered_ids).strip()
        if not cif_text.startswith("data_"):
            cif_text = "data_generated\n" + cif_text
        if not cif_text.endswith("\n"):
            cif_text += "\n"
        return cif_text

    def _fix_symmetry_in_cif(self, cif_text):
        """Post-process CIF: fix symmetry operations to match the declared space group.
        This replicates the official deCIFer experimental_pipeline.fix_symmetry_in_cif()."""
        if not getattr(self, '_has_symmetry_fix', False):
            return cif_text
        try:
            c = self._replace_symmetry_loop_with_P1(cif_text)
            sg = self._extract_space_group_symbol(c)
            if sg != "P 1":
                c = self._reinstate_symmetry_loop(c, sg)
            return c
        except Exception as e:
            print(f"  Warning: symmetry fix gagal ({e}), menggunakan CIF asli")
            return cif_text

    def _validate_cif_structure(self, cif_text):
        """Validate CIF by parsing it into a pymatgen Structure.
        Returns (Structure, None) on success or (None, error_message) on failure."""
        if not getattr(self, '_has_pymatgen', False):
            return None, "pymatgen not available"
        try:
            structure = self._Structure.from_str(cif_text, fmt="cif")
            return structure, None
        except Exception as e:
            return None, str(e)

    def _prepare_condition_vector(self, angles, intensities):
        """Prepare the XRD condition vector for deCIFer.
        Matches the official pipeline: normalize, crop, re-normalize, zero-pad endpoints,
        then interpolate onto the FIXED Q=[0,10] grid with n_points = expected_xrd_length."""
        angles = np.asarray(angles, dtype=np.float32)
        intensities = np.asarray(intensities, dtype=np.float32)
        intensities = np.nan_to_num(intensities, nan=0.0, posinf=0.0, neginf=0.0)

        if angles.size != intensities.size:
            raise ValueError("Panjang angle dan intensity tidak sama")

        # Konversi 2theta -> Q  (sama seperti pipeline resmi)
        if self.wavelength is not None:
            theta_rad = np.radians(angles / 2.0)
            q_values = (4.0 * np.pi / self.wavelength) * np.sin(theta_rad)
        else:
            q_values = angles

        order = np.argsort(q_values)
        q_values = q_values[order]
        intensities = intensities[order]

        q_actual_min = float(np.min(q_values))
        q_actual_max = float(np.max(q_values))
        q_min_crop = max(self.qmin, q_actual_min)
        q_max_crop = min(self.qmax, q_actual_max)
        if q_min_crop >= q_max_crop:
            raise ValueError("Rentang Q input tidak overlap dengan rentang model deCIFer")

        # Step 1: Full-range min-max normalization
        full_min = float(np.min(intensities))
        full_max = float(np.max(intensities))
        if full_max > full_min:
            intensity_normalized = (intensities - full_min) / (full_max - full_min)
        else:
            intensity_normalized = np.zeros_like(intensities, dtype=np.float32)

        # Step 2: Crop to Q range
        crop_mask = (q_values > q_min_crop) & (q_values < q_max_crop)
        q_crop = q_values[crop_mask]
        i_crop = intensity_normalized[crop_mask]
        if q_crop.size == 0:
            raise ValueError("Tidak ada data XRD pada rentang Q crop untuk deCIFer")

        # Step 3: Re-normalize cropped region to [0,1]
        crop_min = float(np.min(i_crop))
        crop_max = float(np.max(i_crop))
        if crop_max > crop_min:
            i_crop = (i_crop - crop_min) / (crop_max - crop_min)
        else:
            i_crop = np.zeros_like(i_crop, dtype=np.float32)

        # Step 4: Add zero endpoints at crop boundaries (match official pipeline)
        q_crop = np.concatenate(([q_min_crop], q_crop, [q_max_crop])).astype(np.float32)
        i_crop = np.concatenate(([0.0], i_crop, [0.0])).astype(np.float32)

        # Step 5: Interpolate onto FIXED Q=[0, 10] grid (critical: matches training data)
        # Official pipeline: Q_std = np.linspace(0, 10, n_points)
        q_std = np.linspace(0.0, 10.0, self.expected_xrd_length, dtype=np.float32)
        cond = np.interp(q_std, q_crop, i_crop).astype(np.float32)
        return torch.tensor(cond, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _space_group_number_to_crystal_system(self, sg_number):
        try:
            n = int(sg_number)
        except Exception:
            return None
        ranges = {
            'triclinic': (1, 2),
            'monoclinic': (3, 15),
            'orthorhombic': (16, 74),
            'tetragonal': (75, 142),
            'trigonal': (143, 167),
            'hexagonal': (168, 194),
            'cubic': (195, 230),
        }
        for cs, (min_n, max_n) in ranges.items():
            if min_n <= n <= max_n:
                return cs
        return None

    def _normalize_formula_for_prompt(self, formula):
        if not formula:
            return None
        return re.sub(r"\s+", "", str(formula))

    def _infer_formula_from_cif_text(self, cif_text):
        if not cif_text:
            return None
        match = re.search(r"_chemical_formula_sum\s+['\"]?([^\n'\"]+)['\"]?", cif_text)
        if match:
            return self._normalize_formula_for_prompt(match.group(1))
        first_line = cif_text.splitlines()[0].strip() if cif_text else ""
        if first_line.startswith("data_") and len(first_line) > 5:
            return self._normalize_formula_for_prompt(first_line[5:])
        return None

    def _infer_spacegroup_from_cif_text(self, cif_text):
        if not cif_text:
            return None
        if getattr(self, '_has_symmetry_fix', False):
            try:
                sg_symbol = self._extract_space_group_symbol(cif_text)
                return sg_symbol if sg_symbol else None
            except Exception:
                pass
        match = re.search(r"_symmetry_space_group_name_H-M\s+['\"]?([^\n'\"]+)['\"]?", cif_text)
        if match:
            return match.group(1).strip()
        return None

    def _resolve_reference_paths(self, reference_path):
        if not reference_path:
            return []
        ref = Path(reference_path)
        paths = []
        if ref.exists():
            if ref.is_file():
                paths = [ref]
            elif ref.is_dir():
                paths = list(ref.rglob("*.cif")) + list(ref.rglob("*.CIF"))
        else:
            for p in glob.glob(reference_path):
                pth = Path(p)
                if pth.exists() and pth.is_file():
                    paths.append(pth)

        # De-duplicate while preserving order
        seen = set()
        uniq = []
        for p in paths:
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    def _load_reference_candidates(self, reference_path, target_q, target_i, reference_max=200):
        ref_paths = self._resolve_reference_paths(reference_path)
        if not ref_paths:
            print(f"  Warning: Reference path tidak ditemukan atau kosong: {reference_path}")
            return []

        if reference_max and len(ref_paths) > reference_max:
            print(f"  Warning: Reference CIF terlalu banyak ({len(ref_paths)}).")
            print(f"  Membatasi ke {reference_max} file pertama untuk scoring.")
            ref_paths = ref_paths[:reference_max]

        reference_candidates = []
        for idx, ref_path in enumerate(ref_paths, start=1):
            try:
                cif_text = ref_path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                try:
                    cif_text = ref_path.read_text(errors='ignore')
                except Exception:
                    print(f"  Warning: Gagal membaca CIF reference: {ref_path}")
                    continue

            structure, val_error = self._validate_cif_structure(cif_text)
            score_info = self._score_candidate(cif_text, target_q, target_i)

            composition = None
            spacegroup = None
            sg_number = None
            crystal_system = None
            if structure is not None:
                try:
                    composition = self._normalize_formula_for_prompt(structure.composition.reduced_formula)
                except Exception:
                    composition = None
                try:
                    sg_symbol, sg_number = structure.get_space_group_info()
                    spacegroup = sg_symbol
                    crystal_system = self._space_group_number_to_crystal_system(sg_number)
                except Exception:
                    pass
            else:
                composition = self._infer_formula_from_cif_text(cif_text)
                spacegroup = self._infer_spacegroup_from_cif_text(cif_text)

            if structure is not None:
                score_info['score'] = score_info.get('score', 0) + 3.0
                score_info.setdefault('validity', {})
                score_info['validity']['pymatgen_valid'] = True
            else:
                score_info.setdefault('validity', {})
                score_info['validity']['pymatgen_valid'] = False
                if val_error:
                    score_info['validity']['pymatgen_error'] = val_error

            reference_candidates.append({
                'index': f"ref{idx}",
                'source': 'reference',
                'path': str(ref_path),
                'composition': composition,
                'spacegroup': spacegroup,
                'sg_number': sg_number,
                'crystal_system': crystal_system,
                'cif_text': cif_text,
                'structure': structure,
                **score_info,
            })

        return reference_candidates

    def _score_candidate(self, cif_text, target_q, target_i):
        result = {
            'score': float('-inf'),
            'xrd_score': -1.0,
            'bond_score': 0.0,
            'validity': {},
            'error': None,
        }
        try:
            if not getattr(self, 'has_advanced_scoring', False):
                header_score = float(cif_text.startswith('data_'))
                field_hits = sum(
                    field in cif_text for field in [
                        '_cell_length_a',
                        '_cell_length_b',
                        '_cell_length_c',
                        '_atom_site_type_symbol',
                        '_atom_site_fract_x',
                        '_symmetry_space_group_name_H-M',
                    ]
                )
                loop_hits = cif_text.count('loop_')
                result.update({
                    'score': 2.0 * header_score + 0.5 * field_hits + 0.25 * loop_hits,
                    'validity': {
                        'heuristic_only': True,
                        'field_hits': field_hits,
                        'loop_hits': loop_hits,
                    },
                })
                return result

            validity = self._evaluate_syntax_validity(cif_text, bond_length_acceptability_cutoff=0.9)
            bond_score = float(self._bond_length_reasonableness_score(cif_text))
            sensible = bool(self._is_sensible(cif_text))
            formula_ok = bool(self._is_formula_consistent(cif_text))
            spacegroup_ok = bool(self._is_space_group_consistent(cif_text))

            pred = self._generate_continuous_xrd_from_cif(
                cif_text,
                wavelength='CuKa',
                qmin=self.qmin,
                qmax=self.qmax,
                qstep=self.qstep,
                fwhm_range=(0.05, 0.05),
                eta_range=(0.5, 0.5),
                noise_range=None,
                intensity_scale_range=None,
                mask_prob=None,
                debug=False,
            )
            if pred is None:
                raise ValueError("generate_continuous_xrd_from_cif returned None")

            pred_q = np.asarray(pred['q'], dtype=np.float32)
            pred_i = np.asarray(pred['iq'], dtype=np.float32)
            pred_i = np.interp(target_q, pred_q, pred_i).astype(np.float32)
            pred_max = float(np.max(pred_i))
            if pred_max > 0:
                pred_i = pred_i / pred_max

            mse = float(np.mean((target_i - pred_i) ** 2))
            if np.std(target_i) > 0 and np.std(pred_i) > 0:
                corr = float(np.corrcoef(target_i, pred_i)[0, 1])
            else:
                corr = -1.0
            xrd_score = max(-1.0, corr) - mse

            score = (
                3.0 * float(validity.get('formula', False))
                + 2.0 * float(validity.get('site_multiplicity', False))
                + 2.0 * float(validity.get('spacegroup', False))
                + 1.0 * float(validity.get('bond_length', False))
                + 1.0 * float(sensible)
                + 2.5 * bond_score
                + 5.0 * xrd_score
                + 0.5 * float(formula_ok)
                + 0.5 * float(spacegroup_ok)
            )

            result.update({
                'score': score,
                'xrd_score': xrd_score,
                'bond_score': bond_score,
                'validity': {
                    **validity,
                    'sensible': sensible,
                    'formula_consistent': formula_ok,
                    'spacegroup_consistent': spacegroup_ok,
                },
            })
        except Exception as e:
            result['error'] = str(e)
        return result

    def generate_cif(self, angles, intensities, output_cif_path, composition=None,
                     num_candidates=32, temperature=0.7,
                     spacegroup=None, crystal_system=None,
                     reference_cif=None, reference_topk=3, reference_max=200,
                     reference_min_score=0.25, reference_only=False):
        if not self.is_loaded:
            return
            
        try:
            print("  Memproses prediksi struktur dari pola XRD...")

            xrd_tensor = self._prepare_condition_vector(angles, intensities)
            target_q = np.linspace(self.qmin, self.qmax, self.expected_xrd_length, dtype=np.float32)
            target_i = xrd_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            target_max = float(np.max(target_i))
            if target_max > 0:
                target_i = target_i / target_max

            # Reference-based matching (opsional)
            reference_candidates = []
            reference_best = None
            if reference_cif:
                if not getattr(self, 'has_advanced_scoring', False):
                    print("  Warning: Reference CIF dipakai tanpa simulasi XRD (advanced scoring tidak tersedia).")
                reference_candidates = self._load_reference_candidates(
                    reference_cif,
                    target_q,
                    target_i,
                    reference_max=reference_max,
                )
                if reference_candidates:
                    reference_candidates.sort(key=lambda item: item.get('score', float('-inf')), reverse=True)
                    reference_best = reference_candidates[0]
                    print("  Reference CIF terbaik (top-k):")
                    for rank, cand in enumerate(reference_candidates[:max(1, int(reference_topk))], start=1):
                        fname = Path(cand.get('path', 'unknown')).name
                        print(
                            f"    {rank}. {fname} | score={cand.get('score', float('-inf')):.3f} "
                            f"xrd={cand.get('xrd_score', -1.0):.3f}"
                        )

                    # Gunakan info dari reference untuk constraint jika user belum memberi
                    if composition is None and reference_best.get('composition'):
                        composition = reference_best.get('composition')
                        print(f"  Constraint dari reference: composition={composition}")
                    if spacegroup is None and reference_best.get('spacegroup'):
                        spacegroup = reference_best.get('spacegroup')
                        print(f"  Constraint dari reference: spacegroup={spacegroup}")
                    if crystal_system is None and reference_best.get('crystal_system'):
                        crystal_system = reference_best.get('crystal_system')
                        print(f"  Constraint dari reference: crystal_system={crystal_system}")

                    if reference_only:
                        if reference_best.get('score', float('-inf')) < float(reference_min_score):
                            print(
                                f"  Warning: Reference best score ({reference_best.get('score', float('-inf')):.3f}) "
                                f"di bawah threshold {reference_min_score:.3f}. Tetap simpan (reference_only)."
                            )

                        output_path = Path(output_cif_path)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, 'w') as f:
                            f.write(reference_best['cif_text'])
                        print(f"  OK Saved reference CIF: {output_path}")

                        summary_path = output_path.with_name(f"{output_path.stem}_ranking.txt")
                        with open(summary_path, 'w') as f:
                            for rank, candidate in enumerate(reference_candidates, start=1):
                                is_valid = "VALID" if candidate.get('structure') is not None else "INVALID"
                                f.write(
                                    f"rank={rank}\t{is_valid}\tindex={candidate['index']}\t"
                                    f"score={candidate['score']:.6f}\t"
                                    f"xrd={candidate['xrd_score']:.6f}\t"
                                    f"bond={candidate['bond_score']:.6f}\t"
                                    f"source={candidate.get('source','reference')}\t"
                                    f"path={candidate.get('path','')}\t"
                                    f"validity={candidate['validity']}\t"
                                    f"error={candidate['error']}\n"
                                )
                        print(f"  OK Saved reference ranking: {summary_path}")
                        return

            # Setup constraints (after reference inference)
            allowed_sg_ids = self._get_allowed_sg_ids(spacegroup=spacegroup, crystal_system=crystal_system)
            use_constrained = allowed_sg_ids is not None
            if use_constrained:
                print(f"  Mode: CONSTRAINED generation (soft SG biasing aktif)")
            else:
                print(f"  Mode: UNCONSTRAINED generation")

            candidates = []
            for candidate_idx in range(int(num_candidates)):
                idx = self._build_prompt(composition=composition)

                with torch.no_grad():
                    if use_constrained:
                        hasil_prediksi = self._generate_constrained(
                            idx=idx,
                            max_new_tokens=2000,
                            cond_vec=xrd_tensor,
                            temperature=temperature,
                            allowed_sg_ids=allowed_sg_ids,
                        )
                    else:
                        hasil_prediksi = self.model.generate(
                            idx=idx,
                            max_new_tokens=2000,
                            cond_vec=xrd_tensor,
                            start_indices_batch=[[0]],
                            temperature=temperature,
                            disable_pbar=True,
                        )

                if isinstance(hasil_prediksi, torch.Tensor):
                    angka_token = hasil_prediksi.detach().cpu().tolist()[0]
                    generated_cif_text = self._decode_tokens_to_cif(angka_token)

                    # POST-PROCESSING: Fix symmetry operations (match official pipeline)
                    generated_cif_text = self._fix_symmetry_in_cif(generated_cif_text)

                    # VALIDASI: cek apakah CIF bisa diparsing jadi struktur kristal
                    structure, val_error = self._validate_cif_structure(generated_cif_text)
                    if structure is not None:
                        try:
                            sg_symbol = structure.get_space_group_info()[0]
                        except Exception:
                            sg_symbol = "?"
                        print(f"    ✓ Candidate {candidate_idx+1}: {structure.formula}, "
                              f"SG={sg_symbol}, "
                              f"a={structure.lattice.a:.3f} b={structure.lattice.b:.3f} c={structure.lattice.c:.3f}")
                    elif val_error:
                        print(f"    ✗ Candidate {candidate_idx+1}: pymatgen validation gagal: {val_error}")

                    score_info = self._score_candidate(generated_cif_text, target_q, target_i)

                    # Bonus skor jika CIF bisa diparsing jadi struktur valid
                    if structure is not None:
                        score_info['score'] = score_info.get('score', 0) + 3.0
                        score_info['validity']['pymatgen_valid'] = True

                        # Bonus skor jika SG cocok dengan constraint
                        if spacegroup or crystal_system:
                            try:
                                sg_info = structure.get_space_group_info()
                                actual_sg = sg_info[0]  # Symbol
                                actual_sg_num = sg_info[1]  # Number
                                sg_match = False
                                if spacegroup:
                                    # Match by symbol
                                    target_sg = spacegroup.replace('_sg', '').strip()
                                    sg_match = (actual_sg == target_sg)
                                if crystal_system:
                                    # Match by crystal system
                                    cs_ranges = {
                                        'triclinic': (1, 2), 'monoclinic': (3, 15), 'orthorhombic': (16, 74),
                                        'tetragonal': (75, 142), 'trigonal': (143, 167),
                                        'hexagonal': (168, 194), 'cubic': (195, 230),
                                    }
                                    cs_key = crystal_system.lower()
                                    if cs_key in cs_ranges:
                                        sg_min, sg_max = cs_ranges[cs_key]
                                        sg_match = sg_match or (sg_min <= actual_sg_num <= sg_max)
                                if sg_match:
                                    score_info['score'] += 10.0  # Bonus besar untuk SG yang cocok
                                    score_info['validity']['sg_match'] = True
                                    print(f"      ★ SG cocok dengan target!")
                                else:
                                    score_info['validity']['sg_match'] = False
                            except Exception:
                                pass
                    else:
                        score_info['validity']['pymatgen_valid'] = False
                        score_info['validity']['pymatgen_error'] = val_error

                    candidate = {
                        'index': candidate_idx + 1,
                        'cif_text': generated_cif_text,
                        'structure': structure,
                        **score_info,
                    }
                else:
                    candidate = {
                        'index': candidate_idx + 1,
                        'cif_text': str(hasil_prediksi),
                        'structure': None,
                        'score': float('-inf'),
                        'xrd_score': -1.0,
                        'bond_score': 0.0,
                        'validity': {},
                        'error': 'Model output is not a tensor',
                    }

                candidates.append(candidate)
                print(
                    f"  Candidate {candidate['index']}/{num_candidates}: "
                    f"score={candidate['score']:.3f}, "
                    f"xrd={candidate['xrd_score']:.3f}, "
                    f"bond={candidate['bond_score']:.3f}"
                )

            if reference_candidates:
                candidates.extend(reference_candidates)

            if not candidates:
                raise ValueError("Tidak ada kandidat CIF yang berhasil digenerate")

            # Filter: hanya simpan kandidat yang VALID (bisa diparsing pymatgen)
            valid_candidates = [c for c in candidates if c.get('structure') is not None]
            all_candidates_sorted = sorted(candidates, key=lambda item: item['score'], reverse=True)
            valid_candidates.sort(key=lambda item: item['score'], reverse=True)

            print(f"\n  === HASIL: {len(valid_candidates)}/{len(candidates)} kandidat valid ===")

            output_path = Path(output_cif_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if valid_candidates:
                best_candidate = valid_candidates[0]

                # Simpan CIF terbaik
                with open(output_path, 'w') as f:
                    f.write(best_candidate['cif_text'])
                print(f"  OK Saved best CIF: {output_path}")
                print(
                    f"  Best candidate: score={best_candidate['score']:.3f}, "
                    f"xrd={best_candidate['xrd_score']:.3f}, "
                    f"bond={best_candidate['bond_score']:.3f}"
                )
                s = best_candidate['structure']
                print(f"  Formula: {s.formula}")
                try:
                    print(f"  Space Group: {s.get_space_group_info()[0]}")
                except Exception:
                    print("  Space Group: (tidak dapat ditentukan)")
                print(f"  Lattice: a={s.lattice.a:.4f} b={s.lattice.b:.4f} c={s.lattice.c:.4f}")
                print(f"  Angles: α={s.lattice.alpha:.2f} β={s.lattice.beta:.2f} γ={s.lattice.gamma:.2f}")
                print(f"  Jumlah atom dalam unit cell: {len(s)}")

            else:
                # Fallback: jika tidak ada kandidat valid, simpan yang skor tertinggi
                best_candidate = all_candidates_sorted[0]
                with open(output_path, 'w') as f:
                    f.write(best_candidate['cif_text'])
                print(f"  ⚠ WARNING: Tidak ada kandidat valid! Menyimpan kandidat terbaik (belum tervalidasi)")
                print(f"  Saved fallback CIF: {output_path}")

            # Simpan ranking lengkap (semua kandidat, valid dan tidak)
            summary_path = output_path.with_name(f"{output_path.stem}_ranking.txt")
            with open(summary_path, 'w') as f:
                for rank, candidate in enumerate(all_candidates_sorted, start=1):
                    is_valid = "VALID" if candidate.get('structure') is not None else "INVALID"
                    f.write(
                        f"rank={rank}\t{is_valid}\tindex={candidate['index']}\tscore={candidate['score']:.6f}\t"
                        f"xrd={candidate['xrd_score']:.6f}\tbond={candidate['bond_score']:.6f}\t"
                        f"source={candidate.get('source','model')}\t"
                        f"validity={candidate['validity']}\terror={candidate['error']}\n"
                    )
            print(f"  OK Saved candidate ranking: {summary_path}")
            
        except Exception as e:
            print("\n" + "🚨"*25)
            print("   DETEKSI LOKASI ERROR SECARA DETAIL (X-RAY)")
            print("🚨"*25)
            traceback.print_exc()
            print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="AI Pipeline: Denoise Experimental XRD Data & Generate CIF",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained denoiser model checkpoint (.pth, .ckpt)')
    parser.add_argument('--decifer', type=str, default=None,
                        help='Path to deCIFer checkpoint (decifer_v1_ckpt.pt) untuk men-generate CIF')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to experimental XRD file (.txt, .xy, .ASC)')
    parser.add_argument('--output', type=str, default=None,
                        help='(Ignored) output denoised otomatis ke 1b.Denoising')
    parser.add_argument('--gui', action='store_true',
                        help='Open Windows Explorer to select input file interactively')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating comparison plot')
    parser.add_argument('--composition', type=str, default=None,
                        help='Optional formula prompt for deCIFer, e.g. Fe3O4')
    parser.add_argument('--num-candidates', type=int, default=32,
                        help='Number of deCIFer candidates to sample before auto-selecting the best')
    parser.add_argument('--decifer-temperature', type=float, default=0.7,
                        help='Sampling temperature for deCIFer candidate generation (lebih rendah = lebih konservatif)')
    parser.add_argument('--spacegroup', type=str, default=None,
                        help='Constrain space group, e.g. "Fd-3m" untuk Fe3O4 magnetite')
    parser.add_argument('--crystal-system', type=str, default=None,
                        choices=['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal',
                                 'trigonal', 'hexagonal', 'cubic'],
                        help='Constrain crystal system (e.g. cubic, hexagonal)')
    parser.add_argument('--reference-cif', type=str, default=None,
                        help='Path ke CIF reference (file/folder/glob) untuk matching XRD')
    parser.add_argument('--reference-topk', type=int, default=3,
                        help='Jumlah reference CIF terbaik yang ditampilkan')
    parser.add_argument('--reference-max', type=int, default=200,
                        help='Batas maksimum CIF reference yang di-scan untuk scoring')
    parser.add_argument('--reference-min-score', type=float, default=0.25,
                        help='Threshold skor reference untuk warning (reference_only)')
    parser.add_argument('--reference-only', action='store_true',
                        help='Jika aktif, simpan CIF terbaik dari reference tanpa generate model')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    if args.gui:
        if not TKINTER_AVAILABLE:
            sys.exit(1)
        input_path = select_file_via_gui(mode='input')
        if not input_path: sys.exit(0)
        args.input = input_path
    
    if not args.input:
        parser.print_help()
        sys.exit(1)

    input_path_obj = Path(args.input)
    DENOISING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output = str(DENOISING_OUTPUT_DIR / f"{input_path_obj.stem}_denoised.txt")
    if args.decifer:
        CIF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. RUN DENOISER
    try:
        denoiser = XRDDenoiser(args.model, device=args.device)
        angles, noisy, denoised = denoiser.denoise_file(
            input_path=args.input,
            output_path=args.output,
            plot=not args.no_plot
        )
    except Exception as e:
        print(f"Error during denoising: {e}")
        sys.exit(1)

    # 2. RUN DECIFER 
    if args.decifer:
        print(f"\n{'='*80}\nPROCESSING CIF GENERATION\n{'='*80}")
        try:
            cif_output_path = str(CIF_OUTPUT_DIR / f"{input_path_obj.stem}.cif")
            composition = args.composition or infer_composition_from_path(args.input)
            if composition:
                print(f"  Using composition prompt: {composition}")
            else:
                print("  Warning: composition prompt tidak terdeteksi; generasi CIF bisa kurang stabil.")
            decifer_model = DeCIFerWrapper(args.decifer, device=args.device)
            decifer_model.generate_cif(
                angles,
                denoised,
                cif_output_path,
                composition=composition,
                num_candidates=args.num_candidates,
                temperature=args.decifer_temperature,
                spacegroup=args.spacegroup,
                crystal_system=args.crystal_system,
                reference_cif=args.reference_cif,
                reference_topk=args.reference_topk,
                reference_max=args.reference_max,
                reference_min_score=args.reference_min_score,
                reference_only=args.reference_only,
            )
        except Exception as e:
             print(f"Error during CIF generation: {e}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Input file      : {args.input}")
    print(f"Denoised file   : {args.output}")
    if args.decifer:
        print(f"CIF file        : {cif_output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
