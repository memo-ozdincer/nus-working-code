# # Google Colab Version
# # If you are running this from Colab, you need to comment out:
# !pip install pytorch_lightning rich

# Experiment
#Causal Masking in SelfAttentionBlock
#Dilated Convolutions in TemporalBlock

#  Physics-Informed Attention-TCN for I-V Curve Reconstruction in PyTorch
#
# MODIFICATION: This script has been updated to align more closely with a
# previous Keras implementation for better comparison.
#   1. Swapped LayerNorm for BatchNorm1d in the parameter MLP.
#   2. Removed weight_norm from TCN convolutional layers.
#   3. Changed scalar feature scaling from StandardScaler to MinMaxScaler.
#   4. Set training precision to full 32-bit float.
#   5. Corrected the convexity loss term to be physically accurate.
#
# REFACTOR & PATCH: This script uses a memory-efficient architecture and
# integrates a user-provided patch for superior plot reconstruction.
#
# --- ARCHITECTURE ---
#   - Hybrid Attention-Augmented Temporal Convolutional Network (TCN).
#   - Swappable positional embeddings (Fourier, Clipped Fourier, Gaussian RBF).
#   - Physics-informed loss function (MSE, Monotonicity, Convexity, Curvature).
#
# --- WORKFLOW ---
#   1. Configuration: All hyperparameters are defined in a single `CONFIG` dict.
#   2. Preprocessing: A one-time function processes raw data, applies PCHIP
#      interpolation, creates features, and saves training slices to a .npz file
#      and fine-grid curves to memory-mapped files.
#   3. Data Loading: A PyTorch Lightning `DataModule` efficiently handles
#      loading data for train, validation, and test sets.
#   4. Model: The `PhysicsIVSystem` LightningModule encapsulates the entire
#      model, training logic, and optimizer configuration.
#   5. Training: The `pl.Trainer` automates the training loop, with a custom
#      callback that generates high-fidelity reconstructed plots on test completion.
#
# ==============================================================================
import os
import logging
from pathlib import Path
from datetime import datetime
import math
import typing

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Added all missing imports for a self-contained script
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import PchipInterpolator
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
#   CONFIGURATIONS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Paths to input/output data
# --- MAKE SURE TO UPDATE THESE PATHS IF USING YOUR OWN DRIVE ---
# Using placeholder paths for non-Colab environments.
try:
    from google.colab import drive
    # If in Colab, assume drive is mounted or will be mounted.
    INPUT_FILE_PARAMS = "/content/drive/MyDrive/Colab Notebooks/Data_100k/LHS_parameters_m.txt"
    INPUT_FILE_IV = "/content/drive/MyDrive/Colab Notebooks/Data_100k/iV_m.txt"
except ImportError:
    # Use local paths if not in Colab. Create dummy files if they don't exist.
    # NOTE: You will need to replace these with your actual data.
    INPUT_FILE_PARAMS = "./LHS_parameters_m.txt"
    INPUT_FILE_IV = "./iV_m.txt"

OUTPUT_DIR = Path("./lightning_output")

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Path("./data/processed").mkdir(parents=True, exist_ok=True)

# Main configuration dictionary with best hyperparameters from HPO
CONFIG = {
    "train": {
        "seed": 42,
        "run_name": f"AttentionTCN-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    },
    "model": {
        "param_dim": 34,
        "dense_units": [256, 128, 128],
        "filters": [128, 64],
        "kernel": 5,
        "heads": 4,
        "dropout": 0.03631232181608377,
        "embedding_type": 'gaussian',
        "gaussian_bands": 18,
        "gaussian_sigma": 0.07749512610240868,
        "loss_weights": {
            "mse": 0.98,
            "mono": 0.005,
            "convex": 0.005,
            "excurv": 0.01,
            "excess_threshold": 0.8,
        },
    },
    "optimizer": {
        "lr": 0.005545402750717978,
        "weight_decay": 5.403751961152276e-05,
        "final_lr_ratio": 0.00666,
        "warmup_epochs": 7,
    },
    "dataset": {
        "paths": {
            "params_csv": INPUT_FILE_PARAMS,
            "iv_raw_txt": INPUT_FILE_IV,
            "output_dir": "./data/processed",
            "preprocessed_npz": "./data/processed/preprocessed_data.npz",
            "param_transformer": "./data/processed/param_transformer.joblib",
            "scalar_transformer": "./data/processed/scalar_transformer.joblib",
            "v_fine_memmap": "./data/processed/v_fine_curves.mmap",
            "i_fine_memmap": "./data/processed/i_fine_curves.mmap",
        },
        "pchip": {
            "v_max": 1.4,
            "n_fine": 2000,
            "n_pre_mpp": 3,
            "n_post_mpp": 4,
            "seq_len": 8,
        },
        "dataloader": {
            "batch_size": 128,
            "num_workers": os.cpu_count() // 2 if os.cpu_count() else 1,
            "pin_memory": True,
        },
        "curvature_weighting": {
            "alpha": 4.0,
            "power": 1.5,
        },
    },
    "trainer": {
        "max_epochs": 12,
        "accelerator": "auto",
        "devices": "auto",
        "precision": "32-true",
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 25,
    },
}

# Column names for the 31 device parameters
COLNAMES = [
    'lH','lP','lE', 'muHh','muPh','muPe','muEe','NvH','NcH','NvE','NcE','NvP','NcP',
    'chiHh','chiHe','chiPh','chiPe','chiEh','chiEe', 'Wlm','Whm', 'epsH','epsP','epsE',
    'Gavg','Aug','Brad','Taue','Tauh','vII','vIII'
]

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#   UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)

# ##############################################################################
#   START OF MODIFIED SECTION
# ##############################################################################
def process_iv_with_pchip(iv_raw, full_v_grid, n_pre, n_post, v_max, n_fine) -> typing.Optional[tuple]:
    """Finds key points (Isc, Voc, MPP) and extracts a fixed-length slice."""
    seq_len = n_pre + 1 + n_post
    try:
        if np.count_nonzero(~np.isnan(iv_raw)) < 4: return None
        pi = PchipInterpolator(full_v_grid, iv_raw, extrapolate=False)
        v_fine = np.linspace(0, v_max, n_fine)
        i_fine = pi(v_fine)
        valid_mask = ~np.isnan(i_fine)
        v_fine, i_fine = v_fine[valid_mask], i_fine[valid_mask]
        if v_fine.size < 2: return None
        zero_cross_idx = np.where(i_fine <= 0)[0]
        voc_v = v_fine[zero_cross_idx[0]] if len(zero_cross_idx) > 0 else v_fine[-1]
        v_search_mask = v_fine <= voc_v
        v_search, i_search = v_fine[v_search_mask], i_fine[v_search_mask]
        if v_search.size == 0: return None
        power = v_search * i_search
        mpp_idx = np.argmax(power)
        v_mpp = v_search[mpp_idx]
        v_pre_mpp = np.linspace(v_search[0], v_mpp, n_pre + 2, endpoint=True)[:-1]
        v_post_mpp = np.linspace(v_mpp, v_search[-1], n_post + 2, endpoint=True)[1:]
        v_mpp_grid = np.unique(np.concatenate([v_pre_mpp, v_post_mpp]))
        v_slice = np.interp(np.linspace(0, 1, seq_len), np.linspace(0, 1, len(v_mpp_grid)), v_mpp_grid)
        i_slice = pi(v_slice)
        if np.any(np.isnan(i_slice)) or i_slice.shape[0] != seq_len: return None

        # FIX: Clip interpolated values to the valid range of float16 before casting.
        # This prevents the `RuntimeWarning: overflow encountered in cast` and avoids
        # saving `inf` values to the memory-mapped files.
        f16_info = np.finfo(np.float16)
        v_fine_clipped = np.clip(v_fine, f16_info.min, f16_info.max)
        i_fine_clipped = np.clip(i_fine, f16_info.min, f16_info.max)

        return (v_slice.astype(np.float32),
                i_slice.astype(np.float32),
                (v_fine_clipped.astype(np.float16), i_fine_clipped.astype(np.float16)))

    except (ValueError, IndexError):
        return None
# ##############################################################################
#   END OF MODIFIED SECTION
# ##############################################################################

def normalize_and_scale_by_isc(curve: np.ndarray) -> tuple[float, np.ndarray]:
    isc_val = float(curve[0])
    return isc_val, (2.0 * (curve / isc_val) - 1.0).astype(np.float32)

def compute_curvature_weights(y_curves: np.ndarray, alpha: float, power: float) -> np.ndarray:
    padded = np.pad(y_curves, ((0, 0), (1, 1)), mode='edge')
    kappa = np.abs(padded[:, 2:] - 2 * padded[:, 1:-1] + padded[:, :-2])
    max_kappa = np.max(kappa, axis=1, keepdims=True)
    max_kappa[max_kappa < 1e-9] = 1.0
    return (1.0 + alpha * np.power(kappa / max_kappa, power)).astype(np.float32)

def get_param_transformer(colnames: list[str]) -> ColumnTransformer:
    param_defs = {
        'layer_thickness': ['lH', 'lP', 'lE'],
        'material_properties': ['muHh', 'muPh', 'muPe', 'muEe', 'NvH', 'NcH', 'NvE', 'NcE', 'NvP', 'NcP', 'chiHh', 'chiHe', 'chiPh', 'chiPe', 'chiEh', 'chiEe', 'epsH', 'epsP', 'epsE'],
        'contacts': ['Wlm', 'Whm'],
        'recombination_gen': ['Gavg', 'Aug', 'Brad', 'Taue', 'Tauh', 'vII', 'vIII']
    }
    transformers = []
    for group, cols in param_defs.items():
        actual_cols = [c for c in cols if c in colnames]
        if not actual_cols: continue
        steps = [('robust', RobustScaler()), ('minmax', MinMaxScaler(feature_range=(-1, 1)))]
        if group == 'material_properties':
            steps.insert(0, ('log1p', FunctionTransformer(func=np.log1p)))
        transformers.append((group, Pipeline(steps), actual_cols))
    return ColumnTransformer(transformers, remainder='passthrough')

def denormalize(scaled_current, isc):
    is_tensor = isinstance(scaled_current, torch.Tensor)
    if is_tensor:
        isc = isc.unsqueeze(1)
    else:
        isc = isc[:, np.newaxis]
    return (scaled_current + 1.0) / 2.0 * isc

# ──────────────────────────────────────────────────────────────────────────────
#   MODEL COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────

class FourierFeatures(nn.Module):
    def __init__(self, num_bands: int, v_max: float = 1.4):
        super().__init__()
        self.v_max = v_max
        B = torch.logspace(0, 3, num_bands)
        self.register_buffer('B', B, persistent=False)
        self.out_dim = num_bands * 2
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        two_pi = torch.tensor(2 * math.pi, device=v.device, dtype=v.dtype)
        v_norm = v / self.v_max
        v_proj = v_norm.unsqueeze(-1) * self.B
        return torch.cat([(two_pi * v_proj).sin(), (two_pi * v_proj).cos()], dim=-1)

class ClippedFourierFeatures(nn.Module):
    def __init__(self, num_bands: int, v_max: float = 1.4):
        super().__init__()
        self.v_max = v_max
        B = torch.logspace(0, 3, num_bands)
        B_mask = (B >= 1.0).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('B', B, persistent=False)
        self.register_buffer('B_mask', B_mask, persistent=False)
        self.out_dim = num_bands * 2
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        two_pi = torch.tensor(2 * math.pi, device=v.device, dtype=v.dtype)
        v_norm = v / self.v_max
        v_proj = v_norm.unsqueeze(-1) * self.B
        sines = (two_pi * v_proj).sin() * self.B_mask
        coses = (two_pi * v_proj).cos() * self.B_mask
        return torch.cat([sines, coses], dim=-1)

class GaussianRBFFeatures(nn.Module):
    def __init__(self, num_bands: int, sigma: float = 0.1, v_max: float = 1.4):
        super().__init__()
        self.v_max = v_max
        self.sigma = sigma
        mu = torch.linspace(0, 1, num_bands)
        self.register_buffer('mu', mu, persistent=False)
        self.out_dim = num_bands
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_norm = v / self.v_max
        diff = v_norm.unsqueeze(-1) - self.mu
        return torch.exp(-0.5 * (diff / self.sigma)**2)

def make_positional_embedding(cfg: dict) -> nn.Module:
    etype = cfg['model']['embedding_type']
    if etype == 'fourier':
        return FourierFeatures(cfg['model']['fourier_bands'], cfg['dataset']['pchip']['v_max'])
    elif etype == 'fourier_clipped':
        return ClippedFourierFeatures(cfg['model']['fourier_bands'], cfg['dataset']['pchip']['v_max'])
    elif etype == 'gaussian':
        return GaussianRBFFeatures(cfg['model']['gaussian_bands'], cfg['model']['gaussian_sigma'], cfg['dataset']['pchip']['v_max'])
    raise ValueError(f"Unknown embedding type: {etype}")

def physics_loss(y_pred: torch.Tensor, y_true: torch.Tensor, sample_w: torch.Tensor, loss_w: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mse_loss = (((y_true - y_pred)**2) * sample_w).mean()
    mono_violations = torch.relu(y_pred[:, 1:] - y_pred[:, :-1])
    mono_loss = mono_violations.pow(2).mean()
    convex_violations = torch.relu(2 * y_pred[:, 1:-1] - y_pred[:, :-2] - y_pred[:, 2:])
    convex_loss = convex_violations.pow(2).mean()
    curvature = torch.abs(y_pred[:, :-2] - 2 * y_pred[:, 1:-1] + y_pred[:, 2:])
    excurv_violations = torch.relu(curvature - loss_w['excess_threshold'])
    excurv_loss = excurv_violations.pow(2).mean()
    total_loss = (loss_w['mse'] * mse_loss + loss_w['mono'] * mono_loss + loss_w['convex'] * convex_loss + loss_w['excurv'] * excurv_loss)
    return total_loss, {'mse': mse_loss, 'mono': mono_loss, 'convex': convex_loss, 'excurv': excurv_loss}

class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class TemporalBlock(nn.Module):
    # I.2: Add dilation parameter to the constructor
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float, dilation: int):
        super().__init__()
        # I.2: Adjust padding based on kernel size and dilation to maintain sequence length
        self.padding = ((kernel_size - 1) * dilation, 0)
        # I.2: Apply dilation to convolutional layers
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.act1 = nn.GELU()
        self.norm1 = ChannelLayerNorm(out_ch)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.act2 = nn.GELU()
        self.norm2 = ChannelLayerNorm(out_ch)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = F.pad(x, self.padding)
        out = self.drop1(self.norm1(self.act1(self.conv1(out))))
        out = F.pad(out, self.padding)
        out = self.drop2(self.norm2(self.act2(self.conv2(out))))
        return out + res

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # I.1: Add causal masking for temporal causality
        seq_len = x.size(1)
        # Create an upper-triangular mask to prevent attention to future positions.
        # True values in the mask indicate positions that should be ignored.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return self.norm(x + self.drop(attn_out))

# ──────────────────────────────────────────────────────────────────────────────
#   PYTORCH LIGHTNING DATA & MODEL MODULES
# ──────────────────────────────────────────────────────────────────────────────

class IVDataset(Dataset):
    def __init__(self, cfg: dict, split: str, param_tf, scalar_tf):
        self.cfg = cfg
        self.split = split
        data = np.load(cfg['dataset']['paths']['preprocessed_npz'], allow_pickle=True)
        split_labels = data['split_labels']
        indices = np.where(split_labels == split)[0]
        self.v_slices = torch.from_numpy(data['v_slices'][indices])
        self.i_slices_scaled = torch.from_numpy(data['i_slices_scaled'][indices])
        self.sample_weights = torch.from_numpy(data['sample_weights'][indices])
        params_df = pd.read_csv(cfg['dataset']['paths']['params_csv'], header=None, names=COLNAMES)
        params_df_valid = params_df.iloc[data['valid_indices']].reset_index(drop=True)
        scalar_df = pd.DataFrame({'I_ref': data['i_slices'][:, 0], 'V_mpp': data['v_slices'][:, 3], 'I_mpp': data['i_slices'][:, 3]})
        X_params_full = param_tf.transform(params_df_valid).astype(np.float32)
        X_scalar_full = scalar_tf.transform(scalar_df).astype(np.float32)
        X_combined = np.concatenate([X_params_full, X_scalar_full], axis=1)
        self.X = torch.from_numpy(X_combined[indices])
        self.isc_vals = torch.from_numpy(data['isc_vals'][indices])
    def __len__(self):
        return len(self.v_slices)
    def __getitem__(self, idx):
        return {'X_combined': self.X[idx], 'voltage': self.v_slices[idx], 'current_scaled': self.i_slices_scaled[idx], 'sample_w': self.sample_weights[idx], 'isc': self.isc_vals[idx]}

class IVDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.param_tf = None
        self.scalar_tf = None
    def prepare_data(self):
        if not Path(self.cfg['dataset']['paths']['preprocessed_npz']).exists():
            log.info("Preprocessed data not found. Running preprocessing...")
            self._preprocess_and_save()
        else:
            log.info("Found preprocessed data. Skipping preprocessing.")
    def setup(self, stage: str | None = None):
        if self.param_tf is None:
            self.param_tf = joblib.load(self.cfg['dataset']['paths']['param_transformer'])
            self.scalar_tf = joblib.load(self.cfg['dataset']['paths']['scalar_transformer'])
        if stage == "fit" or stage is None:
            self.train_dataset = IVDataset(self.cfg, 'train', self.param_tf, self.scalar_tf)
            self.val_dataset = IVDataset(self.cfg, 'val', self.param_tf, self.scalar_tf)
        if stage == "test" or stage is None:
            self.test_dataset = IVDataset(self.cfg, 'test', self.param_tf, self.scalar_tf)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg['dataset']['dataloader'], shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg['dataset']['dataloader'])
    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg['dataset']['dataloader'])

    def _preprocess_and_save(self):
        log.info("--- Starting Memory-Efficient Data Preprocessing ---")
        cfg = self.cfg
        paths = cfg['dataset']['paths']
        pchip_cfg = cfg['dataset']['pchip']

        # Create dummy data files if they don't exist, to prevent crashing
        if not Path(paths['params_csv']).exists():
            log.warning(f"{paths['params_csv']} not found. Creating a dummy file.")
            dummy_params = np.random.rand(100, len(COLNAMES))
            pd.DataFrame(dummy_params, columns=COLNAMES).to_csv(paths['params_csv'], header=False, index=False)
        if not Path(paths['iv_raw_txt']).exists():
            log.warning(f"{paths['iv_raw_txt']} not found. Creating a dummy file.")
            dummy_iv = 20 * np.exp(-np.linspace(0, 5, 43)) + np.random.randn(100, 43) * 0.1
            np.savetxt(paths['iv_raw_txt'], dummy_iv, delimiter=',')

        params_df = pd.read_csv(paths['params_csv'], header=None, names=COLNAMES)
        iv_data_raw = np.loadtxt(paths['iv_raw_txt'], delimiter=',')
        full_v_grid = np.concatenate([np.arange(0, 0.4 + 1e-8, 0.1), np.arange(0.425, 1.4 + 1e-8, 0.025)]).astype(np.float32)

        N_raw = len(iv_data_raw)
        log.info(f"Opening memory-mapped files for {N_raw} curves...")
        v_fine_mm = np.memmap(paths['v_fine_memmap'], dtype=np.float16, mode='w+', shape=(N_raw, pchip_cfg['n_fine']))
        i_fine_mm = np.memmap(paths['i_fine_memmap'], dtype=np.float16, mode='w+', shape=(N_raw, pchip_cfg['n_fine']))
        v_fine_mm[:] = np.nan
        i_fine_mm[:] = np.nan

        valid_indices, v_slices, i_slices = [], [], []
        pchip_args = (full_v_grid, pchip_cfg['n_pre_mpp'], pchip_cfg['n_post_mpp'], pchip_cfg['v_max'], pchip_cfg['n_fine'])

        for i in tqdm(range(N_raw), desc="PCHIP & Streaming to Disk"):
            res = process_iv_with_pchip(iv_data_raw[i], *pchip_args)
            if res is not None and res[1][0] > 1e-9:
                valid_indices.append(i)
                v_slices.append(res[0])
                i_slices.append(res[1])
                v_fine, i_fine = res[2]
                v_fine_mm[i, :len(v_fine)] = v_fine
                i_fine_mm[i, :len(i_fine)] = i_fine

        v_fine_mm.flush(); i_fine_mm.flush()
        del v_fine_mm, i_fine_mm

        log.info(f"Retained {len(valid_indices)} / {N_raw} valid curves after PCHIP & Isc filtering.")
        if not valid_indices:
            raise RuntimeError("No valid curves found after preprocessing. Please check your input data.")

        v_slices, i_slices, valid_indices = np.array(v_slices), np.array(i_slices), np.array(valid_indices)

        isc_vals, i_slices_scaled = zip(*[normalize_and_scale_by_isc(c) for c in i_slices])
        isc_vals, i_slices_scaled = np.array(isc_vals), np.array(i_slices_scaled)
        sample_weights = compute_curvature_weights(i_slices_scaled, **cfg['dataset']['curvature_weighting'])

        params_df_valid = params_df.iloc[valid_indices].reset_index(drop=True)
        param_transformer = get_param_transformer(COLNAMES)
        param_transformer.fit(params_df_valid)
        joblib.dump(param_transformer, paths['param_transformer'])

        scalar_df = pd.DataFrame({'I_ref': i_slices[:, 0], 'V_mpp': v_slices[:, pchip_cfg['n_pre_mpp']], 'I_mpp': i_slices[:, pchip_cfg['n_pre_mpp']]})
        scalar_transformer = Pipeline([('scaler', MinMaxScaler(feature_range=(-1, 1)))])
        scalar_transformer.fit(scalar_df)
        joblib.dump(scalar_transformer, paths['scalar_transformer'])

        param_dim, scalar_dim = param_transformer.transform(params_df_valid).shape[1], scalar_transformer.transform(scalar_df).shape[1]
        self.cfg['model']['param_dim'] = param_dim + scalar_dim
        log.info(f"Total parameter dimension calculated: {self.cfg['model']['param_dim']} ({param_dim} params + {scalar_dim} scalars)")

        all_indices = np.arange(len(valid_indices))
        train_val_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=cfg['train']['seed'])
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=cfg['train']['seed'])
        split_labels = np.array([''] * len(all_indices), dtype=object)
        split_labels[train_idx], split_labels[val_idx], split_labels[test_idx] = 'train', 'val', 'test'

        np.savez(paths['preprocessed_npz'],
                 v_slices=v_slices, i_slices=i_slices, i_slices_scaled=i_slices_scaled,
                 sample_weights=sample_weights, isc_vals=isc_vals,
                 valid_indices=valid_indices, split_labels=split_labels)
        log.info(f"Saved training/validation slice data to {paths['preprocessed_npz']}")
        log.info(f"Saved fine-grid curves to memmap files: {paths['v_fine_memmap']} & {paths['i_fine_memmap']}")


class PhysicsIVSystem(pl.LightningModule):
    def __init__(self, cfg: dict, warmup_steps: int, total_steps: int):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.hparams.warmup_steps = warmup_steps
        self.hparams.total_steps = total_steps
        mcfg = self.hparams
        mlp_layers = []
        in_dim = mcfg['model']['param_dim']
        for units in mcfg['model']['dense_units']:
            mlp_layers.extend([
                nn.Linear(in_dim, units),
                nn.BatchNorm1d(units),
                nn.GELU(),
                nn.Dropout(mcfg['model']['dropout'])
            ])
            in_dim = units
        self.param_mlp = nn.Sequential(*mlp_layers)
        self.pos_embed = make_positional_embedding(mcfg)
        seq_input_dim = mcfg['model']['dense_units'][-1] + self.pos_embed.out_dim
        filters, kernel, dropout, heads = mcfg['model']['filters'], mcfg['model']['kernel'], mcfg['model']['dropout'], mcfg['model']['heads']

        # I.2: Instantiate TCN blocks with exponentially increasing dilation
        self.tcn1 = TemporalBlock(seq_input_dim, filters[0], kernel, dropout, dilation=1) # Dilation = 2**0
        self.attn = SelfAttentionBlock(filters[0], heads, dropout)
        self.tcn2 = TemporalBlock(filters[0], filters[1], kernel, dropout, dilation=2) # Dilation = 2**1

        self.out_head = nn.Linear(filters[1], 1)
        self.apply(self._init_weights)
        self.test_preds, self.test_trues = [], []
        self.all_test_preds_np, self.all_test_trues_np = None, None

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)

    def forward(self, X_combined: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        B, L = voltage.shape
        p = self.param_mlp(X_combined)
        v_emb = self.pos_embed(voltage)
        p_rep = p.unsqueeze(1).expand(-1, L, -1)
        x = torch.cat([p_rep, v_emb], dim=-1).transpose(1, 2)
        x = self.tcn1(x)
        x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        x = self.tcn2(x).transpose(1, 2)
        return self.out_head(x).squeeze(-1)

    def _step(self, batch, stage: str):
        y_pred = self(batch['X_combined'], batch['voltage'])
        loss, comps = physics_loss(y_pred, batch['current_scaled'], batch['sample_w'], self.hparams['model']['loss_weights'])
        self.log_dict({f'{stage}_{k}': v for k, v in comps.items()}, on_step=False, on_epoch=True, batch_size=len(batch['voltage']))
        self.log(f'{stage}_loss', loss, prog_bar=(stage == 'val'), on_step=False, on_epoch=True, batch_size=len(batch['voltage']))
        return loss

    def training_step(self, batch, batch_idx): return self._step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        pred_scaled = self(batch['X_combined'], batch['voltage'])
        self.test_preds.append(denormalize(pred_scaled.cpu(), batch['isc'].cpu()))
        self.test_trues.append(denormalize(batch['current_scaled'].cpu(), batch['isc'].cpu()))
        loss, _ = physics_loss(pred_scaled, batch['current_scaled'], batch['sample_w'], self.hparams['model']['loss_weights'])
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch['voltage']))

    def on_test_epoch_start(self): self.test_preds.clear(); self.test_trues.clear()
    def on_test_epoch_end(self):
        if not self.test_preds: return
        self.all_test_preds_np = torch.cat(self.test_preds, dim=0).numpy()
        self.all_test_trues_np = torch.cat(self.test_trues, dim=0).numpy()
        preds, trues = self.all_test_preds_np, self.all_test_trues_np
        self.log("test/MAE_denorm", mean_absolute_error(trues.ravel(), preds.ravel()), prog_bar=True)
        self.log("test/RMSE_denorm", np.sqrt(mean_squared_error(trues.ravel(), preds.ravel())), prog_bar=True)
        self.log("test/avg_R2", np.mean([r2_score(trues[i], preds[i]) for i in range(len(trues))]), prog_bar=True)

    def configure_optimizers(self):
        opt_cfg = self.hparams['optimizer']
        optimizer = torch.optim.AdamW(self.parameters(), lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt_cfg['lr'], total_steps=self.hparams.total_steps, pct_start=self.hparams.warmup_steps/self.hparams.total_steps, final_div_factor=1/opt_cfg['final_lr_ratio'])
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

# ──────────────────────────────────────────────────────────────────────────────
#   PLOTTING CALLBACK WITH RECONSTRUCTION (PATCH APPLIED)
# ──────────────────────────────────────────────────────────────────────────────
class ExamplePlotsCallback(pl.Callback):
    """
    A callback that generates and logs illustrative plots with full curve
    reconstruction at the end of the test phase.
    """
    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        log.info("--- Generating fully reconstructed plots at the end of testing ---")
        if pl_module.all_test_preds_np is None:
            log.warning("Prediction arrays not found in the model. Skipping plotting.")
            return

        preds, trues = pl_module.all_test_preds_np, pl_module.all_test_trues_np
        metrics_df = pd.DataFrame({'r2': [r2_score(trues[i], preds[i]) for i in range(len(trues))]})
        n_samples = min(self.num_samples, len(trues))
        plot_groups = {
            "Random_Samples": np.random.choice(metrics_df.index, n_samples, replace=False),
            "Best_R2_Samples": metrics_df.nlargest(n_samples, 'r2').index.values,
            "Worst_R2_Samples": metrics_df.nsmallest(n_samples, 'r2').index.values,
        }
        for name, indices in plot_groups.items():
            if not trainer.logger:
                log.warning("No logger found, skipping plot logging.")
                continue
            filename = Path(trainer.logger.log_dir) / f"test_plots_{name.lower()}.png"
            self._generate_and_log_plot(trainer, pl_module, filename, name, indices, preds, trues, metrics_df)

    def _generate_and_log_plot(self, trainer, pl_module, filename, title, indices, preds, trues, metrics_df):
        hparams = pl_module.hparams
        paths = hparams['dataset']['paths']
        try:
            preprocessed_data = np.load(paths['preprocessed_npz'], allow_pickle=True)
        except FileNotFoundError:
            log.error(f"Could not find preprocessed data at {paths['preprocessed_npz']}. Skipping plotting.")
            return

        test_indices_in_valid_set = np.where(preprocessed_data['split_labels'] == 'test')[0]

        # load the full ground-truth fine grid (shape: [N_raw, n_fine])
        try:
            v_fine_mm = np.memmap(paths['v_fine_memmap'], dtype=np.float16, mode='r')\
                             .reshape(-1, hparams['dataset']['pchip']['n_fine'])
            i_fine_mm = np.memmap(paths['i_fine_memmap'], dtype=np.float16, mode='r')\
                             .reshape(-1, hparams['dataset']['pchip']['n_fine'])
        except (FileNotFoundError, ValueError) as e:
            log.error(f"Error loading memmap files: {e}. Skipping plotting.")
            return

        n_samples = len(indices)
        nrows, ncols = (n_samples + 3) // 4, 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows),
                                 squeeze=False, constrained_layout=True)
        axes = axes.flatten()
        fig.suptitle(title.replace("_", " "), fontsize=20, weight='bold')

        for i, test_set_idx in enumerate(indices):
            ax = axes[i]
            valid_set_idx = test_indices_in_valid_set[test_set_idx]
            raw_data_idx = preprocessed_data['valid_indices'][valid_set_idx]

            v_slice = preprocessed_data['v_slices'][valid_set_idx]
            i_true_slice = trues[test_set_idx]
            i_pred_slice = preds[test_set_idx]

            # full ground truth
            v_fine = v_fine_mm[raw_data_idx].astype(np.float32)
            i_fine = i_fine_mm[raw_data_idx].astype(np.float32)
            mask = ~np.isnan(v_fine)
            v_fine, i_fine = v_fine[mask], i_fine[mask]

            # — Ground-truth full curve
            if len(v_fine) > 0:
                ax.plot(v_fine, i_fine, 'k-', alpha=0.7, lw=2,
                        label='Actual (Fine Grid)')

            # — Predicted full curve, PCHIP-interpolated over the exact same fine grid
            pred_interp = PchipInterpolator(v_slice, i_pred_slice, extrapolate=False)
            # run it over the full fine grid so endpoints (Isc @ v=0 and Voc @ v=Voc) are honored
            if len(v_fine) > 0:
                i_pred_full = pred_interp(v_fine)
                ax.plot(v_fine, i_pred_full, 'r--', lw=2,
                        label='Predicted (Reconstructed)')

            # Points for clarity
            ax.plot(v_slice, i_true_slice,  'bo', ms=6, label='Actual Points')
            ax.plot(v_slice, i_pred_slice,  'rx', ms=6, mew=2, label='Predicted Points')

            r2_val = metrics_df.loc[test_set_idx, 'r2']
            ax.set_title(f"Test Sample #{test_set_idx} (R² = {r2_val:.4f})")
            ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Current (mA/cm²)")
            ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
            if len(v_fine) > 0 and len(i_fine) > 0:
                ax.set_xlim(left=-0.05, right=max(v_fine.max() * 1.05, 0.1))
                ax.set_ylim(bottom=-max(i_fine.max()*0.05, 1))

        for j in range(n_samples, len(axes)): fig.delaxes(axes[j])
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        try:
            img = np.array(Image.open(filename))
            trainer.logger.experiment.add_image(title, img, 0, dataformats='HWC')
            log.info(f"Saved and logged reconstructed plot: {filename}")
        except Exception as e:
            log.warning(f"Could not log image to Tensorboard: {e}")

        del v_fine_mm, i_fine_mm, preprocessed_data

# ──────────────────────────────────────────────────────────────────────────────
#   MAIN EXECUTION SCRIPT
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(cfg: dict):
    log.info(f"Starting run '{cfg['train']['run_name']}'")
    seed_everything(cfg['train']['seed'])

    datamodule = IVDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')

    batches_per_epoch = len(datamodule.train_dataloader())
    if batches_per_epoch == 0:
        log.error("Train dataloader is empty. Cannot determine number of steps for optimizer. Aborting.")
        return
    total_steps = cfg['trainer']['max_epochs'] * batches_per_epoch
    model = PhysicsIVSystem(cfg, warmup_steps=cfg['optimizer']['warmup_epochs'] * batches_per_epoch, total_steps=total_steps)
    log.info(f"Model instantiated with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-model-{epoch:02d}")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=20, mode="min")
    logger = TensorBoardLogger(str(OUTPUT_DIR / "tb_logs"), name=cfg['train']['run_name'])
    plot_callback = ExamplePlotsCallback(num_samples=8)

    trainer = pl.Trainer(
        **cfg['trainer'],
        default_root_dir=OUTPUT_DIR,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stopping_cb, RichProgressBar(), plot_callback],
    )

    log.info("--- Starting Training ---")
    trainer.fit(model, datamodule=datamodule)

    log.info("--- Starting Final Testing on Best Checkpoint ---")
    # Use the best model checkpoint for testing
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path and Path(best_ckpt_path).exists():
        log.info(f"Loading best model from: {best_ckpt_path}")
        test_results = trainer.test(datamodule=datamodule, ckpt_path=best_ckpt_path)
    else:
        log.warning("No best checkpoint found or path is invalid. Testing with the last model state.")
        test_results = trainer.test(datamodule=datamodule)

    log.info(f"Final test results (see TensorBoard for details): {test_results[0]}")
    log.info(f"Experiment Finished. Full results in: {trainer.logger.log_dir}")


if __name__ == "__main__":
    # Check for Colab environment and existence of data files
    is_colab = False
    try:
        import google.colab
        is_colab = True
    except ImportError:
        pass

    if is_colab and not (Path(INPUT_FILE_PARAMS).exists() and Path(INPUT_FILE_IV).exists()):
         log.error("="*80)
         log.error("! Google Drive is not mounted or data files not found !")
         log.error("Please mount your drive: from google.colab import drive; drive.mount('/content/drive')")
         log.error(f"And ensure files exist at: {INPUT_FILE_PARAMS} and {INPUT_FILE_IV}")
         log.error("="*80)
    elif not (Path(INPUT_FILE_PARAMS).exists() and Path(INPUT_FILE_IV).exists()):
        log.warning("="*80)
        log.warning("Input data files not found! Please update the paths at the top of the script.")
        log.warning(f"Checked for: {INPUT_FILE_PARAMS} and {INPUT_FILE_IV}")
        log.warning("For a demonstration, dummy data files will be created during preprocessing.")
        log.warning("="*80)
        run_experiment(CONFIG)
    else:
        run_experiment(CONFIG)