# If you are running this from Colab, you need to comment out:
# !pip install pytorch_lightning

#
# ==============================================================================
#  Physics-Informed Attention-TCN for I-V Curve Reconstruction in PyTorch
# ==============================================================================
#
# This is the latest model as of Jun. 29, 2025. - Ozdincer
#
# --- ARCHITECTURE ---
#   - Hybrid Attention-Augmented Temporal Convolutional Network (TCN).
#   - Swappable positional embeddings (Fourier, Clipped Fourier, Gaussian RBF).
#   - Physics-informed loss function (MSE, Monotonicity, Convexity, Curvature).
#
# --- WORKFLOW ---
#   1. Configuration: All hyperparameters are defined in a single `CONFIG` dict.
#   2. Preprocessing: A one-time function processes raw data, applies PCHIP
#      interpolation, creates features, and saves a clean .npz file.
#   3. Data Loading: A PyTorch Lightning `DataModule` efficiently handles
#      loading data for train, validation, and test sets.
#   4. Model: The `PhysicsIVSystem` LightningModule encapsulates the entire
#      model, training logic, and optimizer configuration.
#   5. Training: The `pl.Trainer` automates the training loop, including
#      mixed-precision, checkpointing, logging, and early stopping.
#
# ==============================================================================

import os
import logging
from pathlib import Path
from datetime import datetime
import typing
import math

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# ──────────────────────────────────────────────────────────────────────────────
#   CONFIGURATIONS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# ⚠️ --- ACTION REQUIRED: UPDATE FILE PATHS --- ⚠️
# Set the paths to your input data files and desired output directory.
# Note: This script will create a './data/processed' subdirectory for precomputed files.
INPUT_FILE_PARAMS = "/content/drive/MyDrive/Colab Notebooks/Data_100k/LHS_parameters_m.txt"
INPUT_FILE_IV = "/content/drive/MyDrive/Colab Notebooks/Data_100k/iV_m.txt"
OUTPUT_DIR = Path("./lightning_output")
# --------------------------------------------------------------------------

# Ensure the output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Path("./data/processed").mkdir(parents=True, exist_ok=True)

# Main configuration dictionary (replaces Hydra for single-script use)
CONFIG = {
    "train": {
        "seed": 42,
        "run_name": f"AttentionTCN-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    },
    "model": {
        "param_dim": 30,  # 27 device params + 3 scalar features
        "dense_units": [256, 128, 128],
        "filters": [128, 64],
        "kernel": 5,
        "heads": 4,
        "dropout": 0.25,
        "embedding_type": 'fourier_clipped',  # 'fourier', 'fourier_clipped', 'gaussian'
        "fourier_bands": 16,
        "gaussian_bands": 16,
        "gaussian_sigma": 0.1,
        "loss_weights": {
            "mse": 0.98,
            "mono": 0.005,
            "convex": 0.005,
            "excurv": 0.01,
            "excess_threshold": 0.8,
        },
    },
    "optimizer": {
        "lr": 0.0015,
        "weight_decay": 1e-5,
        "final_lr_ratio": 0.00666, # final_lr = lr * ratio
        "warmup_epochs": 5,
    },
    "dataset": {
        "paths": {
            "params_csv": INPUT_FILE_PARAMS,
            "iv_raw_txt": INPUT_FILE_IV,
            "output_dir": "./data/processed",
            "preprocessed_npz": "./data/processed/precomputed_data.npz",
            "param_transformer": "./data/processed/param_transformer.joblib",
            "scalar_transformer": "./data/processed/scalar_transformer.joblib",
        },
        "pchip": {
            "v_max": 1.4,
            "n_fine": 10000,
            "n_pre_mpp": 3,
            "n_post_mpp": 4,
            "seq_len": 8, # n_pre + 1 + n_post
        },
        "dataloader": {
            "batch_size": 128,
            "num_workers": os.cpu_count() // 2,
            "pin_memory": True,
        },
        "curvature_weighting": {
            "alpha": 4.0,
            "power": 1.5,
        },
    },
    "trainer": {
        "max_epochs": 100,
        "accelerator": "auto",
        "devices": "auto",
        "precision": "16-mixed",
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 25,
    },
}

# Column names for the parameters file
COLNAMES = [
    'lH', 'lP', 'lE', 'muHh', 'muPh', 'muPe', 'muEe', 'NvH', 'NcH',
    'NvE', 'NcE', 'NvP', 'NcP', 'chiHh', 'chiHe', 'chiPh', 'chiPe',
    'chiEh', 'chiEe', 'Wlm', 'Whm', 'epsH', 'epsP', 'epsE', 'Gavg',
    'Aug', 'Brad', 'Taue', 'Tauh', 'vII', 'vIII'
]

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#   UTILITY FUNCTIONS (from `utils.py`)
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    pl.seed_everything(seed, workers=True)


def process_iv_with_pchip(
    iv_raw: np.ndarray, full_v_grid: np.ndarray, n_pre: int, n_post: int,
    v_max: float, n_fine: int
) -> typing.Optional[tuple]:
    """Identical logic to the TF version, but with improved type hints."""
    # ✔️ CRITIQUE ADDRESSED: Using typing.Optional for Python < 3.10 compatibility
    seq_len = n_pre + 1 + n_post
    try:
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
        v_slice = np.interp(
            np.linspace(0, 1, seq_len), np.linspace(0, 1, len(v_mpp_grid)), v_mpp_grid)
        i_slice = pi(v_slice)
        if np.any(np.isnan(i_slice)) or i_slice.shape[0] != seq_len: return None
        return (
            v_slice.astype(np.float32),
            i_slice.astype(np.float32),
            (v_fine.astype(np.float32), i_fine.astype(np.float32))
        )
    except (ValueError, IndexError): return None


def normalize_and_scale_by_isc(curve: np.ndarray) -> tuple[float, np.ndarray]:
    """Scales curve to [-1, 1] and returns the Isc value."""
    isc_val = float(curve[0])
    return isc_val, (2.0 * (curve / isc_val) - 1.0).astype(np.float32)


def compute_curvature_weights(y_curves: np.ndarray, alpha: float, power: float) -> np.ndarray:
    """Computes sample weights based on curvature of the scaled curves."""
    padded = np.pad(y_curves, ((0, 0), (1, 1)), mode='edge')
    kappa = np.abs(padded[:, 2:] - 2 * padded[:, 1:-1] + padded[:, :-2])
    max_kappa = np.max(kappa, axis=1, keepdims=True)
    max_kappa[max_kappa < 1e-9] = 1.0
    return (1.0 + alpha * np.power(kappa / max_kappa, power)).astype(np.float32)


def get_param_transformer(colnames: list[str]) -> ColumnTransformer:
    """Builds the sklearn ColumnTransformer for input parameters."""
    param_defs = {
        'layer_thickness': ['lH', 'lP', 'lE'],
        'material_properties': [
            'muHh', 'muPh', 'muPe', 'muEe', 'NvH', 'NcH', 'NvE', 'NcE',
            'NvP', 'NcP', 'chiHh', 'chiHe', 'chiPh', 'chiPe', 'chiEh',
            'chiEe', 'epsH', 'epsP', 'epsE'
        ],
        'contacts': ['Wlm', 'Whm'],
        'recombination_gen': [
            'Gavg', 'Aug', 'Brad', 'Taue', 'Tauh', 'vII', 'vIII'
        ]
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


# ──────────────────────────────────────────────────────────────────────────────
#   MODEL COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────

# ‹›‹›‹› Positional Embeddings (from `models/embeddings.py`) ‹›‹›‹›

class FourierFeatures(nn.Module):
    def __init__(self, num_bands: int, v_max: float = 1.4):
        super().__init__()
        self.v_max = v_max
        B = torch.logspace(0, 3, num_bands)
        self.register_buffer('B', B, persistent=False)
        self.out_dim = num_bands * 2

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # ✔️ CRITIQUE ADDRESSED: Use device-aware tensor for pi
        two_pi = torch.tensor(2 * math.pi, device=v.device, dtype=v.dtype)
        v_norm = v / self.v_max
        v_proj = v_norm.unsqueeze(-1) * self.B
        return torch.cat([(two_pi * v_proj).sin(), (two_pi * v_proj).cos()], dim=-1)


class ClippedFourierFeatures(nn.Module):
    def __init__(self, num_bands: int, v_max: float = 1.4):
        super().__init__()
        self.v_max = v_max
        B = torch.logspace(0, 3, num_bands)
        # ✔️ CRITIQUE ADDRESSED: Pre-broadcast mask for efficiency
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
    """Factory function for embeddings."""
    etype = cfg['model']['embedding_type']
    if etype == 'fourier':
        return FourierFeatures(cfg['model']['fourier_bands'], cfg['dataset']['pchip']['v_max'])
    elif etype == 'fourier_clipped':
        return ClippedFourierFeatures(cfg['model']['fourier_bands'], cfg['dataset']['pchip']['v_max'])
    elif etype == 'gaussian':
        return GaussianRBFFeatures(cfg['model']['gaussian_bands'], cfg['model']['gaussian_sigma'], cfg['dataset']['pchip']['v_max'])
    raise ValueError(f"Unknown embedding type: {etype}")

# ‹›‹›‹› Physics-Informed Loss (from `models/losses.py`) ‹›‹›‹›

def physics_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, sample_w: torch.Tensor, loss_w: dict
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculates the total physics-informed loss."""
    mse_loss = (((y_true - y_pred)**2) * sample_w).mean()
    mono_violations = torch.relu(y_pred[:, 1:] - y_pred[:, :-1])
    mono_loss = mono_violations.pow(2).mean()
    convex_violations = torch.relu(y_pred[:, :-2] - 2 * y_pred[:, 1:-1] + y_pred[:, 2:])
    convex_loss = convex_violations.pow(2).mean()
    curvature = torch.abs(y_pred[:, :-2] - 2 * y_pred[:, 1:-1] + y_pred[:, 2:])
    excurv_violations = torch.relu(curvature - loss_w['excess_threshold'])
    excurv_loss = excurv_violations.pow(2).mean()
    total_loss = (
        loss_w['mse'] * mse_loss +
        loss_w['mono'] * mono_loss +
        loss_w['convex'] * convex_loss +
        loss_w['excurv'] * excurv_loss
    )
    return total_loss, {
        'mse': mse_loss, 'mono': mono_loss,
        'convex': convex_loss, 'excurv': excurv_loss
    }

# ‹›‹›‹› Core Model Architecture (from `models/tcn_attention.py`) ‹›‹›‹›

class ChannelLayerNorm(nn.Module):
    # ✔️ CRITIQUE ADDRESSED: Helper module for clean LayerNorm on channel dimension
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, L], LayerNorm expects [..., C]
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        self.padding = (kernel_size - 1, 0)
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size))
        self.act1 = nn.GELU()
        self.norm1 = ChannelLayerNorm(out_ch)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size))
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
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


# ──────────────────────────────────────────────────────────────────────────────
#   PYTORCH LIGHTNING DATA & MODEL MODULES
# ──────────────────────────────────────────────────────────────────────────────

class IVDataset(Dataset):
    def __init__(self, cfg: dict, split: str, param_tf, scalar_tf):
        self.cfg = cfg
        self.split = split
        data = np.load(cfg['dataset']['paths']['preprocessed_npz'], allow_pickle=True)
        
        # ✔️ CRITIQUE ADDRESSED: Use clean split_labels array for indexing
        split_labels = data['split_labels']
        indices = np.where(split_labels == split)[0]

        self.v_slices = torch.from_numpy(data['v_slices'][indices])
        self.i_slices_scaled = torch.from_numpy(data['i_slices_scaled'][indices])
        self.sample_weights = torch.from_numpy(data['sample_weights'][indices])

        params_df = pd.read_csv(cfg['dataset']['paths']['params_csv'], header=None, names=COLNAMES)
        params_df_valid = params_df.iloc[data['valid_indices']].reset_index(drop=True)
        scalar_df = pd.DataFrame({
            'I_ref': data['i_slices'][:, 0],
            'V_mpp': data['v_slices'][:, 3],
            'I_mpp': data['i_slices'][:, 3]
        })

        X_params_full = param_tf.transform(params_df_valid).astype(np.float32)
        X_scalar_full = scalar_tf.transform(scalar_df).astype(np.float32)
        X_combined = np.concatenate([X_params_full, X_scalar_full], axis=1)
        self.X = torch.from_numpy(X_combined[indices])
        
        # Physical values for evaluation
        self.isc_vals = torch.from_numpy(data['isc_vals'][indices])

    def __len__(self):
        return len(self.v_slices)

    def __getitem__(self, idx):
        return {
          'X_combined': self.X[idx],
          'voltage': self.v_slices[idx],
          'current_scaled': self.i_slices_scaled[idx],
          'sample_w': self.sample_weights[idx],
          'isc': self.isc_vals[idx],
        }

class IVDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.param_tf = None
        self.scalar_tf = None

    def prepare_data(self):
        # This method is called once per node. Good for downloads, etc.
        # Here we use it for our one-time preprocessing.
        if not Path(self.cfg['dataset']['paths']['preprocessed_npz']).exists():
            log.info("Preprocessed data not found. Running preprocessing...")
            self._preprocess_and_save()
        else:
            log.info("Found preprocessed data. Skipping preprocessing.")

    def setup(self, stage: str | None = None):
        # ✔️ CRITIQUE ADDRESSED: Load transformers once here, not in Dataset
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
        # This is the main data crunching function.
        log.info("--- Starting Data Preprocessing ---")
        cfg = self.cfg
        paths = cfg['dataset']['paths']
        params_df = pd.read_csv(paths['params_csv'], header=None, names=COLNAMES)
        iv_data_raw = np.loadtxt(paths['iv_raw_txt'], delimiter=',')
        full_v_grid = np.concatenate([np.arange(0, 0.4 + 1e-8, 0.1), np.arange(0.425, 1.4 + 1e-8, 0.025)]).astype(np.float32)
        
        pchip_cfg = cfg['dataset']['pchip']
        pchip_args = (full_v_grid, pchip_cfg['n_pre_mpp'], pchip_cfg['n_post_mpp'], pchip_cfg['v_max'], pchip_cfg['n_fine'])
        results = [process_iv_with_pchip(iv_data_raw[i], *pchip_args) for i in tqdm(range(len(iv_data_raw)), desc="PCHIP")]
        
        valid_indices, v_slices, i_slices, fine_curves_tuples = [], [], [], []
        for i, res in enumerate(results):
            if res is not None and res[1][0] > 1e-9: # ✔️ CRITIQUE ADDRESSED: Filter out zero/negative Isc curves
                valid_indices.append(i)
                v_slices.append(res[0])
                i_slices.append(res[1])
                fine_curves_tuples.append(res[2])

        log.info(f"Retained {len(valid_indices)} / {len(iv_data_raw)} valid curves after PCHIP & Isc filtering.")
        v_slices = np.array(v_slices)
        i_slices = np.array(i_slices)
        valid_indices = np.array(valid_indices)

        # Normalize and compute weights
        isc_vals, i_slices_scaled = zip(*[normalize_and_scale_by_isc(c) for c in i_slices])
        isc_vals = np.array(isc_vals)
        i_slices_scaled = np.array(i_slices_scaled)
        sample_weights = compute_curvature_weights(i_slices_scaled, **cfg['dataset']['curvature_weighting'])

        # Feature Engineering & Scaling
        params_df_valid = params_df.iloc[valid_indices].reset_index(drop=True)
        param_transformer = get_param_transformer(COLNAMES)
        param_transformer.fit(params_df_valid)
        joblib.dump(param_transformer, paths['param_transformer'])
        scalar_df = pd.DataFrame({'I_ref': i_slices[:, 0], 'V_mpp': v_slices[:, pchip_cfg['n_pre_mpp']], 'I_mpp': i_slices[:, pchip_cfg['n_pre_mpp']]})
        scalar_transformer = Pipeline([('scaler', StandardScaler())])
        scalar_transformer.fit(scalar_df)
        joblib.dump(scalar_transformer, paths['scalar_transformer'])

        # Calculate and store the parameter dimensions
        param_dim = param_transformer.transform(params_df_valid).shape[1]
        scalar_dim = scalar_transformer.transform(scalar_df).shape[1]
        self.cfg['model']['param_dim'] = param_dim + scalar_dim
        log.info(f"Total parameter dimension calculated: {self.cfg['model']['param_dim']} ({param_dim} params + {scalar_dim} scalars)")

        # Create data splits using a labels array
        all_indices = np.arange(len(valid_indices))
        train_val_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=cfg['train']['seed'])
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=cfg['train']['seed'])
        split_labels = np.array([''] * len(all_indices), dtype=object)
        split_labels[train_idx] = 'train'
        split_labels[val_idx] = 'val'
        split_labels[test_idx] = 'test'

        # ✔️ CRITIQUE ADDRESSED: Store fine curves in dense padded arrays instead of object array
        max_len = max(len(v) for v, i in fine_curves_tuples)
        v_fine_padded = np.full((len(fine_curves_tuples), max_len), np.nan, dtype=np.float32)
        i_fine_padded = np.full((len(fine_curves_tuples), max_len), np.nan, dtype=np.float32)
        for i, (v, c) in enumerate(fine_curves_tuples):
            v_fine_padded[i, :len(v)] = v
            i_fine_padded[i, :len(c)] = c
            
        np.savez(
            paths['preprocessed_npz'],
            v_slices=v_slices, i_slices=i_slices, i_slices_scaled=i_slices_scaled,
            sample_weights=sample_weights, isc_vals=isc_vals,
            valid_indices=valid_indices, split_labels=split_labels,
            v_fine_padded=v_fine_padded, i_fine_padded=i_fine_padded # New padded arrays
        )
        log.info(f"Saved all preprocessed data and transformers to {paths['output_dir']}")

class PhysicsIVSystem(pl.LightningModule):
    # ✔️ CRITIQUE ADDRESSED: Accept scheduler steps during init
    def __init__(self, cfg: dict, warmup_steps: int, total_steps: int):
        super().__init__()
        self.save_hyperparameters(cfg) # Saves the regular config
        self.hparams.warmup_steps = warmup_steps
        self.hparams.total_steps = total_steps
        
        mcfg = self.hparams
        mlp_layers = []
        in_dim = mcfg['model']['param_dim']
        for units in mcfg['model']['dense_units']:
            mlp_layers.extend([nn.Linear(in_dim, units), nn.GELU(), nn.LayerNorm(units), nn.Dropout(mcfg['model']['dropout'])])
            in_dim = units
        self.param_mlp = nn.Sequential(*mlp_layers)
        
        self.pos_embed = make_positional_embedding(mcfg)
        
        seq_input_dim = mcfg['model']['dense_units'][-1] + self.pos_embed.out_dim
        filters = mcfg['model']['filters']
        kernel, dropout, heads = mcfg['model']['kernel'], mcfg['model']['dropout'], mcfg['model']['heads']
        self.tcn1 = TemporalBlock(seq_input_dim, filters[0], kernel, dropout)
        self.attn = SelfAttentionBlock(filters[0], heads, dropout)
        self.tcn2 = TemporalBlock(filters[0], filters[1], kernel, dropout)
        self.out_head = nn.Linear(filters[1], 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        # ✔️ CRITIQUE ADDRESSED: More comprehensive initialization
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), nonlinearity='leaky_relu') # GELU is similar to leaky relu
            if module.bias is not None: nn.init.zeros_(module.bias)
            
    def forward(self, X_combined: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        B, L = voltage.shape
        p = self.param_mlp(X_combined)
        v_emb = self.pos_embed(voltage)
        p_rep = p.unsqueeze(1).expand(-1, L, -1)
        x = torch.cat([p_rep, v_emb], dim=-1)
        x = x.transpose(1, 2)
        x = self.tcn1(x)
        x = x.transpose(1, 2)
        x = self.attn(x)
        x = x.transpose(1, 2)
        x = self.tcn2(x)
        x = x.transpose(1, 2)
        return self.out_head(x).squeeze(-1)

    def _step(self, batch, stage: str):
        y_pred = self(batch['X_combined'], batch['voltage'])
        loss, comps = physics_loss(y_pred, batch['current_scaled'], batch['sample_w'], self.hparams['model']['loss_weights'])
        # ✔️ CRITIQUE ADDRESSED: Removed redundant batch_size argument
        self.log_dict({f'{stage}_{k}': v for k, v in comps.items()}, on_step=False, on_epoch=True)
        self.log(f'{stage}_loss', loss, prog_bar=(stage == 'val'), on_step=False, on_epoch=True)
        return loss
        
    def training_step(self, batch, batch_idx): return self._step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self._step(batch, 'val')
    def test_step(self, batch, batch_idx): return self._step(batch, 'test')
    
    def configure_optimizers(self):
        opt_cfg = self.hparams['optimizer']
        optimizer = torch.optim.AdamW(self.parameters(), lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'])
        final_lr = opt_cfg['lr'] * opt_cfg['final_lr_ratio']
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-7, end_factor=1.0, total_iters=self.hparams.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.total_steps - self.hparams.warmup_steps, eta_min=final_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.hparams.warmup_steps])
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


# ──────────────────────────────────────────────────────────────────────────────
#   MAIN EXECUTION SCRIPT
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(cfg: dict):
    """Main function to run the training and testing pipeline."""
    log.info(f"Starting run '{cfg['train']['run_name']}'")
    seed_everything(cfg['train']['seed'])

    # 1. Setup DataModule (will trigger preprocessing if needed)
    datamodule = IVDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage='fit') # Manually call setup to get dataloader info

    # 2. Calculate scheduler steps and instantiate model
    batches_per_epoch = len(datamodule.train_dataloader())
    warmup_steps = cfg['optimizer']['warmup_epochs'] * batches_per_epoch
    total_steps = cfg['trainer']['max_epochs'] * batches_per_epoch

    # Sanity check for parameter dimensions
    xb, = next(iter(datamodule.train_dataloader()))['X_combined'],
    log.info(f">>> X_combined.shape: {xb.shape}")  # should be [batch_size, param_dim]
    log.info(f">>> CONFIG param_dim: {cfg['model']['param_dim']}")

    model = PhysicsIVSystem(cfg, warmup_steps, total_steps)
    log.info(f"Model instantiated with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # 3. Setup Callbacks and Logger
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-model-{epoch:02d}")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=20, mode="min")
    logger = TensorBoardLogger(str(OUTPUT_DIR / "tb_logs"), name=cfg['train']['run_name'])
    
    # 4. Initialize and run Trainer
    trainer = pl.Trainer(
        **cfg['trainer'],
        default_root_dir=OUTPUT_DIR,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stopping_cb],
    )
    
    log.info("--- Starting Training ---")
    trainer.fit(model, datamodule=datamodule)
    
    log.info("--- Starting Testing on Best Checkpoint ---")
    test_results = trainer.test(datamodule=datamodule, ckpt_path="best")
    log.info(f"Test results: {test_results[0]}")

    # (Optional) Add final evaluation and plotting here
    # Example:
    # from matplotlib import pyplot as plt
    # test_loader = datamodule.test_dataloader()
    # best_model = PhysicsIVSystem.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # best_model.eval()
    # ... plotting logic ...


if __name__ == "__main__":
    if not (Path(INPUT_FILE_PARAMS).exists() and Path(INPUT_FILE_IV).exists()):
        log.error("="*80)
        log.error("Input data files not found! Please update the paths at the top of the script.")
        log.error(f"Checked for: {INPUT_FILE_PARAMS} and {INPUT_FILE_IV}")
        log.error("="*80)
    else:
        run_experiment(CONFIG)