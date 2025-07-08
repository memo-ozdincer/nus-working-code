# For colab:
!pip install pytorch_lightning optuna optuna_integration[pytorch_lightning]
!pip install --upgrade optuna
#!/usr/bin/env python3
#
# ==============================================================================
#  Physics-Informed Attention-TCN for I-V Curve Reconstruction in PyTorch
# ==============================================================================
#
# This single-file script implements a complete, production-ready pipeline for
# training a physics-informed neural network. It ports a TensorFlow model to
# a modern PyTorch & PyTorch Lightning framework, incorporating best practices
# and addressing critiques from the original design.
#
# === HPO UPDATE ===
# This version integrates a full Hyperparameter Optimization (HPO) routine using
# Optuna, following a structured blueprint.
#
# --- HPO WORKFLOW ---
#   1. Objective Function: An `objective` function wraps the training process,
#      where Optuna suggests hyperparameters for each trial.
#   2. Search & Pruning: Optuna's TPE sampler explores the search space, and a
#      pruner callback terminates poor-performing trials early.
#   3. HPO Study: The main script runs the optimization study for a specified
#      number of trials.
#   4. Final Training: After the study, the best hyperparameters are used to
#      train a final model on the full training data.
#   5. Evaluation: The final model is evaluated on the held-out test set.
#
# ==============================================================================

import os
import logging
from pathlib import Path
from datetime import datetime
import typing
import math
import copy
import json

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# --- HPO Imports ---
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch.nn.utils import parametrize
import torch.nn.utils as nn_utils

# ──────────────────────────────────────────────────────────────────────────────
#   CONFIGURATIONS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

INPUT_FILE_PARAMS = "/content/drive/MyDrive/Colab Notebooks/Data_100k/LHS_parameters_m.txt"
INPUT_FILE_IV = "/content/drive/MyDrive/Colab Notebooks/Data_100k/iV_m.txt"
OUTPUT_DIR = Path("./lightning_output_hpo")
# --------------------------------------------------------------------------

# Ensure the output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Path("./data/processed").mkdir(parents=True, exist_ok=True)

# Main configuration dictionary (serves as a base template)
CONFIG = {
    "train": {
        "seed": 42,
        "run_name": f"AttentionTCN-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    },
    "hpo": {
        "n_trials": 2, # Number of HPO trials to run
        "study_name": "PhysicsIV-HPO",
    },
    "model": {
        "param_dim": 34, # This is updated dynamically after data loading
        "dense_units": [256, 128, 128],
        "filters": [128, 64],
        "kernel": 5,
        "heads": 4,
        "dropout": 0.25,
        "embedding_type": 'fourier_clipped',
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
        "final_lr_ratio": 0.00666,
        "warmup_epochs": 5,
    },
    "dataset": {
        "paths": {
            "params_csv": INPUT_FILE_PARAMS,
            "iv_raw_txt": INPUT_FILE_IV,
            "output_dir": "./data/processed",
            "preprocessed_npz": "./data/processed/precomputed_data",
            "param_transformer": "./data/processed/param_transformer.joblib",
            "scalar_transformer": "./data/processed/scalar_transformer.joblib",
        },
        "pchip": {
            "v_max": 1.4,
            "n_fine": 10000,
            "n_pre_mpp": 5,
            "n_post_mpp": 6,
            "seq_len": 10, # Note: This is n_pre + n_post, not a separate param
        },
        "dataloader": {
            "batch_size": 128,
            "num_workers": os.cpu_count() // 2 if os.cpu_count() is not None else 2,
            "pin_memory": True,
        },
        "curvature_weighting": {
            "alpha": 4.0,
            "power": 1.5,
        },
    },
    "trainer": {
        "max_epochs": 10, # Max epochs for a single HPO trial
        "accelerator": "auto",
        "devices": "auto",
        "precision": "16-mixed",
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 50,
    },
    "final_train": { # Settings for the final run with best params
        "max_epochs": 150,
    }
}

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
#   UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)

def process_iv_with_pchip(iv_raw, full_v_grid, n_pre, n_post, v_max, n_fine):
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
        v_slice = np.interp(np.linspace(0, 1, seq_len), np.linspace(0, 1, len(v_mpp_grid)), v_mpp_grid)
        i_slice = pi(v_slice)
        if np.any(np.isnan(i_slice)) or i_slice.shape[0] != seq_len: return None
        return (v_slice.astype(np.float32), i_slice.astype(np.float32))
    except (ValueError, IndexError): return None

def normalize_and_scale_by_isc(curve):
    isc_val = float(curve[0])
    return isc_val, (2.0 * (curve / isc_val) - 1.0).astype(np.float32)

def compute_curvature_weights(y_curves, alpha, power):
    padded = np.pad(y_curves, ((0, 0), (1, 1)), mode='edge')
    kappa = np.abs(padded[:, 2:] - 2 * padded[:, 1:-1] + padded[:, :-2])
    max_kappa = np.max(kappa, axis=1, keepdims=True)
    max_kappa[max_kappa < 1e-9] = 1.0
    return (1.0 + alpha * np.power(kappa / max_kappa, power)).astype(np.float32)

def get_param_transformer(colnames):
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

# ──────────────────────────────────────────────────────────────────────────────
#   MODEL COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────
class FourierFeatures(nn.Module):
    def __init__(self, num_bands, v_max=1.4):
        super().__init__()
        self.v_max = v_max
        B = torch.logspace(0, 3, num_bands)
        self.register_buffer('B', B, persistent=False)
        self.out_dim = num_bands * 2
    def forward(self, v):
        two_pi = torch.tensor(2 * math.pi, device=v.device, dtype=v.dtype)
        v_norm = v / self.v_max
        v_proj = v_norm.unsqueeze(-1) * self.B
        return torch.cat([(two_pi * v_proj).sin(), (two_pi * v_proj).cos()], dim=-1)

class ClippedFourierFeatures(nn.Module):
    def __init__(self, num_bands, v_max=1.4):
        super().__init__()
        self.v_max = v_max
        B = torch.logspace(0, 3, num_bands)
        B_mask = (B >= 1.0).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('B', B, persistent=False)
        self.register_buffer('B_mask', B_mask, persistent=False)
        self.out_dim = num_bands * 2
    def forward(self, v):
        two_pi = torch.tensor(2 * math.pi, device=v.device, dtype=v.dtype)
        v_norm = v / self.v_max
        v_proj = v_norm.unsqueeze(-1) * self.B
        sines = (two_pi * v_proj).sin() * self.B_mask
        coses = (two_pi * v_proj).cos() * self.B_mask
        return torch.cat([sines, coses], dim=-1)

class GaussianRBFFeatures(nn.Module):
    def __init__(self, num_bands, sigma=0.1, v_max=1.4):
        super().__init__()
        self.v_max = v_max
        self.sigma = sigma
        mu = torch.linspace(0, 1, num_bands)
        self.register_buffer('mu', mu, persistent=False)
        self.out_dim = num_bands
    def forward(self, v):
        v_norm = v / self.v_max
        diff = v_norm.unsqueeze(-1) - self.mu
        return torch.exp(-0.5 * (diff / self.sigma)**2)

def make_positional_embedding(cfg):
    etype = cfg['model']['embedding_type']
    if etype == 'fourier': return FourierFeatures(cfg['model']['fourier_bands'], cfg['dataset']['pchip']['v_max'])
    if etype == 'fourier_clipped': return ClippedFourierFeatures(cfg['model']['fourier_bands'], cfg['dataset']['pchip']['v_max'])
    if etype == 'gaussian': return GaussianRBFFeatures(cfg['model']['gaussian_bands'], cfg['model']['gaussian_sigma'], cfg['dataset']['pchip']['v_max'])
    raise ValueError(f"Unknown embedding type: {etype}")

def physics_loss(y_pred, y_true, sample_w, loss_w):
    mse_loss = (((y_true - y_pred)**2) * sample_w).mean()
    mono_violations = torch.relu(y_pred[:, 1:] - y_pred[:, :-1])
    mono_loss = mono_violations.pow(2).mean()
    convex_violations = torch.relu(y_pred[:, :-2] - 2 * y_pred[:, 1:-1] + y_pred[:, 2:])
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
    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout):
        super().__init__()
        self.padding = (kernel_size - 1, 0)
        # Use torch.nn.utils.parametrize for weight_norm
        self.conv1 = nn_utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size))
        self.act1 = nn.GELU()
        self.norm1 = ChannelLayerNorm(out_ch)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn_utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size))
        self.act2 = nn.GELU()
        self.norm2 = ChannelLayerNorm(out_ch)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        out = F.pad(x, self.padding)
        out = self.drop1(self.norm1(self.act1(self.conv1(out))))
        out = F.pad(out, self.padding)
        out = self.drop2(self.norm2(self.act2(self.conv2(out))))
        return out + res

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


# Create a patched callback to work around the library compatibility issue
# This class satisfies the checks in newer PyTorch Lightning versions.

class PatchedPyTorchLightningPruningCallback(PyTorchLightningPruningCallback):
    def state_dict(self):
        """Called when saving a checkpoint to satisfy PyTorch Lightning's checks."""
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Called when loading a checkpoint."""
        pass
# ----------------- END NEW CLASS -----------------

class IVDataset(Dataset):
    def __init__(self, cfg, split, param_tf, scalar_tf):
        self.cfg = cfg
        self.split = split
        data = np.load(cfg['dataset']['paths']['preprocessed_npz'], allow_pickle=True)
        split_labels, indices = data['split_labels'], np.where(data['split_labels'] == split)[0]
        self.v_slices = torch.from_numpy(data['v_slices'][indices])
        self.i_slices_scaled = torch.from_numpy(data['i_slices_scaled'][indices])
        self.sample_weights = torch.from_numpy(data['sample_weights'][indices])
        self.i_slices = torch.from_numpy(data['i_slices'][indices])
        params_df = pd.read_csv(cfg['dataset']['paths']['params_csv'], header=None, names=COLNAMES)
        params_df_valid = params_df.iloc[data['valid_indices']].reset_index(drop=True)
        scalar_df = pd.DataFrame({'I_ref': data['i_slices'][:, 0], 'V_mpp': data['v_slices'][:, 5], 'I_mpp': data['i_slices'][:, 5]})
        X_params_full = param_tf.transform(params_df_valid).astype(np.float32)
        X_scalar_full = scalar_tf.transform(scalar_df).astype(np.float32)
        X_combined = np.concatenate([X_params_full, X_scalar_full], axis=1)
        self.X = torch.from_numpy(X_combined[indices])
        self.isc_vals = torch.from_numpy(data['isc_vals'][indices])
        # Update param_dim dynamically if it's the first time
        if 'param_dim' not in cfg['model'] or cfg['model']['param_dim'] != X_combined.shape[1]:
            log.info(f"Updating model.param_dim from {cfg['model']['param_dim']} to {X_combined.shape[1]}")
            cfg['model']['param_dim'] = X_combined.shape[1]

    def __len__(self):
        return len(self.v_slices)
    def __getitem__(self, idx):
        return {'X_combined': self.X[idx], 'voltage': self.v_slices[idx], 'current_scaled': self.i_slices_scaled[idx], 'sample_w': self.sample_weights[idx], 'isc': self.isc_vals[idx], 'current_original': self.i_slices[idx]}

class IVDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.param_tf, self.scalar_tf = None, None
    def prepare_data(self):
        if not Path(self.cfg['dataset']['paths']['preprocessed_npz']).exists():
            log.info("Preprocessed data not found. Running preprocessing...")
            self._preprocess_and_save()
        else:
            log.info("Found preprocessed data. Skipping preprocessing.")
    def setup(self, stage=None):
        if self.param_tf is None:
            self.param_tf = joblib.load(self.cfg['dataset']['paths']['param_transformer'])
            self.scalar_tf = joblib.load(self.cfg['dataset']['paths']['scalar_transformer'])
        if stage in ("fit", None):
            self.train_dataset = IVDataset(self.cfg, 'train', self.param_tf, self.scalar_tf)
            self.val_dataset = IVDataset(self.cfg, 'val', self.param_tf, self.scalar_tf)
        if stage in ("test", None):
            self.test_dataset = IVDataset(self.cfg, 'test', self.param_tf, self.scalar_tf)
    def train_dataloader(self): return DataLoader(self.train_dataset, **self.cfg['dataset']['dataloader'], shuffle=True)
    def val_dataloader(self): return DataLoader(self.val_dataset, **self.cfg['dataset']['dataloader'])
    def test_dataloader(self): return DataLoader(self.test_dataset, **self.cfg['dataset']['dataloader'])
    def _preprocess_and_save(self):
        log.info("--- Starting Data Preprocessing ---")
        cfg, paths = self.cfg, self.cfg['dataset']['paths']
        params_df, iv_data_raw = pd.read_csv(paths['params_csv'], header=None, names=COLNAMES), np.loadtxt(paths['iv_raw_txt'], delimiter=',')
        full_v_grid = np.concatenate([np.arange(0, 0.4 + 1e-8, 0.1), np.arange(0.425, 1.4 + 1e-8, 0.025)]).astype(np.float32)
        pchip_cfg = cfg['dataset']['pchip']
        seq_len = pchip_cfg['n_pre_mpp'] + 1 + pchip_cfg['n_post_mpp']
        results = [process_iv_with_pchip(iv_data_raw[i], full_v_grid, pchip_cfg['n_pre_mpp'], pchip_cfg['n_post_mpp'], pchip_cfg['v_max'], pchip_cfg['n_fine']) for i in tqdm(range(len(iv_data_raw)), desc="PCHIP")]
        valid_indices, v_slices, i_slices = [], [], []
        for i, res in enumerate(results):
            if res is not None and res[1][0] > 1e-9 and res[0].shape[0] == seq_len:
                valid_indices.append(i); v_slices.append(res[0]); i_slices.append(res[1])
        log.info(f"Retained {len(valid_indices)} / {len(iv_data_raw)} valid curves after PCHIP & Isc filtering.")
        v_slices, i_slices, valid_indices = np.array(v_slices), np.array(i_slices), np.array(valid_indices)
        isc_vals, i_slices_scaled = zip(*[normalize_and_scale_by_isc(c) for c in i_slices])
        isc_vals, i_slices_scaled = np.array(isc_vals), np.array(i_slices_scaled)
        sample_weights = compute_curvature_weights(i_slices_scaled, **cfg['dataset']['curvature_weighting'])
        params_df_valid = params_df.iloc[valid_indices].reset_index(drop=True)
        param_transformer = get_param_transformer(COLNAMES)
        param_transformer.fit(params_df_valid); joblib.dump(param_transformer, paths['param_transformer'])
        # Correctly get mpp index
        mpp_idx = pchip_cfg['n_pre_mpp']
        scalar_df = pd.DataFrame({'I_ref': i_slices[:, 0], 'V_mpp': v_slices[:, mpp_idx], 'I_mpp': i_slices[:, mpp_idx]})
        scalar_transformer = Pipeline([('scaler', StandardScaler())])
        scalar_transformer.fit(scalar_df); joblib.dump(scalar_transformer, paths['scalar_transformer'])
        all_indices = np.arange(len(valid_indices))
        train_val_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=cfg['train']['seed'])
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=cfg['train']['seed'])
        split_labels = np.array([''] * len(all_indices), dtype=object)
        split_labels[train_idx], split_labels[val_idx], split_labels[test_idx] = 'train', 'val', 'test'
        np.savez(paths['preprocessed_npz'], v_slices=v_slices, i_slices=i_slices, i_slices_scaled=i_slices_scaled, sample_weights=sample_weights, isc_vals=isc_vals, valid_indices=valid_indices, split_labels=split_labels)
        log.info(f"Saved all preprocessed data and transformers to {paths['output_dir']}")

class PhysicsIVSystem(pl.LightningModule):
    # UPDATE THE __init__ METHOD SIGNATURE
    def __init__(self, cfg, warmup_steps, total_steps, trial: typing.Optional[optuna.trial.Trial] = None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.hparams.warmup_steps, self.hparams.total_steps = warmup_steps, total_steps
        self.trial = trial # Store the trial object
        mcfg, mlp_layers, in_dim = self.hparams, [], self.hparams['model']['param_dim']
        for units in mcfg['model']['dense_units']:
            mlp_layers.extend([nn.Linear(in_dim, units), nn.GELU(), nn.LayerNorm(units), nn.Dropout(mcfg['model']['dropout'])]); in_dim = units
        self.param_mlp = nn.Sequential(*mlp_layers)
        self.pos_embed = make_positional_embedding(mcfg)
        seq_input_dim = mcfg['model']['dense_units'][-1] + self.pos_embed.out_dim
        filters, kernel, dropout, heads = mcfg['model']['filters'], mcfg['model']['kernel'], mcfg['model']['dropout'], mcfg['model']['heads']
        self.tcn1 = TemporalBlock(seq_input_dim, filters[0], kernel, dropout)
        self.attn = SelfAttentionBlock(filters[0], heads, dropout)
        self.tcn2 = TemporalBlock(filters[0], filters[1], kernel, dropout)
        self.out_head = nn.Linear(filters[1], 1)
        self.apply(self._init_weights)
    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to perform pruning.
        """
        # No pruning on the sanity check run
        if self.trainer.sanity_checking:
            return

        val_loss = self.trainer.callback_metrics.get("val_loss")

        if val_loss is not None and self.trial is not None:
            # 1. Report the validation loss to Optuna
            self.trial.report(val_loss, self.current_epoch)

            # 2. Check if the trial should be pruned
            if self.trial.should_prune():
                message = f"Trial pruned at epoch {self.current_epoch}."
                raise optuna.exceptions.TrialPruned(message)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None: nn.init.zeros_(module.bias)
    def forward(self, X_combined, voltage):
        B, L = voltage.shape
        p = self.param_mlp(X_combined)
        v_emb = self.pos_embed(voltage)
        p_rep = p.unsqueeze(1).expand(-1, L, -1)
        x = torch.cat([p_rep, v_emb], dim=-1).transpose(1, 2)
        x = self.tcn1(x).transpose(1, 2)
        x = self.attn(x).transpose(1, 2)
        x = self.tcn2(x).transpose(1, 2)
        return self.out_head(x).squeeze(-1)
    def _step(self, batch, stage):
        y_pred = self(batch['X_combined'], batch['voltage'])
        loss, comps = physics_loss(y_pred, batch['current_scaled'], batch['sample_w'], self.hparams['model']['loss_weights'])
        self.log_dict({f'{stage}_{k}': v for k, v in comps.items()}, on_step=False, on_epoch=True)
        self.log(f'{stage}_loss', loss, prog_bar=(stage == 'val'), on_step=False, on_epoch=True)
        return loss
    def training_step(self, b, _): return self._step(b, 'train')
    def validation_step(self, b, _): return self._step(b, 'val')
    def test_step(self, b, _): return self._step(b, 'test')
    def predict_step(self, batch, _, __=0): return self(batch['X_combined'], batch['voltage'])
    def configure_optimizers(self):
        opt_cfg, final_lr = self.hparams['optimizer'], self.hparams['optimizer']['lr'] * self.hparams['optimizer']['final_lr_ratio']
        optimizer = torch.optim.AdamW(self.parameters(), lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'])
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-7, end_factor=1.0, total_iters=self.hparams.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.total_steps - self.hparams.warmup_steps, eta_min=final_lr)
        return [optimizer], [{'scheduler': torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.hparams.warmup_steps]), 'interval': 'step'}]

# ──────────────────────────────────────────────────────────────────────────────
#   EVALUATION FUNCTION (IMPROVED OUTPUT CLARITY)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_and_plot(trainer: pl.Trainer, model: pl.LightningModule, datamodule: IVDataModule, run_dir: Path):
    """
    Runs final evaluation on the test set, calculates metrics on denormalized
    data, prints them clearly, and saves plots and metrics to a file.
    """
    log.info("--- Starting Final Evaluation and Plotting ---")
    datamodule.setup('test')
    test_dl = datamodule.test_dataloader()

    # Get predictions on the test set using the best model
    predictions_scaled = trainer.predict(model, dataloaders=test_dl, ckpt_path="best")
    all_preds_scaled = torch.cat([p.cpu() for p in predictions_scaled], dim=0)

    # Gather ground truth data
    all_true_original, all_voltages, all_isc = [], [], []
    for batch in test_dl:
        all_true_original.append(batch['current_original'])
        all_voltages.append(batch['voltage'])
        all_isc.append(batch['isc'])
    all_true_original = torch.cat(all_true_original, dim=0)
    all_voltages = torch.cat(all_voltages, dim=0)
    all_isc = torch.cat(all_isc, dim=0)

    log.info("Denormalizing predicted values for final metrics...")
    all_preds_denorm = ((all_preds_scaled + 1.0) / 2.0) * all_isc.unsqueeze(1)
    y_true_flat = all_true_original.flatten().numpy()
    y_pred_flat = all_preds_denorm.flatten().numpy()
    mae, rmse = mean_absolute_error(y_true_flat, y_pred_flat), np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2_global = r2_score(y_true_flat, y_pred_flat)
    r2_per_curve = np.array([r2_score(all_true_original[i], all_preds_denorm[i]) for i in range(all_true_original.shape[0])])

    print("\n" + "="*80)
    print(" " * 20 + "FINAL MODEL EVALUATION (ON TEST SET)")
    print("="*80 + "\nMetrics are calculated on DENORMALIZED data (original units).\n")
    print(f"-> Mean Absolute Error (MAE):          {mae:.6f}")
    print(f"-> Root Mean Squared Error (RMSE):     {rmse:.6f}")
    print(f"-> Global R^2 Score:                   {r2_global:.6f}")
    print(f"-> R^2 Score per Curve (Mean ± Std):   {r2_per_curve.mean():.4f} ± {r2_per_curve.std():.4f}")
    print("="*80)

    metrics_path = run_dir / "evaluation_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n" + f"Root Mean Squared Error (RMSE): {rmse:.6f}\n" +
                f"Global R^2 Score: {r2_global:.6f}\n" + f"R^2 Score per Curve (Mean): {r2_per_curve.mean():.6f}\n" +
                f"R^2 Score per Curve (Std Dev): {r2_per_curve.std():.6f}\n")
    log.info(f"Saved evaluation metrics to: {metrics_path}")

    log.info("Generating and saving comparison plots...")
    num_plots = min(9, len(all_true_original))
    plot_indices = np.random.choice(len(all_true_original), num_plots, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    for i, idx in enumerate(plot_indices):
        ax = axes[i]
        v, i_true, i_pred = all_voltages[idx].numpy(), all_true_original[idx].numpy(), all_preds_denorm[idx].numpy()
        ax.plot(v, i_true, 'bo-', label='Original', markersize=4, linewidth=2)
        ax.plot(v, i_pred, 'r.--', label='Reconstructed', markersize=4, linewidth=1.5)
        ax.set_title(f"Test Sample #{idx}"), ax.set_xlabel("Voltage (V)"), ax.set_ylabel("Current (A/m$^2$)")
        ax.grid(True, linestyle='--', alpha=0.6), ax.legend()
    for i in range(num_plots, len(axes)): axes[i].set_visible(False)
    fig.suptitle("Original vs. Reconstructed I-V Curves", fontsize=20, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = run_dir / "reconstruction_plots.png"
    plt.savefig(plot_path, dpi=300)
    log.info(f"Saved reconstruction plot to: {plot_path}\n")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
#   HPO OBJECTIVE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
#   HPO OBJECTIVE FUNCTION (Corrected)
# ──────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.trial.Trial, base_cfg: dict, datamodule: IVDataModule) -> float:
    """The Optuna objective function for one HPO trial."""
    cfg = copy.deepcopy(base_cfg)

    # 1. Sample hyperparameters (this part is unchanged)
    cfg['optimizer']['lr'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    cfg['optimizer']['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    cfg['optimizer']['warmup_epochs'] = trial.suggest_int("warmup_epochs", 0, 10)
    cfg['model']['dropout'] = trial.suggest_float("dropout", 0.0, 0.5)
    cfg['model']['heads'] = trial.suggest_categorical("heads", [2, 4, 8])
    cfg['model']['kernel'] = trial.suggest_categorical("kernel", [3, 5, 7])

    # Suggest filter configuration as a categorical choice from predefined pairs
    filter_choice = trial.suggest_categorical("filters", ["64_32", "128_64", "256_128"])
    cfg['model']['filters'] = [int(f) for f in filter_choice.split('_')]

    # Suggest embedding type and its parameters
    cfg['model']['embedding_type'] = trial.suggest_categorical("embedding_type", ['fourier', 'fourier_clipped', 'gaussian'])
    if 'fourier' in cfg['model']['embedding_type']:
        cfg['model']['fourier_bands'] = trial.suggest_int("fourier_bands", 8, 32)
    else: # gaussian
        cfg['model']['gaussian_bands'] = trial.suggest_int("gaussian_bands", 8, 32)
        cfg['model']['gaussian_sigma'] = trial.suggest_float("gaussian_sigma", 0.01, 0.5)

    # 2. Setup model and trainer for this trial
    seed_everything(cfg['train']['seed'])
    batches_per_epoch = len(datamodule.train_dataloader())
    warmup_steps = cfg['optimizer']['warmup_epochs'] * batches_per_epoch
    total_steps = cfg['trainer']['max_epochs'] * batches_per_epoch
    
    # UPDATE THIS LINE: Pass the trial object directly to the model
    model = PhysicsIVSystem(cfg, warmup_steps, total_steps, trial=trial)

    # UPDATE THE TRAINER: Remove the 'callbacks' argument completely
    trainer = pl.Trainer(
        **cfg['trainer'],
        logger=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    # 4. Train and return metric (add explicit exception handling)
    try:
        trainer.fit(model, datamodule=datamodule)
    except optuna.exceptions.TrialPruned:
        # Return a large value to tell Optuna this trial was bad.
        return float('inf')

    # Return the best validation loss
    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss.item() if val_loss is not None else float('inf')

# ──────────────────────────────────────────────────────────────────────────────
#   MAIN EXECUTION SCRIPT
# ──────────────────────────────────────────────────────────────────────────────
def run_hpo_study(cfg: dict):
    """Runs the full Optuna HPO study."""
    log.info("--- Preparing for HPO Study ---")
    seed_everything(cfg['train']['seed'])
    
    # Prepare data once before starting the study
    datamodule = IVDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    
    # Define pruner and study
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=cfg['hpo']['study_name'],
        direction="minimize",
        pruner=pruner
    )
    
    log.info(f"--- Starting HPO Study: {cfg['hpo']['study_name']} for {cfg['hpo']['n_trials']} trials ---")
    study.optimize(
        lambda trial: objective(trial, cfg, datamodule),
        n_trials=cfg['hpo']['n_trials'],
        timeout=None # Can set a time limit in seconds
    )

    log.info(f"--- HPO Study Finished ---")
    log.info(f"Best trial number: {study.best_trial.number}")
    log.info(f"Best validation loss: {study.best_value:.6f}")
    log.info("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        log.info(f"  - {key}: {value}")

    # Save the best parameters to a file
    best_params_path = OUTPUT_DIR / "best_hpo_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)
    log.info(f"Saved best hyperparameters to {best_params_path}")

    return study.best_trial.params

def run_final_training(cfg: dict, best_params: dict):
    """
    Trains the final model using the best hyperparameters found during the HPO study.
    """
    log.info("\n" + "="*80 + "\n--- STARTING FINAL TRAINING WITH BEST HYPERPARAMETERS ---\n" + "="*80)

    # 1. Update config with best params
    cfg['optimizer']['lr'] = best_params['lr']
    cfg['optimizer']['weight_decay'] = best_params['weight_decay']
    cfg['optimizer']['warmup_epochs'] = best_params['warmup_epochs']
    cfg['model']['dropout'] = best_params['dropout']
    cfg['model']['heads'] = best_params['heads']
    cfg['model']['kernel'] = best_params['kernel']
    cfg['model']['filters'] = [int(f) for f in best_params['filters'].split('_')]
    cfg['model']['embedding_type'] = best_params['embedding_type']
    if 'fourier' in best_params['embedding_type']:
        cfg['model']['fourier_bands'] = best_params['fourier_bands']
    else:
        cfg['model']['gaussian_bands'] = best_params['gaussian_bands']
        cfg['model']['gaussian_sigma'] = best_params['gaussian_sigma']

    # Use longer training for the final model
    cfg['trainer']['max_epochs'] = cfg['final_train']['max_epochs']

    # 2. Setup final run
    run_name = f"FINAL-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log.info(f"Starting final run: '{run_name}'")
    seed_everything(cfg['train']['seed'])

    datamodule = IVDataModule(cfg)
    # No need for prepare_data, it's already done.
    datamodule.setup(stage='fit')

    batches_per_epoch = len(datamodule.train_dataloader())
    warmup_steps = cfg['optimizer']['warmup_epochs'] * batches_per_epoch
    total_steps = cfg['trainer']['max_epochs'] * batches_per_epoch
    model = PhysicsIVSystem(cfg, warmup_steps, total_steps)
    log.info(f"Final model instantiated with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # 3. Setup callbacks and logger for the final run
    final_run_dir = OUTPUT_DIR / run_name
    final_run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_cb = ModelCheckpoint(dirpath=final_run_dir, monitor="val_loss", mode="min", save_top_k=1, filename="best-model-{epoch:02d}")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=20, mode="min")
    logger = TensorBoardLogger(str(OUTPUT_DIR / "tb_logs"), name=run_name)
    
    trainer = pl.Trainer(
        **cfg['trainer'],
        default_root_dir=final_run_dir,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stopping_cb],
        enable_progress_bar=True, # Enable progress bar for final run
        enable_checkpointing=True
    )
    
    log.info("--- Starting Final Training ---")
    trainer.fit(model, datamodule=datamodule)
    
    log.info("--- Starting Testing on Best Checkpoint (for logging scaled metrics) ---")
    # The trainer automatically uses the best checkpoint for `test`
    test_results = trainer.test(datamodule=datamodule, ckpt_path="best")
    log.info(f"Test results (on scaled data): {test_results[0]}")

    # --- FINAL EVALUATION AND PLOTTING ---
    evaluate_and_plot(trainer, model, datamodule, run_dir=final_run_dir)


if __name__ == "__main__":
    if not (Path(INPUT_FILE_PARAMS).exists() and Path(INPUT_FILE_IV).exists()):
        log.error("="*80 + "\nInput data files not found! Please update the paths at the top of the script.\n" + f"Checked for: {INPUT_FILE_PARAMS} and {INPUT_FILE_IV}\n" + "="*80)
    else:
        # Step 1: Run the HPO study to find the best parameters
        best_hyperparameters = run_hpo_study(CONFIG)
        
        # Step 2: Run the final, full training and evaluation with the best parameters
        run_final_training(CONFIG, best_hyperparameters)