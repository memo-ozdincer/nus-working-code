# a little more abstract model, it works using a hypernetwork to generate the weights of a SIREN model
# Written in PyTorch, written for Colab. Same preprocessing pipeline as the original model the only difference is the padding value. Also, this is per-curve normalization, not per-batch normalization.

# ## 0. Project-level Imports and Setup
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple, Optional, Callable

import joblib
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting, suitable for Colab saving files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast 
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Constants
COLNAMES = [
    'Eg', 'NCv', 'NCc', 'mu_e', 'mu_h', 'eps', 'A', 'Cn', 'Cp', 'Nt',
    'Et', 'nD', 'nA', 'thickness', 'T', 'Sn', 'Sp', 'Rs', 'Rsh',
    'G', 'light_intensity', 'Voc_ref', 'Jsc_ref', 'FF_ref', 'PCE_ref',
    'Qe_loss', 'R_loss', 'SRH_loss', 'series_loss', 'shunt_loss', 'other_loss'
]
N_FEATURES = len(COLNAMES)
VOLTAGE_POINTS_TARGET = 45
VOLTAGE_GRID_FIXED = np.linspace(0, 1.2, VOLTAGE_POINTS_TARGET, dtype=np.float32) 
RAW_PAD_VALUE = -999.0 
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGNED_LOG_I_REF = 1.0 

TRANSFORMED_PAD_VALUE: Optional[float] = None # Will be set after signed_log_transform is defined

# Rich console for pretty printing
console = Console()

# ## 1. Configuration (Replaces argparse for Colab)

class ColabArgs:
    def __init__(self):
        # Data paths
        self.data_root: str = "/content/data"
        self.input_file: str = "input.txt"
        self.output_file: str = "output.txt"
        self.output_dir: str = "/content/outputs"

        # Training parameters
        self.epochs: int = 200
        self.batch_size: int = 2000 
        self.lr: float = 4e-4 
        self.no_gpu: bool = False 
        self.quick: bool = False 
        self.seed: int = 42
        self.num_workers: int = 2  # For val/test loaders
        self.prefetch_factor: int = 2 
        
        # Model specific parameters
        self.siren_hidden_dim: int = 32 
        self.siren_num_layers: int = 2   
        self.siren_omega_0: float = 30.0
        self.hypernet_ranks: int = 5    
        self.hypernet_layers: List[int] = [128, 128] 

        # Training specific parameters
        self.voltage_samples_train_post_warmup: int = 16 
        self.warmup_epochs_full_curve: int = 10 # Epochs for full curve sampling & diode loss warmup
        
        self.scheduler_type: str = "onecycle" 
        self.warmup_epochs: int = 20 # Used by OneCycleLR's pct_start indirectly if total_steps based on this
        self.onecycle_pct_start: float = 0.2 
        self.onecycle_div_factor: float = 100.0 
        self.onecycle_final_div_factor: float = 1e4
        self.patience: int = 25
        self.min_delta: float = 1e-4
        self.checkpoint_freq: int = 10 
        self.gradient_clip_val: float = 1.0 
        self.huber_delta: float = 1.0 # For Huber loss

        # Loss weights
        self.lambda_reconstruction: float = 1.0 # Main weight for Huber loss
        self.lambda_smoothness: float = 1e-2
        self.lambda_monotonicity: float = 5e-2
        self.lambda_diode: float = 1e-3
        self.lambda_knee_weighted_loss: float = 0.1 # Example: 0.1, tune as needed
        self.knee_weight_alpha: float = 5.0 
        self.knee_weight_sigma_v: float = 0.05 
        self.lambda_curvature: float = 1e-3 # Example: 1e-3, tune as needed
        self.lambda_voc_constraint: float = 0.1 # Example: 0.1, tune as needed
        self.lambda_sc_slope_constraint: float = 0.05 # Example: 0.05, tune as needed

        # Optimizer
        self.diode_predictor_lr_factor: float = 1.0 

        # Optional features
        self.use_cnn_refiner: bool = False
        self.cnn_depth: int = 6
        self.cnn_kernel_size: int = 3
        self.cnn_channels: int = 64
        self.active_learning: bool = False 
        self.predict_only: bool = False 
        self.model_path_hyper_inr: str = "hyper_inr.pt" 
        self.model_path_cnn_refiner: str = "cnn_refiner.pt" 
        self.preprocessor_path: str = "preprocessor.pkl" 
        self.enable_torch_compile: bool = True 
        self.profile_run: bool = False 

        if self.quick:
            self.epochs = 10
            self.batch_size = min(self.batch_size, 128)
            self.warmup_epochs_full_curve = max(1, self.epochs // 10)
            # Disable more complex/experimental losses in quick mode for stability/speed
            self.lambda_knee_weighted_loss = 0.0
            self.lambda_curvature = 0.0
            self.lambda_voc_constraint = 0.0
            self.lambda_sc_slope_constraint = 0.0
            console.print("[yellow]Running in QUICK mode. Complex losses disabled.[/yellow]")

args = ColabArgs()

# ## 1. Determinism and Utility Functions (Continued)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    console.print(f"[green]Global seed set to {seed}[/green]")

# 0. Signed‐log scaling utilities for I–V curves
def signed_log_transform(arr: np.ndarray, I_ref: float = SIGNED_LOG_I_REF) -> np.ndarray:
    return np.sign(arr) * np.log1p(np.abs(arr) / I_ref)

def signed_log_inverse(arr: np.ndarray, I_ref: float = SIGNED_LOG_I_REF) -> np.ndarray:
    return np.sign(arr) * I_ref * (np.expm1(np.abs(arr)))

# FIX ⚠️ Bug 3: Calculate TRANSFORMED_PAD_VALUE
TRANSFORMED_PAD_VALUE = signed_log_transform(np.array([RAW_PAD_VALUE]), I_ref=SIGNED_LOG_I_REF)[0]
console.print(f"Raw PAD_VALUE: {RAW_PAD_VALUE}, Transformed PAD_VALUE: {TRANSFORMED_PAD_VALUE:.4f}")


# ## 2. Data Specification and Preprocessing Pipeline

def load_input_features(filepath: str, colnames: List[str]) -> pd.DataFrame:
    if not os.path.exists(filepath):
        console.print(f"[bold red]Error: Input file not found at {filepath}[/bold red]")
        raise FileNotFoundError(f"Input file not found at {filepath}")
    try:
        df = pd.read_csv(filepath, sep=',', header=None, names=colnames, dtype=np.float64)
        return df
    except Exception as e:
        console.print(f"[bold red]Error reading input file {filepath}: {e}[/bold red]")
        raise

def estimate_v_knee(
    currents_original_curve: np.ndarray, 
    voltages_original_curve: np.ndarray, 
    isc_original_value: float
) -> float:
    """Estimates V_knee as the voltage where current is closest to 50% of Isc."""
    if isc_original_value <= 1e-9 or len(currents_original_curve) == 0: # Avoid issues with zero/negative Isc
        return voltages_original_curve[0] if len(voltages_original_curve) > 0 else 0.0 
        
    target_current_for_knee = 0.5 * isc_original_value
    valid_indices = np.where(currents_original_curve > 0)[0] # Consider only positive current part
    if len(valid_indices) == 0:
         return voltages_original_curve[0] if len(voltages_original_curve) > 0 else 0.0

    # Find the index within valid_indices that's closest to the target current
    closest_idx_in_valid = np.argmin(np.abs(currents_original_curve[valid_indices] - target_current_for_knee))
    actual_closest_idx = valid_indices[closest_idx_in_valid]
    
    # Ensure index is within bounds of voltages_original_curve
    if actual_closest_idx < len(voltages_original_curve):
        return voltages_original_curve[actual_closest_idx]
    else: # Fallback if something is off with indexing
        return voltages_original_curve[0] if len(voltages_original_curve) > 0 else 0.0


def process_iv_curve(
    currents_str: str, 
    target_len: int, 
    transformed_pad_value: float, # Use transformed pad value
    voltage_grid_fixed_np: np.ndarray,
    i_ref_for_transform: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], Optional[float]]:
    """
    Processes a single I-V curve string.
    Returns (transformed_currents, mask, Voc_value, Isc_original_value, V_knee_estimated).
    """
    try:
        currents_original = np.array([float(c) for c in currents_str.strip().split(',')], dtype=np.float64)
    except ValueError:
        return None, None, None, None, None

    isc_original_value = currents_original[0] if len(currents_original) > 0 and voltage_grid_fixed_np[0] == 0 else 0.0

    cut_off_idx = np.where(currents_original <= 0)[0]
    if len(cut_off_idx) > 0:
        actual_len = cut_off_idx[0] + 1
        currents_processed_original = currents_original[:actual_len]
        voltages_processed = voltage_grid_fixed_np[:actual_len]
        voc_val = voltage_grid_fixed_np[cut_off_idx[0]] if cut_off_idx[0] < len(voltage_grid_fixed_np) else (voltage_grid_fixed_np[-1] if actual_len > 0 else 0.0)
    else: 
        actual_len = min(len(currents_original), target_len)
        currents_processed_original = currents_original[:actual_len]
        voltages_processed = voltage_grid_fixed_np[:actual_len]
        voc_val = voltage_grid_fixed_np[actual_len -1] if actual_len > 0 else 0.0

    if actual_len == 0:
        if len(currents_original)>0 and currents_original[0] <= 0: 
            currents_processed_original = currents_original[0:1] 
            voltages_processed = voltage_grid_fixed_np[0:1]
            actual_len = 1
            voc_val = voltage_grid_fixed_np[0] 
            if isc_original_value == 0.0 : isc_original_value = currents_original[0] # Update if Isc was default 0
        else: 
            return None, None, None, None, None

    v_knee_estimated = estimate_v_knee(currents_processed_original, voltages_processed, isc_original_value)

    if actual_len > 0:
        transformed_segment = signed_log_transform(currents_processed_original, I_ref=i_ref_for_transform)
    else: 
        transformed_segment = np.array([])
    
    padded_currents_transformed = np.full(target_len, transformed_pad_value, dtype=np.float32)
    mask = np.zeros(target_len, dtype=np.float32)

    if actual_len > 0:
        padded_currents_transformed[:actual_len] = transformed_segment.astype(np.float32)
        mask[:actual_len] = 1.0
    
    return padded_currents_transformed, mask, voc_val, isc_original_value, v_knee_estimated


def load_output_curves(
    filepath: str, 
    target_len: int = VOLTAGE_POINTS_TARGET, 
    transformed_pad_value: float = TRANSFORMED_PAD_VALUE, 
    voltage_grid_fixed_np: np.ndarray = VOLTAGE_GRID_FIXED,
    i_ref_for_transform: float = SIGNED_LOG_I_REF
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads output I-V curves, processes them (including V_knee estimation and signed-log transform),
    and returns y_transformed, masks, voc_values, isc_original_values, v_knee_estimations.
    """
    if not os.path.exists(filepath):
        console.print(f"[bold red]Error: Output file not found at {filepath}[/bold red]")
        raise FileNotFoundError(f"Output file not found at {filepath}")

    with open(filepath, 'r') as f: lines = f.readlines()
    if not lines:
        console.print(f"[bold red]Error: Output file {filepath} is empty.[/bold red]")
        raise ValueError("Output file is empty.")
        
    results = [] 
    
    console.print(f"Processing {len(lines)} curves from output file (estimating V_knee)...")
    for line_idx, line in enumerate(lines):
        if not line.strip(): 
            console.print(f"[yellow]Skipping empty line {line_idx+1} in {filepath}[/yellow]")
            continue
        results.append(process_iv_curve(line, target_len, transformed_pad_value, voltage_grid_fixed_np, i_ref_for_transform))

    valid_results = [r for r in results if r[0] is not None and r[1] is not None and r[2] is not None and r[3] is not None and r[4] is not None]
    if not valid_results:
        raise ValueError("No valid I-V curves found after processing output file.")

    y_transformed = np.array([r[0] for r in valid_results], dtype=np.float32)
    masks = np.array([r[1] for r in valid_results], dtype=np.float32)
    voc_values = np.array([r[2] for r in valid_results], dtype=np.float32) 
    isc_original_values = np.array([r[3] for r in valid_results], dtype=np.float32)
    v_knee_estimations = np.array([r[4] for r in valid_results], dtype=np.float32)
    
    console.print(f"Successfully processed {len(y_transformed)} I-V curves.")
    return y_transformed, masks, voc_values, isc_original_values, v_knee_estimations


class IVPerovskiteDataset(Dataset):
    def __init__(self, X: np.ndarray, y_transformed: np.ndarray, mask: np.ndarray, 
                 Voc: np.ndarray, Isc_transformed: np.ndarray, 
                 Isc_original: np.ndarray, V_knee: np.ndarray):
        if not (X.shape[0] == y_transformed.shape[0] == mask.shape[0] == Voc.shape[0] == 
                Isc_transformed.shape[0] == Isc_original.shape[0] == V_knee.shape[0]):
            mismatches = {
                "X": X.shape[0], "y_transformed": y_transformed.shape[0], "mask": mask.shape[0],
                "Voc": Voc.shape[0], "Isc_transformed": Isc_transformed.shape[0],
                "Isc_original": Isc_original.shape[0], "V_knee": V_knee.shape[0]
            }
            raise ValueError(f"Mismatch in lengths of inputs during IVPerovskiteDataset init: {mismatches}")

        self.X = torch.from_numpy(X).float()
        self.y_transformed = torch.from_numpy(y_transformed).float()
        self.mask = torch.from_numpy(mask).float()
        self.Voc = torch.from_numpy(Voc).float()
        self.Isc_transformed = torch.from_numpy(Isc_transformed).float()
        self.Isc_original = torch.from_numpy(Isc_original).float()
        self.V_knee = torch.from_numpy(V_knee).float()


    def __len__(self) -> int: return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.X[idx],
            "y_target": self.y_transformed[idx], 
            "mask": self.mask[idx],
            "voc_true": self.Voc[idx],
            "isc_transformed": self.Isc_transformed[idx],
            "isc_original": self.Isc_original[idx], 
            "v_knee": self.V_knee[idx]              
        }

def create_dataloaders(
    current_args: ColabArgs, 
    X_train: np.ndarray, y_train_t: np.ndarray, m_train: np.ndarray, v_train: np.ndarray, isc_train_t: np.ndarray, isc_train_o: np.ndarray, vk_train: np.ndarray,
    X_val: np.ndarray, y_val_t: np.ndarray, m_val: np.ndarray, v_val: np.ndarray, isc_val_t: np.ndarray, isc_val_o: np.ndarray, vk_val: np.ndarray,
    X_test: Optional[np.ndarray]=None, y_test_t: Optional[np.ndarray]=None, m_test: Optional[np.ndarray]=None, v_test: Optional[np.ndarray]=None, isc_test_t: Optional[np.ndarray]=None, isc_test_o: Optional[np.ndarray]=None, vk_test: Optional[np.ndarray]=None
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    
    train_dataset = IVPerovskiteDataset(X_train, y_train_t, m_train, v_train, isc_train_t, isc_train_o, vk_train)
    val_dataset = IVPerovskiteDataset(X_val, y_val_t, m_val, v_val, isc_val_t, isc_val_o, vk_val) if len(X_val) > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_size=current_args.batch_size, shuffle=True,
        num_workers=2, prefetch_factor=2 if 2 > 0 else None, # Fixed for train_loader
        pin_memory=True, drop_last=True
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=current_args.batch_size, shuffle=False, 
            num_workers=current_args.num_workers, pin_memory=True, # Uses args for val/test
            prefetch_factor=current_args.prefetch_factor if current_args.num_workers > 0 else None
        )
    
    test_loader = None
    test_dataset = None 
    if X_test is not None and y_test_t is not None and m_test is not None and v_test is not None and isc_test_t is not None and isc_test_o is not None and vk_test is not None:
        if len(X_test) > 0: 
            test_dataset = IVPerovskiteDataset(X_test, y_test_t, m_test, v_test, isc_test_t, isc_test_o, vk_test)
            test_loader = DataLoader(
                test_dataset, batch_size=current_args.batch_size, shuffle=False, 
                num_workers=current_args.num_workers, pin_memory=True, # Uses args for val/test
                prefetch_factor=current_args.prefetch_factor if current_args.num_workers > 0 else None
            )

    console.print(f"Train DataLoader: {len(train_loader)} batches, {len(train_dataset)} samples.")
    if val_loader and val_dataset: 
        console.print(f"Validation DataLoader: {len(val_loader)} batches, {len(val_dataset)} samples.")
    else:
        console.print(f"Validation DataLoader: Not created (no validation data or empty validation set).")
    if test_loader and test_dataset: 
        console.print(f"Test DataLoader: {len(test_loader)} batches, {len(test_dataset)} samples.")
    else:
        console.print(f"Test DataLoader: Not created (no test data or empty test set).")
        
    return train_loader, val_loader, test_loader


# ## 3. Model Architecture

class SineActivation(nn.Module):
    def __init__(self, omega_0: float = 30.0): super().__init__(); self.omega_0 = omega_0
    def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.sin(self.omega_0 * x)

class SirenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega_0: float, is_first: bool = False, use_bias: bool = True):
        super().__init__(); self.in_features = in_features; self.out_features = out_features; self.linear = nn.Linear(in_features, out_features, bias=use_bias); self.activation = SineActivation(omega_0); self.is_first = is_first; self.omega_0 = omega_0; self.init_weights()
    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first: bound = 1.0 / self.in_features
            else: bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound); 
            if self.linear.bias is not None: self.linear.bias.uniform_(-bound, bound)

class SIREN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int, out_features: int, omega_0: float = 30.0, final_activation: Optional[nn.Module] = None):
        super().__init__(); self.in_features = in_features; self.hidden_features = hidden_features; self.out_features = out_features; self.omega_0 = omega_0; self.final_activation = final_activation
        _module_list = [SirenLayer(in_features, hidden_features, omega_0, is_first=True)]
        for _ in range(hidden_layers): _module_list.append(SirenLayer(hidden_features, hidden_features, omega_0, is_first=False))
        _module_list.append(nn.Linear(hidden_features, out_features)) 
        self.siren_layers_for_shape = nn.ModuleList(_module_list)

    def get_param_shapes(self, rank: Optional[int] = None) -> List[Tuple[Tuple[int, ...], Optional[Tuple[int, ...]], Tuple[int, ...]]]:
        param_shapes = []
        for _, spec_or_layer in enumerate(self.siren_layers_for_shape): # Use _ for i if not used
            if isinstance(spec_or_layer, SirenLayer): W_shape = (spec_or_layer.out_features, spec_or_layer.in_features); b_shape = (spec_or_layer.out_features,) if spec_or_layer.linear.bias is not None else tuple()
            elif isinstance(spec_or_layer, nn.Linear): W_shape = spec_or_layer.weight.shape; b_shape = spec_or_layer.bias.shape if spec_or_layer.bias is not None else tuple()
            else: raise TypeError(f"Unexpected layer type {type(spec_or_layer)}")
            is_siren_like_layer = isinstance(spec_or_layer, SirenLayer)
            if rank is not None and is_siren_like_layer : U_shape = (W_shape[0], rank); V_shape = (rank, W_shape[1]); param_shapes.append( (U_shape, V_shape, b_shape) )
            else: param_shapes.append( (W_shape, None, b_shape) ) 
        return param_shapes

    def forward(self, x_coords_batch: torch.Tensor, flat_params_batch: torch.Tensor, param_shapes: List[Tuple[Tuple[int, ...], Optional[Tuple[int, ...]], Tuple[int, ...]]]) -> torch.Tensor:
        x = x_coords_batch; batch_size = x.shape[0]; current_idx_in_flat_params = 0
        for i in range(len(param_shapes)): # Use i if needed for debugging, else _
            layer_spec_or_module = self.siren_layers_for_shape[i]; W_s, V_s, b_s = param_shapes[i]
            num_W_elements = np.prod(W_s); W_or_U_flat_b = flat_params_batch[:, current_idx_in_flat_params : current_idx_in_flat_params + num_W_elements]; W_or_U_b = W_or_U_flat_b.reshape(batch_size, *W_s); current_idx_in_flat_params += num_W_elements
            if V_s is not None and len(V_s) > 0: num_V_elements = np.prod(V_s); V_flat_b = flat_params_batch[:, current_idx_in_flat_params : current_idx_in_flat_params + num_V_elements]; V_b = V_flat_b.reshape(batch_size, *V_s); current_idx_in_flat_params += num_V_elements; W_actual_b = torch.bmm(W_or_U_b, V_b) 
            else: W_actual_b = W_or_U_b 
            x = torch.bmm(x, W_actual_b.transpose(1, 2))
            if b_s is not None and len(b_s) > 0: num_b_elements = np.prod(b_s); b_flat_b = flat_params_batch[:, current_idx_in_flat_params : current_idx_in_flat_params + num_b_elements]; b_b = b_flat_b.reshape(batch_size, *b_s) ; current_idx_in_flat_params += num_b_elements; x = x + b_b.unsqueeze(1) 
            if isinstance(layer_spec_or_module, SirenLayer): x = torch.sin(layer_spec_or_module.omega_0 * x)
        if self.final_activation is not None: x = self.final_activation(x)
        return x

class HyperNetwork(nn.Module):
    def __init__(self, input_dim: int, hypernet_layers_dims: List[int], target_param_count: int):
        super().__init__(); layers = []; current_dim = input_dim
        for hidden_dim in hypernet_layers_dims: layers.append(nn.Linear(current_dim, hidden_dim)); layers.append(nn.GELU()); current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, target_param_count)); self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)

class SmoothStep(nn.Module):
    def __init__(self, epsilon=1e-6): super().__init__(); self.epsilon = epsilon
    def forward(self, x: torch.Tensor) -> torch.Tensor: t = torch.clamp(x, 0.0, 1.0); return t * t * (3.0 - 2.0 * t)

class PhysicsAwareHyperINR(nn.Module):
    def __init__(self, 
                 device_param_dim: int, siren_in_features: int = 1, siren_hidden_features: int = 64, 
                 siren_hidden_layers: int = 3, siren_out_features: int = 1, siren_omega_0: float = 30.0,
                 hypernet_layers: List[int] = [256, 256, 512], hypernet_rank: Optional[int] = 10,
                 use_cnn_refiner: bool = False, cnn_config: Optional[Dict] = None,
                 predict_diode_params: bool = False,
                 lambda_voc_constraint_active: bool = False): # Changed to bool flag
        super().__init__()
        self.hypernet_rank = hypernet_rank
        self.lambda_voc_constraint_active = lambda_voc_constraint_active
        
        self.siren = SIREN(siren_in_features, siren_hidden_features, siren_hidden_layers, 
                           siren_out_features, siren_omega_0, final_activation=None) 
        
        self.siren_param_shapes_cache = self.siren.get_param_shapes(rank=hypernet_rank) 
        siren_total_params = sum(np.prod(W_s) + (np.prod(V_s) if V_s and len(V_s)>0 else 0) + (np.prod(b_s) if b_s and len(b_s)>0 else 0) for W_s, V_s, b_s in self.siren_param_shapes_cache)
        console.print(f"SIREN network requires {siren_total_params} parameters (rank {hypernet_rank}).")
        self.hyper_net = HyperNetwork(device_param_dim, hypernet_layers, siren_total_params)
        self.smooth_step = SmoothStep()
        
        self.cnn_refiner = None
        if use_cnn_refiner and cnn_config: self.cnn_refiner = CNNRefiner(1, 1, cnn_config.get("depth",6), cnn_config.get("kernel_size",3), cnn_config.get("channels",64)); console.print(f"CNN Refiner initialized.")
            
        self.diode_param_predictor = None
        if predict_diode_params: self.diode_param_predictor = nn.Sequential(nn.Linear(device_param_dim, 64), nn.GELU(), nn.Linear(64, 4) ); console.print("Diode parameter predictor head initialized.")

    def forward(self, device_params: torch.Tensor, voltage_coords: torch.Tensor, 
                true_voc: torch.Tensor, isc_transformed: torch.Tensor # true_voc is [B]
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size = device_params.shape[0]

        if voltage_coords.ndim == 2 and voltage_coords.shape[0] != batch_size : voltage_coords = voltage_coords.unsqueeze(0).expand(batch_size, -1, -1)
        elif voltage_coords.ndim == 1: voltage_coords = voltage_coords.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.siren.in_features)
        
        flat_siren_params_batch = self.hyper_net(device_params).to(torch.float32)

        siren_outputs_on_grid = self.siren(voltage_coords, flat_siren_params_batch, self.siren_param_shapes_cache)
        
        siren_output_at_true_voc: Optional[torch.Tensor] = None
        if self.lambda_voc_constraint_active: # Check if this loss is used
            true_voc_coords_for_siren = true_voc.view(batch_size, 1, 1).to(dtype=voltage_coords.dtype, device=voltage_coords.device)
            siren_output_at_true_voc = self.siren(true_voc_coords_for_siren, flat_siren_params_batch, self.siren_param_shapes_cache)
            siren_output_at_true_voc = siren_output_at_true_voc.squeeze(-1) # [B,1]

        voc_expanded = true_voc.view(batch_size, 1, 1) 
        isc_transformed_expanded = isc_transformed.view(batch_size, 1, 1)
        
        ratio_v_voc = voltage_coords / (voc_expanded + 1e-9) 
        sigma = self.smooth_step(ratio_v_voc) 
        i_pred_boundary_enforced = (1.0 - sigma) * isc_transformed_expanded + sigma * siren_outputs_on_grid
        final_pred = i_pred_boundary_enforced.squeeze(-1) 
        
        if self.cnn_refiner: final_pred = self.cnn_refiner(final_pred.unsqueeze(1)).squeeze(1) 
            
        predicted_diode_params = None
        if self.diode_param_predictor:
            raw = self.diode_param_predictor(device_params)
            I0   = 1e-12 + F.softplus(raw[:,0]); Rs = torch.clamp(F.softplus(raw[:,1]), 0.0, 5.0)
            Rsh  = torch.clamp(1e2 + F.softplus(raw[:,2]), 1e2, 1e4); n_id = torch.clamp(1.0 + F.softplus(raw[:,3]), 1.0, 2.0)
            predicted_diode_params = torch.stack([I0, Rs, Rsh, n_id], dim=1)
            
        return final_pred, predicted_diode_params, siren_output_at_true_voc

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **kwargs): super().__init__(); self.padding = (kernel_size - 1) * dilation; self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)
    def forward(self, x): x = self.conv(x); return x[:, :, :-self.padding] if self.padding != 0 else x

class CNNRefinerBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation): super().__init__(); self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation); self.norm1 = nn.LayerNorm(channels); self.relu1 = nn.GELU(); self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=1); self.norm2 = nn.LayerNorm(channels); self.relu2 = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor: identity = x; out = self.conv1(x); out = self.norm1(out.permute(0, 2, 1)).permute(0, 2, 1); out = self.relu1(out); out = self.conv2(out); out = self.norm2(out.permute(0, 2, 1)).permute(0, 2, 1); out = self.relu2(out); return out + identity

class CNNRefiner(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, num_layers: int, kernel_size: int, channels: int):
        super().__init__(); self.input_conv = nn.Conv1d(input_channels, channels, kernel_size=1) ; cnn_layers_list = []
        for i in range(num_layers): dilation = 2**i; cnn_layers_list.append(CNNRefinerBlock(channels, kernel_size, dilation))
        self.cnn_layers = nn.Sequential(*cnn_layers_list); self.output_conv = nn.Conv1d(channels, output_channels, kernel_size=1) 
    def forward(self, x: torch.Tensor) -> torch.Tensor: x = self.input_conv(x); x = self.cnn_layers(x); x = self.output_conv(x); return x

# ## 4. Custom Loss Functions
def masked_huber_loss(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    loss_no_reduction = F.huber_loss(pred, true, reduction='none', delta=delta)
    masked_loss = loss_no_reduction * mask
    num_valid = mask.sum()
    return masked_loss.sum() / num_valid if num_valid > 0 else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

def smoothness_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = pred[:, 1:] - pred[:, :-1]; valid_mask = mask[:, 1:] * mask[:, :-1] ; sq_diff = (diff**2) * valid_mask; num_valid = valid_mask.sum()
    return sq_diff.sum() / num_valid if num_valid > 0 else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

def monotonicity_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = pred[:, 1:] - pred[:, :-1] ; valid_mask = mask[:, 1:] * mask[:, :-1]; positive_diffs = torch.relu(diff) * valid_mask ; num_valid = valid_mask.sum()
    return (positive_diffs**2).sum() / num_valid if num_valid > 0 else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

def diode_residual_loss(
    pred_currents_transformed: torch.Tensor, voltages: torch.Tensor, mask: torch.Tensor,
    diode_params: Optional[torch.Tensor] = None, fixed_diode_params: Optional[Dict[str, float]] = None, 
    q: float = 1.602e-19, k: float = 1.380e-23, T_batch: Optional[torch.Tensor] = None, T_fixed: float = 300.0,
    i_ref_for_transform: float = SIGNED_LOG_I_REF
) -> torch.Tensor:
    if diode_params is None and fixed_diode_params is None: return torch.tensor(0.0, device=pred_currents_transformed.device, dtype=pred_currents_transformed.dtype)
    bs, _ = pred_currents_transformed.shape # Use _ for N_pts if not used elsewhere
    if voltages.ndim == 1: voltages = voltages.unsqueeze(0).expand(bs, -1) 
    T = T_batch.unsqueeze(1) if T_batch is not None else torch.full((bs, 1), T_fixed, device=pred_currents_transformed.device, dtype=pred_currents_transformed.dtype)
    
    pred_sign = torch.sign(pred_currents_transformed)
    pred_abs_inv_log1p = torch.expm1(torch.abs(pred_currents_transformed))
    pred_currents_original_scale = pred_sign * i_ref_for_transform * pred_abs_inv_log1p

    if diode_params is not None: I0, Rs, Rsh, n_ideal = diode_params[:, 0:1], diode_params[:, 1:2], diode_params[:, 2:3], diode_params[:, 3:4]
    elif fixed_diode_params is not None: 
        dev=pred_currents_transformed.device; dtype=pred_currents_transformed.dtype
        I0=torch.full((bs,1),fixed_diode_params['I0'],device=dev,dtype=dtype); Rs=torch.full((bs,1),fixed_diode_params['Rs'],device=dev,dtype=dtype)
        Rsh=torch.full((bs,1),fixed_diode_params['Rsh'],device=dev,dtype=dtype); n_ideal=torch.full((bs,1),fixed_diode_params['n'],device=dev,dtype=dtype)
    else: return torch.tensor(0.0, device=pred_currents_transformed.device, dtype=pred_currents_transformed.dtype)

    I0 = torch.clamp(I0, min=1e-15); Rs = torch.clamp(Rs, min=1e-6) # Allow Rs to be small positive    
    Rsh = torch.clamp(Rsh, min=1.0); n_ideal = torch.clamp(n_ideal, min=0.5, max=3.0) # Wider plausible range
    
    V_junction = voltages + pred_currents_original_scale * Rs 
    exp_arg_coeff = q / (k * T + 1e-9) # Add epsilon to T for stability
    exp_arg = exp_arg_coeff * V_junction / (n_ideal + 1e-9) # Add epsilon to n_ideal
    exp_term_m1 = torch.exp(torch.clamp(exp_arg, max=50.0, min=-50.0)) - 1.0 
    diode_current_comp = I0 * exp_term_m1 
    shunt_current_comp = V_junction / (Rsh + 1e-9)
    residual_values_original_scale = pred_currents_original_scale + diode_current_comp + shunt_current_comp
    masked_residual_sq = (residual_values_original_scale**2) * mask
    num_valid = mask.sum()
    loss = masked_residual_sq.sum() / num_valid if num_valid > 0 else torch.tensor(0.0, device=pred_currents_transformed.device, dtype=pred_currents_transformed.dtype)
    return torch.clamp(loss, max=1e4) # Clamp large diode losses

# ## 5. Training Regimen
def get_optimizer_scheduler(model: PhysicsAwareHyperINR, current_args: ColabArgs, total_steps_per_epoch: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    
    param_groups = []
    main_params = []
    diode_predictor_params_list = []

    model_to_access_params = model._orig_mod if hasattr(model, '_orig_mod') else model

    for name, param in model_to_access_params.named_parameters():
        if not param.requires_grad:
            continue
        if 'diode_param_predictor' in name:
            diode_predictor_params_list.append(param)
        else:
            main_params.append(param)
    
    param_groups.append({'params': main_params, 'lr': current_args.lr}) # Main group

    if diode_predictor_params_list:
        diode_lr = current_args.lr * current_args.diode_predictor_lr_factor
        param_groups.append({
            'params': diode_predictor_params_list,
            'lr': diode_lr
        })
        console.print(f"Diode predictor params assigned LR: {diode_lr:.2e} (factor: {current_args.diode_predictor_lr_factor})")
    else:
        console.print("[yellow]No diode_param_predictor parameters found for separate LR group.[/yellow]")


    optimizer = torch.optim.AdamW(param_groups, lr=current_args.lr, weight_decay=1e-5) # Default LR here is for groups not explicitly set
    
    total_steps = current_args.epochs * total_steps_per_epoch
    if total_steps_per_epoch == 0 : total_steps = 1 # Handle empty loader case
        
    # OneCycleLR max_lr will be taken from the optimizer's param_groups.
    # It correctly handles different LRs for different groups.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[pg.get('lr', current_args.lr) for pg in optimizer.param_groups], # Pass list of max_lr for each group
        total_steps=total_steps if total_steps > 0 else 1,
        pct_start=current_args.onecycle_pct_start, 
        div_factor=current_args.onecycle_div_factor,
        final_div_factor=current_args.onecycle_final_div_factor, 
        anneal_strategy='cos' 
    )
    console.print(f"Optimizer: AdamW (main max_lr: {current_args.lr:.2e}). Scheduler: OneCycleLR over {total_steps} steps.")
    return optimizer, scheduler

def train_one_epoch(
    model: PhysicsAwareHyperINR, loader: DataLoader, optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler, grad_scaler: GradScaler,
    device: torch.device, epoch: int, current_args: ColabArgs, 
    fixed_voltage_grid_tensor: torch.Tensor, progress: Progress
) -> float:
    model.train()
    total_loss_epoch_val = 0.0 # Store the sum of actual backwarded losses
    task_id = progress.add_task(f"[cyan]Epoch {epoch+1}/{current_args.epochs} Training...", total=len(loader))

    # FIX ⚠️ Bug 4: Dynamic voltage sampling
    num_voltage_samples_this_epoch = VOLTAGE_POINTS_TARGET \
        if epoch < current_args.warmup_epochs_full_curve \
        else current_args.voltage_samples_train_post_warmup

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True) 
        
        x_dev_params = batch["x"].to(device, non_blocking=True)
        y_target_transformed_full = batch["y_target"].to(device, non_blocking=True)
        mask_full = batch["mask"].to(device, non_blocking=True)
        voc_true_batch = batch["voc_true"].to(device, non_blocking=True) # [B]
        isc_transformed_batch = batch["isc_transformed"].to(device, non_blocking=True).unsqueeze(-1) # [B,1]
        isc_original_batch = batch["isc_original"].to(device, non_blocking=True) # [B]
        v_knee_batch = batch["v_knee"].to(device, non_blocking=True) # [B]
        batch_size = x_dev_params.shape[0]

        current_v_grid_sampled_for_model: torch.Tensor # Voltages actually fed to model
        current_y_target_transformed_sampled: torch.Tensor
        current_mask_sampled: torch.Tensor
        current_v_values_for_losses: torch.Tensor # The actual voltage values corresponding to sampled points

        if num_voltage_samples_this_epoch < VOLTAGE_POINTS_TARGET:
            # TODO: Implement more sophisticated adaptive/importance sampling here if desired.
            # For now, random sampling if not using full grid.
            indices = torch.sort(torch.randperm(VOLTAGE_POINTS_TARGET, device=device)[:num_voltage_samples_this_epoch]).values
            current_v_grid_sampled_for_model = fixed_voltage_grid_tensor[indices].unsqueeze(0).expand(batch_size, -1, -1)
            current_y_target_transformed_sampled = y_target_transformed_full[:, indices]
            current_mask_sampled = mask_full[:, indices]
            current_v_values_for_losses = fixed_voltage_grid_tensor[indices] # Shape [num_samples_this_epoch]
        else:
            current_v_grid_sampled_for_model = fixed_voltage_grid_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            current_y_target_transformed_sampled = y_target_transformed_full
            current_mask_sampled = mask_full
            current_v_values_for_losses = fixed_voltage_grid_tensor # Shape [VOLTAGE_POINTS_TARGET]
            
        with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            y_pred_transformed, diode_p, siren_out_at_voc = model(
                x_dev_params, current_v_grid_sampled_for_model, 
                voc_true_batch, isc_transformed_batch
            )
            
            # Main reconstruction loss (Huber on transformed scale)
            loss_reconstruction = masked_huber_loss(
                y_pred_transformed, current_y_target_transformed_sampled, 
                current_mask_sampled, delta=current_args.huber_delta
            )
            loss_smooth = smoothness_loss(y_pred_transformed, current_mask_sampled)
            loss_mono = monotonicity_loss(y_pred_transformed, current_mask_sampled)
            
            # Diode loss (conditional on epoch)
            loss_d = torch.tensor(0.0, device=device, dtype=y_pred_transformed.dtype)
            effective_lambda_diode = current_args.lambda_diode
            if epoch < current_args.warmup_epochs_full_curve:
                effective_lambda_diode = 0.0 
            
            if effective_lambda_diode > 0 and diode_p is not None:
                T_col_idx = COLNAMES.index('T') if 'T' in COLNAMES else -1
                batch_T_values = x_dev_params[:, T_col_idx] if T_col_idx != -1 else None
                loss_d = diode_residual_loss(
                    y_pred_transformed, current_v_grid_sampled_for_model.squeeze(-1), 
                    current_mask_sampled, diode_params=diode_p, T_batch=batch_T_values
                )

            # Knee-weighted loss
            loss_knee = torch.tensor(0.0, device=device, dtype=y_pred_transformed.dtype)
            if current_args.lambda_knee_weighted_loss > 0:
                v_coords_for_weight = current_v_grid_sampled_for_model.squeeze(-1) # [B, N_samples]
                # v_knee_batch is [B], needs to be [B,1] for broadcasting with v_coords_for_weight
                gaussian_exponent = -0.5 * torch.square(
                    (v_coords_for_weight - v_knee_batch.unsqueeze(1)) / current_args.knee_weight_sigma_v
                )
                weights_knee = 1.0 + current_args.knee_weight_alpha * torch.exp(gaussian_exponent)
                
                huber_residuals_for_knee = F.huber_loss(
                    y_pred_transformed, current_y_target_transformed_sampled, 
                    reduction='none', delta=current_args.huber_delta
                )
                loss_knee = (weights_knee * huber_residuals_for_knee * current_mask_sampled).sum() / current_mask_sampled.sum() \
                            if current_mask_sampled.sum() > 0 else torch.tensor(0.0, device=device)

            # Curvature loss
            loss_curv = torch.tensor(0.0, device=device, dtype=y_pred_transformed.dtype)
            if current_args.lambda_curvature > 0 and y_pred_transformed.shape[1] >= 3:
                # y_pred_transformed is [B, N_samples]
                diffs1 = y_pred_transformed[:, 1:] - y_pred_transformed[:, :-1] # [B, N_samples-1]
                diffs2 = diffs1[:, 1:] - diffs1[:, :-1] # [B, N_samples-2]
                mask_curv = current_mask_sampled[:, 2:] # Mask for second derivative points
                loss_curv = (diffs2**2 * mask_curv).sum() / mask_curv.sum() \
                            if mask_curv.sum() > 0 else torch.tensor(0.0, device=device)

            # Voc constraint loss (J=0 at Voc, in transformed space)
            loss_voc_cons = torch.tensor(0.0, device=device, dtype=y_pred_transformed.dtype)
            if current_args.lambda_voc_constraint > 0 and siren_out_at_voc is not None:
                # siren_out_at_voc is [B,1], target is 0 for current at Voc
                target_at_voc = torch.zeros_like(siren_out_at_voc) # Target is 0 (transformed current)
                loss_voc_cons = F.mse_loss(siren_out_at_voc, target_at_voc)
            
            # SC Slope constraint loss (on transformed scale)
            loss_sc_slope_cons = torch.tensor(0.0, device=device, dtype=y_pred_transformed.dtype)
            if current_args.lambda_sc_slope_constraint > 0 and y_pred_transformed.shape[1] >= 2:
                # current_v_values_for_losses is [N_samples]
                v_0 = current_v_values_for_losses[0]
                v_1 = current_v_values_for_losses[1]
                v_diff = v_1 - v_0
                if torch.abs(v_diff) > 1e-6: 
                    # y_pred_transformed is [B, N_samples]
                    pred_slope_sc = (y_pred_transformed[:, 1] - y_pred_transformed[:, 0]) / v_diff
                    target_slope_sc = (current_y_target_transformed_sampled[:, 1] - current_y_target_transformed_sampled[:, 0]) / v_diff
                    loss_sc_slope_cons = F.mse_loss(pred_slope_sc, target_slope_sc)

            # Total loss for backward pass
            total_loss_for_backward = (
                current_args.lambda_reconstruction * loss_reconstruction +
                current_args.lambda_smoothness * loss_smooth +
                current_args.lambda_monotonicity * loss_mono +
                effective_lambda_diode * loss_d + 
                current_args.lambda_knee_weighted_loss * loss_knee +
                current_args.lambda_curvature * loss_curv +
                current_args.lambda_voc_constraint * loss_voc_cons +
                current_args.lambda_sc_slope_constraint * loss_sc_slope_cons
            )
        
        grad_scaler.scale(total_loss_for_backward).backward()
        
        if current_args.gradient_clip_val > 0:
            grad_scaler.unscale_(optimizer) 
            params_to_clip = list(model.parameters()) # Simple: clip all model params
            if hasattr(model, '_orig_mod'): params_to_clip = list(model._orig_mod.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, current_args.gradient_clip_val)
            
        grad_scaler.step(optimizer) 
        grad_scaler.update()
        # FIX ⚠️ Bug 2: Removed standalone optimizer.step()
        scheduler.step() 
        
        total_loss_epoch_val += total_loss_for_backward.item()
        progress.update(task_id, advance=1, description=f"[cyan]E{epoch+1} Train Loss: {total_loss_for_backward.item():.4f} LR: {scheduler.get_last_lr()[0]:.2e}") # LR of first group
        
    progress.remove_task(task_id)
    return total_loss_epoch_val / len(loader) if len(loader) > 0 else 0.0

@torch.no_grad()
def evaluate_model(
    model: PhysicsAwareHyperINR, loader: DataLoader, device: torch.device, 
    current_args: ColabArgs, fixed_voltage_grid_tensor: torch.Tensor, 
    epoch: int, # FIX ⚠️ Bug 1: Added epoch for conditional diode loss
    prefix: str = "Val", progress: Optional[Progress] = None
) -> Dict[str, float]:
    model.eval()
    all_y_pred_transformed_batches, all_y_true_transformed_batches, all_masks_batches = [], [], []
    val_loss_sum_from_batches_mixed_scale = 0.0 
    
    task_id = None
    if progress and loader and len(loader) > 0:
        task_id = progress.add_task(f"[magenta]{prefix} Eval E{epoch+1}...", total=len(loader))

    for batch_idx, batch in enumerate(loader):
        x_dev_params = batch["x"].to(device, non_blocking=True)
        y_target_transformed_batch = batch["y_target"].to(device, non_blocking=True)
        mask_batch = batch["mask"].to(device, non_blocking=True)
        voc_true_batch = batch["voc_true"].to(device, non_blocking=True)
        isc_transformed_batch = batch["isc_transformed"].to(device, non_blocking=True).unsqueeze(-1)
        batch_size = x_dev_params.shape[0]
        
        eval_v_coords_full_grid = fixed_voltage_grid_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            # For eval, always use full grid. siren_out_at_voc not strictly needed for val_loss_total.
            y_pred_transformed_batch, diode_p_batch, _ = model(
                x_dev_params, eval_v_coords_full_grid, 
                voc_true_batch, isc_transformed_batch
            )
        
        # --- Calculate components for Val_loss_total (mixed scale as per reasoning) ---
        # 1. Reconstruction MSE on original scale (as per earlier patch direction for eval)
        y_pred_batch_orig_np = signed_log_inverse(y_pred_transformed_batch.cpu().numpy(), I_ref=SIGNED_LOG_I_REF)
        y_true_batch_orig_np = signed_log_inverse(y_target_transformed_batch.cpu().numpy(), I_ref=SIGNED_LOG_I_REF)
        mask_batch_np = mask_batch.cpu().numpy()

        batch_mse_orig_scale = ((y_pred_batch_orig_np - y_true_batch_orig_np)**2 * mask_batch_np).sum() / mask_batch_np.sum() \
                               if mask_batch_np.sum() > 0 else 0.0
        
        # 2. Smoothness & Monotonicity on transformed scale
        loss_smooth_val_batch = smoothness_loss(y_pred_transformed_batch, mask_batch) 
        loss_mono_val_batch = monotonicity_loss(y_pred_transformed_batch, mask_batch)
        
        # 3. Diode loss (epoch-aware)
        loss_d_val_batch = torch.tensor(0.0, device=device, dtype=y_pred_transformed_batch.dtype)
        effective_lambda_diode_val = current_args.lambda_diode
        if epoch < current_args.warmup_epochs_full_curve: # FIX ⚠️ Bug 1
            effective_lambda_diode_val = 0.0
            
        if effective_lambda_diode_val > 0 and diode_p_batch is not None:
            T_col_idx = COLNAMES.index('T') if 'T' in COLNAMES else -1
            batch_T_values = x_dev_params[:, T_col_idx] if T_col_idx != -1 else None
            loss_d_val_batch = diode_residual_loss(
                y_pred_transformed_batch, eval_v_coords_full_grid.squeeze(-1), 
                mask_batch, diode_params=diode_p_batch, T_batch=batch_T_values
            )
        
        # Summing components for this batch's contribution to Val_loss_total
        current_batch_val_loss_mixed = (
            current_args.lambda_reconstruction * batch_mse_orig_scale + # Using main reco weight
            current_args.lambda_smoothness * loss_smooth_val_batch.item() + 
            current_args.lambda_monotonicity * loss_mono_val_batch.item() + 
            effective_lambda_diode_val * loss_d_val_batch.item()
        )
        val_loss_sum_from_batches_mixed_scale += current_batch_val_loss_mixed
        
        # Store transformed outputs for overall original-scale metric calculation later
        all_y_pred_transformed_batches.append(y_pred_transformed_batch.cpu().numpy())
        all_y_true_transformed_batches.append(y_target_transformed_batch.cpu().numpy())
        all_masks_batches.append(mask_batch_np)
        
        if progress and task_id: progress.update(task_id, advance=1)
    
    if progress and task_id: progress.remove_task(task_id)
        
    avg_val_loss_mixed_scale = val_loss_sum_from_batches_mixed_scale / len(loader) if loader and len(loader) > 0 else 0.0
    
    # Initialize metrics dictionary
    metrics_results = {
        f"{prefix}_loss_total": avg_val_loss_mixed_scale, # This is the "mixed scale" val loss
        f"{prefix}_mae_orig": 0.0, f"{prefix}_rmse_orig": 0.0, f"{prefix}_r2_overall_orig": 0.0, 
        f"{prefix}_cosine_similarity_orig_mean": 0.0, f"{prefix}_cosine_similarity_orig_std": 0.0,
        f"{prefix}_per_curve_r2_orig_mean": 0.0, f"{prefix}_per_curve_r2_orig_std": 0.0,
        f"{prefix}_per_curve_r2_orig_min": 0.0, f"{prefix}_per_curve_r2_orig_max": 0.0,
    }

    if not all_y_pred_transformed_batches: 
        console.print(f"[yellow]No data in {prefix} loader to evaluate fully.[/yellow]")
        return metrics_results

    # Concatenate all batch results
    y_pred_transformed_all = np.concatenate(all_y_pred_transformed_batches)
    y_true_transformed_all = np.concatenate(all_y_true_transformed_batches)
    masks_all_np = np.concatenate(all_masks_batches)
    
    if masks_all_np.sum() == 0: 
        console.print(f"[yellow]All masks are zero in {prefix} set. Original scale metrics will be 0 or NaN.[/yellow]")
        return metrics_results

    # --- Calculate MAE, RMSE, R2 on original scale using all samples ---
    y_pred_orig_all_np = signed_log_inverse(y_pred_transformed_all, I_ref=SIGNED_LOG_I_REF)
    y_true_orig_all_np = signed_log_inverse(y_true_transformed_all, I_ref=SIGNED_LOG_I_REF)

    abs_err_orig = np.abs(y_pred_orig_all_np - y_true_orig_all_np) * masks_all_np; mae_orig = abs_err_orig.sum() / masks_all_np.sum() if masks_all_np.sum() > 0 else 0.0
    sq_err_orig = ((y_pred_orig_all_np - y_true_orig_all_np)**2) * masks_all_np; mse_orig = sq_err_orig.sum() / masks_all_np.sum() if masks_all_np.sum() > 0 else 0.0; rmse_orig = np.sqrt(mse_orig)
    
    if masks_all_np.sum() > 0:
        mean_true_overall_orig = (y_true_orig_all_np * masks_all_np).sum() / masks_all_np.sum()
        ss_tot_overall_orig = (((y_true_orig_all_np - mean_true_overall_orig) * masks_all_np)**2).sum()
        r2_overall_orig = 1-(sq_err_orig.sum()/(ss_tot_overall_orig+1e-9)) if ss_tot_overall_orig > 1e-9 else (1.0 if sq_err_orig.sum()<1e-9 else 0.0)
    else: r2_overall_orig = 0.0
    
    cos_sims_orig, pc_r2s_orig = [], []
    for i in range(y_pred_orig_all_np.shape[0]): 
        mask_i_bool = masks_all_np[i].astype(bool); pred_i_orig = y_pred_orig_all_np[i][mask_i_bool]; true_i_orig = y_true_orig_all_np[i][mask_i_bool]
        if len(true_i_orig) < 2: cos_sims_orig.append(0.0); pc_r2s_orig.append(0.0); continue
        dot_orig=np.dot(pred_i_orig,true_i_orig); norm_p_o=np.linalg.norm(pred_i_orig); norm_t_o=np.linalg.norm(true_i_orig)
        cos_sims_orig.append(dot_orig/(norm_p_o*norm_t_o) if (norm_p_o*norm_t_o)>1e-9 else (1.0 if np.allclose(pred_i_orig,true_i_orig) else 0.0))
        mean_t_i_o=true_i_orig.mean(); ss_tot_i_o=((true_i_orig-mean_t_i_o)**2).sum(); ss_res_i_o=((pred_i_orig-true_i_orig)**2).sum()
        pc_r2s_orig.append(1-(ss_res_i_o/(ss_tot_i_o+1e-9)) if ss_tot_i_o > 1e-9 else (1.0 if ss_res_i_o < 1e-9 else 0.0))

    metrics_results.update({
        f"{prefix}_mae_orig": float(mae_orig), f"{prefix}_rmse_orig": float(rmse_orig), f"{prefix}_r2_overall_orig": float(r2_overall_orig),
        f"{prefix}_cosine_similarity_orig_mean": float(np.mean(cos_sims_orig)) if cos_sims_orig else 0.0, f"{prefix}_cosine_similarity_orig_std": float(np.std(cos_sims_orig)) if cos_sims_orig else 0.0,
        f"{prefix}_per_curve_r2_orig_mean": float(np.mean(pc_r2s_orig)) if pc_r2s_orig else 0.0, f"{prefix}_per_curve_r2_orig_std": float(np.std(pc_r2s_orig)) if pc_r2s_orig else 0.0,
        f"{prefix}_per_curve_r2_orig_min": float(np.min(pc_r2s_orig)) if pc_r2s_orig else 0.0, f"{prefix}_per_curve_r2_orig_max": float(np.max(pc_r2s_orig)) if pc_r2s_orig else 0.0,
    })
    return metrics_results

# ## 6. Plotting and Artifacts Saving
def plot_training_curves(train_losses: List[float], val_losses: List[float], val_metric_orig_scale: List[float], metric_name: str, save_path: str) -> None:
    plt.style.use('seaborn-v0_8-whitegrid'); fig, ax1 = plt.subplots(figsize=(12, 7)) ; epochs_range = range(1, len(train_losses) + 1)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color='tab:red'); ax1.plot(epochs_range, train_losses, color='tab:red', linestyle='-', marker='o', markersize=3, label='Training Loss (Components Sum)')
    if val_losses: ax1.plot(epochs_range[:len(val_losses)], val_losses, color='tab:orange', linestyle='--',  marker='x', markersize=3, label='Validation Loss (Mixed Scale Total)')
    ax1.tick_params(axis='y', labelcolor='tab:red'); ax1.legend(loc='upper left'); ax1.grid(True, linestyle=':', alpha=0.7)
    if val_metric_orig_scale: 
        ax2 = ax1.twinx() ; ax2.set_ylabel(f'{metric_name} (Validation, Original Scale)', color='tab:blue'); ax2.plot(epochs_range[:len(val_metric_orig_scale)], val_metric_orig_scale, color='tab:blue', linestyle=':', marker='s', markersize=3, label=f'Validation {metric_name}'); ax2.tick_params(axis='y', labelcolor='tab:blue'); ax2.legend(loc='upper right')
        valid_metric_values = [x for x in val_metric_orig_scale if x is not None and not np.isnan(x)]
        if "r2" in metric_name.lower() or "similarity" in metric_name.lower(): 
            min_y_metric = 0.0
            if valid_metric_values: min_y_metric = min(0.0, np.min(valid_metric_values) - 0.05) # R2 can be negative
            ax2.set_ylim(min_y_metric if min_y_metric < 0.95 else 0.8, 1.01)
    fig.tight_layout() ; plt.title('Training Progress', fontsize=16); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig); console.print(f"[green]Training curves plot saved to {save_path}[/green]")

def plot_predictions(
    model: PhysicsAwareHyperINR, loader: DataLoader, device: torch.device, 
    current_args: ColabArgs, fixed_voltage_grid_tensor: torch.Tensor, 
    dataset_X_orig_indices_subset: np.ndarray, 
    num_samples_to_plot: int, save_path: str,
    apply_mc_dropout: bool = False # MC dropout not fully implemented here
) -> None:
    model.eval()
    if not loader or not loader.dataset or len(loader.dataset) == 0: console.print(f"[yellow]Plot predictions: Loader empty. Skipping.[/yellow]"); return
    try: sample_batch = next(iter(loader)) 
    except StopIteration: console.print(f"[yellow]Plot predictions: Loader empty (StopIteration). Skipping.[/yellow]"); return

    actual_num_to_plot = min(num_samples_to_plot, sample_batch["x"].shape[0])
    if actual_num_to_plot == 0: console.print(f"[yellow]Plot predictions: No samples. Skipping.[/yellow]"); return

    x_dev_params = sample_batch["x"][:actual_num_to_plot].to(device); y_target_t_batch = sample_batch["y_target"][:actual_num_to_plot].to(device)
    mask_batch = sample_batch["mask"][:actual_num_to_plot].to(device); voc_true_batch = sample_batch["voc_true"][:actual_num_to_plot].to(device)
    isc_t_batch = sample_batch["isc_transformed"][:actual_num_to_plot].to(device).unsqueeze(-1)
    
    eval_v_coords_full = fixed_voltage_grid_tensor.unsqueeze(0).expand(actual_num_to_plot, -1, -1)
    with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
        y_pred_t_batch, _, _ = model(x_dev_params, eval_v_coords_full, voc_true_batch, isc_t_batch)
    
    y_pred_orig_list, y_true_orig_list = [], []
    for i in range(actual_num_to_plot):
        pred_orig = signed_log_inverse(y_pred_t_batch[i].cpu().numpy().reshape(-1,1), SIGNED_LOG_I_REF).flatten()
        true_orig = signed_log_inverse(y_target_t_batch[i].cpu().numpy().reshape(-1,1), SIGNED_LOG_I_REF).flatten()
        y_pred_orig_list.append(pred_orig); y_true_orig_list.append(true_orig)

    if not y_pred_orig_list: console.print(f"[yellow]Plot predictions: No data to plot. Skipping.[/yellow]"); return
    
    plt.style.use('seaborn-v0_8-whitegrid'); num_plots = len(y_pred_orig_list); fig_rows = min(num_plots,2 if num_plots>1 else 1); fig_cols = (num_plots+fig_rows-1)//fig_rows if num_plots>0 else 1
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(7.5*fig_cols, 6.0*fig_rows), squeeze=False); axes = axes.flatten() 
    voltages_np = fixed_voltage_grid_tensor.cpu().numpy().flatten()

    for i in range(num_plots):
        ax=axes[i]; mask_i_bool = mask_batch[i].cpu().numpy().astype(bool)
        ax.plot(voltages_np[mask_i_bool], y_true_orig_list[i][mask_i_bool], 'bo-', ms=4, lw=1.5, label='True (Orig Scale)')
        ax.plot(voltages_np[mask_i_bool], y_pred_orig_list[i][mask_i_bool], 'rx--', ms=4, lw=1.5, label='Pred (Orig Scale)')
        # MC Dropout std plotting would go here if y_pred_std_orig_list was properly calculated
        
        pred_o_pts=y_pred_orig_list[i][mask_i_bool]; true_o_pts=y_true_orig_list[i][mask_i_bool]; r2_o,cs_o="N/A","N/A"
        if len(true_o_pts)>=2: 
            mean_t_o=true_o_pts.mean();ss_tot_o=((true_o_pts-mean_t_o)**2).sum();ss_res_o=((pred_o_pts-true_o_pts)**2).sum()
            r2_o_val = (1-(ss_res_o/(ss_tot_o+1e-9)) if ss_tot_o > 1e-9 else (1.0 if ss_res_o < 1e-9 else 0.0))
            r2_o=f"{r2_o_val:.3f}"
            dot_o=np.dot(pred_o_pts,true_o_pts);n_p_o=np.linalg.norm(pred_o_pts);n_t_o=np.linalg.norm(true_o_pts)
            cs_o_val = (dot_o/(n_p_o*n_t_o) if (n_p_o*n_t_o)>1e-9 else (1.0 if np.allclose(pred_o_pts,true_o_pts) else 0.0))
            cs_o=f"{cs_o_val:.3f}"
        orig_idx_val = dataset_X_orig_indices_subset[i] if dataset_X_orig_indices_subset is not None and i < len(dataset_X_orig_indices_subset) else i
        ax.set_title(f'Sample (Orig Idx {orig_idx_val})\nOrig R²:{r2_o}|Orig CosSim:{cs_o}',fontsize=10); ax.legend(fontsize=10); ax.grid(True,ls=':',alpha=0.7); ax.set_xlabel('V (V)',fontsize=10); ax.set_ylabel('I (mA/cm²)',fontsize=10)
    for j_ax in range(num_plots,len(axes)): fig.delaxes(axes[j_ax]) # Use j_ax to avoid conflict
    plt.tight_layout(pad=2.0); plt.savefig(save_path,dpi=300,bbox_inches='tight'); plt.close(fig); console.print(f"[green]Predictions plot saved: {save_path}[/green]")

def save_artifacts(current_args: ColabArgs, model: nn.Module, preprocessor: Optional[ColumnTransformer], metrics: Dict) -> None:
    os.makedirs(current_args.output_dir,exist_ok=True); model_to_save=model._orig_mod if hasattr(model,'_orig_mod') else model
    if hasattr(model_to_save,'hyper_net'): h_path=os.path.join(current_args.output_dir,current_args.model_path_hyper_inr); torch.save(model_to_save.hyper_net.state_dict(),h_path); console.print(f"[green]HyperNet saved: {h_path}[/green]")
    if current_args.use_cnn_refiner and hasattr(model_to_save,'cnn_refiner') and model_to_save.cnn_refiner is not None: c_path=os.path.join(current_args.output_dir,current_args.model_path_cnn_refiner); torch.save(model_to_save.cnn_refiner.state_dict(),c_path); console.print(f"[green]CNN Refiner saved: {c_path}[/green]")
    if hasattr(model_to_save,'diode_param_predictor') and model_to_save.diode_param_predictor is not None: d_path=os.path.join(current_args.output_dir,"diode_predictor.pt"); torch.save(model_to_save.diode_param_predictor.state_dict(),d_path); console.print(f"[green]Diode Predictor saved: {d_path}[/green]")
    if preprocessor: p_path=os.path.join(current_args.output_dir,current_args.preprocessor_path); joblib.dump(preprocessor,p_path); console.print(f"[green]Preprocessor saved: {p_path}[/green]")
    console.print("[yellow]Scalers not saved (signed-log transform global).[/yellow]")
    m_path=os.path.join(current_args.output_dir,"metrics.json"); serializable_metrics={k:(float(v) if isinstance(v,(np.float32,np.float64,float,np.ndarray, torch.Tensor)) else v) for k,v in metrics.items()}; 
    with open(m_path,'w') as f: json.dump(serializable_metrics,f,indent=4); console.print(f"[green]Metrics saved: {m_path}[/green]")

# ## 7. Active Data Refinement & Calibration (Stubs)
def active_data_refinement_callback(epoch: int, model: nn.Module, val_loader: Optional[DataLoader], current_args: ColabArgs) -> None: 
    if not current_args.active_learning or (epoch+1)%5 !=0: return ; console.print(f"[blue]Epoch {epoch+1}: Active data refinement (stub)...[/blue]"); console.print("[yellow]Active learning callback is a stub.[/yellow]")
def calibrate_model_analytically(model: PhysicsAwareHyperINR, experimental_data: Dict, current_args: ColabArgs) -> None:
    console.print("[blue]Analytic calibration (stub)...[/blue]"); console.print("[yellow]Analytic calibration is a stub.[/yellow]")

# ## 8. Main Execution Block
def main_colab(current_args: ColabArgs): 
    set_seed(current_args.seed)
    effective_device = torch.device("cuda" if torch.cuda.is_available() and not current_args.no_gpu else "cpu")
    console.print(f"Using device: {effective_device}")
    pt_major, pt_minor = int(torch.__version__.split('.')[0]), int(torch.__version__.split('.')[1])
    if current_args.enable_torch_compile and pt_major < 2: current_args.enable_torch_compile=False; console.print(f"[yellow]torch.compile disabled (PyTorch {torch.__version__} < 2.0).[/yellow]")
    os.makedirs(current_args.output_dir, exist_ok=True)
    
    profiler_context = None
    if current_args.profile_run : # Profiler setup (condensed for brevity)
        from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
        prof_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if effective_device.type == 'cuda' else [ProfilerActivity.CPU]
        prof_path = os.path.join(current_args.output_dir, 'profiler_log' if effective_device.type == 'cuda' else 'profiler_log_cpu')
        profiler_context = profile(activities=prof_activities, schedule=schedule(wait=1,warmup=1,active=3,repeat=1), on_trace_ready=tensorboard_trace_handler(prof_path), record_shapes=True, profile_memory=(pt_major>=1 and pt_minor>=9), with_stack=True)
        console.print(f"[cyan]Profiler enabled. Traces to: {prof_path}[/cyan]")

    console.print("[cyan]Loading and preprocessing data (signed-log, V_knee estimation)...[/cyan]")
    input_path = os.path.join(current_args.data_root, current_args.input_file)
    output_path = os.path.join(current_args.data_root, current_args.output_file)
    if not os.path.exists(input_path) or not os.path.exists(output_path): console.print(f"[bold red]Data files not found. Check paths.[/bold red]"); return

    try:
        X_df_all = load_input_features(input_path, COLNAMES)
        y_transformed_all, masks_all, voc_values_all, isc_original_all, v_knee_all = load_output_curves(
            output_path, transformed_pad_value=TRANSFORMED_PAD_VALUE, voltage_grid_fixed_np=VOLTAGE_GRID_FIXED 
        )
    except Exception as e: console.print(f"[bold red]Error loading/processing data: {e}. Exiting.[/bold red]"); return

    min_len = min(X_df_all.shape[0], y_transformed_all.shape[0])
    if X_df_all.shape[0] != y_transformed_all.shape[0]: console.print(f"[yellow]Input/Output sample count mismatch. Aligning to {min_len} samples.[/yellow]")
    if min_len == 0: console.print("[bold red]No common samples after alignment. Exiting.[/bold red]"); return
    X_df_all=X_df_all.iloc[:min_len]; y_transformed_all=y_transformed_all[:min_len]; masks_all=masks_all[:min_len]
    voc_values_all=voc_values_all[:min_len]; isc_original_all=isc_original_all[:min_len]; v_knee_all=v_knee_all[:min_len]
    
    isc_transformed_all = signed_log_transform(isc_original_all, I_ref=SIGNED_LOG_I_REF)
    
    numeric_features_indices = list(range(N_FEATURES)) 
    preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_features_indices)], remainder='passthrough')
    original_indices_all = np.arange(X_df_all.shape[0])

    X_df_s, y_t_s, m_s, v_s, isc_t_s, isc_o_s, vk_s, orig_idx_s = \
        X_df_all, y_transformed_all, masks_all, voc_values_all, isc_transformed_all, isc_original_all, v_knee_all, original_indices_all

    if current_args.quick:
        num_total = X_df_s.shape[0]; num_q = min(num_total, current_args.batch_size*5, 1000 if num_total>1000 else num_total)
        if num_q==0 and num_total>0: num_q=num_total
        if num_q==0: console.print("[bold red]No data for quick mode. Exiting.[/bold red]"); return
        console.print(f"[yellow]QUICK MODE: Subsetting to {num_q} samples.[/yellow]")
        perm = np.random.permutation(num_total)[:num_q]
        X_df_s=X_df_s.iloc[perm].copy(); y_t_s,m_s,v_s,isc_t_s,isc_o_s,vk_s = y_t_s[perm],m_s[perm],v_s[perm],isc_t_s[perm],isc_o_s[perm],vk_s[perm]
        orig_idx_s = orig_idx_s[perm]

    if X_df_s.empty: console.print("[bold red]Dataset empty. Exiting.[/bold red]"); return
    
    current_data_indices = np.arange(X_df_s.shape[0]) 
    if len(current_data_indices) < 2: console.print("[bold red]Not enough samples for split. Exiting.[/bold red]"); return
    tv_ratio = 0.2; tv_ratio = 1/len(current_data_indices) if len(current_data_indices)*tv_ratio<1 and len(current_data_indices)>1 else tv_ratio
    train_idx_loc, temp_idx_loc = (current_data_indices, np.array([],dtype=int)) if (len(current_data_indices)==1 or tv_ratio==0) else \
        train_test_split(current_data_indices, test_size=tv_ratio, random_state=current_args.seed, shuffle=True)
    val_idx_loc, test_idx_loc = np.array([],dtype=int), np.array([],dtype=int)
    if len(temp_idx_loc)>0:
        sub_test_ratio=0.5; sub_test_ratio = 1/len(temp_idx_loc) if len(temp_idx_loc)*sub_test_ratio<1 and len(temp_idx_loc)>1 else sub_test_ratio
        if len(temp_idx_loc)==1 or sub_test_ratio==0: val_idx_loc = temp_idx_loc
        else: val_idx_loc, test_idx_loc = train_test_split(temp_idx_loc, test_size=sub_test_ratio, random_state=current_args.seed, shuffle=True)
    if len(train_idx_loc)==0: console.print("[bold red]No training samples. Exiting.[/bold red]"); return

    preprocessor.fit(X_df_s.iloc[train_idx_loc])
    X_train_p = preprocessor.transform(X_df_s.iloc[train_idx_loc])
    X_val_p = preprocessor.transform(X_df_s.iloc[val_idx_loc]) if len(val_idx_loc)>0 else np.array([]).reshape(0,N_FEATURES)
    X_test_p = preprocessor.transform(X_df_s.iloc[test_idx_loc]) if len(test_idx_loc)>0 else np.array([]).reshape(0,N_FEATURES)

    def get_split_data(indices_loc):
        if len(indices_loc) == 0: 
            empty_y_shape, empty_meta_shape = (0,VOLTAGE_POINTS_TARGET),(0,)
            return tuple([np.array([]).reshape(empty_y_shape)]*2 + [np.array([]).reshape(empty_meta_shape)]*5)
        return y_t_s[indices_loc], m_s[indices_loc], v_s[indices_loc], isc_t_s[indices_loc], isc_o_s[indices_loc], vk_s[indices_loc]

    y_train_t, m_train, v_train, isc_train_t, isc_train_o, vk_train = get_split_data(train_idx_loc)
    y_val_t, m_val, v_val, isc_val_t, isc_val_o, vk_val = get_split_data(val_idx_loc)
    y_test_t, m_test, v_test, isc_test_t, isc_test_o, vk_test = get_split_data(test_idx_loc)
    
    val_orig_global_indices = orig_idx_s[val_idx_loc] if len(val_idx_loc) > 0 else np.array([],dtype=int)
    test_orig_global_indices = orig_idx_s[test_idx_loc] if len(test_idx_loc) > 0 else np.array([],dtype=int)
    
    console.print(f"Data split: Train ({len(X_train_p)}), Val ({len(X_val_p)}), Test ({len(X_test_p)})")
    train_loader, val_loader, test_loader = create_dataloaders(
        current_args, X_train_p, y_train_t, m_train, v_train, isc_train_t, isc_train_o, vk_train,
        X_val_p, y_val_t, m_val, v_val, isc_val_t, isc_val_o, vk_val, 
        X_test_p, y_test_t, m_test, v_test, isc_test_t, isc_test_o, vk_test
    )
    
    console.print("[cyan]Initializing model...[/cyan]")
    cnn_conf = {"depth":current_args.cnn_depth, "kernel_size":current_args.cnn_kernel_size, "channels":current_args.cnn_channels} if current_args.use_cnn_refiner else None
    model = PhysicsAwareHyperINR(
        device_param_dim=N_FEATURES, siren_hidden_features=current_args.siren_hidden_dim, 
        siren_hidden_layers=current_args.siren_num_layers, siren_omega_0=current_args.siren_omega_0,        
        hypernet_layers=current_args.hypernet_layers, hypernet_rank=current_args.hypernet_ranks,
        use_cnn_refiner=current_args.use_cnn_refiner, cnn_config=cnn_conf,
        predict_diode_params=(current_args.lambda_diode > 0),
        lambda_voc_constraint_active=(current_args.lambda_voc_constraint > 0) # Pass bool flag
    ).to(effective_device)

    if current_args.enable_torch_compile:
        try: model = torch.compile(model,mode="reduce-overhead"); console.print("[green]Model JIT compiled.[/green]")
        except Exception as e: console.print(f"[yellow]torch.compile() failed: {e}. Proceeding without.[/yellow]")
    
    fixed_voltage_grid_tensor = torch.from_numpy(VOLTAGE_GRID_FIXED).float().unsqueeze(-1).to(effective_device)

    if current_args.predict_only:
        console.print("[yellow]Prediction only mode. Loading model...[/yellow]")
        model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
        # Load HyperNet
        h_load_path = os.path.join(current_args.output_dir, current_args.model_path_hyper_inr)
        if os.path.exists(h_load_path) and hasattr(model_to_load, 'hyper_net'): model_to_load.hyper_net.load_state_dict(torch.load(h_load_path, map_location=effective_device)); console.print(f"Loaded HyperNet from {h_load_path}")
        else: console.print(f"[bold red]HyperNet model not found: {h_load_path}. Cannot predict.[/bold red]"); return
        # Load CNN Refiner if used
        if current_args.use_cnn_refiner and hasattr(model_to_load, 'cnn_refiner') and model_to_load.cnn_refiner:
            c_load_path = os.path.join(current_args.output_dir, current_args.model_path_cnn_refiner); 
            if os.path.exists(c_load_path): model_to_load.cnn_refiner.load_state_dict(torch.load(c_load_path, map_location=effective_device)); console.print(f"Loaded CNN Refiner from {c_load_path}")
            else: console.print(f"[yellow]CNN Refiner model not found: {c_load_path}.[/yellow]")
        # Load Diode Predictor if used
        if hasattr(model_to_load, 'diode_param_predictor') and model_to_load.diode_param_predictor:
            d_load_path = os.path.join(current_args.output_dir, "diode_predictor.pt")
            if os.path.exists(d_load_path): model_to_load.diode_param_predictor.load_state_dict(torch.load(d_load_path, map_location=effective_device)); console.print(f"Loaded Diode Predictor from {d_load_path}")
            else: console.print(f"[yellow]Diode Predictor model not found: {d_load_path}.[/yellow]")
        
        console.print("Evaluating loaded model on test set...")
        if test_loader and test_loader.dataset and len(test_loader.dataset) > 0:
            test_metrics = evaluate_model(model, test_loader, effective_device, current_args, fixed_voltage_grid_tensor, epoch=current_args.epochs, prefix="Test") # Use large epoch for eval
            table = Table(title="Test Set Eval Metrics (Loaded Model)"); table.add_column("Metric",style="cyan"); table.add_column("Value",style="magenta")
            for k,v in test_metrics.items(): table.add_row(k,f"{float(v):.4f}")
            console.print(table)
            plot_predictions(model, test_loader, effective_device, current_args, fixed_voltage_grid_tensor, 
                             test_orig_global_indices, min(4,len(test_loader.dataset or [])), 
                             os.path.join(current_args.output_dir, "test_predictions_loaded_model.png"))
        else: console.print("[yellow]Test loader empty. Skipping eval & plot.[/yellow]")
        return

    console.print("[cyan]Starting training...[/cyan]")
    if not train_loader or not train_loader.dataset or len(train_loader.dataset)==0: console.print("[bold red]Training loader empty. Exiting.[/bold red]"); return

    grad_scaler = GradScaler(enabled=(effective_device.type == 'cuda'))
    optimizer, scheduler = get_optimizer_scheduler(model, current_args, len(train_loader))
    
    best_val_metric_for_early_stop = -float('inf') 
    epochs_no_improve = 0
    train_loss_history, val_loss_history, val_r2_history = [], [], []

    if profiler_context: profiler_context.start()

    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn(), console=console, transient=False) as prog_bar_manager:
        overall_task_id = prog_bar_manager.add_task("Overall Training Progress", total=current_args.epochs)
        for epoch in range(current_args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, grad_scaler, 
                                         effective_device, epoch, current_args, fixed_voltage_grid_tensor, prog_bar_manager)
            train_loss_history.append(train_loss)
            current_lr_group0 = optimizer.param_groups[0]['lr'] # LR for main group
            log_message = f"E{epoch+1}/{current_args.epochs}: Train Loss {train_loss:.4f}, LR {current_lr_group0:.2e}"

            if val_loader and val_loader.dataset and len(val_loader.dataset) > 0:
                val_metrics = evaluate_model(model, val_loader, effective_device, current_args, fixed_voltage_grid_tensor, epoch, prefix="Val", progress=prog_bar_manager)
                val_loss_history.append(val_metrics["Val_loss_total"])
                
                current_val_r2_orig = val_metrics.get("Val_r2_overall_orig", -float('inf'))
                val_r2_history.append(current_val_r2_orig)
                log_message += f", Val Loss (mixed) {val_metrics['Val_loss_total']:.4f}, Val R2 (orig) {current_val_r2_orig:.4f}"
                
                if current_val_r2_orig > best_val_metric_for_early_stop + current_args.min_delta:
                    best_val_metric_for_early_stop = current_val_r2_orig; epochs_no_improve = 0
                    model_to_save_best = model._orig_mod if hasattr(model,'_orig_mod') else model
                    if hasattr(model_to_save_best,'hyper_net'): torch.save(model_to_save_best.hyper_net.state_dict(), os.path.join(current_args.output_dir,"best_hyper_inr.pt"))
                    if hasattr(model_to_save_best,'diode_param_predictor') and model_to_save_best.diode_param_predictor: torch.save(model_to_save_best.diode_param_predictor.state_dict(), os.path.join(current_args.output_dir,"best_diode_predictor.pt"))
                    if current_args.use_cnn_refiner and hasattr(model_to_save_best,'cnn_refiner') and model_to_save_best.cnn_refiner: torch.save(model_to_save_best.cnn_refiner.state_dict(), os.path.join(current_args.output_dir,"best_cnn_refiner.pt"))
                    log_message += f" ([green]New best R2: {best_val_metric_for_early_stop:.4f}. Saved.[/green])"
                else:
                    epochs_no_improve += 1; log_message += f" (No R2 improvement for {epochs_no_improve} epochs)"
            else: 
                if (epoch+1)%current_args.checkpoint_freq==0 or (epoch+1)==current_args.epochs:
                    m_save_no_val=model._orig_mod if hasattr(model,'_orig_mod') else model
                    if hasattr(m_save_no_val,'hyper_net'): torch.save(m_save_no_val.hyper_net.state_dict(),os.path.join(current_args.output_dir,f"last_e{epoch+1}_hyper_inr.pt"))
                    log_message += f" ([blue]Saved model at epoch {epoch+1}[/blue])"
            console.print(log_message)
            
            if (epoch+1)%current_args.checkpoint_freq==0:
                ckpt_dir=os.path.join(current_args.output_dir,f"checkpoint_epoch_{epoch+1}"); os.makedirs(ckpt_dir,exist_ok=True)
                m_save_ckpt=model._orig_mod if hasattr(model,'_orig_mod') else model
                if hasattr(m_save_ckpt,'hyper_net'): torch.save(m_save_ckpt.hyper_net.state_dict(),os.path.join(ckpt_dir,"hyper_inr.pt"))
                # Save other components for checkpoint if needed
                console.print(f"[blue]Checkpoint saved at epoch {epoch+1}[/blue]")
            
            if val_loader and epochs_no_improve >= current_args.patience : console.print(f"[yellow]Early stopping after {current_args.patience} epochs of no R2 improvement.[/yellow]"); break
            active_data_refinement_callback(epoch, model, val_loader, current_args)
            prog_bar_manager.update(overall_task_id, advance=1)
            if profiler_context: profiler_context.step()

    if profiler_context: profiler_context.stop()

    console.print("[cyan]Training finished. Loading best/last model for final eval...[/cyan]")
    loaded_model_type_final = "current_state"; model_for_final_eval = model._orig_mod if hasattr(model,'_orig_mod') else model
    best_h_path_final = os.path.join(current_args.output_dir,"best_hyper_inr.pt")
    if os.path.exists(best_h_path_final) and hasattr(model_for_final_eval,'hyper_net'):
        model_for_final_eval.hyper_net.load_state_dict(torch.load(best_h_path_final,map_location=effective_device))
        loaded_model_type_final="best_val"; console.print(f"Loaded best HyperNet from {best_h_path_final}")
        # Load other "best" components
        if hasattr(model_for_final_eval,'diode_param_predictor') and os.path.exists(os.path.join(current_args.output_dir,"best_diode_predictor.pt")):
            model_for_final_eval.diode_param_predictor.load_state_dict(torch.load(os.path.join(current_args.output_dir,"best_diode_predictor.pt"), map_location=effective_device))
        if current_args.use_cnn_refiner and hasattr(model_for_final_eval,'cnn_refiner') and os.path.exists(os.path.join(current_args.output_dir,"best_cnn_refiner.pt")):
             model_for_final_eval.cnn_refiner.load_state_dict(torch.load(os.path.join(current_args.output_dir,"best_cnn_refiner.pt"), map_location=effective_device))
    else: console.print(f"[yellow]No 'best_hyper_inr.pt'. Using model from end of training for final eval.[/yellow]")

    final_metrics_to_save = {}
    if test_loader and test_loader.dataset and len(test_loader.dataset) > 0:
        console.print(f"Evaluating on Test Set with '{loaded_model_type_final}' model...")
        eval_model_for_test = model if current_args.enable_torch_compile and (pt_major>=2) else model_for_final_eval # Use compiled if available
        final_test_metrics = evaluate_model(eval_model_for_test, test_loader, effective_device, current_args, fixed_voltage_grid_tensor, epoch=current_args.epochs, prefix="TestFinal")
        final_metrics_to_save = final_test_metrics
        final_tab = Table(title=f"Final Test Set Eval Metrics ({loaded_model_type_final} model)")
        final_tab.add_column("Metric",style="cyan"); final_tab.add_column("Value",style="magenta")
        for k,v_met in final_test_metrics.items(): final_tab.add_row(k,f"{float(v_met):.4f}")
        console.print(final_tab)
        plot_predictions(eval_model_for_test, test_loader, effective_device, current_args, fixed_voltage_grid_tensor,
                         test_orig_global_indices, min(4,len(test_loader.dataset or [])), 
                         os.path.join(current_args.output_dir, "test_predictions_final.png"))
    else: console.print("[yellow]Test loader empty. Skipping final test evaluation.[/yellow]")
    
    save_artifacts(current_args, model, preprocessor, final_metrics_to_save) 
    if train_loss_history : plot_training_curves(train_loss_history, val_loss_history, val_r2_history, "Val_R2_Overall_Orig", os.path.join(current_args.output_dir, "training_curves.png"))
    console.print("[bold green]Colab run completed successfully![/bold green]")

if __name__ == "__main__": 
    main_colab(args)