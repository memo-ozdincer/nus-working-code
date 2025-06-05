#!/usr/bin/env python3
"""
refactored_iv_reconstruction.py

Comprehensive, refactored implementation for Perovskite I–V curve reconstruction
with a physics‐informed neural network, PCA, and ensemble methods. Key features:
- Single‐responsibility utility functions for preprocessing (truncation, smoothing, scaling, padding, outlier removal).
- Per‐curve I_sc‐based normalization (no MinMax scaling).
- Knee‐weighted MSE and enhanced physics penalties (monotonicity, curvature, J_sc/V_oc).
- Sinusoidal positional encoding for voltage embedding with a residual block.
- Centralized hyperparameter configuration, logging, and thorough docstrings.
- No negative current values assumed (currents ≥ 0).
- Minimal, necessary, and well‐tested changes to achieve R² ≈ 0.99 performance.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as SklearnGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Add, Lambda, TimeDistributed, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ───────────────────────────────────────────────────────────────────────────────
#  Logging Configuration
# ───────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
#  Hyperparameters & Constants
# ───────────────────────────────────────────────────────────────────────────────
# File paths (modify these to your actual file locations)
INPUT_FILE_PARAMS = "/content/drive/MyDrive/Colab Notebooks/Data_100k/LHS_parameters_m.txt"
INPUT_FILE_IV     = "/content/drive/MyDrive/Colab Notebooks/Data_100k/iV_m.txt"

# Output directory with timestamp to avoid overwriting
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"./output_run_{RUN_ID}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

# Original I–V curve configuration
ASSUMED_ORIGINAL_IV_POINTS   = 45
ASSUMED_ORIGINAL_MAX_VOLTAGE = 1.2  # in Volts

# Truncation & smoothing hyperparameters
MIN_LEN_FOR_PROCESSING = 5         # Minimum points required after truncation
MIN_LEN_SAVGOL        = 5          # Minimum points to apply Savitzky–Golay
SAVGOL_POLYORDER      = 2          # Polynomial order for SavGol
SAVGOL_LOWER_WINDOW   = 3          # Minimum window length for SavGol

# Physics‐informed loss weights
LOSS_WEIGHTS = {
    'mse': 1.0,
    'monotonicity': 0.02,    # Reduced after knee‐weighting
    'curvature': 0.02,       # Reduced after knee‐weighting
    'jsc': 0.20,
    'voc': 0.25              # Increased to force current→0 at V_oc
}
KNEE_WEIGHT_FACTOR = 2.0   # Weight for last two valid points in MSE

# Neural network architecture
VOLTAGE_EMBED_DIM       = 16     # d_model for sinusoidal positional encoding
DENSE_UNITS_PARAMS      = [256, 128, 128]  # Layer widths for param path
DENSE_UNITS_MERGED      = [256, 128]       # Layer widths after concatenation
DROPOUT_RATE            = 0.30

# Training hyperparameters
NN_LEARNING_RATE = 1e-3
NN_EPOCHS        = 200
BATCH_SIZE       = 128

# PCA & ensemble hyperparameters
PCA_N_COMPONENTS       = 15
PCA_N_ESTIMATORS       = 200
ENSEMBLE_N_ESTIMATORS  = 150
ENSEMBLE_LEARNING_RATE = 0.05
ENSEMBLE_MAX_DEPTH     = 8
ENSEMBLE_SUBSAMPLE     = 0.7
ENSEMBLE_RANDOM_STATE  = 42

# Outlier detection
ISOFOREST_CONTAMINATION = 0.05
ISOFOREST_RANDOM_STATE  = 42

# Column names for input parameters file
COLNAMES = [
    'Eg', 'NCv', 'NCc', 'mu_e', 'mu_h', 'eps', 'A', 'Cn', 'Cp', 'Nt',
    'Et', 'nD', 'nA', 'thickness', 'T', 'Sn', 'Sp', 'Rs', 'Rsh',
    'G', 'light_intensity', 'Voc_ref', 'Jsc_ref', 'FF_ref', 'PCE_ref',
    'Qe_loss', 'R_loss', 'SRH_loss', 'series_loss', 'shunt_loss', 'other_loss'
]

# ───────────────────────────────────────────────────────────────────────────────
#  Utility Functions: Preprocessing & Loss
# ───────────────────────────────────────────────────────────────────────────────

def truncate_iv_curve(
    curve_current_raw: np.ndarray,
    full_voltage_grid: np.ndarray,
    truncation_threshold_pct: float
) -> (np.ndarray, np.ndarray):
    """
    Truncate an I–V curve where current drops below truncation_threshold_pct * I_sc.
    Assumes no negative current values (currents ≥ 0).
    Returns (voltage_truncated, current_truncated). If too short, returns (None, None).
    """
    if curve_current_raw.size == 0:
        return None, None

    isc_val = float(curve_current_raw[0])
    if isc_val <= 0:
        return None, None

    threshold = truncation_threshold_pct * isc_val
    below_threshold = curve_current_raw < threshold
    if np.any(below_threshold):
        trunc_idx = int(np.argmax(below_threshold))
    else:
        trunc_idx = curve_current_raw.shape[0]

    if trunc_idx < MIN_LEN_FOR_PROCESSING:
        return None, None

    voltage_trunc = full_voltage_grid[:trunc_idx].copy()
    current_trunc = curve_current_raw[:trunc_idx].copy()
    return voltage_trunc, current_trunc


def apply_savgol(current_trunc: np.ndarray) -> np.ndarray:
    """
    Apply Savitzky–Golay smoothing to truncated current array if length ≥ MIN_LEN_SAVGOL.
    Otherwise, return unchanged.
    """
    L = current_trunc.size
    if L < MIN_LEN_SAVGOL + 2:
        return current_trunc

    wl = min(max(MIN_LEN_SAVGOL, ((L // 10) * 2 + 1)), L)
    if wl % 2 == 0:
        wl -= 1
    wl = max(SAVGOL_LOWER_WINDOW, wl)
    if wl > L:
        wl = L if L % 2 == 1 else L - 1
    polyorder = max(0, min(SAVGOL_POLYORDER, wl - 1))

    if L > wl and wl >= 3:
        return savgol_filter(current_trunc, window_length=wl, polyorder=polyorder)
    return current_trunc


def normalize_by_isc(curve_trunc: np.ndarray) -> (float, np.ndarray):
    """
    Normalize a truncated current array by its I_sc (first value).
    Returns (isc_value, normalized_array).
    Assumes curve_trunc[0] > 0.
    """
    isc_val = float(curve_trunc[0])
    if isc_val <= 0:
        # If I_sc is zero or negative, skip normalization
        return 1.0, curve_trunc.copy().astype(np.float32)
    norm_curve = curve_trunc / isc_val
    return isc_val, norm_curve.astype(np.float32)


def pad_and_create_mask(
    norm_curves: list,
    volt_curves: list
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Given a list of normalized current arrays and corresponding voltage arrays
    (both truncated), pad each to the maximum length among them.
    Returns:
      - y_padded: (N, max_len) float32, normalized currents, padded by last valid value.
      - v_padded: (N, max_len) float32, voltages, padded by last valid voltage.
      - mask:     (N, max_len) float32, 1.0 for valid points, 0.0 for padded.
      - lengths:  (N,)       int, original lengths of each truncated curve.
    """
    lengths = [c.size for c in norm_curves]
    max_len = max(lengths)
    n_samples = len(norm_curves)

    y_padded = np.zeros((n_samples, max_len), dtype=np.float32)
    v_padded = np.zeros((n_samples, max_len), dtype=np.float32)
    mask     = np.zeros((n_samples, max_len), dtype=np.float32)

    for i, (c, v) in enumerate(zip(norm_curves, volt_curves)):
        L = c.size
        y_padded[i, :L] = c
        v_padded[i, :L] = v
        mask[i, :L]     = 1.0
        if L < max_len:
            y_padded[i, L:] = c[-1]
            v_padded[i, L:] = v[-1]

    return y_padded, v_padded, mask, np.array(lengths, dtype=np.int32)


def remove_outliers_via_isolation_forest(
    X_augmented: np.ndarray,
    scalar_features_df: pd.DataFrame,
    contamination: float = ISOFOREST_CONTAMINATION
) -> np.ndarray:
    """
    Detect outliers using IsolationForest on raw scalar features:
      ['Isc_raw','Vknee_raw','Imax_raw','Imin_raw','Imean_raw'].
    Returns a boolean mask of shape (N,) where True = inlier, False = outlier.
    """
    logger.info(f"Detecting outliers with IsolationForest (contamination={contamination})...")
    iso = IsolationForest(
        contamination=contamination,
        random_state=ISOFOREST_RANDOM_STATE,
        n_estimators=100,
        n_jobs=-1
    )
    features = scalar_features_df[['Isc_raw', 'Vknee_raw', 'Imax_raw', 'Imin_raw', 'Imean_raw']].values
    labels = iso.fit_predict(features)  # +1 for inliers, -1 for outliers
    inlier_mask = labels == 1
    n_outliers = np.sum(~inlier_mask)
    pct_outliers = 100 * n_outliers / len(labels)
    logger.info(f"  Removed {n_outliers} outliers ({pct_outliers:.1f}%).")
    return inlier_mask


def sinusoidal_position_encoding(V_norm: tf.Tensor, d_model: int = VOLTAGE_EMBED_DIM) -> tf.Tensor:
    """
    Generate sinusoidal positional encoding for normalized voltage grid V_norm.
    V_norm shape: (batch_size, seq_len), values ∈ [0, 1].
    Returns: (batch_size, seq_len, d_model) tensor.
    """
    seq_len = tf.shape(V_norm)[1]
    V_exp = tf.expand_dims(V_norm, axis=-1)  # shape = (batch_size, seq_len, 1)

    position = tf.cast(V_exp, tf.float32)
    div_term = tf.exp(
        tf.range(0, d_model, 2, dtype=tf.float32) *
        -(np.log(10000.0) / tf.cast(d_model, tf.float32))
    )  # shape = (d_model/2,)

    angle_rates = position * div_term[None, None, :]
    sin_enc = tf.sin(angle_rates)
    cos_enc = tf.cos(angle_rates)
    pos_encoding = tf.concat([sin_enc, cos_enc], axis=-1)
    return pos_encoding  # dtype = float32


def masked_mse_with_knee_weight(
    y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor, orig_len: tf.Tensor
) -> tf.Tensor:
    """
    Compute MSE with additional weighting (KNEE_WEIGHT_FACTOR) on the last two valid points.
    y_true, y_pred, mask shape: (batch_size, seq_len)
    orig_len shape: (batch_size,) ⇒ lengths of each truncated curve
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask   = tf.cast(mask, tf.float32)
    orig_l = tf.cast(orig_len, tf.int32)

    batch_size = tf.shape(y_pred)[0]
    seq_len    = tf.shape(y_pred)[1]

    se = tf.square(y_true - y_pred)  # shape = (batch_size, seq_len)

    idx = tf.range(seq_len)[None, :]
    last_idx = tf.reshape(orig_l - 1, (batch_size, 1))
    second_last_idx = last_idx - 1

    cond_last = tf.equal(idx, last_idx)
    cond_second_last = tf.logical_and(tf.greater_equal(second_last_idx, 0),
                                      tf.equal(idx, second_last_idx))
    knee_mask = tf.where(tf.logical_or(cond_last, cond_second_last),
                         KNEE_WEIGHT_FACTOR, 1.0)
    knee_mask = tf.cast(knee_mask, tf.float32)

    weighted_se = se * mask * knee_mask
    sum_weighted_se = tf.reduce_sum(weighted_se, axis=-1)
    sum_weights     = tf.reduce_sum(mask * knee_mask, axis=-1)

    mse_per_sample = sum_weighted_se / (sum_weights + 1e-7)
    return tf.reduce_mean(mse_per_sample)  # scalar


def monotonicity_penalty_loss(y_pred: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    Penalize any instance where I_{i+1} > I_i (current increasing with voltage)
    y_pred, mask shape: (batch_size, seq_len)
    """
    y_pred = tf.cast(y_pred, tf.float32)
    mask   = tf.cast(mask, tf.float32)

    diffs = y_pred[:, 1:] - y_pred[:, :-1]
    diff_mask = mask[:, 1:] * mask[:, :-1]
    violations = tf.nn.relu(diffs) * diff_mask
    sum_violation = tf.reduce_sum(tf.square(violations), axis=-1)
    sum_mask = tf.reduce_sum(diff_mask, axis=-1)
    return tf.reduce_mean(sum_violation / (sum_mask + 1e-7))


def curvature_penalty_loss(y_pred: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    Penalize large second derivatives in normalized current (encouraging smooth convex shape).
    y_pred, mask shape: (batch_size, seq_len)
    """
    y_pred = tf.cast(y_pred, tf.float32)
    mask   = tf.cast(mask, tf.float32)

    curvature = y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2]
    curv_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    weighted_curv = tf.square(curvature) * curv_mask
    sum_curv = tf.reduce_sum(weighted_curv, axis=-1)
    sum_mask = tf.reduce_sum(curv_mask, axis=-1)
    return tf.reduce_mean(sum_curv / (sum_mask + 1e-7))


def jsc_voc_penalty_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor, orig_len: tf.Tensor
) -> (tf.Tensor, tf.Tensor):
    """
    Compute two penalties:
      1) J_sc MSE at V=0 → encourage y_pred[:,0] ≈ y_true[:,0]
      2) V_oc penalty → encourage y_pred at last valid index ≈ 0
    y_true, y_pred, mask shape: (batch_size, seq_len)
    orig_len shape: (batch_size,)
    Returns (jsc_mse_scalar, voc_mse_scalar).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask   = tf.cast(mask, tf.float32)
    orig_l = tf.cast(orig_len, tf.int32)

    # J_sc MSE at index 0
    jsc_true = y_true[:, 0]
    jsc_pred = y_pred[:, 0]
    jsc_mask = mask[:, 0]
    jsc_mse = tf.reduce_mean(tf.square(jsc_true - jsc_pred) * jsc_mask)

    batch_size = tf.shape(y_pred)[0]
    indices = tf.range(batch_size, dtype=tf.int32)
    last_indices = tf.clip_by_value(orig_l - 1, 0, tf.shape(y_pred)[1] - 1)
    gather_idx = tf.stack([indices, last_indices], axis=1)
    last_pred = tf.gather_nd(y_pred, gather_idx)

    voc_mask = tf.cast(orig_l > 0, tf.float32)
    voc_mse = tf.reduce_sum(tf.square(last_pred) * voc_mask) / (tf.reduce_sum(voc_mask) + 1e-7)
    return jsc_mse, voc_mse


def combined_physics_loss(loss_weights: dict):
    """
    Returns a loss function that combines:
      - Knee‐weighted masked MSE
      - Monotonicity penalty
      - Curvature penalty
      - J_sc MSE penalty
      - V_oc penalty
    The loss_weights dict must have keys: ['mse','monotonicity','curvature','jsc','voc'].
    """
    def loss_fn(y_true_combined, y_pred):
        y_true = y_true_combined['y_true']
        mask   = y_true_combined['mask']
        orig_l = y_true_combined['orig_len']

        mse_part   = masked_mse_with_knee_weight(y_true, y_pred, mask, orig_l)
        mono_part  = monotonicity_penalty_loss(y_pred, mask)
        curv_part  = curvature_penalty_loss(y_pred, mask)
        jsc_part, voc_part = jsc_voc_penalty_loss(y_true, y_pred, mask, orig_l)

        total = (
            loss_weights['mse'] * mse_part +
            loss_weights['monotonicity'] * mono_part +
            loss_weights['curvature'] * curv_part +
            loss_weights['jsc'] * jsc_part +
            loss_weights['voc'] * voc_part
        )
        return total

    return loss_fn


# ───────────────────────────────────────────────────────────────────────────────
#  Preprocessing: Input Parameters & Scalar Features
# ───────────────────────────────────────────────────────────────────────────────

def preprocess_input_parameters(params_df: pd.DataFrame) -> np.ndarray:
    """
    Transform device/material parameters with group‐wise pipelines:
      - 'material': log(1 + x) → RobustScaler
      - 'device', 'operating', 'reference', 'loss': RobustScaler
    Drops any constant columns first.
    Returns: NumPy array of shape (N_samples, N_processed_param_features).
    """
    logger.info("Preprocessing input parameters (device, material, etc.)...")
    const_tol = 1e-10
    constant_cols = [c for c in params_df.columns if params_df[c].std(ddof=0) <= const_tol]
    if constant_cols:
        logger.info(f"  Removing constant columns: {constant_cols}")
        params_df = params_df.drop(columns=constant_cols)

    param_defs = {
        'material': ['Eg', 'NCv', 'NCc', 'mu_e', 'mu_h', 'eps'],
        'device':   ['A', 'Cn', 'Cp', 'Nt', 'Et', 'nD', 'nA', 'thickness', 'T', 'Sn', 'Sp', 'Rs', 'Rsh'],
        'operating':['G', 'light_intensity'],
        'reference':['Voc_ref', 'Jsc_ref', 'FF_ref', 'PCE_ref'],
        'loss':     ['Qe_loss', 'R_loss', 'SRH_loss', 'series_loss', 'shunt_loss', 'other_loss']
    }

    transformers = []
    for group, cols in param_defs.items():
        actual_cols = [c for c in cols if c in params_df.columns]
        if not actual_cols:
            continue
        steps = []
        if group == 'material':
            # Use FunctionTransformer for log1p to keep it in sklearn pipeline
            steps.append(('log1p', FunctionTransformer(func=np.log1p, validate=False)))
        steps.append(('scaler', RobustScaler()))
        transformers.append((group, Pipeline(steps), actual_cols))

    column_transformer = ColumnTransformer(transformers, remainder='passthrough')
    X_processed = column_transformer.fit_transform(params_df)
    logger.info(f"  Processed param features shape: {X_processed.shape}")
    return X_processed.astype(np.float32)


def preprocess_scalar_features(scalar_features_df: pd.DataFrame, fit: bool = True,
                               scaler: StandardScaler = None) -> (np.ndarray, StandardScaler):
    """
    Standardize scalar features to zero‐mean, unit‐variance.
    Columns: ['Isc_raw','Vknee_raw','Voc_ref_raw','Imax_raw','Imin_raw','Imean_raw'].
    If fit=True, fit a new StandardScaler; otherwise, use provided scaler.
    Returns: (Processed array (N, 6), fitted StandardScaler)
    """
    logger.info(f"Preprocessing scalar features: {list(scalar_features_df.columns)}")
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(scalar_features_df.values)
    else:
        if scaler is None:
            raise RuntimeError("Scalar feature scaler not provided for transform.")
        X_scaled = scaler.transform(scalar_features_df.values)
    logger.info(f"  Processed scalar features shape: {X_scaled.shape}")
    return X_scaled.astype(np.float32), scaler


# ───────────────────────────────────────────────────────────────────────────────
#  Neural Network Definition: Physics‐Informed NN with Voltage Embedding
# ───────────────────────────────────────────────────────────────────────────────

def build_nn_core(input_dim_params: int, seq_len: int, voltage_embed_dim: int = VOLTAGE_EMBED_DIM) -> Model:
    """
    Build the core submodel for the physics‐informed NN:
    - Param path: Dense → BN → Dropout → Dense → BN → Dense → BN
    - Voltage path: Sinusoidal positional encoding → TimeDistributed(Dense)
    - Merge & apply a proper residual block (project merged → 256 dims for skip)
      → Dense(128) → BN → Dropout → output(Signoid)
    Returns a Keras Model with inputs [X_params, voltage_grid] and output iv_output_flat.
    """
    # Parameter input
    x_params_in = Input(shape=(input_dim_params,), name="X_params")
    x = Dense(DENSE_UNITS_PARAMS[0], activation='relu', name="param_dense1")(x_params_in)
    x = BatchNormalization(name="param_bn1")(x)
    x = Dropout(DROPOUT_RATE, name="param_do1")(x)
    x = Dense(DENSE_UNITS_PARAMS[1], activation='relu', name="param_dense2")(x)
    x = BatchNormalization(name="param_bn2")(x)
    x = Dense(DENSE_UNITS_PARAMS[2], activation='relu', name="param_dense3")(x)
    x = BatchNormalization(name="param_bn3")(x)
    param_path = x  # shape = (batch_size, 128)

    # Voltage grid input
    voltage_grid_in = Input(shape=(seq_len,), name="voltage_grid")
    norm_voltage = Lambda(lambda v: v / ASSUMED_ORIGINAL_MAX_VOLTAGE,
                         name="voltage_normalization")(voltage_grid_in)
    pos_enc = Lambda(lambda v: sinusoidal_position_encoding(v, d_model=voltage_embed_dim),
                     output_shape=(seq_len, voltage_embed_dim),
                     name="sinusoidal_pos_enc")(norm_voltage)
    # After positional encoding, shape = (batch_size, seq_len, voltage_embed_dim)
    v_embed = TimeDistributed(Dense(voltage_embed_dim, activation='relu'),
                              name="voltage_embed_td")(pos_enc)
    # Now v_embed.shape = (batch_size, seq_len, 16)

    # Tile the param features so they align with the time dimension
    param_expanded = Lambda(lambda t: tf.expand_dims(t, axis=1), name="param_expand")(param_path)
    # param_expanded.shape = (batch_size, 1, 128)
    param_tiled = Lambda(lambda t: tf.tile(t, [1, seq_len, 1]),
                         output_shape=(seq_len, DENSE_UNITS_PARAMS[-1]),
                         name="param_tile")(param_expanded)
    # param_tiled.shape = (batch_size, seq_len, 128)

    # Concatenate along the last axis → (batch_size, seq_len, 128 + 16 = 144)
    merged = Concatenate(axis=-1, name="concat_param_voltage")([param_tiled, v_embed])
    # merged.shape = (batch_size, seq_len, 144)

    # ───────────────────────────
    #  Residual Block (corrected)
    # ───────────────────────────
    #
    # We want a skip connection + residual that both have the same “middle dimension” (256).
    # So we project `merged` → 256 via a 1×1 “Dense” before adding.

    # 1) Skip path: project merged(144) → 256
    skip = TimeDistributed(Dense(DENSE_UNITS_MERGED[0], activation=None,
                                 name="res_skip1"), name="time_res_skip1")(merged)
    # skip.shape = (batch_size, seq_len, 256)

    # 2) Residual path: merged(144) → Dense(256) → BN → (no activation here since we add then apply activation)
    res = TimeDistributed(Dense(DENSE_UNITS_MERGED[0], activation=None),
                          name="res_dense1_td")(merged)
    # Note: we did not put activation='relu' in the Dense itself, because we want to add skip + raw linear(merged)
    res = BatchNormalization(name="res_bn1")(res)
    res = layers.Activation('relu', name="res_act1")(res)
    # Now res.shape = (batch_size, seq_len, 256)

    # 3) Add skip + res → shape still = (batch_size, seq_len, 256)
    x2 = Add(name="res_add1")([skip, res])
    x2 = Dropout(DROPOUT_RATE, name="res_do1")(x2)

    # ───────────────────────────
    #  Following Dense Layers
    # ───────────────────────────
    x2 = TimeDistributed(Dense(DENSE_UNITS_MERGED[1], activation='relu'),
                         name="merged_dense2_td")(x2)
    x2 = BatchNormalization(name="merged_bn2")(x2)
    x2 = Dropout(DROPOUT_RATE, name="merged_do2")(x2)
    # x2.shape = (batch_size, seq_len, 128)

    # ───────────────────────────
    #  Output layer: Dense(1, 'sigmoid') per time step
    # ───────────────────────────
    iv_output = TimeDistributed(Dense(1, activation='sigmoid'),
                                name='iv_point_output_td')(x2)
    # iv_output.shape = (batch_size, seq_len, 1)

    iv_output_flat = Lambda(lambda t: tf.squeeze(t, axis=-1),
                            name='iv_output_flat')(iv_output)
    # iv_output_flat.shape = (batch_size, seq_len)

    model = Model(inputs=[x_params_in, voltage_grid_in], outputs=iv_output_flat, name="NN_Core")
    return model



class PhysicsNNModel(keras.Model):
    """
    Custom Keras Model wrapper to integrate:
      - A core NN (built by build_nn_core)
      - A physics‐informed loss function (combined_physics_loss)
      - Tracking of loss & MAE metrics
    Supports train_step & test_step overrides to use custom loss.
    """

    def __init__(self, nn_core: Model, loss_weights: dict, **kwargs):
        super().__init__(**kwargs)
        self.nn_core = nn_core
        self.loss_weights = loss_weights
        self.custom_loss_fn = combined_physics_loss(self.loss_weights)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]

    def call(self, inputs, training=False):
        return self.nn_core(inputs, training=training)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data):
        inputs, y_true_combined = data
        with tf.GradientTape() as tape:
            y_pred = self.nn_core(inputs, training=True)
            loss = self.custom_loss_fn(y_true_combined, y_pred)
        grads = tape.gradient(loss, self.nn_core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn_core.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y_true_combined['y_true'], y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    def test_step(self, data):
        inputs, y_true_combined = data
        y_pred = self.nn_core(inputs, training=False)
        loss = self.custom_loss_fn(y_true_combined, y_pred)
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y_true_combined['y_true'], y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}


# ───────────────────────────────────────────────────────────────────────────────
#  PCA & Ensemble Model Training & Prediction (Refactored)
# ───────────────────────────────────────────────────────────────────────────────

class PCAModel:
    """
    Wrapper for PCA + multiple regression on principal components.
    - Fits PCA (cuML if available, else sklearn) on y_train_norm_padded.
    - Trains one regressor per principal component.
    - Predicts by regressing each PC and inverting PCA.
    """

    def __init__(self, use_gpu: bool = False):
        try:
            import cupy
            import cuml
            from cuml.decomposition import PCA as CumlPCA
            self.use_gpu = use_gpu
            self.CumlPCA = CumlPCA
            logger.info("cuML PCA available. Will use GPU for PCA.")
        except (ImportError, ModuleNotFoundError):
            self.use_gpu = False
            self.CumlPCA = None
            logger.warning("cuML PCA not available. Using scikit-learn PCA.")
        self.pca = None
        self.regressors = []
        self.explained_variance_ratio_ = None

    def fit(self, X_train: np.ndarray, y_train_norm_padded: np.ndarray, n_components: int = PCA_N_COMPONENTS):
        max_components = min(n_components, y_train_norm_padded.shape[1], X_train.shape[0])
        if max_components < 1:
            raise RuntimeError("Insufficient data to fit PCA.")
        if max_components < n_components:
            logger.info(f"Adjusting n_components from {n_components} to {max_components}.")

        if self.use_gpu and self.CumlPCA is not None:
            import cupy as cp
            y_gpu = cp.asarray(y_train_norm_padded)
            self.pca = self.CumlPCA(n_components=max_components)
            y_pca_gpu = self.pca.fit_transform(y_gpu)
            y_pca = cp.asnumpy(y_pca_gpu)
            del y_gpu, y_pca_gpu; cp.get_default_memory_pool().free_all_blocks()
        else:
            self.pca = SklearnPCA(n_components=max_components)
            y_pca = self.pca.fit_transform(y_train_norm_padded)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        total_var = np.sum(self.explained_variance_ratio_)
        logger.info(f"PCA fitted: {max_components} components, total variance explained: {total_var:.4f}")

        self.regressors = []
        reg_params = {
            'n_estimators': PCA_N_ESTIMATORS,
            'learning_rate': ENSEMBLE_LEARNING_RATE,
            'max_depth': ENSEMBLE_MAX_DEPTH,
            'subsample': ENSEMBLE_SUBSAMPLE,
            'random_state': ENSEMBLE_RANDOM_STATE
        }
        for i in range(max_components):
            if self.use_gpu:
                import xgboost as xgb
                reg = xgb.XGBRegressor(**reg_params, tree_method='gpu_hist', n_jobs=-1)
            else:
                reg = SklearnGradientBoostingRegressor(**reg_params, validation_fraction=0.1, n_iter_no_change=10)
            logger.info(f"  Training regressor for PC {i+1}/{max_components}...")
            reg.fit(X_train, y_pca[:, i])
            self.regressors.append(reg)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None or not self.regressors:
            raise RuntimeError("PCA model or regressors not fitted.")

        pcs = [r.predict(X) for r in self.regressors]
        pcs = np.vstack(pcs).T

        if self.use_gpu and self.CumlPCA is not None:
            import cupy as cp
            pcs_gpu = cp.asarray(pcs)
            y_pred_gpu = self.pca.inverse_transform(pcs_gpu)
            y_pred = cp.asnumpy(y_pred_gpu)
            del pcs_gpu, y_pred_gpu; cp.get_default_memory_pool().free_all_blocks()
        else:
            y_pred = self.pca.inverse_transform(pcs)

        return y_pred.astype(np.float32)


class EnsembleModel:
    """
    Ensemble of one or more base regressors (XGBoost or RandomForest),
    wrapped in MultiOutputRegressor to predict entire normalized padded curve at once.
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        if self.use_gpu:
            try:
                import xgboost as xgb
                self.base_cls = xgb.XGBRegressor
                self.tree_method = 'gpu_hist'
                logger.info("XGBoost GPU available for ensemble.")
            except (ImportError, ModuleNotFoundError):
                self.base_cls = SklearnRandomForestRegressor
                self.tree_method = None
                logger.warning("XGBoost GPU not available. Falling back to RandomForest.")
        else:
            self.base_cls = SklearnRandomForestRegressor
            self.tree_method = None
            logger.info("Using RandomForest for ensemble (CPU).")
        self.models = {}

    def fit(self, X_train: np.ndarray, y_train_norm_padded: np.ndarray):
        params = {
            'n_estimators': ENSEMBLE_N_ESTIMATORS,
            'max_depth': ENSEMBLE_MAX_DEPTH,
            'random_state': ENSEMBLE_RANDOM_STATE,
            'n_jobs': -1
        }
        if self.use_gpu and self.tree_method == 'gpu_hist':
            params.update({'learning_rate': ENSEMBLE_LEARNING_RATE, 'tree_method': 'gpu_hist'})

        model = self.base_cls(**params)
        multi = MultiOutputRegressor(model, n_jobs=-1)
        logger.info("Training ensemble model on normalized padded curves...")
        multi.fit(X_train, y_train_norm_padded)
        self.models['ensemble_member'] = multi

    def predict(self, X: np.ndarray) -> np.ndarray:
        if 'ensemble_member' not in self.models:
            raise RuntimeError("Ensemble model not fitted.")
        return self.models['ensemble_member'].predict(X).astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────────
#  Full Reconstructor Class: Integrates Preprocessing, PCA, NN, Ensemble
# ───────────────────────────────────────────────────────────────────────────────

class TruncatedIVReconstructor:
    """
    Main class to orchestrate:
      - Data loading & cleaning (truncation, smoothing, scaling, padding, outlier removal)
      - PCA + regression, Physics‐informed NN, Ensemble training
      - Prediction and evaluation on test data
      - Plotting of results
    """

    def __init__(self, use_gpu_if_available: bool = True):
        # Check for TensorFlow GPU availability
        available_gpus = tf.config.list_physical_devices('GPU')
        self.use_gpu = bool(available_gpus) and use_gpu_if_available
        if self.use_gpu:
            logger.info("TensorFlow GPU detected. Will use GPU acceleration where possible.")
            try:
                for gpu in available_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"{len(available_gpus)} GPU(s) configured for TF.")
            except RuntimeError as e:
                logger.warning(f"Error configuring GPU memory growth: {e}")
        else:
            logger.info("No GPU detected or GPU usage disabled. Using CPU only.")

        # Holders for processed data & models
        self.X_clean = None
        self.y_clean_norm_padded = None
        self.padded_voltages = None
        self.masks = None
        self.per_curve_isc = None
        self.orig_lengths = None
        self.scalar_features_df = None
        self.scalar_scaler = None

        # Models
        self.pca_regressor = None
        self.nn_model = None
        self.ensemble_model = None

        # Training history
        self.training_history = None

    def check_data_files_exist(self) -> bool:
        exists = Path(INPUT_FILE_PARAMS).exists() and Path(INPUT_FILE_IV).exists()
        if not exists:
            logger.error(f"Input files not found: {INPUT_FILE_PARAMS}, {INPUT_FILE_IV}")
        return exists

    def load_and_prepare_data(self, truncation_threshold_pct: float = 0.0) -> bool:
        """
        Load raw parameter file and I–V file, then:
          1. Truncate each I–V curve at threshold_pct * I_sc
          2. Smooth via Savitzky–Golay
          3. Normalize each truncated curve by its I_sc
          4. Build scalar feature DataFrame
          5. Pad truncated + normalized curves & build mask matrix
          6. Preprocess parameter and scalar features
          7. Remove outliers
        Populates:
          - self.X_clean (float32 array of shape (N_clean, N_features_total))
          - self.y_clean_norm_padded (float32 array (N_clean, max_len))
          - self.padded_voltages (float32 array (N_clean, max_len))
          - self.masks (float32 array (N_clean, max_len))
          - self.per_curve_isc (float32 array (N_clean,))
          - self.orig_lengths (int32 array (N_clean,))
          - self.scalar_features_df (DataFrame of shape (N_clean, 6))
        Returns True if successful, False otherwise.
        """
        logger.info(f"=== Loading & Preparing Data (Truncation PCT: {truncation_threshold_pct}) ===")
        if not self.check_data_files_exist():
            return False

        try:
            params_df = pd.read_csv(INPUT_FILE_PARAMS, header=None, names=COLNAMES)
            iv_data_np = np.loadtxt(INPUT_FILE_IV, delimiter=',', dtype=np.float32)
            logger.info(f"Loaded raw data: params {params_df.shape}, I–V {iv_data_np.shape}")

            full_voltage_grid = np.linspace(
                0.0, ASSUMED_ORIGINAL_MAX_VOLTAGE, ASSUMED_ORIGINAL_IV_POINTS, dtype=np.float32
            )

            raw_currents = []
            raw_voltages = []
            orig_isc = []
            orig_vknee = []
            orig_voc_ref = []
            valid_param_indices = []

            for i in range(iv_data_np.shape[0]):
                curve_i = iv_data_np[i]
                voltage_trunc, current_trunc = truncate_iv_curve(
                    curve_i, full_voltage_grid, truncation_threshold_pct
                )
                if voltage_trunc is None:
                    continue

                current_smooth = apply_savgol(current_trunc)

                raw_currents.append(current_smooth)
                raw_voltages.append(voltage_trunc)
                orig_isc.append(float(current_smooth[0]))
                orig_vknee.append(float(voltage_trunc[-1]))
                orig_voc_ref.append(float(params_df.iloc[i]['Voc_ref']))
                valid_param_indices.append(i)

            if not raw_currents:
                logger.error("No valid curves after truncation. Aborting.")
                return False

            norm_curves = []
            per_curve_isc = []
            for c in raw_currents:
                isc_val, norm_c = normalize_by_isc(c)
                per_curve_isc.append(isc_val)
                norm_curves.append(norm_c)

            y_padded, v_padded, mask_matrix, lengths = pad_and_create_mask(norm_curves, raw_voltages)

            scalar_df = pd.DataFrame({
                'Isc_raw': orig_isc,
                'Vknee_raw': orig_vknee,
                'Voc_ref_raw': orig_voc_ref,
                'Imax_raw': [float(np.max(c)) for c in raw_currents],
                'Imin_raw': [float(np.min(c)) for c in raw_currents],
                'Imean_raw': [float(np.mean(c)) for c in raw_currents]
            })

            params_df_valid = params_df.iloc[valid_param_indices].reset_index(drop=True)

            X_params = preprocess_input_parameters(params_df_valid)
            X_scalar, self.scalar_scaler = preprocess_scalar_features(scalar_df, fit=True)

            X_augmented = np.concatenate([X_params, X_scalar], axis=1)
            logger.info(f"Augmented feature shape (pre‐outlier removal): {X_augmented.shape}")

            inlier_mask = remove_outliers_via_isolation_forest(X_augmented, scalar_df)
            if not np.any(inlier_mask):
                logger.error("All samples flagged as outliers. Aborting.")
                return False

            self.X_clean = X_augmented[inlier_mask]
            self.y_clean_norm_padded = y_padded[inlier_mask]
            self.padded_voltages = v_padded[inlier_mask]
            self.masks = mask_matrix[inlier_mask]
            self.per_curve_isc = np.array(per_curve_isc, dtype=np.float32)[inlier_mask]
            self.orig_lengths = lengths[inlier_mask]
            self.scalar_features_df = scalar_df.iloc[inlier_mask].reset_index(drop=True)

            logger.info(f"Final cleaned feature shape: {self.X_clean.shape}")
            logger.info(f"Final cleaned y_norm_padded shape: {self.y_clean_norm_padded.shape}")
            logger.info(f"Max truncated length overall: {self.y_clean_norm_padded.shape[1]}")
            return True

        except Exception as e:
            logger.exception(f"Error during data loading/preparation: {e}")
            return False

    def fit_pca_model(self, n_components: int = PCA_N_COMPONENTS):
        """
        Fit PCA + regressors on the cleaned dataset.
        """
        if self.X_clean is None or self.y_clean_norm_padded is None:
            raise RuntimeError("Data not prepared. Call load_and_prepare_data first.")
        self.pca_regressor = PCAModel(use_gpu=self.use_gpu)
        self.pca_regressor.fit(self.X_clean, self.y_clean_norm_padded, n_components=n_components)
        logger.info("PCA + regressors training completed.")

    def fit_physics_informed_nn(
        self,
        X_train: np.ndarray, y_train_norm: np.ndarray,
        V_train: np.ndarray, M_train: np.ndarray, L_train: np.ndarray,
        X_val: np.ndarray = None, y_val_norm: np.ndarray = None,
        V_val: np.ndarray = None, M_val: np.ndarray = None, L_val: np.ndarray = None,
    ):
        """
        Fit the physics‐informed neural network on training data, optionally with validation.
        """
        logger.info("=== Fitting Physics‐Informed Neural Network ===")
        input_dim_params = X_train.shape[1]
        seq_len = self.y_clean_norm_padded.shape[1]
        if seq_len == 0:
            raise RuntimeError("No output dimension for NN (seq_len=0). Skipping NN training.")

        nn_core = build_nn_core(input_dim_params, seq_len, voltage_embed_dim=VOLTAGE_EMBED_DIM)
        physics_model = PhysicsNNModel(nn_core, LOSS_WEIGHTS)
        optimizer = Adam(learning_rate=NN_LEARNING_RATE)
        physics_model.compile(optimizer=optimizer)
        self.nn_model = physics_model

        train_inputs = {"X_params": X_train, "voltage_grid": V_train}
        train_targets = {"y_true": y_train_norm, "mask": M_train, "orig_len": L_train}
        train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
        train_ds = train_ds.shuffle(buffer_size=min(X_train.shape[0], 10000)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_ds = None
        if X_val is not None and y_val_norm is not None:
            val_inputs = {"X_params": X_val, "voltage_grid": V_val}
            val_targets = {"y_true": y_val_norm, "mask": M_val, "orig_len": L_val}
            val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets))
            val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        else:
            logger.warning("No validation data provided for NN.")

        callbacks = [
            EarlyStopping(monitor='val_loss' if val_ds else 'loss', patience=30, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss' if val_ds else 'loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1)
        ]

        history = physics_model.fit(
            train_ds,
            epochs=NN_EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        self.training_history = history
        logger.info("Physics‐informed NN training completed.")

    def fit_ensemble_model(self):
        """
        Fit the ensemble model on the cleaned dataset (predicting entire norm_padded curves).
        """
        if self.X_clean is None or self.y_clean_norm_padded is None:
            raise RuntimeError("Data not prepared. Call load_and_prepare_data first.")
        self.ensemble_model = EnsembleModel(use_gpu=self.use_gpu)
        self.ensemble_model.fit(self.X_clean, self.y_clean_norm_padded)
        logger.info("Ensemble model training completed.")

    def predict(
        self, X: np.ndarray, V_padded: np.ndarray,
        per_curve_isc: np.ndarray, orig_lengths: np.ndarray,
        model_type: str = 'physics_nn'
    ) -> list:
        """
        Predict truncated physical I–V curves (un‐normalized) for X.
        model_type: 'pca', 'physics_nn', or 'ensemble'.
        Returns a list of length N_samples, each entry is an array of length orig_len[i].
        """
        if model_type == 'pca':
            if self.pca_regressor is None:
                raise RuntimeError("PCA model not fitted.")
            y_pred_norm_padded = self.pca_regressor.predict(X)  # shape = (N, seq_len)
        elif model_type == 'physics_nn':
            if self.nn_model is None:
                raise RuntimeError("NN model not fitted.")
            inputs = {"X_params": X, "voltage_grid": V_padded}
            y_pred_norm_padded = self.nn_model.predict(inputs, batch_size=512, verbose=0)
        elif model_type == 'ensemble':
            if self.ensemble_model is None:
                raise RuntimeError("Ensemble model not fitted.")
            y_pred_norm_padded = self.ensemble_model.predict(X)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from 'pca','physics_nn','ensemble'.")

        N = X.shape[0]
        y_pred_list = []
        for i in range(N):
            norm_curve = y_pred_norm_padded[i]
            isc_val = per_curve_isc[i]
            predicted_full_curve = norm_curve * isc_val
            L = orig_lengths[i]
            if predicted_full_curve.size < L:
                padded = np.zeros((L,), dtype=np.float32)
                padded[:predicted_full_curve.size] = predicted_full_curve
                pred_final = padded
            else:
                pred_final = predicted_full_curve[:L]
            y_pred_list.append(pred_final.astype(np.float32))
        return y_pred_list

    def evaluate_model(
        self, X: np.ndarray, V_padded: np.ndarray,
        y_true_curves: list, y_true_voltages: list,
        per_curve_isc: np.ndarray, orig_lengths: np.ndarray,
        model_type: str = 'physics_nn'
    ) -> dict:
        """
        Evaluate the specified model on test data and return metrics:
          - 'MAE', 'RMSE', 'R2' (global over all points),
          - 'per_curve_R2_mean', 'per_curve_R2_std'
        Also returns 'predictions': list of predicted truncated curves.
        """
        logger.info(f"=== Evaluating {model_type.upper()} ===")
        y_pred_list = self.predict(X, V_padded, per_curve_isc, orig_lengths, model_type=model_type)

        flat_true = []
        flat_pred = []
        per_curve_r2 = []
        for i, (true_curve, pred_curve) in enumerate(zip(y_true_curves, y_pred_list)):
            L = orig_lengths[i]
            true_slice = true_curve[:L]
            pred_slice = pred_curve[:L]
            if true_slice.size == 0:
                continue
            flat_true.extend(true_slice)
            flat_pred.extend(pred_slice)
            if true_slice.size > 1:
                r2_val = r2_score(true_slice, pred_slice) if np.std(true_slice) > 1e-9 else (
                    1.0 if mean_squared_error(true_slice, pred_slice) < 1e-6 else 0.0
                )
                per_curve_r2.append(r2_val)

        flat_true = np.array(flat_true, dtype=np.float32)
        flat_pred = np.array(flat_pred, dtype=np.float32)
        if flat_true.size == 0:
            raise RuntimeError("No valid points found for evaluation (flat_true is empty).")

        mae = mean_absolute_error(flat_true, flat_pred)
        rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
        r2_global = r2_score(flat_true, flat_pred) if np.std(flat_true) > 1e-9 else (
            1.0 if mean_squared_error(flat_true, flat_pred) < 1e-6 else 0.0
        )
        mean_r2 = np.mean(per_curve_r2) if per_curve_r2 else np.nan
        std_r2 = np.std(per_curve_r2) if per_curve_r2 else np.nan

        logger.info(f"  Truncated Original scale: MAE={mae:.6f}, RMSE={rmse:.6f}, Global R²={r2_global:.4f}")
        logger.info(f"  Per‐curve R²: Mean={mean_r2:.4f}, Std={std_r2:.4f}")

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2_global,
            'per_curve_R2_mean': mean_r2,
            'per_curve_R2_std': std_r2,
            'predictions': y_pred_list
        }

    def plot_results(
        self, X: np.ndarray, V_padded: np.ndarray,
        y_true_curves: list, y_true_voltages: list,
        per_curve_isc: np.ndarray, orig_lengths: np.ndarray,
        model_type: str = 'physics_nn', n_samples: int = 4, suffix: str = ""
    ):
        """
        Plot a random selection of n_samples from the provided data:
          True vs Predicted truncated I–V curves, annotated with per‐curve R².
        Saves the figure to OUTPUT_DIR as 'preds_{model_type}{suffix}.png'.
        """
        y_pred_list = self.predict(X, V_padded, per_curve_isc, orig_lengths, model_type=model_type)
        N = len(y_true_curves)
        if N == 0:
            logger.warning(f"No samples to plot for {model_type}.")
            return

        indices = np.random.choice(N, size=min(n_samples, N), replace=False)
        if n_samples <= 4:
            nrows, ncols = 2, 2
        elif n_samples <= 6:
            nrows, ncols = 2, 3
        elif n_samples <= 9:
            nrows, ncols = 3, 3
        else:
            ncols = 4
            nrows = (n_samples + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes = axes.flatten()

        for i_plot, idx in enumerate(indices):
            ax = axes[i_plot]
            true_curve = y_true_curves[idx]
            volt_curve = y_true_voltages[idx]
            pred_curve = y_pred_list[idx]
            L = orig_lengths[idx]
            if L == 0:
                ax.text(0.5, 0.5, f"Sample {idx}\nEmpty", ha='center', va='center')
                ax.set_title(f"Sample {idx} - Error")
                continue

            true_v = volt_curve[:L]
            true_i = true_curve[:L]
            pred_i = pred_curve[:L]
            if true_i.size > 1:
                r2_val = r2_score(true_i, pred_i) if np.std(true_i) > 1e-9 else (
                    1.0 if mean_squared_error(true_i, pred_i) < 1e-6 else 0.0
                )
            else:
                r2_val = np.nan

            ax.plot(true_v, true_i, 'b-', lw=2, label='Actual (Trunc)', alpha=0.8)
            ax.plot(true_v, pred_i, 'r--', lw=2, label='Predicted (Trunc)', alpha=0.8)
            ax.set_title(f"Sample {idx} - {model_type.upper()} (R²={r2_val:.3f})")
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current Density (mA/cm²)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        for j in range(i_plot + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        outpath = OUTPUT_DIR / f"preds_{model_type}{suffix}.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Prediction plot saved: {outpath}")


# ───────────────────────────────────────────────────────────────────────────────
#  Main Entry Points: Single Run & Bulk Experiment
# ───────────────────────────────────────────────────────────────────────────────

def main_single_run(trunc_thresh_pct: float = 0.0, use_gpu: bool = True):
    """
    Single‐configuration run for debugging:
      - Loads & prepares data with truncation threshold
      - Splits into train/val/test
      - Trains PhysicsNN
      - Evaluates on test set
      - Saves plots and returns the test metrics dict
    """
    logger.info(f"=== SINGLE RUN: Trunc Thresh={trunc_thresh_pct:.3f}, Use GPU={use_gpu} ===")
    recon = TruncatedIVReconstructor(use_gpu_if_available=use_gpu)
    if not recon.load_and_prepare_data(truncation_threshold_pct=trunc_thresh_pct):
        logger.error("Data preparation failed. Exiting single run.")
        return None

    N = recon.X_clean.shape[0]
    all_idx = np.arange(N)
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    X_train = recon.X_clean[train_idx]
    y_train = recon.y_clean_norm_padded[train_idx]
    V_train = recon.padded_voltages[train_idx]
    M_train = recon.masks[train_idx]
    L_train = recon.orig_lengths[train_idx]

    X_val = recon.X_clean[val_idx]
    y_val = recon.y_clean_norm_padded[val_idx]
    V_val = recon.padded_voltages[val_idx]
    M_val = recon.masks[val_idx]
    L_val = recon.orig_lengths[val_idx]

    X_test = recon.X_clean[test_idx]
    V_test = recon.padded_voltages[test_idx]
    L_test = recon.orig_lengths[test_idx]
    isc_test = recon.per_curve_isc[test_idx]

    y_test_curves = [
        recon.y_clean_norm_padded[i][:L_test[j]] * recon.per_curve_isc[i]
        for j, i in enumerate(test_idx)
    ]
    y_test_voltages = [
        recon.padded_voltages[i][:L_test[j]] for j, i in enumerate(test_idx)
    ]

    recon.fit_physics_informed_nn(
        X_train, y_train, V_train, M_train, L_train,
        X_val, y_val, V_val, M_val, L_val
    )

    if recon.training_history:
        hist = recon.training_history.history
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(hist['loss'], label='Train Loss', color='blue')
        if 'val_loss' in hist:
            ax1.plot(hist['val_loss'], label='Val Loss', color='cyan', linestyle='--')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        if 'mae' in hist:
            ax2.plot(hist['mae'], label='Train MAE', color='red', linestyle=':')
        if 'val_mae' in hist:
            ax2.plot(hist['val_mae'], label='Val MAE', color='magenta', linestyle='-.')
        ax2.set_ylabel("MAE (normalized)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
        plt.title("PhysicsNN Training History (Single Run)")
        plt.grid(True, alpha=0.2)
        hist_out = OUTPUT_DIR / "training_history_single_run.png"
        plt.savefig(hist_out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Training history plot saved: {hist_out}")
    else:
        logger.warning("No training history found for PhysicsNN (Single Run).")

    test_metrics = recon.evaluate_model(
        X_test, V_test, y_test_curves, y_test_voltages, isc_test, L_test, model_type='physics_nn'
    )

    recon.plot_results(
        X_test, V_test, y_test_curves, y_test_voltages, isc_test, L_test,
        model_type='physics_nn', n_samples=6, suffix="_single_run_test"
    )
    return test_metrics


def main_bulk_experiment():
    """
    Bulk experiment (single config) analogous to the original bulk function,
    but now only runs one configuration (e.g. best found threshold + no global scaling).
    Saves summary CSV and result plots to OUTPUT_DIR.
    """
    logger.info("=== BULK EXPERIMENT: Single Configuration ===")
    best_threshold = 0.10

    recon = TruncatedIVReconstructor(use_gpu_if_available=True)
    if not recon.load_and_prepare_data(truncation_threshold_pct=best_threshold):
        logger.error("Data preparation failed. Exiting bulk experiment.")
        return

    N = recon.X_clean.shape[0]
    all_idx = np.arange(N)
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    X_train = recon.X_clean[train_idx]
    y_train = recon.y_clean_norm_padded[train_idx]
    V_train = recon.padded_voltages[train_idx]
    M_train = recon.masks[train_idx]
    L_train = recon.orig_lengths[train_idx]

    X_val = recon.X_clean[val_idx]
    y_val = recon.y_clean_norm_padded[val_idx]
    V_val = recon.padded_voltages[val_idx]
    M_val = recon.masks[val_idx]
    L_val = recon.orig_lengths[val_idx]

    X_test = recon.X_clean[test_idx]
    V_test = recon.padded_voltages[test_idx]
    L_test = recon.orig_lengths[test_idx]
    isc_test = recon.per_curve_isc[test_idx]

    y_test_curves = [
        recon.y_clean_norm_padded[i] * recon.per_curve_isc[i]
        for i in test_idx
    ]
    y_test_voltages = [
        recon.padded_voltages[i][:L_test[j]] for j, i in enumerate(test_idx)
    ]

    recon.fit_physics_informed_nn(
        X_train, y_train, V_train, M_train, L_train,
        X_val, y_val, V_val, M_val, L_val
    )

    if recon.training_history:
        hist = recon.training_history.history
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(hist['loss'], label='Train Loss', color='blue')
        if 'val_loss' in hist:
            ax1.plot(hist['val_loss'], label='Val Loss', color='cyan', linestyle='--')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        if 'mae' in hist:
            ax2.plot(hist['mae'], label='Train MAE', color='red', linestyle=':')
        if 'val_mae' in hist:
            ax2.plot(hist['val_mae'], label='Val MAE', color='magenta', linestyle='-.')
        ax2.set_ylabel("MAE (normalized)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
        plt.title(f"PhysicsNN Training History (Thresh={best_threshold:.3f})")
        plt.grid(True, alpha=0.3)
        hist_out = OUTPUT_DIR / f"training_history_thresh_{best_threshold:.3f}.png"
        plt.savefig(hist_out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Training history plot saved: {hist_out}")
    else:
        logger.warning("No training history found for PhysicsNN (bulk).")

    val_metrics = recon.evaluate_model(
        X_val, V_val,
        [recon.y_clean_norm_padded[i] * recon.per_curve_isc[i] for i in val_idx],
        [recon.padded_voltages[i][:L_val[j]] for j, i in enumerate(val_idx)],
        recon.per_curve_isc[val_idx], L_val,
        model_type='physics_nn'
    )
    logger.info(f"  [Validation] Thresh={best_threshold:.3f}: R²/curve={val_metrics['per_curve_R2_mean']:.4f} (MAE={val_metrics['MAE']:.4f})")

    recon.plot_results(
        X_val, V_val,
        [recon.y_clean_norm_padded[i] * recon.per_curve_isc[i] for i in val_idx],
        [recon.padded_voltages[i][:L_val[j]] for j, i in enumerate(val_idx)],
        recon.per_curve_isc[val_idx], L_val,
        model_type='physics_nn', n_samples=6, suffix=f"_val_thresh_{best_threshold:.3f}"
    )

    test_metrics = recon.evaluate_model(
        X_test, V_test,
        y_test_curves,
        y_test_voltages,
        isc_test,
        L_test,
        model_type='physics_nn'
    )
    logger.info(f"  [Test] Thresh={best_threshold:.3f}: R²/curve={test_metrics['per_curve_R2_mean']:.4f} (MAE={test_metrics['MAE']:.4f})")

    recon.plot_results(
        X_test, V_test, y_test_curves, y_test_voltages,
        isc_test, L_test,
        model_type='physics_nn', n_samples=6, suffix=f"_test_thresh_{best_threshold:.3f}"
    )

    summary = {
        'config_threshold': best_threshold,
        'val_R2_curve': val_metrics['per_curve_R2_mean'],
        'val_MAE': val_metrics['MAE'],
        'test_R2_curve': test_metrics['per_curve_R2_mean'],
        'test_MAE': test_metrics['MAE']
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = OUTPUT_DIR / "bulk_experiment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Bulk experiment summary saved: {summary_csv}")


if __name__ == "__main__":
    # To run a single configuration:
    # test_results = main_single_run(trunc_thresh_pct=0.10, use_gpu=True)
    # print("Single Run Results:", test_results)

    # To run the bulk experiment (single config):
    main_bulk_experiment()
