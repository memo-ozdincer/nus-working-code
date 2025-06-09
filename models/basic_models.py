#!/usr/bin/env python3
"""
refactored_iv_reconstruction.py

A lean, focused implementation for Perovskite I–V curve reconstruction using a
physics‐informed neural network. This version focuses exclusively on the NN
approach, removing PCA and ensemble methods for clarity and simplicity.

Key features:
- Single‐responsibility utility functions for preprocessing (truncation, smoothing, padding, outlier removal).
- Per‐curve I_sc‐based normalization.
- A sophisticated physics‐informed loss function including:
  - Knee‐weighted Mean Squared Error (MSE).
  - Penalties for monotonicity, curvature, J_sc, and V_oc.
- Sinusoidal positional encoding for voltage embedding with a residual block architecture.
- Centralized hyperparameter configuration and logging.
- A streamlined main execution path for training, evaluation, and visualization.

"""

import os
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
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
    max_len = max(lengths) if lengths else 0
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
    scalar_features_df: pd.DataFrame,
    contamination: float = ISOFOREST_CONTAMINATION
) -> np.ndarray:
    """
    Detect outliers using IsolationForest on raw scalar features derived from the curves.
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
    return pos_encoding


def masked_mse_with_knee_weight(
    y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor, orig_len: tf.Tensor
) -> tf.Tensor:
    """
    Compute MSE with additional weighting (KNEE_WEIGHT_FACTOR) on the last two valid points.
    y_true, y_pred, mask shape: (batch_size, seq_len)
    orig_len shape: (batch_size,) ⇒ lengths of each truncated curve
    """
    y_true, y_pred, mask = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), tf.cast(mask, tf.float32)
    orig_l = tf.cast(orig_len, tf.int32)

    batch_size, seq_len = tf.shape(y_pred)[0], tf.shape(y_pred)[1]
    se = tf.square(y_true - y_pred)

    idx = tf.range(seq_len)[None, :]
    last_idx = tf.reshape(orig_l - 1, (batch_size, 1))
    second_last_idx = last_idx - 1

    is_last = tf.equal(idx, last_idx)
    is_second_last = tf.logical_and(tf.greater_equal(second_last_idx, 0), tf.equal(idx, second_last_idx))
    knee_mask = tf.where(tf.logical_or(is_last, is_second_last), KNEE_WEIGHT_FACTOR, 1.0)
    knee_mask = tf.cast(knee_mask, tf.float32)

    weighted_se = se * mask * knee_mask
    mse_per_sample = tf.reduce_sum(weighted_se, axis=-1) / (tf.reduce_sum(mask * knee_mask, axis=-1) + 1e-7)
    return tf.reduce_mean(mse_per_sample)


def monotonicity_penalty_loss(y_pred: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    Penalize any instance where I_{i+1} > I_i (current increasing with voltage).
    """
    y_pred, mask = tf.cast(y_pred, tf.float32), tf.cast(mask, tf.float32)
    diffs = y_pred[:, 1:] - y_pred[:, :-1]
    diff_mask = mask[:, 1:] * mask[:, :-1]
    violations = tf.nn.relu(diffs) * diff_mask
    sum_violation = tf.reduce_sum(tf.square(violations), axis=-1)
    return tf.reduce_mean(sum_violation / (tf.reduce_sum(diff_mask, axis=-1) + 1e-7))


def curvature_penalty_loss(y_pred: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    Penalize large second derivatives in normalized current (encouraging smooth convex shape).
    """
    y_pred, mask = tf.cast(y_pred, tf.float32), tf.cast(mask, tf.float32)
    curvature = y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2]
    curv_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    weighted_curv = tf.square(curvature) * curv_mask
    return tf.reduce_mean(tf.reduce_sum(weighted_curv, axis=-1) / (tf.reduce_sum(curv_mask, axis=-1) + 1e-7))


def jsc_voc_penalty_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor, orig_len: tf.Tensor
) -> (tf.Tensor, tf.Tensor):
    """
    Compute two penalties: J_sc MSE at V=0 and V_oc MSE (current at last point should be 0).
    """
    y_true, y_pred, mask = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), tf.cast(mask, tf.float32)
    orig_l = tf.cast(orig_len, tf.int32)

    # J_sc MSE at index 0
    jsc_mse = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]) * mask[:, 0])

    # V_oc penalty: current at the last valid index should be zero
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
    Factory function for a combined physics-informed loss.
    """
    def loss_fn(y_true_combined, y_pred):
        y_true, mask, orig_l = y_true_combined['y_true'], y_true_combined['mask'], y_true_combined['orig_len']
        mse_part = masked_mse_with_knee_weight(y_true, y_pred, mask, orig_l)
        mono_part = monotonicity_penalty_loss(y_pred, mask)
        curv_part = curvature_penalty_loss(y_pred, mask)
        jsc_part, voc_part = jsc_voc_penalty_loss(y_true, y_pred, mask, orig_l)

        return (loss_weights['mse'] * mse_part +
                loss_weights['monotonicity'] * mono_part +
                loss_weights['curvature'] * curv_part +
                loss_weights['jsc'] * jsc_part +
                loss_weights['voc'] * voc_part)
    return loss_fn

# ───────────────────────────────────────────────────────────────────────────────
#  Preprocessing: Input Parameters & Scalar Features
# ───────────────────────────────────────────────────────────────────────────────

def preprocess_input_parameters(params_df: pd.DataFrame) -> np.ndarray:
    """
    Transform device/material parameters with group‐wise pipelines.
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
        if not actual_cols: continue
        steps = [('log1p', FunctionTransformer(func=np.log1p, validate=False))] if group == 'material' else []
        steps.append(('scaler', RobustScaler()))
        transformers.append((group, Pipeline(steps), actual_cols))

    column_transformer = ColumnTransformer(transformers, remainder='passthrough')
    X_processed = column_transformer.fit_transform(params_df)
    logger.info(f"  Processed param features shape: {X_processed.shape}")
    return X_processed.astype(np.float32)


def preprocess_scalar_features(scalar_features_df: pd.DataFrame, fit: bool = True,
                               scaler: StandardScaler = None) -> (np.ndarray, StandardScaler):
    """
    Standardize scalar features derived from the I-V curves.
    """
    logger.info(f"Preprocessing scalar features: {list(scalar_features_df.columns)}")
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(scalar_features_df.values)
    else:
        if scaler is None: raise RuntimeError("Scaler not provided for transform.")
        X_scaled = scaler.transform(scalar_features_df.values)
    logger.info(f"  Processed scalar features shape: {X_scaled.shape}")
    return X_scaled.astype(np.float32), scaler

# ───────────────────────────────────────────────────────────────────────────────
#  Neural Network Definition
# ───────────────────────────────────────────────────────────────────────────────

def build_nn_core(input_dim_params: int, seq_len: int, voltage_embed_dim: int = VOLTAGE_EMBED_DIM) -> Model:
    """
    Build the core Keras Model for the physics‐informed NN.
    """
    # Parameter input path
    x_params_in = Input(shape=(input_dim_params,), name="X_params")
    x = x_params_in
    for i, units in enumerate(DENSE_UNITS_PARAMS):
        x = Dense(units, activation='relu', name=f"param_dense{i+1}")(x)
        x = BatchNormalization(name=f"param_bn{i+1}")(x)
        if i == 0: x = Dropout(DROPOUT_RATE, name=f"param_do{i+1}")(x)
    param_path = x

    # Voltage grid input path
    voltage_grid_in = Input(shape=(seq_len,), name="voltage_grid")
    norm_voltage = Lambda(lambda v: v / ASSUMED_ORIGINAL_MAX_VOLTAGE)(voltage_grid_in)
    pos_enc = Lambda(lambda v: sinusoidal_position_encoding(v, d_model=voltage_embed_dim))(norm_voltage)
    v_embed = TimeDistributed(Dense(voltage_embed_dim, activation='relu'))(pos_enc)

    # Merge paths
    param_tiled = Lambda(lambda t: tf.tile(tf.expand_dims(t, 1), [1, seq_len, 1]))(param_path)
    merged = Concatenate(axis=-1)([param_tiled, v_embed])

    # Residual Block
    skip = TimeDistributed(Dense(DENSE_UNITS_MERGED[0], activation=None))(merged)
    res = TimeDistributed(Dense(DENSE_UNITS_MERGED[0], activation='relu'))(merged)
    res = BatchNormalization()(res)
    x2 = Add()([skip, res])
    x2 = Dropout(DROPOUT_RATE)(x2)

    # Final dense layers
    x2 = TimeDistributed(Dense(DENSE_UNITS_MERGED[1], activation='relu'))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(DROPOUT_RATE)(x2)

    # Output layer
    iv_output = TimeDistributed(Dense(1, activation='sigmoid'))(x2)
    iv_output_flat = Lambda(lambda t: tf.squeeze(t, axis=-1))(iv_output)

    return Model(inputs=[x_params_in, voltage_grid_in], outputs=iv_output_flat, name="NN_Core")


class PhysicsNNModel(keras.Model):
    """
    Custom Keras Model wrapper to integrate the core NN with the physics-informed loss.
    """
    def __init__(self, nn_core: Model, loss_weights: dict, **kwargs):
        super().__init__(**kwargs)
        self.nn_core = nn_core
        self.custom_loss_fn = combined_physics_loss(loss_weights)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]

    def call(self, inputs, training=False):
        return self.nn_core(inputs, training=training)

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
#  Full Reconstructor Class
# ───────────────────────────────────────────────────────────────────────────────

class TruncatedIVReconstructor:
    """
    Main class to orchestrate data preparation, NN training, prediction, and evaluation.
    """
    def __init__(self, use_gpu_if_available: bool = True):
        available_gpus = tf.config.list_physical_devices('GPU')
        self.use_gpu = bool(available_gpus) and use_gpu_if_available
        if self.use_gpu:
            logger.info("TensorFlow GPU detected. Will use GPU acceleration.")
            try:
                for gpu in available_gpus: tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"Error configuring GPU memory growth: {e}")
        else:
            logger.info("No GPU detected or GPU usage disabled. Using CPU only.")

        self.X_clean = self.y_clean_norm_padded = self.padded_voltages = None
        self.masks = self.per_curve_isc = self.orig_lengths = None
        self.scalar_scaler = None
        self.nn_model = self.training_history = None

    def load_and_prepare_data(self, truncation_threshold_pct: float = 0.0) -> bool:
        """
        Load, preprocess, and clean all input data.
        Returns True if successful, False otherwise.
        """
        logger.info(f"=== Loading & Preparing Data (Truncation PCT: {truncation_threshold_pct}) ===")
        if not (Path(INPUT_FILE_PARAMS).exists() and Path(INPUT_FILE_IV).exists()):
            logger.error(f"Input files not found: {INPUT_FILE_PARAMS}, {INPUT_FILE_IV}")
            return False
        try:
            params_df = pd.read_csv(INPUT_FILE_PARAMS, header=None, names=COLNAMES)
            iv_data_np = np.loadtxt(INPUT_FILE_IV, delimiter=',', dtype=np.float32)
            logger.info(f"Loaded raw data: params {params_df.shape}, I–V {iv_data_np.shape}")
            full_voltage_grid = np.linspace(0.0, ASSUMED_ORIGINAL_MAX_VOLTAGE, ASSUMED_ORIGINAL_IV_POINTS, dtype=np.float32)

            raw_currents, raw_voltages, valid_indices = [], [], []
            for i in range(iv_data_np.shape[0]):
                v_trunc, c_trunc = truncate_iv_curve(iv_data_np[i], full_voltage_grid, truncation_threshold_pct)
                if v_trunc is None: continue
                raw_voltages.append(v_trunc)
                raw_currents.append(apply_savgol(c_trunc))
                valid_indices.append(i)

            if not raw_currents:
                logger.error("No valid curves after truncation. Aborting."); return False

            per_curve_isc, norm_curves = zip(*[normalize_by_isc(c) for c in raw_currents])
            y_padded, v_padded, mask_matrix, lengths = pad_and_create_mask(list(norm_curves), raw_voltages)

            scalar_df = pd.DataFrame({
                'Isc_raw': [c[0] for c in raw_currents],
                'Vknee_raw': [v[-1] for v in raw_voltages],
                'Imax_raw': [np.max(c) for c in raw_currents],
                'Imin_raw': [np.min(c) for c in raw_currents],
                'Imean_raw': [np.mean(c) for c in raw_currents]
            })
            inlier_mask = remove_outliers_via_isolation_forest(scalar_df)

            params_df_valid = params_df.iloc[valid_indices].reset_index(drop=True)
            X_params = preprocess_input_parameters(params_df_valid[inlier_mask])
            X_scalar, self.scalar_scaler = preprocess_scalar_features(scalar_df[inlier_mask], fit=True)
            self.X_clean = np.concatenate([X_params, X_scalar], axis=1)

            self.y_clean_norm_padded = y_padded[inlier_mask]
            self.padded_voltages = v_padded[inlier_mask]
            self.masks = mask_matrix[inlier_mask]
            self.per_curve_isc = np.array(per_curve_isc, dtype=np.float32)[inlier_mask]
            self.orig_lengths = lengths[inlier_mask]

            logger.info(f"Final cleaned data shapes: X={self.X_clean.shape}, y={self.y_clean_norm_padded.shape}")
            return True
        except Exception as e:
            logger.exception(f"Error during data loading/preparation: {e}")
            return False

    def fit_physics_informed_nn(self, X_train, y_train, V_train, M_train, L_train, X_val, y_val, V_val, M_val, L_val):
        """
        Fit the physics‐informed neural network.
        """
        logger.info("=== Fitting Physics‐Informed Neural Network ===")
        input_dim_params, seq_len = X_train.shape[1], self.y_clean_norm_padded.shape[1]
        if seq_len == 0: raise RuntimeError("NN output dimension is zero. Aborting training.")

        nn_core = build_nn_core(input_dim_params, seq_len, voltage_embed_dim=VOLTAGE_EMBED_DIM)
        self.nn_model = PhysicsNNModel(nn_core, LOSS_WEIGHTS)
        self.nn_model.compile(optimizer=Adam(learning_rate=NN_LEARNING_RATE))

        train_ds = tf.data.Dataset.from_tensor_slices((
            {"X_params": X_train, "voltage_grid": V_train},
            {"y_true": y_train, "mask": M_train, "orig_len": L_train}
        )).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((
            {"X_params": X_val, "voltage_grid": V_val},
            {"y_true": y_val, "mask": M_val, "orig_len": L_val}
        )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1)
        ]
        self.training_history = self.nn_model.fit(train_ds, epochs=NN_EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)
        logger.info("Physics‐informed NN training completed.")

    def predict(self, X: np.ndarray, V_padded: np.ndarray, per_curve_isc: np.ndarray, orig_lengths: np.ndarray) -> list:
        """
        Predict truncated physical I–V curves (un‐normalized) for input X.
        """
        if self.nn_model is None: raise RuntimeError("NN model not fitted.")
        inputs = {"X_params": X, "voltage_grid": V_padded}
        y_pred_norm_padded = self.nn_model.predict(inputs, batch_size=512, verbose=0)

        y_pred_list = []
        for i in range(X.shape[0]):
            predicted_curve = y_pred_norm_padded[i] * per_curve_isc[i]
            y_pred_list.append(predicted_curve[:orig_lengths[i]].astype(np.float32))
        return y_pred_list

    def evaluate_model(self, X, V_padded, y_true_curves, per_curve_isc, orig_lengths) -> dict:
        """
        Evaluate the NN model and return a dictionary of metrics.
        """
        logger.info(f"=== Evaluating Model ===")
        y_pred_list = self.predict(X, V_padded, per_curve_isc, orig_lengths)

        flat_true, flat_pred, per_curve_r2 = [], [], []
        for true_curve, pred_curve in zip(y_true_curves, y_pred_list):
            if true_curve.size == 0: continue
            flat_true.extend(true_curve)
            flat_pred.extend(pred_curve)
            if true_curve.size > 1 and np.std(true_curve) > 1e-9:
                per_curve_r2.append(r2_score(true_curve, pred_curve))

        flat_true, flat_pred = np.array(flat_true), np.array(flat_pred)
        if flat_true.size == 0: raise RuntimeError("No valid points found for evaluation.")

        mae = mean_absolute_error(flat_true, flat_pred)
        rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
        r2_global = r2_score(flat_true, flat_pred)
        mean_r2, std_r2 = (np.mean(per_curve_r2), np.std(per_curve_r2)) if per_curve_r2 else (np.nan, np.nan)

        logger.info(f"  Metrics: MAE={mae:.6f}, RMSE={rmse:.6f}, Global R²={r2_global:.4f}")
        logger.info(f"  Per‐curve R²: Mean={mean_r2:.4f}, Std={std_r2:.4f}")

        return {'MAE': mae, 'RMSE': rmse, 'R2': r2_global, 'per_curve_R2_mean': mean_r2, 'per_curve_R2_std': std_r2}

    def plot_results(self, X, V_padded, y_true_curves, y_true_voltages, per_curve_isc, orig_lengths, n_samples=4, suffix=""):
        """
        Plot a random selection of true vs. predicted I–V curves.
        """
        y_pred_list = self.predict(X, V_padded, per_curve_isc, orig_lengths)
        N = len(y_true_curves)
        if N == 0: logger.warning("No samples to plot."); return

        indices = np.random.choice(N, size=min(n_samples, N), replace=False)
        nrows, ncols = (n_samples + 1) // 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
        axes = axes.flatten()

        for i_plot, idx in enumerate(indices):
            ax = axes[i_plot]
            true_v, true_i = y_true_voltages[idx], y_true_curves[idx]
            pred_i = y_pred_list[idx]
            r2_val = r2_score(true_i, pred_i) if true_i.size > 1 and np.std(true_i) > 1e-9 else np.nan

            ax.plot(true_v, true_i, 'b-', lw=2, label='Actual')
            ax.plot(true_v, pred_i, 'r--', lw=2, label='Predicted')
            ax.set_title(f"Sample {idx} (R²={r2_val:.3f})")
            ax.set_xlabel("Voltage (V)"), ax.set_ylabel("Current Density (mA/cm²)"), ax.legend(), ax.grid(True, alpha=0.3)

        for j in range(len(indices), len(axes)): fig.delaxes(axes[j])
        plt.tight_layout()
        outpath = OUTPUT_DIR / f"preds_plot{suffix}.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Prediction plot saved: {outpath}")

# ───────────────────────────────────────────────────────────────────────────────
#  Main Execution
# ───────────────────────────────────────────────────────────────────────────────

def run_experiment(trunc_thresh_pct: float = 0.1, use_gpu: bool = True):
    """
    Executes a full experiment run: data loading, training, evaluation, and plotting.
    """
    logger.info(f"=== STARTING EXPERIMENT: Truncation={trunc_thresh_pct}, GPU={use_gpu} ===")
    recon = TruncatedIVReconstructor(use_gpu_if_available=use_gpu)
    if not recon.load_and_prepare_data(truncation_threshold_pct=trunc_thresh_pct):
        logger.error("Data preparation failed. Aborting experiment.")
        return

    # --- Data Splitting ---
    all_idx = np.arange(recon.X_clean.shape[0])
    train_val_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=42) # 0.15 * 0.8 = 12% of total

    # --- Training Data ---
    X_train, y_train, V_train, M_train, L_train = (recon.X_clean[train_idx], recon.y_clean_norm_padded[train_idx],
        recon.padded_voltages[train_idx], recon.masks[train_idx], recon.orig_lengths[train_idx])

    # --- Validation Data ---
    X_val, y_val, V_val, M_val, L_val = (recon.X_clean[val_idx], recon.y_clean_norm_padded[val_idx],
        recon.padded_voltages[val_idx], recon.masks[val_idx], recon.orig_lengths[val_idx])

    # --- NN Training ---
    recon.fit_physics_informed_nn(X_train, y_train, V_train, M_train, L_train, X_val, y_val, V_val, M_val, L_val)

    # --- Plot Training History ---
    if recon.training_history:
        hist = recon.training_history.history
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist['loss'], label='Train Loss')
        ax.plot(hist['val_loss'], label='Val Loss', linestyle='--')
        ax.set_xlabel("Epoch"), ax.set_ylabel("Loss"), ax.set_title("Training & Validation Loss")
        ax.legend(), ax.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / "training_history.png", dpi=300)
        plt.close(fig)

    # --- Evaluation ---
    def get_true_curves(indices, lengths):
        isc_vals = recon.per_curve_isc[indices]
        voltages, currents = [], []
        for i, master_idx in enumerate(indices):
            L = lengths[i]
            currents.append(recon.y_clean_norm_padded[master_idx][:L] * isc_vals[i])
            voltages.append(recon.padded_voltages[master_idx][:L])
        return currents, voltages

    test_currents, test_voltages = get_true_curves(test_idx, recon.orig_lengths[test_idx])
    test_metrics = recon.evaluate_model(recon.X_clean[test_idx], recon.padded_voltages[test_idx],
                                        test_currents, recon.per_curve_isc[test_idx], recon.orig_lengths[test_idx])

    recon.plot_results(recon.X_clean[test_idx], recon.padded_voltages[test_idx],
                       test_currents, test_voltages, recon.per_curve_isc[test_idx], recon.orig_lengths[test_idx],
                       n_samples=8, suffix="_test_set")

    # --- Save Summary ---
    summary = {'config_threshold': trunc_thresh_pct, **{f'test_{k}': v for k, v in test_metrics.items()}}
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)
    logger.info(f"Experiment finished. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Ensure input files exist or point to the correct location before running.
    # This example call runs the full pipeline with a 10% truncation threshold.
    if not Path(INPUT_FILE_PARAMS).exists() or not Path(INPUT_FILE_IV).exists():
        logger.error("="*80)
        logger.error("Input data files not found!")
        logger.error(f"Please place '{Path(INPUT_FILE_PARAMS).name}' and '{Path(INPUT_FILE_IV).name}' in the current directory,")
        logger.error(f"or update the INPUT_FILE_PARAMS and INPUT_FILE_IV constants in the script.")
        logger.error("="*80)
    else:
        run_experiment(trunc_thresh_pct=0.10, use_gpu=True)