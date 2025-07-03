#!/usr/bin/env python3
"""
This is the final code before the method was switched to interpolation.
Hyperparameter optimization was performed using the `optuna` library. The HPO code is not in the repo.

Note from early June: A lean, focused implementation for Perovskite I–V curve reconstruction using a
physics‐informed neural network. This version focuses exclusively on the NN
approach, removing PCA and ensemble methods for clarity and simplicity.

Key features:
- Single‐responsibility utility functions for preprocessing (truncation, padding).
- Uniform [-1, 1] scaling for all inputs and outputs for improved convergence.
- A sophisticated physics‐informed loss function including:
  - Knee‐weighted Mean Squared Error (MSE).
  - Penalties for monotonicity, curvature, J_sc, and V_oc, adapted for [-1, 1] scale.
- Fourier Feature encoding for voltage embedding with a TCN architecture.
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Add, Lambda, TimeDistributed, Concatenate,
    Conv1D, SpatialDropout1D  # ### CHANGE 4: Added Conv1D and SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay # ### CHANGE 5: Added CosineDecay
from tensorflow.keras.callbacks import EarlyStopping

import typing

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
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
v1 = np.arange(0.0, 0.4 + 1e-8, 0.1)           # [0.0, 0.1, …, 0.4]
v2 = np.arange(0.425, 1.4 + 1e-8, 0.025)      # [0.425, 0.450, …, 1.4]

# Truncation hyperparameters
MIN_LEN_FOR_PROCESSING = 5         # Minimum points required after truncation

# Tuned loss weights and window parameters
LOSS_WEIGHTS = {
    'mse': 0.1803342607307616,         # updated
    'monotonicity': 0.0008268117,      # keep
    'curvature':    0.0030637258,      # keep
    'jsc': 0.041028679818554695,       # updated
    'voc': 0.1632580476                # keep
}
KNEE_WEIGHT_FACTOR = 2.1355485479     # keep
KNEE_WINDOW_SIZE   = 4                # keep
RANDOM_SEED = 42

# Neural network architecture (tuned hyperparameters)
FOURIER_NUM_BANDS       = 8           # was 16
DENSE_UNITS_PARAMS      = [256, 128, 128]  # unchanged
TCN_FILTERS             = [256, 32]   # second layer down from 64 → 32
TCN_KERNEL_SIZE         = 7           # was 5
DROPOUT_RATE            = 0.17623268240751816  # was 0.2801

# Training hyperparameters
NN_INITIAL_LEARNING_RATE = 0.004277022251867502  # was 0.0020778887
NN_FINAL_LEARNING_RATE   = 5.3622e-06            # unchanged
NN_EPOCHS                = 70                    # unchanged
BATCH_SIZE               = 64                    # was 128

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
) -> tuple[typing.Optional[np.ndarray], typing.Optional[np.ndarray]]:
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


def normalize_and_scale_by_isc(curve_trunc: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Normalize a truncated current array by its I_sc (first value) to [0, 1],
    then scale it to the [-1, 1] range.
    Returns (isc_value, scaled_array).
    Assumes curve_trunc[0] > 0.
    """
    isc_val = float(curve_trunc[0])
    if isc_val <= 0:
        # If I_sc is zero or negative, return unscaled and unnormalized
        return 1.0, curve_trunc.copy().astype(np.float32)
    norm_curve = curve_trunc / isc_val
    scaled_curve = 2.0 * norm_curve - 1.0 # Scale from [0, 1] to [-1, 1]
    return isc_val, scaled_curve.astype(np.float32)


def pad_and_create_mask(
    scaled_curves: list,
    volt_curves: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a list of [-1, 1] scaled current arrays and corresponding voltage arrays
    (both truncated), pad each to the maximum length among them.
    Returns:
      - y_padded: (N, max_len) float32, scaled currents, padded by last valid value.
      - v_padded: (N, max_len) float32, voltages, padded by last valid voltage.
      - mask:     (N, max_len) float32, 1.0 for valid points, 0.0 for padded.
      - lengths:  (N,)       int, original lengths of each truncated curve.
    """
    lengths = [c.size for c in scaled_curves]
    max_len = max(lengths) if lengths else 0
    n_samples = len(scaled_curves)

    y_padded = np.zeros((n_samples, max_len), dtype=np.float32)
    v_padded = np.zeros((n_samples, max_len), dtype=np.float32)
    mask     = np.zeros((n_samples, max_len), dtype=np.float32)

    for i, (c, v) in enumerate(zip(scaled_curves, volt_curves)):
        L = c.size
        y_padded[i, :L] = c
        v_padded[i, :L] = v
        mask[i, :L]     = 1.0
        if L < max_len:
            y_padded[i, L:] = c[-1]
            v_padded[i, L:] = v[-1]

    return y_padded, v_padded, mask, np.array(lengths, dtype=np.int32)


### CHANGE 2: RANDOM FOURIER FEATURES ###
B = tf.constant(np.logspace(0, 3, num=FOURIER_NUM_BANDS), dtype=tf.float32)

def fourier_features(V_norm, B_matrix=B):
    """
    Generate random Fourier features for normalized voltage grid V_norm.
    V_norm shape: (batch_size, seq_len), values ∈ [0, 1].
    Returns: (batch_size, seq_len, 2 * FOURIER_NUM_BANDS) tensor.
    """
    # V_norm: [batch, seq] -> V: [batch, seq, num_bands]
    V = V_norm[..., None] * B_matrix[None, None, :]
    # Result: [batch, seq, 2 * num_bands]
    return tf.concat([tf.sin(2 * np.pi * V), tf.cos(2 * np.pi * V)], axis=-1)


### CHANGE 3: MASKED LOSS WINDOW ###
def masked_loss_window(mask: tf.Tensor, orig_len: tf.Tensor, window: int = KNEE_WINDOW_SIZE) -> tf.Tensor:
    """
    Creates a mask that is 1.0 only for points within +/- window of the MPP knee.
    The MPP knee is assumed to be at the last valid point (orig_len - 1).
    """
    idx = tf.range(tf.shape(mask)[1])[None, :]  # Shape: [1, seq_len]
    mpp_idx = tf.reshape(orig_len - 1, (-1, 1)) # Shape: [batch_size, 1]

    # Create a boolean mask for the window around the MPP
    window_mask = tf.logical_and(idx >= mpp_idx - window, idx <= mpp_idx + window)
    return tf.cast(window_mask, tf.float32)


def masked_mse_with_knee_weight(
    y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor, orig_len: tf.Tensor
) -> tf.Tensor:
    """
    Compute MSE with additional weighting on points inside the knee window.
    """
    y_true, y_pred, mask = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), tf.cast(mask, tf.float32)
    se = tf.square(y_true - y_pred)

    # ### CHANGE 3: Use windowed mask for knee weighting ###
    w = masked_loss_window(mask, orig_len, window=KNEE_WINDOW_SIZE)
    # Apply base weight of 1.0 everywhere, and add (FACTOR-1.0) inside the window
    knee_weights = 1.0 + (KNEE_WEIGHT_FACTOR - 1.0) * w

    weighted_se = se * mask * knee_weights
    mse_per_sample = tf.reduce_sum(weighted_se, axis=-1) / (tf.reduce_sum(mask * knee_weights, axis=-1) + 1e-7)
    return tf.reduce_mean(mse_per_sample)


def monotonicity_penalty_loss(y_pred: tf.Tensor, mask: tf.Tensor, orig_len: tf.Tensor) -> tf.Tensor:
    """
    Penalize I_{i+1} > I_i, but only within the specified knee window.
    """
    y_pred, mask = tf.cast(y_pred, tf.float32), tf.cast(mask, tf.float32)
    diffs = y_pred[:, 1:] - y_pred[:, :-1]

    # Base mask for valid differences
    diff_mask = mask[:, 1:] * mask[:, :-1]

    # ### CHANGE 3: Localize penalty to the knee window ###
    window_w = masked_loss_window(mask, orig_len, window=KNEE_WINDOW_SIZE)
    # Ensure the window mask applies to the difference pairs
    window_mask_for_diffs = window_w[:, :-1] * window_w[:, 1:]

    final_mask = diff_mask * window_mask_for_diffs

    violations = tf.nn.relu(diffs) * final_mask
    sum_violation = tf.reduce_sum(tf.square(violations), axis=-1)
    return tf.reduce_mean(sum_violation / (tf.reduce_sum(final_mask, axis=-1) + 1e-7))


def curvature_penalty_loss(y_pred: tf.Tensor, mask: tf.Tensor, orig_len: tf.Tensor) -> tf.Tensor:
    """
    Penalize large second derivatives, but only within the specified knee window.
    """
    y_pred, mask = tf.cast(y_pred, tf.float32), tf.cast(mask, tf.float32)
    curvature = y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2]

    # Base mask for valid curvature points
    curv_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]

    # ### CHANGE 3: Localize penalty to the knee window ###
    window_w = masked_loss_window(mask, orig_len, window=KNEE_WINDOW_SIZE)
    # Ensure window mask applies to the 3-point stencil
    window_mask_for_curv = window_w[:, :-2] * window_w[:, 1:-1] * window_w[:, 2:]

    final_mask = curv_mask * window_mask_for_curv

    weighted_curv = tf.square(curvature) * final_mask
    return tf.reduce_mean(tf.reduce_sum(weighted_curv, axis=-1) / (tf.reduce_sum(final_mask, axis=-1) + 1e-7))


def jsc_voc_penalty_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor, orig_len: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute two penalties: J_sc MSE at V=0 and V_oc MSE (current at last point should be -1.0).
    """
    y_true, y_pred, mask = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), tf.cast(mask, tf.float32)
    orig_l = tf.cast(orig_len, tf.int32)

    # J_sc MSE at index 0. Target is y_true[:, 0], which should be 1.0.
    jsc_mse = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]) * mask[:, 0])

    # V_oc penalty: current at the last valid index should be -1.0 (scaled from 0).
    batch_size = tf.shape(y_pred)[0]
    indices = tf.range(batch_size, dtype=tf.int32)
    last_indices = tf.clip_by_value(orig_l - 1, 0, tf.shape(y_pred)[1] - 1)
    gather_idx = tf.stack([indices, last_indices], axis=1)
    last_pred = tf.gather_nd(y_pred, gather_idx)

    # The target value for the last point is -1.0
    voc_target = -1.0
    voc_mask = tf.cast(orig_l > 0, tf.float32)
    voc_mse = tf.reduce_sum(tf.square(last_pred - voc_target) * voc_mask) / (tf.reduce_sum(voc_mask) + 1e-7)
    return jsc_mse, voc_mse


def get_all_loss_components(y_true_combined, y_pred):
    """Helper to compute all loss components for logging."""
    y_true, mask, orig_l = y_true_combined['y_true'], y_true_combined['mask'], y_true_combined['orig_len']
    mse_part = masked_mse_with_knee_weight(y_true, y_pred, mask, orig_l)
    # ### CHANGE 3: Pass orig_len to mono and curv losses ###
    mono_part = monotonicity_penalty_loss(y_pred, mask, orig_l)
    curv_part = curvature_penalty_loss(y_pred, mask, orig_l)
    jsc_part, voc_part = jsc_voc_penalty_loss(y_true, y_pred, mask, orig_l)
    return mse_part, mono_part, curv_part, jsc_part, voc_part


def combined_physics_loss(loss_weights: dict):
    """Factory function for a combined physics-informed loss."""
    def loss_fn(y_true_combined, y_pred):
        mse, mono, curv, jsc, voc = get_all_loss_components(y_true_combined, y_pred)
        return (loss_weights['mse'] * mse +
                loss_weights['monotonicity'] * mono +
                loss_weights['curvature'] * curv +
                loss_weights['jsc'] * jsc +
                loss_weights['voc'] * voc)
    return loss_fn

# ───────────────────────────────────────────────────────────────────────────────
#  Preprocessing: Input Parameters & Scalar Features
# ───────────────────────────────────────────────────────────────────────────────

def preprocess_input_parameters(params_df: pd.DataFrame) -> tuple[np.ndarray, ColumnTransformer]:
    """
    Transform device/material parameters with group‐wise pipelines, scaling to [-1, 1].
    Returns the processed data and the fitted transformer.
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
        steps = []
        if group == 'material':
             steps.append(('log1p', FunctionTransformer(func=np.log1p, validate=False)))
        steps.extend([
            ('robust', RobustScaler()),
            ('minmax', MinMaxScaler(feature_range=(-1, 1)))
        ])
        transformers.append((group, Pipeline(steps), actual_cols))

    column_transformer = ColumnTransformer(transformers, remainder='passthrough')
    X_processed = column_transformer.fit_transform(params_df)
    logger.info(f"  Processed param features shape: {X_processed.shape}")
    return X_processed.astype(np.float32), column_transformer


def preprocess_scalar_features(scalar_features_df: pd.DataFrame, fit: bool = True,
                               scaler: Pipeline = None) -> tuple[np.ndarray, Pipeline]:
    """
    Scale scalar features derived from the I-V curves to [-1, 1].
    """
    logger.info(f"Preprocessing scalar features: {list(scalar_features_df.columns)}")
    if fit:
        scaler = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])
        X_scaled = scaler.fit_transform(scalar_features_df.values)
    else:
        if scaler is None: raise RuntimeError("Scaler not provided for transform.")
        X_scaled = scaler.transform(scalar_features_df.values)
    logger.info(f"  Processed scalar features shape: {X_scaled.shape}")
    return X_scaled.astype(np.float32), scaler

# ───────────────────────────────────────────────────────────────────────────────
#  Neural Network Definition
# ───────────────────────────────────────────────────────────────────────────────

def build_nn_core(input_dim_params: int, seq_len: int) -> Model:
    """
    Build the core Keras Model for the physics‐informed NN.
    This version uses Fourier Features and a TCN head.
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
    # ### CHANGE 2: Use Fourier Features for voltage embedding ###
    norm_voltage = Lambda(lambda v: v / 1.4)(voltage_grid_in) # Normalize voltage to [0, 1]
    v_embed = Lambda(fourier_features)(norm_voltage)

    # Merge paths
    param_tiled = Lambda(lambda t: tf.tile(tf.expand_dims(t, 1), [1, seq_len, 1]))(param_path)
    x = Concatenate(axis=-1)([param_tiled, v_embed])

    # ### CHANGE 4: Replace TimeDistributed MLP with a TCN stack ###
    x = Conv1D(TCN_FILTERS[1], kernel_size=TCN_KERNEL_SIZE, padding='same', activation='relu')(x)
    x = SpatialDropout1D(DROPOUT_RATE)(x)

    # ### CHANGE 1 & 4: Final projection with linear activation ###
    iv_output = Conv1D(1, kernel_size=1, padding='same', activation=None)(x)
    iv_output_flat = Lambda(lambda t: tf.squeeze(t, axis=-1))(iv_output)

    return Model(inputs=[x_params_in, voltage_grid_in], outputs=iv_output_flat, name="NN_Core")


class PhysicsNNModel(keras.Model):
    """
    Custom Keras Model wrapper to integrate the core NN with the physics-informed loss
    and detailed metric tracking.
    """
    def __init__(self, nn_core: Model, loss_weights: dict, **kwargs):
        super().__init__(**kwargs)
        self.nn_core = nn_core
        self.loss_weights = loss_weights
        self.custom_loss_fn = combined_physics_loss(loss_weights)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        self.mono_loss_tracker = keras.metrics.Mean(name="mono_loss")
        self.curv_loss_tracker = keras.metrics.Mean(name="curv_loss")
        self.jsc_loss_tracker = keras.metrics.Mean(name="jsc_loss")
        self.voc_loss_tracker = keras.metrics.Mean(name="voc_loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae_scaled") # MAE on scaled data

    @property
    def metrics(self):
        # Return all trackers to be displayed by Keras
        return [
            self.loss_tracker, self.mse_loss_tracker, self.mono_loss_tracker,
            self.curv_loss_tracker, self.jsc_loss_tracker, self.voc_loss_tracker,
            self.mae_metric
        ]

    def call(self, inputs, training=False):
        return self.nn_core(inputs, training=training)

    def train_step(self, data):
        inputs, y_true_combined = data
        with tf.GradientTape() as tape:
            y_pred = self.nn_core(inputs, training=True)
            mse, mono, curv, jsc, voc = get_all_loss_components(y_true_combined, y_pred)
            total_loss = (self.loss_weights['mse'] * mse +
                          self.loss_weights['monotonicity'] * mono +
                          self.loss_weights['curvature'] * curv +
                          self.loss_weights['jsc'] * jsc +
                          self.loss_weights['voc'] * voc)

        grads = tape.gradient(total_loss, self.nn_core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn_core.trainable_variables))
        y_true = y_true_combined['y_true']
        mask   = y_true_combined['mask']
        y_pred = y_pred

        bool_mask = tf.cast(mask, tf.bool)
        y_true_masked = tf.boolean_mask(y_true, bool_mask)
        y_pred_masked = tf.boolean_mask(y_pred, bool_mask)

        # Update all metric trackers
        self.loss_tracker.update_state(total_loss)
        self.mse_loss_tracker.update_state(mse)
        self.mono_loss_tracker.update_state(mono)
        self.curv_loss_tracker.update_state(curv)
        self.jsc_loss_tracker.update_state(jsc)
        self.voc_loss_tracker.update_state(voc)
        self.mae_metric.update_state(y_true_masked, y_pred_masked)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, y_true_combined = data
        y_pred = self.nn_core(inputs, training=False)

        mse, mono, curv, jsc, voc = get_all_loss_components(y_true_combined, y_pred)
        total_loss = (self.loss_weights['mse'] * mse +
                      self.loss_weights['monotonicity'] * mono +
                      self.loss_weights['curvature'] * curv +
                      self.loss_weights['jsc'] * jsc +
                      self.loss_weights['voc'] * voc)

        # Update all metric trackers
        self.loss_tracker.update_state(total_loss)
        self.mse_loss_tracker.update_state(mse)
        self.mono_loss_tracker.update_state(mono)
        self.curv_loss_tracker.update_state(curv)
        self.jsc_loss_tracker.update_state(jsc)
        self.voc_loss_tracker.update_state(voc)
        y_true = y_true_combined['y_true']
        y_pred = y_pred
        mask   = y_true_combined['mask']

        # select only valid positions
        bool_mask    = tf.cast(mask, tf.bool)
        y_true_valid = tf.boolean_mask(y_true, bool_mask)
        y_pred_valid = tf.boolean_mask(y_pred, bool_mask)

        self.mae_metric.update_state(y_true_valid, y_pred_valid)

        return {m.name: m.result() for m in self.metrics}

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

        self.X_clean = self.y_clean_scaled_padded = self.padded_voltages = None
        self.masks = self.per_curve_isc = self.orig_lengths = None
        self.param_transformer = self.scalar_scaler = None
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
            full_voltage_grid = np.concatenate([v1, v2]).astype(np.float32)
            assert full_voltage_grid.size == 45

            raw_currents, raw_voltages, valid_indices = [], [], []
            for i in range(iv_data_np.shape[0]):
                v_trunc, c_trunc = truncate_iv_curve(iv_data_np[i], full_voltage_grid, truncation_threshold_pct)
                if v_trunc is None: continue
                raw_voltages.append(v_trunc)
                raw_currents.append(c_trunc)
                valid_indices.append(i)

            if not raw_currents:
                logger.error("No valid curves after truncation. Aborting."); return False

            per_curve_isc, scaled_curves = zip(*[normalize_and_scale_by_isc(c) for c in raw_currents])
            y_padded, v_padded, mask_matrix, lengths = pad_and_create_mask(list(scaled_curves), raw_voltages)

            params_df_valid = params_df.iloc[valid_indices].reset_index(drop=True)

            scalar_df = pd.DataFrame({
                'Isc_raw': [c[0] for c in raw_currents],
                'Vknee_raw': [v[-1] for v in raw_voltages],
                'Imax_raw': [np.max(c) for c in raw_currents],
                'Imin_raw': [np.min(c) for c in raw_currents],
                'Imean_raw': [np.mean(c) for c in raw_currents]
            })

            X_params, self.param_transformer = preprocess_input_parameters(params_df_valid)
            X_scalar, self.scalar_scaler = preprocess_scalar_features(scalar_df, fit=True)
            self.X_clean = np.concatenate([X_params, X_scalar], axis=1)

            self.y_clean_scaled_padded = y_padded
            self.padded_voltages = v_padded
            self.masks = mask_matrix
            self.per_curve_isc = np.array(per_curve_isc, dtype=np.float32)
            self.orig_lengths = lengths

            logger.info(f"Final cleaned data shapes: X={self.X_clean.shape}, y={self.y_clean_scaled_padded.shape}")
            return True
        except Exception as e:
            logger.exception(f"Error during data loading/preparation: {e}")
            return False

    def fit_physics_informed_nn(self, X_train, y_train, V_train, M_train, L_train, X_val, y_val, V_val, M_val, L_val):
        """
        Fit the physics‐informed neural network.
        """
        logger.info("=== Fitting Physics‐Informed Neural Network ===")
        input_dim_params, seq_len = X_train.shape[1], self.y_clean_scaled_padded.shape[1]
        if seq_len == 0: raise RuntimeError("NN output dimension is zero. Aborting training.")

        nn_core = build_nn_core(input_dim_params, seq_len)
        self.nn_model = PhysicsNNModel(nn_core, LOSS_WEIGHTS)

        # ### CHANGE 5: Implement CosineDecay learning rate schedule ###
        steps_per_epoch = X_train.shape[0] // BATCH_SIZE
        total_steps     = steps_per_epoch * NN_EPOCHS
        cosine_lr_schedule = CosineDecay(
            initial_learning_rate=NN_INITIAL_LEARNING_RATE,  # 0.0020778887
            decay_steps=total_steps,
            alpha=NN_FINAL_LEARNING_RATE / NN_INITIAL_LEARNING_RATE  # 5.3622e-06 / 0.0020778887
        )
        optimizer = Adam(learning_rate=cosine_lr_schedule)

        self.nn_model.compile(optimizer=optimizer)

        train_ds = tf.data.Dataset.from_tensor_slices((
            {"X_params": X_train, "voltage_grid": V_train},
            {"y_true": y_train, "mask": M_train, "orig_len": L_train}
        )).shuffle(10000, seed=RANDOM_SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((
            {"X_params": X_val, "voltage_grid": V_val},
            {"y_true": y_val, "mask": M_val, "orig_len": L_val}
        )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # ### CHANGE 5: Removed ReduceLROnPlateau ###
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ]
        self.training_history = self.nn_model.fit(train_ds, epochs=NN_EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)
        logger.info("Physics‐informed NN training completed.")

    def predict(self, X: np.ndarray, V_padded: np.ndarray, per_curve_isc: np.ndarray, orig_lengths: np.ndarray) -> list:
        """
        Predict truncated physical I–V curves (un‐scaled) for input X.
        """
        if self.nn_model is None: raise RuntimeError("NN model not fitted.")
        inputs = {"X_params": X, "voltage_grid": V_padded}
        y_pred_scaled_padded = self.nn_model.predict(inputs, batch_size=512, verbose=0)

        y_pred_list = []
        for i in range(X.shape[0]):
            # Reverse the scaling: from [-1, 1] back to physical units.
            # This logic is still correct even with a linear output layer,
            # as the model is trained to predict values in the [-1, 1] space.
            y_scaled = y_pred_scaled_padded[i]
            y_norm = (y_scaled + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            predicted_curve = y_norm * per_curve_isc[i] # [0, 1] -> physical
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
        nrows, ncols = (n_samples + 3) // 4 * 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
        axes = axes.flatten()


        for i_plot, idx in enumerate(indices):
            ax = axes[i_plot]
            true_v, true_i = y_true_voltages[idx], y_true_curves[idx]
            pred_i = y_pred_list[idx]
            r2_val = r2_score(true_i, pred_i) if true_i.size > 1 and np.std(true_i) > 1e-9 else np.nan

            ax.plot(true_v, true_i, 'b-', lw=2, label='Actual')
            ax.plot(true_v, pred_i, 'r--', lw=2, label='Predicted')
            ax.set_title(f"Sample {idx} (R²={r2_val:.4f})")
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

def run_experiment(trunc_thresh_pct: float = 0.09991097180491529, use_gpu: bool = True):
    """
    Executes a full experiment run: data loading, training, evaluation, and plotting.
    """
    logger.info(f"=== STARTING EXPERIMENT: Truncation={trunc_thresh_pct}, GPU={use_gpu} ===")

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    recon = TruncatedIVReconstructor(use_gpu_if_available=use_gpu)
    if not recon.load_and_prepare_data(truncation_threshold_pct=trunc_thresh_pct):
        logger.error("Data preparation failed. Aborting experiment.")
        return

    # --- Data Splitting ---
    all_idx = np.arange(recon.X_clean.shape[0])
    train_val_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=RANDOM_SEED) # 0.15 * 0.8 = 12% of total

    # --- Training Data ---
    X_train, y_train, V_train, M_train, L_train = (recon.X_clean[train_idx], recon.y_clean_scaled_padded[train_idx],
        recon.padded_voltages[train_idx], recon.masks[train_idx], recon.orig_lengths[train_idx])

    # --- Validation Data ---
    X_val, y_val, V_val, M_val, L_val = (recon.X_clean[val_idx], recon.y_clean_scaled_padded[val_idx],
        recon.padded_voltages[val_idx], recon.masks[val_idx], recon.orig_lengths[val_idx])

    # --- NN Training ---
    recon.fit_physics_informed_nn(X_train, y_train, V_train, M_train, L_train, X_val, y_val, V_val, M_val, L_val)

    # --- Plot Training History ---
    if recon.training_history:
        hist_df = pd.DataFrame(recon.training_history.history)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(hist_df.index, hist_df['loss'], label='Train Total Loss')
        ax1.plot(hist_df.index, hist_df['val_loss'], label='Val Total Loss', linestyle='--')
        ax1.set_ylabel("Total Loss"), ax1.set_title("Training & Validation Loss"), ax1.legend(), ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        loss_cols = [c for c in hist_df.columns if c.endswith('_loss') and c not in ('loss','val_loss')]
        for col in loss_cols:
            ax2.plot(hist_df.index, hist_df[col], label=col.replace('_loss', ''))
        ax2.set_xlabel("Epoch"), ax2.set_ylabel("Loss Component Value"), ax2.set_title("Training Loss Components")
        ax2.legend(), ax2.grid(True, alpha=0.3), ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "training_history.png", dpi=300)
        plt.close(fig)

    # --- Evaluation ---
    def get_true_curves(indices, lengths):
        isc_vals = recon.per_curve_isc[indices]
        voltages, currents = [], []
        for i, master_idx in enumerate(indices):
            L = lengths[i]
            # De-scale the true curves from [-1, 1] back to physical units for evaluation
            y_scaled = recon.y_clean_scaled_padded[master_idx][:L]
            y_norm = (y_scaled + 1.0) / 2.0
            currents.append(y_norm * isc_vals[i])
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
    if not Path(INPUT_FILE_PARAMS).exists() or not Path(INPUT_FILE_IV).exists():
        logger.error("="*80)
        logger.error("Input data files not found!")
        logger.error(f"Please place '{Path(INPUT_FILE_PARAMS).name}' and '{Path(INPUT_FILE_IV).name}' in the current directory,")
        logger.error(f"or update the INPUT_FILE_PARAMS and INPUT_FILE_IV constants in the script.")
        logger.error("="*80)
    else:
        # Run with new defaults: 1% truncation threshold
        run_experiment(trunc_thresh_pct=0.01, use_gpu=True)

