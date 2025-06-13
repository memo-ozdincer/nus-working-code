#!/usr/bin/env python3
"""
refactored_iv_reconstruction.py

A lean, focused implementation for Perovskite I–V curve reconstruction using a
physics‐informed neural network. This version focuses exclusively on the NN
approach, removing PCA and ensemble methods for clarity and simplicity.

Key changes from original:
- All 31 input parameters are retained (no column dropping).
- I-V curves are preprocessed with PCHIP interpolation to extract a fixed-length
  sequence of 8 points centered around the Maximum Power Point (MPP). This
  eliminates the need for padding/masking and simplifies the model.
- Loss function adapted for fixed-length sequences (Jsc/Voc penalties removed).
- Static "knee" weighting replaced with dynamic, curvature-based weighting.
- Added a curvature regularization penalty to improve physical plausibility.

"""

import os
import logging
from pathlib import Path
from datetime import datetime
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

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
    Conv1D, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping

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
    datefmt="%Y-%m-%d %H%M%S",
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
#  Hyperparameters & Constants
# ───────────────────────────────────────────────────────────────────────────────
# File paths
INPUT_FILE_PARAMS = "/content/drive/MyDrive/Colab Notebooks/Data_100k/LHS_parameters_m.txt"
INPUT_FILE_IV     = "/content/drive/MyDrive/Colab Notebooks/Data_100k/iV_m.txt"

# Output directory with timestamp
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"./output_run_{RUN_ID}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

# Original I–V curve configuration
v1 = np.arange(0.0, 0.4 + 1e-8, 0.1)
v2 = np.arange(0.425, 1.4 + 1e-8, 0.025)

# PCHIP processing hyperparameters
PCHIP_FINE_GRID_POINTS = 10000
PCHIP_VOLTAGE_MAX = 1.4
NUM_PRE_MPP_POINTS = 3
NUM_POST_MPP_POINTS = 4
FIXED_SEQUENCE_LENGTH = NUM_PRE_MPP_POINTS + 1 + NUM_POST_MPP_POINTS

### MODIFIED ###: Replaced static knee weight with curvature-based parameters
# --- Improvement #1: Curvature-Magnitude Weighting Parameters ---
CURVATURE_WEIGHT_ALPHA = 4.0  # Controls how much extra weight is given to high-curvature points.
CURVATURE_WEIGHT_POWER = 1.5  # Exponent >= 1 to accentuate extreme curvature points.

# --- Improvement #4: Curvature Regularization Parameter ---
# Penalizes model predictions where curvature magnitude exceeds this threshold.
# This is applied to the scaled [-1, 1] data. A value around 0.5-1.0 is a good start.
EXCESS_CURVATURE_THRESHOLD = 0.8

# Updated loss weights for the new components
LOSS_WEIGHTS = {
    'mse': 0.98,
    'monotonicity': 0.005,      # Penalizes non-decreasing current
    'convexity': 0.005,         # Penalizes non-physical convex regions
    'excess_curvature': 0.01,   # Penalizes unrealistic sharp bends
}
RANDOM_SEED = 42

# Neural network architecture
FOURIER_NUM_BANDS       = 16
DENSE_UNITS_PARAMS      = [256, 128, 128]
TCN_FILTERS             = [256, 64]
TCN_KERNEL_SIZE         = 5
DROPOUT_RATE            = 0.2801020847

# Training hyperparameters
NN_INITIAL_LEARNING_RATE = 0.0020778887
NN_FINAL_LEARNING_RATE   = 5.3622e-06
NN_EPOCHS                = 70
BATCH_SIZE               = 128

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

def process_iv_with_pchip(
    curve_current_raw: np.ndarray,
    full_voltage_grid: np.ndarray
) -> typing.Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Processes a raw I-V curve using PCHIP interpolation to extract a fixed
    number of points around the Maximum Power Point (MPP).
    """
    try:
        interpolator = PchipInterpolator(full_voltage_grid, curve_current_raw, extrapolate=False)
        v_fine = np.linspace(0, PCHIP_VOLTAGE_MAX, PCHIP_FINE_GRID_POINTS)
        i_fine = interpolator(v_fine)
        valid_mask = ~np.isnan(i_fine)
        v_fine, i_fine = v_fine[valid_mask], i_fine[valid_mask]
        if v_fine.size < 2: return None
        zero_cross_idx = np.where(i_fine <= 0)[0]
        voc_v = v_fine[zero_cross_idx[0]] if len(zero_cross_idx) > 0 else v_fine[-1]
        voc_mask = v_fine <= voc_v
        v_search, i_search = v_fine[voc_mask], i_fine[voc_mask]
        if v_search.size == 0: return None
        power = v_search * i_search
        mpp_idx = np.argmax(power)
        v_mpp = v_search[mpp_idx]
        v_pre_mpp = np.linspace(v_search[0], v_mpp, NUM_PRE_MPP_POINTS + 2, endpoint=True)[:-1]
        v_post_mpp = np.linspace(v_mpp, v_search[-1], NUM_POST_MPP_POINTS + 2, endpoint=True)[1:]
        v_mpp_grid = np.unique(np.concatenate([v_pre_mpp, v_post_mpp]))
        v_mpp_grid_final = np.interp(
            np.linspace(0, 1, FIXED_SEQUENCE_LENGTH),
            np.linspace(0, 1, len(v_mpp_grid)),
            v_mpp_grid
        )
        i_mpp_slice = interpolator(v_mpp_grid_final)
        if np.any(np.isnan(i_mpp_slice)) or i_mpp_slice.shape[0] != FIXED_SEQUENCE_LENGTH:
            return None
        return v_mpp_grid_final.astype(np.float32), i_mpp_slice.astype(np.float32)

    except (ValueError, IndexError):
        return None

def normalize_and_scale_by_isc(curve: np.ndarray) -> tuple[float, np.ndarray]:
    isc_val = float(curve[0])
    if isc_val <= 0:
        return 1.0, (2.0 * curve.copy().astype(np.float32) - 1.0)
    norm_curve = curve / isc_val
    scaled_curve = 2.0 * norm_curve - 1.0
    return isc_val, scaled_curve.astype(np.float32)

# --- Fourier Features (unchanged) ---
B = tf.constant(np.logspace(0, 3, num=FOURIER_NUM_BANDS), dtype=tf.float32)
def fourier_features(V_norm, B_matrix=B):
    V = V_norm[..., None] * B_matrix[None, None, :]
    return tf.concat([tf.sin(2 * np.pi * V), tf.cos(2 * np.pi * V)], axis=-1)

### NEW ### --- Implementation of Improvement #1: Curvature-Magnitude Weighting ---
def compute_curvature_weights(y_curves: np.ndarray, alpha: float, power: float) -> np.ndarray:
    """
    Computes a sample weight mask based on the curvature of each I-V curve.
    This replaces the static knee weight with a dynamic, data-driven mask.
    
    1. Computes per-point curvature: kappa = |I(i+1) - 2*I(i) + I(i-1)|
    2. Normalizes kappa for each curve to lie in [0, 1].
    3. Forms the weight mask: w = 1 + alpha * normalized_kappa^power
    
    Args:
        y_curves (np.ndarray): Batch of I-V curves, shape (num_samples, seq_len).
        alpha (float): Controls how much to up-weight high-curvature points.
        power (float): Exponent to accentuate extreme points.
        
    Returns:
        np.ndarray: A weight mask of the same shape as y_curves.
    """
    # Pad curves on the left and right to handle boundaries for the 2nd derivative
    padded_curves = np.pad(y_curves, ((0, 0), (1, 1)), mode='edge')
    
    # 1. Compute curvature magnitude (finite second difference)
    kappa = np.abs(padded_curves[:, 2:] - 2 * padded_curves[:, 1:-1] + padded_curves[:, :-2])
    
    # 2. Normalize kappa per curve
    max_kappa = np.max(kappa, axis=1, keepdims=True)
    # Avoid division by zero for perfectly linear curves (rare)
    max_kappa[max_kappa < 1e-9] = 1.0
    norm_kappa = kappa / max_kappa
    
    # 3. Form the weight mask
    weights = 1.0 + alpha * np.power(norm_kappa, power)
    
    return weights.astype(np.float32)


### MODIFIED ### --- Loss Functions ---
# The static `mse_with_knee_weight` is removed. MSE is now calculated in the
# train step using the pre-computed curvature weights.

def monotonicity_penalty_loss(y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.cast(y_pred, tf.float32)
    diffs = y_pred[:, 1:] - y_pred[:, :-1]
    # Penalize positive differences (I should be non-increasing)
    violations = tf.nn.relu(diffs)
    return tf.reduce_mean(tf.square(violations))

### MODIFIED ###: Renamed from `curvature_penalty_loss` for clarity.
def convexity_penalty_loss(y_pred: tf.Tensor) -> tf.Tensor:
    """Penalizes non-physical convex regions in the I-V curve."""
    y_pred = tf.cast(y_pred, tf.float32)
    # Curvature (2nd derivative) should be negative (concave).
    curvature = y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2]
    # Penalize positive curvature values (convexity)
    violations = tf.nn.relu(curvature)
    return tf.reduce_mean(tf.square(violations))

### NEW ### --- Implementation of Improvement #4: Curvature Regularization Loss ---
def excess_curvature_penalty_loss(y_pred: tf.Tensor, threshold: float) -> tf.Tensor:
    """
    Penalizes the model for predicting curves with unrealistically sharp bends
    by regularizing the magnitude of the curvature.
    
    Loss = mean( (max(0, |kappa| - threshold))^2 )
    """
    y_pred = tf.cast(y_pred, tf.float32)
    # kappa = |I_{i+1} - 2I_i + I_{i-1}|
    kappa = tf.abs(y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2])
    
    # Penalize curvature magnitude that exceeds the allowed physical threshold
    violations = tf.nn.relu(kappa - threshold)
    return tf.reduce_mean(tf.square(violations))


### MODIFIED ###
def get_all_loss_components(y_true, y_pred, sample_weight):
    """Calculates all individual loss components."""
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    sample_weight = tf.cast(sample_weight, dtype=tf.float32)
    
    # 1. Weighted MSE (using pre-computed curvature weights)
    se = tf.square(y_true - y_pred)
    weighted_se = se * sample_weight
    mse_part = tf.reduce_mean(weighted_se)
    
    # 2. Monotonicity Penalty
    mono_part = monotonicity_penalty_loss(y_pred)
    
    # 3. Convexity Penalty
    convex_part = convexity_penalty_loss(y_pred)
    
    # 4. Excess Curvature Regularization
    excess_curv_part = excess_curvature_penalty_loss(y_pred, EXCESS_CURVATURE_THRESHOLD)
    
    return mse_part, mono_part, convex_part, excess_curv_part

# The `combined_physics_loss` function is no longer needed as the logic
# is handled directly inside the custom train_step.

# ───────────────────────────────────────────────────────────────────────────────
#  Preprocessing: Input Parameters & Scalar Features
# ───────────────────────────────────────────────────────────────────────────────
def preprocess_input_parameters(params_df: pd.DataFrame) -> tuple[np.ndarray, ColumnTransformer]:
    logger.info("Preprocessing input parameters (device, material, etc.)...")
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
        steps.extend([('robust', RobustScaler()), ('minmax', MinMaxScaler(feature_range=(-1, 1)))])
        transformers.append((group, Pipeline(steps), actual_cols))

    column_transformer = ColumnTransformer(transformers, remainder='passthrough')
    X_processed = column_transformer.fit_transform(params_df)
    logger.info(f"  Processed param features shape: {X_processed.shape}")
    return X_processed.astype(np.float32), column_transformer

def preprocess_scalar_features(scalar_features_df: pd.DataFrame, fit: bool = True, scaler: Pipeline = None) -> tuple[np.ndarray, Pipeline]:
    logger.info(f"Preprocessing scalar features: {list(scalar_features_df.columns)}")
    if fit:
        scaler = Pipeline([('scaler', MinMaxScaler(feature_range=(-1, 1)))])
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
    x_params_in = Input(shape=(input_dim_params,), name="X_params")
    x = x_params_in
    for i, units in enumerate(DENSE_UNITS_PARAMS):
        x = Dense(units, activation='relu', name=f"param_dense{i+1}")(x)
        x = BatchNormalization(name=f"param_bn{i+1}")(x)
        if i == 0: x = Dropout(DROPOUT_RATE, name=f"param_do{i+1}")(x)
    param_path = x

    voltage_grid_in = Input(shape=(seq_len,), name="voltage_grid")
    norm_voltage = Lambda(lambda v: v / PCHIP_VOLTAGE_MAX)(voltage_grid_in)
    v_embed = Lambda(fourier_features)(norm_voltage)

    param_tiled = Lambda(lambda t: tf.tile(tf.expand_dims(t, 1), [1, seq_len, 1]))(param_path)
    x = Concatenate(axis=-1)([param_tiled, v_embed])

    x = Conv1D(TCN_FILTERS[1], kernel_size=TCN_KERNEL_SIZE, padding='same', activation='relu')(x)
    x = SpatialDropout1D(DROPOUT_RATE)(x)

    iv_output = Conv1D(1, kernel_size=1, padding='same', activation=None)(x)
    iv_output_flat = Lambda(lambda t: tf.squeeze(t, axis=-1))(iv_output)

    return Model(inputs=[x_params_in, voltage_grid_in], outputs=iv_output_flat, name="NN_Core")

### MODIFIED ###: Updated custom model to handle new loss structure
class PhysicsNNModel(keras.Model):
    def __init__(self, nn_core: Model, loss_weights: dict, **kwargs):
        super().__init__(**kwargs)
        self.nn_core = nn_core
        self.loss_weights = loss_weights
        # Updated loss trackers for the new components
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        self.mono_loss_tracker = keras.metrics.Mean(name="mono_loss")
        self.convex_loss_tracker = keras.metrics.Mean(name="convex_loss")
        self.excess_curv_loss_tracker = keras.metrics.Mean(name="excess_curv_loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae_scaled")

    @property
    def metrics(self):
        return [
            self.loss_tracker, self.mse_loss_tracker, self.mono_loss_tracker,
            self.convex_loss_tracker, self.excess_curv_loss_tracker, self.mae_metric
        ]

    def call(self, inputs, training=False):
        return self.nn_core(inputs, training=training)

    def train_step(self, data):
        # Data now includes sample_weight from the tf.data.Dataset
        inputs, y_true, sample_weight = data
        
        with tf.GradientTape() as tape:
            y_pred = self.nn_core(inputs, training=True)
            
            # Get all loss components, passing the sample_weight
            mse, mono, convex, excess_curv = get_all_loss_components(y_true, y_pred, sample_weight)
            
            # Combine them using the weights dictionary
            total_loss = (self.loss_weights['mse'] * mse +
                          self.loss_weights['monotonicity'] * mono +
                          self.loss_weights['convexity'] * convex +
                          self.loss_weights['excess_curvature'] * excess_curv)
                          
        grads = tape.gradient(total_loss, self.nn_core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn_core.trainable_variables))
        
        # Update all trackers
        self.loss_tracker.update_state(total_loss)
        self.mse_loss_tracker.update_state(mse)
        self.mono_loss_tracker.update_state(mono)
        self.convex_loss_tracker.update_state(convex)
        self.excess_curv_loss_tracker.update_state(excess_curv)
        self.mae_metric.update_state(y_true, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, y_true, sample_weight = data
        y_pred = self.nn_core(inputs, training=False)
        
        mse, mono, convex, excess_curv = get_all_loss_components(y_true, y_pred, sample_weight)
        
        total_loss = (self.loss_weights['mse'] * mse +
                      self.loss_weights['monotonicity'] * mono +
                      self.loss_weights['convexity'] * convex +
                      self.loss_weights['excess_curvature'] * excess_curv)

        self.loss_tracker.update_state(total_loss)
        self.mse_loss_tracker.update_state(mse)
        self.mono_loss_tracker.update_state(mono)
        self.convex_loss_tracker.update_state(convex)
        self.excess_curv_loss_tracker.update_state(excess_curv)
        self.mae_metric.update_state(y_true, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

# ───────────────────────────────────────────────────────────────────────────────
#  Full Reconstructor Class
# ───────────────────────────────────────────────────────────────────────────────

class MPPIVReconstructor:
    def __init__(self, use_gpu_if_available: bool = True):
        available_gpus = tf.config.list_physical_devices('GPU')
        self.use_gpu = bool(available_gpus) and use_gpu_if_available
        self.X_clean = self.y_clean_scaled_fixed = self.v_clean_fixed = None
        ### NEW ###: Added attribute to store sample weights
        self.y_sample_weights = None
        self.per_curve_isc = None
        self.param_transformer = self.scalar_scaler = None
        self.nn_model = self.training_history = None

    def load_and_prepare_data(self) -> bool:
        logger.info("=== Loading & Preparing Data (PCHIP MPP-centric method) ===")
        if not (Path(INPUT_FILE_PARAMS).exists() and Path(INPUT_FILE_IV).exists()):
            logger.error(f"Input files not found: {INPUT_FILE_PARAMS}, {INPUT_FILE_IV}")
            return False
        try:
            params_df = pd.read_csv(INPUT_FILE_PARAMS, header=None, names=COLNAMES)
            iv_data_np = np.loadtxt(INPUT_FILE_IV, delimiter=',', dtype=np.float32)
            logger.info(f"Loaded raw data: params {params_df.shape}, I–V {iv_data_np.shape}")
            full_voltage_grid = np.concatenate([v1, v2]).astype(np.float32)

            processed_voltages, processed_currents, valid_indices = [], [], []
            for i in range(iv_data_np.shape[0]):
                result = process_iv_with_pchip(iv_data_np[i], full_voltage_grid)
                if result is None: continue
                v_slice, i_slice = result
                processed_voltages.append(v_slice)
                processed_currents.append(i_slice)
                valid_indices.append(i)

            if not processed_currents:
                logger.error("No valid curves after PCHIP processing. Aborting."); return False

            self.v_clean_fixed = np.array(processed_voltages, dtype=np.float32)
            i_clean_arr = np.array(processed_currents, dtype=np.float32)
            per_curve_isc, scaled_curves = zip(*[normalize_and_scale_by_isc(c) for c in i_clean_arr])
            self.y_clean_scaled_fixed = np.array(list(scaled_curves), dtype=np.float32)
            self.per_curve_isc = np.array(per_curve_isc, dtype=np.float32)

            ### NEW ###: Pre-compute curvature weights for the cleaned, scaled true data
            logger.info("Computing curvature-based sample weights...")
            self.y_sample_weights = compute_curvature_weights(
                self.y_clean_scaled_fixed,
                alpha=CURVATURE_WEIGHT_ALPHA,
                power=CURVATURE_WEIGHT_POWER
            )

            params_df_valid = params_df.iloc[valid_indices].reset_index(drop=True)
            scalar_df = pd.DataFrame({
                'I_ref': [c[0] for c in i_clean_arr],
                'V_mpp': [v[NUM_PRE_MPP_POINTS] for v in self.v_clean_fixed],
                'I_mpp': [c[NUM_PRE_MPP_POINTS] for c in i_clean_arr]
            })

            X_params, self.param_transformer = preprocess_input_parameters(params_df_valid)
            X_scalar, self.scalar_scaler = preprocess_scalar_features(scalar_df, fit=True)
            self.X_clean = np.concatenate([X_params, X_scalar], axis=1)

            logger.info(f"Final cleaned data shapes: X={self.X_clean.shape}, y={self.y_clean_scaled_fixed.shape}, weights={self.y_sample_weights.shape}")
            return True
        except Exception as e:
            logger.exception(f"Error during data loading/preparation: {e}")
            return False

    ### MODIFIED ###: Function signature updated to accept weights
    def fit_physics_informed_nn(self, X_train, y_train, V_train, W_train, X_val, y_val, V_val, W_val):
        logger.info("=== Fitting Physics‐Informed Neural Network ===")
        input_dim_params, seq_len = X_train.shape[1], FIXED_SEQUENCE_LENGTH
        nn_core = build_nn_core(input_dim_params, seq_len)
        self.nn_model = PhysicsNNModel(nn_core, LOSS_WEIGHTS)

        steps_per_epoch = X_train.shape[0] // BATCH_SIZE
        total_steps = steps_per_epoch * NN_EPOCHS
        cosine_lr_schedule = CosineDecay(
            initial_learning_rate=NN_INITIAL_LEARNING_RATE,
            decay_steps=total_steps,
            alpha=NN_FINAL_LEARNING_RATE / NN_INITIAL_LEARNING_RATE
        )
        optimizer = Adam(learning_rate=cosine_lr_schedule)
        self.nn_model.compile(optimizer=optimizer)

        ### MODIFIED ###: Pass the sample weights into the tf.data.Dataset
        train_ds = tf.data.Dataset.from_tensor_slices((
            {"X_params": X_train, "voltage_grid": V_train}, y_train, W_train
        )).shuffle(10000, seed=RANDOM_SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((
            {"X_params": X_val, "voltage_grid": V_val}, y_val, W_val
        )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        callbacks = [EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)]
        self.training_history = self.nn_model.fit(train_ds, epochs=NN_EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)
        logger.info("Physics‐informed NN training completed.")

    def predict(self, X: np.ndarray, V: np.ndarray, per_curve_isc: np.ndarray) -> np.ndarray:
        if self.nn_model is None: raise RuntimeError("NN model not fitted.")
        inputs = {"X_params": X, "voltage_grid": V}
        y_pred_scaled = self.nn_model.predict(inputs, batch_size=512, verbose=0)
        y_norm = (y_pred_scaled + 1.0) / 2.0
        predicted_curves = y_norm * per_curve_isc[:, np.newaxis]
        return predicted_curves.astype(np.float32)

    def evaluate_model(self, X, V, y_true_curves, per_curve_isc) -> dict:
        logger.info(f"=== Evaluating Model ===")
        y_pred_curves = self.predict(X, V, per_curve_isc)
        flat_true = y_true_curves.flatten()
        flat_pred = y_pred_curves.flatten()
        per_curve_r2 = [r2_score(y_true_curves[i], y_pred_curves[i]) for i in range(y_true_curves.shape[0]) if np.std(y_true_curves[i]) > 1e-9]
        mae = mean_absolute_error(flat_true, flat_pred)
        rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
        r2_global = r2_score(flat_true, flat_pred)
        mean_r2, std_r2 = (np.mean(per_curve_r2), np.std(per_curve_r2)) if per_curve_r2 else (np.nan, np.nan)
        logger.info(f"  Metrics: MAE={mae:.6f}, RMSE={rmse:.6f}, Global R²={r2_global:.4f}")
        logger.info(f"  Per‐curve R²: Mean={mean_r2:.4f}, Std={std_r2:.4f}")
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2_global, 'per_curve_R2_mean': mean_r2, 'per_curve_R2_std': std_r2}

    def plot_results(self, X, V, y_true_curves, per_curve_isc, n_samples=4, suffix=""):
        y_pred_curves = self.predict(X, V, per_curve_isc)
        N = X.shape[0]
        if N == 0: logger.warning("No samples to plot."); return
        indices = np.random.choice(N, size=min(n_samples, N), replace=False)
        nrows, ncols = (n_samples + 1) // 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False, constrained_layout=True)
        axes = axes.flatten()

        for i_plot, idx in enumerate(indices):
            ax = axes[i_plot]
            true_v, true_i = V[idx], y_true_curves[idx]
            pred_i = y_pred_curves[idx]
            r2_val = r2_score(true_i, pred_i) if np.std(true_i) > 1e-9 else np.nan
            ax.plot(true_v, true_i, 'b-', lw=2, label='Actual')
            ax.plot(true_v, pred_i, 'r--', lw=2, label='Predicted')
            ax.set_title(f"Sample {idx} (R²={r2_val:.4f})")
            ax.set_xlabel("Voltage (V)"), ax.set_ylabel("Current Density (mA/cm²)"), ax.legend(), ax.grid(True, alpha=0.3)
        for j in range(len(indices), len(axes)): fig.delaxes(axes[j])
        outpath = OUTPUT_DIR / f"preds_plot{suffix}.png"
        plt.savefig(outpath, dpi=300)
        plt.close(fig)
        logger.info(f"Prediction plot saved: {outpath}")

# ───────────────────────────────────────────────────────────────────────────────
#  Main Execution
# ───────────────────────────────────────────────────────────────────────────────
def run_experiment(use_gpu: bool = True):
    logger.info(f"=== STARTING EXPERIMENT: Curvature-Aware PCHIP, GPU={use_gpu} ===")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    recon = MPPIVReconstructor(use_gpu_if_available=use_gpu)
    if not recon.load_and_prepare_data():
        logger.error("Data preparation failed. Aborting experiment.")
        return

    all_idx = np.arange(recon.X_clean.shape[0])
    train_val_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=RANDOM_SEED)

    ### MODIFIED ###: Split all data arrays, including the new sample weights
    X_train, y_train, V_train, W_train = (
        recon.X_clean[train_idx],
        recon.y_clean_scaled_fixed[train_idx],
        recon.v_clean_fixed[train_idx],
        recon.y_sample_weights[train_idx]
    )
    X_val, y_val, V_val, W_val = (
        recon.X_clean[val_idx],
        recon.y_clean_scaled_fixed[val_idx],
        recon.v_clean_fixed[val_idx],
        recon.y_sample_weights[val_idx]
    )
    X_test, y_test, V_test = (
        recon.X_clean[test_idx],
        recon.y_clean_scaled_fixed[test_idx],
        recon.v_clean_fixed[test_idx]
    )
    isc_test = recon.per_curve_isc[test_idx]

    ### MODIFIED ###: Pass the weight arrays to the fitting function
    recon.fit_physics_informed_nn(X_train, y_train, V_train, W_train, X_val, y_val, V_val, W_val)

    if recon.training_history:
        hist_df = pd.DataFrame(recon.training_history.history)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(hist_df.index, hist_df['loss'], label='Train Total Loss')
        ax1.plot(hist_df.index, hist_df['val_loss'], label='Val Total Loss', linestyle='--')
        ax1.set_ylabel("Total Loss"), ax1.set_title("Training & Validation Loss"), ax1.legend(), ax1.grid(True, alpha=0.3), ax1.set_yscale('log')
        
        ### MODIFIED ###: Plot the new loss components
        loss_cols = [c for c in hist_df.columns if c.endswith('_loss') and 'val_' not in c and c != 'loss']
        for col in loss_cols:
            ax2.plot(hist_df.index, hist_df[col], label=col.replace('_loss', ''))
        
        ax2.set_xlabel("Epoch"), ax2.set_ylabel("Loss Component Value"), ax2.set_title("Training Loss Components"), ax2.legend(), ax2.grid(True, alpha=0.3), ax2.set_yscale('log')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "training_history.png", dpi=300)
        plt.close(fig)

    y_test_physical = (y_test + 1.0) / 2.0 * isc_test[:, np.newaxis]
    test_metrics = recon.evaluate_model(X_test, V_test, y_test_physical, isc_test)
    recon.plot_results(X_test, V_test, y_test_physical, isc_test, n_samples=8, suffix="_test_set")

    summary = {f'test_{k}': v for k, v in test_metrics.items()}
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)
    logger.info(f"Experiment finished. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    if not Path(INPUT_FILE_PARAMS).exists() or not Path(INPUT_FILE_IV).exists():
        logger.error("="*80)
        logger.error("Input data files not found!")
        logger.error(f"Please check INPUT_FILE_PARAMS and INPUT_FILE_IV constants.")
        logger.error("="*80)
    else:
        run_experiment(use_gpu=True)