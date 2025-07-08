#!/usr/bin/env python3
"""
This is the working implementation, after implementing interpolation.
Now its legacy, the new one is ported to lightning
Approximate numbers:
MAE: 4, RMSE: 10, R²: 0.996, R^2 per‐curve: ~0.991

A lean, focused implementation for Perovskite I–V curve reconstruction using a
physics‐informed neural network.

--- HYBRID ARCHITECTURE ---
This version uses a high-performance Attention-Augmented TCN. It combines the
proven effectiveness of Temporal Convolutional Networks (TCNs) with a
self-attention layer to capture both local and long-range dependencies in the
I-V curve sequence. This hybrid approach aims to outperform both the pure TCN
and pure Transformer models.

- Core processing is done via Conv1D (TCN) layers.
- A Multi-Head Self-Attention block is injected between TCN layers.
- Features advanced, configurable positional embeddings (Clipped Fourier, Gaussian).
- Uses AdamW optimizer with a warmup and cosine decay schedule.

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
    Conv1D, MultiHeadAttention, LayerNormalization
)
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

# Data Processing
PCHIP_FINE_GRID_POINTS = 10000
PCHIP_VOLTAGE_MAX = 1.4
NUM_PRE_MPP_POINTS = 3
NUM_POST_MPP_POINTS = 4
FIXED_SEQUENCE_LENGTH = NUM_PRE_MPP_POINTS + 1 + NUM_POST_MPP_POINTS

# Physics-informed parameters
CURVATURE_WEIGHT_ALPHA = 4.0
CURVATURE_WEIGHT_POWER = 1.5
EXCESS_CURVATURE_THRESHOLD = 0.8
LOSS_WEIGHTS = {
    'mse': 0.98, 'monotonicity': 0.005, 'convexity': 0.005, 'excess_curvature': 0.01
}
RANDOM_SEED = 42

EMBEDDING_TYPE = 'fourier_clipped'  # Options: 'fourier', 'fourier_clipped', 'gaussian'

# Architecture
DENSE_UNITS_PARAMS      = [256, 128, 128]
TCN_FILTERS             = [128, 64] # Filters for the TCN blocks
TCN_KERNEL_SIZE         = 5
ATTENTION_HEADS         = 4 # Fewer heads for a shorter sequence
DROPOUT_RATE            = 0.25

# Embedding Dimensions
FOURIER_NUM_BANDS       = 16
GAUSSIAN_NUM_BANDS      = 16
GAUSSIAN_SIGMA          = 0.1

# Training hyperparameters
NN_INITIAL_LEARNING_RATE = 0.0015
NN_FINAL_LEARNING_RATE   = 1e-6
NN_EPOCHS                = 100
BATCH_SIZE               = 128
WARMUP_EPOCHS            = 5
WEIGHT_DECAY             = 1e-5

# Column names for input parameters file (MODIFIED)
COLNAMES = [
    'lH',   # H layer thickness (nm)
    'lP',   # P layer thickness (nm)
    'lE',   # E layer thickness (nm)
    'muHh', # μₕᴴ: Hole mobility in H (m²/V·s)
    'muPh', # μₕᴾ: Hole mobility in P
    'muPe', # μₑᴾ: Electron mobility in P
    'muEe', # μₑᴱ: Electron mobility in E
    'NvH',  # Nᵥᴴ: Valence DOS in H (m⁻³)
    'NcH',  # N꜀ᴴ: Conduction DOS in H
    'NvE',  # Nᵥᴱ: Valence DOS in E
    'NcE',  # N꜀ᴱ: Conduction DOS in E
    'NvP',  # Nᵥᴾ: Valence DOS in P
    'NcP',  # N꜀ᴾ: Conduction DOS in P
    'chiHh',# χₕᴴ: Hole ionization in H (eV)
    'chiHe',# χₑᴴ: Electron affinity in H
    'chiPh',# χₕᴾ: Hole ionization in P
    'chiPe',# χₑᴾ: Electron affinity in P
    'chiEh',# χₕᴱ: Hole ionization in E
    'chiEe',# χₑᴱ: Electron affinity in E
    'Wlm',  # Wᴮ: Back‐contact work function (eV)
    'Whm',  # Wᶠ: Front‐contact work function (eV)
    'epsH', # εᴴ: Permittivity in H
    'epsP', # εᴾ: Permittivity in P
    'epsE', # εᴱ: Permittivity in E
    'Gavg', # 𝐺avg: Generation rate (m⁻³·s⁻¹)
    'Aug',  # A(e,h): Auger recomb coeff (m⁶·s⁻¹)
    'Brad', # Bᵣₐ𝒹: Radiative recomb coeff (m³·s⁻¹)
    'Taue', # τₑ: Electron lifetime (s)
    'Tauh', # τₕ: Hole lifetime (s)
    'vII',  # vII: Interface recomb velocity II (m⁴·s⁻¹)
    'vIII', # vIII: Interface recomb velocity III (m⁴·s⁻¹)
]

# ───────────────────────────────────────────────────────────────────────────────
#  Advanced Positional Embedding Functions
# ───────────────────────────────────────────────────────────────────────────────

def fourier_features(V_norm):
    B = tf.constant(np.logspace(0, 3, num=FOURIER_NUM_BANDS), dtype=tf.float32)
    V = V_norm[..., None] * B[None, None, :]
    return tf.concat([tf.sin(2 * np.pi * V), tf.cos(2 * np.pi * V)], axis=-1)

def clipped_fourier_features(V_norm):
    B = tf.constant(np.logspace(0, 3, num=FOURIER_NUM_BANDS), dtype=tf.float32)
    B_mask = tf.cast(B >= 1.0, tf.float32)
    V = V_norm[..., None] * B[None, None, :]
    sines = tf.sin(2 * np.pi * V) * B_mask[None, None, :]
    coses = tf.cos(2 * np.pi * V) * B_mask[None, None, :]
    return tf.concat([sines, coses], axis=-1)

def gaussian_features(V_norm):
    mu = tf.linspace(0.0, 1.0, GAUSSIAN_NUM_BANDS)
    diff = V_norm[..., None] - mu[None, None, :]
    return tf.exp(-0.5 * (diff / GAUSSIAN_SIGMA)**2)

# ───────────────────────────────────────────────────────────────────────────────
#  Utility & Preprocessing Functions
# ───────────────────────────────────────────────────────────────────────────────

def process_iv_with_pchip(
    curve_current_raw: np.ndarray, full_voltage_grid: np.ndarray
) -> typing.Optional[tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]]:
    try:
        interpolator = PchipInterpolator(full_voltage_grid, curve_current_raw, extrapolate=False)
        v_fine = np.linspace(0, PCHIP_VOLTAGE_MAX, PCHIP_FINE_GRID_POINTS)
        i_fine = interpolator(v_fine)
        valid_mask = ~np.isnan(i_fine); v_fine, i_fine = v_fine[valid_mask], i_fine[valid_mask]
        if v_fine.size < 2: return None
        zero_cross_idx = np.where(i_fine <= 0)[0]
        voc_v = v_fine[zero_cross_idx[0]] if len(zero_cross_idx) > 0 else v_fine[-1]
        v_search, i_search = v_fine[v_fine <= voc_v], i_fine[v_fine <= voc_v]
        if v_search.size == 0: return None
        power = v_search * i_search; mpp_idx = np.argmax(power)
        v_mpp = v_search[mpp_idx]
        v_pre_mpp = np.linspace(v_search[0], v_mpp, NUM_PRE_MPP_POINTS + 2, endpoint=True)[:-1]
        v_post_mpp = np.linspace(v_mpp, v_search[-1], NUM_POST_MPP_POINTS + 2, endpoint=True)[1:]
        v_mpp_grid = np.unique(np.concatenate([v_pre_mpp, v_post_mpp]))
        v_mpp_grid_final = np.interp(np.linspace(0, 1, FIXED_SEQUENCE_LENGTH), np.linspace(0, 1, len(v_mpp_grid)), v_mpp_grid)
        i_mpp_slice = interpolator(v_mpp_grid_final)
        if np.any(np.isnan(i_mpp_slice)) or i_mpp_slice.shape[0] != FIXED_SEQUENCE_LENGTH: return None

        return (
            v_mpp_grid_final.astype(np.float32),
            i_mpp_slice.astype(np.float32),
            (v_fine.astype(np.float32), i_fine.astype(np.float32))
        )
    except (ValueError, IndexError): return None

def normalize_and_scale_by_isc(curve: np.ndarray) -> tuple[float, np.ndarray]:
    isc_val = float(curve[0]);
    if isc_val <= 0: return 1.0, (2.0 * curve.copy().astype(np.float32) - 1.0)
    return isc_val, (2.0 * (curve / isc_val) - 1.0).astype(np.float32)

def compute_curvature_weights(y_curves: np.ndarray, alpha: float, power: float) -> np.ndarray:
    padded = np.pad(y_curves, ((0, 0), (1, 1)), mode='edge')
    kappa = np.abs(padded[:, 2:] - 2 * padded[:, 1:-1] + padded[:, :-2])
    max_kappa = np.max(kappa, axis=1, keepdims=True); max_kappa[max_kappa < 1e-9] = 1.0
    return (1.0 + alpha * np.power(kappa / max_kappa, power)).astype(np.float32)

def get_all_loss_components(y_true, y_pred, sample_weight):
    y_true, y_pred, sample_weight = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), tf.cast(sample_weight, tf.float32)
    se = tf.square(y_true - y_pred); weighted_se = se * sample_weight
    mono_violations = tf.nn.relu(y_pred[:, 1:] - y_pred[:, :-1])
    convex_violations = tf.nn.relu(y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2])
    excess_curv_violations = tf.nn.relu(tf.abs(y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2]) - EXCESS_CURVATURE_THRESHOLD)
    return (
        tf.reduce_mean(weighted_se),
        tf.reduce_mean(tf.square(mono_violations)),
        tf.reduce_mean(tf.square(convex_violations)),
        tf.reduce_mean(tf.square(excess_curv_violations))
    )

def preprocess_input_parameters(params_df: pd.DataFrame) -> tuple[np.ndarray, ColumnTransformer]:
    # MODIFIED: Grouping based on new 3-layer device physics parameters
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
        actual_cols = [c for c in cols if c in params_df.columns]
        if not actual_cols: continue
        steps = [('robust', RobustScaler()), ('minmax', MinMaxScaler(feature_range=(-1, 1)))]
        # Apply log transform to material properties which can span orders of magnitude
        if group == 'material_properties':
            steps.insert(0, ('log1p', FunctionTransformer(func=np.log1p)))
        transformers.append((group, Pipeline(steps), actual_cols))

    ct = ColumnTransformer(transformers, remainder='passthrough')
    return ct.fit_transform(params_df).astype(np.float32), ct

def preprocess_scalar_features(df: pd.DataFrame, fit: bool = True, scaler: Pipeline = None) -> tuple[np.ndarray, Pipeline]:
    if fit: scaler = Pipeline([('scaler', MinMaxScaler(feature_range=(-1, 1)))])
    return scaler.fit_transform(df.values).astype(np.float32) if fit else scaler.transform(df.values).astype(np.float32), scaler

# ───────────────────────────────────────────────────────────────────────────────
#  HYBRID MODEL DEFINITION: Attention-Augmented TCN 
# ───────────────────────────────────────────────────────────────────────────────

def build_nn_core(input_dim_params: int, seq_len: int) -> Model:
    # 1. Parameter Path (from TCN model)
    param_in = Input(shape=(input_dim_params,), name="X_params")
    param_path = param_in
    for i, units in enumerate(DENSE_UNITS_PARAMS):
        param_path = Dense(units, activation='gelu', name=f"param_dense_{i}")(param_path)
        param_path = BatchNormalization()(param_path)
        if i == 0: param_path = Dropout(DROPOUT_RATE)(param_path)

    # 2. Voltage Embedding Path (Configurable)
    voltage_in = Input(shape=(seq_len,), name="voltage_grid")
    norm_voltage = Lambda(lambda v: v / PCHIP_VOLTAGE_MAX)(voltage_in)

    # Select and define shape for the Lambda layer
    if EMBEDDING_TYPE == 'gaussian':
        shape = (seq_len, GAUSSIAN_NUM_BANDS)
        v_embed = Lambda(gaussian_features, output_shape=shape, name="gaussian_embed")(norm_voltage)
    elif EMBEDDING_TYPE == 'fourier_clipped':
        shape = (seq_len, 2 * FOURIER_NUM_BANDS)
        v_embed = Lambda(clipped_fourier_features, output_shape=shape, name="fourier_clipped_embed")(norm_voltage)
    else: # 'fourier'
        shape = (seq_len, 2 * FOURIER_NUM_BANDS)
        v_embed = Lambda(fourier_features, output_shape=shape, name="fourier_embed")(norm_voltage)

    # 3. Combine and process with Hybrid TCN-Attention Core
    param_tiled = Lambda(lambda t: tf.tile(tf.expand_dims(t, 1), [1, seq_len, 1]))(param_path)
    x = Concatenate(axis=-1)([param_tiled, v_embed])

    # --- First TCN Block ---
    x = Conv1D(TCN_FILTERS[0], kernel_size=TCN_KERNEL_SIZE, padding='causal', activation='gelu')(x)
    x = LayerNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    # --- Injected Attention Block ---
    attn_input = x
    key_dim = attn_input.shape[-1]
    attn_output = MultiHeadAttention(num_heads=ATTENTION_HEADS, key_dim=key_dim)(x, x)
    attn_output = Dropout(DROPOUT_RATE)(attn_output)
    x = LayerNormalization()(attn_input + attn_output) # Add & Norm

    # --- Second TCN Block ---
    x = Conv1D(TCN_FILTERS[1], kernel_size=TCN_KERNEL_SIZE, padding='causal', activation='gelu')(x)
    x = LayerNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    # 4. Output Head
    iv_output = Dense(1, activation=None)(x)
    iv_output_flat = Lambda(lambda t: tf.squeeze(t, axis=-1))(iv_output)

    return Model(inputs=[param_in, voltage_in], outputs=iv_output_flat, name="Attention_TCN_Hybrid")

# The PhysicsNNModel and WarmupCosineDecay classes are perfect as-is.
class PhysicsNNModel(keras.Model):
    def __init__(self, nn_core: Model, loss_weights: dict, **kwargs):
        super().__init__(**kwargs)
        self.nn_core = nn_core; self.loss_weights = loss_weights
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        self.mono_loss_tracker = keras.metrics.Mean(name="mono_loss")
        self.convex_loss_tracker = keras.metrics.Mean(name="convex_loss")
        self.excess_curv_loss_tracker = keras.metrics.Mean(name="excess_curv_loss")
    @property
    def metrics(self): return [self.loss_tracker, self.mse_loss_tracker, self.mono_loss_tracker, self.convex_loss_tracker, self.excess_curv_loss_tracker]
    def call(self, inputs, training=False): return self.nn_core(inputs, training=training)
    def train_step(self, data):
        inputs, y_true, sample_weight = data
        with tf.GradientTape() as tape:
            y_pred = self.nn_core(inputs, training=True)
            mse, mono, convex, excess_curv = get_all_loss_components(y_true, y_pred, sample_weight)
            total_loss = ( self.loss_weights['mse'] * mse + self.loss_weights['monotonicity'] * mono + self.loss_weights['convexity'] * convex + self.loss_weights['excess_curvature'] * excess_curv)
        grads = tape.gradient(total_loss, self.nn_core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn_core.trainable_variables))
        self.loss_tracker.update_state(total_loss); self.mse_loss_tracker.update_state(mse); self.mono_loss_tracker.update_state(mono)
        self.convex_loss_tracker.update_state(convex); self.excess_curv_loss_tracker.update_state(excess_curv)
        return {m.name: m.result() for m in self.metrics}
    def test_step(self, data):
        inputs, y_true, sample_weight = data
        y_pred = self.nn_core(inputs, training=False)
        mse, mono, convex, excess_curv = get_all_loss_components(y_true, y_pred, sample_weight)
        total_loss = ( self.loss_weights['mse'] * mse + self.loss_weights['monotonicity'] * mono + self.loss_weights['convexity'] * convex + self.loss_weights['excess_curvature'] * excess_curv)
        self.loss_tracker.update_state(total_loss); self.mse_loss_tracker.update_state(mse); self.mono_loss_tracker.update_state(mono)
        self.convex_loss_tracker.update_state(convex); self.excess_curv_loss_tracker.update_state(excess_curv)
        return {m.name: m.result() for m in self.metrics}

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, total_steps, warmup_steps, final_lr, name=None):
        super().__init__(); self.initial_lr, self.total_steps, self.warmup_steps, self.final_lr, self.name = initial_lr, total_steps, warmup_steps, final_lr, name
        self.warmup = tf.keras.optimizers.schedules.PolynomialDecay(1e-7, warmup_steps, initial_lr, power=1.0)
        self.cosine = tf.keras.optimizers.schedules.CosineDecay(initial_lr, total_steps - warmup_steps, alpha=final_lr / initial_lr if initial_lr > 0 else 0.0)
    def __call__(self, step):
        return tf.cond(tf.cast(step, tf.float32) < self.warmup_steps, lambda: self.warmup(step), lambda: self.cosine(step - self.warmup_steps))
    def get_config(self): return {"initial_lr": self.initial_lr, "total_steps": self.total_steps, "warmup_steps": self.warmup_steps, "final_lr": self.final_lr, "name": self.name}

class MPPIVReconstructor:
    def __init__(self, use_gpu_if_available: bool = True):
        self.use_gpu = bool(tf.config.list_physical_devices('GPU')) and use_gpu_if_available
        self.X_clean = self.y_clean_scaled_fixed = self.v_clean_fixed = None
        self.y_sample_weights = self.per_curve_isc = None
        self.param_transformer = self.scalar_scaler = None
        self.nn_model = self.training_history = None
        self.original_indices = None
        self.fine_grid_curves = None

    def load_and_prepare_data(self) -> bool:
        logger.info("=== Loading & Preparing Data ===")
        try:
            params_df = pd.read_csv(INPUT_FILE_PARAMS, header=None, names=COLNAMES)
            iv_data = np.loadtxt(INPUT_FILE_IV, delimiter=',', dtype=np.float32)
            full_v_grid = np.concatenate([np.arange(0.0, 0.4 + 1e-8, 0.1), np.arange(0.425, 1.4 + 1e-8, 0.025)]).astype(np.float32)

            processed = [process_iv_with_pchip(iv_data[i], full_v_grid) for i in range(iv_data.shape[0])]
            valid_results = [(res[0], res[1], res[2], idx) for idx, res in enumerate(processed) if res is not None]

            if not valid_results: logger.error("No valid curves after PCHIP. Aborting."); return False

            v_slices, i_slices, fine_curves, valid_indices = zip(*valid_results)
            self.v_clean_fixed = np.array(v_slices, dtype=np.float32)
            i_clean_arr = np.array(i_slices, dtype=np.float32)

            self.original_indices = np.array(valid_indices, dtype=np.int32)
            self.fine_grid_curves = list(fine_curves) # Store as list of tuples

            isc_vals, scaled_curves = zip(*[normalize_and_scale_by_isc(c) for c in i_clean_arr])
            self.y_clean_scaled_fixed = np.array(list(scaled_curves), dtype=np.float32)
            self.per_curve_isc = np.array(isc_vals, dtype=np.float32)

            self.y_sample_weights = compute_curvature_weights(self.y_clean_scaled_fixed, CURVATURE_WEIGHT_ALPHA, CURVATURE_WEIGHT_POWER)

            params_df_valid = params_df.iloc[list(valid_indices)].reset_index(drop=True)
            scalar_df = pd.DataFrame({'I_ref': [c[0] for c in i_clean_arr], 'V_mpp': self.v_clean_fixed[:, NUM_PRE_MPP_POINTS], 'I_mpp': i_clean_arr[:, NUM_PRE_MPP_POINTS]})

            X_params, self.param_transformer = preprocess_input_parameters(params_df_valid)
            X_scalar, self.scalar_scaler = preprocess_scalar_features(scalar_df, fit=True)
            self.X_clean = np.concatenate([X_params, X_scalar], axis=1)

            logger.info(f"Final data shapes: X={self.X_clean.shape}, y={self.y_clean_scaled_fixed.shape}, v={self.v_clean_fixed.shape}")
            logger.info(f"Retained {len(self.original_indices)} valid curves out of {iv_data.shape[0]}.")
            return True
        except Exception as e:
            logger.exception(f"Error during data loading/preparation: {e}"); return False

    def fit_physics_informed_nn(self, X_train, y_train, V_train, W_train, X_val, y_val, V_val, W_val):
        logger.info(f"=== Fitting Hybrid Attention-TCN (Embedding: {EMBEDDING_TYPE}) ===")
        nn_core = build_nn_core(X_train.shape[1], FIXED_SEQUENCE_LENGTH)
        self.nn_model = PhysicsNNModel(nn_core, LOSS_WEIGHTS)

        steps_per_epoch = X_train.shape[0] // BATCH_SIZE
        total_steps = steps_per_epoch * NN_EPOCHS
        warmup_steps = WARMUP_EPOCHS * steps_per_epoch

        lr_schedule = WarmupCosineDecay(NN_INITIAL_LEARNING_RATE, total_steps, warmup_steps, NN_FINAL_LEARNING_RATE)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
        self.nn_model.compile(optimizer=optimizer)

        train_ds = tf.data.Dataset.from_tensor_slices( ({"X_params": X_train, "voltage_grid": V_train}, y_train, W_train) ).shuffle(10000, seed=RANDOM_SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices( ({"X_params": X_val, "voltage_grid": V_val}, y_val, W_val) ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)]
        self.training_history = self.nn_model.fit(train_ds, epochs=NN_EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)
        logger.info("NN training completed.")

    def predict(self, X: np.ndarray, V: np.ndarray, per_curve_isc: np.ndarray) -> np.ndarray:
        if self.nn_model is None: raise RuntimeError("Model not fitted.")
        y_pred_scaled = self.nn_model.predict({"X_params": X, "voltage_grid": V}, batch_size=1024, verbose=0)
        y_norm = (y_pred_scaled + 1.0) / 2.0
        return (y_norm * per_curve_isc[:, np.newaxis]).astype(np.float32)

    def evaluate_model(self, X, V, y_true, per_curve_isc, clean_set_indices) -> dict:
        y_pred = self.predict(X, V, per_curve_isc)
        flat_true, flat_pred = y_true.flatten(), y_pred.flatten()
        mae = mean_absolute_error(flat_true, flat_pred)
        rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
        r2 = r2_score(flat_true, flat_pred)
        logger.info(f"Evaluation Metrics: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}")

        # MODIFIED: Per‐Curve R² calculation
        r2_per_curve = [
          r2_score(y_true[i], y_pred[i])
          for i in range(len(y_true))
        ]
        
        # Get the original row numbers for the current data split (e.g., test set)
        original_rows_for_this_set = self.original_indices[clean_set_indices]

        pd.DataFrame({
          'original_row': original_rows_for_this_set,
          'r2_curve':    r2_per_curve
        }).to_csv(OUTPUT_DIR/'r2_per_curve.csv', index=False)
        logger.info(f"Saved per-curve R² values to {OUTPUT_DIR / 'r2_per_curve.csv'}")

        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    def plot_results(self, test_indices: np.ndarray, n_samples: int = 8, suffix: str = ""):
        logger.info("--- Generating comparison plots ---")
        # 1. Slice all required data using the provided test indices
        X_test = self.X_clean[test_indices]
        V_test_slices = self.v_clean_fixed[test_indices]
        isc_test = self.per_curve_isc[test_indices]
        y_scaled_test = self.y_clean_scaled_fixed[test_indices]

        # Denormalize the true y-values (8-point slices)
        y_true_physical_slices = (y_scaled_test + 1.0) / 2.0 * isc_test[:, np.newaxis]

        # Get predictions for the 8-point slices
        y_pred_physical_slices = self.predict(X_test, V_test_slices, isc_test)

        # Get the corresponding original row numbers and full fine-grid curves
        original_row_numbers_test = self.original_indices[test_indices]
        fine_grid_curves_test = [self.fine_grid_curves[i] for i in test_indices]

        # 2. Setup plotting grid
        N = X_test.shape[0]
        if N == 0: logger.warning("No samples to plot."); return
        plot_indices = np.random.choice(N, size=min(n_samples, N), replace=False)
        nrows, ncols = (len(plot_indices) + 3) // 4, 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False, constrained_layout=True)
        axes = axes.flatten()

        # 3. Loop through random samples and plot
        for i, idx in enumerate(plot_indices):
            ax = axes[i]

            # Get data for this specific sample
            original_row = original_row_numbers_test[idx]
            v_slice, i_true_slice = V_test_slices[idx], y_true_physical_slices[idx]
            i_pred_slice = y_pred_physical_slices[idx]
            v_fine_actual, i_fine_actual = fine_grid_curves_test[idx]

            try:
                recon_interp = PchipInterpolator(v_slice, i_pred_slice, extrapolate=False)
                v_full_recon = np.linspace(v_slice[0], v_slice[-1], 500)
                i_full_recon = recon_interp(v_full_recon)
            except ValueError:
                v_full_recon, i_full_recon = v_slice, i_pred_slice

            r2 = r2_score(i_true_slice, i_pred_slice) if np.std(i_true_slice) > 1e-9 else np.nan

            # Plotting layers
            ax.plot(v_fine_actual, i_fine_actual, 'k-', alpha=0.6, lw=2, label='Actual (Fine Grid)')
            ax.plot(v_full_recon, i_full_recon, 'r--', lw=2, label='Predicted (Reconstructed)')
            ax.plot(v_slice, i_true_slice, 'bo', ms=6, label='Actual (Training Points)')
            ax.plot(v_slice, i_pred_slice, 'rx', ms=6, mew=2, label='Predicted (8 Points)')

            ax.set_title(f"Original Row {original_row} (R²={r2:.4f})")
            ax.set_xlabel("Voltage (V)"), ax.set_ylabel("Current (A/m²)") # Changed label to reflect likely units
            ax.legend(), ax.grid(True, alpha=0.4)

        for j in range(len(plot_indices), len(axes)): fig.delaxes(axes[j])
        
        # Save the full-range plots
        outpath = OUTPUT_DIR / f"predictions_sample{suffix}.png"
        plt.savefig(outpath, dpi=200)
        logger.info(f"Saved prediction plot: {outpath}")

        # MODIFIED: Create and save the zoomed versions
        zoom_dir = OUTPUT_DIR / "zoomed"
        zoom_dir.mkdir(exist_ok=True)
        for i, idx in enumerate(plot_indices):
            ax = axes[i]
            i_true_slice = y_true_physical_slices[idx]
            V_test_slice = V_test_slices[idx]
            try:
                # Find Voc from the true 8-point slice to set the x-limit
                voc_idx = np.where(i_true_slice <= 0)[0][0]
                voc_true = V_test_slice[voc_idx]
                ax.set_xlim(0, voc_true)
            except IndexError:
                # If no Voc is found (e.g., curve doesn't cross zero), use max voltage
                ax.set_xlim(0, V_test_slice[-1])

        zoom_outpath = zoom_dir / f"predictions_sample{suffix}.png"
        plt.savefig(zoom_outpath, dpi=200)
        logger.info(f"Saved zoomed prediction plot: {zoom_outpath}")
        plt.close(fig)

        if self.training_history:
            hist_df = pd.DataFrame(self.training_history.history)
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax1.plot(hist_df.index, hist_df['loss'], label='Train Loss')
            ax1.plot(hist_df.index, hist_df['val_loss'], label='Validation Loss', linestyle='--')
            ax1.set_ylabel("Total Loss"), ax1.set_xlabel("Epoch"), ax1.set_title("Training & Validation Loss")
            ax1.legend(), ax1.grid(True, alpha=0.3), ax1.set_yscale('log')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "training_history.png", dpi=200); plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
#  Main Execution
# ───────────────────────────────────────────────────────────────────────────────
def run_experiment(use_gpu: bool = True):
    logger.info(f"=== STARTING EXPERIMENT: {RUN_ID}, GPU={use_gpu} ===")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    recon = MPPIVReconstructor(use_gpu_if_available=use_gpu)
    if not recon.load_and_prepare_data():
        logger.error("Data preparation failed. Aborting."); return

    indices = np.arange(recon.X_clean.shape[0])
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=RANDOM_SEED)

    data_splits = {
        'X': recon.X_clean, 'y': recon.y_clean_scaled_fixed,
        'V': recon.v_clean_fixed, 'W': recon.y_sample_weights
    }
    X_train, y_train, V_train, W_train = (data_splits[k][train_idx] for k in data_splits)
    X_val, y_val, V_val, W_val = (data_splits[k][val_idx] for k in data_splits)
    X_test, y_test, V_test = recon.X_clean[test_idx], recon.y_clean_scaled_fixed[test_idx], recon.v_clean_fixed[test_idx]
    isc_test = recon.per_curve_isc[test_idx]

    recon.fit_physics_informed_nn(X_train, y_train, V_train, W_train, X_val, y_val, V_val, W_val)

    y_test_physical = (y_test + 1.0) / 2.0 * isc_test[:, np.newaxis]
    
    # MODIFIED: Pass test_idx to evaluate_model for per-curve R² indexing
    test_metrics = recon.evaluate_model(X_test, V_test, y_test_physical, isc_test, test_idx)

    recon.plot_results(test_idx, n_samples=8, suffix="_test")

    summary = {f'test_{k}': v for k, v in test_metrics.items()}
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)
    logger.info(f"Experiment finished. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if not (Path(INPUT_FILE_PARAMS).exists() and Path(INPUT_FILE_IV).exists()):
        logger.error("="*80 + "\nInput data files not found! Check paths.\n" + "="*80)
    else:
        run_experiment(use_gpu=True)