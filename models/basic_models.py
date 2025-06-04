# corrected_iv_reconstruction_TRUNCATED_ENHANCED.py
# Implementation with CUDA support, scaling, I-V curve truncation,
# explicit voltage grid, masking, scalar physics features, and enhanced NN loss.
# WITH NEW IMPROVEMENTS (Flexible Truncation, Global Scaling, More Features, Tuned Loss, Bulk Experiment)

import os
# Suppress TensorFlow INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as SklearnGradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.multioutput import MultiOutputRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Concatenate, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Attempt to import GPU-accelerated libraries
try:
    import cupy
    import cuml
    from cuml.decomposition import PCA as CumlPCA
    print("cuML found. Will use GPU for PCA.")
    CUML_AVAILABLE = True
except ImportError:
    print("Warning: cuML not found. PCA will run on CPU using scikit-learn.")
    CUML_AVAILABLE = False
    CumlPCA = SklearnPCA  # Fallback

try:
    import xgboost
    try:
        # Quick test: can we fit a trivial sample with gpu_hist?
        xgboost.XGBRegressor(tree_method='gpu_hist').fit(
            np.array([[1]]), np.array([1])
        )
        print("XGBoost with GPU support found. Will use GPU for GBR alternatives.")
        XGBOOST_GPU_AVAILABLE = True
    except xgboost.core.XGBoostError:
        print("Warning: XGBoost found, but GPU support (gpu_hist) seems unavailable. XGBoost will run on CPU.")
        XGBOOST_GPU_AVAILABLE = False
    BaseXGBRegressor = xgboost.XGBRegressor
except ImportError:
    print("Warning: XGBoost not found. Ensemble and PCA regressors will use scikit-learn CPU versions.")
    XGBOOST_GPU_AVAILABLE = False
    BaseXGBRegressor = SklearnGradientBoostingRegressor  # Fallback

# === BEGIN: Standalone configuration ===
INPUT_FILE = "/content/drive/MyDrive/Colab Notebooks/Data_100k/LHS_parameters_m.txt"
OUTPUT_FILE = "/content/drive/MyDrive/Colab Notebooks/Data_100k/iV_m.txt"
COLNAMES = [
    'Eg', 'NCv', 'NCc', 'mu_e', 'mu_h', 'eps', 'A', 'Cn', 'Cp', 'Nt',
    'Et', 'nD', 'nA', 'thickness', 'T', 'Sn', 'Sp', 'Rs', 'Rsh',
    'G', 'light_intensity', 'Voc_ref', 'Jsc_ref', 'FF_ref', 'PCE_ref',
    'Qe_loss', 'R_loss', 'SRH_loss', 'series_loss', 'shunt_loss', 'other_loss'
]
OUTPUT_DIR_TRUNCATED = Path("./output_data_truncated_enhanced_cuda_bulk") # Changed for bulk experiments output
OUTPUT_DIR_TRUNCATED.mkdir(parents=True, exist_ok=True)

ASSUMED_ORIGINAL_IV_POINTS = 45
ASSUMED_ORIGINAL_MAX_VOLTAGE = 1.2

def check_data_files_exist():
    return Path(INPUT_FILE).exists() and Path(OUTPUT_FILE).exists()

if not check_data_files_exist():
    print(f"Creating dummy data files for testing ({INPUT_FILE}, {OUTPUT_FILE})...")
    num_dummy_samples = 1000 # Reduced for faster testing if dummy data is used
    params_df_dummy = pd.DataFrame(np.random.rand(num_dummy_samples, len(COLNAMES)), columns=COLNAMES)
    if 'Voc_ref' in COLNAMES:
        params_df_dummy['Voc_ref'] = np.random.rand(num_dummy_samples) * 0.5 + 0.5
    if 'light_intensity' in COLNAMES: # Ensure positive light intensity
        params_df_dummy['light_intensity'] = np.random.rand(num_dummy_samples) * 0.8 + 0.2 # e.g. 0.2 to 1.0
    params_df_dummy.to_csv(INPUT_FILE, header=False, index=False)

    dummy_output_data = np.zeros((num_dummy_samples, ASSUMED_ORIGINAL_IV_POINTS))
    for i in range(num_dummy_samples):
        isc_dummy = (np.random.rand() * 20 + 10) * params_df_dummy.loc[i, 'light_intensity'] # mA/cm^2, somewhat related to light
        voc_dummy = params_df_dummy.loc[i, 'Voc_ref'] # V
        # Simple exponential decay for dummy IV
        voltage_points = np.linspace(0, ASSUMED_ORIGINAL_MAX_VOLTAGE, ASSUMED_ORIGINAL_IV_POINTS)
        current_points = isc_dummy * (1 - np.exp((voltage_points - voc_dummy) / 0.05)) # Simplified diode model
        current_points[voltage_points > voc_dummy] = isc_dummy * (1 - np.exp((voc_dummy - voc_dummy) / 0.05)) * \
                                                    (1 - (voltage_points[voltage_points > voc_dummy] - voc_dummy)/(ASSUMED_ORIGINAL_MAX_VOLTAGE-voc_dummy+1e-6))
        current_points = np.maximum(current_points, -0.1 * isc_dummy) # Limit negative current
        current_points += np.random.normal(0, 0.01 * isc_dummy, ASSUMED_ORIGINAL_IV_POINTS) # Add some noise
        dummy_output_data[i,:] = current_points

    np.savetxt(OUTPUT_FILE, dummy_output_data, delimiter=',')
    print(f"Dummy data created. {num_dummy_samples} samples.")
# === END: Standalone configuration ===

# Patch FunctionTransformer to always return input_features when calling get_feature_names_out
def _ft_get_feature_names_out(self, input_features=None):
    return input_features if input_features is not None else []
FunctionTransformer.get_feature_names_out = _ft_get_feature_names_out


# --- Masked Physics Loss Functions (with dtype casts) --------------------------------

def masked_mse_loss(y_true, y_pred, mask):
    y_true = tf.cast(y_true, y_pred.dtype)
    mask = tf.cast(mask, y_pred.dtype)
    squared_difference = tf.square(y_true - y_pred)
    masked_squared = squared_difference * mask
    sum_masked = tf.reduce_sum(masked_squared, axis=-1)
    sum_mask = tf.reduce_sum(mask, axis=-1)
    return tf.reduce_mean(sum_masked / (sum_mask + 1e-7))


def monotonicity_penalty_loss(y_pred, mask):
    mask = tf.cast(mask, y_pred.dtype)
    diffs = y_pred[:, 1:] - y_pred[:, :-1]
    diff_mask = mask[:, 1:] * mask[:, :-1]
    violations = tf.nn.relu(diffs) * diff_mask # Penalize I_i+1 > I_i (current increasing with voltage)
    sum_violation = tf.reduce_sum(tf.square(violations), axis=-1)
    sum_mask = tf.reduce_sum(diff_mask, axis=-1)
    return tf.reduce_mean(sum_violation / (sum_mask + 1e-7))


def curvature_penalty_loss(y_pred, mask):
    mask = tf.cast(mask, y_pred.dtype)
    # d2I/dV2 should generally be positive (concave up for typical solar cell I-V curve where I is positive and decreases)
    # If y_pred is normalized (0,1) and represents current from Isc down to ~0,
    # then the curve should be convex (d2y/dx2 < 0 or small positive).
    # The original instruction implies penalizing large abs(curvature).
    curvature = y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2]
    curvature_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    # Using tf.square as per original, can be changed to tf.abs if L1 preferred
    sum_curvature = tf.reduce_sum(tf.square(curvature) * curvature_mask, axis=-1)
    sum_mask = tf.reduce_sum(curvature_mask, axis=-1)
    return tf.reduce_mean(sum_curvature / (sum_mask + 1e-7))


def jsc_voc_penalty_loss(y_true, y_pred, mask, original_lengths):
    y_true = tf.cast(y_true, y_pred.dtype)
    mask = tf.cast(mask, y_pred.dtype)
    orig_len = tf.cast(original_lengths, tf.int32)

    jsc_true = y_true[:, 0]
    jsc_pred = y_pred[:, 0]
    jsc_mask = mask[:, 0]
    jsc_mse = tf.reduce_mean(tf.square(jsc_true - jsc_pred) * jsc_mask)

    batch_indices = tf.range(tf.shape(y_pred)[0])
    last_indices = tf.clip_by_value(orig_len - 1, 0, tf.shape(y_pred)[1] - 1)
    gather_indices = tf.stack([batch_indices, last_indices], axis=1)

    last_pred_current_norm = tf.gather_nd(y_pred, gather_indices) # This is normalized current
    # Voc means current is zero. In normalized space, this also means current is zero (or close to feature_range[0] of MinMaxScaler)
    voc_mask = tf.cast(orig_len > 0, y_pred.dtype) # Only apply if curve has points
    # The penalty is on the normalized current at Voc point, which should be close to 0.
    voc_loss = tf.reduce_sum(tf.square(last_pred_current_norm) * voc_mask) / (tf.reduce_sum(voc_mask) + 1e-7)

    return jsc_mse, voc_loss


def full_masked_physics_loss(loss_weights):
    def loss_fn(y_true_combined, y_pred):
        y_true = y_true_combined['y_true']
        mask = y_true_combined['mask']
        orig_len = y_true_combined['orig_len']

        mse_part = masked_mse_loss(y_true, y_pred, mask)
        mono_part = monotonicity_penalty_loss(y_pred, mask)
        curv_part = curvature_penalty_loss(y_pred, mask)
        jsc_part, voc_part = jsc_voc_penalty_loss(y_true, y_pred, mask, orig_len)

        total = (
            loss_weights['mse'] * mse_part +
            loss_weights['monotonicity'] * mono_part +
            loss_weights['curvature'] * curv_part +
            loss_weights['jsc'] * jsc_part +
            loss_weights['voc'] * voc_part
        )
        return total
    return loss_fn


# --- NN Model with Voltage Embedding (wrapped tf.expand_dims) -------------------------

def build_nn_core(input_dim_params, seq_len, voltage_embed_dim=16):
    x_params_in = Input(shape=(input_dim_params,), name="X_params")
    param_path = Dense(128, activation='relu')(x_params_in)
    param_path = BatchNormalization()(param_path)
    param_path = Dropout(0.3)(param_path)
    param_path = Dense(128, activation='relu')(param_path)

    voltage_grid_in = Input(shape=(seq_len,), name="voltage_grid")
    norm_voltage_grid = Lambda(
        lambda v: v / ASSUMED_ORIGINAL_MAX_VOLTAGE,
        name="voltage_normalization"
    )(voltage_grid_in)
    norm_voltage_grid_expanded = Lambda(
        lambda v: tf.expand_dims(v, axis=-1),
        name="voltage_expand_dims"
    )(norm_voltage_grid)

    v_embed = Dense(voltage_embed_dim, activation="relu", name="voltage_embed_dense1")(norm_voltage_grid_expanded)
    v_embed = Dense(voltage_embed_dim, activation="relu", name="voltage_embed_dense2")(v_embed)

    param_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1), name="param_expand_dims")(param_path)
    param_tiled = Lambda(lambda x: tf.tile(x, [1, seq_len, 1]), name="param_tile")(param_expanded)

    merged = Concatenate(axis=-1)([param_tiled, v_embed])

    x = Dense(256, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    iv_output = Dense(1, activation='sigmoid', name='iv_point_output')(x)
    iv_output_flat = Lambda(lambda t: tf.squeeze(t, axis=-1), name='iv_output_flat')(iv_output)

    return Model(inputs=[x_params_in, voltage_grid_in], outputs=iv_output_flat, name="NN_Core")


class PhysicsNNModel(tf.keras.Model):
    def __init__(self, nn_core_model, loss_fn_config, **kwargs):
        super().__init__(**kwargs)
        self.nn_core_model = nn_core_model
        self.loss_fn_config = loss_fn_config
        self.custom_loss_fn = full_masked_physics_loss(loss_fn_config)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae") # Tracks MAE on normalized scale

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data):
        inputs_dict, y_true_for_loss = data
        with tf.GradientTape() as tape:
            y_pred = self.nn_core_model(inputs_dict, training=True)
            loss = self.custom_loss_fn(y_true_for_loss, y_pred)

        grads = tape.gradient(loss, self.nn_core_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn_core_model.trainable_variables))

        self.loss_tracker.update_state(loss)
        # MAE on normalized y_true vs y_pred
        self.mae_metric.update_state(y_true_for_loss['y_true'], y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    def test_step(self, data):
        inputs_dict, y_true_for_loss = data
        y_pred = self.nn_core_model(inputs_dict, training=False)
        loss = self.custom_loss_fn(y_true_for_loss, y_pred)
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y_true_for_loss['y_true'], y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    @property
    def metrics(self): return [self.loss_tracker, self.mae_metric]
    def call(self, inputs, training=False): return self.nn_core_model(inputs, training=training)

    def save(self, filepath, overwrite=True, **kwargs):
        Path(filepath).mkdir(parents=True, exist_ok=True)
        self.nn_core_model.save(Path(filepath) / "core_model.keras", overwrite=overwrite, **kwargs)
        joblib.dump(self.loss_fn_config, Path(filepath) / "loss_fn_config.joblib")
        print(f"PhysicsNNModel saved to {filepath}")

    @classmethod
    def load_custom_model(cls, filepath):
        core_model = tf.keras.models.load_model(Path(filepath) / "core_model.keras", compile=False)
        loss_fn_config = joblib.load(Path(filepath) / "loss_fn_config.joblib")
        loaded = cls(nn_core_model=core_model, loss_fn_config=loss_fn_config)
        print(f"PhysicsNNModel loaded from {filepath}")
        return loaded


# --- TruncatedIVReconstructor ------------------------

class TruncatedIVReconstructor:
    def __init__(self, use_gpu_if_available=True, use_global_scaling=False): # Added use_global_scaling
        self.input_preprocessor = None
        self.scalar_feature_preprocessor = None
        self.scalar_features_df_columns_ = None # To store scalar feature names

        self.final_scalers_for_clean_data = None
        self.final_original_lengths_for_clean_data = None
        self.max_truncated_len_overall = None

        self.X_clean = None
        self.y_clean_norm_padded = None
        self.padded_voltages_all_clean = None
        self.masks_all_clean = None
        self.y_clean_raw_truncated_curves = None
        self.y_clean_raw_truncated_voltages = None

        self.pca_model = None
        self.models = {}
        self.outlier_detector = None
        self.param_group_names = ['material', 'device', 'operating', 'reference', 'loss']
        
        # Improvement 4.1: Update loss weights
        self.loss_weights_nn = {
            'mse': 1.0, 'monotonicity': 0.05, 'curvature': 0.05,  # Bumped curvature
            'jsc': 0.2, 'voc': 0.1
        }

        self.use_gpu = use_gpu_if_available
        self.cuml_ok = CUML_AVAILABLE if self.use_gpu else False
        self.xgb_gpu_ok = XGBOOST_GPU_AVAILABLE if self.use_gpu else False
        self.use_global_scaling = use_global_scaling # Improvement 2.1
        print(f"Reconstructor Config: GPU PCA: {self.cuml_ok}, GPU XGBoost: {self.xgb_gpu_ok}, Global Scaling: {self.use_global_scaling}")

    # Improvement 1.1: Add truncation_threshold_pct parameter
    def _truncate_filter_normalize_iv_data(self, params_df_raw, iv_data_np_raw,
                                           min_len_for_processing=5, min_len_savgol=5,
                                           truncation_threshold_pct=0.0):
        print(f"Truncating (threshold_pct={truncation_threshold_pct}), filtering, and normalizing I-V data...")
        full_voltage_grid = np.linspace(0, ASSUMED_ORIGINAL_MAX_VOLTAGE, ASSUMED_ORIGINAL_IV_POINTS)

        # Lists for collecting data during the first pass (truncation & SavGol)
        collected_raw_currents = []
        collected_raw_voltages = []
        collected_original_lengths = []
        collected_isc_raw = []
        collected_vknee_raw = []
        collected_voc_ref_raw = []
        collected_valid_indices = []
        
        all_current_points_for_global_scaler = [] # Improvement 2.2: Accumulate for global scaler

        # First pass: Truncate, filter, apply SavGol, and collect raw data
        for i in range(iv_data_np_raw.shape[0]):
            curve_current_raw = iv_data_np_raw[i]

            # Improvement 1.1: New truncation logic
            # Ensure Isc (curve_current_raw[0]) is not zero to prevent issues with threshold calculation.
            # If Isc is zero or very small, threshold might become zero, leading to no truncation if currents are negative.
            isc_val = curve_current_raw[0] if len(curve_current_raw) > 0 else 0.0
            # Define truncation_threshold, handle case where isc_val might be negative or zero.
            # If Isc is positive, threshold will be negative (e.g., -0.03 * Isc).
            # If Isc is zero or negative, a small negative absolute threshold might be better, or rely on first negative.
            if isc_val > 1e-6 : # Isc is positive and meaningful
                 truncation_threshold = truncation_threshold_pct * isc_val
            else: # Isc is zero or negative, or too small. Fallback to slight negative current as threshold.
                 truncation_threshold = -0.001 # A small absolute negative current
            
            first_crossing_idx = -1
            for k_idx, k_val in enumerate(curve_current_raw):
                if k_val < truncation_threshold:
                    first_crossing_idx = k_idx
                    break
            trunc_idx = first_crossing_idx if first_crossing_idx != -1 else len(curve_current_raw)

            if trunc_idx < min_len_for_processing:
                continue

            current_truncated = curve_current_raw[:trunc_idx].copy()
            voltage_truncated = full_voltage_grid[:trunc_idx].copy()

            if current_truncated.size == 0: # Should be caught by min_len_for_processing
                continue
            
            current_len = len(current_truncated) # Use this for SavGol

            if current_len >= min_len_savgol:
                # Ensure window_length (wl) is odd, less than current_len, and polyorder < wl.
                wl = min(max(min_len_savgol, (current_len // 10) * 2 + 1), current_len)
                if wl % 2 == 0: wl -=1 # make it odd
                wl = max(3, wl) # must be at least 3
                
                polyorder = min(2, wl - 1) # polyorder must be less than window_length
                if polyorder < 0: polyorder = 0 # handle wl < 2 edge case (should not happen if wl >=3)

                if current_len > wl and wl >=3 : # Savgol only if enough points and valid params
                    current_truncated = savgol_filter(current_truncated, window_length=wl, polyorder=polyorder)
                # else: current_truncated remains as is (not filtered)
            
            collected_raw_currents.append(current_truncated)
            collected_raw_voltages.append(voltage_truncated)
            collected_original_lengths.append(current_len)
            
            collected_isc_raw.append(current_truncated[0] if current_len > 0 else 0.0)
            collected_vknee_raw.append(voltage_truncated[-1] if current_len > 0 else 0.0) # Voltage at truncation point
            collected_voc_ref_raw.append(params_df_raw.iloc[i]['Voc_ref'])
            collected_valid_indices.append(i)

            if self.use_global_scaling and current_truncated.size > 0: # Improvement 2.2
                all_current_points_for_global_scaler.extend(current_truncated)

        if not collected_raw_currents: # No valid curves found
            print("Warning: No valid curves after truncation and filtering.")
            return None

        # Improvement 2.2: Compute global scaler (if enabled)
        global_scaler = None
        if self.use_global_scaling:
            if all_current_points_for_global_scaler:
                global_scaler = MinMaxScaler(feature_range=(0, 1))
                global_scaler.fit(np.array(all_current_points_for_global_scaler).reshape(-1, 1))
                print(f"  Global scaler fitted. Data range: {global_scaler.data_min_[0]:.2f} to {global_scaler.data_max_[0]:.2f}")
            else:
                print("Warning: Global scaling enabled, but no data points found to fit the scaler. Will use per-curve scaling.")
                # Fallback to per-curve scaling if global_scaler cannot be fitted
                self.use_global_scaling = False # Temporarily disable for this call if fitting failed

        # Second pass: Normalize curves
        final_normalized_currents = []
        final_scalers_list = []

        for current_to_norm in collected_raw_currents:
            if self.use_global_scaling and global_scaler: # Improvement 2.3
                # Check for constant array before transform if scaler might have issues.
                # MinMaxScaler handles constant arrays by scaling them to min_range (0.0 by default).
                norm_current = global_scaler.transform(current_to_norm.reshape(-1, 1)).flatten()
                final_scalers_list.append(global_scaler)
            else: # Per-curve scaling
                scaler = MinMaxScaler(feature_range=(0, 1))
                # MinMaxScaler handles constant arrays appropriately (scales to 0.0 if min=max)
                norm_current = scaler.fit_transform(current_to_norm.reshape(-1, 1)).flatten()
                final_scalers_list.append(scaler)
            final_normalized_currents.append(norm_current)
            
        max_len_after_trunc = max(collected_original_lengths) if collected_original_lengths else 0
        num_valid_curves = len(final_normalized_currents)

        # Pad normalized currents and corresponding voltages
        y_norm_trunc_padded_np = np.zeros((num_valid_curves, max_len_after_trunc))
        padded_voltages_np = np.zeros((num_valid_curves, max_len_after_trunc))
        mask_matrix_np = np.zeros((num_valid_curves, max_len_after_trunc), dtype=np.float32)

        for i in range(num_valid_curves):
            norm_curve = final_normalized_currents[i]
            volt_curve = collected_raw_voltages[i] # Use voltages corresponding to this curve
            curr_len = len(norm_curve) # Should be same as collected_original_lengths[i]

            y_norm_trunc_padded_np[i, :curr_len] = norm_curve
            padded_voltages_np[i, :curr_len] = volt_curve 
            mask_matrix_np[i, :curr_len] = 1.0

            # Pad with last value to avoid sudden jumps if model predicts full length
            if curr_len > 0 and curr_len < max_len_after_trunc:
                y_norm_trunc_padded_np[i, curr_len:] = norm_curve[-1]
                padded_voltages_np[i, curr_len:] = volt_curve[-1]
        
        # Improvement 3.1: Add more scalar features
        scalar_features_df = pd.DataFrame({
            'Isc_raw': collected_isc_raw,
            'Vknee_raw': collected_vknee_raw, # This is V at truncation point
            'Voc_ref_raw': collected_voc_ref_raw, # From input parameters
            'Imax_raw': [np.max(c) if len(c) > 0 else 0.0 for c in collected_raw_currents],
            'Imin_raw': [np.min(c) if len(c) > 0 else 0.0 for c in collected_raw_currents],
            'Imean_raw': [np.mean(c) if len(c) > 0 else 0.0 for c in collected_raw_currents],
        })

        return {
            "y_norm_padded": y_norm_trunc_padded_np,
            "padded_voltages": padded_voltages_np,
            "masks": mask_matrix_np,
            "scalers": final_scalers_list, # List of scalers (global or local)
            "lengths": collected_original_lengths, # List of original lengths post-truncation
            "raw_currents": collected_raw_currents, # List of raw (truncated, SavGol'd) current arrays
            "raw_voltages": collected_raw_voltages, # List of raw (truncated) voltage arrays
            "scalar_features_df": scalar_features_df,
            "valid_indices": collected_valid_indices,
            "max_len": max_len_after_trunc
        }

    def _preprocess_input_parameters(self, params_df):
        print("Preprocessing input parameters (device, material, etc.)...")
        const_tol = 1e-10
        constant_cols = [c for c in params_df.columns if params_df[c].std(ddof=0) <= const_tol]
        if constant_cols:
            print(f"  Removing constant columns from X_params: {constant_cols}")
            params_df = params_df.drop(columns=constant_cols)

        param_definitions = {
            'material': ['Eg', 'NCv', 'NCc', 'mu_e', 'mu_h', 'eps'],
            'device': ['A', 'Cn', 'Cp', 'Nt', 'Et', 'nD', 'nA', 'thickness', 'T', 'Sn', 'Sp', 'Rs', 'Rsh'],
            'operating': ['G', 'light_intensity'],
            'reference': ['Voc_ref', 'Jsc_ref', 'FF_ref', 'PCE_ref'],
            'loss': ['Qe_loss', 'R_loss', 'SRH_loss', 'series_loss', 'shunt_loss', 'other_loss']
        }
        transformers = []
        current_cols = params_df.columns.tolist()
        for group_name in self.param_group_names:
            group_cols_def = param_definitions.get(group_name, [])
            actual_cols = [c for c in group_cols_def if c in current_cols]
            if not actual_cols: continue
            steps = []
            if group_name == 'material': # Example: Log transform for material properties
                steps.append(('log_transform', FunctionTransformer(np.log1p, validate=False)))
            steps.append(('scaler', RobustScaler())) # RobustScaler is good for features with outliers
            transformers.append((group_name, Pipeline(steps), actual_cols))

        self.input_preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        X_processed = self.input_preprocessor.fit_transform(params_df)
        print(f"  Processed device parameters shape: {X_processed.shape}")
        return X_processed

    def _preprocess_scalar_features(self, scalar_features_df, fit=True):
        # Using name from prompt for consistency: 'Isc_raw', 'Vknee_raw', 'Voc_ref_raw', 'Imax_raw', 'Imin_raw', 'Imean_raw'
        print(f"Preprocessing scalar features ({', '.join(scalar_features_df.columns.tolist())})...")
        if fit:
            self.scalar_features_df_columns_ = scalar_features_df.columns.tolist() # Store for get_feature_names_out
            self.scalar_feature_preprocessor = StandardScaler()
            X_scalar_processed = self.scalar_feature_preprocessor.fit_transform(scalar_features_df)
        else:
            if not self.scalar_feature_preprocessor:
                raise RuntimeError("Scalar feature preprocessor not fitted.")
            X_scalar_processed = self.scalar_feature_preprocessor.transform(scalar_features_df)
        print(f"  Processed scalar features shape: {X_scalar_processed.shape}")
        return X_scalar_processed

    def _remove_outlier_samples(self, data_dict, contamination=0.05):
        print("Detecting and removing outlier samples...")
        X_full = data_dict['X_augmented']
        # Use raw scalar features for outlier detection as they are more interpretable
        # and less prone to scaling artifacts affecting outlier scores.
        # The original code used a mix of normalized curve stats and raw scalar features.
        # Using only raw scalar features from scalar_features_df:
        features_for_outlier_df = data_dict['scalar_features_df'][['Isc_raw', 'Vknee_raw', 'Imax_raw', 'Imin_raw', 'Imean_raw']]

        self.outlier_detector = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100, n_jobs=-1
        )
        outlier_labels = self.outlier_detector.fit_predict(features_for_outlier_df)
        normal_mask = outlier_labels == 1
        n_outliers = np.sum(~normal_mask)
        print(f"  Removed {n_outliers} outlier samples ({n_outliers/len(outlier_labels)*100:.1f}%) using scalar IV features.")

        cleaned = {}
        cleaned['X_clean_augmented'] = X_full[normal_mask]
        cleaned['y_clean_norm_padded'] = data_dict['y_norm_padded'][normal_mask]
        cleaned['padded_voltages_clean'] = data_dict['padded_voltages'][normal_mask]
        cleaned['masks_clean'] = data_dict['masks'][normal_mask]
        cleaned['scalers_clean'] = [s for s, m in zip(data_dict['scalers'], normal_mask) if m]
        cleaned['lengths_clean'] = [l for l, m in zip(data_dict['lengths'], normal_mask) if m]
        cleaned['raw_currents_clean'] = [c for c, m in zip(data_dict['raw_currents'], normal_mask) if m]
        cleaned['raw_voltages_clean'] = [v for v, m in zip(data_dict['raw_voltages'], normal_mask) if m]
        # Filter scalar_features_df as well, as it's used by main_bulk_enhanced for X_val construction
        cleaned['scalar_features_df_clean'] = data_dict['scalar_features_df'].iloc[normal_mask].reset_index(drop=True)


        if not cleaned['lengths_clean']:
            print("Warning: All samples removed as outliers or no valid samples to begin with.")
            return None

        cleaned['max_len_clean'] = max(cleaned['lengths_clean']) if cleaned['lengths_clean'] else 0
        return cleaned

    # Modified to accept truncation_threshold_pct
    def load_and_prepare_data(self, truncation_threshold_pct=0.0):
        print(f"=== Loading and Preparing Data (Truncation PCT: {truncation_threshold_pct}, Global Scaling: {self.use_global_scaling}) ===")
        if not check_data_files_exist():
            print(f"Data files not found: {INPUT_FILE}, {OUTPUT_FILE}")
            return False
        try:
            params_df_raw_full = pd.read_csv(INPUT_FILE, header=None, names=COLNAMES if COLNAMES else None)
            iv_data_np_raw_full = np.loadtxt(OUTPUT_FILE, delimiter=',')
            print(f"Loaded raw data: Parameters={params_df_raw_full.shape}, I-V curves={iv_data_np_raw_full.shape}")

            iv_bundle = self._truncate_filter_normalize_iv_data(
                params_df_raw_full, iv_data_np_raw_full,
                truncation_threshold_pct=truncation_threshold_pct # Pass the parameter
            )
            if iv_bundle is None or not iv_bundle.get('valid_indices'):
                print("Data processing returned no valid IV curves.")
                return False

            params_df_filtered = params_df_raw_full.iloc[iv_bundle['valid_indices']].reset_index(drop=True)
            # This is the unscaled scalar_features_df for all valid (pre-outlier) curves
            scalar_features_df_unscaled_valid = iv_bundle['scalar_features_df'].reset_index(drop=True)

            X_params_processed = self._preprocess_input_parameters(params_df_filtered)
            X_scalar_processed = self._preprocess_scalar_features(scalar_features_df_unscaled_valid, fit=True)

            X_augmented_full = np.concatenate([X_params_processed, X_scalar_processed], axis=1)
            print(f"  Augmented X shape (pre-outlier removal): {X_augmented_full.shape}")

            data_for_outlier = {
                'X_augmented': X_augmented_full, # Augmented features for valid curves
                'y_norm_padded': iv_bundle['y_norm_padded'], # Normalized IVs for valid curves
                'padded_voltages': iv_bundle['padded_voltages'],
                'masks': iv_bundle['masks'],
                'scalers': iv_bundle['scalers'],
                'lengths': iv_bundle['lengths'],
                'raw_currents': iv_bundle['raw_currents'],
                'raw_voltages': iv_bundle['raw_voltages'],
                'scalar_features_df': scalar_features_df_unscaled_valid # Unscaled scalar features for valid curves
            }

            cleaned_data = self._remove_outlier_samples(data_for_outlier, contamination=0.05)
            if cleaned_data is None or not cleaned_data['lengths_clean']:
                print("No data left after outlier removal.")
                return False

            self.X_clean = cleaned_data['X_clean_augmented']
            self.y_clean_norm_padded = cleaned_data['y_clean_norm_padded']
            self.padded_voltages_all_clean = cleaned_data['padded_voltages_clean']
            self.masks_all_clean = cleaned_data['masks_clean']
            self.final_scalers_for_clean_data = cleaned_data['scalers_clean']
            self.final_original_lengths_for_clean_data = cleaned_data['lengths_clean']
            self.y_clean_raw_truncated_curves = cleaned_data['raw_currents_clean']
            self.y_clean_raw_truncated_voltages = cleaned_data['raw_voltages_clean']
            # Storing the cleaned scalar_features_df might be useful for inspection or other tasks.
            self.scalar_features_df_clean_ = cleaned_data['scalar_features_df_clean'] 
            self.max_truncated_len_overall = cleaned_data['max_len_clean']

            print(f"Final X_clean shape: {self.X_clean.shape}")
            print(f"Final y_clean_norm_padded shape: {self.y_clean_norm_padded.shape}")
            print(f"Max truncated length: {self.max_truncated_len_overall} points.")
            if self.max_truncated_len_overall == 0:
                print("Warning: Max truncated length is 0. No data for modeling.")
                return False
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback; traceback.print_exc()
            return False

    def fit_pca_model(self, X_train_np, y_train_norm_padded_np, n_components=15):
        actual_n_components = min(n_components, self.max_truncated_len_overall, X_train_np.shape[0])
        if actual_n_components < n_components:
            print(f"  PCA n_components adjusted from {n_components} to {actual_n_components} due to data dimensions.")
        if actual_n_components <= 1:
            print(f"Warning: PCA cannot be effectively fitted with {actual_n_components}. Skipping PCA.")
            self.models['pca'] = None
            return self

        print(f"\n=== Fitting PCA Model (n_components={actual_n_components}) ===")
        if self.cuml_ok:
            print("Using cuML PCA (GPU)...")
            y_cu = cupy.asarray(y_train_norm_padded_np)
            self.pca_model = CumlPCA(n_components=actual_n_components)
            y_pca_cu = self.pca_model.fit_transform(y_cu)
            y_pca_np = cupy.asnumpy(y_pca_cu)
            del y_cu, y_pca_cu; cupy.get_default_memory_pool().free_all_blocks()
        else:
            print("Using scikit-learn PCA (CPU)...")
            self.pca_model = SklearnPCA(n_components=actual_n_components)
            y_pca_np = self.pca_model.fit_transform(y_train_norm_padded_np)

        print(f"PCA Explained Variance (Top {min(10, actual_n_components)}): {self.pca_model.explained_variance_ratio_[:min(10, actual_n_components)]}")
        print(f"Total variance by {actual_n_components}: {np.sum(self.pca_model.explained_variance_ratio_):.4f}")

        self.pca_regressors = []
        regressor_params = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.8, 'random_state': 42}

        for i in range(actual_n_components):
            print(f"Training regressor for PC{i+1}...")
            if self.xgb_gpu_ok:
                reg = BaseXGBRegressor(**regressor_params, tree_method='gpu_hist', n_jobs=-1)
            elif BaseXGBRegressor == SklearnGradientBoostingRegressor: # Scikit-learn GBR
                reg = SklearnGradientBoostingRegressor(**regressor_params, validation_fraction=0.1, n_iter_no_change=10)
            else: # XGBoost CPU
                reg = BaseXGBRegressor(**regressor_params, tree_method='hist', n_jobs=-1)
            reg.fit(X_train_np, y_pca_np[:, i])
            self.pca_regressors.append(reg)
        self.models['pca'] = {'pca_obj': self.pca_model, 'regressors': self.pca_regressors}
        print("PCA model fitting completed!")
        return self

    def fit_physics_informed_nn(
        self, X_train_np, y_train_norm_padded_np, V_train_padded_voltages, Mask_train, Origlen_train,
        X_val_np, y_val_norm_padded_np, V_val_padded_voltages, Mask_val, Origlen_val,
        epochs=200, batch_size=128 # Added epochs, batch_size for flexibility
    ):
        print("\n=== Fitting Physics-Informed Neural Network (Enhanced) ===")
        input_dim_params = X_train_np.shape[1]
        seq_len = self.max_truncated_len_overall
        if seq_len == 0:
            print("Warning: Output dimension for NN is 0. Skipping.")
            self.models['physics_nn'] = None
            return self

        nn_core = build_nn_core(input_dim_params, seq_len, voltage_embed_dim=16)
        physics_model = PhysicsNNModel(nn_core, self.loss_weights_nn)
        physics_model.compile(optimizer=Adam(learning_rate=1e-3)) # MAE metric is auto-added by custom model

        train_inputs = {"X_params": X_train_np, "voltage_grid": V_train_padded_voltages}
        train_targets = {"y_true": y_train_norm_padded_np, "mask": Mask_train, "orig_len": Origlen_train}
        train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
        train_ds = train_ds.shuffle(buffer_size=min(X_train_np.shape[0], 10000)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = None
        if X_val_np is not None and len(X_val_np) > 0:
            val_inputs = {"X_params": X_val_np, "voltage_grid": V_val_padded_voltages}
            val_targets = {"y_true": y_val_norm_padded_np, "mask": Mask_val, "orig_len": Origlen_val}
            val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets))
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else: print("  Warning: No validation set provided for NN.")

        callbacks = [
            EarlyStopping(monitor='val_loss' if val_ds else 'loss', patience=30, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss' if val_ds else 'loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1)
        ]
        self.training_history = physics_model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=1)
        self.models['physics_nn'] = physics_model
        return self

    def fit_ensemble_model(self, X_train_np, y_train_norm_padded_np):
        print("\n=== Fitting Ensemble Model (Enhanced X, Padded Y) ===")
        if self.max_truncated_len_overall == 0:
            print("Warning: Output dim for ensemble is 0. Skipping.")
            self.models['ensemble'] = None; return self

        regressor_params = {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 8, 'subsample': 0.7, 'random_state': 42}
        base_configs = []
        if self.xgb_gpu_ok:
            print("  Using XGBoost (GPU) for ensemble members...")
            base_configs.append(('xgb1', BaseXGBRegressor(**regressor_params, tree_method='gpu_hist', n_jobs=-1)))
        elif BaseXGBRegressor != SklearnGradientBoostingRegressor: # XGBoost CPU
            print("  Using XGBoost (CPU) for ensemble members...")
            base_configs.append(('xgb1_cpu', BaseXGBRegressor(**regressor_params, tree_method='hist', n_jobs=-1)))
        else: # Fallback to RandomForest
            print("  Using RandomForestRegressor (CPU) for ensemble...")
            base_configs.append(('rf_cpu', SklearnRandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)))

        self.ensemble_models = {}
        for name, base_model in base_configs:
            print(f"  Training {name.upper()} ensemble member…")
            multi = MultiOutputRegressor(base_model, n_jobs=1) # MultiOutputRegressor does not accept n_jobs=-1 for some base estimators
            multi.fit(X_train_np, y_train_norm_padded_np)
            self.ensemble_models[name] = multi
        self.models['ensemble'] = self.ensemble_models
        return self

    def predict(self, X_np, V_padded_voltages_np, scalers_for_X, original_lengths_for_X, model_type='pca'):
        if model_type not in self.models or self.models[model_type] is None:
            print(f"Model {model_type} not fitted. Returning empty.")
            return [np.array([]) for _ in range(X_np.shape[0])]

        M = self.models[model_type]
        y_pred_norm_padded = None

        if model_type == 'pca':
            pca_obj, regs = M['pca_obj'], M['regressors']
            if not regs: return [np.array([]) for _ in range(X_np.shape[0])]
            y_pca_pred = np.array([r.predict(X_np) for r in regs]).T
            if self.cuml_ok and isinstance(pca_obj, CumlPCA):
                y_pred_norm_padded = cupy.asnumpy(pca_obj.inverse_transform(cupy.asarray(y_pca_pred)))
            else:
                y_pred_norm_padded = pca_obj.inverse_transform(y_pca_pred)
        elif model_type == 'physics_nn':
            y_pred_norm_padded = M.predict({"X_params": X_np, "voltage_grid": V_padded_voltages_np}, batch_size=512, verbose=0)
        elif model_type == 'ensemble':
            preds = [model.predict(X_np) for _, model in M.items()]
            y_pred_norm_padded = np.mean(preds, axis=0)

        y_pred_orig_trunc_list = []
        for i in range(y_pred_norm_padded.shape[0]):
            norm_curve, scaler, orig_len = y_pred_norm_padded[i], scalers_for_X[i], original_lengths_for_X[i]
            if orig_len == 0: y_pred_orig_trunc_list.append(np.array([])); continue
            # Ensure scaler is fitted, global scaler might be passed without fitting if no data.
            # However, earlier logic should prevent this by falling back to per-curve or erroring.
            if not hasattr(scaler, 'data_min_') and not hasattr(scaler, 'scale_'): # Basic check if scaler is fitted
                print(f"Warning: Scaler for sample {i} appears unfitted. Global scaler issue? Using identity.")
                full_curve_unscaled = norm_curve * 1.0 # Bogus inverse transform
            else:
                full_curve_unscaled = scaler.inverse_transform(norm_curve.reshape(-1, 1)).flatten()
            y_pred_orig_trunc_list.append(full_curve_unscaled[:orig_len])
        return y_pred_orig_trunc_list

    def evaluate_model(self, X_test, V_test_padded_voltages, y_test_orig_truncated_curves_list,
                       scalers_test, original_lengths_test, model_type='pca'):
        print(f"\n=== Evaluating {model_type.upper()} (Enhanced Truncated) ===")
        y_pred_list = self.predict(X_test, V_test_padded_voltages, scalers_test, original_lengths_test, model_type)

        if not y_test_orig_truncated_curves_list or not y_pred_list:
            print("  No data for evaluation."); return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'per_curve_R2_mean': np.nan, 'predictions': []}

        flat_true, flat_pred = [], []
        for t, p in zip(y_test_orig_truncated_curves_list, y_pred_list):
            if len(t)>0 and len(t)==len(p): # Ensure comparable
                 flat_true.extend(t)
                 flat_pred.extend(p)
        flat_true, flat_pred = np.array(flat_true), np.array(flat_pred)

        if flat_true.size == 0: mae, rmse, r2_global = np.nan, np.nan, np.nan
        else:
            mae = mean_absolute_error(flat_true, flat_pred)
            rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
            r2_global = r2_score(flat_true, flat_pred) if np.std(flat_true) > 1e-9 else (1.0 if mean_squared_error(flat_true, flat_pred) < 1e-6 else 0.0)
        print(f"Truncated Original scale: MAE: {mae:.6f}, RMSE: {rmse:.6f}, Global R²: {r2_global:.4f}")

        per_curve_r2 = []
        for i in range(len(y_test_orig_truncated_curves_list)):
            true_c, pred_c = y_test_orig_truncated_curves_list[i], y_pred_list[i]
            if len(true_c) > 1 and len(pred_c) == len(true_c):
                r2_val = r2_score(true_c, pred_c) if np.std(true_c) > 1e-9 else (1.0 if mean_squared_error(true_c, pred_c) < 1e-6 else 0.0)
                per_curve_r2.append(r2_val)
        
        mean_r2_pc, std_r2_pc = (np.mean(per_curve_r2), np.std(per_curve_r2)) if per_curve_r2 else (np.nan, np.nan)
        print(f"  Per-curve R² (truncated) - Mean: {mean_r2_pc:.4f}, Std: {std_r2_pc:.4f}")
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2_global, 'per_curve_R2_mean': mean_r2_pc, 'per_curve_R2_std': std_r2_pc if per_curve_r2 else np.nan, 'predictions': y_pred_list}

    def plot_results(self, X_test, V_test_padded_voltages, y_test_orig_truncated_curves_list, y_test_orig_truncated_voltages_list,
                     scalers_test, original_lengths_test, model_type='pca', n_samples=4, suffix=""):
        y_pred_list = self.predict(X_test, V_test_padded_voltages, scalers_test, original_lengths_test, model_type)
        num_avail = len(y_test_orig_truncated_curves_list);
        if num_avail == 0: print(f"No samples to plot for {model_type}."); return

        indices = np.random.choice(num_avail, min(n_samples, num_avail), replace=False)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10)); axes = axes.ravel()
        for i, idx in enumerate(indices):
            ax, true_c, true_v, pred_c = axes[i], y_test_orig_truncated_curves_list[idx], y_test_orig_truncated_voltages_list[idx], y_pred_list[idx]
            if not (len(true_c) > 0 and len(pred_c) > 0 and len(true_c) == len(pred_c) and len(true_c) == len(true_v)):
                ax.text(0.5, 0.5, f"Sample {idx}\nMismatch/Empty", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Sample {idx} - Error"); continue
            ax.plot(true_v, true_c, 'b-', lw=2, label='Actual (Trunc)', alpha=0.8)
            ax.plot(true_v, pred_c, 'r--', lw=2, label='Predicted (Trunc)', alpha=0.8)
            curve_r2 = r2_score(true_c, pred_c) if np.std(true_c) > 1e-9 else (1.0 if mean_squared_error(true_c, pred_c) < 1e-6 else 0.0)
            ax.set_title(f"Sample {idx} - {model_type.upper()} (R²={curve_r2:.3f})"); ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Current Density (mA/cm²)")
            ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        outpath = OUTPUT_DIR_TRUNCATED / f"preds_{model_type}{suffix}.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight'); print(f"Prediction plot saved: {outpath}"); plt.close(fig)

    def plot_pca_analysis(self, suffix=""):
        if 'pca' not in self.models or not self.models['pca'] or not self.models['pca']['pca_obj']:
            print("PCA model not fitted for plotting!"); return
        pca_obj, regs = self.models['pca']['pca_obj'], self.models['pca']['regressors']
        explained_var = cupy.asnumpy(pca_obj.explained_variance_ratio_) if isinstance(pca_obj.explained_variance_ratio_, cupy.ndarray) else pca_obj.explained_variance_ratio_
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0,0].bar(range(1,len(explained_var)+1),explained_var); axes[0,0].set_title("PCA Explained Variance")
        axes[0,1].plot(range(1,len(explained_var)+1), np.cumsum(explained_var),'bo-'); axes[0,1].set_title("Cumulative Variance"); axes[0,1].axhline(0.95,c='r',ls='--'); axes[0,1].legend(['Cum. Var.', '95% Threshold'])
        components = cupy.asnumpy(pca_obj.components_) if isinstance(pca_obj.components_, cupy.ndarray) else pca_obj.components_
        for i in range(min(4, components.shape[0])): axes[1,0].plot(np.arange(components.shape[1]), components[i], label=f"PC{i+1}")
        axes[1,0].set_title("Principal Components"); axes[1,0].legend()

        if hasattr(regs[0], 'feature_importances_'):
            imp = regs[0].feature_importances_
            feature_names = []
            if self.input_preprocessor: feature_names.extend(self.input_preprocessor.get_feature_names_out())
            if self.scalar_feature_preprocessor and hasattr(self, 'scalar_features_df_columns_'):
                feature_names.extend([f"scalar__{col}" for col in self.scalar_features_df_columns_])
            
            if len(feature_names) != len(imp):
                print(f"Warning: Feature name count ({len(feature_names)}) mismatch with importance count ({len(imp)}). Using generic names.")
                feature_names = [f"F{i}" for i in range(len(imp))]
            
            top_idx = np.argsort(imp)[-10:]; labels = [feature_names[i] for i in top_idx]
            axes[1,1].barh(range(len(top_idx)), imp[top_idx]); axes[1,1].set_yticks(range(len(top_idx))); axes[1,1].set_yticklabels(labels)
            axes[1,1].set_title("Top Features for PC1 Regressor")
        else: axes[1,1].text(0.5,0.5,"Feat. importance N/A", ha='center',va='center',transform=axes[1,1].transAxes)
        plt.tight_layout(); outpath = OUTPUT_DIR_TRUNCATED / f"pca_analysis{suffix}.png"
        plt.savefig(outpath,dpi=300,bbox_inches='tight'); print(f"PCA analysis plot saved: {outpath}"); plt.close(fig)


# --- MAIN FUNCTION (SINGLE RUN) -------------------------------------------------------------
def main_single_run(trunc_thresh_pct=0.0, use_global_scaling=False):
    print(f"=== SINGLE RUN: Trunc Thresh={trunc_thresh_pct}, Global Scaling={use_global_scaling} ===")
    reconstructor = TruncatedIVReconstructor(use_gpu_if_available=True, use_global_scaling=use_global_scaling)
    if not reconstructor.load_and_prepare_data(truncation_threshold_pct=trunc_thresh_pct):
        print("Failed to load/prepare data. Exiting.")
        return None

    N = reconstructor.X_clean.shape[0]
    all_indices = np.arange(N)
    train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    X_train, y_train_norm = reconstructor.X_clean[train_idx], reconstructor.y_clean_norm_padded[train_idx]
    V_train_pad, Mask_train = reconstructor.padded_voltages_all_clean[train_idx], reconstructor.masks_all_clean[train_idx]
    Origlen_train = np.array(reconstructor.final_original_lengths_for_clean_data)[train_idx]

    X_val, y_val_norm = reconstructor.X_clean[val_idx], reconstructor.y_clean_norm_padded[val_idx]
    V_val_pad, Mask_val = reconstructor.padded_voltages_all_clean[val_idx], reconstructor.masks_all_clean[val_idx]
    Origlen_val = np.array(reconstructor.final_original_lengths_for_clean_data)[val_idx]

    X_test, V_test_pad = reconstructor.X_clean[test_idx], reconstructor.padded_voltages_all_clean[test_idx]
    scalers_test = [reconstructor.final_scalers_for_clean_data[i] for i in test_idx]
    lengths_test = [reconstructor.final_original_lengths_for_clean_data[i] for i in test_idx]
    y_test_curves = [reconstructor.y_clean_raw_truncated_curves[i] for i in test_idx]
    y_test_voltages = [reconstructor.y_clean_raw_truncated_voltages[i] for i in test_idx]

    plot_suffix = f"_thresh{trunc_thresh_pct:.2f}_glob{use_global_scaling}"

    # --- Physics-Informed NN ---
    reconstructor.fit_physics_informed_nn(
        X_train, y_train_norm, V_train_pad, Mask_train, Origlen_train,
        X_val, y_val_norm, V_val_pad, Mask_val, Origlen_val, epochs=2 # Short epochs for quick test
    )
    nn_res = None
    if reconstructor.models.get('physics_nn'):
        nn_res = reconstructor.evaluate_model(
            X_test, V_test_pad, y_test_curves, scalers_test, lengths_test, 'physics_nn'
        )
        reconstructor.plot_results(
            X_test, V_test_pad, y_test_curves, y_test_voltages, scalers_test, lengths_test, 'physics_nn', suffix=plot_suffix
        )
    else: print("Physics NN model skipped.")
    
    # Save models (optional, for single run can be skipped or adapted)
    return nn_res # or full reconstructor object


# --- MAIN BULK EXPERIMENT FUNCTION (Improvement 5) -------------------------------------
def main_bulk_enhanced():
    print("=== BULK EXPERIMENTS for Perovskite I-V Curve Reconstruction ===")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU(s) available to TensorFlow.")
        except RuntimeError as e: print(f"Error setting memory growth: {e}")
    else: print("No GPUs available to TensorFlow.")

    # Define ranges for experiments
    thresholds_to_try = np.arange(-0.03, 0.031, 0.01) # Example: -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03
    # thresholds_to_try = [0.0] # For a quicker test run
    
    best_r2_val = -np.inf # Using validation R2 for model selection
    best_config = None
    all_results = []

    for use_global_scaling_exp in [False, True]:
        for thresh_exp in thresholds_to_try:
            current_config_str = f"Trunc_thresh={thresh_exp:.3f}, Global_scaling={use_global_scaling_exp}"
            print(f"\n=== TESTING CONFIG: {current_config_str} ===")
            
            reconstructor = TruncatedIVReconstructor(
                use_gpu_if_available=True, 
                use_global_scaling=use_global_scaling_exp
            )
            # Pass truncation_threshold_pct to load_and_prepare_data
            ok = reconstructor.load_and_prepare_data(truncation_threshold_pct=thresh_exp)
            if not ok or reconstructor.X_clean is None or reconstructor.X_clean.shape[0] < 50: # Min samples for robust split
                print(f"  Skipped {current_config_str} (failed to load/prepare sufficient data: {reconstructor.X_clean.shape[0] if reconstructor.X_clean is not None else 0} samples)")
                all_results.append({'config': current_config_str, 'val_R2_curve': np.nan, 'status': 'DataPrepFailed'})
                continue

            N = reconstructor.X_clean.shape[0]
            all_indices = np.arange(N)
            # Stratified split could be better if there are classes, but not applicable here.
            # Ensure enough samples for train/val/test split.
            # test_size=0.2, val_size relative to train = 0.15 => 0.8 * 0.15 = 0.12 for val. So train=0.68, val=0.12, test=0.2
            if N < 20 : # Need at least a few samples for each set.
                 print(f"  Skipped {current_config_str} (not enough samples ({N}) for train/val/test split)")
                 all_results.append({'config': current_config_str, 'val_R2_curve': np.nan, 'status': 'NotEnoughSamplesForSplit'})
                 continue

            train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42, shuffle=True)
            # Ensure train_idx is not too small before splitting for validation
            if len(train_idx) < 10 : # Need enough for training and validation from this
                print(f"  Skipped {current_config_str} (not enough samples in initial train set ({len(train_idx)}) for val split)")
                # Use test_idx as val_idx if train_idx is too small and test_idx is larger, or skip
                # For simplicity, skipping here
                all_results.append({'config': current_config_str, 'val_R2_curve': np.nan, 'status': 'NotEnoughForTrainValSplit'})
                continue
            train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42) # 0.15 of (0.8*N)

            # Check if any split is empty
            if not all([len(train_idx)>0, len(val_idx)>0, len(test_idx)>0]):
                print(f"  Skipped {current_config_str} (one or more data splits are empty after splitting)")
                all_results.append({'config': current_config_str, 'val_R2_curve': np.nan, 'status': 'EmptySplit'})
                continue


            X_train = reconstructor.X_clean[train_idx]
            y_train_norm = reconstructor.y_clean_norm_padded[train_idx]
            V_train_pad = reconstructor.padded_voltages_all_clean[train_idx]
            Mask_train = reconstructor.masks_all_clean[train_idx]
            Origlen_train = np.array(reconstructor.final_original_lengths_for_clean_data)[train_idx]

            X_val = reconstructor.X_clean[val_idx]
            y_val_norm = reconstructor.y_clean_norm_padded[val_idx]
            V_val_pad = reconstructor.padded_voltages_all_clean[val_idx]
            Mask_val = reconstructor.masks_all_clean[val_idx]
            Origlen_val = np.array(reconstructor.final_original_lengths_for_clean_data)[val_idx]

            # For bulk experiments, focus on the Physics NN model
            # Using fewer epochs for faster iteration in bulk experiments
            reconstructor.fit_physics_informed_nn(
                X_train, y_train_norm, V_train_pad, Mask_train, Origlen_train,
                X_val, y_val_norm, V_val_pad, Mask_val, Origlen_val,
                epochs=50, batch_size=128 # Reduced epochs for bulk; adjust as needed
            )

            if reconstructor.models.get('physics_nn'):
                # Evaluate on validation set to find best hyperparameters
                val_res = reconstructor.evaluate_model(
                    X_val, V_val_pad,
                    [reconstructor.y_clean_raw_truncated_curves[i] for i in val_idx],
                    [reconstructor.final_scalers_for_clean_data[i] for i in val_idx],
                    [reconstructor.final_original_lengths_for_clean_data[i] for i in val_idx],
                    model_type='physics_nn'
                )
                val_r2_curve_mean = val_res['per_curve_R2_mean']
                print(f"  [Validation] {current_config_str}: R²/curve={val_r2_curve_mean:.4f} (MAE={val_res['MAE']:.4f})")
                
                current_run_results = {
                    'config_dict': {'threshold': thresh_exp, 'global_scaling': use_global_scaling_exp, 'curvature_penalty': reconstructor.loss_weights_nn['curvature']},
                    'config_str': current_config_str,
                    'val_R2_curve': val_r2_curve_mean,
                    'val_MAE': val_res['MAE'],
                    'status': 'Completed'
                }
                all_results.append(current_run_results)

                if not np.isnan(val_r2_curve_mean) and val_r2_curve_mean > best_r2_val:
                    best_r2_val = val_r2_curve_mean
                    best_config = current_run_results['config_dict']
                    print(f"    New best validation R²/curve: {best_r2_val:.4f} with {current_config_str}")
                    
                    # Optionally, save the best model or evaluate on test set here
                    # For now, just tracking the best config based on validation set.
            else:
                print(f"  Skipped {current_config_str} (Physics NN not fitted)")
                all_results.append({'config': current_config_str, 'val_R2_curve': np.nan, 'status': 'NNFitFailed'})
    
    print("\n" + "="*30 + " BULK EXPERIMENT SUMMARY " + "="*30)
    if best_config:
        print(f"Best config based on validation R²/curve: Threshold={best_config['threshold']:.3f}, "
              f"Global Scaling={best_config['global_scaling']}, Curvature Penalty={best_config['curvature_penalty']:.3f} "
              f"-> Validation R²/curve={best_r2_val:.4f}")
    else:
        print("No successful configurations found or all results were NaN.")

    print("\nAll results (Validation R²/curve):")
    # Header for the summary table
    print(f"{'Trunc. Threshold':<18} | {'Global Scaling':<15} | {'Curv. Penalty':<15} | {'Val R²/curve':<15} | {'Val MAE':<10} | {'Status':<20}")
    print("-" * 100)
    for res in all_results:
        if res['status'] == 'Completed':
            cfg = res['config_dict']
            print(f"{cfg['threshold']:<18.3f} | {str(cfg['global_scaling']):<15} | {cfg['curvature_penalty']:<15.3f} | "
                  f"{res['val_R2_curve']:<15.4f} | {res['val_MAE']:<10.4f} | {res['status']:<20}")
        else: # Handle failed runs
             print(f"{res['config_str']:<52} | {'N/A':<15} | {'N/A':<10} | {res['status']:<20}")
    
    # Save all_results to a file
    results_df = pd.DataFrame([
        {
            'threshold': r['config_dict']['threshold'] if 'config_dict' in r else np.nan,
            'global_scaling': r['config_dict']['global_scaling'] if 'config_dict' in r else np.nan,
            'curvature_penalty': r['config_dict']['curvature_penalty'] if 'config_dict' in r else np.nan,
            'val_R2_curve': r['val_R2_curve'] if 'val_R2_curve' in r else np.nan,
            'val_MAE': r['val_MAE'] if 'val_MAE' in r else np.nan,
            'status': r['status']
        } for r in all_results
    ])
    results_df.to_csv(OUTPUT_DIR_TRUNCATED / "bulk_experiment_results.csv", index=False)
    print(f"\nBulk experiment results saved to {OUTPUT_DIR_TRUNCATED / 'bulk_experiment_results.csv'}")


# --- Plotting for multi-model comparison (can be used by single run if needed) ---
def plot_multi_model_comparison_enhanced(y_true_curves_list, y_true_voltages_list, y_preds_dict,
                                         n_samples_plot, output_dir, suffix=""):
    # (This function is mostly unchanged from baseline, just ensuring it's callable)
    num_curves = len(y_true_curves_list)
    if num_curves == 0: return
    sample_idxs = np.random.choice(num_curves, min(n_samples_plot, num_curves), replace=False)
    n_models = len(y_preds_dict)
    if n_models == 0: return

    fig, axes = plt.subplots(len(sample_idxs), n_models + 1, figsize=(5*(n_models+1), 4*len(sample_idxs)), squeeze=False)
    for row, idx in enumerate(sample_idxs):
        true_cur, true_vol = y_true_curves_list[idx], y_true_voltages_list[idx]
        if len(true_cur) == 0 or len(true_cur) != len(true_vol):
            for col_ax in range(n_models+1): axes[row, col_ax].text(0.5,0.5, f"Sample {idx}\nInvalid", ha='center',va='center',transform=axes[row, col_ax].transAxes)
            continue
        ax0 = axes[row,0]; ax0.plot(true_vol, true_cur, 'b-', lw=2, label='True (Trunc)'); ax0.set_title(f"Sample {idx}\nGround Truth"); ax0.set_xlabel("Voltage (V)"); ax0.set_ylabel("Current"); ax0.grid(True,alpha=0.5); ax0.legend()
        for col, (mname, pred_list) in enumerate(y_preds_dict.items()):
            axp = axes[row, col+1]; pred_cur = pred_list[idx]
            if len(pred_cur) != len(true_cur): axp.text(0.5,0.5,"Length Mismatch",ha='center',va='center',transform=axp.transAxes); axp.set_title(f"{mname}\nError"); continue
            axp.plot(true_vol, true_cur, 'b-', lw=1.5, label='True', alpha=0.7); axp.plot(true_vol, pred_cur, 'r--', lw=2, label='Pred')
            r2v = r2_score(true_cur, pred_cur) if np.std(true_cur) > 1e-9 else (1.0 if mean_squared_error(true_cur, pred_cur) < 1e-6 else 0.0)
            axp.set_title(f"{mname}\nR²={r2v:.3f}"); axp.set_xlabel("Voltage (V)"); axp.grid(True,alpha=0.5); axp.legend()
    plt.tight_layout(); outfile = output_dir / f"multi_model_comparison{suffix}.png"
    plt.savefig(outfile,dpi=300,bbox_inches='tight'); print(f"\nMulti-model IV plot saved: {outfile}"); plt.close(fig)


if __name__ == "__main__":
    # To run a single configuration (e.g. with best found settings or for debugging)
    # print("Running a single configuration example...")
    # main_single_run(trunc_thresh_pct=0.0, use_global_scaling=False) # Example values

    # To run the bulk experiments
    print("Starting bulk experiments...")
    main_bulk_enhanced()