# corrected_iv_reconstruction_TRUNCATED_ENHANCED.py
# Implementation with CUDA support, scaling, I-V curve truncation,
# explicit voltage grid, masking, scalar physics features, and enhanced NN loss.

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
OUTPUT_DIR_TRUNCATED = Path("./output_data_truncated_enhanced_cuda")
OUTPUT_DIR_TRUNCATED.mkdir(parents=True, exist_ok=True)

ASSUMED_ORIGINAL_IV_POINTS = 45
ASSUMED_ORIGINAL_MAX_VOLTAGE = 1.2

def check_data_files_exist():
    return Path(INPUT_FILE).exists() and Path(OUTPUT_FILE).exists()

if not check_data_files_exist():
    print(f"Creating dummy data files for testing ({INPUT_FILE}, {OUTPUT_FILE})...")
    num_dummy_samples = 1000
    params_df_dummy = pd.DataFrame(np.random.rand(num_dummy_samples, len(COLNAMES)))
    # Ensure Voc_ref column in dummy data has plausible values
    if 'Voc_ref' in COLNAMES:
        params_df_dummy.iloc[:, COLNAMES.index('Voc_ref')] = np.random.rand(num_dummy_samples) * 0.5 + 0.5
    params_df_dummy.to_csv(INPUT_FILE, header=False, index=False)

    dummy_output_data = np.random.randn(num_dummy_samples, ASSUMED_ORIGINAL_IV_POINTS) * 0.8 + 0.3
    np.savetxt(OUTPUT_FILE, dummy_output_data, delimiter=',')
# === END: Standalone configuration ===

# Patch FunctionTransformer to always return input_features when calling get_feature_names_out
def _ft_get_feature_names_out(self, input_features=None):
    return input_features if input_features is not None else []
FunctionTransformer.get_feature_names_out = _ft_get_feature_names_out


# --- Masked Physics Loss Functions (with dtype casts) --------------------------------

def masked_mse_loss(y_true, y_pred, mask):
    # Cast y_true and mask to y_pred dtype (float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    mask = tf.cast(mask, y_pred.dtype)

    squared_difference = tf.square(y_true - y_pred)
    masked_squared = squared_difference * mask
    sum_masked = tf.reduce_sum(masked_squared, axis=-1)
    sum_mask = tf.reduce_sum(mask, axis=-1)
    return tf.reduce_mean(sum_masked / (sum_mask + 1e-7))


def monotonicity_penalty_loss(y_pred, mask):
    # Cast mask to y_pred dtype
    mask = tf.cast(mask, y_pred.dtype)

    diffs = y_pred[:, 1:] - y_pred[:, :-1]
    diff_mask = mask[:, 1:] * mask[:, :-1]
    violations = tf.nn.relu(diffs) * diff_mask
    sum_violation = tf.reduce_sum(tf.square(violations), axis=-1)
    sum_mask = tf.reduce_sum(diff_mask, axis=-1)
    return tf.reduce_mean(sum_violation / (sum_mask + 1e-7))


def curvature_penalty_loss(y_pred, mask):
    # Cast mask to y_pred dtype
    mask = tf.cast(mask, y_pred.dtype)

    curvature = y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2]
    curvature_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    sum_curvature = tf.reduce_sum(tf.square(curvature) * curvature_mask, axis=-1)
    sum_mask = tf.reduce_sum(curvature_mask, axis=-1)
    return tf.reduce_mean(sum_curvature / (sum_mask + 1e-7))


def jsc_voc_penalty_loss(y_true, y_pred, mask, original_lengths):
    # Cast everything to appropriate dtypes
    y_true = tf.cast(y_true, y_pred.dtype)
    mask = tf.cast(mask, y_pred.dtype)
    orig_len = tf.cast(original_lengths, tf.int32)

    # Jsc (first point)
    jsc_true = y_true[:, 0]
    jsc_pred = y_pred[:, 0]
    jsc_mask = mask[:, 0]
    jsc_mse = tf.reduce_mean(tf.square(jsc_true - jsc_pred) * jsc_mask)

    # Voc: last valid index
    batch_indices = tf.range(tf.shape(y_pred)[0])
    last_indices = tf.clip_by_value(orig_len - 1, 0, tf.shape(y_pred)[1] - 1)
    gather_indices = tf.stack([batch_indices, last_indices], axis=1)

    last_pred_current = tf.gather_nd(y_pred, gather_indices)
    voc_mask = tf.cast(orig_len > 0, y_pred.dtype)
    voc_loss = tf.reduce_sum(tf.square(last_pred_current) * voc_mask) / (tf.reduce_sum(voc_mask) + 1e-7)

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
    """
    Returns a Keras Functional model that takes:
      - X_params: shape=(batch, input_dim_params)
      - voltage_grid: shape=(batch, seq_len)
    and outputs a (batch, seq_len) vector of normalized currents (sigmoid).
    """
    # 1) Parameter input
    x_params_in = Input(shape=(input_dim_params,), name="X_params")
    param_path = Dense(128, activation='relu')(x_params_in)
    param_path = BatchNormalization()(param_path)
    param_path = Dropout(0.3)(param_path)
    param_path = Dense(128, activation='relu')(param_path)

    # 2) Voltage grid input
    voltage_grid_in = Input(shape=(seq_len,), name="voltage_grid")
    # Normalize voltage to [0,1] by dividing by ASSUMED_ORIGINAL_MAX_VOLTAGE
    norm_voltage_grid = Lambda(
        lambda v: v / ASSUMED_ORIGINAL_MAX_VOLTAGE,
        name="voltage_normalization"
    )(voltage_grid_in)

    # Expand dims via Lambda layer
    norm_voltage_grid_expanded = Lambda(
        lambda v: tf.expand_dims(v, axis=-1),
        name="voltage_expand_dims"
    )(norm_voltage_grid)  # shape -> (batch, seq_len, 1)

    v_embed = Dense(voltage_embed_dim, activation="relu", name="voltage_embed_dense1")(norm_voltage_grid_expanded)
    v_embed = Dense(voltage_embed_dim, activation="relu", name="voltage_embed_dense2")(v_embed)
    # Now v_embed is (batch, seq_len, voltage_embed_dim)

    # 3) Tile the param_path to (batch, seq_len, features)
    param_expanded = Lambda(
        lambda x: tf.expand_dims(x, axis=1),
        name="param_expand_dims"
    )(param_path)  # shape -> (batch, 1, features)
    param_tiled = Lambda(
        lambda x: tf.tile(x, [1, seq_len, 1]),
        name="param_tile"
    )(param_expanded)  # shape -> (batch, seq_len, features)

    # 4) Concatenate along last axis
    merged = Concatenate(axis=-1)([param_tiled, v_embed])  # (batch, seq_len, features + voltage_embed_dim)

    # 5) Per-step processing
    x = Dense(256, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 6) Output layer: sigmoid per time step
    iv_output = Dense(1, activation='sigmoid', name='iv_point_output')(x)  # (batch, seq_len, 1)
    iv_output_flat = Lambda(
        lambda t: tf.squeeze(t, axis=-1),
        name='iv_output_flat'
    )(iv_output)  # (batch, seq_len)

    return Model(
        inputs=[x_params_in, voltage_grid_in],
        outputs=iv_output_flat,
        name="NN_Core"
    )


class PhysicsNNModel(tf.keras.Model):
    """
    Custom Keras Model that wraps `nn_core_model` and implements
    train_step/test_step using full_masked_physics_loss.
    """
    def __init__(self, nn_core_model, loss_fn_config, **kwargs):
        super().__init__(**kwargs)
        self.nn_core_model = nn_core_model
        self.loss_fn_config = loss_fn_config
        self.custom_loss_fn = full_masked_physics_loss(loss_fn_config)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data):
        # data = (inputs_dict, y_true_for_loss_dict)
        inputs_dict, y_true_for_loss = data
        with tf.GradientTape() as tape:
            y_pred = self.nn_core_model(inputs_dict, training=True)
            loss = self.custom_loss_fn(y_true_for_loss, y_pred)

        grads = tape.gradient(loss, self.nn_core_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn_core_model.trainable_variables))

        self.loss_tracker.update_state(loss)
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
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]

    def call(self, inputs, training=False):
        return self.nn_core_model(inputs, training=training)

    def save(self, filepath, overwrite=True, **kwargs):
        """
        Save the core model (Functional) and loss_fn_config.
        """
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


# --- TruncatedIVReconstructor (unchanged except for loss fixes) ------------------------

class TruncatedIVReconstructor:
    def __init__(self, use_gpu_if_available=True):
        self.input_preprocessor = None
        self.scalar_feature_preprocessor = None

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
        self.loss_weights_nn = {
            'mse': 1.0, 'monotonicity': 0.05, 'curvature': 0.01,
            'jsc': 0.2, 'voc': 0.1
        }

        self.use_gpu = use_gpu_if_available
        self.cuml_ok = CUML_AVAILABLE if self.use_gpu else False
        self.xgb_gpu_ok = XGBOOST_GPU_AVAILABLE if self.use_gpu else False
        print(f"Reconstructor Config: GPU PCA: {self.cuml_ok}, GPU XGBoost: {self.xgb_gpu_ok}")

    def _truncate_filter_normalize_iv_data(self, params_df_raw, iv_data_np_raw,
                                           min_len_for_processing=5, min_len_savgol=5):
        print("Truncating, filtering, and normalizing I-V data...")
        full_voltage_grid = np.linspace(0, ASSUMED_ORIGINAL_MAX_VOLTAGE, ASSUMED_ORIGINAL_IV_POINTS)

        truncated_currents_list = []
        truncated_voltages_list = []
        normalized_truncated_currents_list = []
        scalers_list = []
        original_lengths_list = []

        isc_list_raw = []
        vknee_list_raw = []
        voc_ref_list_raw = []
        valid_indices_from_raw = []

        for i in range(iv_data_np_raw.shape[0]):
            curve_current_raw = iv_data_np_raw[i]
            first_negative_idx = -1
            for k_idx, k_val in enumerate(curve_current_raw):
                if k_val < 0:
                    first_negative_idx = k_idx
                    break

            trunc_idx = first_negative_idx if first_negative_idx != -1 else len(curve_current_raw)
            if trunc_idx < min_len_for_processing:
                continue

            current_truncated = curve_current_raw[:trunc_idx].copy()
            voltage_truncated = full_voltage_grid[:trunc_idx].copy()

            if current_truncated.size == 0:
                continue

            current_len = len(current_truncated)
            if current_len >= min_len_savgol:
                wl = min(max(min_len_savgol, (current_len // 10) * 2 + 1),
                         (current_len - 1) if current_len % 2 == 0 else (current_len - 2))
                wl = max(3, wl if wl % 2 != 0 else wl - 1)
                if wl < current_len and wl >= 3:
                    current_truncated = savgol_filter(current_truncated, window_length=wl, polyorder=min(2, wl - 1))

            scaler = MinMaxScaler(feature_range=(0, 1))
            if np.all(current_truncated == current_truncated[0]) and len(current_truncated) > 1:
                norm_current_for_list = np.zeros_like(current_truncated)
                scaler.fit(current_truncated.reshape(-1, 1))
            else:
                norm_current_for_list = scaler.fit_transform(current_truncated.reshape(-1, 1)).flatten()

            truncated_currents_list.append(current_truncated)
            truncated_voltages_list.append(voltage_truncated)
            normalized_truncated_currents_list.append(norm_current_for_list)
            scalers_list.append(scaler)
            original_lengths_list.append(current_len)

            isc_list_raw.append(current_truncated[0] if current_len > 0 else 0.0)
            vknee_list_raw.append(voltage_truncated[-1] if current_len > 0 else 0.0)
            voc_ref_list_raw.append(params_df_raw.iloc[i]['Voc_ref'])
            valid_indices_from_raw.append(i)

        if len(original_lengths_list) == 0:
            print("Warning: No valid curves after truncation and filtering.")
            return None

        max_len = max(original_lengths_list)
        num_valid_curves = len(normalized_truncated_currents_list)

        y_norm_trunc_padded_np = np.zeros((num_valid_curves, max_len))
        padded_voltages_np = np.zeros((num_valid_curves, max_len))
        mask_matrix_np = np.zeros((num_valid_curves, max_len), dtype=np.float32)

        for i in range(num_valid_curves):
            norm_curve = normalized_truncated_currents_list[i]
            volt_curve = truncated_voltages_list[i]
            curr_len = len(norm_curve)

            y_norm_trunc_padded_np[i, :curr_len] = norm_curve
            padded_voltages_np[i, :curr_len] = volt_curve
            mask_matrix_np[i, :curr_len] = 1.0

            if curr_len < max_len:
                y_norm_trunc_padded_np[i, curr_len:] = norm_curve[-1]
                padded_voltages_np[i, curr_len:] = volt_curve[-1]

        scalar_features_df = pd.DataFrame({
            'Isc_raw': isc_list_raw,
            'Vknee_raw': vknee_list_raw,
            'Voc_ref_raw': voc_ref_list_raw
        })

        return {
            "y_norm_padded": y_norm_trunc_padded_np,
            "padded_voltages": padded_voltages_np,
            "masks": mask_matrix_np,
            "scalers": scalers_list,
            "lengths": original_lengths_list,
            "raw_currents": truncated_currents_list,
            "raw_voltages": truncated_voltages_list,
            "scalar_features_df": scalar_features_df,
            "valid_indices": valid_indices_from_raw,
            "max_len": max_len
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
            if not actual_cols:
                continue
            steps = []
            if group_name == 'material':
                steps.append(('log_transform', FunctionTransformer(np.log1p, validate=False)))
            steps.append(('scaler', RobustScaler()))
            transformers.append((group_name, Pipeline(steps), actual_cols))

        self.input_preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        X_processed = self.input_preprocessor.fit_transform(params_df)
        print(f"  Processed device parameters shape: {X_processed.shape}")
        return X_processed

    def _preprocess_scalar_features(self, scalar_features_df, fit=True):
        print("Preprocessing scalar features (Isc, V_knee, Voc_ref)...")
        if fit:
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
        y_norm_padded = data_dict['y_norm_padded']

        iv_features_for_outlier = []
        for i in range(y_norm_padded.shape[0]):
            norm_curve = y_norm_padded[i, : data_dict['lengths'][i]]
            if len(norm_curve) > 1:
                std_val = np.std(norm_curve)
                mean_val = np.mean(norm_curve)
                isc_raw = data_dict['scalar_features_df']['Isc_raw'].iloc[i]
                vknee_raw = data_dict['scalar_features_df']['Vknee_raw'].iloc[i]
                iv_features_for_outlier.append([std_val, mean_val, isc_raw, vknee_raw])
            elif len(norm_curve) == 1:
                isc_raw = data_dict['scalar_features_df']['Isc_raw'].iloc[i]
                vknee_raw = data_dict['scalar_features_df']['Vknee_raw'].iloc[i]
                iv_features_for_outlier.append([0.0, norm_curve[0], isc_raw, vknee_raw])
            else:
                iv_features_for_outlier.append([0.0, 0.0, 0.0, 0.0])

        iv_features_np = np.array(iv_features_for_outlier)
        self.outlier_detector = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100, n_jobs=-1
        )
        outlier_labels = self.outlier_detector.fit_predict(iv_features_np)
        normal_mask = outlier_labels == 1
        n_outliers = np.sum(~normal_mask)
        print(f"  Removed {n_outliers} outlier samples ({n_outliers/len(outlier_labels)*100:.1f}%)")

        cleaned = {}
        cleaned['X_clean_augmented'] = X_full[normal_mask]
        cleaned['y_clean_norm_padded'] = data_dict['y_norm_padded'][normal_mask]
        cleaned['padded_voltages_clean'] = data_dict['padded_voltages'][normal_mask]
        cleaned['masks_clean'] = data_dict['masks'][normal_mask]
        cleaned['scalers_clean'] = [s for s, m in zip(data_dict['scalers'], normal_mask) if m]
        cleaned['lengths_clean'] = [l for l, m in zip(data_dict['lengths'], normal_mask) if m]
        cleaned['raw_currents_clean'] = [
            c for c, m in zip(data_dict['raw_currents'], normal_mask) if m
        ]
        cleaned['raw_voltages_clean'] = [
            v for v, m in zip(data_dict['raw_voltages'], normal_mask) if m
        ]
        cleaned['scalar_features_df'] = data_dict['scalar_features_df'].iloc[normal_mask].reset_index(drop=True)

        if not cleaned['lengths_clean']:
            print("Warning: All samples removed as outliers or no valid samples to begin with.")
            return None

        cleaned['max_len_clean'] = max(cleaned['lengths_clean'])
        return cleaned

    def load_and_prepare_data(self):
        print("=== Loading and Preparing Data (ENHANCED TRUNCATED IV) ===")
        if not check_data_files_exist():
            print(f"Data files not found: {INPUT_FILE}, {OUTPUT_FILE}")
            return False

        try:
            params_df_raw_full = pd.read_csv(INPUT_FILE, header=None, names=COLNAMES if COLNAMES else None)
            iv_data_np_raw_full = np.loadtxt(OUTPUT_FILE, delimiter=',')
            print(f"Loaded raw data: Parameters={params_df_raw_full.shape}, I-V curves={iv_data_np_raw_full.shape}")

            iv_bundle = self._truncate_filter_normalize_iv_data(
                params_df_raw_full, iv_data_np_raw_full
            )
            if iv_bundle is None:
                return False

            params_df_filtered = params_df_raw_full.iloc[iv_bundle['valid_indices']].reset_index(drop=True)
            scalar_features_df_unscaled = iv_bundle['scalar_features_df'].reset_index(drop=True)

            X_params_processed = self._preprocess_input_parameters(params_df_filtered)
            X_scalar_processed = self._preprocess_scalar_features(scalar_features_df_unscaled, fit=True)

            X_augmented_full = np.concatenate([X_params_processed, X_scalar_processed], axis=1)
            print(f"  Augmented X shape: {X_augmented_full.shape}")

            data_for_outlier = {
                'X_augmented': X_augmented_full,
                'y_norm_padded': iv_bundle['y_norm_padded'],
                'padded_voltages': iv_bundle['padded_voltages'],
                'masks': iv_bundle['masks'],
                'scalers': iv_bundle['scalers'],
                'lengths': iv_bundle['lengths'],
                'raw_currents': iv_bundle['raw_currents'],
                'raw_voltages': iv_bundle['raw_voltages'],
                'scalar_features_df': scalar_features_df_unscaled
            }

            cleaned_data = self._remove_outlier_samples(data_for_outlier, contamination=0.05)
            if cleaned_data is None:
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
            elif BaseXGBRegressor == SklearnGradientBoostingRegressor:
                reg = SklearnGradientBoostingRegressor(**regressor_params, validation_fraction=0.1, n_iter_no_change=10)
            else:
                reg = BaseXGBRegressor(**regressor_params, tree_method='hist', n_jobs=-1)

            reg.fit(X_train_np, y_pca_np[:, i])
            self.pca_regressors.append(reg)

        self.models['pca'] = {'pca_obj': self.pca_model, 'regressors': self.pca_regressors}
        print("PCA model fitting completed!")
        return self

    def fit_physics_informed_nn(
        self,
        X_train_np, y_train_norm_padded_np,
        V_train_padded_voltages, Mask_train, Origlen_train,
        X_val_np, y_val_norm_padded_np,
        V_val_padded_voltages, Mask_val, Origlen_val
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

        physics_model.compile(optimizer=Adam(learning_rate=1e-3), metrics=['mae'])

        train_inputs = {
            "X_params": X_train_np,
            "voltage_grid": V_train_padded_voltages
        }
        train_targets = {
            "y_true": y_train_norm_padded_np,
            "mask": Mask_train,
            "orig_len": Origlen_train
        }

        train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
        train_ds = train_ds.shuffle(buffer_size=min(X_train_np.shape[0], 10000)).batch(128).prefetch(tf.data.AUTOTUNE)

        val_ds = None
        if X_val_np is not None and len(X_val_np) > 0:
            val_inputs = {"X_params": X_val_np, "voltage_grid": V_val_padded_voltages}
            val_targets = {"y_true": y_val_norm_padded_np, "mask": Mask_val, "orig_len": Origlen_val}
            val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets))
            val_ds = val_ds.batch(128).prefetch(tf.data.AUTOTUNE)
        else:
            print("  Warning: No validation set provided for NN.")

        callbacks = [
            EarlyStopping(monitor='val_loss' if val_ds else 'loss', patience=30, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss' if val_ds else 'loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1)
        ]

        self.training_history = physics_model.fit(
            train_ds, epochs=200, validation_data=val_ds, callbacks=callbacks, verbose=1
        )
        self.models['physics_nn'] = physics_model
        return self

    def fit_ensemble_model(self, X_train_np, y_train_norm_padded_np):
        print("\n=== Fitting Ensemble Model (Enhanced X, Padded Y) ===")
        if self.max_truncated_len_overall == 0:
            print("Warning: Output dim for ensemble is 0. Skipping.")
            self.models['ensemble'] = None
            return self

        regressor_params = {
            'n_estimators': 150,
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.7,
            'random_state': 42
        }
        base_configs = []

        if self.xgb_gpu_ok:
            print("  Using XGBoost (GPU) for ensemble members...")
            base_configs.append((
                'xgb1', BaseXGBRegressor(**regressor_params, tree_method='gpu_hist', n_jobs=-1)
            ))
        elif BaseXGBRegressor != SklearnGradientBoostingRegressor:
            print("  Using XGBoost (CPU) for ensemble members...")
            base_configs.append((
                'xgb1_cpu', BaseXGBRegressor(**regressor_params, tree_method='hist', n_jobs=-1)
            ))
        else:
            print("  Using RandomForestRegressor (CPU) for ensemble...")
            base_configs.append((
                'rf_cpu',
                SklearnRandomForestRegressor(
                    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                )
            ))

        self.ensemble_models = {}
        for name, base_model in base_configs:
            print(f"  Training {name.upper()} ensemble member…")
            multi = MultiOutputRegressor(base_model, n_jobs=1)
            multi.fit(X_train_np, y_train_norm_padded_np)
            self.ensemble_models[name] = multi

        self.models['ensemble'] = self.ensemble_models
        return self

    def predict(
        self,
        X_np,
        V_padded_voltages_np,  # only needed for NN
        scalers_for_X,
        original_lengths_for_X,
        model_type='pca'
    ):
        if model_type not in self.models or self.models[model_type] is None:
            print(f"Model {model_type} not fitted. Returning empty.")
            return [np.array([]) for _ in range(X_np.shape[0])]

        M = self.models[model_type]
        y_pred_norm_padded = None

        if model_type == 'pca':
            pca_obj = M['pca_obj']
            regs = M['regressors']
            if not regs:
                print("PCA model has no regressors. Returning empty.")
                return [np.array([]) for _ in range(X_np.shape[0])]

            y_pca_pred = np.zeros((X_np.shape[0], len(regs)))
            for i, r in enumerate(regs):
                y_pca_pred[:, i] = r.predict(X_np)

            if self.cuml_ok and isinstance(pca_obj, CumlPCA):
                y_pca_cu = cupy.asarray(y_pca_pred)
                y_norm_cu = pca_obj.inverse_transform(y_pca_cu)
                y_pred_norm_padded = cupy.asnumpy(y_norm_cu)
                del y_pca_cu, y_norm_cu; cupy.get_default_memory_pool().free_all_blocks()
            else:
                y_pred_norm_padded = pca_obj.inverse_transform(y_pca_pred)

        elif model_type == 'physics_nn':
            nn_input = {
                "X_params": X_np,
                "voltage_grid": V_padded_voltages_np
            }
            y_pred_norm_padded = M.predict(nn_input, batch_size=512, verbose=0)

        elif model_type == 'ensemble':
            preds = [model.predict(X_np) for _, model in M.items()]
            y_pred_norm_padded = np.mean(preds, axis=0)

        if y_pred_norm_padded.shape[0] != len(scalers_for_X):
            raise ValueError(f"Sample-scaler mismatch: {y_pred_norm_padded.shape[0]} vs {len(scalers_for_X)}")

        y_pred_orig_trunc_list = []
        for i in range(y_pred_norm_padded.shape[0]):
            norm_curve = y_pred_norm_padded[i]
            scaler = scalers_for_X[i]
            orig_len = original_lengths_for_X[i]

            if orig_len == 0:
                y_pred_orig_trunc_list.append(np.array([]))
                continue

            full_curve_unscaled = scaler.inverse_transform(norm_curve.reshape(-1, 1)).flatten()
            truncated_curve = full_curve_unscaled[:orig_len]
            y_pred_orig_trunc_list.append(truncated_curve)

        return y_pred_orig_trunc_list

    def evaluate_model(
        self,
        X_test,
        V_test_padded_voltages,
        y_test_orig_truncated_curves_list,
        scalers_test,
        original_lengths_test,
        model_type='pca'
    ):
        print(f"\n=== Evaluating {model_type.upper()} (Enhanced Truncated) ===")
        y_pred_list = self.predict(
            X_test,
            V_test_padded_voltages,
            scalers_test,
            original_lengths_test,
            model_type=model_type
        )

        if not y_test_orig_truncated_curves_list or not y_pred_list:
            print("  No data for evaluation.")
            return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'per_curve_R2_mean': np.nan, 'predictions': []}

        flat_true = np.concatenate([arr for arr in y_test_orig_truncated_curves_list if len(arr) > 0])
        flat_pred = np.concatenate([arr for arr in y_pred_list if len(arr) > 0])

        if flat_true.size == 0:
            mae, rmse, r2_global = np.nan, np.nan, np.nan
        else:
            mae = mean_absolute_error(flat_true, flat_pred)
            mse = mean_squared_error(flat_true, flat_pred)
            rmse = np.sqrt(mse)
            r2_global = (
                r2_score(flat_true, flat_pred)
                if np.std(flat_true) > 1e-9
                else (1.0 if mse < 1e-6 else 0.0)
            )

        print(f"Truncated Original scale: MAE: {mae:.6f}, RMSE: {rmse:.6f}, Global R²: {r2_global:.4f}")

        per_curve_r2 = []
        for i in range(len(y_test_orig_truncated_curves_list)):
            true_c = y_test_orig_truncated_curves_list[i]
            pred_c = y_pred_list[i]
            if len(true_c) > 1 and len(pred_c) == len(true_c):
                if np.std(true_c) > 1e-9:
                    per_curve_r2.append(r2_score(true_c, pred_c))
                elif mean_squared_error(true_c, pred_c) < 1e-6:
                    per_curve_r2.append(1.0)
                else:
                    per_curve_r2.append(0.0)

        if per_curve_r2:
            mean_r2_pc = np.mean(per_curve_r2)
            std_r2_pc = np.std(per_curve_r2)
            print(f"  Per-curve R² (truncated) - Mean: {mean_r2_pc:.4f}, Std: {std_r2_pc:.4f}")
        else:
            mean_r2_pc = np.nan
            print("  Not enough valid curves for per-curve R².")

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2_global,
            'per_curve_R2_mean': mean_r2_pc,
            'per_curve_R2_std': std_r2_pc,
            'predictions': y_pred_list
        }

    def plot_results(
        self,
        X_test,
        V_test_padded_voltages,
        y_test_orig_truncated_curves_list,
        y_test_orig_truncated_voltages_list,
        scalers_test,
        original_lengths_test,
        model_type='pca',
        n_samples=4
    ):
        y_pred_list = self.predict(
            X_test,
            V_test_padded_voltages,
            scalers_test,
            original_lengths_test,
            model_type=model_type
        )

        num_avail = len(y_test_orig_truncated_curves_list)
        if num_avail == 0:
            print(f"No samples to plot for {model_type}.")
            return

        indices = np.random.choice(num_avail, min(n_samples, num_avail), replace=False)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for i, idx in enumerate(indices):
            ax = axes[i]
            true_c = y_test_orig_truncated_curves_list[idx]
            true_v = y_test_orig_truncated_voltages_list[idx]
            pred_c = y_pred_list[idx]

            if not (len(true_c) > 0 and len(pred_c) > 0 and
                    len(true_c) == len(pred_c) and
                    len(true_c) == len(true_v)):
                ax.text(0.5, 0.5, f"Sample {idx}\nMismatch/Empty",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Sample {idx} - Error")
                continue

            ax.plot(true_v, true_c, 'b-', lw=2, label='Actual (Trunc)', alpha=0.8)
            ax.plot(true_v, pred_c, 'r--', lw=2, label='Predicted (Trunc)', alpha=0.8)

            if np.std(true_c) > 1e-9:
                curve_r2 = r2_score(true_c, pred_c)
            else:
                curve_r2 = 1.0 if mean_squared_error(true_c, pred_c) < 1e-6 else 0.0

            ax.set_title(f"Sample {idx} - {model_type.upper()} (R²={curve_r2:.3f})")
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current Density (mA/cm²)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outpath = OUTPUT_DIR_TRUNCATED / f"enhanced_cuda_{model_type}_predictions.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Enhanced prediction plot saved: {outpath}")

    def plot_pca_analysis(self):
        if 'pca' not in self.models or not self.models['pca'] or not self.models['pca']['pca_obj']:
            print("PCA model not fitted for plotting!")
            return

        pca_obj = self.models['pca']['pca_obj']
        explained_var = pca_obj.explained_variance_ratio_
        if isinstance(explained_var, cupy.ndarray):
            explained_var = cupy.asnumpy(explained_var)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].bar(range(1, len(explained_var) + 1), explained_var)
        axes[0, 0].set_title("PCA Explained Variance")

        cum_var = np.cumsum(explained_var)
        axes[0, 1].plot(range(1, len(cum_var) + 1), cum_var, 'bo-')
        axes[0, 1].set_title("Cumulative Explained Variance")
        axes[0, 1].axhline(y=0.95, c='r', ls='--', label='95%')
        axes[0, 1].legend()

        components = pca_obj.components_
        if isinstance(components, cupy.ndarray):
            components = cupy.asnumpy(components)
        component_x = np.arange(components.shape[1])

        for i in range(min(4, components.shape[0])):
            axes[1, 0].plot(component_x, components[i], label=f"PC{i+1}")
        axes[1, 0].set_title("Principal Components")
        axes[1, 0].set_xlabel("Component Index")
        axes[1, 0].legend()

        if hasattr(self.models['pca']['regressors'][0], 'feature_importances_'):
            feature_names = []
            if self.input_preprocessor:
                feature_names.extend(self.input_preprocessor.get_feature_names_out())
            if self.scalar_feature_preprocessor:
                feature_names.extend(["scalar__Isc", "scalar__Vknee", "scalar__Voc_ref"])

            imp = self.models['pca']['regressors'][0].feature_importances_
            if len(feature_names) != len(imp):
                feature_names = [f"F{i}" for i in range(len(imp))]

            top_idx = np.argsort(imp)[-10:]
            labels = [feature_names[i] for i in top_idx]
            axes[1, 1].barh(range(len(top_idx)), imp[top_idx])
            axes[1, 1].set_yticks(range(len(top_idx)))
            axes[1, 1].set_yticklabels(labels)
            axes[1, 1].set_title("Top Features for PC1 Prediction")
        else:
            axes[1, 1].text(
                0.5, 0.5, "Feat. importance N/A",
                ha='center', va='center', transform=axes[1, 1].transAxes
            )

        plt.tight_layout()
        outpath = OUTPUT_DIR_TRUNCATED / "enhanced_cuda_pca_analysis.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Enhanced PCA analysis plot saved: {outpath}")


# --- MAIN FUNCTION ----------------------------------------------------------------------

def main_truncated_enhanced():
    print("=== ENHANCED Perovskite I-V Curve Reconstruction (CUDA & Advanced) ===")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU(s) available to TensorFlow.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPUs available to TensorFlow.")

    reconstructor = TruncatedIVReconstructor(use_gpu_if_available=True)
    if not reconstructor.load_and_prepare_data():
        print("Failed to load/prepare data. Exiting.")
        return

    # Retrieve cleaned data
    N = reconstructor.X_clean.shape[0]
    all_indices = np.arange(N)

    train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)  # 12% val

    # TRAIN SET
    X_train = reconstructor.X_clean[train_idx]
    y_train_norm = reconstructor.y_clean_norm_padded[train_idx]
    V_train_pad = reconstructor.padded_voltages_all_clean[train_idx]
    Mask_train = reconstructor.masks_all_clean[train_idx]
    Origlen_train = np.array(reconstructor.final_original_lengths_for_clean_data)[train_idx]

    # VALIDATION SET (for NN)
    X_val = reconstructor.X_clean[val_idx]
    y_val_norm = reconstructor.y_clean_norm_padded[val_idx]
    V_val_pad = reconstructor.padded_voltages_all_clean[val_idx]
    Mask_val = reconstructor.masks_all_clean[val_idx]
    Origlen_val = np.array(reconstructor.final_original_lengths_for_clean_data)[val_idx]

    # TEST SET
    X_test = reconstructor.X_clean[test_idx]
    V_test_pad = reconstructor.padded_voltages_all_clean[test_idx]

    scalers_test = [reconstructor.final_scalers_for_clean_data[i] for i in test_idx]
    lengths_test = [reconstructor.final_original_lengths_for_clean_data[i] for i in test_idx]
    y_test_curves = [reconstructor.y_clean_raw_truncated_curves[i] for i in test_idx]
    y_test_voltages = [reconstructor.y_clean_raw_truncated_voltages[i] for i in test_idx]

    print(f"\nShapes: X_train {X_train.shape}, y_train_norm {y_train_norm.shape}")
    print(f"Val: X_val {X_val.shape}, Test samples {len(y_test_curves)}")

    # --- 1) PCA model ---
    reconstructor.fit_pca_model(X_train, y_train_norm, n_components=20)
    if reconstructor.models.get('pca'):
        reconstructor.plot_pca_analysis()
        pca_res = reconstructor.evaluate_model(
            X_test, V_test_pad,
            y_test_curves, scalers_test, lengths_test, 'pca'
        )
        reconstructor.plot_results(
            X_test, V_test_pad,
            y_test_curves, y_test_voltages,
            scalers_test, lengths_test, 'pca'
        )
    else:
        pca_res = None
        print("PCA model was skipped.")

    # --- 2) Physics-Informed NN ---
    reconstructor.fit_physics_informed_nn(
        X_train, y_train_norm, V_train_pad, Mask_train, Origlen_train,
        X_val, y_val_norm, V_val_pad, Mask_val, Origlen_val
    )
    if reconstructor.models.get('physics_nn'):
        nn_res = reconstructor.evaluate_model(
            X_test, V_test_pad,
            y_test_curves, scalers_test, lengths_test, 'physics_nn'
        )
        reconstructor.plot_results(
            X_test, V_test_pad,
            y_test_curves, y_test_voltages,
            scalers_test, lengths_test, 'physics_nn'
        )
    else:
        nn_res = None
        print("Physics NN model skipped.")

    # --- 3) Ensemble model ---
    reconstructor.fit_ensemble_model(X_train, y_train_norm)
    if reconstructor.models.get('ensemble'):
        ens_res = reconstructor.evaluate_model(
            X_test, V_test_pad,
            y_test_curves, scalers_test, lengths_test, 'ensemble'
        )
        reconstructor.plot_results(
            X_test, V_test_pad,
            y_test_curves, y_test_voltages,
            scalers_test, lengths_test, 'ensemble'
        )
    else:
        ens_res = None
        print("Ensemble model skipped.")

    # --- RESULTS SUMMARY ---
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON (ENHANCED TRUNCATED)")
    print("="*70)
    summary = {}
    if pca_res: summary['PCA (Enhanced)'] = pca_res
    if nn_res: summary['Physics NN (Enhanced)'] = nn_res
    if ens_res: summary['Ensemble (Enhanced)'] = ens_res

    print(f"{'Model':<25} {'Global R²':<12} {'MAE':<10} {'RMSE':<10} {'Mean R²/curve':<15}")
    print("-"*80)
    best_name, best_r2_pc = "N/A", -np.inf
    for name, res in summary.items():
        if res:
            g = res['R2']; m = res['MAE']; r = res['RMSE']; mr2 = res['per_curve_R2_mean']
            print(f"{name:<25} {g:<12.4f} {m:<10.6f} {r:<10.6f} {mr2:<15.4f}")
            if not np.isnan(mr2) and mr2 > best_r2_pc:
                best_r2_pc = mr2
                best_name = name
        else:
            print(f"{name:<25} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<15}")
    print(f"\nBest performing model (Mean R²/curve): {best_name} (R² = {best_r2_pc:.4f})")

        # --- SAVE MODELS + PREPROCESSORS ---
    joblib.dump(
        reconstructor.input_preprocessor,
        OUTPUT_DIR_TRUNCATED / "input_preprocessor_enhanced.joblib"
    )
    if reconstructor.scalar_feature_preprocessor:
        joblib.dump(
            reconstructor.scalar_feature_preprocessor,
            OUTPUT_DIR_TRUNCATED / "scalar_feature_preprocessor_enhanced.joblib"
        )

    if reconstructor.models.get('pca'):
        joblib.dump(
            reconstructor.models['pca'],
            OUTPUT_DIR_TRUNCATED / "pca_model_bundle_enhanced.joblib"
        )

    if reconstructor.models.get('physics_nn'):
        # Use the custom .save(...) to persist the Keras model + loss‐config; do NOT joblib.dump it.
        reconstructor.models['physics_nn'].save(
            OUTPUT_DIR_TRUNCATED / "physics_nn_model_enhanced"
        )

    if reconstructor.models.get('ensemble'):
        joblib.dump(
            reconstructor.models['ensemble'],
            OUTPUT_DIR_TRUNCATED / "ensemble_models_bundle_enhanced.joblib"
        )

    # Save per‐curve current‐scalers and other metadata
    joblib.dump({
        'scalers': reconstructor.final_scalers_for_clean_data,
        'lengths': reconstructor.final_original_lengths_for_clean_data,
        'max_len': reconstructor.max_truncated_len_overall,
        'ASSUMED_ORIGINAL_MAX_VOLTAGE': ASSUMED_ORIGINAL_MAX_VOLTAGE,
        'ASSUMED_ORIGINAL_IV_POINTS': ASSUMED_ORIGINAL_IV_POINTS,
        'loss_weights_nn': reconstructor.loss_weights_nn
    }, OUTPUT_DIR_TRUNCATED / "data_processing_info_enhanced.joblib")

    print("\n=== Models, preprocessors, and processing info (enhanced) saved! ===")

    # --- MULTI-MODEL COMPARISON PLOT ---
    if y_test_curves and any(summary.values()):
        preds_dict = {}
        if pca_res and 'predictions' in pca_res: preds_dict["PCA (Enh)"] = pca_res['predictions']
        if nn_res and 'predictions' in nn_res: preds_dict["NN (Enh)"] = nn_res['predictions']
        if ens_res and 'predictions' in ens_res: preds_dict["Ensemble (Enh)"] = ens_res['predictions']

        if preds_dict:
            plot_multi_model_comparison_enhanced(
                y_test_curves,
                y_test_voltages,
                preds_dict,
                n_samples_plot=4,
                output_dir=OUTPUT_DIR_TRUNCATED
            )


def plot_multi_model_comparison_enhanced(
    y_true_curves_list,
    y_true_voltages_list,
    y_preds_dict,
    n_samples_plot,
    output_dir
):
    num_curves = len(y_true_curves_list)
    if num_curves == 0:
        return

    sample_idxs = np.random.choice(num_curves, min(n_samples_plot, num_curves), replace=False)
    n_models = len(y_preds_dict)

    fig, axes = plt.subplots(
        len(sample_idxs),
        n_models + 1,
        figsize=(5 * (n_models + 1), 4 * len(sample_idxs)),
        squeeze=False
    )

    for row, idx in enumerate(sample_idxs):
        true_cur = y_true_curves_list[idx]
        true_vol = y_true_voltages_list[idx]
        if len(true_cur) == 0 or len(true_cur) != len(true_vol):
            for col_ax in range(n_models + 1):
                axes[row, col_ax].text(
                    0.5, 0.5, f"Sample {idx}\nInvalid",
                    ha='center', va='center',
                    transform=axes[row, col_ax].transAxes
                )
            continue

        # Plot ground truth
        ax0 = axes[row, 0]
        ax0.plot(true_vol, true_cur, 'b-', lw=2, label='True (Trunc)')
        ax0.set_title(f"Sample {idx}\nGround Truth")
        ax0.set_xlabel("Voltage (V)")
        ax0.set_ylabel("Current")
        ax0.grid(True, alpha=0.5)
        ax0.legend()

        for col, (mname, pred_list) in enumerate(y_preds_dict.items()):
            axp = axes[row, col + 1]
            pred_cur = pred_list[idx]
            if len(pred_cur) != len(true_cur):
                axp.text(0.5, 0.5, "Length Mismatch", ha='center', va='center', transform=axp.transAxes)
                axp.set_title(f"{mname}\nError")
                continue

            axp.plot(true_vol, true_cur, 'b-', lw=1.5, label='True', alpha=0.7)
            axp.plot(true_vol, pred_cur, 'r--', lw=2, label='Pred')
            r2v = (
                r2_score(true_cur, pred_cur)
                if np.std(true_cur) > 1e-9
                else (1.0 if mean_squared_error(true_cur, pred_cur) < 1e-6 else 0.0)
            )
            axp.set_title(f"{mname}\nR²={r2v:.3f}")
            axp.set_xlabel("Voltage (V)")
            axp.grid(True, alpha=0.5)
            axp.legend()

    plt.tight_layout()
    outfile = output_dir / "enhanced_model_comparison_iv_curves.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"\n[Visualization] Enhanced multi-model IV curve comparison plot saved: {outfile}")


if __name__ == "__main__":
    main_truncated_enhanced()