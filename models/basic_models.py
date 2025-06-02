# Preprocessor embedded in the model file
# 1 - Regressor (PCA refers to regressor, just called that for simplicity)
# 2 - Physics-informed neural network (PINN)
# 3 - Ensemble model (Random Forest + Gradient Boosting)
import os
# Set TF_CPP_MIN_LOG_LEVEL to 2 before importing TensorFlow to suppress INFO logs
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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Lambda
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
    CumlPCA = SklearnPCA

try:
    import xgboost
    try:
        xgboost.XGBRegressor(tree_method='gpu_hist').fit(np.array([[1]]), np.array([1]))
        print("XGBoost with GPU support found. Will use GPU for GBR alternatives.")
        XGBOOST_GPU_AVAILABLE = True
    except xgboost.core.XGBoostError:
        print("Warning: XGBoost found, but GPU support (gpu_hist) seems unavailable. XGBoost will run on CPU.")
        XGBOOST_GPU_AVAILABLE = False
    BaseXGBRegressor = xgboost.XGBRegressor
except ImportError:
    print("Warning: XGBoost not found. Ensemble and PCA regressors will use scikit-learn CPU versions.")
    XGBOOST_GPU_AVAILABLE = False
    BaseXGBRegressor = SklearnGradientBoostingRegressor

# === BEGIN: Standalone configuration (replaces common_header.py) ===
INPUT_FILE = "/content/drive/MyDrive/Colab Notebooks/Data_100k/LHS_parameters_m.txt" # made for colab
OUTPUT_FILE = "/content/drive/MyDrive/Colab Notebooks/Data_100k/iV_m.txt"
COLNAMES = [
    'Eg', 'NCv', 'NCc', 'mu_e', 'mu_h', 'eps', 'A', 'Cn', 'Cp', 'Nt',
    'Et', 'nD', 'nA', 'thickness', 'T', 'Sn', 'Sp', 'Rs', 'Rsh',
    'G', 'light_intensity', 'Voc_ref', 'Jsc_ref', 'FF_ref', 'PCE_ref',
    'Qe_loss', 'R_loss', 'SRH_loss', 'series_loss', 'shunt_loss', 'other_loss'
]
# IMPORTANT: New output directory for the truncated version. At this point I don't really have any use for the negative value model anyway though.
OUTPUT_DIR_TRUNCATED = Path("./output_data_truncated_cuda")
OUTPUT_DIR_TRUNCATED.mkdir(parents=True, exist_ok=True)

# PLEASE NOTE, THE INDICES SHOULD BE REPLACED W/ REAL VOLTAGE VALUES AND THE PADDED VALUES SHOULD NOT BE TRAINED ON!! Its a 2x improvement if  you do this.
ASSUMED_ORIGINAL_IV_POINTS = 45
ASSUMED_ORIGINAL_MAX_VOLTAGE = 1.2

def check_data_files_exist():
    return Path(INPUT_FILE).exists() and Path(OUTPUT_FILE).exists()

if not check_data_files_exist():
    print(f"Creating dummy data files for testing ({INPUT_FILE}, {OUTPUT_FILE})...")
    num_dummy_samples = 1000
    pd.DataFrame(np.random.rand(num_dummy_samples, len(COLNAMES))).to_csv(INPUT_FILE, header=False, index=False)
    # Generate dummy output data that includes negative values to test truncation
    dummy_output_data = np.random.randn(num_dummy_samples, ASSUMED_ORIGINAL_IV_POINTS) * 0.8 + 0.3 # Mean 0.3, std 0.8 -> some negatives
    np.savetxt(OUTPUT_FILE, dummy_output_data, delimiter=',')
# === END: Standalone configuration ===

def _ft_get_feature_names_out(self, input_features=None):
    return input_features if input_features is not None else []
FunctionTransformer.get_feature_names_out = _ft_get_feature_names_out


class TruncatedIVReconstructor: # Renamed class
    def __init__(self, use_gpu_if_available=True):
        self.input_preprocessor = None
        # For truncated data:
        self.final_scalers_for_clean_data = None # List of scalers
        self.final_original_lengths_for_clean_data = None # List of original lengths after truncation
        self.max_truncated_len_overall = None # Max length after truncation, for padding

        self.pca_model = None
        self.models = {}
        self.outlier_detector = None
        self.param_group_names = ['material', 'device', 'operating', 'reference', 'loss']

        self.use_gpu = use_gpu_if_available
        if self.use_gpu:
            self.cuml_ok = CUML_AVAILABLE
            self.xgb_gpu_ok = XGBOOST_GPU_AVAILABLE
        else:
            self.cuml_ok = False
            self.xgb_gpu_ok = False
        print(f"TruncatedReconstructor Config: GPU PCA (cuML): {self.cuml_ok}, GPU XGBoost: {self.xgb_gpu_ok}")

    def _truncate_raw_iv_data(self, iv_data_raw):
        truncated_curves_list = []
        original_lengths_list = []
        min_len_for_processing = 3 # Min length for Savgol filter; MinMaxScaler needs >1 if range is 0

        for i in range(iv_data_raw.shape[0]):
            curve = iv_data_raw[i]

            first_negative_idx = -1
            for k_idx, k_val in enumerate(curve):
                if k_val < 0:
                    first_negative_idx = k_idx
                    break

            if first_negative_idx == -1: # All non-negative
                truncated_curve = curve[:]
            elif first_negative_idx == 0: # Starts with a negative value
                # "Part before dip" is empty. Take first point(s) for min_len.
                truncated_curve = curve[:min(min_len_for_processing, len(curve))]
            else: # Negative value found after some non-negative ones
                truncated_curve = curve[:first_negative_idx]

            if len(truncated_curve) < min_len_for_processing:
                # If truncated is too short, try to take min_len_for_processing from original start
                # This might include negative values if original positive part was shorter than min_len
                if len(curve) >= min_len_for_processing:
                    truncated_curve = curve[:min_len_for_processing]
                else:
                    truncated_curve = curve[:] # Take whatever is available from original
                # Final check for empty
                if len(truncated_curve) == 0:
                    truncated_curve = np.array([0.0] * min_len_for_processing) # Fallback for safety

            truncated_curves_list.append(truncated_curve)
            original_lengths_list.append(len(truncated_curve))
        return truncated_curves_list, original_lengths_list

    def _pad_normalized_truncated_data(self, normalized_truncated_curves_list, max_len, original_lengths_list):
        num_curves = len(normalized_truncated_curves_list)
        padded_data = np.zeros((num_curves, max_len))

        for i in range(num_curves):
            norm_trunc_curve = normalized_truncated_curves_list[i]
            # original_lengths_list[i] should be equal to len(norm_trunc_curve) here
            current_len = len(norm_trunc_curve)

            if current_len > 0:
                padded_data[i, :current_len] = norm_trunc_curve
                if current_len < max_len:
                    padding_value = norm_trunc_curve[current_len-1] # Pad with last normalized value
                    padded_data[i, current_len:] = padding_value
            elif max_len > 0: # current_len is 0 (or <0, which is impossible)
                padded_data[i, :] = 0.0 # Pad with 0 for normalized data if original was empty
        return padded_data

    def load_and_prepare_data(self):
        print("=== Loading and Preparing Data (TRUNCATED IV) ===")
        if not check_data_files_exist():
            print(f"Data files not found: {INPUT_FILE}, {OUTPUT_FILE}")
            return None, None, None, None
        try:
            params_df_raw = pd.read_csv(INPUT_FILE, header=None, names=COLNAMES if COLNAMES else None)
            iv_data_np_raw = np.loadtxt(OUTPUT_FILE, delimiter=',')
            print(f"Loaded raw data shapes: Parameters={params_df_raw.shape}, I-V curves={iv_data_np_raw.shape}")

            X_processed = self._preprocess_input_parameters(params_df_raw)

            # 1. Truncate raw I-V data
            truncated_iv_list, original_lengths_full_list = self._truncate_raw_iv_data(iv_data_np_raw)

            # Filter out curves that became too short after truncation for reliable processing
            valid_indices = [i for i, length in enumerate(original_lengths_full_list) if length >= 3] # Min 3 for Savgol
            if len(valid_indices) < len(original_lengths_full_list):
                print(f"  Filtered out {len(original_lengths_full_list) - len(valid_indices)} curves that were too short after truncation.")

            X_processed = X_processed[valid_indices]
            truncated_iv_list = [truncated_iv_list[i] for i in valid_indices]
            original_lengths_full_list = [original_lengths_full_list[i] for i in valid_indices]
            iv_data_np_raw = iv_data_np_raw[valid_indices] # Keep raw aligned for y_original later

            if not truncated_iv_list:
                print("No valid curves left after truncation and length filtering. Exiting.")
                return None, None, None, None

            self.max_truncated_len_overall = max(original_lengths_full_list) if original_lengths_full_list else 0
            print(f"  Max length of I-V curve after truncation: {self.max_truncated_len_overall} points.")

            # 2. Normalize truncated I-V curves (variable lengths)
            # _preprocess_iv_curves now returns a list of normalized truncated curves
            y_norm_trunc_list, scalers_list_full = self._preprocess_iv_curves(truncated_iv_list, original_lengths_full_list)

            # 3. Pad normalized truncated I-V curves to max_truncated_len_overall
            y_norm_trunc_padded_full_np = self._pad_normalized_truncated_data(
                y_norm_trunc_list, self.max_truncated_len_overall, original_lengths_full_list
            )

            # 4. Outlier removal (operates on padded normalized data, using original lengths for feature calculation)
            X_clean, y_clean_norm_padded, scalers_clean_list, original_lengths_clean_list, iv_data_clean_raw = \
                self._remove_outlier_samples(
                    X_processed, y_norm_trunc_padded_full_np, scalers_list_full, original_lengths_full_list, iv_data_np_raw
                )

            self.final_scalers_for_clean_data = scalers_clean_list
            self.final_original_lengths_for_clean_data = original_lengths_clean_list

            # Store the raw (but truncated) IV data corresponding to X_clean for y_original in evaluation
            self.y_clean_raw_truncated = []
            for i in range(iv_data_clean_raw.shape[0]):
                curve = iv_data_clean_raw[i]
                length = original_lengths_clean_list[i]
                self.y_clean_raw_truncated.append(curve[:length])


            return X_clean, y_clean_norm_padded, scalers_clean_list, original_lengths_clean_list

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

    def _preprocess_input_parameters(self, params_df):
        # Same as original, ensure it's available or copy
        print("Preprocessing input parameters...")
        const_tol = 1e-10
        constant_cols = [c for c in params_df.columns if params_df[c].std(ddof=0) <= const_tol]
        if constant_cols:
            print(f"Removing constant columns: {constant_cols}")
            params_df = params_df.drop(columns=constant_cols)

        param_definitions = {
            'material': ['Eg', 'NCv', 'NCc', 'mu_e', 'mu_h', 'eps'],
            'device': ['A', 'Cn', 'Cp', 'Nt', 'Et', 'nD', 'nA', 'thickness', 'T', 'Sn', 'Sp', 'Rs', 'Rsh'],
            'operating': ['G', 'light_intensity'],
            'reference': ['Voc_ref', 'Jsc_ref', 'FF_ref', 'PCE_ref'],
            'loss': ['Qe_loss', 'R_loss', 'SRH_loss', 'series_loss', 'shunt_loss', 'other_loss']
        }
        transformers = []
        for group_name in self.param_group_names:
            group_cols_def = param_definitions.get(group_name, [])
            actual_cols_in_group = [p for p in group_cols_def if p in params_df.columns]
            if actual_cols_in_group:
                pipe_steps = [('log_transform', FunctionTransformer(np.log1p, validate=False))] if group_name == 'material' else []
                pipe_steps.append(('scaler', RobustScaler()))
                transformers.append((group_name, Pipeline(pipe_steps), actual_cols_in_group))

        self.input_preprocessor = ColumnTransformer(transformers, remainder='drop')
        X_processed = self.input_preprocessor.fit_transform(params_df)
        print(f"Processed input shape: {X_processed.shape}")
        return X_processed

    def _preprocess_iv_curves(self, truncated_iv_data_list, original_lengths_list):
        print("Preprocessing I-V curves (TRUNCATED, INDIVIDUAL NORMALIZATION)...")
        n_samples = len(truncated_iv_data_list)
        processed_normalized_curves_list = []
        scalers_list = []
        min_len_savgol = 5 # Savgol needs window_length < N, and window_length must be odd and >=3

        for i in range(n_samples):
            if i % 10000 == 0 and i > 0:
                 print(f"  Preprocessing I-V curve {i}/{n_samples}...")

            curve_truncated = truncated_iv_data_list[i].copy() # Already truncated
            current_len = original_lengths_list[i] # Length of this specific truncated curve

            if current_len >= min_len_savgol: # Only apply Savgol if long enough
                # Window length must be odd, less than num points, and typically > polyorder
                wl = min(max(min_len_savgol, (current_len // 10) * 2 + 1), current_len -1 if current_len % 2 == 0 else current_len -2 )
                wl = max(3, wl if wl % 2 != 0 else wl - 1) # Ensure odd and at least 3

                if wl < current_len and wl >=3 : # Final check
                    po = min(2, wl - 1) # Polyorder should be less than window length
                    curve_truncated = savgol_filter(curve_truncated, window_length=wl, polyorder=po)

            scaler = MinMaxScaler(feature_range=(0, 1))
            # Fit and transform the truncated curve (must be 2D for scaler)
            # Handle cases where curve_truncated might be all same values after filtering (min/max equal)
            if np.all(curve_truncated == curve_truncated[0]) and len(curve_truncated) > 1: # Constant value
                normalized_curve_truncated = np.zeros_like(curve_truncated) # Map constant to 0
            elif len(curve_truncated) == 0: # Should not happen due to pre-filtering
                normalized_curve_truncated = np.array([])
            else:
                normalized_curve_truncated = scaler.fit_transform(curve_truncated.reshape(-1, 1)).flatten()

            processed_normalized_curves_list.append(normalized_curve_truncated)
            scalers_list.append(scaler)

        print(f"I-V curves truncated, (optionally) filtered, and individually normalized.")
        return processed_normalized_curves_list, scalers_list

    def _remove_outlier_samples(self, X, y_normalized_padded, scalers_list_full, original_lengths_full, iv_data_raw_full, contamination=0.05):
        print("Detecting and removing outlier samples (from TRUNCATED data)...")
        # Calculate features for IsolationForest based on the actual data part of y_normalized_padded
        iv_features_for_outlier_detection = []
        for i in range(y_normalized_padded.shape[0]):
            norm_padded_curve = y_normalized_padded[i]
            length = original_lengths_full[i]
            actual_norm_data = norm_padded_curve[:length]
            if length > 1: # Need some data for these features
                std_val = np.std(actual_norm_data)
                trapz_val = np.trapz(actual_norm_data)
                sum_diff_neg = np.sum(np.diff(actual_norm_data) < 0)
                max_abs_diff = np.max(np.abs(np.diff(actual_norm_data))) if length > 1 else 0
                iv_features_for_outlier_detection.append([std_val, trapz_val, sum_diff_neg, max_abs_diff])
            else: # Not enough data, use safe defaults or skip
                iv_features_for_outlier_detection.append([0,0,0,0]) # Or consider this an outlier implicitly

        iv_features_np = np.array(iv_features_for_outlier_detection)

        self.outlier_detector = IsolationForest(contamination=contamination, random_state=42, n_estimators=100, n_jobs=-1)
        outliers = self.outlier_detector.fit_predict(iv_features_np) # Use features from normalized, truncated data

        normal_mask = outliers == 1
        n_outliers = np.sum(~normal_mask)
        print(f"Removed {n_outliers} outlier samples ({n_outliers/len(outliers)*100:.1f}%)")

        scalers_clean = [s for s, m in zip(scalers_list_full, normal_mask) if m]
        original_lengths_clean = [l for l, m in zip(original_lengths_full, normal_mask) if m]

        return X[normal_mask], y_normalized_padded[normal_mask], scalers_clean, original_lengths_clean, iv_data_raw_full[normal_mask]


    def fit_pca_model(self, X_train_np, y_train_norm_padded_np, n_components=15):
        # n_components for PCA should not exceed number of features (max_truncated_len_overall)
        # or number of samples.
        actual_n_components = min(n_components, self.max_truncated_len_overall, X_train_np.shape[0])
        if actual_n_components < n_components:
             print(f"  PCA n_components adjusted from {n_components} to {actual_n_components} due to data dimensions.")
        if actual_n_components <=1: # PCA needs at least 1 component, usually more.
            print(f"Warning: PCA cannot be effectively fitted with {actual_n_components} components. Skipping PCA model.")
            self.models['pca'] = None # Indicate PCA model is not available
            return self

        print(f"\n=== Fitting PCA Model (TRUNCATED DATA, n_components={actual_n_components}) ===")
        if self.cuml_ok:
            print("Using cuML PCA (GPU)...")
            y_train_cupy = cupy.asarray(y_train_norm_padded_np)
            self.pca_model = CumlPCA(n_components=actual_n_components)
            y_pca_cupy = self.pca_model.fit_transform(y_train_cupy)
            y_pca_np = cupy.asnumpy(y_pca_cupy)
            del y_train_cupy, y_pca_cupy; cupy.get_default_memory_pool().free_all_blocks()
        else:
            print("Using scikit-learn PCA (CPU)...")
            self.pca_model = SklearnPCA(n_components=actual_n_components)
            y_pca_np = self.pca_model.fit_transform(y_train_norm_padded_np)

        print(f"PCA Explained Variance (Top {min(10, actual_n_components)}): {self.pca_model.explained_variance_ratio_[:min(10, actual_n_components)]}")
        print(f"Total variance by {actual_n_components} comps: {np.sum(self.pca_model.explained_variance_ratio_):.4f}")

        self.pca_regressors = []
        regressor_params = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.8, 'random_state': 42}

        for i in range(actual_n_components):
            print(f"Training regressor for PC{i+1}...")
            if self.xgb_gpu_ok:
                regressor = BaseXGBRegressor(**regressor_params, tree_method='gpu_hist', n_jobs=-1)
            elif BaseXGBRegressor == SklearnGradientBoostingRegressor:
                regressor = SklearnGradientBoostingRegressor(**regressor_params, validation_fraction=0.1, n_iter_no_change=10)
            else:
                regressor = BaseXGBRegressor(**regressor_params, tree_method='hist', n_jobs=-1)

            regressor.fit(X_train_np, y_pca_np[:, i])
            self.pca_regressors.append(regressor)

        self.models['pca'] = {'pca_obj': self.pca_model, 'regressors': self.pca_regressors}
        print("PCA model fitting completed!")
        return self

    def _enhanced_physics_loss(self, y_true, y_pred): # y_true, y_pred are padded
        # Loss should ideally be calculated on the non-padded part.
        # This requires knowing original lengths inside the loss function, which is complex with tf.data.
        # For now, loss is on padded data. User can refine if needed.
        mse_fn = tf.keras.losses.MeanSquaredError()
        main_mse_loss = mse_fn(y_true, y_pred)

        # Penalties should also ideally apply to non-padded parts.
        # For simplicity, applying to full padded prediction.
        # Note: If padding is with last value, diff will be 0 in padded region.
        diff = y_pred[:, 1:] - y_pred[:, :-1]
        smoothness_penalty = tf.reduce_mean(tf.square(diff))
        monotonicity_penalty = tf.reduce_mean(tf.maximum(0.0, diff + 0.01)) # Assuming current generally decreases or stays flat

        # Jsc (first point) and Voc (last significant point) loss - tricky with variable length
        # For padded data, last point is not Voc.
        jsc_point_loss = mse_fn(y_true[:, 0:1], y_pred[:, 0:1])
        # Voc loss on padded data is not meaningful. We might skip or use a proxy.
        # For now, removing Voc point loss or making it less weighted due to padding.
        # Let's try to mask the loss calculation if possible, or focus on first point.

        w_smooth, w_mono, w_jsc = 0.01, 0.05, 0.1
        return main_mse_loss + w_smooth * smoothness_penalty + w_mono * monotonicity_penalty + \
               w_jsc * jsc_point_loss


    def fit_physics_informed_nn(self, X_train_np, y_train_norm_padded_np):
        print("\n=== Fitting Physics-Informed Neural Network (TRUNCATED DATA) ===")
        input_dim = X_train_np.shape[1]
        output_dim = self.max_truncated_len_overall # NN output matches padded length
        if output_dim == 0:
            print("Warning: Output dimension for NN is 0 (max_truncated_len_overall). Skipping NN model.")
            self.models['physics_nn'] = None
            return self

        inputs_layer = Input(shape=(input_dim,))
        processed_feature_names = self.input_preprocessor.get_feature_names_out()
        branch_outputs, branch_configs = [], {'material': 96, 'device': 96, 'operating': 48, 'reference': 48, 'loss': 48}
        all_grouped_indices = set()

        for group_prefix in self.param_group_names:
            indices = [i for i, name in enumerate(processed_feature_names) if name.startswith(group_prefix + "__")]
            if indices:
                all_grouped_indices.update(indices)
                branch_input = Lambda(lambda x, idx=indices: tf.gather(x, idx, axis=1))(inputs_layer)
                units = branch_configs.get(group_prefix, 32)
                path = Dense(units, activation='relu', name=f'{group_prefix}_d')(branch_input)
                path = BatchNormalization(name=f'{group_prefix}_bn')(path)
                path = Dropout(0.2, name=f'{group_prefix}_dr')(path)
                branch_outputs.append(path)

        remaining_indices = [i for i in range(input_dim) if i not in all_grouped_indices]
        if remaining_indices:
            rem_input = Lambda(lambda x, idx=remaining_indices: tf.gather(x, idx, axis=1))(inputs_layer)
            rem_path = Dense(32, activation='relu', name='rem_d')(rem_input)
            rem_path = BatchNormalization(name='rem_bn')(rem_path)
            rem_path = Dropout(0.2, name='rem_dr')(rem_path)
            branch_outputs.append(rem_path)

        combined = Concatenate()(branch_outputs) if len(branch_outputs) > 1 else (branch_outputs[0] if branch_outputs else inputs_layer)

        x = Dense(384, activation='relu')(combined)
        x = BatchNormalization()(x); x = Dropout(0.3)(x)
        x = Dense(768, activation='relu')(x)
        x = BatchNormalization()(x); x = Dropout(0.3)(x)
        # Output activation: sigmoid for [0,1] normalized data.
        outputs_layer = Dense(output_dim, activation='sigmoid', name='iv_output')(x)

        model = Model(inputs=inputs_layer, outputs=outputs_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=self._enhanced_physics_loss, metrics=['mae'])

        batch_size = 128
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_np, y_train_norm_padded_np))
        train_dataset = train_dataset.shuffle(buffer_size=min(X_train_np.shape[0], 10000)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        num_val_samples = int(0.2 * X_train_np.shape[0])
        if num_val_samples > 0 :
            val_dataset = tf.data.Dataset.from_tensor_slices((X_train_np[-num_val_samples:], y_train_norm_padded_np[-num_val_samples:]))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            actual_train_dataset = tf.data.Dataset.from_tensor_slices((X_train_np[:-num_val_samples], y_train_norm_padded_np[:-num_val_samples]))
            actual_train_dataset = actual_train_dataset.shuffle(buffer_size=min(X_train_np.shape[0]-num_val_samples, 10000)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else: # Not enough samples for validation split
            val_dataset = None
            actual_train_dataset = train_dataset
            print("  Warning: Not enough samples for a validation set during NN training.")


        callbacks = [
            EarlyStopping(monitor='val_loss' if val_dataset else 'loss', patience=30, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss' if val_dataset else 'loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1)
        ]

        history = model.fit(actual_train_dataset, epochs=200,
                            validation_data=val_dataset, callbacks=callbacks, verbose=1)
        self.models['physics_nn'] = model
        self.training_history = history
        return self

    def fit_ensemble_model(self, X_train_np, y_train_norm_padded_np):
        print("\n=== Fitting Ensemble Model (TRUNCATED DATA) ===")
        if self.max_truncated_len_overall == 0:
            print("Warning: Output dimension for Ensemble is 0. Skipping Ensemble model.")
            self.models['ensemble'] = None
            return self

        regressor_params = {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 8, 'subsample': 0.7, 'random_state': 42}
        base_models_config = []

        if self.xgb_gpu_ok:
            print("  Using XGBoost (GPU) for ensemble members...")
            base_models_config.append(('xgb1', BaseXGBRegressor(**regressor_params, tree_method='gpu_hist', n_jobs=-1)))
            base_models_config.append(('xgb2', BaseXGBRegressor(n_estimators=100, max_depth=10, tree_method='gpu_hist', n_jobs=-1)))
        elif BaseXGBRegressor != SklearnGradientBoostingRegressor :
             print("  Using XGBoost (CPU) for ensemble members...")
             base_models_config.append(('xgb1_cpu', BaseXGBRegressor(**regressor_params, tree_method='hist', n_jobs=-1)))
             base_models_config.append(('xgb2_cpu', BaseXGBRegressor(n_estimators=100, max_depth=10, tree_method='hist', n_jobs=-1)))
        else:
            print("  Using scikit-learn RandomForestRegressor (CPU) for ensemble...")
            base_models_config.append(('rf_cpu', SklearnRandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)))

        self.ensemble_models = {}
        for name, base_model_instance in base_models_config:
            print(f"Training {name.upper()} ensemble member...")
            multi_model = MultiOutputRegressor(base_model_instance, n_jobs=1)
            multi_model.fit(X_train_np, y_train_norm_padded_np) # Target is padded
            self.ensemble_models[name] = multi_model

        self.models['ensemble'] = self.ensemble_models
        return self

    def predict(self, X_np, scalers_for_X, original_lengths_for_X, model_type='pca'):
        if model_type not in self.models or self.models[model_type] is None:
            print(f"Model {model_type} not fitted or available. Returning empty predictions.")
            # Return list of empty arrays or handle as error, matching evaluate_model expectations
            return [np.array([]) for _ in range(X_np.shape[0])]


        model_bundle = self.models[model_type]
        y_pred_normalized_padded = None

        if model_type == 'pca':
            pca_obj, regressors = model_bundle['pca_obj'], model_bundle['regressors']
            if not regressors : # PCA might have been skipped if n_components was too low
                 print(f"PCA model '{model_type}' has no regressors. Returning empty predictions.")
                 return [np.array([]) for _ in range(X_np.shape[0])]
            y_pca_pred_np = np.zeros((X_np.shape[0], len(regressors)))
            for i, reg in enumerate(regressors):
                y_pca_pred_np[:, i] = reg.predict(X_np)

            if self.cuml_ok and isinstance(pca_obj, CumlPCA):
                y_pca_pred_cupy = cupy.asarray(y_pca_pred_np)
                y_pred_normalized_padded_cupy = pca_obj.inverse_transform(y_pca_pred_cupy)
                y_pred_normalized_padded = cupy.asnumpy(y_pred_normalized_padded_cupy)
                del y_pca_pred_cupy, y_pred_normalized_padded_cupy; cupy.get_default_memory_pool().free_all_blocks()
            else:
                y_pred_normalized_padded = pca_obj.inverse_transform(y_pca_pred_np)

        elif model_type == 'physics_nn':
            y_pred_normalized_padded = model_bundle.predict(X_np, batch_size=512, verbose=0)

        elif model_type == 'ensemble':
            predictions = [model.predict(X_np) for _, model in model_bundle.items()]
            y_pred_normalized_padded = np.mean(predictions, axis=0)

        if y_pred_normalized_padded.shape[0] != len(scalers_for_X):
            raise ValueError(f"Sample-scaler mismatch: {y_pred_normalized_padded.shape[0]} vs {len(scalers_for_X)}")

        # Inverse transform and slice to original truncated length
        y_pred_original_truncated_list = []
        for i in range(y_pred_normalized_padded.shape[0]):
            p_norm_padded = y_pred_normalized_padded[i]
            scaler = scalers_for_X[i]
            original_len = original_lengths_for_X[i]

            if original_len == 0: # Should not happen if filtered, but for safety
                y_pred_original_truncated_list.append(np.array([]))
                continue

            # Scaler expects 2D input
            p_orig_padded = scaler.inverse_transform(p_norm_padded.reshape(-1, 1)).flatten()
            p_orig_truncated = p_orig_padded[:original_len]
            y_pred_original_truncated_list.append(p_orig_truncated)

        return y_pred_original_truncated_list # List of arrays with variable lengths

    def evaluate_model(self, X_test, y_test_original_truncated_list, # True values are already truncated lists
                       scalers_test, original_lengths_test, model_type='pca'):
        print(f"\n=== Evaluating {model_type.upper()} Model (TRUNCATED DATA) ===")

        # Predictions will be a list of truncated arrays
        y_pred_original_truncated_list = self.predict(X_test, scalers_test, original_lengths_test, model_type=model_type)

        if not y_test_original_truncated_list or not y_pred_original_truncated_list:
            print("  No data for evaluation.")
            return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'per_curve_R2_mean': np.nan, 'predictions': []}

        # Flatten all curves for overall MAE/RMSE
        # Ensure corresponding curves are not empty before concatenating
        flat_true_truncated = np.concatenate([arr for arr in y_test_original_truncated_list if len(arr)>0])
        flat_pred_truncated = np.concatenate([arr for arr in y_pred_original_truncated_list if len(arr)>0])

        if len(flat_true_truncated) == 0: # No points to evaluate
            print("  No valid data points for MAE/RMSE/R2 calculation.")
            mae, rmse, r2_global = np.nan, np.nan, np.nan
        else:
            mae = mean_absolute_error(flat_true_truncated, flat_pred_truncated)
            mse = mean_squared_error(flat_true_truncated, flat_pred_truncated)
            rmse = np.sqrt(mse)
            if np.std(flat_true_truncated) > 1e-9:
                r2_global = r2_score(flat_true_truncated, flat_pred_truncated)
            else: # True is constant
                r2_global = 1.0 if mse < 1e-6 else 0.0

        print(f"Truncated Original scale: MAE: {mae:.6f}, RMSE: {rmse:.6f}, Global R²: {r2_global:.4f}")

        per_curve_r2 = []
        for i in range(len(y_test_original_truncated_list)):
            true_c = y_test_original_truncated_list[i]
            pred_c = y_pred_original_truncated_list[i]
            if len(true_c) > 1 and len(pred_c) == len(true_c): # Need at least 2 points for R2 and matching length
                if np.std(true_c) > 1e-9: # Check for variance in true curve
                    per_curve_r2.append(r2_score(true_c, pred_c))
                elif mean_squared_error(true_c, pred_c) < 1e-6 : # Constant true, prediction is also close
                    per_curve_r2.append(1.0)
                else: # Constant true, prediction differs
                    per_curve_r2.append(0.0) # Or other value indicating poor fit for constant

        if per_curve_r2:
            mean_r2_pc = np.mean(per_curve_r2)
            std_r2_pc = np.std(per_curve_r2)
            min_r2_pc = np.min(per_curve_r2)
            max_r2_pc = np.max(per_curve_r2)
            print(f"  Per-curve R² (truncated) - Mean: {mean_r2_pc:.4f}, Std: {std_r2_pc:.4f}, Min: {min_r2_pc:.4f}, Max: {max_r2_pc:.4f}")
        else:
            mean_r2_pc, std_r2_pc = np.nan, np.nan
            print("  Not enough valid curves for per-curve R² statistics.")

        return {'MAE': mae, 'RMSE': rmse, 'R2': r2_global,
                'per_curve_R2_mean': mean_r2_pc,
                'per_curve_R2_std': std_r2_pc,
                'predictions': y_pred_original_truncated_list}

    def plot_results(self, X_test, y_test_original_truncated_list, scalers_test, original_lengths_test, model_type='pca', n_samples=4):
        y_pred_list = self.predict(X_test, scalers_test, original_lengths_test, model_type=model_type)

        num_available_samples = len(y_test_original_truncated_list)
        if num_available_samples == 0:
            print(f"No samples available to plot for model {model_type}.")
            return

        indices = np.random.choice(num_available_samples, min(n_samples, num_available_samples), replace=False)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10)); axes = axes.ravel()
        for i, idx in enumerate(indices):
            ax = axes[i]
            true_curve = y_test_original_truncated_list[idx]
            pred_curve = y_pred_list[idx]

            if len(true_curve) == 0 or len(pred_curve) == 0 or len(true_curve) != len(pred_curve):
                ax.text(0.5, 0.5, f"Sample {idx}\nData Mismatch or Empty", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Sample {idx} - Error')
                continue

            # Voltage axis scaled to truncated length
            n_points_curve = len(true_curve)
            # voltage_points = np.linspace(0, ASSUMED_ORIGINAL_MAX_VOLTAGE * n_points_curve / ASSUMED_ORIGINAL_IV_POINTS, n_points_curve)
            # Or, just use point indices if voltage scaling is complex/uncertain for truncated
            voltage_points = np.arange(n_points_curve)
            x_label = 'Voltage Points (Scaled)' # Or 'Voltage (V)' if properly scaled

            ax.plot(voltage_points, true_curve, 'b-', lw=2, label='Actual (Trunc)', alpha=0.8)
            ax.plot(voltage_points, pred_curve, 'r--', lw=2, label='Predicted (Trunc)', alpha=0.8)

            current_r2 = np.nan
            if len(true_curve) > 1 and np.std(true_curve) > 1e-9:
                current_r2 = r2_score(true_curve, pred_curve)
            elif len(true_curve) > 1 and mean_squared_error(true_curve, pred_curve) < 1e-6:
                current_r2 = 1.0
            elif len(true_curve) > 1:
                current_r2 = 0.0

            ax.set_title(f'Sample {idx} - {model_type.upper()} (Trunc R² = {current_r2:.3f})')
            ax.set_xlabel(x_label); ax.set_ylabel('Current Density (mA/cm²)')
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR_TRUNCATED / f"truncated_cuda_{model_type}_predictions.png", dpi=300, bbox_inches='tight')
        print(f"Truncated prediction plot saved: {OUTPUT_DIR_TRUNCATED / f'truncated_cuda_{model_type}_predictions.png'}")

    def plot_pca_analysis(self):
        if 'pca' not in self.models or not self.models['pca'] or not self.models['pca']['pca_obj']:
            print("PCA model not fitted or available for plotting!"); return

        pca_obj = self.models['pca']['pca_obj']
        pca_regressors = self.models['pca']['regressors']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        explained_var = pca_obj.explained_variance_ratio_
        if isinstance(explained_var, cupy.ndarray): explained_var = cupy.asnumpy(explained_var)
        cumulative_var = np.cumsum(explained_var)

        axes[0,0].bar(range(1,len(explained_var)+1), explained_var); axes[0,0].set_title('PCA Explained Variance (Truncated)')
        axes[0,1].plot(range(1,len(cumulative_var)+1), cumulative_var,'bo-'); axes[0,1].set_title('Cumulative Explained Var (Truncated)')
        axes[0,1].axhline(y=0.95, c='r', ls='--', label='95%'); axes[0,1].legend()

        components = pca_obj.components_ # Shape (n_components, max_truncated_len_overall)
        if isinstance(components, cupy.ndarray): components = cupy.asnumpy(components)

        # X-axis for components is now up to max_truncated_len_overall
        # voltage_points = np.linspace(0, 1.2, components.shape[1]) # This may not be accurate for truncated
        component_x_axis = np.arange(components.shape[1])


        for i in range(min(4, components.shape[0])): axes[1,0].plot(component_x_axis, components[i], label=f'PC{i+1}')
        axes[1,0].set_title('Principal Components (Truncated)'); axes[1,0].set_xlabel("Component Feature Index"); axes[1,0].legend()

        if pca_regressors and hasattr(pca_regressors[0], 'feature_importances_'):
            importance = pca_regressors[0].feature_importances_
            f_names = self.input_preprocessor.get_feature_names_out() if self.input_preprocessor else [f"F{i}" for i in range(len(importance))]
            s_indices = np.argsort(importance)[-10:]; labels = [f_names[j] for j in s_indices]
            axes[1,1].barh(range(len(s_indices)),importance[s_indices]); axes[1,1].set_yticks(range(len(s_indices))); axes[1,1].set_yticklabels(labels)
            axes[1,1].set_title('Top Features for PC1 Prediction (Truncated)')
        else: axes[1,1].text(0.5,0.5,'Feat. importance N/A', ha='center',va='center',transform=axes[1,1].transAxes)

        for r_ax in axes:
            for ax in r_ax: ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR_TRUNCATED / "truncated_cuda_pca_analysis.png", dpi=300, bbox_inches='tight')


def main_truncated(): # New main function for this script
    print("=== TRUNCATED Perovskite I-V Curve Reconstruction (CUDA & Scaling) ===")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} Physical GPUs available to TensorFlow.")
        except RuntimeError as e: print(f"Error setting memory growth: {e}")
    else: print("No GPUs available to TensorFlow.")

    OUTPUT_DIR_TRUNCATED.mkdir(parents=True, exist_ok=True)
    reconstructor = TruncatedIVReconstructor(use_gpu_if_available=True)

    # X_clean, y_clean_norm_padded, scalers_clean_list, original_lengths_clean_list
    X_clean, y_clean_norm_padded, scalers_clean, lengths_clean = reconstructor.load_and_prepare_data()

    if X_clean is None or y_clean_norm_padded is None or not scalers_clean or not lengths_clean:
        print("Failed to load and prepare data. Exiting.")
        return

    # y_clean_raw_truncated is a list of 1D numpy arrays (the true, truncated, original scale IV curves)
    y_clean_raw_truncated_list = reconstructor.y_clean_raw_truncated

    # Ensure all inputs to train_test_split are arrays or lists of consistent first dimension
    # Convert lists to object arrays for train_test_split if they contain non-numerical objects like scalers
    # or arrays of varying lengths (though y_clean_raw_truncated_list needs careful handling if split directly)

    # We need to split indices, then use indices to get corresponding items
    num_samples = X_clean.shape[0]
    indices = np.arange(num_samples)

    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

    X_train = X_clean[train_indices]
    X_test = X_clean[test_indices]

    y_train_norm_padded = y_clean_norm_padded[train_indices]
    # y_test_norm_padded is not directly used by evaluate, but good to have
    y_test_norm_padded = y_clean_norm_padded[test_indices]

    scalers_train_list = [scalers_clean[i] for i in train_indices] # Not used directly in fit, but good for consistency
    scalers_test_list = [scalers_clean[i] for i in test_indices]

    lengths_train_list = [lengths_clean[i] for i in train_indices] # Not used directly in fit
    lengths_test_list = [lengths_clean[i] for i in test_indices]

    # This is the crucial ground truth for evaluation: list of truncated original-scale curves
    y_test_original_truncated_list = [y_clean_raw_truncated_list[i] for i in test_indices]

    print(f"\nData shapes: X_train: {X_train.shape}, y_train_norm_padded: {y_train_norm_padded.shape}")
    print(f"Test set: {len(y_test_original_truncated_list)} original truncated curves, {len(scalers_test_list)} scalers, {len(lengths_test_list)} lengths.")

    # --- Fit PCA model ---
    reconstructor.fit_pca_model(X_train, y_train_norm_padded, n_components=20)
    if reconstructor.models.get('pca'): # Check if PCA model was successfully fitted
        reconstructor.plot_pca_analysis()
        pca_results = reconstructor.evaluate_model(X_test, y_test_original_truncated_list, scalers_test_list, lengths_test_list, 'pca')
        reconstructor.plot_results(X_test, y_test_original_truncated_list, scalers_test_list, lengths_test_list, 'pca')
    else:
        pca_results = None
        print("PCA model was not fitted, skipping evaluation and plotting for PCA.")

    # --- Fit Physics-Informed NN ---
    reconstructor.fit_physics_informed_nn(X_train, y_train_norm_padded)
    if reconstructor.models.get('physics_nn'):
        nn_results = reconstructor.evaluate_model(X_test, y_test_original_truncated_list, scalers_test_list, lengths_test_list, 'physics_nn')
        reconstructor.plot_results(X_test, y_test_original_truncated_list, scalers_test_list, lengths_test_list, 'physics_nn')
    else:
        nn_results = None
        print("Physics NN model was not fitted, skipping evaluation and plotting for NN.")

    # --- Fit Ensemble Model ---
    reconstructor.fit_ensemble_model(X_train, y_train_norm_padded)
    if reconstructor.models.get('ensemble'):
        ensemble_results = reconstructor.evaluate_model(X_test, y_test_original_truncated_list, scalers_test_list, lengths_test_list, 'ensemble')
        reconstructor.plot_results(X_test, y_test_original_truncated_list, scalers_test_list, lengths_test_list, 'ensemble')
    else:
        ensemble_results = None
        print("Ensemble model was not fitted, skipping evaluation and plotting for Ensemble.")

    print("\n" + "="*70 + "\nFINAL RESULTS COMPARISON (TRUNCATED DATA)\n" + "="*70)
    results_summary = {}
    if pca_results: results_summary['PCA (Truncated)'] = pca_results
    if nn_results: results_summary['Physics NN (Truncated)'] = nn_results
    if ensemble_results: results_summary['Ensemble (Truncated)'] = ensemble_results

    print(f"{'Model':<25} {'Global R²':<12} {'MAE':<10} {'RMSE':<10} {'Mean R²/curve':<15}")
    print("-" * 80)
    best_model_name, best_r2_per_curve = "N/A", -np.inf

    for name, res_dict in results_summary.items():
        if res_dict: # Check if results dict is not None
            glob_r2 = res_dict.get('R2', np.nan)
            mae_val = res_dict.get('MAE', np.nan)
            rmse_val = res_dict.get('RMSE', np.nan)
            mean_r2_pc_val = res_dict.get('per_curve_R2_mean', np.nan)
            print(f"{name:<25} {glob_r2:<12.4f} {mae_val:<10.6f} {rmse_val:<10.6f} {mean_r2_pc_val:<15.4f}")
            if not np.isnan(mean_r2_pc_val) and mean_r2_pc_val > best_r2_per_curve:
                best_r2_per_curve = mean_r2_pc_val
                best_model_name = name
        else:
            print(f"{name:<25} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<15}")

    print(f"\nBest performing model (Mean R²/curve on Truncated Data): {best_model_name} (R² = {best_r2_per_curve:.4f})")

    joblib.dump(reconstructor.input_preprocessor, OUTPUT_DIR_TRUNCATED / "input_preprocessor_truncated.joblib")
    if reconstructor.models.get('pca'): joblib.dump(reconstructor.models['pca'], OUTPUT_DIR_TRUNCATED / "pca_model_bundle_truncated.joblib")
    if reconstructor.models.get('physics_nn'): reconstructor.models['physics_nn'].save(OUTPUT_DIR_TRUNCATED / "physics_nn_model_truncated.keras")
    if reconstructor.models.get('ensemble'): joblib.dump(reconstructor.models['ensemble'], OUTPUT_DIR_TRUNCATED / "ensemble_models_bundle_truncated.joblib")

    # Save scalers and original lengths (they are crucial for reproducing predictions)
    joblib.dump({
        'scalers': reconstructor.final_scalers_for_clean_data,
        'lengths': reconstructor.final_original_lengths_for_clean_data,
        'max_len': reconstructor.max_truncated_len_overall
    }, OUTPUT_DIR_TRUNCATED / "data_processing_info_truncated.joblib")
    print("\n=== Models, preprocessor, and processing info (truncated) saved! ===")

    # --- Visualization: Compare True vs Predicted IV Curves for All Models (Truncated) ---
    # This plotting part can be adapted from your original script, using the truncated data
    if y_test_original_truncated_list and any(results_summary.values()):
        y_preds_dict_truncated = {}
        if pca_results and 'predictions' in pca_results:
            y_preds_dict_truncated["PCA (Trunc)"] = pca_results["predictions"]
        if nn_results and 'predictions' in nn_results:
            y_preds_dict_truncated["NN (Trunc)"] = nn_results["predictions"]
        if ensemble_results and 'predictions' in ensemble_results:
            y_preds_dict_truncated["Ensemble (Trunc)"] = ensemble_results["predictions"]

        if y_preds_dict_truncated:
            plot_multi_model_comparison_truncated(
                y_test_original_truncated_list,
                y_preds_dict_truncated,
                n_samples_plot=4,
                output_dir=OUTPUT_DIR_TRUNCATED
            )

def plot_multi_model_comparison_truncated(y_true_list, y_preds_dict, n_samples_plot, output_dir):
    """
    Plots actual vs predicted for truncated IV curves from multiple models.
    y_true_list: list of 1D numpy arrays (true truncated curves)
    y_preds_dict: dict of {model_name: list_of_prediction_arrays}
    """
    num_curves_available = len(y_true_list)
    if num_curves_available == 0: return

    sample_indices = np.random.choice(num_curves_available, min(n_samples_plot, num_curves_available), replace=False)
    n_models = len(y_preds_dict)

    fig, axes = plt.subplots(len(sample_indices), n_models + 1, figsize=(5 * (n_models + 1), 4 * len(sample_indices)), squeeze=False)

    for row, idx in enumerate(sample_indices):
        true_curve_idx = y_true_list[idx]

        # Voltage scaling for this specific truncated curve
        n_points_curve = len(true_curve_idx)
        if n_points_curve == 0: continue # Skip if empty

        # voltage_points_curve = np.linspace(0, ASSUMED_ORIGINAL_MAX_VOLTAGE * n_points_curve / ASSUMED_ORIGINAL_IV_POINTS, n_points_curve)
        # Simpler: use point indices for x-axis for truncated plots to avoid complexity with voltage scaling
        voltage_points_curve = np.arange(n_points_curve)
        x_label_plot = "Voltage Points (Scaled)"


        # Plot True
        ax_true = axes[row, 0]
        ax_true.plot(voltage_points_curve, true_curve_idx, 'b-', lw=2, label='True (Trunc)')
        ax_true.set_title(f"Sample {idx}\nGround Truth (Trunc)")
        ax_true.set_xlabel(x_label_plot); ax_true.set_ylabel("Current")
        ax_true.grid(True, alpha=0.5); ax_true.legend()

        # Plot predictions
        for col, (model_name, y_pred_list_model) in enumerate(y_preds_dict.items()):
            ax_pred = axes[row, col + 1]
            pred_curve_idx = y_pred_list_model[idx]

            if len(pred_curve_idx) != n_points_curve: # Should match from predict() logic
                 ax_pred.text(0.5,0.5, "Length Mismatch", ha='center', va='center')
                 ax_pred.set_title(f"{model_name}\nError")
                 continue

            ax_pred.plot(voltage_points_curve, true_curve_idx, 'b-', lw=1.5, label='True (Trunc)', alpha=0.7)
            ax_pred.plot(voltage_points_curve, pred_curve_idx, 'r--', lw=2, label='Pred (Trunc)')

            r2_val = np.nan
            if n_points_curve > 1 and np.std(true_curve_idx) > 1e-9:
                r2_val = r2_score(true_curve_idx, pred_curve_idx)
            elif n_points_curve > 1 and mean_squared_error(true_curve_idx, pred_curve_idx) < 1e-6:
                r2_val = 1.0
            elif n_points_curve > 1:
                 r2_val = 0.0

            ax_pred.set_title(f"{model_name}\nR²={r2_val:.3f}")
            ax_pred.set_xlabel(x_label_plot); ax_pred.grid(True, alpha=0.5); ax_pred.legend()

    plt.tight_layout()
    fname = output_dir / "truncated_model_comparison_iv_curves.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\n[Visualization] Truncated multi-model IV curve comparison plot saved: {fname}")


if __name__ == "__main__":
    main_truncated() # Call the main function for the truncated version
