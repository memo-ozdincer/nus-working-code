# ==============================================================================
#  CELL 1: THE DEFINITIVE SCALAR PREDICTOR FACTORY (FINAL, RECON-COMPATIBLE)
# ==============================================================================
#
#  Abstract:
#  This script is the final, production-ready version of the scalar model
#  factory. It incorporates all best practices for robustness and is now
#  updated to produce output files directly compatible with downstream
#  reconstruction pipelines.
#
#  FINAL HARDENING & RECON-COMPATIBILITY:
#  - Added `bagging_seed` and `feature_fraction_seed` for full LGBM determinism.
#  - Expanded the `manifest.json` to include MLP architecture and voltage grid.
#  - **[NEW]** Exports four separate .txt files (Isc, Voc, FF, Vmpp) with NaN
#    padding for invalid curves, ensuring an output length of N_raw.
#  - **[NEW]** The manifest is updated to point to these new .txt files for
#    seamless integration with loader scripts.
#
# ==============================================================================
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from pathlib import Path
from tqdm.auto import tqdm
import os
import joblib
import matplotlib.pyplot as plt
import json
import pkg_resources

# --- PyTorch Imports ---
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- 1. Configuration & Full Reproducibility Setup ---
print("--- Initializing The Definitive Model Factory ---")
SEED = 42
# Set all seeds for full reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    DRIVE_PATH = "/content/drive/MyDrive/NewDatas/two"
    print(f"Google Drive mounted. Using path: {DRIVE_PATH}")
except (ImportError, RuntimeError):
    DRIVE_PATH = "./"
    print("Not in Colab or Drive mount failed. Using local path.")

Path(DRIVE_PATH).mkdir(parents=True, exist_ok=True)
INPUT_FILE_IV = Path(DRIVE_PATH) / "iV_m.txt"
INPUT_FILE_PARAMS = Path(DRIVE_PATH) / "LHS_parameters_m.txt"
COLNAMES = ['lH','lP','lE','muHh','muPh','muPe','muEe','NvH','NcH','NvE','NcE','NvP','NcP','chiHh','chiHe','chiPh','chiPe','chiEh','chiEe','Wlm','Whm','epsH','epsP','epsE','Gavg','Aug','Brad','Taue','Tauh','vII','vIII']

# --- 2. Data Loading & Preprocessing ---
print("\n--- Loading and Processing Data ---")
params_df_raw = pd.read_csv(INPUT_FILE_PARAMS, header=None, names=COLNAMES)
iv_data_raw = np.loadtxt(INPUT_FILE_IV, delimiter=',')
full_v_grid = np.concatenate([np.arange(0, 0.4 + 1e-8, 0.1), np.arange(0.425, 1.4 + 1e-8, 0.025)])

def calculate_pv_params(voltage_grid, current_curve):
    try:
        jsc = current_curve[0];
        if jsc <= 1e-9: return None
        interpolator = PchipInterpolator(voltage_grid, current_curve, extrapolate=False)
        v_fine = np.linspace(0, 1.5, 2000); i_fine = interpolator(v_fine)
        zero_cross_idx = np.where(i_fine <= 0)[0]
        if not len(zero_cross_idx): return None
        voc = v_fine[zero_cross_idx[0]]; power = v_fine * i_fine; power[v_fine > voc] = 0
        v_mpp, i_mpp = v_fine[np.argmax(power)], i_fine[np.argmax(power)]
        if voc * jsc < 1e-9: return None
        return {"Jsc": jsc, "Voc": voc, "FF": (v_mpp * i_mpp) / (voc * jsc), "Vmpp": v_mpp}
    except: return None

results, valid_indices = [], []
for i in tqdm(range(len(iv_data_raw)), desc="Processing Curves"):
    params = calculate_pv_params(full_v_grid, iv_data_raw[i])
    if params: results.append(params); valid_indices.append(i)

iv_summary_df = pd.DataFrame(results)
X = params_df_raw.iloc[valid_indices].reset_index(drop=True)
y_jsc, y_voc, y_ff, y_vmpp = iv_summary_df['Jsc'], iv_summary_df['Voc'], iv_summary_df['FF'], iv_summary_df['Vmpp']

X_train, X_test, y_jsc_train, y_jsc_test, y_voc_train, y_voc_test, y_ff_train, y_ff_test, y_vmpp_train, y_vmpp_test = train_test_split(
    X, y_jsc, y_voc, y_ff, y_vmpp, test_size=0.2, random_state=SEED, shuffle=True
)
print(f"Data prepared. Training on {len(X_train)}, testing on {len(X_test)}.")

# --- 3. Master Data Scaling ---
base_scaler = StandardScaler()
X_train_scaled = base_scaler.fit_transform(X_train)
X_test_scaled = base_scaler.transform(X_test)
joblib.dump(base_scaler, Path(DRIVE_PATH) / "standard_scaler.joblib")
print("\nBase parameter scaler has been fitted and saved.")

# --- 4. PyTorch Model Definitions and Helpers ---
MLP_CONFIG = {
    "layers": [
        {"type": "Linear", "out_features": 384},
        {"type": "ReLU"},
        {"type": "BatchNorm1d", "num_features": 384},
        {"type": "Dropout", "p": 0.3},
        {"type": "Linear", "out_features": 256},
        {"type": "ReLU"},
        {"type": "BatchNorm1d", "num_features": 256},
        {"type": "Dropout", "p": 0.3},
        {"type": "Linear", "out_features": 1}
    ]
}

class MLP(nn.Module):
    def __init__(self, input_features):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 384), nn.ReLU(), nn.BatchNorm1d(384), nn.Dropout(0.3),
            nn.Linear(384, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.layers(x)

def train_pytorch_model(model, X_train_np, y_train_np, device, epochs=80):
    X_tensor = torch.tensor(X_train_np, dtype=torch.float32); y_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor); train_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)
    model.to(device); criterion = nn.MSELoss(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); loss = criterion(outputs, labels); optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model

def predict_pytorch_model(model, X_np, device, batch_size=1024):
    model.to(device); model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            batch_x = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
            outputs = model(batch_x); all_preds.append(outputs.cpu().numpy())
    return np.vstack(all_preds).flatten()

# --- 5. Define Model Hyperparameters for Full Determinism ---
LGBM_BASE_PARAMS = {'random_state': SEED, 'bagging_seed': SEED, 'feature_fraction_seed': SEED}
LGBM_JSC_PARAMS = {**LGBM_BASE_PARAMS, 'n_estimators': 400}
LGBM_FF_PARAMS = {**LGBM_BASE_PARAMS, 'n_estimators': 500}
LGBM_VMPP_PARAMS = {**LGBM_BASE_PARAMS, 'n_estimators': 600, 'num_leaves': 60}

# --- 6. Train Foundational Models (Jsc & Voc) ---
print("\n--- Training and Saving Foundational Models ---")
jsc_lgbm = lgb.LGBMRegressor(**LGBM_JSC_PARAMS)
jsc_lgbm.fit(X_train_scaled, y_jsc_train)
joblib.dump(jsc_lgbm, Path(DRIVE_PATH) / "jsc_model.joblib")
print("  - Jsc LGBM model trained and saved.")

voc_mlp_final = MLP(input_features=X_train_scaled.shape[1])
voc_mlp_final = train_pytorch_model(voc_mlp_final, X_train_scaled, y_voc_train.values, device)
torch.save(voc_mlp_final.state_dict(), Path(DRIVE_PATH) / "voc_mlp_model.pth")
print("  - Voc MLP final model (trained on all data) saved.")

# --- 7. Generate LEAK-PROOF OOF Predictions for ALL Scalars ---
print("\n--- Generating LEAK-PROOF OOF Predictions for Downstream Models ---")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

jsc_oof_preds = cross_val_predict(lgb.LGBMRegressor(**LGBM_JSC_PARAMS), X_train_scaled, y_jsc_train, cv=kf)
print("  - Generated OOF predictions for Jsc.")

voc_oof_preds = np.zeros(len(X_train))
pbar = tqdm(kf.split(X_train_scaled), total=5, desc="Voc OOF Preds")
for fold, (train_idx, val_idx) in enumerate(pbar):
    X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train_fold, y_val_fold = y_voc_train.iloc[train_idx], y_voc_train.iloc[val_idx]
    fold_model = MLP(X_train_fold.shape[1])
    trained_fold_model = train_pytorch_model(fold_model, X_train_fold, y_train_fold.values, device, epochs=50)
    val_preds = predict_pytorch_model(trained_fold_model, X_val_fold, device)
    voc_oof_preds[val_idx] = val_preds
    pbar.set_postfix({'Fold R²': f'{r2_score(y_val_fold, val_preds):.4f}'})
print("  - Generated OOF predictions for Voc.")

X_ff_train_features_oof = np.hstack([X_train_scaled, jsc_oof_preds[:, np.newaxis], voc_oof_preds[:, np.newaxis]])
ff_oof_preds = cross_val_predict(lgb.LGBMRegressor(**LGBM_FF_PARAMS), X_ff_train_features_oof, y_ff_train, cv=kf)
print("  - Generated OOF predictions for FF.")

vmpp_oof_preds = np.zeros(len(X_train))
vmpp_meta_feature_oof = voc_oof_preds * ff_oof_preds
X_vmpp_train_features_oof = np.hstack([X_ff_train_features_oof, ff_oof_preds[:, np.newaxis], vmpp_meta_feature_oof[:, np.newaxis]])

for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train_scaled), total=5, desc="Vmpp OOF Preds")):
    vmpp_fold_model = lgb.LGBMRegressor(**LGBM_VMPP_PARAMS)
    vmpp_fold_model.fit(X_vmpp_train_features_oof[train_idx], y_vmpp_train.iloc[train_idx])
    vmpp_oof_preds[val_idx] = vmpp_fold_model.predict(X_vmpp_train_features_oof[val_idx])
print("  - Generated OOF predictions for Vmpp.")

# --- 8. Train FINAL Downstream Models on FULL Training Data ---
print("\n--- Training Final Downstream Models for Inference ---")
ff_lgbm_final = lgb.LGBMRegressor(**LGBM_FF_PARAMS); ff_lgbm_final.fit(X_ff_train_features_oof, y_ff_train)
joblib.dump(ff_lgbm_final, Path(DRIVE_PATH) / "ff_model.joblib"); print("  - FF LGBM final model trained and saved.")

vmpp_lgbm_final = lgb.LGBMRegressor(**LGBM_VMPP_PARAMS); vmpp_lgbm_final.fit(X_vmpp_train_features_oof, y_vmpp_train)
joblib.dump(vmpp_lgbm_final, Path(DRIVE_PATH) / "vmpp_model.joblib"); print("  - Vmpp LGBM final model trained and saved.")

# --- 9. FINAL VALIDATION & PREDICTION on TEST SET ---
print("\n--- Final Validation & Prediction on Held-Out Test Set ---")
jsc_preds_test = jsc_lgbm.predict(X_test_scaled)
voc_preds_test = predict_pytorch_model(voc_mlp_final, X_test_scaled, device)
X_ff_test_features = np.hstack([X_test_scaled, jsc_preds_test[:, np.newaxis], voc_preds_test[:, np.newaxis]])
ff_preds_test = ff_lgbm_final.predict(X_ff_test_features)
vmpp_meta_feature_test = voc_preds_test * ff_preds_test
X_vmpp_test_final = np.hstack([X_ff_test_features, ff_preds_test[:, np.newaxis], vmpp_meta_feature_test[:, np.newaxis]])
vmpp_preds_test = vmpp_lgbm_final.predict(X_vmpp_test_final)

# --- 10. Assemble, Sanitize, and Export ALL Predictions (compat with recon) ---
print("\n--- Assembling, Sanitizing, and Exporting All Predictions ---")

# preds_valid holds [Jsc_hat, Voc_hat, FF_hat, Vmpp_hat] for the *valid_indices subset only*
preds_valid = np.empty((len(X), 4), dtype=np.float32)
# Re-order the OOF and Test predictions back to their original positions in the 'valid' dataframe (X)
preds_valid[X_train.index] = np.column_stack([jsc_oof_preds, voc_oof_preds, ff_oof_preds, vmpp_oof_preds])
preds_valid[X_test.index]  = np.column_stack([jsc_preds_test, voc_preds_test, ff_preds_test, vmpp_preds_test])

# Clamp per-physics
jsc_hat, voc_hat, ff_hat, vmpp_hat = preds_valid.T
jsc_hat  = np.clip(jsc_hat,  1e-9, None)
voc_hat  = np.clip(voc_hat,  1e-6, None)
ff_hat   = np.clip(ff_hat,   0.0,  1.0)
vmpp_hat = np.clip(vmpp_hat, 0.2 * voc_hat, 0.95 * voc_hat)

# Recompute Impp for convenience (not saved as a .txt in recon, but handy)
impp_hat = (jsc_hat * voc_hat * ff_hat) / np.maximum(vmpp_hat, 1e-9)

preds_valid = np.column_stack([jsc_hat, voc_hat, ff_hat, vmpp_hat]).astype(np.float32)
assert np.isfinite(preds_valid).all(), "FATAL: Non-finite scalar predictions after clamping."
assert np.isfinite(impp_hat).all(),    "FATAL: Non-finite Impp_hat after clamping."

# --- 10.1 Scatter into full-length arrays (N_raw), NaN elsewhere ---
N_raw = len(iv_data_raw)
Isc_all  = np.full(N_raw, np.nan, dtype=np.float32)
Voc_all  = np.full(N_raw, np.nan, dtype=np.float32)
FF_all   = np.full(N_raw, np.nan, dtype=np.float32)
Vmpp_all = np.full(N_raw, np.nan, dtype=np.float32)
Impp_all = np.full(N_raw, np.nan, dtype=np.float32)

valid_indices_arr = np.array(valid_indices, dtype=np.int64)
Isc_all[valid_indices_arr]  = preds_valid[:, 0]
Voc_all[valid_indices_arr]  = preds_valid[:, 1]
FF_all[valid_indices_arr]   = preds_valid[:, 2]
Vmpp_all[valid_indices_arr] = preds_valid[:, 3]
Impp_all[valid_indices_arr] = impp_hat.astype(np.float32)

# Final guardrails
for name, arr in [("Isc_all", Isc_all), ("Voc_all", Voc_all), ("FF_all", FF_all), ("Vmpp_all", Vmpp_all)]:
    # Allow NaNs for invalid rows; only check finiteness on the valid subset
    assert np.isfinite(arr[valid_indices_arr]).all(), f"FATAL: Non-finite values in {name} at valid indices."
print("  - All predicted scalars have been clamped and scattered into full-length arrays.")
print("  - NaN/inf check passed on valid subset.")

# --- 10.2 Write the 4 separate .txt files expected by the recon pipeline ---
out_dir = Path(DRIVE_PATH)
isc_txt  = out_dir / "factory_Isc.txt"
vmpp_txt = out_dir / "factory_Vmpp.txt"
voc_txt  = out_dir / "factory_Voc.txt"
ff_txt   = out_dir / "factory_FF.txt"

np.savetxt(isc_txt,  Isc_all,  fmt="%.8g")
np.savetxt(vmpp_txt, Vmpp_all, fmt="%.8g")
np.savetxt(voc_txt,  Voc_all,  fmt="%.8g")
np.savetxt(ff_txt,   FF_all,   fmt="%.8g")

print(f"  - Wrote scalar files:\n    {isc_txt}\n    {vmpp_txt}\n    {voc_txt}\n    {ff_txt}")

# --- 10.3 Save the compact NPZ (optional) + manifest update ---
output_path = out_dir / "scalar_predictions.npz"
np.savez(
    output_path,
    valid_indices=valid_indices_arr,
    train_indices=X_train.index.values,
    test_indices=X_test.index.values,
    preds_valid=preds_valid,   # only for valid subset (debugging)
    impp_hat_valid=impp_hat,   # only for valid subset (debugging)
    Isc_all=Isc_all, Voc_all=Voc_all, FF_all=FF_all, Vmpp_all=Vmpp_all, Impp_all=Impp_all
)
print(f"  - Unified predictions (plus full-length arrays) saved to: {output_path.name}")


# --- 11. Create Manifest File for Full Traceability ---
print("\n--- Creating Manifest File ---")
manifest = {
    "output_files": {
        "predictions_npz": str(output_path.name),
        "scaler": "standard_scaler.joblib",
        "jsc_model": "jsc_model.joblib",
        "ff_model": "ff_model.joblib",
        "vmpp_model": "vmpp_model.joblib",
        "voc_model": "voc_mlp_model.pth",
        "txt_scalars": {
            "isc":  isc_txt.name,
            "vmpp": vmpp_txt.name,
            "voc":  voc_txt.name,
            "ff":   ff_txt.name
        }
    },
    "mlp_config": MLP_CONFIG,
    "data_source": {"full_v_grid": full_v_grid.tolist()},
    "prediction_array_columns": ["Jsc_hat", "Voc_hat", "FF_hat", "Vmpp_hat"],
    "array_shapes": {
        "preds_valid": preds_valid.shape,
        "valid_indices": (len(valid_indices_arr),),
        "Isc_all": Isc_all.shape, "Voc_all": Voc_all.shape,
        "FF_all": FF_all.shape,   "Vmpp_all": Vmpp_all.shape, "Impp_all": Impp_all.shape
    },
    "clamping_rules": {"Jsc_hat": "clip(x, 1e-9, None)", "Voc_hat": "clip(x, 1e-6, None)",
        "FF_hat": "clip(x, 0.0, 1.0)", "Vmpp_hat": "clip(x, 0.2 * Voc_hat, 0.95 * Voc_hat)"},
    "library_versions": {p.project_name: p.version for p in pkg_resources.working_set if p.project_name in
        ['torch', 'pandas', 'scikit-learn', 'lightgbm', 'numpy', 'scipy']}
}

manifest_path = Path(DRIVE_PATH) / "manifest.json"
with open(manifest_path, 'w') as f: json.dump(manifest, f, indent=4)
print(f"  - Manifest saved to: {manifest_path.name}")


# --- 12. Final Performance Visualization ---
print("\n--- Generating Final Performance Plots (on Test Set) ---")
fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
fig.suptitle("Final Cascade Performance on Unseen Test Data", fontsize=24, weight='bold')
# Note: plot_data uses the original test-set predictions before scattering, which is correct for model validation.
plot_data = {"Jsc (LGBM)": (y_jsc_test, jsc_preds_test), "Voc (MLP)": (y_voc_test, voc_preds_test),
             "FF (LGBM)": (y_ff_test, ff_preds_test), "Vmpp (LGBM)": (y_vmpp_test, vmpp_preds_test)}
for i, (title, (actual, pred)) in enumerate(plot_data.items()):
    ax = axes.flatten()[i]; r2 = r2_score(actual, pred)
    ax.scatter(actual, pred, alpha=0.2, s=15, c='blue', edgecolors='k', linewidths=0.5)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r-', lw=2.5, label='Perfect Prediction')
    ax.set_title(f"{title} Prediction (R² = {r2:.4f})", fontsize=16, weight='bold')
    ax.set_xlabel("Actual Value", fontsize=12); ax.set_ylabel("Predicted Value", fontsize=12)
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n\n✅✅✅ MODEL FACTORY WORKFLOW COMPLETE! ✅✅✅")