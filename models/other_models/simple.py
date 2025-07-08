# -*- coding: utf-8 -*-
"""
The simplest implementation, it significantly underfits, and adheres to the most common mode, 
which is long concave down curves with large fill factors. Also linear curves seem to be fine. 
Scores for those of MAE ~ 3, RMSE ~ 8, R² ~ 0.997, R^2 per‐curve ~ 0.992.
Negative R^2 for the non-typical curves, usually ones with low FF.
Most efficient model, mostly used for experiment with loss functions and other functionaliy right now.


Written in PyTorch. I will eventually make everything into PyTorch because of Lightning :/, but I am dreading
that day. At least it means its production-level if we get to that point

This version is tuned for better GPU utilization and
includes comprehensive evaluation in both normalized and physical units.

Key Features:
1.  **Dataset & DataLoader:** Wraps NumPy arrays in a custom `torch.utils.data.Dataset`
    for efficient batching, shuffling, and data loading. It now includes original
    sample indices to prevent lookup errors.
2.  **PyTorch `nn.Module`:** The core `Conv1D` architecture is implemented as a
    subclass of `torch.nn.Module`, providing a clear and modular structure.
3.  **Inline Physics-Informed Loss:** The custom loss function (MSE + monotonicity +
    curvature penalties) is calculated directly within the training loop.
4.  **Optimized Training Loop:** The training loop is configured for high performance on
    a dedicated GPU (like a Colab T4) by using a larger batch size and enabling
    asynchronous data loading with multiple workers.
5.  **Comprehensive Per-Curve & Global Metrics:** Includes robust evaluation over the
    entire test set, reporting RMSE, MAE, and R² for each curve.
6.  **Advanced Multi-Panel Plotting:** Generates detailed plots including I-V, P-V,
    and residual analysis for the best, worst, and random samples.
7.  **Aggregate Performance Plots:** Includes histograms and scatter plots to
    visualize overall model performance distribution (e.g., R² histogram).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# User Configuration
# -----------------------------------------------------------------------------
# --- File Paths ---
PARAMS_CSV = '/content/drive/MyDrive/Colab Notebooks/Data_100k/LHS_parameters_m.txt'
IV_RAW_CSV = '/content/drive/MyDrive/Colab Notebooks/Data_100k/iV_m.txt'

# --- NEW: Data Filtering Configuration ---
FF_PERCENTILE_THRESHOLD = 0.05 # Filter out curves in the bottom 5% of Fill Factor

# --- Model & Sampling ---
MODEL_TYPE = 'conv1d' # 'mlp' or 'conv1d'
NUM_KNOTS  = 9
KNOT_ALPHA = 0.5   # Power-law for knot spacing (<1 pushes points towards MPP)

# --- Physics-Informed Loss Hyperparameters ---
LAMBDA_MONO = 0.05   # Weight for the monotonicity penalty
LAMBDA_CURV = 0.1    # Weight for the curvature penalty

# --- Machine Learning & Validation (Tuned for GPU) ---
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
LEARNING_RATE    = 0.001
EPOCHS           = 50
BATCH_SIZE       = 256  # Increased for better GPU utilization
PATIENCE         = 25
NUM_WORKERS      = 2    # For asynchronous data loading

# --- Plotting & Analysis ---
NUM_PLOT_SAMPLES = 4  # Number of best/worst/random samples to plot
INTERP_GRID_SIZE = 500

# -----------------------------------------------------------------------------
# NEW & MODIFIED Data Preprocessing Functions
# -----------------------------------------------------------------------------
def analyze_and_filter_by_ff(iv_raw_all, voltage_raw, ff_percentile_threshold):
    """
    Analyzes all raw I-V curves to calculate their physical properties and filters
    them based on a Fill Factor (FF) percentile.

    Args:
        iv_raw_all (np.ndarray): The raw I-V data for all curves.
        voltage_raw (np.ndarray): The corresponding voltage sweep.
        ff_percentile_threshold (float): The percentile to use for filtering (e.g., 0.05 for 5%).

    Returns:
        pd.DataFrame: A DataFrame with properties of the curves that passed the filter.
    """
    print("\n--- Pre-analysis: Calculating Fill Factors for all curves ---")
    all_curve_properties = []

    for i in range(iv_raw_all.shape[0]):
        if (i + 1) % 5000 == 0:
            print(f"\rAnalyzing curve {i+1}/{iv_raw_all.shape[0]}...", end="")

        jsc = iv_raw_all[i, 0]
        if jsc <= 1e-6: continue

        interp_func = PchipInterpolator(voltage_raw, iv_raw_all[i])
        v_fine = np.linspace(voltage_raw[0], voltage_raw[-1], 2000)
        j_fine = interp_func(v_fine)

        neg_idx = np.where(j_fine <= 0)[0]
        voc = v_fine[neg_idx[0]] if len(neg_idx) > 0 else v_fine[-1]
        if voc <= 1e-6: continue

        power = v_fine * j_fine
        valid_mask = (v_fine <= voc) & (j_fine >= 0)
        if not np.any(valid_mask): continue

        mpp_idx = np.argmax(power[valid_mask])
        v_mpp = v_fine[valid_mask][mpp_idx]
        j_mpp = j_fine[valid_mask][mpp_idx]

        fill_factor = (v_mpp * j_mpp) / (voc * jsc) if (voc * jsc) > 0 else 0

        all_curve_properties.append({
            'original_index': i, 'jsc': jsc, 'voc': voc,
            'v_mpp': v_mpp, 'fill_factor': fill_factor
        })
    print(f"\nAnalyzed {len(all_curve_properties)} initially valid curves.")

    properties_df = pd.DataFrame(all_curve_properties)

    print("\n--- Fill Factor Distribution (Pre-filtering) ---")
    print(properties_df['fill_factor'].describe(percentiles=[.01, .05, .25, .5, .75, .95]).round(4))

    ff_threshold = properties_df['fill_factor'].quantile(ff_percentile_threshold)
    print(f"\nFiltering out curves with Fill Factor < {ff_percentile_threshold*100:.1f}th percentile ({ff_threshold:.4f})...")

    initial_count = len(properties_df)
    filtered_df = properties_df[properties_df['fill_factor'] >= ff_threshold].copy()
    final_count = len(filtered_df)

    print(f"Dropped {initial_count - final_count} curves. Kept {final_count} high-quality curves.")
    return filtered_df

def build_normalized_curve_dataset(params_all, iv_raw_all, voltage_raw, filtered_properties_df):
    """
    MODIFIED: Builds the ML-ready dataset (X, y) using the pre-filtered curves.
    This function is now more efficient as it uses pre-calculated Jsc, Voc, etc.
    """
    num_curves = len(filtered_properties_df)
    num_features = params_all.shape[1] + 2 + NUM_KNOTS
    X = np.zeros((num_curves, num_features))
    y = np.zeros((num_curves, NUM_KNOTS))

    # We create interpolators for only the curves we need, which is more memory-efficient
    interpolators = {int(idx): PchipInterpolator(voltage_raw, iv_raw_all[int(idx)])
                     for idx in filtered_properties_df['original_index']}

    for i, (_, row) in enumerate(filtered_properties_df.iterrows()):
        if (i+1) % 5000 == 0:
            print(f"\rBuilding feature vectors for curve {i+1}/{num_curves}...", end="")

        original_idx = int(row['original_index'])
        jsc, voc, v_mpp = row['jsc'], row['voc'], row['v_mpp']

        u = np.linspace(0, 1, NUM_KNOTS)
        v_sel = np.piecewise(u, [u < 0.5, u >= 0.5], [
            lambda u_val: v_mpp * (2*u_val)**KNOT_ALPHA,
            lambda u_val: v_mpp + (voc - v_mpp) * (2*u_val-1)**(1/KNOT_ALPHA)
        ])
        v_sel = np.unique(v_sel)
        if len(v_sel) < NUM_KNOTS:
             v_sel = np.pad(v_sel, (0, NUM_KNOTS - len(v_sel)), 'edge')

        j_sel = interpolators[original_idx](v_sel)
        v_sel_norm, j_sel_norm = v_sel / voc, j_sel / jsc

        X[i, :] = np.concatenate([params_all[original_idx], [jsc, voc], v_sel_norm])
        y[i, :] = j_sel_norm

    original_indices = filtered_properties_df['original_index'].values
    print("\nFinished building feature vectors.")
    return X, y, original_indices

# -----------------------------------------------------------------------------
# PyTorch Dataset and Model Definitions (Unchanged)
# -----------------------------------------------------------------------------
class IVCurveDataset(Dataset):
    # ... (No changes here)
    def __init__(self, X, y, original_indices):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.original_indices = torch.from_numpy(original_indices).long()
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.original_indices[idx]

class ConvKnotNet(nn.Module):
    # ... (No changes here)
    def __init__(self, input_dim, params_dim, num_knots):
        super().__init__()
        self.params_dim = params_dim
        self.num_knots = num_knots
        self.fc_p1 = nn.Linear(params_dim, 128)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.fc_k1 = nn.Linear(8 * num_knots, 128)
        self.fc_f1 = nn.Linear(128 + 128, 256)
        self.fc_f2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, num_knots)
    def forward(self, x):
        params_vec = x[:, :self.params_dim]
        v_knots = x[:, -self.num_knots:]
        h_p = F.relu(self.fc_p1(params_vec))
        h_k = v_knots.unsqueeze(1)
        h_k = F.relu(self.conv1(h_k))
        h_k = F.relu(self.conv2(h_k))
        h_k = h_k.view(h_k.size(0), -1)
        h_k = F.relu(self.fc_k1(h_k))
        h = torch.cat([h_p, h_k], dim=1)
        h = F.relu(self.fc_f1(h))
        h = F.relu(self.fc_f2(h))
        return self.fc_out(h)

def physics_informed_loss(y_pred, y_true, lambda_mono, lambda_curv):
    # ... (No changes here)
    mse_loss = F.mse_loss(y_pred, y_true, reduction='mean')
    diffs = y_pred[:, 1:] - y_pred[:, :-1]
    mono_loss = torch.mean(torch.relu(diffs)**2)
    curvature = y_pred[:, 2:] - 2 * y_pred[:, 1:-1] + y_pred[:, :-2]
    curv_loss = torch.mean(torch.relu(curvature)**2)
    return mse_loss + lambda_mono * mono_loss + lambda_curv * curv_loss

# -----------------------------------------------------------------------------
# Plotting and Analysis Functions (Unchanged)
# -----------------------------------------------------------------------------
def plot_reconstruction_details(results_df, plot_idx, voltage_raw, iv_raw):
    # ... (No changes here)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Detailed Reconstruction for Test Sample (Original Index: {results_df.loc[plot_idx, 'original_index']})", fontsize=16)
    jsc = results_df.loc[plot_idx, 'jsc']
    voc = results_df.loc[plot_idx, 'voc']
    v_grid_phys = np.linspace(0, voc, INTERP_GRID_SIZE)
    v_knots_norm = results_df.loc[plot_idx, 'v_knots_norm']
    j_knots_pred_norm = results_df.loc[plot_idx, 'j_pred_norm']
    f_pred = PchipInterpolator(v_knots_norm * voc, j_knots_pred_norm * jsc, extrapolate=False)
    j_pred_phys = f_pred(v_grid_phys)
    f_true = PchipInterpolator(voltage_raw, iv_raw[results_df.loc[plot_idx, 'original_index']], extrapolate=False)
    j_true_phys = f_true(v_grid_phys)
    valid_mask = ~np.isnan(j_pred_phys) & ~np.isnan(j_true_phys)
    v_grid_phys, j_pred_phys, j_true_phys = v_grid_phys[valid_mask], j_pred_phys[valid_mask], j_true_phys[valid_mask]
    ax = axes[0, 0]
    ax.plot(v_grid_phys, j_true_phys, 'b-', label='Ground Truth', lw=2)
    ax.plot(v_grid_phys, j_pred_phys, 'r--', label='Reconstruction', lw=2)
    ax.plot(v_knots_norm * voc, j_knots_pred_norm * jsc, 'ko', label='Predicted Knots', ms=6, zorder=10)
    ax.set_xlabel("Voltage (V)"), ax.set_ylabel("Current Density (mA/cm²)"), ax.set_title(f"I-V Curve (Physical Units) | $R^2$ = {results_df.loc[plot_idx, 'phys_r2']:.4f}"), ax.legend(loc='upper right'), ax.set_ylim(bottom=0), ax.set_xlim(left=0)
    ax = axes[0, 1]
    power_true, power_pred = v_grid_phys * j_true_phys, v_grid_phys * j_pred_phys
    mpp_true_idx, mpp_pred_idx = np.argmax(power_true), np.argmax(power_pred)
    ax.plot(v_grid_phys, power_true, 'b-', label=f'True Power (MPP: {power_true[mpp_true_idx]:.2f} mW)', lw=2)
    ax.plot(v_grid_phys, power_pred, 'r--', label=f'Predicted Power (MPP: {power_pred[mpp_pred_idx]:.2f} mW)', lw=2)
    ax.axvline(v_grid_phys[mpp_true_idx], color='b', ls=':', ymax=0.95, label=f'$V_{{mpp-true}}$={v_grid_phys[mpp_true_idx]:.2f}V')
    ax.axvline(v_grid_phys[mpp_pred_idx], color='r', ls=':', ymax=0.95, label=f'$V_{{mpp-pred}}$={v_grid_phys[mpp_pred_idx]:.2f}V')
    ax.set_xlabel("Voltage (V)"), ax.set_ylabel("Power Density (mW/cm²)"), ax.set_title("P-V Curve Comparison"), ax.legend(loc='upper left', fontsize='small'), ax.set_ylim(bottom=0), ax.set_xlim(left=0)
    ax = axes[1, 0]
    residuals = j_true_phys - j_pred_phys
    ax.plot(v_grid_phys, residuals, 'g-'), ax.axhline(0, color='k', linestyle='--', lw=1), ax.set_xlabel("Voltage (V)"), ax.set_ylabel("Current Error (mA/cm²)"), ax.set_title("Reconstruction Error")
    ax = axes[1, 1]
    ax.axis('off')
    metrics_text = [["Metric", "Normalized", "Physical"], ["-"*10, "-"*10, "-"*10], ["RMSE", f"{results_df.loc[plot_idx, 'norm_rmse']:.4f}", f"{results_df.loc[plot_idx, 'phys_rmse']:.4f}"], ["MAE", f"{results_df.loc[plot_idx, 'norm_mae']:.4f}", f"{results_df.loc[plot_idx, 'phys_mae']:.4f}"], ["$R^2$", f"{results_df.loc[plot_idx, 'norm_r2']:.4f}", f"{results_df.loc[plot_idx, 'phys_r2']:.4f}"]]
    table = ax.table(cellText=metrics_text, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False), table.set_fontsize(12), table.scale(1.1, 1.4), ax.set_title("Per-Curve Statistics", y=0.8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_aggregate_performance(results_df):
    # ... (No changes here)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(results_df['phys_r2'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribution of $R^2$ Scores (Physical Units)'), axes[0].set_xlabel('$R^2$ Score'), axes[0].set_ylabel('Frequency')
    mean_r2 = results_df['phys_r2'].mean()
    axes[0].axvline(mean_r2, color='r', linestyle='--', label=f'Mean $R^2$: {mean_r2:.3f}'), axes[0].legend()
    p_mpp_true, p_mpp_pred = results_df['p_mpp_true'], results_df['p_mpp_pred']
    axes[1].scatter(p_mpp_true, p_mpp_pred, alpha=0.5, s=10, c='coral', edgecolors='k', lw=0.5)
    axes[1].set_title('Maximum Power Point Correlation'), axes[1].set_xlabel('True $P_{mpp}$ (mW/cm²)'), axes[1].set_ylabel('Predicted $P_{mpp}$ (mW/cm²)')
    lims = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]), max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
    axes[1].plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y = x'), axes[1].set_xlim(lims), axes[1].set_ylim(lims)
    axes[1].set_aspect('equal', adjustable='box'), axes[1].legend()
    fig.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------
def main():
    """Main function to run the entire PyTorch workflow."""
    print("--- IV Curve Reconstruction | PyTorch Implementation (Optimized) ---")

    if not (os.path.exists(PARAMS_CSV) and os.path.exists(IV_RAW_CSV)):
        # ... (Dummy file creation logic remains the same)
        print("ERROR: Input data files not found!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading and Preparation ---
    print("\n[Phase 1/5] Loading and preprocessing data...")
    params = pd.read_csv(PARAMS_CSV, header=None).values
    iv_raw = pd.read_csv(IV_RAW_CSV, header=None).values
    voltage_raw = np.concatenate([np.arange(0, 0.4+1e-8, 0.1), np.arange(0.425, 1.4+1e-8, 0.025)])

    # --- MODIFIED WORKFLOW: Pre-filter data before building dataset ---
    # 1. Analyze all curves and get a list of high-quality ones
    filtered_properties_df = analyze_and_filter_by_ff(iv_raw, voltage_raw, FF_PERCENTILE_THRESHOLD)

    # 2. Build the final ML dataset using only the filtered curves
    X_valid, y_valid, original_indices_valid = build_normalized_curve_dataset(
        params, iv_raw, voltage_raw, filtered_properties_df
    )
    # --- END OF MODIFIED WORKFLOW ---

    # --- Train/Test Split and DataLoader creation ---
    print("\n[Phase 2/5] Creating data loaders...")
    # ... The rest of the script continues as before, now operating on cleaner data ...
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_valid, y_valid, original_indices_valid, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

    train_ds = IVCurveDataset(X_train_scaled, y_train, indices_train)
    test_ds = IVCurveDataset(X_test_scaled, y_test, indices_test)

    loader_args = {'num_workers': NUM_WORKERS, 'pin_memory': True} if device.type == 'cuda' else {}
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_args)

    # --- Model, Optimizer, and Training Loop ---
    print("\n[Phase 3/5] Building and training the PyTorch model...")
    model = ConvKnotNet(X_train_scaled.shape[1], params.shape[1] + 2, NUM_KNOTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss, patience_counter = float('inf'), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = physics_informed_loss(y_pred, y_batch, LAMBDA_MONO, LAMBDA_CURV)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss = physics_informed_loss(y_pred, y_batch, LAMBDA_MONO, LAMBDA_CURV)
                val_losses.append(val_loss.item())

        avg_train_loss, avg_val_loss = np.mean(train_losses), np.mean(val_losses)
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Stopping early at epoch {epoch}.")
                break

    # --- Final Evaluation on Test Set ---
    # ... (This entire block is unchanged) ...
    print("\n[Phase 4/5] Evaluating final model on the entire test set...")
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    per_curve_results = []
    v_grid_norm_eval = np.linspace(0, 1, INTERP_GRID_SIZE)
    with torch.no_grad():
        for X_batch_scaled, y_batch_true_norm, idx_batch in val_loader:
            y_batch_pred_norm = model(X_batch_scaled.to(device)).cpu().numpy()
            X_batch_unscaled = scaler.inverse_transform(X_batch_scaled.cpu().numpy())
            for j in range(y_batch_pred_norm.shape[0]):
                original_idx = idx_batch[j].item()
                jsc = X_batch_unscaled[j, params.shape[1]]
                voc = X_batch_unscaled[j, params.shape[1] + 1]
                v_knots_norm = X_batch_unscaled[j, -NUM_KNOTS:]
                j_pred_norm_knots = y_batch_pred_norm[j]
                f_pred_norm = PchipInterpolator(v_knots_norm, j_pred_norm_knots, extrapolate=False)
                j_rec_norm = f_pred_norm(v_grid_norm_eval)
                true_curve_phys = iv_raw[original_idx]
                f_true_norm = PchipInterpolator(voltage_raw / voc, true_curve_phys / jsc, extrapolate=False)
                j_true_norm_grid = f_true_norm(v_grid_norm_eval)
                mask = ~np.isnan(j_rec_norm) & ~np.isnan(j_true_norm_grid)
                if np.sum(mask) < 2: continue
                norm_rmse = np.sqrt(mean_squared_error(j_true_norm_grid[mask], j_rec_norm[mask]))
                norm_mae = mean_absolute_error(j_true_norm_grid[mask], j_rec_norm[mask])
                norm_r2 = r2_score(j_true_norm_grid[mask], j_rec_norm[mask])
                j_rec_phys = j_rec_norm[mask] * jsc
                j_true_phys_grid = j_true_norm_grid[mask] * jsc
                phys_rmse = np.sqrt(mean_squared_error(j_true_phys_grid, j_rec_phys))
                phys_mae = mean_absolute_error(j_true_phys_grid, j_rec_phys)
                phys_r2 = r2_score(j_true_phys_grid, j_rec_phys)
                v_grid_phys = v_grid_norm_eval[mask] * voc
                p_mpp_true = np.max(v_grid_phys * j_true_phys_grid)
                p_mpp_pred = np.max(v_grid_phys * j_rec_phys)
                per_curve_results.append({'original_index': original_idx, 'jsc': jsc, 'voc': voc, 'v_knots_norm': v_knots_norm, 'j_pred_norm': j_pred_norm_knots, 'norm_rmse': norm_rmse, 'norm_mae': norm_mae, 'norm_r2': norm_r2, 'phys_rmse': phys_rmse, 'phys_mae': phys_mae, 'phys_r2': phys_r2, 'p_mpp_true': p_mpp_true, 'p_mpp_pred': p_mpp_pred})
    results_df = pd.DataFrame(per_curve_results).set_index(pd.Index(range(len(per_curve_results))))

    # --- Print Global and Per-Curve Statistics ---
    def print_stats_summary(df):
        print("\n" + "="*50), print(">>> GLOBAL TEST SET STATS (PHYSICAL UNITS) <<<"), print("="*50)
        print(df[['phys_rmse', 'phys_mae', 'phys_r2']].describe(percentiles=[.25, .5, .75]).round(4))
        print("\n" + "="*50), print(">>> PER-CURVE PERFORMANCE HIGHLIGHTS <<<"), print("="*50)
        print("\n--- Top 5 Best Performing Curves (by Physical R²) ---")
        print(df.sort_values('phys_r2', ascending=False).head(5)[['original_index', 'phys_r2', 'phys_rmse', 'phys_mae']].round(4))
        print("\n--- Top 5 Worst Performing Curves (by Physical R²) ---")
        print(df.sort_values('phys_r2', ascending=True).head(5)[['original_index', 'phys_r2', 'phys_rmse', 'phys_mae']].round(4))
        print("\n" + "="*50)
    print_stats_summary(results_df)

    # --- Plotting Phase ---
    # ... (This entire block is unchanged) ...
    print("\n[Phase 5/5] Plotting analysis and individual curve examples...")
    plot_aggregate_performance(results_df)
    best_indices = results_df.sort_values('phys_r2', ascending=False).head(NUM_PLOT_SAMPLES).index
    worst_indices = results_df.sort_values('phys_r2', ascending=True).head(NUM_PLOT_SAMPLES).index
    remaining_indices = results_df.index.difference(best_indices).difference(worst_indices)
    random_indices = np.random.choice(remaining_indices, size=min(NUM_PLOT_SAMPLES, len(remaining_indices)), replace=False)
    print(f"\nPlotting {len(best_indices)} best-fit curves...")
    for idx in best_indices: plot_reconstruction_details(results_df, idx, voltage_raw, iv_raw)
    print(f"\nPlotting {len(worst_indices)} worst-fit curves...")
    for idx in worst_indices: plot_reconstruction_details(results_df, idx, voltage_raw, iv_raw)
    print(f"\nPlotting {len(random_indices)} random curves...")
    for idx in random_indices: plot_reconstruction_details(results_df, idx, voltage_raw, iv_raw)


if __name__ == '__main__':
    main()
