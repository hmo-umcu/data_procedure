"""
model_gpr.py
=============
Gaussian Process Regression model for SF_mean (shape fidelity) prediction
from printing parameters (Pressure_kPa, NozzleSpeed_mms, Zoffset_mm).
Outputs saved to: results/GPR/

Model notes:
    - Matern kernel with ARD (one lengthscale per feature) + WhiteKernel
      for observation noise
    - ARD lengthscales give a direct feature-sensitivity ranking: a SHORT
      lengthscale means the GP needs to move only a little along that
      feature axis before its prediction changes a lot (high sensitivity);
      a LONG lengthscale means the feature barely matters
    - Native prediction uncertainty (predictive std) — useful directly for
      the downstream Bayesian Optimization acquisition function (e.g. EI,
      UCB), since GPR's predictive variance is exactly what BO needs
    - Train R² close to 1.0 is EXPECTED (GPs interpolate training points
      near-exactly by design) — this is not a sign of overfitting on its
      own; judge by validation R² only (see overfitting_check.png)

Hyperparameters (tunable via CLI):
    --nu              Matern smoothness parameter (default 1.5; 0.5/1.5/2.5
                       common choices, higher = smoother function assumed)
    --n_restarts      Number of optimizer restarts for kernel hyperparameters
                       (default 10; higher = more robust but slower)
    --alpha           Fixed noise added to the kernel diagonal for numerical
                       stability (default 1e-6; separate from the WhiteKernel,
                       which learns the actual observation noise level)

Usage:
    python model_gpr.py --data sample_sf_summary.csv
    python model_gpr.py --data sample_sf_summary.csv --nu 2.5 --n_restarts 20
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_utils import (load_and_preprocess, run_cv, aggregate_metrics,
                      print_summary, apply_plot_style, plot_predicted_vs_actual,
                      plot_residuals, plot_cv_metrics, plot_error_per_sample,
                      plot_overfitting, FEATURE_LABELS, FEATURES, PLOT_STYLE)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
apply_plot_style()

parser = argparse.ArgumentParser(description='Gaussian Process Regression for SF_mean prediction')
parser.add_argument('--data',        type=str,   default='sample_sf_summary.csv')
parser.add_argument('--outdir',      type=str,   default='results/GPR')
parser.add_argument('--n_folds',     type=int,   default=5)
parser.add_argument('--seed',        type=int,   default=42)
parser.add_argument('--title',       action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--nu',          type=float, default=1.5,
                    help='Matern kernel smoothness (default 1.5; try 0.5 or 2.5)')
parser.add_argument('--n_restarts',  type=int,   default=10,
                    help='Optimizer restarts for kernel hyperparameters (default 10)')
parser.add_argument('--alpha',       type=float, default=1e-6,
                    help='Fixed diagonal noise for numerical stability (default 1e-6)')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

print("=" * 60)
print("  Gaussian Process Regression — SF_mean Prediction")
print("=" * 60)
print(f"  Matern nu:       {args.nu}")
print(f"  n_restarts:      {args.n_restarts}")
print(f"  CV folds:        {args.n_folds}")
print(f"  Output:          {args.outdir}\n")

df, X, y, strat_label, feature_cols = load_and_preprocess(args.data)
n_features = X.shape[1]


def make_kernel():
    return (ConstantKernel(1.0, (1e-3, 1e3))
           * Matern(length_scale=np.ones(n_features),
                    length_scale_bounds=(1e-2, 1e2), nu=args.nu)
           + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1)))


model = GaussianProcessRegressor(
    kernel=make_kernel(), alpha=args.alpha,
    n_restarts_optimizer=args.n_restarts,
    normalize_y=False,   # we already standardise y ourselves in run_cv
    random_state=args.seed,
)

print(f"Running {args.n_folds}-fold stratified CV...\n")
val_rows, train_rows, fold_metrics = run_cv(
    model, X, y, strat_label, df, feature_cols,
    n_folds=args.n_folds, seed=args.seed
)

df_agg = aggregate_metrics(fold_metrics)
print_summary('GPR', df_agg)
print("\n[NOTE] Train R\u00b2 near 1.0 is expected for GPR (interpolates training "
      "points by design) — judge generalisation from validation R\u00b2 only.")

# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────
df_val_log   = pd.DataFrame(val_rows)
df_train_log = pd.DataFrame(train_rows)
df_full_log  = pd.concat([df_val_log, df_train_log], ignore_index=True)
df_full_log.sort_values(['split', 'fold', 'sample_id'], inplace=True)

log_path = os.path.join(args.outdir, 'prediction_log.csv')
df_full_log.to_csv(log_path, index=False, sep=',', decimal='.')
log_excel = os.path.join(args.outdir, 'prediction_log_excel.csv')
df_full_log.to_csv(log_excel, index=False, sep=';', decimal=',')
print(f"\n  Prediction log saved: {log_path}")

metrics_path = os.path.join(args.outdir, 'cv_metrics_summary.csv')
df_agg.to_csv(metrics_path, index=False)
folds_path = os.path.join(args.outdir, 'cv_metrics_per_fold.csv')
pd.DataFrame(fold_metrics).to_csv(folds_path, index=False)
print(f"  CV metrics saved:     {metrics_path}")
print(f"  Per-fold metrics:     {folds_path}")

# ─── STANDARD FIGURES ─────────────────────────────────────────────────────────
print("\nGenerating figures...")
COLOR = '#2E7D32'   # GPR color (green)

plot_predicted_vs_actual(val_rows, 'GPR', COLOR, args.outdir, args.title)
plot_residuals(val_rows, 'GPR', COLOR, args.outdir, args.title)
plot_cv_metrics(fold_metrics, 'GPR', COLOR, args.outdir, args.title)
plot_error_per_sample(val_rows, 'GPR', COLOR, args.outdir, args.title)
plot_overfitting(val_rows, train_rows, fold_metrics,
                 'GPR', COLOR, args.outdir, args.title)

# ─── GPR-SPECIFIC: ARD lengthscale plot (refitted on full data) ──────────────
print("  Generating ARD lengthscale plot...")
scaler_X_full = StandardScaler()
scaler_y_full = StandardScaler()
X_full_s = scaler_X_full.fit_transform(X)
y_full_s = scaler_y_full.fit_transform(y.reshape(-1, 1)).ravel()

gpr_full = GaussianProcessRegressor(
    kernel=make_kernel(), alpha=args.alpha,
    n_restarts_optimizer=args.n_restarts,
    normalize_y=False, random_state=args.seed,
)
gpr_full.fit(X_full_s, y_full_s)
print(f"  Fitted kernel (full-data refit): {gpr_full.kernel_}")

# Extract ARD lengthscales from the Matern part of the product kernel
# kernel_ structure: (Constant * Matern) + White
matern_k = gpr_full.kernel_.k1.k2
lengthscales = np.atleast_1d(matern_k.length_scale)
sensitivity = 1.0 / lengthscales   # shorter lengthscale = more sensitive

ps = PLOT_STYLE
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
if args.title:
    fig.suptitle('GPR — ARD Lengthscales  (refitted on full dataset)',
                 fontsize=ps['FONTSIZE_TITLE'] + 1, fontweight='bold', y=1.03)

feature_names = [FEATURE_LABELS[fc] for fc in feature_cols]

ax0 = axes[0]
ax0.barh(feature_names, lengthscales, color=COLOR, alpha=0.82,
         edgecolor='black', linewidth=1.2)
ax0.set_xlabel('Lengthscale  (standardised feature units)',
              fontsize=ps['FONTSIZE_LABEL'])
ax0.tick_params(labelsize=ps['FONTSIZE_TICK'])
ax0.grid(axis='x', alpha=0.3)
ax0.set_title('Lengthscale  (shorter = more sensitive)',
             fontsize=ps['FONTSIZE_LABEL'])

ax1 = axes[1]
order = np.argsort(sensitivity)
ax1.barh([feature_names[i] for i in order], [sensitivity[i] for i in order],
         color=COLOR, alpha=0.82, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Sensitivity  (1 / lengthscale)', fontsize=ps['FONTSIZE_LABEL'])
ax1.tick_params(labelsize=ps['FONTSIZE_TICK'])
ax1.grid(axis='x', alpha=0.3)
ax1.set_title('Feature Sensitivity Ranking', fontsize=ps['FONTSIZE_LABEL'])

plt.tight_layout()
ard_path = os.path.join(args.outdir, 'gpr_ard_lengthscales.png')
plt.savefig(ard_path, dpi=ps['DPI'], bbox_inches='tight')
plt.close()
print(f"  Saved: {ard_path}")

ard_df = pd.DataFrame({'feature': feature_cols, 'lengthscale': lengthscales,
                       'sensitivity_1_over_lengthscale': sensitivity})
ard_df.sort_values('sensitivity_1_over_lengthscale', ascending=False, inplace=True)
ard_df.to_csv(os.path.join(args.outdir, 'gpr_ard_lengthscales.csv'), index=False)

# ─── GPR-SPECIFIC: Prediction uncertainty plot (last CV fold) ────────────────
print("  Generating prediction uncertainty plot...")
cv_unc = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
train_idx_u, val_idx_u = list(cv_unc.split(X, strat_label))[-1]

scaler_X_u = StandardScaler()
X_tr_u  = scaler_X_u.fit_transform(X[train_idx_u])
X_val_u = scaler_X_u.transform(X[val_idx_u])
scaler_y_u = StandardScaler()
y_tr_u  = scaler_y_u.fit_transform(y[train_idx_u].reshape(-1, 1)).ravel()
y_true_u = y[val_idx_u]

gpr_u = GaussianProcessRegressor(
    kernel=make_kernel(), alpha=args.alpha,
    n_restarts_optimizer=args.n_restarts,
    normalize_y=False, random_state=args.seed,
)
gpr_u.fit(X_tr_u, y_tr_u)
y_pred_mean_s, y_pred_std_s = gpr_u.predict(X_val_u, return_std=True)
y_pred_mean = y_pred_mean_s * scaler_y_u.scale_[0] + scaler_y_u.mean_[0]
y_pred_std  = y_pred_std_s * scaler_y_u.scale_[0]

fig, ax = plt.subplots(figsize=(max(10, len(val_idx_u) * 0.9), 7))
if args.title:
    ax.set_title('GPR — Prediction Uncertainty on Last CV Fold\n'
                 '(error bars = predicted ±1 std)',
                 fontsize=ps['FONTSIZE_TITLE'], fontweight='bold', pad=14)

x_pos = np.arange(len(y_true_u))
ax.scatter(x_pos, y_true_u, color='black', zorder=5, s=110, label='Actual', marker='D')
ax.errorbar(x_pos, y_pred_mean, yerr=y_pred_std, fmt='o', color=COLOR,
           alpha=0.82, capsize=5, capthick=2, elinewidth=1.8,
           markersize=9, label='Predicted (±1 std)')

ax.set_xticks(x_pos)
ax.set_xticklabels(df['Sample_ID'].iloc[val_idx_u].values, fontsize=ps['FONTSIZE_TICK'] - 2)
ax.set_xlabel('Sample ID (last validation fold)', fontsize=ps['FONTSIZE_LABEL'])
ax.set_ylabel('SF_mean', fontsize=ps['FONTSIZE_LABEL'])
ax.tick_params(labelsize=ps['FONTSIZE_TICK'])
ax.grid(alpha=0.25)
ax.legend(fontsize=ps['FONTSIZE_LEGEND'])

plt.tight_layout()
unc_path = os.path.join(args.outdir, 'gpr_uncertainty.png')
plt.savefig(unc_path, dpi=ps['DPI'], bbox_inches='tight')
plt.close()
print(f"  Saved: {unc_path}")

print(f"\n{'='*60}")
print(f"  GPR complete. All results in: {args.outdir}/")
print(f"{'='*60}")
