"""
model_ngboost.py
==================
NGBoost (Natural Gradient Boosting) model for SF_mean (shape fidelity)
prediction from printing parameters (Pressure_kPa, NozzleSpeed_mms,
Zoffset_mm).
Outputs saved to: results/NGBoost/

Model notes:
    - Gradient boosting that predicts a full probability distribution
      (mean + std), not just a point estimate
    - Key regularisation levers at small N (~30 here):
        n_estimators : number of boosting rounds (keep low, N<50)
        learning_rate: step size per round (lower = more conservative)
        minibatch_frac: fraction of samples per round (adds stochasticity)
        col_sample   : fraction of features per round (adds diversity)
    - If train R\u00b2 \u2248 1.0 but val R\u00b2 << 1.0 -> reduce n_estimators further

Hyperparameters (tunable via CLI):
    --n_estimators    int     default 30   (lower than the 4-output
                                            original since N is smaller here)
    --learning_rate   float   default 0.1
    --minibatch_frac  float   default 0.6  (0 < x <= 1.0)
    --col_sample      float   default 0.8  (0 < x <= 1.0; only 3 features
                                            here so don't over-subsample)

Usage:
    python model_ngboost.py --data sample_sf_summary.csv
    python model_ngboost.py --data sample_sf_summary.csv --n_estimators 20
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
                      plot_overfitting, PLOT_STYLE)

from ngboost import NGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error as mse_fn

warnings.filterwarnings('ignore')
apply_plot_style()

parser = argparse.ArgumentParser(description='NGBoost for SF_mean prediction')
parser.add_argument('--data',           type=str,   default='sample_sf_summary.csv')
parser.add_argument('--outdir',         type=str,   default='results/NGBoost')
parser.add_argument('--n_folds',        type=int,   default=5)
parser.add_argument('--seed',           type=int,   default=42)
parser.add_argument('--title',          action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--n_estimators',   type=int,   default=30,
                    help='Number of boosting rounds (default 30; lower = less overfitting)')
parser.add_argument('--learning_rate',  type=float, default=0.1)
parser.add_argument('--minibatch_frac', type=float, default=0.6)
parser.add_argument('--col_sample',     type=float, default=0.8)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

print("=" * 60)
print("  NGBoost — SF_mean Prediction")
print("=" * 60)
print(f"  n_estimators:   {args.n_estimators}")
print(f"  learning_rate:  {args.learning_rate}")
print(f"  minibatch_frac: {args.minibatch_frac}")
print(f"  col_sample:     {args.col_sample}")
print(f"  CV folds:       {args.n_folds}")
print(f"  Output:         {args.outdir}\n")

df, X, y, strat_label, feature_cols = load_and_preprocess(args.data)


def make_model():
    return NGBRegressor(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        minibatch_frac=args.minibatch_frac, col_sample=args.col_sample,
        random_state=args.seed, verbose=False,
    )


model = make_model()

print(f"Running {args.n_folds}-fold stratified CV...\n")
val_rows, train_rows, fold_metrics = run_cv(
    model, X, y, strat_label, df, feature_cols,
    n_folds=args.n_folds, seed=args.seed
)

df_agg = aggregate_metrics(fold_metrics)
print_summary('NGBoost', df_agg)

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

# ─── STANDARD FIGURES ─────────────────────────────────────────────────────────
print("\nGenerating figures...")
COLOR = '#C62828'

plot_predicted_vs_actual(val_rows, 'NGBoost', COLOR, args.outdir, args.title)
plot_residuals(val_rows, 'NGBoost', COLOR, args.outdir, args.title)
plot_cv_metrics(fold_metrics, 'NGBoost', COLOR, args.outdir, args.title)
plot_error_per_sample(val_rows, 'NGBoost', COLOR, args.outdir, args.title)
plot_overfitting(val_rows, train_rows, fold_metrics,
                 'NGBoost', COLOR, args.outdir, args.title)

# ─── NGBOOST-SPECIFIC: Staged score plot (val loss vs n_estimators) ──────────
print("  Generating staged score plot...")

cv_staged = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
train_idx_s, val_idx_s = list(cv_staged.split(X, strat_label))[-1]

scaler_X_s = StandardScaler()
X_tr_s  = scaler_X_s.fit_transform(X[train_idx_s])
X_val_s = scaler_X_s.transform(X[val_idx_s])
scaler_y_s = StandardScaler()
y_tr_s  = scaler_y_s.fit_transform(y[train_idx_s].reshape(-1, 1)).ravel()
y_val_true = y[val_idx_s]
y_scale = scaler_y_s.scale_[0]
y_mean  = scaler_y_s.mean_[0]

ngb_single = make_model()
ngb_single.fit(X_tr_s, y_tr_s)

train_rmse_list, val_rmse_list = [], []
for y_tr_pred_s_staged, y_val_pred_s_staged in zip(
        ngb_single.staged_predict(X_tr_s), ngb_single.staged_predict(X_val_s)):
    y_tr_pred  = y_tr_pred_s_staged  * y_scale + y_mean
    y_val_pred = y_val_pred_s_staged * y_scale + y_mean
    y_tr_true  = y[train_idx_s]
    train_rmse_list.append(np.sqrt(mse_fn(y_tr_true,  y_tr_pred)))
    val_rmse_list.append(  np.sqrt(mse_fn(y_val_true, y_val_pred)))

n_iters = np.arange(1, len(train_rmse_list) + 1)
ps = PLOT_STYLE
fig, ax = plt.subplots(figsize=(10, 7))
if args.title:
    ax.set_title('NGBoost — RMSE vs Number of Estimators  (last CV fold)',
                 fontsize=ps['FONTSIZE_TITLE'], fontweight='bold', pad=14)

ax.plot(n_iters, train_rmse_list, color='#455A64', linewidth=2.2, label='Train')
ax.plot(n_iters, val_rmse_list,   color=COLOR,      linewidth=2.2, label='Validation')

best_iter = np.argmin(val_rmse_list) + 1
best_rmse = min(val_rmse_list)
ax.axvline(best_iter, color='darkorange', linewidth=2.0, linestyle='--',
          label=f'Best val iter={best_iter}')
ax.text(0.97, 0.97, f'Best iter={best_iter}\nVal RMSE={best_rmse:.4f}',
       transform=ax.transAxes, fontsize=ps['FONTSIZE_ANNOTATION'],
       va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                       alpha=0.85, ec='gray'))

ax.set_xlabel('Number of estimators', fontsize=ps['FONTSIZE_LABEL'])
ax.set_ylabel('RMSE', fontsize=ps['FONTSIZE_LABEL'])
ax.tick_params(labelsize=ps['FONTSIZE_TICK'])
ax.grid(alpha=0.25)
ax.legend(fontsize=ps['FONTSIZE_LEGEND'] - 1)

if best_iter < args.n_estimators * 0.6:
    print(f"  [NOTE] Best validation iteration ({best_iter}) is well below "
          f"n_estimators ({args.n_estimators}) — consider lowering "
          f"--n_estimators to around {best_iter} to reduce overfitting risk.")

plt.tight_layout()
staged_path = os.path.join(args.outdir, 'ngboost_staged_score.png')
plt.savefig(staged_path, dpi=ps['DPI'], bbox_inches='tight')
plt.close()
print(f"  Saved: {staged_path}")

# ─── NGBOOST-SPECIFIC: Prediction uncertainty plot ────────────────────────────
print("  Generating prediction uncertainty plot...")

cv_unc = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
train_idx_u, val_idx_u = list(cv_unc.split(X, strat_label))[-1]

scaler_X_u = StandardScaler()
X_tr_u  = scaler_X_u.fit_transform(X[train_idx_u])
X_val_u = scaler_X_u.transform(X[val_idx_u])
scaler_y_u = StandardScaler()
y_tr_u  = scaler_y_u.fit_transform(y[train_idx_u].reshape(-1, 1)).ravel()
y_true_u = y[val_idx_u]

ngb_u = make_model()
ngb_u.fit(X_tr_u, y_tr_u)
dist = ngb_u.pred_dist(X_val_u)
y_pred_mean = dist.loc   * scaler_y_u.scale_[0] + scaler_y_u.mean_[0]
y_pred_std  = dist.scale * scaler_y_u.scale_[0]

fig, ax = plt.subplots(figsize=(max(10, len(val_idx_u) * 0.9), 7))
if args.title:
    ax.set_title('NGBoost — Prediction Uncertainty on Last CV Fold\n'
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
unc_path = os.path.join(args.outdir, 'ngboost_uncertainty.png')
plt.savefig(unc_path, dpi=ps['DPI'], bbox_inches='tight')
plt.close()
print(f"  Saved: {unc_path}")

print(f"\n{'='*60}")
print(f"  NGBoost complete. All results in: {args.outdir}/")
print(f"{'='*60}")
