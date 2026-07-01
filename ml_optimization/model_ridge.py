"""
model_ridge.py
===============
Ridge Regression model for SF_mean (shape fidelity) prediction from
printing parameters (Pressure_kPa, NozzleSpeed_mms, Zoffset_mm).
Outputs saved to: results/Ridge/

Hyperparameters:
    alpha: regularisation strength — selected automatically via RidgeCV
           using internal leave-one-out CV on each training fold
           Candidates: [0.01, 0.1, 1, 10, 100, 1000]

Usage:
    python model_ridge.py --data sample_sf_summary.csv
    python model_ridge.py --data sample_sf_summary.csv --n_folds 4 --no-title
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
                      plot_overfitting, FEATURE_LABELS, PLOT_STYLE)

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
apply_plot_style()

parser = argparse.ArgumentParser(description='Ridge Regression for SF_mean prediction')
parser.add_argument('--data',    type=str, default='sample_sf_summary.csv')
parser.add_argument('--outdir',  type=str, default='results/Ridge')
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--seed',    type=int, default=42)
parser.add_argument('--title',   action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--alphas',  type=float, nargs='+',
                    default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                    help='Ridge alpha candidates for RidgeCV (space-separated)')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

print("=" * 60)
print("  Ridge Regression — SF_mean Prediction")
print("=" * 60)
print(f"  Alpha candidates: {args.alphas}")
print(f"  CV folds:         {args.n_folds}")
print(f"  Output directory: {args.outdir}\n")

df, X, y, strat_label, feature_cols = load_and_preprocess(args.data)

model = RidgeCV(alphas=args.alphas, fit_intercept=True)

print(f"Running {args.n_folds}-fold stratified CV...\n")
val_rows, train_rows, fold_metrics = run_cv(
    model, X, y, strat_label, df, feature_cols,
    n_folds=args.n_folds, seed=args.seed
)

df_agg = aggregate_metrics(fold_metrics)
print_summary('Ridge', df_agg)

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

# ─── FIGURES ──────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
COLOR = '#1565C0'

plot_predicted_vs_actual(val_rows, 'Ridge', COLOR, args.outdir, args.title)
plot_residuals(val_rows, 'Ridge', COLOR, args.outdir, args.title)
plot_cv_metrics(fold_metrics, 'Ridge', COLOR, args.outdir, args.title)
plot_error_per_sample(val_rows, 'Ridge', COLOR, args.outdir, args.title)
plot_overfitting(val_rows, train_rows, fold_metrics,
                 'Ridge', COLOR, args.outdir, args.title)

# ─── RIDGE-SPECIFIC: Coefficient plot (refitted on full data) ─────────────────
print("  Generating Ridge coefficient plot...")
scaler_X_full = StandardScaler()
scaler_y_full = StandardScaler()
X_full_s = scaler_X_full.fit_transform(X)
y_full_s = scaler_y_full.fit_transform(y.reshape(-1, 1)).ravel()

ridge_full = RidgeCV(alphas=args.alphas)
ridge_full.fit(X_full_s, y_full_s)
print(f"  Selected alpha (full-data refit): {ridge_full.alpha_:.4g}")

ps = PLOT_STYLE
fig, ax = plt.subplots(figsize=(9, 6))
if args.title:
    ax.set_title('Ridge — Standardised Coefficients  (refitted on full dataset)',
                 fontsize=ps['FONTSIZE_TITLE'], fontweight='bold', pad=14)

coefs = ridge_full.coef_
order = np.argsort(np.abs(coefs))
feature_names = [FEATURE_LABELS[fc] for fc in feature_cols]
colors_bar = ['#C62828' if c > 0 else '#1565C0' for c in coefs]

ax.barh([feature_names[i] for i in order], [coefs[i] for i in order],
       color=[colors_bar[i] for i in order], alpha=0.82,
       edgecolor='black', linewidth=1.2)
ax.axvline(0, color='black', linewidth=1.8)
ax.set_xlabel('Standardised Coefficient', fontsize=ps['FONTSIZE_LABEL'])
ax.tick_params(axis='both', labelsize=ps['FONTSIZE_TICK'])
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
ax.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#C62828', label='Positive effect on SF', alpha=0.82),
                   Patch(facecolor='#1565C0', label='Negative effect on SF', alpha=0.82)],
         fontsize=ps['FONTSIZE_LEGEND'] - 2, loc='lower right')

plt.tight_layout()
coef_path = os.path.join(args.outdir, 'ridge_coefficients.png')
plt.savefig(coef_path, dpi=ps['DPI'], bbox_inches='tight')
plt.close()
print(f"  Saved: {coef_path}")

coef_df = pd.DataFrame({'feature': feature_cols, 'standardised_coef': coefs})
coef_df.to_csv(os.path.join(args.outdir, 'ridge_coefficients.csv'), index=False)

print(f"\n{'='*60}")
print(f"  Ridge complete. All results in: {args.outdir}/")
print(f"{'='*60}")
