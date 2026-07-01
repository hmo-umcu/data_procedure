"""
model_comparison.py
=====================
Compare Ridge, GPR, and NGBoost performance for SF_mean prediction.
Reads each model's cv_metrics_summary.csv / cv_metrics_per_fold.csv /
prediction_log.csv (already produced by model_ridge.py, model_gpr.py,
model_ngboost.py) and produces side-by-side comparison figures + a summary
table — does not retrain anything itself.

Run the three model scripts first:
    python model_ridge.py    --data sample_sf_summary.csv
    python model_gpr.py      --data sample_sf_summary.csv
    python model_ngboost.py  --data sample_sf_summary.csv

Then:
    python model_comparison.py --results_dir results

Outputs (saved to --outdir, default results/comparison/):
    comparison_metrics_summary.csv     — RMSE/MAE/R²/MAPE mean±std per model
    comparison_bar_metrics.png         — bar chart, all 4 metrics, all 3 models
    comparison_predicted_vs_actual.png — predicted-vs-actual, 1 panel/model
    comparison_per_fold_r2.png         — per-fold R² across models
    comparison_error_distribution.png  — abs error boxplot per model
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Compare Ridge / GPR / NGBoost results')
parser.add_argument('--results_dir', type=str, default='results',
                    help='Parent dir containing Ridge/, GPR/, NGBoost/ subfolders')
parser.add_argument('--outdir',      type=str, default='results/comparison')
parser.add_argument('--models',      type=str, nargs='+',
                    default=['Ridge', 'GPR', 'NGBoost'],
                    help='Subfolder names to compare (must match model script --outdir leaf names)')
parser.add_argument('--title',       action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

FONTSIZE_TITLE, FONTSIZE_LABEL = 20, 18
FONTSIZE_TICK, FONTSIZE_LEGEND, FONTSIZE_ANNOTATION = 16, 16, 15
LINE_WIDTH, DPI = 2.0, 200

plt.rcParams.update({
    'font.size': FONTSIZE_TICK, 'axes.titlesize': FONTSIZE_TITLE,
    'axes.labelsize': FONTSIZE_LABEL, 'xtick.labelsize': FONTSIZE_TICK,
    'ytick.labelsize': FONTSIZE_TICK, 'legend.fontsize': FONTSIZE_LEGEND,
    'axes.linewidth': LINE_WIDTH, 'xtick.major.width': LINE_WIDTH,
    'ytick.major.width': LINE_WIDTH, 'xtick.major.size': 7,
    'ytick.major.size': 7, 'figure.dpi': DPI,
})

MODEL_COLORS = {'Ridge': '#1565C0', 'GPR': '#2E7D32', 'NGBoost': '#C62828'}

# ─── LOAD EACH MODEL'S RESULTS ────────────────────────────────────────────────
metrics_summaries = {}
fold_metrics_all  = {}
prediction_logs    = {}

for model_name in args.models:
    model_dir = os.path.join(args.results_dir, model_name)
    summary_path = os.path.join(model_dir, 'cv_metrics_summary.csv')
    folds_path   = os.path.join(model_dir, 'cv_metrics_per_fold.csv')
    log_path     = os.path.join(model_dir, 'prediction_log.csv')

    if not os.path.exists(summary_path):
        print(f"[WARN] {summary_path} not found — skipping {model_name}. "
              f"Did you run model_{model_name.lower()}.py first?")
        continue

    metrics_summaries[model_name] = pd.read_csv(summary_path)
    fold_metrics_all[model_name]  = pd.read_csv(folds_path)
    prediction_logs[model_name]   = pd.read_csv(log_path)
    print(f"Loaded results for {model_name} from {model_dir}")

available_models = list(metrics_summaries.keys())
if not available_models:
    print("[ERROR] No model results found. Run the model scripts first.")
    raise SystemExit(1)

# ─── SUMMARY TABLE ────────────────────────────────────────────────────────────
summary_rows = []
for model_name in available_models:
    sub = metrics_summaries[model_name].set_index('Metric')
    summary_rows.append({
        'Model': model_name,
        'RMSE_mean': sub.loc['RMSE', 'Mean'], 'RMSE_std': sub.loc['RMSE', 'Std'],
        'MAE_mean':  sub.loc['MAE',  'Mean'], 'MAE_std':  sub.loc['MAE',  'Std'],
        'R2_mean':   sub.loc['R2',   'Mean'], 'R2_std':   sub.loc['R2',   'Std'],
        'MAPE_mean': sub.loc['MAPE', 'Mean'], 'MAPE_std': sub.loc['MAPE', 'Std'],
    })
df_summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(args.outdir, 'comparison_metrics_summary.csv')
df_summary.to_csv(summary_path, index=False)

print(f"\n{'='*72}")
print("  MODEL COMPARISON — CV RESULTS (mean ± std across folds)")
print(f"{'='*72}")
print(f"  {'Model':<10} {'RMSE':>14} {'MAE':>14} {'R2':>12} {'MAPE':>12}")
print(f"  {'-'*68}")
for _, row in df_summary.iterrows():
    print(f"  {row['Model']:<10} "
          f"{row['RMSE_mean']:>6.4f}\u00b1{row['RMSE_std']:.4f}  "
          f"{row['MAE_mean']:>6.4f}\u00b1{row['MAE_std']:.4f}  "
          f"{row['R2_mean']:>5.3f}\u00b1{row['R2_std']:.3f}  "
          f"{row['MAPE_mean']:>5.2f}\u00b1{row['MAPE_std']:.2f}%")
best_model = df_summary.loc[df_summary['R2_mean'].idxmax(), 'Model']
print(f"\n  Best model by mean validation R\u00b2: {best_model}")
print(f"  Saved: {summary_path}")

# ─── FIGURE 1: BAR CHART, ALL METRICS ────────────────────────────────────────
metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
metric_labels = {'RMSE': 'RMSE', 'MAE': 'MAE', 'R2': 'R\u00b2', 'MAPE': 'MAPE (%)'}

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
if args.title:
    fig.suptitle('Model Comparison — CV Metrics  (mean ± std across folds)',
                 fontsize=FONTSIZE_TITLE + 2, fontweight='bold', y=1.05)

x = np.arange(len(available_models))
for col, metric in enumerate(metrics):
    ax = axes[col]
    means = [df_summary[df_summary['Model'] == m][f'{metric}_mean'].values[0]
            for m in available_models]
    stds  = [df_summary[df_summary['Model'] == m][f'{metric}_std'].values[0]
            for m in available_models]
    colors = [MODEL_COLORS.get(m, 'gray') for m in available_models]

    ax.bar(x, means, 0.6, yerr=stds, capsize=6, color=colors, alpha=0.82,
          edgecolor='black', linewidth=1.5)
    for i, (m_, s_) in enumerate(zip(means, stds)):
        ax.text(i, m_ + s_ + (0.02 if metric != 'MAPE' else 1.0),
               f'{m_:.3f}' if metric != 'MAPE' else f'{m_:.1f}%',
               ha='center', fontsize=FONTSIZE_ANNOTATION, fontweight='bold')

    if metric == 'R2':
        ax.axhline(0, color='black', linewidth=1.0, alpha=0.4)
        ax.axhline(1.0, color='gray', linewidth=1.2, linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(available_models, fontsize=FONTSIZE_TICK)
    ax.set_ylabel(metric_labels[metric], fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path = os.path.join(args.outdir, 'comparison_bar_metrics.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

# ─── FIGURE 2: PREDICTED vs ACTUAL, SIDE BY SIDE ─────────────────────────────
fig, axes = plt.subplots(1, len(available_models),
                         figsize=(7.5 * len(available_models), 7.5))
if len(available_models) == 1:
    axes = [axes]
if args.title:
    fig.suptitle('Predicted vs Actual SF — Model Comparison  (CV validation folds)',
                 fontsize=FONTSIZE_TITLE + 2, fontweight='bold', y=1.03)

all_y = np.concatenate([
    prediction_logs[m][prediction_logs[m]['split'] == 'val']['label'].values
    for m in available_models
] + [
    prediction_logs[m][prediction_logs[m]['split'] == 'val']['pred'].values
    for m in available_models
])
margin = (all_y.max() - all_y.min()) * 0.08
lims = [all_y.min() - margin, all_y.max() + margin]

for ax, model_name in zip(axes, available_models):
    df_val = prediction_logs[model_name]
    df_val = df_val[df_val['split'] == 'val']
    y_true, y_pred = df_val['label'].values, df_val['pred'].values
    color = MODEL_COLORS.get(model_name, 'gray')

    ax.scatter(y_true, y_pred, color=color, s=100, alpha=0.78,
              edgecolor='white', linewidth=0.7)
    ax.plot(lims, lims, 'k--', linewidth=2.0, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.04, 0.95, f'R\u00b2 = {r2:.3f}\nRMSE = {rmse:.4f}',
           transform=ax.transAxes, fontsize=FONTSIZE_ANNOTATION, va='top',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88, ec='gray'))
    ax.set_title(model_name, fontsize=FONTSIZE_LABEL, fontweight='bold', color=color)
    ax.set_xlabel('Actual SF', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Predicted SF', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(alpha=0.25)

plt.tight_layout()
path = os.path.join(args.outdir, 'comparison_predicted_vs_actual.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

# ─── FIGURE 3: PER-FOLD R² ACROSS MODELS ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
if args.title:
    ax.set_title('Per-Fold Validation R\u00b2 — Model Comparison',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=14)

n_folds = len(fold_metrics_all[available_models[0]])
x = np.arange(n_folds)
w = 0.8 / len(available_models)

for i, model_name in enumerate(available_models):
    vals = fold_metrics_all[model_name]['R2'].values
    color = MODEL_COLORS.get(model_name, 'gray')
    offset = (i - (len(available_models) - 1) / 2) * w
    ax.bar(x + offset, vals, w, label=model_name, color=color, alpha=0.82,
          edgecolor='black', linewidth=1.2)

ax.axhline(0, color='black', linewidth=1.0, alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)], fontsize=FONTSIZE_TICK)
ax.set_ylabel('Validation R\u00b2', fontsize=FONTSIZE_LABEL)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=FONTSIZE_LEGEND)

plt.tight_layout()
path = os.path.join(args.outdir, 'comparison_per_fold_r2.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

# ─── FIGURE 4: ABS ERROR DISTRIBUTION BOXPLOT ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
if args.title:
    ax.set_title('Absolute Error Distribution — Model Comparison\n'
                 '(CV validation folds, all samples pooled)',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=14)

data_plot, colors_plot = [], []
for model_name in available_models:
    df_val = prediction_logs[model_name]
    df_val = df_val[df_val['split'] == 'val']
    data_plot.append(df_val['abs_error'].values)
    colors_plot.append(MODEL_COLORS.get(model_name, 'gray'))

bp = ax.boxplot(data_plot, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2.5))
for patch, color in zip(bp['boxes'], colors_plot):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for i, vals in enumerate(data_plot):
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
              color='black', s=25, alpha=0.5, zorder=3)

ax.set_xticks(range(1, len(available_models) + 1))
ax.set_xticklabels(available_models, fontsize=FONTSIZE_TICK)
ax.set_ylabel('Absolute Error (SF)', fontsize=FONTSIZE_LABEL)
ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path = os.path.join(args.outdir, 'comparison_error_distribution.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

print(f"\n{'='*60}")
print(f"  Comparison complete. All results in: {args.outdir}/")
print(f"  Best model by mean validation R\u00b2: {best_model}")
print(f"{'='*60}")
