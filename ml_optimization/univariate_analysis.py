"""
univariate_analysis.py
=======================
Univariate analysis of the printing-parameter -> SF_mean dataset.

Tasks:
    1. Data quality check — missing values, sample count
    2. Distribution of each input feature and the output (SF_mean)
       — histogram + KDE
    3. Summary statistics table (mean, std, min, max, CV)
    4. SF_mean vs SF_std relationship (replicate consistency sanity check)
    5. CV-fold distribution check (if 'fold' column present) — confirms
       SF_mean is reasonably balanced across folds, not concentrated

Outputs saved to --outdir (default: figures/univariate/)

Usage:
    python univariate_analysis.py --data sample_sf_summary.csv
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data',   type=str, default='sample_sf_summary.csv')
parser.add_argument('--outdir', type=str, default='figures/univariate')
parser.add_argument('--title',  action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

FONTSIZE_TITLE, FONTSIZE_LABEL = 24, 22
FONTSIZE_TICK, FONTSIZE_LEGEND, FONTSIZE_ANNOTATION = 20, 20, 18
LINE_WIDTH, DPI = 2.0, 200

plt.rcParams.update({
    'font.size': FONTSIZE_TICK, 'axes.titlesize': FONTSIZE_TITLE,
    'axes.labelsize': FONTSIZE_LABEL, 'xtick.labelsize': FONTSIZE_TICK,
    'ytick.labelsize': FONTSIZE_TICK, 'legend.fontsize': FONTSIZE_LEGEND,
    'axes.linewidth': LINE_WIDTH, 'xtick.major.width': LINE_WIDTH,
    'ytick.major.width': LINE_WIDTH, 'xtick.major.size': 8,
    'ytick.major.size': 8, 'figure.dpi': DPI,
})

POINT_COLOR = '#1565C0'

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv(args.data, sep=';')
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

FEATURES = ['Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
OUTPUT   = 'SF_mean'
FEATURE_LABELS = {
    'Pressure_kPa':    'Pressure  (kPa)',
    'NozzleSpeed_mms': 'Nozzle Speed  (mm/s)',
    'Zoffset_mm':      'Z-offset  (mm)',
    'SF_mean':         'Shape Fidelity SF_mean',
    'SF_std':          'SF_std  (within-sample replicate std)',
}

# ─── 1. DATA QUALITY CHECK ────────────────────────────────────────────────────
print("\n=== DATA QUALITY CHECK ===")
print(f"Missing values:\n{df[FEATURES + [OUTPUT]].isnull().sum()}")
print(f"\nN samples: {len(df)}")
if 'n_images' in df.columns:
    print(f"Replicate count per sample:\n{df['n_images'].value_counts().sort_index()}")
    if (df['n_images'] != 6).any():
        n_off = (df['n_images'] != 6).sum()
        print(f"[NOTE] {n_off} sample(s) have fewer than 6 replicate images "
              f"— check these for missing/failed prints before modeling.")

# ─── 2. SUMMARY STATISTICS ───────────────────────────────────────────────────
summary_cols = FEATURES + [OUTPUT] + (['SF_std'] if 'SF_std' in df.columns else [])
summary_rows = []
for col in summary_cols:
    vals = df[col].dropna()
    cv = (vals.std() / vals.mean() * 100) if vals.mean() != 0 else np.nan
    summary_rows.append({
        'Variable': col, 'N': len(vals),
        'Mean': round(vals.mean(), 4), 'Std': round(vals.std(), 4),
        'Min': round(vals.min(), 4), 'Max': round(vals.max(), 4),
        'CV (%)': round(cv, 2),
    })
df_summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(args.outdir, 'summary_statistics.csv')
df_summary.to_csv(summary_path, index=False)
print(f"\n{df_summary.to_string(index=False)}")
print(f"\nSummary stats saved: {summary_path}")

# ─── 3. HISTOGRAM GRID — FEATURES + OUTPUT ───────────────────────────────────
def plot_histogram_grid(variables, title, filename, ncols=2):
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows))
    if args.title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE + 2, fontweight='bold', y=1.01)
    axes_flat = np.array(axes).flat if nrows * ncols > 1 else [axes]

    for ax, col in zip(axes_flat, variables):
        vals = df[col].dropna()
        ax.hist(vals, bins=min(12, max(5, len(vals) // 3)), alpha=0.7,
               color=POINT_COLOR, edgecolor='white', linewidth=0.8)
        if len(vals) > 3 and vals.std() > 1e-9:
            kde_x = np.linspace(vals.min(), vals.max(), 200)
            kde = stats.gaussian_kde(vals, bw_method='scott')
            bw = (vals.max() - vals.min()) / min(12, max(5, len(vals) // 3))
            ax.plot(kde_x, kde(kde_x) * len(vals) * bw,
                   color='#C62828', linewidth=3.0, label='KDE')
        ax.axvline(vals.mean(), color='black', linestyle='--', linewidth=2.5,
                  label=f'Mean={vals.mean():.3g} ± {vals.std():.3g}')
        ax.set_xlabel(FEATURE_LABELS.get(col, col), fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Count', fontsize=FONTSIZE_LABEL)
        ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=FONTSIZE_LEGEND - 2, loc='upper right')

    all_axes = list(axes_flat)
    for ax in all_axes[len(variables):]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(args.outdir, filename)
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

plot_histogram_grid(FEATURES, 'Univariate Distribution — Input Parameters',
                    'univariate_inputs.png', ncols=3)
plot_histogram_grid([OUTPUT], 'Univariate Distribution — Output (SF_mean)',
                    'univariate_output.png', ncols=1)

# ─── 4. SF_mean vs SF_std (replicate consistency check) ──────────────────────
if 'SF_std' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 8))
    if args.title:
        ax.set_title('Replicate Consistency — SF_std vs SF_mean\n'
                     '(Does higher SF correlate with more/less replicate variability?)',
                     fontsize=FONTSIZE_TITLE, fontweight='bold', pad=14)
    ax.scatter(df[OUTPUT], df['SF_std'], color=POINT_COLOR, s=110, alpha=0.78,
              edgecolor='white', linewidth=0.8)
    rho, pval = stats.spearmanr(df[OUTPUT], df['SF_std'])
    ax.text(0.04, 0.95, f'Spearman ρ = {rho:.3f}\np = {pval:.4f}',
           transform=ax.transAxes, fontsize=FONTSIZE_ANNOTATION, va='top',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='gray'))
    ax.set_xlabel('SF_mean', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('SF_std  (within-sample, across 6 replicates)', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path = os.path.join(args.outdir, 'sf_mean_vs_sf_std.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ─── 5. SF_mean DISTRIBUTION BY CV FOLD ──────────────────────────────────────
if 'fold' in df.columns:
    folds = sorted(df['fold'].dropna().unique())
    fig, ax = plt.subplots(figsize=(11, 8))
    if args.title:
        ax.set_title('SF_mean Distribution by CV Fold\n'
                     '(Check folds are reasonably balanced, not skewed)',
                     fontsize=FONTSIZE_TITLE, fontweight='bold', pad=14)
    data_plot = [df[df['fold'] == f][OUTPUT].dropna().values for f in folds]
    bp = ax.boxplot(data_plot, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2.5))
    for patch in bp['boxes']:
        patch.set_facecolor(POINT_COLOR)
        patch.set_alpha(0.7)
    for i, vals in enumerate(data_plot):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                  color='black', s=30, alpha=0.6, zorder=3)
    ax.set_xticks(range(1, len(folds) + 1))
    ax.set_xticklabels(folds, fontsize=FONTSIZE_TICK)
    ax.set_ylabel('SF_mean', fontsize=FONTSIZE_LABEL)
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.outdir, 'sf_mean_by_fold.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

print("\n=== UNIVARIATE ANALYSIS COMPLETE ===")
print(f"All figures saved to: {args.outdir}/")
