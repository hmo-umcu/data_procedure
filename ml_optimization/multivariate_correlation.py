"""
multivariate_correlation.py
============================
Multivariate correlation analysis of the printing-parameter -> SF_mean dataset.

Tasks:
    1. Spearman correlation — each feature vs SF_mean (single-output bar,
       not a heatmap since there is only one output here)
    2. Spearman correlation heatmap — features vs features (multicollinearity check)
    3. Pairplot of all 3 features + SF_mean
    4. PCA of the 3 input features (dimensionality / redundancy check)

Outputs (saved to ./figures/correlation/):
    corr_features_vs_sf.png        — bar chart, Spearman rho per feature
    corr_features_vs_features.png  — heatmap, feature-feature multicollinearity
    pairplot_features_sf.png       — pairplot of features + SF_mean
    pca_analysis.png               — PCA variance explained + biplot

Usage:
    python multivariate_correlation.py --data sample_sf_summary.csv
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data',   type=str, default='sample_sf_summary.csv')
parser.add_argument('--outdir', type=str, default='figures/correlation')
parser.add_argument('--title',  action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

FONTSIZE_TITLE, FONTSIZE_LABEL = 18, 15
FONTSIZE_TICK, FONTSIZE_LEGEND, FONTSIZE_ANNOTATION = 14, 14, 13
LINE_WIDTH, DPI = 1.5, 200

plt.rcParams.update({
    'font.size': FONTSIZE_TICK, 'axes.titlesize': FONTSIZE_TITLE,
    'axes.labelsize': FONTSIZE_LABEL, 'xtick.labelsize': FONTSIZE_TICK,
    'ytick.labelsize': FONTSIZE_TICK, 'legend.fontsize': FONTSIZE_LEGEND,
    'axes.linewidth': LINE_WIDTH, 'xtick.major.width': LINE_WIDTH,
    'ytick.major.width': LINE_WIDTH, 'xtick.major.size': 6,
    'ytick.major.size': 6, 'figure.dpi': DPI,
})

POINT_COLOR = '#1565C0'

df = pd.read_csv(args.data, sep=';')
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

FEATURES = ['Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
OUTPUT   = 'SF_mean'
SHORT = {'Pressure_kPa': 'Pressure\n(kPa)', 'NozzleSpeed_mms': 'Speed\n(mm/s)',
        'Zoffset_mm': 'Z-offset\n(mm)', 'SF_mean': 'SF_mean'}
LONG  = {'Pressure_kPa': 'Pressure  (kPa)', 'NozzleSpeed_mms': 'Nozzle Speed  (mm/s)',
        'Zoffset_mm': 'Z-offset  (mm)', 'SF_mean': 'Shape Fidelity (SF_mean)'}

df_use = df.dropna(subset=FEATURES + [OUTPUT]).copy().reset_index(drop=True)
print(f"Rows used (non-missing features/output): {len(df_use)}")

# ─── 1. SPEARMAN: FEATURES vs SF_mean ────────────────────────────────────────
rho_rows = []
for feat in FEATURES:
    rho, pval = stats.spearmanr(df_use[feat], df_use[OUTPUT])
    rho_rows.append({'Feature': feat, 'Spearman_rho': rho, 'p_value': pval,
                     'Significant (p<0.05)': pval < 0.05})
df_rho = pd.DataFrame(rho_rows)
rho_path = os.path.join(args.outdir, 'corr_features_vs_sf.csv')
df_rho.to_csv(rho_path, index=False)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#C62828' if r > 0 else '#1565C0' for r in df_rho['Spearman_rho']]
bars = ax.barh([SHORT.get(f, f).replace('\n', ' ') for f in df_rho['Feature']],
              df_rho['Spearman_rho'], color=colors, alpha=0.82,
              edgecolor='black', linewidth=1.2)
for i, (rho, pval) in enumerate(zip(df_rho['Spearman_rho'], df_rho['p_value'])):
    sig = '*' if pval < 0.05 else ''
    ax.text(rho + (0.02 if rho >= 0 else -0.02), i, f'{rho:.3f}{sig}',
           va='center', ha='left' if rho >= 0 else 'right',
           fontsize=FONTSIZE_ANNOTATION, fontweight='bold')
ax.axvline(0, color='black', linewidth=1.8)
if args.title:
    ax.set_title('Spearman Correlation — Input Features vs SF_mean\n(* p < 0.05)',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=14)
ax.set_xlabel('Spearman ρ', fontsize=FONTSIZE_LABEL)
ax.set_xlim(-1, 1)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
path = os.path.join(args.outdir, 'corr_features_vs_sf.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"\n=== FEATURE vs SF_mean (Spearman) ===")
print(df_rho.to_string(index=False))
print(f"Saved: {path}\nSaved: {rho_path}")

# ─── 2. SPEARMAN: FEATURES vs FEATURES (multicollinearity) ──────────────────
def compute_spearman_matrix(df_in, vars_):
    rho_mat = pd.DataFrame(index=vars_, columns=vars_, dtype=float)
    for r in vars_:
        for c in vars_:
            rho, _ = stats.spearmanr(df_in[r], df_in[c])
            rho_mat.loc[r, c] = rho
    return rho_mat.astype(float)

rho_ff = compute_spearman_matrix(df_use, FEATURES)
mask = np.triu(np.ones_like(rho_ff, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(rho_ff, mask=mask, annot=True, fmt='.2f',
           annot_kws={'size': FONTSIZE_ANNOTATION + 1, 'weight': 'bold'},
           cmap='RdBu_r', vmin=-1, vmax=1, center=0,
           linewidths=0.6, linecolor='white',
           xticklabels=[SHORT.get(f, f) for f in FEATURES],
           yticklabels=[SHORT.get(f, f) for f in FEATURES],
           ax=ax, cbar_kws={'label': 'Spearman ρ', 'shrink': 0.85})
if args.title:
    ax.set_title('Spearman Correlation — Feature × Feature\n'
                 '(Multicollinearity check)', fontsize=FONTSIZE_TITLE,
                 fontweight='bold', pad=14)
ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
plt.tight_layout()
path = os.path.join(args.outdir, 'corr_features_vs_features.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"\n=== FEATURE-FEATURE CORRELATION ===")
print(rho_ff.round(3).to_string())

flagged = []
for i, r in enumerate(FEATURES):
    for c in FEATURES[i+1:]:
        if abs(rho_ff.loc[r, c]) > 0.7:
            flagged.append((r, c, rho_ff.loc[r, c]))
if flagged:
    print("\n[WARNING] Multicollinearity flagged (|rho| > 0.7):")
    for r, c, rho in flagged:
        print(f"  {r} <-> {c}: rho={rho:.3f}")
else:
    print("\nNo feature pairs with |rho| > 0.7 — no multicollinearity concern")
print(f"Saved: {path}")

# ─── 3. PAIRPLOT — FEATURES + SF_mean ────────────────────────────────────────
plot_df = df_use[FEATURES + [OUTPUT]].rename(
    columns={c: LONG.get(c, c) for c in FEATURES + [OUTPUT]})
g = sns.pairplot(plot_df, diag_kind='kde',
                 plot_kws={'alpha': 0.7, 's': 60, 'color': POINT_COLOR,
                          'edgecolor': 'white', 'linewidth': 0.5},
                 diag_kws={'color': POINT_COLOR, 'linewidth': 2.5})
if args.title:
    g.figure.suptitle('Pairplot — Input Features + SF_mean',
                      fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.02)
for ax in g.axes.flat:
    if ax is not None:
        ax.tick_params(labelsize=FONTSIZE_TICK)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=FONTSIZE_LABEL - 1)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=FONTSIZE_LABEL - 1)
path = os.path.join(args.outdir, 'pairplot_features_sf.png')
g.figure.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

# ─── 4. PCA ───────────────────────────────────────────────────────────────────
X = df_use[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sf_vals = df_use[OUTPUT].values

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100
cumulative = np.cumsum(explained)
n_components = len(FEATURES)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
if args.title:
    fig.suptitle('PCA of Input Features', fontsize=FONTSIZE_TITLE + 2, fontweight='bold')

axes[0].bar(range(1, n_components + 1), explained, color=POINT_COLOR, alpha=0.8,
           edgecolor='black', linewidth=1.2)
axes[0].plot(range(1, n_components + 1), cumulative, 'o-', color='#C62828',
            linewidth=2.5, markersize=8, label='Cumulative')
axes[0].axhline(90, color='gray', linestyle='--', linewidth=1.8, label='90% threshold')
for i, (e, c) in enumerate(zip(explained, cumulative)):
    axes[0].text(i + 1, e + 1, f'{e:.1f}%', ha='center',
                fontsize=FONTSIZE_ANNOTATION, fontweight='bold')
axes[0].set_xlabel('Principal Component', fontsize=FONTSIZE_LABEL)
axes[0].set_ylabel('Variance Explained (%)', fontsize=FONTSIZE_LABEL)
axes[0].set_xticks(range(1, n_components + 1))
axes[0].tick_params(labelsize=FONTSIZE_TICK)
axes[0].legend(fontsize=FONTSIZE_LEGEND)
axes[0].set_ylim(0, 110)
axes[0].grid(axis='y', alpha=0.3)

sc = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=sf_vals, cmap='viridis',
                     s=90, alpha=0.85, edgecolor='white', linewidth=0.7)
cbar = plt.colorbar(sc, ax=axes[1], shrink=0.85)
cbar.set_label('SF_mean', fontsize=FONTSIZE_LABEL - 2)
axes[1].set_xlabel(f'PC1  ({explained[0]:.1f}% variance)', fontsize=FONTSIZE_LABEL)
axes[1].set_ylabel(f'PC2  ({explained[1]:.1f}% variance)', fontsize=FONTSIZE_LABEL)
axes[1].tick_params(labelsize=FONTSIZE_TICK)
axes[1].grid(alpha=0.25)
axes[1].axhline(0, color='gray', linewidth=0.8, alpha=0.5)
axes[1].axvline(0, color='gray', linewidth=0.8, alpha=0.5)

loadings = pca.components_.T
scale = 2.5
for i, feat in enumerate(FEATURES):
    axes[2].annotate('', xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=POINT_COLOR, lw=2.5))
    axes[2].text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15,
                SHORT.get(feat, feat).replace('\n', ' '),
                fontsize=FONTSIZE_ANNOTATION, color=POINT_COLOR,
                fontweight='bold', ha='center', va='center')
axes[2].axhline(0, color='gray', linewidth=0.8, alpha=0.5)
axes[2].axvline(0, color='gray', linewidth=0.8, alpha=0.5)
axes[2].set_xlabel(f'PC1  ({explained[0]:.1f}%)', fontsize=FONTSIZE_LABEL)
axes[2].set_ylabel(f'PC2  ({explained[1]:.1f}%)', fontsize=FONTSIZE_LABEL)
axes[2].tick_params(labelsize=FONTSIZE_TICK)
axes[2].grid(alpha=0.25)
lim = loadings[:, :2].max() * scale * 1.5
axes[2].set_xlim(-lim, lim)
axes[2].set_ylim(-lim, lim)

plt.tight_layout()
path = os.path.join(args.outdir, 'pca_analysis.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"\n=== PCA SUMMARY ===")
for i, (e, c) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1}: {e:.1f}%  (cumulative: {c:.1f}%)")
n_for_90 = np.searchsorted(cumulative, 90) + 1
print(f"  -> {n_for_90} component(s) explain >=90% of variance")
print(f"Saved: {path}")

print("\n=== CORRELATION ANALYSIS COMPLETE ===")
print(f"All figures saved to: {args.outdir}/")
