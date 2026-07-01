"""
multivariate_linearity.py
==========================
Linearity analysis between input features and SF_mean.
Informs model selection — if relationships are linear, Ridge may suffice;
if nonlinear, GP or NGBoost is justified.

Tasks:
    1. Scatter plots: each feature vs SF_mean, linear fit + R² annotation
    2. Residual plots from linear fit per feature, with LOWESS smoothing
    3. Linearity summary: R² (linear fit) per feature
    4. Partial regression plots (controls for the other 2 features)

Outputs (saved to ./figures/linearity/):
    scatter_sf.png             — scatter grid, all 3 features vs SF_mean
    residuals_sf.png           — residual grid
    linearity_r2_summary.png   — R² bar chart
    partial_regression_sf.png  — partial regression plots

Usage:
    python multivariate_linearity.py --data sample_sf_summary.csv
    python multivariate_linearity.py --data sample_sf_summary.csv --fit_line
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data',     type=str, default='sample_sf_summary.csv')
parser.add_argument('--outdir',   type=str, default='figures/linearity')
parser.add_argument('--fit_line', action='store_true', default=False,
                    help='Draw linear fit line on scatter plots (default: off)')
parser.add_argument('--annotate', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--title',    action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
print(f'Options: fit_line={args.fit_line}, annotate={args.annotate}, title={args.title}')

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
FEATURES = ['Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
OUTPUT   = 'SF_mean'
df_use = df.dropna(subset=FEATURES + [OUTPUT]).copy().reset_index(drop=True)
print(f"Loaded: {len(df)} rows, using {len(df_use)} with complete features/output")

SHORT_FEAT = {'Pressure_kPa': 'Pressure (kPa)', 'NozzleSpeed_mms': 'Speed (mm/s)',
             'Zoffset_mm': 'Z-offset (mm)'}

def linear_fit(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return slope, intercept, r**2, p

# ─── 1. SCATTER: FEATURE vs SF_mean ──────────────────────────────────────────
r2_rows = []
ncols = 3
fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
if args.title:
    fig.suptitle('Scatter Plots — Input Features vs SF_mean',
                 fontsize=FONTSIZE_TITLE + 2, fontweight='bold', y=1.03)

for ax, feat in zip(axes, FEATURES):
    x = df_use[feat].values
    y = df_use[OUTPUT].values
    ax.scatter(x, y, c=POINT_COLOR, s=80, alpha=0.78,
              edgecolor='white', linewidth=0.6)

    slope, intercept, r2, pval = linear_fit(x, y)
    r2_rows.append({'Feature': feat, 'R2_linear': r2, 'p_value': pval})

    if args.fit_line and np.isfinite(r2):
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color='black',
               linewidth=2.2, linestyle='--', label='Linear fit')
    if args.annotate and np.isfinite(r2):
        sig = '*' if pval < 0.05 else ''
        ax.text(0.04, 0.95, f'R\u00b2 = {r2:.3f}{sig}\np = {pval:.4f}',
               transform=ax.transAxes, fontsize=FONTSIZE_ANNOTATION, va='top',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='gray'))

    ax.set_xlabel(SHORT_FEAT.get(feat, feat), fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('SF_mean', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    if args.fit_line:
        ax.legend(fontsize=FONTSIZE_LEGEND - 2)

plt.tight_layout()
path = os.path.join(args.outdir, 'scatter_sf.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

# ─── 2. RESIDUAL PLOTS ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
if args.title:
    fig.suptitle('Residual Plots — Linear Fit per Feature vs SF_mean',
                 fontsize=FONTSIZE_TITLE + 2, fontweight='bold', y=1.03)

for ax, feat in zip(axes, FEATURES):
    x = df_use[feat].values
    y = df_use[OUTPUT].values
    slope, intercept, r2, _ = linear_fit(x, y)

    if np.isfinite(r2):
        y_pred = slope * x + intercept
        residuals = y - y_pred
        ax.scatter(y_pred, residuals, c=POINT_COLOR, s=70, alpha=0.78,
                  edgecolor='white', linewidth=0.6)
        ax.axhline(0, color='black', linewidth=2, linestyle='--')
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(residuals, y_pred, frac=0.6)
            ax.plot(smooth[:, 0], smooth[:, 1], color='darkorange',
                   linewidth=2.5, label='LOWESS', zorder=5)
            ax.legend(fontsize=FONTSIZE_LEGEND - 2)
        except ImportError:
            pass
        ax.text(0.04, 0.95, f'R\u00b2 (linear) = {r2:.3f}',
               transform=ax.transAxes, fontsize=FONTSIZE_ANNOTATION, va='top',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='gray'))
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', transform=ax.transAxes)

    ax.set_xlabel(f'Fitted values  [{SHORT_FEAT.get(feat, feat)}]', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Residual', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

plt.tight_layout()
path = os.path.join(args.outdir, 'residuals_sf.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

# ─── 3. R² SUMMARY ─────────────────────────────────────────────────────────────
df_r2 = pd.DataFrame(r2_rows)
r2_path = os.path.join(args.outdir, 'linearity_r2_summary.csv')
df_r2.to_csv(r2_path, index=False)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#C62828' if p < 0.05 else '#1565C0' for p in df_r2['p_value']]
bars = ax.bar([SHORT_FEAT.get(f, f) for f in df_r2['Feature']],
             df_r2['R2_linear'], color=colors, alpha=0.82,
             edgecolor='black', linewidth=1.2)
for i, (r2, pval) in enumerate(zip(df_r2['R2_linear'], df_r2['p_value'])):
    sig = '*' if pval < 0.05 else ''
    ax.text(i, r2 + 0.02, f'{r2:.3f}{sig}', ha='center',
           fontsize=FONTSIZE_ANNOTATION, fontweight='bold')
if args.title:
    ax.set_title('Linearity Summary — R\u00b2 of Simple Linear Fit per Feature\n'
                 '(* p < 0.05; red = significant linear relationship)',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=14)
ax.set_ylabel('R\u00b2 (linear fit)', fontsize=FONTSIZE_LABEL)
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=FONTSIZE_TICK)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(args.outdir, 'linearity_r2_summary.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"\n=== LINEARITY SUMMARY (R\u00b2 of simple linear fit vs SF_mean) ===")
print(df_r2.to_string(index=False))
print(f"Saved: {path}\nSaved: {r2_path}")

# ─── 4. PARTIAL REGRESSION PLOTS ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_use[FEATURES].values)
y_arr = df_use[OUTPUT].values

fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
if args.title:
    fig.suptitle('Partial Regression Plots — SF_mean\n'
                 '(Controls for the other 2 features — isolates unique contribution)',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.05)

for ax, (feat_idx, feat) in zip(axes, enumerate(FEATURES)):
    other_idx = [i for i in range(len(FEATURES)) if i != feat_idx]
    X_others = X_scaled[:, other_idx]
    x_feat   = X_scaled[:, feat_idx]

    reg_y = LinearRegression().fit(X_others, y_arr)
    resid_y = y_arr - reg_y.predict(X_others)
    reg_x = LinearRegression().fit(X_others, x_feat)
    resid_x = x_feat - reg_x.predict(X_others)

    ax.scatter(resid_x, resid_y, c=POINT_COLOR, s=75, alpha=0.78,
              edgecolor='white', linewidth=0.6)

    slope, intercept, r2_part, pval_part = linear_fit(resid_x, resid_y)
    if np.isfinite(r2_part):
        x_line = np.linspace(resid_x.min(), resid_x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color='black',
               linewidth=2.5, linestyle='--')
        sig_str = '*' if pval_part < 0.05 else ''
        ax.text(0.04, 0.95, f'Partial R\u00b2 = {r2_part:.3f}{sig_str}',
               transform=ax.transAxes, fontsize=FONTSIZE_ANNOTATION, va='top',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='gray'))

    ax.axhline(0, color='gray', linewidth=1, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=1, alpha=0.5)
    ax.set_xlabel(f'{SHORT_FEAT.get(feat, feat)} (residualized)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('SF_mean (residualized)', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

plt.tight_layout()
path = os.path.join(args.outdir, 'partial_regression_sf.png')
plt.savefig(path, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

print("\n=== LINEARITY ANALYSIS COMPLETE ===")
print(f"All figures saved to: {args.outdir}/")
print("\nInterpretation guide:")
print("  R\u00b2 > 0.6   -> strong linear relationship -> Ridge may work")
print("  R\u00b2 0.3-0.6 -> moderate linear relationship -> nonlinear model likely better")
print("  R\u00b2 < 0.3   -> weak/nonlinear relationship -> GP or NGBoost needed")
