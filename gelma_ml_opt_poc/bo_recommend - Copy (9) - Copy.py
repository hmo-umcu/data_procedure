"""
bo_recommend.py
================
The actual Bayesian Optimization step: given the GPR surrogate trained on
ALL currently collected (Pressure_kPa, NozzleSpeed_mms, Zoffset_mm) ->
SF_mean data, recommend the single next print parameter combination
expected to most improve SF, via Expected Improvement (EI).

How this differs from model_gpr.py
-----------------------------------
model_gpr.py cross-validates the GPR surrogate on held-out folds to
answer "how trustworthy is this surrogate?" (R²=0.78 there). This script
assumes that question is already answered, refits the GPR on the FULL
dataset (every sample contributes, no held-out fold — you want maximum
information when actually recommending a real print), and performs the
forward search step: maximizing an acquisition function over the
continuous parameter space to propose where to print next. This is the
step that didn't exist yet in the EDA/CV scripts.

Search bounds
-------------
Default to the observed data range:
    Pressure_kPa:    [min, max] of collected data
    NozzleSpeed_mms: [min, max] of collected data
    Zoffset_mm:      [min, max] of collected data
Override explicitly via --pressure_bounds / --speed_bounds / --zoffset_bounds
if you want to search outside the observed range (extrapolation — GPR
uncertainty grows quickly outside the training region, so EI will tend to
favor points near the boundary if you do this; sensible, but be aware the
surrogate is least trustworthy there).

Acquisition function: Expected Improvement (EI), maximizing SF_mean.
    EI(x) = (mu(x) - f_best - xi) * Phi(Z) + sigma(x) * phi(Z),  if sigma(x) > 0
    EI(x) = 0,                                                   if sigma(x) = 0
    Z = (mu(x) - f_best - xi) / sigma(x)
All computed in raw SF units (not standardized) so --xi is directly
interpretable as an SF improvement threshold. xi (default 0.01) controls
explore/exploit: larger xi -> more exploration of uncertain regions,
smaller xi -> more exploitation near the current best.

EI itself is maximized via multi-start L-BFGS-B (--n_restarts random
starts within bounds, default 50), since EI is smooth but can have more
than one local maximum over a 3D box. By default only the single best
(highest-EI) point across all restarts is reported and logged — pass
--top_k N to also print the top N distinct local optima side by side
(with their own predicted mean/std/EI), so you can see what alternatives
were considered and why the winner beat them, rather than taking the
final choice on faith.

Usage:
    python bo_recommend.py --data sample_sf_summary.csv
    python bo_recommend.py --data sample_sf_summary.csv --xi 0.02
    python bo_recommend.py --data sample_sf_summary.csv \\
        --pressure_bounds 70 130 --speed_bounds 4 16 --zoffset_bounds 0.05 0.8

Output (appended across runs, so repeated calls build a recommendation history):
    bo_recommendation_log.csv   — one row per run: recommended point,
                                   predicted SF mean ± std, EI value,
                                   current best observed SF, iteration number
    bo_recommendation_<iter>.png — 1D slices through the recommended point:
                                   predicted SF (±1 std) and EI, one panel
                                   per parameter, holding the other two fixed
                                   at the recommended value
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_utils import FEATURES, FEATURE_LABELS, apply_plot_style, PLOT_STYLE

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
apply_plot_style()

DEFAULT_PRESSURE_BOUNDS = None   # filled from data if not given
DEFAULT_SPEED_BOUNDS    = None
DEFAULT_ZOFFSET_BOUNDS  = None

parser = argparse.ArgumentParser(description='BO recommendation step — Expected Improvement over GPR surrogate')
parser.add_argument('--data',    type=str, default='sample_sf_summary.csv')
parser.add_argument('--outdir',  type=str, default='results/BO')
parser.add_argument('--seed',    type=int, default=42)
parser.add_argument('--title',   action=argparse.BooleanOptionalAction, default=True)
# GPR kernel hyperparameters — kept consistent with model_gpr.py defaults
parser.add_argument('--nu',          type=float, default=1.5)
parser.add_argument('--n_restarts_kernel', type=int, default=10,
                    help='Optimizer restarts for GPR kernel hyperparameters (default 10)')
parser.add_argument('--alpha',       type=float, default=1e-6)
# Acquisition function
parser.add_argument('--xi', type=float, default=0.01,
                    help='EI exploration parameter, in raw SF units (default 0.01)')
parser.add_argument('--n_restarts', type=int, default=50,
                    help='Multi-start restarts for maximizing EI (default 50)')
parser.add_argument('--top_k', type=int, default=1,
                    help='Number of distinct local EI maxima to report '
                         '(default 1, matching previous behavior). The '
                         'recommendation actually logged/plotted is always '
                         'the rank-1 (highest EI) candidate; --top_k > 1 '
                         'only adds a side-by-side comparison table so you '
                         'can see what was traded off against the winner.')
# Search bounds — default to observed data range if not given
parser.add_argument('--pressure_bounds', type=float, nargs=2, default=None,
                    metavar=('MIN', 'MAX'))
parser.add_argument('--speed_bounds',    type=float, nargs=2, default=None,
                    metavar=('MIN', 'MAX'))
parser.add_argument('--zoffset_bounds',  type=float, nargs=2, default=None,
                    metavar=('MIN', 'MAX'))
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

print("=" * 60)
print("  Bayesian Optimization — Next-Point Recommendation (EI)")
print("=" * 60)

# ─── LOAD DATA (all of it — no CV split, BO wants full information) ─────────
df_raw = pd.read_csv(args.data, sep=';')
df = df_raw.dropna(subset=FEATURES + ['SF_mean']).copy().reset_index(drop=True)
X = df[FEATURES].values.astype(float)
y = df['SF_mean'].values.astype(float)
print(f"Loaded {len(df)} samples (full dataset, no held-out fold)")

# ─── BOUNDS ────────────────────────────────────────────────────────────────────
def resolve_bounds(cli_bounds, data_col):
    if cli_bounds is not None:
        return tuple(cli_bounds)
    return (float(data_col.min()), float(data_col.max()))

bounds_dict = {
    'Pressure_kPa':    resolve_bounds(args.pressure_bounds, df['Pressure_kPa']),
    'NozzleSpeed_mms': resolve_bounds(args.speed_bounds,    df['NozzleSpeed_mms']),
    'Zoffset_mm':      resolve_bounds(args.zoffset_bounds,  df['Zoffset_mm']),
}
bounds = [bounds_dict[f] for f in FEATURES]
print("Search bounds:")
for f, b in bounds_dict.items():
    print(f"  {f}: [{b[0]:g}, {b[1]:g}]")

# ─── FIT GPR ON FULL DATASET ──────────────────────────────────────────────────
scaler_X = StandardScaler()
X_s = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

kernel = (ConstantKernel(1.0, (1e-3, 1e3))
         * Matern(length_scale=np.ones(X.shape[1]),
                  length_scale_bounds=(1e-2, 1e2), nu=args.nu)
         + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1)))

gpr = GaussianProcessRegressor(
    kernel=kernel, alpha=args.alpha,
    n_restarts_optimizer=args.n_restarts_kernel,
    normalize_y=False, random_state=args.seed,
)
gpr.fit(X_s, y_s)
print(f"\nFitted GPR kernel (full-data refit): {gpr.kernel_}")

f_best = y.max()
best_idx = np.argmax(y)
print(f"Current best observed SF_mean: {f_best:.4f}  "
      f"(Sample_ID={df['Sample_ID'].iloc[best_idx] if 'Sample_ID' in df.columns else best_idx}, "
      f"at {dict(zip(FEATURES, X[best_idx]))})")

# ─── PREDICTION + EI HELPERS (operate in raw units) ──────────────────────────
def predict_raw(x_raw):
    """x_raw: (n,3) array in raw parameter units. Returns mu, sigma in raw SF units."""
    x_s = scaler_X.transform(np.atleast_2d(x_raw))
    mu_s, sigma_s = gpr.predict(x_s, return_std=True)
    mu = mu_s * scaler_y.scale_[0] + scaler_y.mean_[0]
    sigma = sigma_s * scaler_y.scale_[0]
    return mu, np.maximum(sigma, 0.0)

def expected_improvement(x_raw):
    mu, sigma = predict_raw(x_raw)
    mu, sigma = mu[0], sigma[0]
    if sigma < 1e-9:
        return 0.0
    z = (mu - f_best - args.xi) / sigma
    ei = (mu - f_best - args.xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return max(ei, 0.0)

def neg_ei(x_raw):
    return -expected_improvement(x_raw)

# ─── MAXIMIZE EI: MULTI-START L-BFGS-B ───────────────────────────────────────
rng = np.random.default_rng(args.seed)
lows  = np.array([b[0] for b in bounds])
highs = np.array([b[1] for b in bounds])

# ─── MAXIMIZE EI: MULTI-START L-BFGS-B ───────────────────────────────────────
rng = np.random.default_rng(args.seed)
lows  = np.array([b[0] for b in bounds])
highs = np.array([b[1] for b in bounds])

all_candidates = []   # (x, ei) from every restart, kept for --top_k reporting
for i in range(args.n_restarts):
    x0 = lows + rng.random(len(bounds)) * (highs - lows)
    res = minimize(neg_ei, x0, method='L-BFGS-B', bounds=bounds)
    ei_val = -res.fun
    all_candidates.append((res.x, ei_val))

# Sort all restart-converged points by EI descending. Restarts often converge
# to the same handful of local optima, so de-duplicate near-identical points
# (standardized distance < 0.05) before taking the top_k — otherwise top_k
# candidates can just be the same optimum found by several different restarts.
all_candidates.sort(key=lambda c: c[1], reverse=True)
deduped = []
for x_c, ei_c in all_candidates:
    x_c_s = scaler_X.transform(np.atleast_2d(x_c))
    is_dup = any(
        np.linalg.norm(x_c_s - scaler_X.transform(np.atleast_2d(x_d))) < 0.05
        for x_d, _ in deduped
    )
    if not is_dup:
        deduped.append((x_c, ei_c))

top_k = deduped[:args.top_k]
best_x, best_ei = top_k[0]

mu_rec, sigma_rec = predict_raw(best_x)
mu_rec, sigma_rec = mu_rec[0], sigma_rec[0]

print(f"\n{'='*60}")
print(f"  RECOMMENDATION")
print(f"{'='*60}")
for f, val in zip(FEATURES, best_x):
    print(f"  {f:<18} = {val:.4g}")
print(f"  Predicted SF_mean  = {mu_rec:.4f} ± {sigma_rec:.4f}")
print(f"  Expected Improvement (EI) = {best_ei:.5f}")
print(f"  Current best observed SF_mean = {f_best:.4f}")

if len(top_k) > 1:
    print(f"\n  Top {len(top_k)} distinct local EI maxima found "
          f"(from {args.n_restarts} restarts, {len(deduped)} distinct after "
          f"de-duplication):")
    print(f"  {'Rank':<5} {'Pressure':>10} {'Speed':>8} {'Zoffset':>9} "
          f"{'Pred.SF':>10} {'±std':>8} {'EI':>10}")
    for rank, (x_c, ei_c) in enumerate(top_k, start=1):
        mu_c, sigma_c = predict_raw(x_c)
        marker = '  <- chosen' if rank == 1 else ''
        print(f"  {rank:<5} {x_c[0]:>10.3g} {x_c[1]:>8.3g} {x_c[2]:>9.4g} "
              f"{mu_c[0]:>10.4f} {sigma_c[0]:>8.4f} {ei_c:>10.5f}{marker}")
if best_ei < 1e-5:
    print(f"\n  [NOTE] EI is essentially zero everywhere searched — the surrogate "
          f"believes further improvement within these bounds is unlikely. "
          f"Consider widening --pressure_bounds/--speed_bounds/--zoffset_bounds, "
          f"or this may indicate you're near a local/global optimum.")

# distance check: how close is the recommendation to an existing observed point?
dists = np.linalg.norm(scaler_X.transform(np.atleast_2d(best_x)) - X_s, axis=1)
nearest_dist = dists.min()
nearest_idx  = dists.argmin()
print(f"  Nearest already-observed point (standardized distance): {nearest_dist:.3f}  "
      f"(Sample_ID={df['Sample_ID'].iloc[nearest_idx] if 'Sample_ID' in df.columns else nearest_idx})")
if nearest_dist < 0.15:
    print(f"  [NOTE] Recommendation is very close to an already-printed point — "
          f"consider this a confirmatory replicate rather than new information.")

# ─── LOG TO CSV (append across runs) ─────────────────────────────────────────
log_path = os.path.join(args.outdir, 'bo_recommendation_log.csv')
iteration = 1
if os.path.exists(log_path):
    existing = pd.read_csv(log_path, sep=';')
    iteration = existing['iteration'].max() + 1

row = {
    'iteration': iteration,
    'Pressure_kPa':    round(best_x[0], 3),
    'NozzleSpeed_mms': round(best_x[1], 3),
    'Zoffset_mm':      round(best_x[2], 4),
    'predicted_SF_mean': round(mu_rec, 4),
    'predicted_SF_std':  round(sigma_rec, 4),
    'expected_improvement': round(best_ei, 5),
    'current_best_observed_SF': round(f_best, 4),
    'xi': args.xi,
    'n_training_samples': len(df),
}
df_row = pd.DataFrame([row])
if os.path.exists(log_path):
    df_row.to_csv(log_path, mode='a', header=False, index=False, sep=';')
else:
    df_row.to_csv(log_path, mode='w', header=True, index=False, sep=';')
print(f"\n  Recommendation logged -> {log_path}  (iteration {iteration})")

# ─── FIGURE: 1D SLICES THROUGH THE RECOMMENDED POINT ─────────────────────────
ps = PLOT_STYLE
fig, axes = plt.subplots(2, len(FEATURES), figsize=(7 * len(FEATURES), 11))
if args.title:
    fig.suptitle(f'BO Recommendation #{iteration} — 1D Slices Through Recommended Point\n'
                f'(other 2 parameters held fixed at the recommended value)',
                fontsize=ps['FONTSIZE_TITLE'], fontweight='bold', y=1.02)

for col, feat in enumerate(FEATURES):
    feat_idx = FEATURES.index(feat)
    lo, hi = bounds[feat_idx]
    grid = np.linspace(lo, hi, 200)

    X_slice = np.tile(best_x, (len(grid), 1))
    X_slice[:, feat_idx] = grid
    mu_slice, sigma_slice = predict_raw(X_slice)
    ei_slice = np.array([expected_improvement(x) for x in X_slice])

    ax0 = axes[0, col]
    ax0.plot(grid, mu_slice, color='#2E7D32', linewidth=2.5, label='Predicted SF')
    ax0.fill_between(grid, mu_slice - sigma_slice, mu_slice + sigma_slice,
                     color='#2E7D32', alpha=0.2, label='±1 std')
    ax0.axhline(f_best, color='black', linewidth=1.5, linestyle=':',
               label=f'Best observed={f_best:.3f}')
    ax0.scatter(df[feat].values, df['SF_mean'].values, color='black', s=40,
               alpha=0.5, zorder=4, label='Observed data')
    ax0.axvline(best_x[feat_idx], color='darkorange', linewidth=2.0, linestyle='--',
               label='Recommended')
    ax0.set_xlabel(FEATURE_LABELS[feat], fontsize=ps['FONTSIZE_LABEL'])
    ax0.set_ylabel('Predicted SF_mean', fontsize=ps['FONTSIZE_LABEL'])
    ax0.tick_params(labelsize=ps['FONTSIZE_TICK'])
    ax0.grid(alpha=0.25)
    if col == 0:
        ax0.legend(fontsize=ps['FONTSIZE_LEGEND'] - 4, loc='best')

    ax1 = axes[1, col]
    ax1.plot(grid, ei_slice, color='#C62828', linewidth=2.5)
    ax1.fill_between(grid, 0, ei_slice, color='#C62828', alpha=0.2)
    ax1.axvline(best_x[feat_idx], color='darkorange', linewidth=2.0, linestyle='--')
    ax1.set_xlabel(FEATURE_LABELS[feat], fontsize=ps['FONTSIZE_LABEL'])
    ax1.set_ylabel('Expected Improvement', fontsize=ps['FONTSIZE_LABEL'])
    ax1.tick_params(labelsize=ps['FONTSIZE_TICK'])
    ax1.grid(alpha=0.25)

plt.tight_layout()
fig_path = os.path.join(args.outdir, f'bo_recommendation_{iteration}.png')
plt.savefig(fig_path, dpi=ps['DPI'], bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

print(f"\n{'='*60}")
print(f"  BO recommendation complete.")
print(f"{'='*60}")