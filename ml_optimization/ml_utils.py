"""
ml_utils.py
===========
Shared preprocessing, evaluation, and plotting utilities used by all three
model scripts (Ridge, GPR, NGBoost) for predicting SF_mean (shape fidelity)
from printing parameters.

Adapted from the BioRT/ExplorerONE calibration ML pipeline, simplified for:
    - 3 continuous inputs only (Pressure_kPa, NozzleSpeed_mms, Zoffset_mm)
      — no categorical material/needle variables
    - single continuous output (SF_mean) — no MultiOutputRegressor wrapper
    - small N (~30 samples) — stratified CV uses quantile bins of SF_mean
      itself (since there's no categorical variable to stratify by)

Import with:
    from ml_utils import (load_and_preprocess, run_cv, aggregate_metrics,
                          print_summary, apply_plot_style,
                          plot_predicted_vs_actual, plot_residuals,
                          plot_cv_metrics, plot_error_per_sample,
                          plot_overfitting, FEATURES, FEATURE_LABELS,
                          FEATURE_UNITS, OUTPUT_NAME, OUTPUT_LABEL,
                          OUTPUT_UNIT, PLOT_STYLE)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
FEATURES = ['Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
FEATURE_LABELS = {
    'Pressure_kPa':    'Pressure  (kPa)',
    'NozzleSpeed_mms': 'Nozzle Speed  (mm/s)',
    'Zoffset_mm':      'Z-offset  (mm)',
}
FEATURE_SHORT = {
    'Pressure_kPa':    'Pressure\n(kPa)',
    'NozzleSpeed_mms': 'Speed\n(mm/s)',
    'Zoffset_mm':      'Z-offset\n(mm)',
}
FEATURE_UNITS = {
    'Pressure_kPa':    'kPa',
    'NozzleSpeed_mms': 'mm/s',
    'Zoffset_mm':      'mm',
}
OUTPUT_NAME  = 'SF_mean'
OUTPUT_LABEL = 'Shape Fidelity  (SF)'
OUTPUT_UNIT  = ''   # SF is dimensionless

POINT_COLOR = '#1565C0'

# ─── PLOT STYLE ───────────────────────────────────────────────────────────────
PLOT_STYLE = {
    'FONTSIZE_TITLE':      22,
    'FONTSIZE_LABEL':      20,
    'FONTSIZE_TICK':       20,
    'FONTSIZE_LEGEND':     18,
    'FONTSIZE_ANNOTATION': 18,
    'LINE_WIDTH':          2.0,
    'DPI':                 200,
}

def apply_plot_style():
    ps = PLOT_STYLE
    plt.rcParams.update({
        'font.size':         ps['FONTSIZE_TICK'],
        'axes.titlesize':    ps['FONTSIZE_TITLE'],
        'axes.labelsize':    ps['FONTSIZE_LABEL'],
        'xtick.labelsize':   ps['FONTSIZE_TICK'],
        'ytick.labelsize':   ps['FONTSIZE_TICK'],
        'legend.fontsize':   ps['FONTSIZE_LEGEND'],
        'axes.linewidth':    ps['LINE_WIDTH'],
        'xtick.major.width': ps['LINE_WIDTH'],
        'ytick.major.width': ps['LINE_WIDTH'],
        'xtick.major.size':  7,
        'ytick.major.size':  7,
        'figure.dpi':        ps['DPI'],
    })

# ─── LOAD AND PREPROCESS ──────────────────────────────────────────────────────
def load_and_preprocess(data_path, n_strat_bins=4):
    """
    Load CSV (semicolon-delimited), drop rows missing SF_mean or any feature.
    Builds quantile bins of SF_mean for stratified CV, since there's no
    categorical variable (material/needle) to stratify by here — without
    this, random folds at N~30 can easily land all the high- or low-SF
    samples in one fold, skewing per-fold metrics.

    Returns:
        df           — filtered DataFrame
        X            — feature matrix (np.ndarray), shape (N, 3)
        y            — target vector (np.ndarray), shape (N,)
        strat_label  — array of quantile-bin labels for stratification
        feature_cols — list of feature column names (== FEATURES)
    """
    df_raw = pd.read_csv(data_path, sep=';')
    needed = FEATURES + [OUTPUT_NAME]
    df = df_raw.dropna(subset=needed).copy().reset_index(drop=True)

    print(f"Total rows: {len(df_raw)}  |  Usable (non-missing): {len(df)}")
    if 'Sample_ID' not in df.columns:
        df['Sample_ID'] = df.index

    X = df[FEATURES].values.astype(float)
    y = df[OUTPUT_NAME].values.astype(float)

    # Quantile-bin SF_mean for stratified k-fold (small-N CV best practice
    # when there's no categorical variable available to stratify by)
    n_bins = min(n_strat_bins, len(df) // 4) if len(df) >= 8 else 2
    n_bins = max(n_bins, 2)
    try:
        strat_label = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        strat_label = np.zeros(len(y), dtype=int)

    print(f"Feature matrix X: {X.shape}  (features: {FEATURES})")
    print(f"Target vector y:  {y.shape}  (target: {OUTPUT_NAME})")
    print(f"Stratification bins (quantiles of {OUTPUT_NAME}): "
          f"{len(np.unique(strat_label))}\n")

    return df, X, y, strat_label, FEATURES

# ─── CROSS-VALIDATION LOOP ────────────────────────────────────────────────────
def run_cv(model, X, y, strat_label, df, feature_cols, n_folds=5, seed=42):
    """
    Run stratified k-fold CV (stratified on quantile bins of y, see
    load_and_preprocess). Single-output version — no MultiOutputRegressor
    reshaping needed.

    Returns:
        val_rows     — list of per-sample prediction log dicts (validation only)
        train_rows   — list of per-sample prediction log dicts (training)
        fold_metrics — list of per-fold metric dicts
    """
    n_folds = min(n_folds, int(np.min(np.bincount(strat_label.astype(int)))))
    if n_folds < 2:
        raise ValueError(
            f"Cannot run CV: smallest stratification bin has fewer than 2 "
            f"samples. Reduce --n_strat_bins or use plain (non-stratified) "
            f"KFold for this dataset size.")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    val_rows, train_rows, fold_metrics = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, strat_label)):
        print(f"  Fold {fold_idx + 1}/{n_folds}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler_X = StandardScaler()
        X_tr_s   = scaler_X.fit_transform(X_tr)
        X_val_s  = scaler_X.transform(X_val)

        scaler_y = StandardScaler()
        y_tr_s   = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

        model.fit(X_tr_s, y_tr_s)

        y_val_pred_s = np.asarray(model.predict(X_val_s)).reshape(-1)
        y_val_pred   = scaler_y.inverse_transform(
            y_val_pred_s.reshape(-1, 1)).ravel()

        y_tr_pred_s = np.asarray(model.predict(X_tr_s)).reshape(-1)
        y_tr_pred   = scaler_y.inverse_transform(
            y_tr_pred_s.reshape(-1, 1)).ravel()

        for i, orig_idx in enumerate(val_idx):
            row = {
                'sample_id': df['Sample_ID'].iloc[orig_idx],
                'fold': fold_idx + 1, 'split': 'val',
            }
            for fc in feature_cols:
                row[fc] = df[fc].iloc[orig_idx]
            row['label']     = round(y_val[i], 6)
            row['pred']      = round(y_val_pred[i], 6)
            row['residual']  = round(y_val_pred[i] - y_val[i], 6)
            row['abs_error'] = round(abs(y_val_pred[i] - y_val[i]), 6)
            val_rows.append(row)

        for i, orig_idx in enumerate(train_idx):
            row = {
                'sample_id': df['Sample_ID'].iloc[orig_idx],
                'fold': fold_idx + 1, 'split': 'train',
            }
            for fc in feature_cols:
                row[fc] = df[fc].iloc[orig_idx]
            row['label']     = round(y_tr[i], 6)
            row['pred']      = round(y_tr_pred[i], 6)
            row['residual']  = round(y_tr_pred[i] - y_tr[i], 6)
            row['abs_error'] = round(abs(y_tr_pred[i] - y_tr[i]), 6)
            train_rows.append(row)

        mask = np.abs(y_val) > 1e-8
        mape = (np.mean(np.abs((y_val[mask] - y_val_pred[mask]) / y_val[mask])) * 100
               if mask.sum() > 0 else np.nan)
        fold_metrics.append({
            'fold': fold_idx + 1,
            'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'MAE':  mean_absolute_error(y_val, y_val_pred),
            'R2':   r2_score(y_val, y_val_pred),
            'MAPE': mape,
        })

    return val_rows, train_rows, fold_metrics

# ─── AGGREGATE FOLD METRICS ───────────────────────────────────────────────────
def aggregate_metrics(fold_metrics):
    df_folds = pd.DataFrame(fold_metrics)
    rows = []
    for metric in ['RMSE', 'MAE', 'R2', 'MAPE']:
        vals = df_folds[metric].dropna()
        rows.append({
            'Metric': metric,
            'Mean': round(vals.mean(), 5), 'Std': round(vals.std(), 5),
            'Min':  round(vals.min(),  5), 'Max': round(vals.max(), 5),
        })
    return pd.DataFrame(rows)

def print_summary(model_name, df_agg):
    print(f"\n{'='*60}")
    print(f"  {model_name} — CV RESULTS (mean ± std across folds)")
    print(f"{'='*60}")
    sub = df_agg.set_index('Metric')
    print(f"  RMSE = {sub.loc['RMSE','Mean']:.4f} ± {sub.loc['RMSE','Std']:.4f}")
    print(f"  MAE  = {sub.loc['MAE','Mean']:.4f} ± {sub.loc['MAE','Std']:.4f}")
    print(f"  R²   = {sub.loc['R2','Mean']:.4f} ± {sub.loc['R2','Std']:.4f}")
    print(f"  MAPE = {sub.loc['MAPE','Mean']:.2f}% ± {sub.loc['MAPE','Std']:.2f}%")

# ─── FIGURE: PREDICTED vs ACTUAL ──────────────────────────────────────────────
def plot_predicted_vs_actual(val_rows, model_name, model_color,
                             outdir, show_title=True):
    ps = PLOT_STYLE
    df_val = pd.DataFrame(val_rows)
    y_true, y_pred = df_val['label'].values, df_val['pred'].values

    fig, ax = plt.subplots(figsize=(9, 9))
    if show_title:
        ax.set_title(f'{model_name} — Predicted vs Actual SF  (CV validation folds)',
                     fontsize=ps['FONTSIZE_TITLE'], fontweight='bold', pad=14)

    sc = ax.scatter(y_true, y_pred, c=df_val['Zoffset_mm'].values,
                    cmap='viridis', s=140, alpha=0.85,
                    edgecolor='white', linewidth=0.8)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label('Z-offset (mm)', fontsize=ps['FONTSIZE_LABEL'] - 2)
    cbar.ax.tick_params(labelsize=ps['FONTSIZE_TICK'] - 2)

    all_vals = np.concatenate([y_true, y_pred])
    margin = (all_vals.max() - all_vals.min()) * 0.08
    lims   = [all_vals.min() - margin, all_vals.max() + margin]
    ax.plot(lims, lims, 'k--', linewidth=2.2, alpha=0.6, label='Perfect prediction')
    ax.set_xlim(lims); ax.set_ylim(lims)

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    ax.text(0.04, 0.97, f'R² = {r2:.3f}\nRMSE = {rmse:.4f}\n'
                        f'MAE  = {mae:.4f}\nMAPE = {mape:.1f}%',
            transform=ax.transAxes, fontsize=ps['FONTSIZE_ANNOTATION'],
            va='top', bbox=dict(boxstyle='round,pad=0.35', fc='white',
                                alpha=0.9, ec='gray', linewidth=1.2))

    ax.set_xlabel(f'Actual {OUTPUT_LABEL}', fontsize=ps['FONTSIZE_LABEL'])
    ax.set_ylabel(f'Predicted {OUTPUT_LABEL}', fontsize=ps['FONTSIZE_LABEL'])
    ax.tick_params(labelsize=ps['FONTSIZE_TICK'])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=ps['FONTSIZE_LEGEND'], loc='lower right')

    plt.tight_layout()
    path = os.path.join(outdir, 'predicted_vs_actual.png')
    plt.savefig(path, dpi=ps['DPI'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# ─── FIGURE: RESIDUALS ────────────────────────────────────────────────────────
def plot_residuals(val_rows, model_name, model_color, outdir, show_title=True):
    ps = PLOT_STYLE
    df_val = pd.DataFrame(val_rows)
    y_pred, residuals = df_val['pred'].values, df_val['residual'].values
    bias, res_std = residuals.mean(), residuals.std()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    if show_title:
        fig.suptitle(f'{model_name} — Residual Analysis  (CV validation folds)',
                     fontsize=ps['FONTSIZE_TITLE'] + 2, fontweight='bold', y=1.02)

    ax0 = axes[0]
    ax0.scatter(y_pred, residuals, c=model_color, s=85, alpha=0.78,
               edgecolor='white', linewidth=0.6)
    ax0.axhline(0, color='black', linewidth=2.5, linestyle='--')
    ax0.axhline(bias, color='darkorange', linewidth=2.0, linestyle=':',
               label=f'Bias={bias:.4f}')
    ax0.axhline(bias + res_std, color='gray', linewidth=1.2, linestyle=':', alpha=0.6)
    ax0.axhline(bias - res_std, color='gray', linewidth=1.2, linestyle=':', alpha=0.6)
    ax0.set_xlabel(f'Predicted {OUTPUT_LABEL}', fontsize=ps['FONTSIZE_LABEL'])
    ax0.set_ylabel('Residual', fontsize=ps['FONTSIZE_LABEL'])
    ax0.tick_params(labelsize=ps['FONTSIZE_TICK'])
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=ps['FONTSIZE_LEGEND'] - 1)
    ax0.text(0.97, 0.97, f'Bias={bias:.4f}\nStd={res_std:.4f}',
            transform=ax0.transAxes, fontsize=ps['FONTSIZE_ANNOTATION'],
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                            alpha=0.88, ec='gray'))

    ax1 = axes[1]
    n_bins = max(6, min(12, len(residuals) // 2))
    ax1.hist(residuals, bins=n_bins, color=model_color, alpha=0.78,
             edgecolor='white', linewidth=0.8)
    ax1.axvline(0,    color='black',      linewidth=2.5, linestyle='--')
    ax1.axvline(bias, color='darkorange', linewidth=2.0, linestyle=':')
    if res_std > 1e-9:
        x_fit = np.linspace(residuals.min(), residuals.max(), 200)
        y_fit = stats.norm.pdf(x_fit, bias, res_std)
        bin_w = (residuals.max() - residuals.min()) / n_bins
        ax1.plot(x_fit, y_fit * len(residuals) * bin_w,
                color='black', linewidth=2.5, label='Normal fit')
    ax1.set_xlabel('Residual', fontsize=ps['FONTSIZE_LABEL'])
    ax1.set_ylabel('Count', fontsize=ps['FONTSIZE_LABEL'])
    ax1.tick_params(labelsize=ps['FONTSIZE_TICK'])
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=ps['FONTSIZE_LEGEND'] - 1)

    plt.tight_layout()
    path = os.path.join(outdir, 'residual_analysis.png')
    plt.savefig(path, dpi=ps['DPI'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# ─── FIGURE: CV METRICS BAR CHART ────────────────────────────────────────────
def plot_cv_metrics(fold_metrics, model_name, model_color, outdir, show_title=True):
    ps = PLOT_STYLE
    df_folds = pd.DataFrame(fold_metrics)
    metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
    metric_labels = {'RMSE': 'RMSE', 'MAE': 'MAE', 'R2': 'R²', 'MAPE': 'MAPE (%)'}

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    if show_title:
        fig.suptitle(f'{model_name} — CV Metrics per Fold',
                     fontsize=ps['FONTSIZE_TITLE'] + 2, fontweight='bold', y=1.05)

    fold_ids = df_folds['fold'].values
    x = np.arange(len(fold_ids))

    for col, metric in enumerate(metrics):
        ax = axes[col]
        vals = df_folds[metric].values
        mean_ = np.nanmean(vals)
        ax.bar(x, vals, 0.6, color=model_color, alpha=0.78,
               edgecolor='black', linewidth=1.5)
        ax.axhline(mean_, color='black', linewidth=2.0, linestyle='--',
                  label=f'Mean={mean_:.3f}')
        if metric == 'R2':
            ax.axhline(1.0, color='gray', linewidth=1.5, linestyle=':', alpha=0.5)
            ax.set_ylim(bottom=min(-0.1, np.nanmin(vals) - 0.1))
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {f}' for f in fold_ids], fontsize=ps['FONTSIZE_TICK'])
        ax.tick_params(labelsize=ps['FONTSIZE_TICK'])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=ps['FONTSIZE_LEGEND'] - 2)
        ax.set_ylabel(metric_labels[metric], fontsize=ps['FONTSIZE_LABEL'])

    plt.tight_layout()
    path = os.path.join(outdir, 'cv_metrics_per_fold.png')
    plt.savefig(path, dpi=ps['DPI'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# ─── FIGURE: ERROR PER SAMPLE ─────────────────────────────────────────────────
def plot_error_per_sample(val_rows, model_name, model_color, outdir, show_title=True):
    ps = PLOT_STYLE
    df_val = pd.DataFrame(val_rows).sort_values('sample_id').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(14, len(df_val) * 0.45), 7))
    if show_title:
        ax.set_title(f'{model_name} — Absolute Error per Sample  (CV validation folds)',
                     fontsize=ps['FONTSIZE_TITLE'], fontweight='bold', pad=14)

    abs_err = df_val['abs_error'].values
    mean_e  = abs_err.mean()
    x_pos   = np.arange(len(df_val))

    sc = ax.bar(x_pos, abs_err, color=model_color, alpha=0.78,
               edgecolor='black', linewidth=0.8)
    ax.axhline(mean_e, color='black', linewidth=2.2, linestyle='--',
              label=f'Mean abs error = {mean_e:.4f}')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_val['sample_id'].values, rotation=45, ha='right',
                       fontsize=max(8, ps['FONTSIZE_TICK'] - 6))
    ax.set_xlabel('Sample ID', fontsize=ps['FONTSIZE_LABEL'])
    ax.set_ylabel('Absolute Error (SF)', fontsize=ps['FONTSIZE_LABEL'])
    ax.tick_params(axis='y', labelsize=ps['FONTSIZE_TICK'])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=ps['FONTSIZE_LEGEND'])

    plt.tight_layout()
    path = os.path.join(outdir, 'error_per_sample.png')
    plt.savefig(path, dpi=ps['DPI'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# ─── FIGURE: TRAIN vs VAL R² OVERFITTING CHECK ────────────────────────────────
def plot_overfitting(val_rows, train_rows, fold_metrics,
                     model_name, model_color, outdir, show_title=True):
    ps = PLOT_STYLE
    df_folds = pd.DataFrame(fold_metrics)
    df_train = pd.DataFrame(train_rows)

    fig, ax = plt.subplots(figsize=(10, 7))
    if show_title:
        ax.set_title(f'{model_name} — Overfitting Check: Train vs Validation R²',
                     fontsize=ps['FONTSIZE_TITLE'], fontweight='bold', pad=14)

    fold_ids = df_folds['fold'].values
    x, w = np.arange(len(fold_ids)), 0.35
    val_r2 = df_folds['R2'].values

    train_r2_per_fold = []
    for fold_id in fold_ids:
        sub = df_train[df_train['fold'] == fold_id]
        yt, yp = sub['label'].values, sub['pred'].values
        train_r2_per_fold.append(r2_score(yt, yp) if len(yt) > 1 else np.nan)
    train_r2 = np.array(train_r2_per_fold)

    ax.bar(x - w/2, train_r2, w, label='Train', color='#455A64', alpha=0.75,
          edgecolor='black', linewidth=1.5)
    ax.bar(x + w/2, val_r2, w, label='Validation', color=model_color, alpha=0.82,
          edgecolor='black', linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in fold_ids], fontsize=ps['FONTSIZE_TICK'])
    ax.axhline(1.0, color='gray', linewidth=1.5, linestyle=':', alpha=0.5)
    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='-', alpha=0.3)
    ax.set_ylim(bottom=min(-0.3, min(val_r2.min(), train_r2.min()) - 0.15))
    ax.set_ylabel('R²', fontsize=ps['FONTSIZE_LABEL'])
    ax.tick_params(labelsize=ps['FONTSIZE_TICK'])
    ax.grid(axis='y', alpha=0.3)

    gap = np.mean(train_r2 - val_r2)
    ax.text(0.03, 0.04, f'Mean gap = {gap:.3f}',
            transform=ax.transAxes, fontsize=ps['FONTSIZE_ANNOTATION'],
            va='bottom', color='darkred' if gap > 0.2 else 'black',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='gray'))
    ax.legend(fontsize=ps['FONTSIZE_LEGEND'])

    plt.tight_layout()
    path = os.path.join(outdir, 'overfitting_check.png')
    plt.savefig(path, dpi=ps['DPI'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
