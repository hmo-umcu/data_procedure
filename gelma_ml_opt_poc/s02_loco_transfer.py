"""
s02_loco_transfer.py
--------------------
Protocol stages 2 and 3: retrospective transfer test with the held-out unit
being the CATEGORY, never the row.
 
You choose which categories are tested; everything else trains.
 
    C-F2 standard LOCO      --test_categories F
    C-F1 strict ablation    --test_categories F --exclude_from_train B
    full six-fold LOCO      --loco
 
Models run here (protocol 5.3 and 5.4):
 
    B0        matched-condition average of the training categories
    B1        P, Speed, Z                          -> SF     Ridge and GPR
    M_onset   P, Speed, Z, onset_kPa               -> SF     Ridge and GPR
    M_full    P, Speed, Z + all 4 fingerprint      -> SF     Ridge and GPR
 
The PC1 variant from protocol 5.4 item 6 is deliberately not implemented.
 
B2 (the target's own sweep curve) is NOT here: it needs the sweep tables rather
than the complete tables, so it lives in s03. Run s03 too, then s04 compares
everything. B2 is the baseline that decides whether the fingerprint model is
doing more than re-reading the target's own measurement, so do not skip it.
 
Scaling is fitted on training categories only, inside each split.
 
Usage
-----
    python s02_loco_transfer.py --data_dir <folder> --test_categories F
    python s02_loco_transfer.py --data_dir <folder> --loco
"""
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
 
import sf_data as S
 
COORD = S.PRINT_FEATURES
 
 
def coord_key(df):
    """One hashable key per LHS coordinate, so matching is by setting not row order."""
    return pd.Series(list(zip(*[df[c].round(6) for c in COORD])), index=df.index)
 
 
def predict_b0(train_df, test_df):
    """Matched-condition average: mean training SF at the same LHS coordinate."""
    g = train_df.groupby(coord_key(train_df).values)[S.TARGET].mean().to_dict()
    fallback = float(train_df[S.TARGET].mean())
    return np.array([g.get(k, fallback) for k in coord_key(test_df)], float)
 
 
 
 
def fit_predict(kind, feats, train_df, test_df, seed, n_restarts, noise_mode):
    Xtr = train_df[feats].values.astype(float)
    Xte = test_df[feats].values.astype(float)
    ytr = train_df[S.TARGET].values.astype(float)
 
    sx = StandardScaler().fit(Xtr)          # fitted on TRAINING categories only
    Xtr_s, Xte_s = sx.transform(Xtr), sx.transform(Xte)
 
    alpha = None
    if kind == 'GPR':
        alpha = S.alpha_from_std(train_df, noise_mode)
    if kind == 'GPR':
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(Xtr_s.shape[1]),
                           length_scale_bounds=(1e-2, 1e3), nu=2.5)
                  + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1)))
        m = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                     n_restarts_optimizer=n_restarts,
                                     alpha=alpha if alpha is not None else 1e-10,
                                     random_state=seed)
    else:
        m = Ridge(alpha=1.0)
    m.fit(Xtr_s, ytr)
    if kind == 'GPR':
        mu, sd = m.predict(Xte_s, return_std=True)
        return mu, sd, m
    return m.predict(Xte_s), np.full(len(Xte_s), np.nan), m
 
 
def run_split(df, train_cats, test_cats, args, excl_cats=()):
    tr = df[df['category'].isin(train_cats)].copy()
    te = df[df['category'].isin(test_cats)].copy()
    # raises if anything held out reached the frame that gets fitted
    S.audit_split(df, train_cats, test_cats, excl_cats, fit_frame=tr,
                  verbose=args.audit)
 
    variants = [('B1', S.PRINT_FEATURES),
                ('M_onset', S.PRINT_FEATURES + S.FINGERPRINT_ONSET),
                ('M_full', S.PRINT_FEATURES + S.FINGERPRINT_ALL)]
    have = set(tr.columns)
    variants = [(n, f) for n, f in variants if set(f) <= have]
 
    preds = {'B0': (predict_b0(tr, te), np.full(len(te), np.nan))}
    for vname, feats in variants:
        for kind in ('Ridge', 'GPR'):
            mu, sd, _ = fit_predict(kind, feats, tr, te, args.seed,
                                    args.n_restarts, args.noise_mode)
            preds[f'{vname}:{kind}'] = (mu, sd)
 
    rows, mets = [], []
    y = te[S.TARGET].values.astype(float)
    for name, (mu, sd) in preds.items():
        m = S.metrics(y, mu)
        m['model'] = name
        m['test_categories'] = ','.join(S.NAME_TO_LETTER.get(c, c) for c in test_cats)
        m['n_train_categories'] = len(train_cats)
        mets.append(m)
        for i in range(len(te)):
            rows.append({
                'model': name,
                'test_category': te['category'].iloc[i],
                'Sample': te['Sample'].iloc[i],
                **{c: te[c].iloc[i] for c in COORD},
                'label_SF_mean': round(float(y[i]), 6),
                'pred_SF_mean': round(float(mu[i]), 6),
                'pred_std': ('' if not np.isfinite(sd[i]) else round(float(sd[i]), 6)),
                'residual': round(float(mu[i] - y[i]), 6),
                'abs_error': round(float(abs(mu[i] - y[i])), 6),
            })
    return rows, mets
 
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = S.load_categories(a.data_dir)
    cats = sorted(df['category'].unique(), key=lambda c: S.NAME_TO_LETTER.get(c, 'Z'))
 
    excl_all = []
    if a.loco:
        splits = [([c for c in cats if c != t], [t]) for t in cats]
        tag = 'loco'
    else:
        if not a.test_categories:
            raise SystemExit('Give --test_categories (e.g. F, or "B,F"), '
                             'or use --loco for all six folds.')
        train, test, excl = S.split_train_test(df, a.test_categories,
                                               a.exclude_from_train)
        splits = [(train, test)]
        excl_all = excl
        tag = 'test-' + '-'.join(S.NAME_TO_LETTER.get(c, c) for c in test)
        if excl:
            tag += '_excl-' + '-'.join(S.NAME_TO_LETTER.get(c, c) for c in excl)
 
    print('=' * 74)
    print(f'  RETROSPECTIVE TRANSFER   ({tag})')
    print('=' * 74)
    print(f'  GPR noise mode: {a.noise_mode}'
          + ('  (SF_std^2/6, variance of the mean of six images)'
             if a.noise_mode == 'mean_of_6' else ''))
 
    all_rows, all_mets = [], []
    for train_cats, test_cats in splits:
        print(f'\n-- held out: {", ".join(S.label(c) for c in test_cats)}')
        print(f'   training on {len(train_cats)}: '
              f'{", ".join(S.NAME_TO_LETTER.get(c, c) for c in train_cats)}')
        rows, mets = run_split(df, train_cats, test_cats, a,
                               excl_cats=excl_all)
        all_rows += rows
        all_mets += mets
        S.print_metrics_table(mets)
 
    pd.DataFrame(all_rows).to_csv(outdir / f'predictions_{tag}.csv', sep=';', index=False)
    md = pd.DataFrame(all_mets)
    md.to_csv(outdir / f'metrics_{tag}.csv', sep=';', index=False)
 
    if len(splits) > 1:
        print('\n' + '=' * 74)
        print('  MEAN ACROSS FOLDS')
        print('=' * 74)
        agg = (md.groupby('model')[['RMSE', 'MAE', 'R2']]
                 .agg(['mean', 'std']).round(4))
        print(agg.to_string())
        agg.to_csv(outdir / f'metrics_{tag}_aggregate.csv', sep=';')
 
    print(f'\nWrote -> {outdir}')
    print('\nNext: run s03_baseline_sweep_only.py, then s04_compare_models.py.')
    print('Without B2 you cannot tell a transfer result from re-reading the '
          'target sweep.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stage 2/3: category-held-out transfer.')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--outdir', default='results/02_transfer')
    ap.add_argument('--test_categories', default=None,
                    help='Held-out categories, letters or names, comma separated')
    ap.add_argument('--exclude_from_train', default=None,
                    help='Also withheld from training but not evaluated '
                         '(C-F1 uses B here while testing on F)')
    ap.add_argument('--loco', action='store_true',
                    help='Ignore --test_categories and run every category as a fold')
    ap.add_argument('--noise_mode', default='mean_of_6',
                    choices=['mean_of_6', 'raw', 'none'],
                    help='Per-point GPR noise from SF_std (default: mean_of_6)')
    ap.add_argument('--no_audit', dest='audit', action='store_false',
                    help='Hide the per-split training-set audit table '
                         '(the leak assertion still runs)')
    ap.add_argument('--n_restarts', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args())