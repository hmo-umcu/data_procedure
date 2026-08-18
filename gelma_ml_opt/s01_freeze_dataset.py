"""
s01_freeze_dataset.py
---------------------
Protocol stage 1: data freeze.
 
Reads every <category>_sf_complete_48well.csv, checks the design, writes
combined_6category_table.csv, and freezes the benchmark numbers for the target
category BEFORE any model touches them.
 
Run this first and read the output. If it reports a problem, later steps will
produce numbers that look fine and are not.
 
Usage
-----
    python s01_freeze_dataset.py --data_dir <folder> --outdir results/01_freeze \
        [--target F]
 
Outputs
-------
    combined_6category_table.csv   one row per category x LHS condition
    category_manifest.csv          per-category provenance and fingerprint
    target_benchmark.csv           frozen mean/median/best for the target
"""
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import sf_data as S
 
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = S.load_categories(a.data_dir)
 
    print('=' * 74)
    print('  DATA FREEZE')
    print('=' * 74)
    ok, common = S.check_design(df)
 
    # ---------------------------------------------------------------- manifest
    rows = []
    for c in sorted(df['category'].unique(), key=lambda x: S.NAME_TO_LETTER.get(x, 'Z')):
        sub = df[df['category'] == c]
        L = S.NAME_TO_LETTER.get(c, '?')
        conc, dof, cell, temp = S.META.get(L, (None,) * 4)
        r = {'letter': L, 'category': c, 'conc_pct': conc, 'DoF': dof,
             'cell_state': cell, 'print_temp_C': temp, 'n_conditions': len(sub),
             'SF_mean_mean': round(sub[S.TARGET].mean(), 4),
             'SF_mean_median': round(sub[S.TARGET].median(), 4),
             'SF_mean_best': round(sub[S.TARGET].max(), 4)}
        for f in S.FINGERPRINT_ALL:
            if f in sub.columns:
                r[f] = sub[f].iloc[0]
        if S.TARGET_STD in sub.columns:
            r['SF_std_median'] = round(sub[S.TARGET_STD].median(), 4)
        rows.append(r)
    man = pd.DataFrame(rows)
    man.to_csv(outdir / 'category_manifest.csv', sep=';', index=False)
 
    print('\n-- category manifest --')
    print(man[['letter', 'category', 'conc_pct', 'DoF', 'cell_state',
               'n_conditions', 'SF_mean_mean', 'SF_mean_best']].to_string(index=False))
 
    # ------------------------------------------------ pairs available for Aim 2
    present = set(df['category'].unique())
    print('\n-- Aim 2 matched pairs --')
    for f, l in S.PAIRS:
        fn, ln = S.LETTER_TO_NAME[f], S.LETTER_TO_NAME[l]
        have = fn in present and ln in present
        print(f'  {f}/{l}  {fn:<18} vs {ln:<18} '
              f'{"available" if have else "INCOMPLETE"}')
 
    # ------------------------------------------------------- target benchmark
    tgt = S.resolve(a.target)
    if tgt not in present:
        print(f'\n[WARN] target {S.label(tgt)} is not in the data; '
              f'benchmark skipped.')
    else:
        sub = df[df['category'] == tgt].copy()
        k = sub[S.TARGET].idxmax()
        best = sub.loc[k]
        ref = sub[(sub['Pressure_kPa'] == 90) & (sub['NozzleSpeed_mms'] == 10)
                  & (np.isclose(sub['Zoffset_mm'], 0.2))]
        bench = [
            {'reference': 'mean across LHS conditions',
             'value': round(sub[S.TARGET].mean(), 4), 'params': '',
             'use': 'secondary descriptive'},
            {'reference': 'median across LHS conditions',
             'value': round(sub[S.TARGET].median(), 4), 'params': '',
             'use': 'secondary robust'},
            {'reference': 'best observed LHS SF',
             'value': round(float(best[S.TARGET]), 4),
             'params': f'{best["Pressure_kPa"]:g} kPa; '
                       f'{best["NozzleSpeed_mms"]:g} mm/s; Z={best["Zoffset_mm"]:g} mm',
             'use': 'PRIMARY optimization baseline, reprint fresh'},
        ]
        if len(ref):
            bench.append({'reference': 'standard matched reference 90/10/0.2',
                          'value': round(float(ref[S.TARGET].iloc[0]), 4),
                          'params': '90 kPa; 10 mm/s; Z=0.2 mm',
                          'use': 'batch consistency check'})
        bdf = pd.DataFrame(bench)
        bdf.insert(0, 'category', tgt)
        bdf.to_csv(outdir / 'target_benchmark.csv', sep=';', index=False)
        print(f'\n-- frozen benchmark for target {S.label(tgt)} --')
        print(bdf.to_string(index=False))
        print('\n  The primary baseline is the BEST observed LHS condition, '
              'reprinted fresh in the\n  validation batch. The 32-point mean is '
              'context only, not the bar to beat.')
 
        # the protocol's warning: is the historical best at the domain edge?
        if float(best['Pressure_kPa']) >= sub['Pressure_kPa'].max():
            print(f'\n  [NOTE] The best observed condition sits at the maximum '
                  f'sampled pressure\n  ({best["Pressure_kPa"]:g} kPa). Any '
                  f'recommendation is the best inside the tested box,\n  not a '
                  f'physical optimum. Say so when reporting.')
 
    df.to_csv(outdir / 'combined_6category_table.csv', sep=';', index=False)
    print(f'\nWrote {len(df)} rows -> {outdir / "combined_6category_table.csv"}')
    if not ok:
        print('\n[ACTION] The warnings above affect cross-category validity. '
              'Resolve them before running s02.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stage 1: freeze the dataset.')
    ap.add_argument('--data_dir', required=True,
                    help='Folder with the *_sf_complete_48well.csv files')
    ap.add_argument('--outdir', default='results/01_freeze')
    ap.add_argument('--target', default='F',
                    help='Headline target category, letter or name (default: F)')
    main(ap.parse_args())