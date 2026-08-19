"""
build_complete_table.py
------------------------
Join the per-category pressure-sweep fingerprint onto that category's 48-well
SF table, producing the model-ready design matrix.
 
For each row of fingerprints.csv it loads <category>_sf_summary_48well.csv,
appends the selected fingerprint columns (identical on every row of that file,
because a fingerprint describes the material, not the individual print), and
writes <category>_sf_complete_48well.csv.
 
Resulting columns:
    Sample_ID, Pressure_kPa, NozzleSpeed_mms, Zoffset_mm, fold, n_images,
    SF_mean, SF_std, category, <selected fingerprint features...>
 
`category` is added as well as the features. The filename already carries it,
but having it as a column is what makes the optional --combined_csv usable for
leave-one-category-out grouping without re-parsing filenames.
 
Choosing features
-----------------
Default: onset_kPa, rise_slope, peak_sf, sf_at_p_max.
 
Three of the plan's six descriptors are unusable as features on this dataset,
and the script refuses them rather than emitting blanks:
 
    peak_kPa        blank in 3/6 categories
    window_kPa      blank in 3/6
    collapse_slope  blank in 4/6
 
They are blank because those sweeps never peaked inside 30-120 kPa, so the
quantity does not exist there. Any feature with a blank in even one category
would put an empty cell into the design matrix, so this script fails loudly
with the list of offending categories instead of writing it.
 
Two things to keep in mind about the result
--------------------------------------------
1. The fingerprint columns are CONSTANT within a category. They carry no
   within-category signal at all; their entire job is to tell categories apart.
   A model fitted on one category alone gains nothing from them.
 
2. In fingerprint space there are only as many distinct points as there are
   categories (6 here), whatever the row count of the joined table. A
   leave-one-category-out fit therefore extrapolates from 5 points in that
   space. The row count (192 per category) makes the table look larger than
   the evidence for the transfer claim actually is.
 
Usage
-----
    python build_complete_table.py \
        --fingerprints_csv pressure_sweep_c6_fingerprints.csv \
        --data_dir <folder with the *_sf_summary_48well.csv files> \
        [--output_dir <folder>] \
        [--features onset_kPa,rise_slope,peak_sf,sf_at_p_max] \
        [--combined_csv all_categories_complete.csv]
"""
 
import argparse
import csv
import itertools
from pathlib import Path
 
DEFAULT_FEATURES = ['onset_kPa', 'rise_slope', 'peak_sf', 'sf_at_p_max']
 
 
def sniff(path):
    with open(path, newline='') as f:
        head = f.readline()
    return ';' if head.count(';') >= head.count(',') else ','
 
 
def load(path):
    d = sniff(path)
    with open(path, newline='') as f:
        r = csv.DictReader(f, delimiter=d)
        return list(r), r.fieldnames
 
 
def main(args):
    fp_path = Path(args.fingerprints_csv)
    data_dir = Path(args.data_dir) if args.data_dir else fp_path.parent
    out_dir = Path(args.output_dir) if args.output_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
 
    fp_rows, fp_fields = load(fp_path)
    if not fp_rows:
        raise SystemExit(f'{fp_path} has no rows.')
    if 'category' not in fp_fields:
        raise SystemExit(f'{fp_path} has no "category" column. Found: {fp_fields}')
 
    features = [f.strip() for f in args.features.split(',') if f.strip()]
    unknown = [f for f in features if f not in fp_fields]
    if unknown:
        raise SystemExit(f'Unknown feature(s): {unknown}\n'
                         f'Columns available: '
                         f'{[c for c in fp_fields if c != "category"]}')
 
    # --- refuse blanks and non-numerics before writing anything --------------
    problems = {}
    for f in features:
        bad = [r['category'] for r in fp_rows if not str(r.get(f, '')).strip()]
        if bad:
            problems[f] = ('blank', bad)
            continue
        nan = []
        for r in fp_rows:
            try:
                float(r[f])
            except ValueError:
                nan.append(r['category'])
        if nan:
            problems[f] = ('non-numeric', nan)
    if problems:
        print('[ERROR] These features cannot go into a design matrix:')
        for f, (why, cats) in problems.items():
            print(f'    {f:<16} {why} in {len(cats)}/{len(fp_rows)}: {cats}')
        print('\n    A descriptor is blank when it does not exist for that '
              'category, e.g. peak_kPa\n    for a sweep that never peaked inside '
              'the swept range. Drop the feature, or\n    re-sweep those '
              'categories over a wider pressure range.')
        raise SystemExit(1)
 
    print(f'Fingerprints : {fp_path}  ({len(fp_rows)} categories)')
    print(f'Features     : {features}\n')
    w = max(len(r['category']) for r in fp_rows) + 2
    print(f'{"category":<{w}}' + ''.join(f'{f:>16}' for f in features))
    for r in fp_rows:
        print(f'{r["category"]:<{w}}' + ''.join(f'{r[f]:>16}' for f in features))
 
    # --- collinearity, since these are the only distinct points in the model -
    if len(features) >= 2 and len(fp_rows) >= 3:
        try:
            import numpy as np
            M = {f: np.array([float(r[f]) for r in fp_rows]) for f in features}
            worst = []
            for a, b in itertools.combinations(features, 2):
                if M[a].std() == 0 or M[b].std() == 0:
                    continue
                rr = float(np.corrcoef(M[a], M[b])[0, 1])
                if abs(rr) > 0.9:
                    worst.append((a, b, rr))
            print(f'\nFeature correlation across the {len(fp_rows)} categories '
                  f'(n is small, read loosely):')
            if worst:
                for a, b, rr in worst:
                    print(f'  [WARN] {a} vs {b}: r={rr:+.3f}. Near-collinear, so '
                          f'they add\n         roughly one dimension between '
                          f'them, not two.')
            else:
                print('  no pair above |r| = 0.9')
        except ImportError:
            pass
 
    # --- join ---------------------------------------------------------------
    SFX_IN, SFX_OUT = args.suffix_in, args.suffix_out
    made, missing, combined = [], [], []
    print()
    for r in fp_rows:
        cat = r['category']
        src = data_dir / f'{cat}{SFX_IN}'
        if not src.exists():
            print(f'[SKIP] {cat}: {src.name} not found in {data_dir}')
            missing.append(cat)
            continue
        rows48, fields48 = load(src)
        if not rows48:
            print(f'[SKIP] {cat}: {src.name} is empty')
            missing.append(cat)
            continue
 
        extra = {f: r[f] for f in features}
        out_fields = ['category', 'Sample'] + features + [
                      'Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm',
                      'SF_mean', 'SF_std']
        out_rows = [{
            'category': cat,
            'Sample': row.get('Sample', row.get('Sample_ID', '')),
            **extra,
            'Pressure_kPa': row.get('Pressure_kPa', ''),
            'NozzleSpeed_mms': row.get('NozzleSpeed_mms', ''),
            'Zoffset_mm': row.get('Zoffset_mm', ''),
            'SF_mean': row.get('SF_mean', ''),
            'SF_std': row.get('SF_std', ''),
        } for row in rows48]
 
        dst = out_dir / f'{cat}{SFX_OUT}'
        with open(dst, 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=out_fields, delimiter=';',
                                 extrasaction='ignore')
            wtr.writeheader()
            wtr.writerows(out_rows)
        made.append((cat, len(out_rows), dst))
        combined.extend(out_rows)
        print(f'[OK] {cat:<{w}} {len(out_rows):>4} rows -> {dst.name}')
 
    if args.combined_csv and combined:
        allf = list(combined[0].keys())
        with open(args.combined_csv, 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=allf, delimiter=';',
                                 extrasaction='ignore')
            wtr.writeheader()
            wtr.writerows(combined)
        print(f'\nCombined: {len(combined)} rows across {len(made)} categories '
              f'-> {args.combined_csv}')
 
    print(f'\n{len(made)}/{len(fp_rows)} category table(s) written to {out_dir}')
    if missing:
        print(f'[WARN] no 48-well file for: {missing}')
    if made:
        n = sum(c for _, c, _ in made)
        print(f'\n{n} rows total, but only {len(made)} distinct point(s) in '
              f'fingerprint space.\nThat is the sample size that matters for a '
              f'cross-category claim.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Join sweep fingerprints onto the 48-well SF tables.')
    ap.add_argument('--fingerprints_csv', required=True)
    ap.add_argument('--data_dir', default=None,
                    help='Folder with the *_sf_summary_48well.csv files '
                         '(default: next to the fingerprints CSV)')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--features', default=','.join(DEFAULT_FEATURES),
                    help=f'Comma-separated fingerprint columns to attach '
                         f'(default: {",".join(DEFAULT_FEATURES)})')
    ap.add_argument('--suffix_in', default='_sf_summary_48well.csv')
    ap.add_argument('--suffix_out', default='_sf_complete_48well.csv')
    ap.add_argument('--combined_csv', default=None,
                    help='Also write one stacked table across all categories, '
                         'which is what the leave-one-category-out step needs')
    main(ap.parse_args())