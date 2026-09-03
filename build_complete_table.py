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
Default: onset_kPa, rise_slope, peak_sf, auc_norm, tail_slope,
         sf_at_p_max_three_mean  (falling back to sf_at_p_max).
 
Those six span six things a sweep curve can say that are not each other: WHERE
it starts, HOW FAST it rises, HOW HIGH it gets, ITS AVERAGE LEVEL, WHETHER it is
still rising or collapsing at the top, and WHAT VALUE it holds there.
 
tail_slope, not collapse_slope. collapse_slope is blank whenever the sweep never
peaked, and blank again when it peaked too near the end to fit a line through the
descent: 4 of 6 categories on pressure_sweep_c6_fingerprints.csv. One blank
disqualifies the whole column. Filling those blanks with 0 would be worse than
blank, because a sweep still climbing at its last well has not been shown not to
collapse. tail_slope fits the last few wells regardless of where the peak is, so
it exists for every curve, and its sign carries the same meaning: positive still
rising, near zero plateaued, negative collapsing. collapse_slope remains in the
fingerprints file for reporting.
 
auc_norm is the MEAN SF over the swept range: the trapezoid integral divided by
the pressure span, which on a uniform grid is the arithmetic mean with the
endpoints half-weighted (correlation 0.9998 with the plain mean over 2000
simulated sweeps). It is included as a default at the project's request. Be aware
that across the six real categories it correlates with onset_kPa at r = -0.986,
so expect the C3 comparison to show it adding little beyond onset. Watch that
comparison rather than assuming either way.
 
Defaults DEGRADE, explicit choices DO NOT. A default feature that is blank or
missing in the supplied fingerprints file is dropped with a message, so the
same command works on an old 30-120 fingerprint file (where peak_kPa,
window_kPa and collapse_slope do not exist for the truncated sweeps) and on a
30-140 one where they do. A feature named explicitly with --features is still a
hard error if it is blank, because you asked for it by name.
 
auc_norm is deliberately NOT a default. Measured on
pressure_sweep_c6_fingerprints.csv it correlates with onset_kPa at r = -0.986
across the six categories, so it is onset re-expressed. It is also an average
over the swept range, which changes when the range changes: extending a sweep
from 120 to 140 kPa moved that kind of number by a factor of two on real data.
Pass it explicitly if you want it for reporting.
 
How many features you can actually afford
-----------------------------------------
This is the part that matters more than which ones. The fingerprint columns are
constant within a category, so in fingerprint space there are exactly as many
distinct points as there are CATEGORIES. Leave-one-category-out trains on N-1 of
them. A linear model in k features plus an intercept has k+1 parameters, so it
is already exactly determined at k = N-2 and underdetermined beyond that: it can
reproduce every training category perfectly while saying nothing about a held-out
one.
 
    6 categories -> 5 training points -> at most 3 features, comfortably 1 to 2
    8 categories -> 7 training points -> at most 5 features, comfortably 2 to 3
 
Measured on the six real categories, five complete features already had rank 5,
the maximum the six points allow, with 91% of their variance in just two
principal components.
 
So attaching six features here is fine and useful, because this script builds
the TABLE. Choosing to FIT on all six is the mistake. Follow plan item C3 and
run the transfer step three ways, reporting all of them: onset alone, PC1 of the
standardised set, and the full set. If one feature matches six, report one.
 
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
 
DEFAULT_FEATURES = ['onset_kPa', 'rise_slope', 'peak_sf', 'auc_norm',
                    'tail_slope', 'sf_at_p_max_three_mean']
 
# Features whose value depends on WHERE THE SWEEP STOPPED, not only on the
# material. They are only comparable across categories swept over the same
# pressure range, and the run refuses to pretend otherwise.
RANGE_DEPENDENT = {'auc_norm', 'sf_at_p_max', 'sf_at_p_max_three_mean',
                   'sf_median', 'sf_iqr', 'tail_slope', 'peak_kPa',
                   'window_kPa', 'sf_at_p_max_three', 'peak_three_sf_mean'}
 
# If a default column is absent from the fingerprints file, use the fallback.
# sf_at_p_max_three_mean is the 3-well average of the top of the sweep and only
# exists in fingerprint files written by the updated sweep_fingerprint.py.
FEATURE_FALLBACKS = {'sf_at_p_max_three_mean': 'sf_at_p_max',
                     'peak_three_sf_mean': 'peak_sf'}
 
 
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
    using_defaults = (features == DEFAULT_FEATURES)
 
    # Defaults adapt to whatever the fingerprints file actually contains, so one
    # command works on an old 30-120 file and a new 30-140 one. An explicitly
    # named feature is never silently swapped or dropped.
    if using_defaults:
        resolved, notes = [], []
        for f in features:
            if f in fp_fields:
                resolved.append(f)
                continue
            alt = FEATURE_FALLBACKS.get(f)
            if alt and alt in fp_fields:
                resolved.append(alt)
                notes.append(f'{f} not in this file, using {alt} instead')
            else:
                notes.append(f'{f} not in this file, dropped')
        features = resolved
        for n in notes:
            print(f'[default] {n}')
    else:
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
    if problems and using_defaults:
        # Drop rather than fail: the defaults are a wish list, and a descriptor
        # is blank when it does not exist for that category (a sweep that never
        # peaked), which is a fact about the data, not a mistake by the user.
        for f, (why, bad) in problems.items():
            print(f'[default] {f} dropped: {why} in {len(bad)}/{len(fp_rows)} '
                  f'categor(ies) {bad}')
        features = [f for f in features if f not in problems]
        problems = {}
        if not features:
            raise SystemExit('No usable default feature survived. Widen the '
                             'sweep range or pass --features explicitly.')
    if problems:
        print('[ERROR] These features cannot go into a design matrix:')
        for f, (why, cats) in problems.items():
            print(f'    {f:<16} {why} in {len(cats)}/{len(fp_rows)}: {cats}')
        print('\n    A descriptor is blank when it does not exist for that '
              'category, e.g. peak_kPa\n    for a sweep that never peaked inside '
              'the swept range. Drop the feature, or\n    re-sweep those '
              'categories over a wider pressure range.')
        raise SystemExit(1)
 
    n_cat = len(fp_rows)
    print(f'Fingerprints : {fp_path}  ({n_cat} categories)')
    print(f'Features     : {len(features)} -> {features}\n')
 
    # --- how many features these categories can actually support -------------
    # LOCO trains on n_cat-1 distinct points in fingerprint space. A linear
    # model in k features plus an intercept has k+1 parameters, so k = n_cat-2
    # is already exactly determined.
    budget = n_cat - 2
    print(f'Dimensionality budget')
    print(f'  {n_cat} categories -> leave-one-category-out trains on '
          f'{n_cat - 1} distinct point(s) in fingerprint space')
    print(f'  a linear model in k features has k+1 parameters, so k = {budget} '
          f'is already exactly determined')
    if len(features) > budget:
        print(f'  [WARN] you are attaching {len(features)} features against a '
              f'budget of {budget}.')
        print(f'         Attaching them to the TABLE is fine and useful for '
              f'reporting. FITTING on\n'
              f'         all {len(features)} is not: the model can reproduce '
              f'every training category\n'
              f'         exactly and still say nothing about a held-out one. '
              f'Run plan item C3 and\n'
              f'         report one feature, PC1, and the full set side by '
              f'side.')
    else:
        print(f'  {len(features)} features is within budget.')
 
    # --- range-dependent features need a common swept range ------------------
    rng_used = sorted(set(features) & RANGE_DEPENDENT)
    if rng_used and {'p_min', 'p_max'} <= set(fp_fields):
        spans = {}
        for r in fp_rows:
            try:
                spans[r['category']] = (float(r['p_min']), float(r['p_max']))
            except (TypeError, ValueError):
                pass
        distinct = set(spans.values())
        if len(distinct) > 1:
            print(f'\n  [WARN] {len(rng_used)} selected feature(s) depend on '
                  f'where the sweep STOPPED:\n         {rng_used}')
            print(f'         but the categories were not all swept over the '
                  f'same range:')
            for c, (a, b) in sorted(spans.items(), key=lambda t: t[1]):
                print(f'           {c:<24} {a:g}-{b:g} kPa')
            print('         An average or an end-of-range value taken over '
                  'different ranges is not\n         the same quantity. '
                  'Re-sweep every category over one range, or drop these\n'
                  '         features. Extending 120 to 140 moved this kind of '
                  'number by a factor\n         of two on real curves.')
        else:
            a, b = next(iter(distinct))
            print(f'  Range-dependent features {rng_used}\n'
                  f'    are comparable: all {len(spans)} categories swept '
                  f'{a:g}-{b:g} kPa.')
    print()
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
 
            # Pairwise r misses the case where three features are jointly
            # redundant without any pair being. The singular values do not.
            cols = [f for f in features if M[f].std() > 0]
            if len(cols) >= 2:
                Z = np.column_stack([(M[f] - M[f].mean()) / M[f].std(ddof=1)
                                     for f in cols])
                Z = Z - Z.mean(0)
                sv = np.linalg.svd(Z, compute_uv=False)
                var = sv ** 2 / np.sum(sv ** 2)
                cum = np.cumsum(var)
                k90 = int(np.searchsorted(cum, 0.90) + 1)
                print(f'\n  Effective dimensionality of the {len(cols)} '
                      f'feature(s):')
                print('    ' + '  '.join(f'PC{i+1} {v:.0%}'
                                         for i, v in enumerate(var) if v > 0.005))
                print(f'    {k90} component(s) carry 90% of the variance, and '
                      f'the matrix rank is\n    {np.linalg.matrix_rank(Z)} '
                      f'(it can never exceed n_categories-1 = {len(fp_rows) - 1}).')
                if k90 < len(cols):
                    print(f'    So these {len(cols)} columns are really about '
                          f'{k90} independent number(s).')
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