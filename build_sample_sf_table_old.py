"""
build_sample_sf_table.py
-------------------------
Merge pore_scores_all_folds.csv (per-image SF scores from pore_analysis.py)
with rename_conversion_table.csv (maps {Sample_ID}_{row} stems to their
print parameters) into one final per-sample summary table.

For each Sample_ID (typically 6 replicate images, rows 0-5), reports:
    Sample_ID, Pressure_kPa, NozzleSpeed_mms, Zoffset_mm,
    fold, n_images, SF_mean, SF_std

"fold" is read from pore_scores_all_folds.csv's own fold column (present
when that file came from pore_analysis.py's --cv_parent_dir multi-fold
mode). Since cross-validation splits at the SAMPLE level, all replicates
of a sample should land in the same fold — if a sample's images are found
split across more than one fold, that's flagged as a warning (it would
indicate a CV leakage issue worth investigating).

Sample_ID and replicate row are parsed directly from the "stem" column in
pore_scores_all_folds.csv (stems follow the {Sample_ID}_{row} convention
used throughout the pipeline), so this works regardless of which fold an
image ended up in during cross-validation.

Print parameters (Pressure_kPa, NozzleSpeed_mms, Zoffset_mm) are looked up
per Sample_ID from rename_conversion_table.csv. Column names in that table
are auto-detected (case-insensitive, matching "sample_id", "pressure",
"speed"/"nozzlespeed", "zoffset"/"z_offset") — override with the
--sample_id_col/--pressure_col/--speed_col/--zoffset_col flags if your
table uses different headers; the script will print the columns it found
if auto-detection fails, so you can copy the exact name from there.

SF_std uses population std (ddof=0), matching the convention used in
unetplusplus_aggregate.py elsewhere in this pipeline.

Usage
-----
    python build_sample_sf_table.py \
        --pore_scores_csv     /path/to/pore_scores_all_folds.csv \
        --rename_table_csv    /path/to/rename_conversion_table.csv \
        --output_csv          /path/to/sample_sf_summary.csv \
        [--sample_id_col Sample_ID] \
        [--pressure_col  Pressure_kPa] \
        [--speed_col     NozzleSpeed_mms] \
        [--zoffset_col   Zoffset_mm]

Output
------
    <output_csv>   one row per Sample_ID, ;-separated
"""

import argparse
import csv
import re
import numpy as np
from pathlib import Path
from collections import defaultdict


STEM_RE = re.compile(r'^(\d+)_(\d+)$')


def sniff_delimiter(path):
    """Detect ; vs , delimiter from the header line."""
    with open(path, newline='') as f:
        first_line = f.readline()
    return ';' if first_line.count(';') >= first_line.count(',') else ','


def load_csv(path):
    delim = sniff_delimiter(path)
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return rows, fieldnames, delim


def auto_detect_column(fieldnames, candidates, label):
    """Case-insensitive substring match against a list of candidate names."""
    lower_map = {fn.lower(): fn for fn in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for fn_lower, fn in lower_map.items():
        for cand in candidates:
            if cand.lower() in fn_lower:
                return fn
    print(f'[ERROR] Could not auto-detect the "{label}" column.')
    print(f'        Columns found in rename_conversion_table.csv: {fieldnames}')
    print(f'        Pass it explicitly, e.g. --{label}_col <exact_column_name>')
    return None


def build_table(pore_scores_csv, rename_table_csv, output_csv,
                sample_id_col, pressure_col, speed_col, zoffset_col):

    # ── load pore_scores_all_folds.csv ────────────────────────────────────────
    pore_rows, pore_fields, _ = load_csv(pore_scores_csv)
    if 'stem' not in pore_fields or 'SF' not in pore_fields:
        print(f'[ERROR] {pore_scores_csv} is missing "stem" or "SF" columns. '
              f'Found: {pore_fields}')
        return
    has_fold_col = 'fold' in pore_fields
    if not has_fold_col:
        print(f'[WARN] {pore_scores_csv} has no "fold" column — '
              f'fold info will be left blank in the output. '
              f'(Expected if this came from single-folder mode rather than '
              f'--cv_parent_dir.)')

    print(f'Loaded {len(pore_rows)} image rows from {pore_scores_csv}')

    # ── group SF (and fold) by Sample_ID (parsed from stem) ───────────────────
    sf_by_sample   = defaultdict(list)
    fold_by_sample = defaultdict(set)
    unparsed = []
    for row in pore_rows:
        stem = row['stem']
        m = STEM_RE.match(stem)
        if not m:
            unparsed.append(stem)
            continue
        sid = m.group(1)
        sf_str = row.get('SF', '').strip()
        if sf_str:
            try:
                sf_by_sample[sid].append(float(sf_str))
            except ValueError:
                pass
        if has_fold_col:
            fold_val = row.get('fold', '').strip()
            if fold_val:
                fold_by_sample[sid].add(fold_val)

    if unparsed:
        print(f'[WARN] {len(unparsed)} stem(s) did not match the '
              f'{{Sample_ID}}_{{row}} pattern and were skipped: '
              f'{unparsed[:5]}{"..." if len(unparsed) > 5 else ""}')

    sample_ids = sorted(sf_by_sample.keys(), key=lambda x: int(x))
    print(f'Found {len(sample_ids)} unique Sample_IDs with SF scores: '
          f'{sample_ids}\n')

    # ── load rename_conversion_table.csv and auto-detect columns ─────────────
    rename_rows, rename_fields, _ = load_csv(rename_table_csv)
    print(f'Loaded {len(rename_rows)} rows from {rename_table_csv}')
    print(f'  Columns: {rename_fields}\n')

    sid_col = sample_id_col or auto_detect_column(
        rename_fields, ['sample_id', 'sampleid', 'sample'], 'sample_id')
    p_col = pressure_col or auto_detect_column(
        rename_fields, ['pressure_kpa', 'pressure'], 'pressure')
    s_col = speed_col or auto_detect_column(
        rename_fields, ['nozzlespeed_mms', 'nozzlespeed', 'speed'], 'speed')
    z_col = zoffset_col or auto_detect_column(
        rename_fields, ['zoffset_mm', 'zoffset', 'z_offset'], 'zoffset')

    if sid_col is None:
        print('[ERROR] Cannot proceed without identifying the Sample_ID '
              'column in rename_conversion_table.csv.')
        return

    print(f'Using columns -> Sample_ID: "{sid_col}", Pressure: "{p_col}", '
          f'Speed: "{s_col}", Zoffset: "{z_col}"\n')

    # build Sample_ID -> params lookup (first occurrence wins; warn on conflict)
    params_by_sample = {}
    for row in rename_rows:
        sid_raw = str(row.get(sid_col, '')).strip()
        if not sid_raw:
            continue
        # normalise possible "3_0" style stems in this column down to Sample_ID
        m = STEM_RE.match(sid_raw)
        sid = m.group(1) if m else sid_raw

        params = {
            'Pressure_kPa':    row.get(p_col, '') if p_col else '',
            'NozzleSpeed_mms': row.get(s_col, '') if s_col else '',
            'Zoffset_mm':      row.get(z_col, '') if z_col else '',
        }
        if sid in params_by_sample and params_by_sample[sid] != params:
            print(f'[WARN] Sample_ID {sid} has conflicting parameter rows '
                  f'in {rename_table_csv} — keeping the first one seen.')
            continue
        params_by_sample[sid] = params

    # ── build final table ──────────────────────────────────────────────────────
    out_rows = []
    missing_params = []
    multi_fold_samples = []
    for sid in sample_ids:
        sf_vals = sf_by_sample[sid]
        params  = params_by_sample.get(sid)
        if params is None:
            missing_params.append(sid)
            params = {'Pressure_kPa': '', 'NozzleSpeed_mms': '', 'Zoffset_mm': ''}

        folds_seen = sorted(fold_by_sample.get(sid, set()))
        if len(folds_seen) > 1:
            multi_fold_samples.append(sid)
        fold_str = ','.join(folds_seen)

        sf_arr = np.array(sf_vals)
        out_rows.append({
            'Sample_ID':       sid,
            'Pressure_kPa':    params['Pressure_kPa'],
            'NozzleSpeed_mms': params['NozzleSpeed_mms'],
            'Zoffset_mm':      params['Zoffset_mm'],
            'fold':            fold_str,
            'n_images':        len(sf_vals),
            'SF_mean':         f'{np.mean(sf_arr):.4f}' if len(sf_vals) else '',
            'SF_std':          f'{np.std(sf_arr):.4f}'  if len(sf_vals) else '',
        })

    if missing_params:
        print(f'[WARN] {len(missing_params)} Sample_ID(s) had SF scores but '
              f'no matching row in rename_conversion_table.csv: '
              f'{missing_params}')
    if multi_fold_samples:
        print(f'[WARN] {len(multi_fold_samples)} Sample_ID(s) had replicate '
              f'images split across MORE THAN ONE fold — this should not '
              f'happen with sample-level CV splitting and may indicate a '
              f'data leakage issue: {multi_fold_samples}')

    fieldnames = ['Sample_ID', 'Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm',
                  'fold', 'n_images', 'SF_mean', 'SF_std']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(out_rows)

    print(f'\n── Per-sample SF summary ──')
    header = f'{"Sample":>8}  {"Pressure":>10}  {"Speed":>10}  {"Zoffset":>10}  '\
             f'{"fold":>10}  {"n":>3}  {"SF_mean":>9}  {"SF_std":>9}'
    print(header)
    print('─' * len(header))
    for r in out_rows:
        print(f'{r["Sample_ID"]:>8}  {str(r["Pressure_kPa"]):>10}  '
              f'{str(r["NozzleSpeed_mms"]):>10}  {str(r["Zoffset_mm"]):>10}  '
              f'{r["fold"]:>10}  {r["n_images"]:>3}  {r["SF_mean"]:>9}  '
              f'{r["SF_std"]:>9}')

    print(f'\n✓ Final table → {output_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build per-sample SF summary table from '
                    'pore_scores_all_folds.csv + rename_conversion_table.csv.'
    )
    parser.add_argument('--pore_scores_csv', required=True)
    parser.add_argument('--rename_table_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--sample_id_col', default=None,
        help='Column name for Sample_ID in rename_conversion_table.csv '
             '(auto-detected if omitted)')
    parser.add_argument('--pressure_col', default=None,
        help='Column name for Pressure_kPa (auto-detected if omitted)')
    parser.add_argument('--speed_col', default=None,
        help='Column name for NozzleSpeed_mms (auto-detected if omitted)')
    parser.add_argument('--zoffset_col', default=None,
        help='Column name for Zoffset_mm (auto-detected if omitted)')
    args = parser.parse_args()

    build_table(
        pore_scores_csv=args.pore_scores_csv,
        rename_table_csv=args.rename_table_csv,
        output_csv=args.output_csv,
        sample_id_col=args.sample_id_col,
        pressure_col=args.pressure_col,
        speed_col=args.speed_col,
        zoffset_col=args.zoffset_col,
    )