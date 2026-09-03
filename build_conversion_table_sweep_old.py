"""
build_conversion_table_sweep.py
--------------------------------
Build rename_conversion_table.csv for a pressure-sweep folder (sweep_0.tif ...
sweep_18.tif) by reading the print parameters straight out of the .nc G-code.
 
Why parse the G-code instead of assuming a formula
---------------------------------------------------
The sweep is nominally "30-120 kPa in 5 kPa steps", and for
pressure_sweap_30120_step5.nc that formula does hold exactly (P = 30 + 5*N,
verified against all 19 wells). But the human-readable comments in that same
file disagree with the machine codes:
 
    ; -- Col 1 | Sample 0  | P=10-20kPa  ...   <- wrong, wells are 30-55 kPa
    ; -- Col 2 | Sample 7  | P=55kPa     ...   <- wrong, wells are 60-85 kPa
    ; -- Col 3 | Sample 14 | P=85kPa     ...   <- wrong, wells are 90-115 kPa
    ; -- Col 4 | Sample 21 | P=115kPa    ...   <- wrong, well is 120 kPa
 
Those comments are stale generator output. The M200 codes the printer actually
executes are the ground truth, so this script reads those (M200=300 -> 30.0 kPa,
i.e. value/10), plus the F feedrate and the print-height Z, per well. If you
ever run a sweep with a different range or step, this keeps working without
edits.
 
Well order is column-major (A1..F1, A2..F2, A3..F3, A4), which is the order the
images are numbered: sweep_N is the Nth well printed.
 
Usage
-----
    python build_conversion_table_sweep.py \
        --nc_file    pressure_sweap_30120_step5.nc \
        --output_csv rename_conversion_table_sweep.csv \
        [--data_dir  .../pressure_sweep_cell_gelma_7_60] \
        [--stem_prefix sweep]
 
Output columns (;-separated)
----------------------------
    Sample_ID;row;stem;well;Pressure_kPa;NozzleSpeed_mms;Zoffset_mm
 
Sample_ID is the sweep index N (each sweep well is its own condition, so unlike
the 48-well plates there is one image per sample, not six replicates).
"""
 
import argparse
import csv
import re
from pathlib import Path
 
 
def natural_key(stem):
    """Sort 'sweep_2' before 'sweep_10', and '0_5' before '1_0'."""
    parts = re.findall(r'\d+', stem)
    return ([int(x) for x in parts], stem)
 
 
def scan_stems(data_dir):
    """Real image stems in a folder, excluding anything the pipeline generated."""
    data_dir = Path(data_dir)
    stems = set()
    for ext in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
        for p in data_dir.glob(ext):
            if any(t in p.stem.lower() for t in
                   ('mask', 'overlay', 'visible', 'pred', 'target')):
                continue
            stems.add(p.stem)
    return sorted(stems, key=natural_key)
 
 
FIELDNAMES = ['Sample_ID', 'row', 'stem', 'well',
              'Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
 
ORIGIN_RE = re.compile(r'G55\s+origin:\s*([A-H]\d+)', re.IGNORECASE)
M200_RE   = re.compile(r'^\s*M200\s*=\s*(\d+)')
F_RE      = re.compile(r'^\s*F(\d+(?:\.\d+)?)\s*$')
Z_RE      = re.compile(r'^\s*(?:G0?0\s+)?Z(\d+\.\d+)\s*(?:;.*)?$')
 
# the imaging block at the end of the file also moves in Z; stop before it
END_MARKER = 'CAMERA IMAGING POSITIONS'
 
# Z moves are also used for lifts between wells (Z1.2, Z18.4, Z20, Z40).
# The print height is the small one, so ignore anything above this.
MAX_PRINT_Z_MM = 1.0
 
 
def parse_nc(nc_path):
    """Return a list of dicts, one per printed well, in print order."""
    text = Path(nc_path).read_text(errors='replace')
    body = text.split(END_MARKER)[0]
 
    wells, cur = [], None
    for line in body.splitlines():
        m = ORIGIN_RE.search(line)
        if m:
            if cur is not None:
                wells.append(cur)
            cur = {'well': m.group(1).upper(), 'P': None, 'F': None, 'Z': None}
            continue
        if cur is None:
            continue
 
        m = M200_RE.match(line)
        if m and cur['P'] is None:
            cur['P'] = int(m.group(1)) / 10.0        # M200=300 -> 30.0 kPa
            continue
        m = F_RE.match(line)
        if m and cur['F'] is None:
            cur['F'] = float(m.group(1))
            continue
        m = Z_RE.match(line)
        if m and cur['Z'] is None:
            z = float(m.group(1))
            if z <= MAX_PRINT_Z_MM:
                cur['Z'] = z
    if cur is not None:
        wells.append(cur)
    return wells
 
 
def fmt(v):
    """Trim trailing .0 so 30.0 prints as 30, matching the LHS CSV style."""
    if v is None:
        return ''
    return str(int(v)) if float(v).is_integer() else f'{v:g}'
 
 
def main(args):
    wells = parse_nc(args.nc_file)
    if not wells:
        raise SystemExit(f'No printed wells found in {args.nc_file}. '
                         f'Expected lines like "G805[...] ; G55 origin: A1".')
 
    incomplete = [w for w in wells if w['P'] is None]
    if incomplete:
        print(f'[WARN] {len(incomplete)} well(s) had no M200 pressure code and '
              f'will have a blank Pressure_kPa: {[w["well"] for w in incomplete]}')
 
    print(f'NC file: {args.nc_file}')
    print(f'         {len(wells)} printed well(s), column-major order\n')
    # sanity: is it a clean arithmetic sweep? report, do not assume
    ps = [w['P'] for w in wells if w['P'] is not None]
    if len(ps) >= 2:
        steps = {round(b - a, 6) for a, b in zip(ps, ps[1:])}
        if len(steps) == 1:
            print(f'\nUniform pressure step of {steps.pop():g} kPa, '
                  f'{fmt(ps[0])} to {fmt(ps[-1])} kPa.')
        else:
            print(f'\n[NOTE] Pressure step is not uniform: {sorted(steps)}. '
                  f'Values are taken from the M200 codes regardless.')
 
    # ---------------------------------------------------------------- stems
    # The image stems are taken from the folder and matched to wells BY ORDER,
    # not by assuming they are called sweep_0..sweep_N. Sweep folders are not
    # named consistently across runs (sweep_0.tif in some, col_row.tif in
    # others), and a name-based match silently produced an empty result table
    # when the guess was wrong. Print order is column-major, which is also
    # natural-sort order for both conventions, so the i-th image is the i-th
    # well in this NC.
    if args.data_dir:
        stems = scan_stems(args.data_dir)
        print(f'\nFolder : {args.data_dir}  ({len(stems)} image(s))')
        if not stems:
            raise SystemExit(f'No images found in {args.data_dir}. Expected .tif '
                             f'files; check the path.')
        if len(stems) != len(wells):
            print(f'\n[ERROR] {len(stems)} image(s) on disk but {len(wells)} '
                  f'well(s) in the NC. They must match one-to-one for the '
                  f'order-based mapping to be valid.')
            print(f'        images: {stems}')
            print(f'        wells : {[w["well"] for w in wells]}')
            raise SystemExit('Refusing to guess the mapping. Use --stem_prefix, '
                             'or check that the folder and the NC belong together.')
        print(f'\n{"image stem":<16} {"well":<5} {"P(kPa)":>7} {"F(mm/s)":>8} {"Z(mm)":>6}')
        for st, wl in zip(stems, wells):
            print(f'{st:<16} {wl["well"]:<5} {fmt(wl["P"]):>7} '
                  f'{fmt(wl["F"]):>8} {fmt(wl["Z"]):>6}')
        print('\nCheck this mapping. If the image order does not match the print '
              'order,\nthe pressures are attached to the wrong wells.')
    else:
        stems = [f'{args.stem_prefix}_{i}' for i in range(len(wells))]
        print(f'\nNo --data_dir given: assuming stems {stems[0]} .. {stems[-1]}. '
              f'Pass --data_dir to\nmap against the images actually present.')
 
    rows = []
    for i, (st, wl) in enumerate(zip(stems, wells)):
        rows.append({
            'Sample_ID':       i,
            'row':             0,      # one image per sweep condition
            'stem':            st,
            'well':            wl['well'],
            'Pressure_kPa':    fmt(wl['P']),
            'NozzleSpeed_mms': fmt(wl['F']),
            'Zoffset_mm':      fmt(wl['Z']),
        })
 
    with open(args.output_csv, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=';')
        wtr.writeheader()
        wtr.writerows(rows)
 
    print(f'\nWrote {len(rows)} row(s) -> {args.output_csv}')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Build a conversion table for a pressure-sweep folder from its .nc file.')
    ap.add_argument('--nc_file', required=True,
                    help='The pressure-sweep .nc G-code file')
    ap.add_argument('--output_csv', required=True,
                    help='Where to write the conversion table')
    ap.add_argument('--data_dir', default=None,
                    help='Sweep image folder; if given, cross-checks that every '
                         'well in the NC has an image and vice versa')
    ap.add_argument('--stem_prefix', default='sweep',
                    help='Filename stem prefix (default: sweep, i.e. sweep_0.tif)')
    main(ap.parse_args())
 