"""
s08_make_validation_nc.py
-------------------------
Read recommendation_F.csv and emit an .nc that prints one column of six wells
at the recommended condition, then images those six wells.
 
Everything geometric is PARSED OUT OF YOUR TEMPLATE, not recomputed:
 
  * `G805[x, y, z] ; G55 origin: A1`  -> the print origin of every well
  * the imaging block                 -> the camera XY of every well
  * the strand pattern between M110 and the end-of-well lift -> reused verbatim
 
so the cross-hatch, the tail-equalisation moves, the inter-strand lift and the
work offsets are byte-identical to the sweep you already validated. Only
pressure, feedrate and print height change.
 
What is substituted
-------------------
  M200      = round(P_kPa * 10)
  F         = Speed_mms
  Z<print>  = Z_star                       (was 0.200 in the template)
  Z<lift>   = Z_star + inter-strand lift   (was 1.200, i.e. 0.200 + 1.000)
 
The end-of-well lift (Z18.400) and the imaging heights (Z40 safe, Z20 imaging)
are copied unchanged, because I cannot tell from the file whether they are
absolute clearances or offsets from the print height. They are absolute in every
well of the template regardless of Z, so they are treated as absolute. Override
with --well_lift_z if that is wrong for your setup.
 
Usage
-----
    python s08_make_validation_nc.py \
        --recommendation_csv results/05_recommendation/recommendation_F.csv \
        --template pressure_sweap_30120_step5.nc \
        --out validation_F_col1.nc \
        [--column 1] [--n_wells 6]
        [--also_historical --column2 2]   # protocol 5.8: V1 and V2 on one plate
 
`--also_historical` adds a second column at the historical-best condition read
from the same CSV, which is the V1-vs-V2 comparison the protocol asks for in a
single fresh batch.
"""
 
import argparse
import re
from pathlib import Path
 
import pandas as pd
 
ROWS = ['A', 'B', 'C', 'D', 'E', 'F']
ORIGIN_RE = re.compile(r'^G805\[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]\s*;\s*G55 origin:\s*([A-H]\d+)')
IMG_WELL_RE = re.compile(r'^;\s*---\s*Well\s+([A-H]\d+)\s*---')
IMG_XY_RE = re.compile(r'^G00\s+X([-\d.]+)\s+Y([-\d.]+)')
 
 
def parse_template(path):
    """Pull the well tables and the strand pattern out of the template."""
    text = Path(path).read_text(errors='replace')
    lines = text.splitlines()
    split = next((i for i, l in enumerate(lines)
                  if 'CAMERA IMAGING POSITIONS' in l), len(lines))
    print_part, image_part = lines[:split], lines[split:]
 
    origins = {}
    for l in print_part:
        m = ORIGIN_RE.match(l.strip())
        if m:
            origins.setdefault(m.group(4).upper(),
                               (float(m.group(1)), float(m.group(2)), float(m.group(3))))
 
    imaging, cur = {}, None
    for l in image_part:
        m = IMG_WELL_RE.match(l.strip())
        if m:
            cur = m.group(1).upper()
            continue
        if cur:
            m = IMG_XY_RE.match(l.strip())
            if m:
                imaging[cur] = (float(m.group(1)), float(m.group(2)))
                cur = None
 
    # strand pattern: from the first "G807[2," after an M110 to the tail-eq line
    start = end = None
    for i, l in enumerate(print_part):
        if start is None and l.strip().startswith('M110=') and i + 1 < len(print_part) \
                and print_part[i + 1].strip().startswith('G807[2,'):
            start = i + 1
        elif start is not None and 'tail-eq' in l and 'V1/V2' in l:
            end = i
            break
    if start is None or end is None:
        raise SystemExit('Could not locate the strand pattern in the template.')
    pattern = print_part[start:end + 1]
 
    if not origins or not imaging:
        raise SystemExit(f'Parsed {len(origins)} print origin(s) and '
                         f'{len(imaging)} imaging position(s) from {path}. '
                         f'Expected both.')
    return origins, imaging, pattern
 
 
def extrapolate(table, col, n_wells, kind):
    """Wells for a column the template does not cover, from its measured steps."""
    have = {w: v for w, v in table.items() if w[1:] == str(col)}
    if len(have) >= n_wells:
        return {f'{ROWS[i]}{col}': have[f'{ROWS[i]}{col}'] for i in range(n_wells)}
 
    cols = sorted({int(w[1:]) for w in table})
    base = cols[0]
    base_wells = {w: v for w, v in table.items() if int(w[1:]) == base}
    if len(base_wells) < n_wells:
        raise SystemExit(f'Template column {base} has only {len(base_wells)} '
                         f'{kind} well(s); cannot build {n_wells}.')
    if len(cols) < 2:
        raise SystemExit(f'Only one column of {kind} positions in the template, '
                         f'so column {col} cannot be extrapolated.')
    dx = ((table[f'A{cols[1]}'][0] - table[f'A{base}'][0]) / (cols[1] - base))
    out = {}
    for i in range(n_wells):
        w0 = f'{ROWS[i]}{base}'
        v = table[w0]
        shift = dx * (col - base)
        out[f'{ROWS[i]}{col}'] = ((v[0] + shift, v[1], v[2]) if len(v) == 3
                                  else (v[0] + shift, v[1]))
    print(f'  [NOTE] column {col} {kind} positions extrapolated from column '
          f'{base} using dx={dx:+.3f} mm per column. Verify before running.')
    return out
 
 
def print_block(wells, origins, pattern, P, Fs, Z, lift, well_lift, first, label):
    out = [f'; -- {label}: P={P:g} kPa  F={Fs:g} mm/s  Z={Z:.3f} mm --']
    m200 = int(round(P * 10))
    for j, w in enumerate(wells):
        x, y, z = origins[w]
        out += [f'G805[{x:.3f}, {y:.3f}, {z:.3f}] ; G55 origin: {w}', 'G55']
        if first and j == 0:
            out += ['', "; Changing tool to 'PSD 1'", '#FLUSH WAIT', 'T1',
                    'G807[1, 0.002, 0.002] ; time-based start/stop delays [s]',
                    f'M200={m200} ; pressure {P:g}kPa', f'F{Fs:.3f}',
                    'G00 X-2.000 Y-2.500', 'M151 ; Engage tool for printing',
                    f'Z{Z:.3f}']
        else:
            out += [f'G00 Z{well_lift:.3f}', f'M200={m200}', f'F{Fs:.3f}',
                    'X-2.000 Y-2.500', f'Z{Z:.3f}']
        out.append(f'M110={min(95, 10 + int(80 * (j + 1) / max(1, len(wells))))}')
        for l in pattern:
            s = l
            if re.match(r'^G00 Z1\.200', s.strip()):
                s = re.sub(r'Z1\.200', f'Z{lift:.3f}', s)
            elif re.match(r'^G00 Z0\.200', s.strip()):
                s = re.sub(r'Z0\.200', f'Z{Z:.3f}', s)
            out.append(s)
        out.append(f'G00 Z{well_lift:.3f} ; lift after well {w}')
    out.append('')
    return out
 
 
def imaging_block(wells, imaging, safe_z, img_z):
    out = ['; CAMERA IMAGING POSITIONS',
           f'; Wells: {", ".join(wells)}',
           f'; Z safe = {safe_z:g} mm  |  Z imaging = {img_z:g} mm',
           '; ' + '=' * 58, '', '#FLUSH WAIT',
           '#CONTOUR MODE OFF          ; Exit tracking mode from printing',
           '#FLUSH WAIT', '', 'T1                         ; Select slot 1',
           'G803                       ; Move to system safe height', '']
    for w in wells:
        x, y = imaging[w]
        out += [f'; --- Well {w} ---',
                f'G00 G54 G90 Z{safe_z:.3f}          ; Raise to safe Z first',
                f'G00 X{x:.3f} Y{y:.3f}              ; Move XY to {w}',
                f'G00 Z{img_z:.3f}                ; Lower to imaging height',
                f'V.E.UserInteraction.Message = "Camera at {w} - trigger imaging, '
                f'then click OK"',
                'M121                       ; Pause for manual camera trigger', '']
    out += ['; --- Return home ---',
            f'G00 G54 G90 Z{safe_z:.3f}          ; Safe Z before going home',
            'G800                       ; Go home', 'M110=1000', 'M30']
    return out
 
 
def main(a):
    rec = pd.read_csv(a.recommendation_csv, sep=';').iloc[0]
    P, Fs, Z = float(rec['P_star']), float(rec['Speed_star']), float(rec['Z_star'])
    origins, imaging, pattern = parse_template(a.template)
    print(f'Template: {len(origins)} print origin(s), {len(imaging)} imaging '
          f'position(s), {len(pattern)} pattern line(s)')
 
    conds = [(a.column, P, Fs, Z, f'V1 ML recommendation ({rec["category"]})')]
    if a.also_historical:
        hp = str(rec['historical_best_params']).split('/')
        conds.append((a.column2, float(hp[0]), float(hp[1]), float(hp[2]),
                      'V2 historical best LHS'))
 
    body, all_wells = [], []
    for i, (col, p, f, z, lab) in enumerate(conds):
        og = extrapolate(origins, col, a.n_wells, 'print')
        wells = [f'{ROWS[k]}{col}' for k in range(a.n_wells)]
        all_wells += wells
        body += print_block(wells, og, pattern, p, f, z,
                            z + a.strand_lift, a.well_lift_z, i == 0, lab)
        print(f'  column {col}: {a.n_wells} well(s) at P={p:g} F={f:g} Z={z:g} '
              f'-> M200={int(round(p * 10))}')
 
    img = {}
    for col, *_ in conds:
        img.update(extrapolate(imaging, col, a.n_wells, 'imaging'))
 
    head = [f'% {Path(a.out).name}',
            '; Generated by s08_make_validation_nc.py',
            f'; Template: {Path(a.template).name}',
            f'; Recommendation: {Path(a.recommendation_csv).name}',
            '; Geometry: 2D cross-hatch grid | 3+3 strands | 2x2 open pores | single layer',
            '; Temperature: set manually in Architect UI (not in G-code)',
            ';']
    for col, p, f, z, lab in conds:
        head.append(f'; Col {col}  {ROWS[0]}{col}-{ROWS[a.n_wells - 1]}{col}   '
                    f'P={p:g} kPa  F={f:g} mm/s  Z={z:g} mm   {lab}')
    head += [';', f'; Model: {rec.get("model", "unrecorded")}',
             f'; Predicted SF_mean: {rec.get("pred_SF_mean", "")} '
             f'+/- {rec.get("pred_std", "")} ({rec.get("uncertainty_source", "")})',
             f'; Trained on: {rec.get("train_categories", "")}',
             f'; Target SF values were NOT used to choose this condition.',
             ';', '; INITIALIZATION', 'T1',
             f'M200={int(round(conds[0][1] * 10))} ; Set pressure to {conds[0][1]:g}kPa',
             'T0', 'G803 ; Move to safe height', '; INITIALIZATION', '',
             'M110=0 ; Set printing progress to 0%', 'T1', 'G801 ; Measure tool',
             '', '#CONTOUR MODE ON [DEV PATH_DEV=0.08]', '',
             'M312 ; Wait for work zone temperature', '']
 
    nc = head + body + imaging_block(all_wells, img, a.safe_z, a.imaging_z)
    Path(a.out).write_text('\n'.join(nc) + '\n')
    print(f'\nWrote {len(nc)} lines, {len(all_wells)} well(s) -> {a.out}')
    print(f'  print  : {len(conds)} column(s) x {a.n_wells} replicate(s)')
    print(f'  imaging: {len(all_wells)} well(s) in column-major order')
    print('\nCheck the first G805/M200/F/Z block against the template before '
          'running on the machine.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Build a validation .nc from a frozen recommendation.')
    ap.add_argument('--recommendation_csv', required=True)
    ap.add_argument('--template', required=True,
                    help='An existing sweep .nc, used for geometry and offsets')
    ap.add_argument('--out', required=True)
    ap.add_argument('--column', type=int, default=1,
                    help='Plate column for the recommendation (default: 1)')
    ap.add_argument('--n_wells', type=int, default=6,
                    help='Replicates, i.e. wells down the column (default: 6)')
    ap.add_argument('--also_historical', action='store_true',
                    help='Add a second column at the historical-best condition')
    ap.add_argument('--column2', type=int, default=2)
    ap.add_argument('--strand_lift', type=float, default=1.0,
                    help='Lift above print height between H and V strands (mm)')
    ap.add_argument('--well_lift_z', type=float, default=18.4,
                    help='Absolute Z between wells (mm)')
    ap.add_argument('--safe_z', type=float, default=40.0)
    ap.add_argument('--imaging_z', type=float, default=20.0)
    main(ap.parse_args())