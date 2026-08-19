"""
sweep_fingerprint.py
--------------------
Step 1 of the intermediate analysis: turn each category's pressure-sweep curve
into the per-category fingerprint the plan uses as the model input, plot all
curves together, and test whether they actually separate the formulations.
 
Why this runs first
-------------------
The plan makes Aim 1 conditional on one check: "the pressure-sweep curves must
actually separate the formulations in a replicate-consistent way ... If they do
not, the fingerprint carries little information and Aim 1 should fall back to
the optimum-comparison form." Everything downstream depends on the answer, and
this is the cheapest thing to compute, so it goes first.
 
Descriptors, and which are actually measurable
-----------------------------------------------
The plan names six: extrusion onset, printable-window width, peak-SF pressure,
rise slope, collapse slope, area under the curve.
 
A sweep that is still rising at its highest pressure has no peak inside the
sampled range, so peak pressure, window width and collapse slope do not exist
for it. This script reports those as empty rather than inventing a value at the
range edge, and sets `truncated=yes`. On cell_gelma_10_60 that is the case: SF
climbs monotonically to the last well at 120 kPa.
 
Noise for the separation check
-------------------------------
Each sweep pressure is printed once, so the sweep carries NO replicate-noise
estimate of its own. The check the plan specifies ("beyond replicate noise")
therefore cannot be computed from sweep data alone. As a stand-in this script
uses the median SF_std from the SAME category's 48-well plate, where each
condition has 6 replicates. That is a proxy from a different plate, not the
sweep's own repeatability, and it is labelled as such everywhere it is used.
Treat a "separated" verdict from it as suggestive, not settled.
 
Usage
-----
    python sweep_fingerprint.py --data_dir <folder with the summary CSVs> \
        [--output_dir <folder>] [--onset_sf 0.02]
 
    The folder should hold the per-category files named
        <category>_sf_summary_sweep.csv     (required)
        <category>_sf_summary_48well.csv    (optional, used for the noise proxy)
 
Outputs
-------
    sweep_curves.png     all curves on one axis
    fingerprints.csv     one row per category
    console              the separation check
"""
 
import argparse
import csv
import re
from pathlib import Path
 
import numpy as np
 
# Validated categorical palette (light mode). Checked with the dataviz
# validator: worst adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6.
# The contrast WARN on slots 3/4/5 is discharged by the legend plus the
# fingerprints.csv table view.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#4a3aa7', '#e34948']
INK      = '#1a1a19'
INK_SOFT = '#6b6a63'
GRID     = '#e5e4df'
SURFACE  = '#fcfcfb'
 
 
def load_summary(path):
    with open(path, newline='') as f:
        head = f.readline()
    delim = ';' if head.count(';') >= head.count(',') else ','
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter=delim))
 
 
def category_of(path, suffix):
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem


def legend_label(cat):
    m = re.fullmatch(r'(cell_)?gelma_(\d+(?:\.\d+)?)_(\d+)', cat)
    if not m:
        return cat
    cell_prefix, concentration, dof = m.groups()
    if concentration == '7':
        concentration = '7.5'
    state = 'cell-laden' if cell_prefix else 'cell-free'
    return f'GelMA {concentration}% DoF {dof}  {state}'
 
 
def curve_from(rows):
    """(pressures, SF) sorted by pressure."""
    pts = []
    for r in rows:
        try:
            pts.append((float(r['Pressure_kPa']), float(r['SF_mean'])))
        except (KeyError, TypeError, ValueError):
            continue
    pts.sort()
    return np.array([p for p, _ in pts]), np.array([s for _, s in pts])
 
 
def fingerprint(P, S, onset_sf):
    """Descriptors from one SF-vs-pressure curve. Unmeasurable ones are None."""
    fp = {
        'n_points':      len(P),
        'n_zero':        int(np.sum(S <= 0)),
        'p_min':         float(P.min()),
        'p_max':         float(P.max()),
        'sf_at_p_max':   float(S[-1]),
    }
 
    above = np.flatnonzero(S > onset_sf)
    if above.size == 0:
        fp.update(onset_kPa=None, rise_slope=None, peak_sf=None, peak_kPa=None,
                  truncated='n/a', window_kPa=None, collapse_slope=None,
                  auc_norm=0.0)
        return fp
 
    i = int(above[0])
    if i == 0:
        onset = float(P[0])          # already extruding at the lowest pressure
        fp['onset_censored'] = 'left'
    else:
        # linear interpolation between the last sub-threshold point and this one
        p0, p1, s0, s1 = P[i - 1], P[i], S[i - 1], S[i]
        onset = float(p0 + (onset_sf - s0) * (p1 - p0) / (s1 - s0)) \
            if s1 != s0 else float(p1)
        fp['onset_censored'] = ''
    fp['onset_kPa'] = round(onset, 2)
 
    k = int(np.argmax(S))
    fp['peak_sf'] = round(float(S[k]), 4)
    truncated = (k == len(S) - 1)
    fp['truncated'] = 'yes' if truncated else 'no'
 
    # rise slope over onset -> peak
    if k > i:
        sl = np.polyfit(P[i:k + 1], S[i:k + 1], 1)[0]
        fp['rise_slope'] = round(float(sl), 6)
    else:
        fp['rise_slope'] = None
 
    if truncated:
        # No maximum inside the sampled range: these three are not measurable.
        # Reporting P[-1] as "the peak" would be an artefact of where the sweep
        # stopped, and window width and collapse slope would be pure fiction.
        fp['peak_kPa'] = None
        fp['window_kPa'] = None
        fp['collapse_slope'] = None
    else:
        fp['peak_kPa'] = round(float(P[k]), 2)
        half = 0.5 * S[k]
        inw = np.flatnonzero(S >= half)
        fp['window_kPa'] = round(float(P[inw[-1]] - P[inw[0]]), 2)
        sl = np.polyfit(P[k:], S[k:], 1)[0] if len(P) - k >= 3 else None
        fp['collapse_slope'] = round(float(sl), 6) if sl is not None else None
 
    # AUC normalised by pressure span, so it is a mean SF over the swept range
    fp['auc_norm'] = round(float(np.trapezoid(S, P) / (P[-1] - P[0])), 4)
    return fp
 
 
def noise_proxy(rows48):
    """Median SF_std across the 48-well plate: a stand-in for sweep repeatability."""
    stds = []
    for r in rows48 or []:
        try:
            if int(r.get('n_images', 0)) > 1:
                stds.append(float(r['SF_std']))
        except (TypeError, ValueError):
            continue
    return float(np.median(stds)) if stds else None
 
 
def plot_curves(cats, curves, out_path, onset_sf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
 
    for n, cat in enumerate(cats):
        P, S = curves[cat]
        ax.plot(P, S, color=SERIES[n % len(SERIES)], linewidth=2,
                marker='o', markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=legend_label(cat), zorder=3)
 
    ax.axhline(onset_sf, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 4)),
               zorder=1)
    ax.annotate(f'onset threshold SF={onset_sf:g}',
                xy=(ax.get_xlim()[0], onset_sf), xytext=(4, 4),
                textcoords='offset points', fontsize=8, color=INK_SOFT)
 
    ax.set_xlabel('Pressure (kPa)', fontsize=10, color=INK)
    ax.set_ylabel('Shape fidelity SF', fontsize=10, color=INK)
    ax.set_title('Pressure-sweep fingerprint curves', fontsize=12, color=INK,
                 pad=10, loc='left')
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    leg = ax.legend(frameon=False, fontsize=9, loc='upper left',
                    labelcolor=INK)
    for t in leg.get_texts():
        t.set_color(INK)
 
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
 
 
def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) if args.output_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
 
    SW, LH = '_sf_summary_sweep.csv', '_sf_summary_48well.csv'
    sweep_files = sorted(data_dir.glob(f'*{SW}'))
    if not sweep_files:
        raise SystemExit(f'No *{SW} files in {data_dir}')
 
    cats, curves, fps = [], {}, {}
    for p in sweep_files:
        cat = category_of(p, SW)
        P, S = curve_from(load_summary(p))
        if len(P) < 3:
            print(f'[SKIP] {cat}: only {len(P)} usable point(s)')
            continue
        cats.append(cat)
        curves[cat] = (P, S)
        fp = fingerprint(P, S, args.onset_sf)
        lh = data_dir / f'{cat}{LH}'
        fp['noise_proxy_sf_std'] = (round(noise_proxy(load_summary(lh)), 4)
                                    if lh.exists() else None)
        fps[cat] = fp
 
    print(f'Categories: {len(cats)}')
    print(f'{"category":<24}{"onset":>8}{"peak SF":>9}{"peak P":>8}'
          f'{"trunc":>7}{"rise":>10}{"AUC":>8}{"zeros":>7}')
    for cat in cats:
        f = fps[cat]
        def g(k, fmtstr='{:.4g}'):
            v = f.get(k)
            return '-' if v is None else fmtstr.format(v)
        print(f'{cat:<24}{g("onset_kPa"):>8}{g("peak_sf"):>9}'
              f'{g("peak_kPa"):>8}{f["truncated"]:>7}{g("rise_slope"):>10}'
              f'{g("auc_norm"):>8}{f["n_zero"]:>7}')
 
    trunc = [c for c in cats if fps[c]['truncated'] == 'yes']
    if trunc:
        print(f'\n[NOTE] {len(trunc)}/{len(cats)} categor(ies) are still rising at '
              f'the highest swept pressure:')
        print(f'       {trunc}')
        print('       For these, peak pressure / window width / collapse slope do '
              'not exist in the\n       sampled range and are left empty. The '
              'fingerprint reduces to onset, rise\n       slope and AUC.')
 
    # ------------------------------------------------------- separation check
    print('\n' + '=' * 72)
    print('  SEPARATION CHECK')
    print('=' * 72)
    if len(cats) < 2:
        print('  Need at least 2 categories.')
    else:
        grid = curves[cats[0]][0]
        for c in cats[1:]:
            grid = np.union1d(grid, curves[c][0])
        interp = {c: np.interp(grid, *curves[c]) for c in cats}
        proxies = [fps[c]['noise_proxy_sf_std'] for c in cats
                   if fps[c]['noise_proxy_sf_std'] is not None]
        noise = float(np.median(proxies)) if proxies else None
 
        print(f'  Common pressure grid: {len(grid)} point(s), '
              f'{grid.min():g} to {grid.max():g} kPa')
        if noise is None:
            print('  No 48-well files found, so there is no noise proxy at all. '
                  'Pairwise\n  distances are reported without a reference scale.')
        else:
            print(f'  Noise proxy (median 48-well SF_std, NOT sweep '
                  f'repeatability): {noise:.4f}')
        print(f'\n  Pairwise RMS difference between curves'
              f'{" (x noise proxy)" if noise else ""}:')
        pairs = []
        for a in range(len(cats)):
            for b in range(a + 1, len(cats)):
                d = float(np.sqrt(np.mean((interp[cats[a]] - interp[cats[b]]) ** 2)))
                pairs.append((d, cats[a], cats[b]))
        pairs.sort()
        for d, ca, cb in pairs:
            ratio = f'{d / noise:>6.1f}x' if noise else ''
            print(f'    {ca:<24} vs {cb:<24} RMS {d:.4f}  {ratio}')
 
        if noise:
            worst, ca, cb = pairs[0]
            print(f'\n  Closest pair: {ca} vs {cb}, RMS {worst:.4f} '
                  f'= {worst / noise:.1f}x the noise proxy.')
            if worst < 2 * noise:
                print('  VERDICT: the closest pair is within ~2x the noise proxy. '
                      'The curves may not\n  separate these formulations. This is '
                      'the case the plan says should trigger the\n  fallback '
                      'framing for Aim 1.')
            else:
                print('  VERDICT: every pair is separated by more than 2x the '
                      'noise proxy. Suggestive\n  that the fingerprint carries '
                      'category information, but remember the proxy comes\n  '
                      'from a different plate, so this is not the replicate-'
                      'consistency test itself.')
 
    # ------------------------------------------------------------- write out
    cols = ['category', 'onset_kPa', 'onset_censored', 'rise_slope', 'peak_sf',
            'peak_kPa', 'window_kPa', 'collapse_slope', 'auc_norm', 'truncated',
            'n_points', 'n_zero', 'p_min', 'p_max', 'sf_at_p_max',
            'noise_proxy_sf_std']
    fp_path = out_dir / 'fingerprints.csv'
    with open(fp_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=';',
                           extrasaction='ignore')
        w.writeheader()
        for cat in cats:
            row = {'category': cat}
            row.update({k: ('' if fps[cat].get(k) is None else fps[cat].get(k))
                        for k in cols[1:]})
            w.writerow(row)
 
    png = out_dir / 'sweep_curves.png'
    try:
        plot_curves(cats, curves, png, args.onset_sf)
        print(f'\nCurves       -> {png}')
    except ImportError:
        print('\n[WARN] matplotlib not installed, no plot written. '
              'pip install matplotlib')
    print(f'Fingerprints -> {fp_path}')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Extract pressure-sweep fingerprints and test separation.')
    ap.add_argument('--data_dir', required=True,
                    help='Folder holding *_sf_summary_sweep.csv (and optionally '
                         '*_sf_summary_48well.csv for the noise proxy)')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--onset_sf', type=float, default=0.1,
                    help='SF above which extrusion counts as started '
                         '(default: 0.02)')
    main(ap.parse_args())
 