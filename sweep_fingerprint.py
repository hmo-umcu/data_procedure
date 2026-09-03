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
 
 
def _tail_only(P, S, tail_kPa, tail_n):
    """tail_slope alone, for the branch where nothing cleared the onset."""
    if tail_n is not None:
        n = int(min(max(2, tail_n), len(S)))
    else:
        n = int(min(max(2, np.sum(P >= P[-1] - float(tail_kPa))), len(S)))
    return {'tail_slope': round(float(np.polyfit(P[-n:], S[-n:], 1)[0]), 6),
            'tail_slope_n': n, 'tail_p_lo': round(float(P[-n]), 2)}
 
 
def fingerprint(P, S, onset_sf, smooth_n=3, tail_kPa=40.0, tail_n=None):
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
                  auc_norm=0.0, sf_median=round(float(np.median(S)), 4),
                  sf_iqr=0.0, peak_three_sf_mean=None, peak_three_sf_std=None,
                  peak_three_n=None, peak_three_p_lo=None, peak_three_p_hi=None,
                  sf_at_p_max_three_mean=None, last_three_sf_std=None,
                  last_three_n=None, last_three_p_lo=None,
                  **_tail_only(P, S, tail_kPa, tail_n))
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
 
    # ---- level descriptors over the swept range -----------------------------
    # There is deliberately NO sf_mean column. auc_norm above already IS the
    # mean: on a uniform pressure grid the trapezoid integral over the span is
    # the arithmetic mean with the endpoints half-weighted. Measured over 2000
    # simulated sweeps the two correlated at 0.9998 and differed by at most
    # 0.013, and on the four real 7.5% curves they ordered the categories
    # identically (Spearman +1.000). A second column for the same number would
    # only add a dimension to a fingerprint that finding A5 already says has
    # too many for the number of training categories.
    #
    # sf_median survives because each sweep pressure is printed ONCE, so a
    # single bad well moves the mean while the median ignores it. That matters
    # given finding A4, where one sweep point sat 4.3 LHS standard deviations
    # below the plate value at the same parameters.
    #
    # sf_iqr is the spread of SF across the swept range: steep curves have a
    # wide IQR, flat or dead ones a narrow one.
    fp['sf_median'] = round(float(np.median(S)), 4)
    q1, q3 = np.percentile(S, [25, 75])
    fp['sf_iqr']    = round(float(q3 - q1), 4)
 
    # ---- neighbourhood-averaged descriptors ---------------------------------
    # Both of the single-well descriptors above are read off ONE printed well,
    # and a sweep has no replicates, so a single bad well sets the value with
    # nothing to contradict it. These two average a short run of adjacent
    # pressures instead.
    #
    #   peak_three_sf_mean   the peak well and its two neighbours
    #   sf_at_p_max_three_mean   the last three wells of the sweep
    #
    # The window is centred on the RAW argmax, not on a smoothed curve, so the
    # location still comes from the data as measured. n and the pressure span
    # actually averaged are recorded, because at an edge only two wells exist
    # and a 3-well window at a 5 kPa step spans 10 kPa of real pressure, which
    # is a different quantity from a point reading.
    half = max(0, (smooth_n - 1) // 2)
 
    lo = max(0, k - half)
    hi = min(len(S), k + half + 1)
    seg = S[lo:hi]
    fp['peak_three_sf_mean'] = round(float(np.mean(seg)), 4)
    fp['peak_three_sf_std']  = round(float(np.std(seg)), 4)
    fp['peak_three_n']       = int(seg.size)
    fp['peak_three_p_lo']    = round(float(P[lo]), 2)
    fp['peak_three_p_hi']    = round(float(P[hi - 1]), 2)
 
    # ---- tail_slope: the always-defined replacement for collapse_slope ------
    # collapse_slope is blank whenever the sweep never peaked, and blank again
    # when it peaked too near the end to fit a line through the descent. On the
    # six-category file that is 4 blanks out of 6, and a single blank disqualifies
    # the whole column from a design matrix.
    #
    # Filling those blanks with 0 would be WRONG. A sweep still climbing at its
    # last well has not been shown not to collapse; it was not swept far enough
    # to find out. Zero would assert "no collapse" from no evidence, and put an
    # untested category at the same value as a genuinely flat one.
    #
    # tail_slope instead fits a line through the LAST tail_n wells regardless of
    # where the peak is. It is defined for every curve, and its sign carries the
    # meaning collapse_slope was reaching for:
    #     positive  still rising at the top of the swept range
    #     ~zero     plateaued
    #     negative  collapsing
    # collapse_slope is kept unchanged alongside it, for the protocol's six
    # named descriptors and for reporting.
    # The window is specified in kPa, not in wells, because a well count means
    # different things at different sweep steps. Measured on the four 30-140
    # curves, a 40 kPa tail reproduces collapse_slope almost exactly wherever
    # collapse_slope exists (-0.00155 vs -0.00155, -0.00411 vs -0.00407) while
    # a 20 kPa tail reported the one genuinely collapsing category as flat.
    if tail_n is not None:
        n_tail = int(min(max(2, tail_n), len(S)))
    else:
        n_tail = int(max(2, np.sum(P >= P[-1] - float(tail_kPa))))
        n_tail = min(n_tail, len(S))
    tail_fit = S[-n_tail:]
    P_tail = P[-n_tail:]
    fp['tail_slope']   = round(float(np.polyfit(P_tail, tail_fit, 1)[0]), 6)
    fp['tail_slope_n'] = n_tail
    fp['tail_p_lo']    = round(float(P_tail[0]), 2)
 
    n_last = min(smooth_n, len(S))
    tail = S[-n_last:]
    fp['sf_at_p_max_three_mean'] = round(float(np.mean(tail)), 4)
    # Spread of the last few wells. On a plateau these are near-replicates of
    # the same SF at slightly different pressures, so this is the closest thing
    # the sweep has to a repeatability estimate of its OWN. It is not a true
    # replicate std, because the pressures genuinely differ and any residual
    # slope inflates it, but it beats borrowing SF_std from a different plate,
    # which is what plan item B2 objects to.
    fp['last_three_sf_std'] = round(float(np.std(tail)), 4)
    fp['last_three_n']      = int(n_last)
    fp['last_three_p_lo']   = round(float(P[-n_last]), 2)
    return fp
 
 
def common_window_stats(cats, curves, step=None):
    """
    Level descriptors recomputed on the pressure window COMMON to every
    category, so they can be compared across categories at all.
 
    Why this is not optional. sf_mean, sf_median and auc_norm are averages over
    whatever range that category happened to be swept. Some plates here were
    printed 30-120 kPa and later ones 30-140 kPa. The extra four wells sit at
    the top of the rising flank where SF is highest, so a category swept to 140
    gets a higher mean than an identical material swept to 120, purely from
    where the sweep stopped. Feeding that to a cross-category model teaches it
    the sweep protocol, not the material.
 
    The experiment plan says the same thing in B1: "if some categories are
    characterised over a wider range than others, the fingerprints are not
    comparable and the transfer input is inconsistent by construction."
 
    Returns (lo, hi, {cat: {...}}) or (None, None, {}) if there is no overlap.
    """
    los = [P.min() for P, _ in curves.values()]
    his = [P.max() for P, _ in curves.values()]
    lo, hi = max(los), min(his)
    if not (hi > lo):
        return None, None, {}
    if step is None:
        gaps = []
        for P, _ in curves.values():
            gaps += list(np.diff(np.unique(P)))
        step = float(np.median(gaps)) if gaps else 5.0
    grid = np.arange(lo, hi + 1e-9, step)
    out = {}
    for c in cats:
        P, S = curves[c]
        Sc = np.interp(grid, P, S)
        # cw_auc_norm is the common-window mean; no separate cw_sf_mean for the
        # same reason there is no sf_mean.
        n_last = min(3, len(Sc))
        out[c] = {
            'cw_sf_median':      round(float(np.median(Sc)), 4),
            'cw_auc_norm':       round(float(np.trapezoid(Sc, grid) / (hi - lo)), 4),
            'cw_sf_at_hi':       round(float(Sc[-1]), 4),
            'cw_sf_at_hi_three': round(float(np.mean(Sc[-n_last:])), 4),
        }
    return float(lo), float(hi), out
 
 
def _rank(a):
    """Average ranks, ties shared. Avoids a scipy dependency."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    # average ranks within tie groups
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r
 
 
def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])
 
 
def collinearity_report(cats, fps, extra, thresh=0.90):
    """
    Rank-correlate every pair of descriptors ACROSS categories and name the
    redundant ones.
 
    The fingerprint is the model input, and finding A5 already notes that only
    four descriptors are usable while leave-one-category-out fits them from 5
    to 7 training points. Adding columns that carry the same ordering makes
    that worse, not better. This is the check that says whether a new
    descriptor earns its place, on the real data rather than by argument.
    """
    names = ['onset_kPa', 'rise_slope', 'tail_slope', 'peak_sf',
             'peak_three_sf_mean',
             'sf_at_p_max', 'sf_at_p_max_three_mean', 'auc_norm',
             'sf_median', 'sf_iqr',
             'cw_sf_median', 'cw_auc_norm', 'cw_sf_at_hi', 'cw_sf_at_hi_three']
    cols = {}
    for n in names:
        vals = []
        for c in cats:
            v = extra.get(c, {}).get(n, fps[c].get(n))
            vals.append(v)
        if all(v is not None for v in vals) and len(set(vals)) > 1:
            cols[n] = np.array(vals, float)
    if len(cols) < 2:
        print('  Not enough complete descriptors to correlate.')
        return
    if len(cats) < 4:
        print(f'  [NOTE] only {len(cats)} categories, so these correlations '
              f'are very noisy.')
        print( '         Treat them as a smell test, not evidence.')
    ks = list(cols)
    pairs = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            r = spearman(cols[ks[i]], cols[ks[j]])
            if np.isfinite(r):
                pairs.append((abs(r), r, ks[i], ks[j]))
    pairs.sort(reverse=True)
    hi = [p for p in pairs if p[0] >= thresh]
    if hi:
        print(f'  Descriptor pairs with |Spearman| >= {thresh:g} '
              f'(one of each pair is redundant):')
        for a, r, x, y in hi:
            print(f'    {x:<14} vs {y:<14}  rho = {r:+.3f}')
    else:
        print(f'  No descriptor pair reaches |Spearman| {thresh:g}.')
    print(f'\n  Least redundant descriptors (lowest max |rho| against any other):')
    worst = {}
    for a, r, x, y in pairs:
        worst[x] = max(worst.get(x, 0), a)
        worst[y] = max(worst.get(y, 0), a)
    for k, v in sorted(worst.items(), key=lambda t: t[1])[:5]:
        print(f'    {k:<14} max |rho| {v:.3f}')
 
 
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
        fp = fingerprint(P, S, args.onset_sf, args.smooth_n,
                         args.tail_kPa, args.tail_n)
        fp['onset_sf_threshold'] = args.onset_sf
        # B4: onset is one of only a few surviving descriptors and it moves a
        # lot with the threshold, so the threshold is recorded in the output and
        # two alternatives are reported alongside it.
        for t in (0.02, 0.10):
            fp[f'onset_kPa_at_{t:g}'] = fingerprint(
                P, S, t, args.smooth_n, args.tail_kPa,
                args.tail_n).get('onset_kPa')
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
 
    print(f'\n{"category":<24}{"collapse_slope":>16}{"tail_slope":>13}'
          f'{"over kPa":>12}  meaning')
    for cat in cats:
        f = fps[cat]
        cs = f.get('collapse_slope')
        ts = f.get('tail_slope')
        # label only. 1e-3 SF/kPa over a 20 kPa tail is 0.02 SF, about one
        # single-well noise, so anything flatter is called a plateau.
        mean_txt = ('still rising' if ts is not None and ts > 1e-3 else
                    'collapsing' if ts is not None and ts < -1e-3 else
                    'plateaued')
        print(f'{cat:<24}{("-" if cs is None else f"{cs:+.6f}"):>16}'
              f'{("-" if ts is None else f"{ts:+.6f}"):>13}'
              f'{f.get("tail_p_lo", "-"):>8}-{f["p_max"]:g}  {mean_txt}')
    n_blank = sum(1 for c in cats if fps[c].get('collapse_slope') is None)
    if n_blank:
        print(f'\ncollapse_slope is blank for {n_blank}/{len(cats)} categor(ies): '
              f'either the sweep never\npeaked, or it peaked too close to the '
              f'end to fit a line through the descent.\nA blank disqualifies the '
              f'whole column from a design matrix, and filling it with 0\nwould '
              f'assert "does not collapse" from no evidence. Use tail_slope as '
              f'the feature\nand keep collapse_slope for reporting.')
 
    trunc = [c for c in cats if fps[c]['truncated'] == 'yes']
    if trunc:
        print(f'\n[NOTE] {len(trunc)}/{len(cats)} categor(ies) are still rising at '
              f'the highest swept pressure:')
        print(f'       {trunc}')
        print('       For these, peak pressure / window width / collapse slope do '
              'not exist in the\n       sampled range and are left empty. The '
              'fingerprint reduces to onset, rise\n       slope and AUC.')
 
    # ------------------------------------------- common-window level metrics
    print('\n' + '=' * 72)
    print('  LEVEL DESCRIPTORS  (mean / median / AUC)')
    print('=' * 72)
    lo, hi, extra = common_window_stats(cats, curves)
    if lo is None:
        print('  Categories share no overlapping pressure window, so mean, '
              'median and AUC\n  cannot be compared across them at all.')
        extra = {}
    else:
        spans = {c: (curves[c][0].min(), curves[c][0].max()) for c in cats}
        widest = max(b - a for a, b in spans.values())
        print(f'  Common window: {lo:g} to {hi:g} kPa '
              f'({hi - lo:g} kPa wide; widest single sweep is {widest:g} kPa)')
        odd = [c for c in cats if spans[c] != (lo, hi)]
        if odd:
            print(f'  {len(odd)} categor(ies) are swept OUTSIDE this window and '
                  f'are truncated to it:')
            for c in odd:
                a, b = spans[c]
                print(f'    {c:<24} own range {a:g}-{b:g} kPa')
            print('  Their raw sf_mean / sf_median / auc_norm are NOT comparable '
                  'with the others.\n  Use the cw_ columns for anything '
                  'cross-category.')
        print(f'\n  {"category":<24}{"sf_median":>11}{"sf_iqr":>8}'
              f'{"cw_median":>11}{"cw_auc":>9}{"cw_hi":>8}{"cw_hi_3":>9}')
        for c in cats:
            f, e = fps[c], extra[c]
            print(f'  {c:<24}{f["sf_median"]:>11.4f}{f["sf_iqr"]:>8.4f}'
                  f'{e["cw_sf_median"]:>11.4f}{e["cw_auc_norm"]:>9.4f}'
                  f'{e["cw_sf_at_hi"]:>8.4f}{e["cw_sf_at_hi_three"]:>9.4f}')
        for c in cats:
            fps[c].update(extra[c])
 
    print('\n' + '=' * 72)
    print(f'  SINGLE-WELL vs {args.smooth_n}-WELL AVERAGED DESCRIPTORS')
    print('=' * 72)
    print('  A sweep prints each pressure ONCE, so peak_sf and sf_at_p_max are '
          'each read off a\n  single well with no replicate to contradict it. '
          'The averaged columns take a run\n  of adjacent pressures instead.')
    print(f'\n  {"category":<24}{"peak_sf":>9}{"peak_3":>9}{"delta":>8}'
          f'{"span kPa":>11}{"sf@pmax":>9}{"last_3":>8}{"delta":>8}{"sd_3":>7}')
    for c in cats:
        f = fps[c]
        if f.get('peak_three_sf_mean') is None:
            print(f'  {c:<24}  (no extrusion above the onset threshold)')
            continue
        dp = f['peak_three_sf_mean'] - f['peak_sf']
        dl = f['sf_at_p_max_three_mean'] - f['sf_at_p_max']
        span = f'{f["peak_three_p_lo"]:g}-{f["peak_three_p_hi"]:g}'
        flag = '' if f['peak_three_n'] == args.smooth_n else f' n={f["peak_three_n"]}'
        print(f'  {c:<24}{f["peak_sf"]:>9.4f}{f["peak_three_sf_mean"]:>9.4f}'
              f'{dp:>+8.4f}{span:>11}{f["sf_at_p_max"]:>9.4f}'
              f'{f["sf_at_p_max_three_mean"]:>8.4f}{dl:>+8.4f}'
              f'{f["last_three_sf_std"]:>7.4f}{flag}')
    edge = [c for c in cats
            if fps[c].get('peak_three_n') not in (None, args.smooth_n)]
    if edge:
        print(f'\n  [NOTE] {len(edge)} categor(ies) peak at an END of the swept '
              f'range, so only {args.smooth_n - 1} wells\n         were '
              f'available to average: {edge}. Their peak_three_sf_mean is not on '
              f'the\n         same footing as the others.')
    sds = [fps[c]['last_three_sf_std'] for c in cats
           if fps[c].get('last_three_sf_std') is not None]
    if sds:
        print(f'\n  Median spread of the last {args.smooth_n} wells across '
              f'categories: {np.median(sds):.4f}')
        print('  On a plateau those wells are near-replicates, so this is the '
              'closest thing the\n  sweep has to a repeatability estimate of '
              'its OWN. It is inflated by any residual\n  slope, so it is an '
              'UPPER bound on the noise, but unlike the 48-well proxy below it\n'
              '  comes from the same plate.')
 
    print('\n' + '=' * 72)
    print('  DESCRIPTOR REDUNDANCY')
    print('=' * 72)
    collinearity_report(cats, fps, extra)
 
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
    cols = ['category', 'onset_kPa', 'onset_censored', 'onset_sf_threshold',
            'onset_kPa_at_0.02', 'onset_kPa_at_0.1',
            'rise_slope', 'peak_sf', 'peak_kPa', 'window_kPa', 'collapse_slope',
            'tail_slope', 'tail_slope_n', 'tail_p_lo',
            'peak_three_sf_mean', 'peak_three_sf_std', 'peak_three_n',
            'peak_three_p_lo', 'peak_three_p_hi',
            'auc_norm', 'sf_median', 'sf_iqr',
            'sf_at_p_max_three_mean', 'last_three_sf_std', 'last_three_n',
            'last_three_p_lo',
            'cw_sf_median', 'cw_auc_norm', 'cw_sf_at_hi', 'cw_sf_at_hi_three',
            'truncated', 'n_points', 'n_zero', 'p_min', 'p_max', 'sf_at_p_max',
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
    ap.add_argument('--tail_kPa', type=float, default=40.0,
                    help='Width in kPa of the window at the top of the sweep '
                         'that tail_slope is fitted through (default: 40). '
                         'tail_slope is the always-defined stand-in for '
                         'collapse_slope, which is blank whenever the sweep '
                         'never peaked. Specified in kPa rather than in wells '
                         'so it means the same thing at any sweep step. At 40 '
                         'kPa it reproduced collapse_slope on the 30-140 '
                         'curves; at 20 kPa it read a genuinely collapsing '
                         'category as flat.')
    ap.add_argument('--tail_n', type=int, default=None,
                    help='Override --tail_kPa with an exact well count')
    ap.add_argument('--smooth_n', type=int, default=3,
                    help='How many adjacent wells to average for '
                         'peak_three_sf_mean and sf_at_p_max_three_mean '
                         '(default: 3). Odd values keep the peak window '
                         'centred on the peak well.')
    ap.add_argument('--onset_sf', type=float, default=0.2,
                    help='SF above which extrusion counts as started '
                         '(default: 0.2). NOTE: the help here used to say 0.02 '
                         'while the code used 0.2, so any onset value produced '
                         'before this was fixed was computed at 0.2, not 0.02. '
                         'The threshold is now written into fingerprints.csv '
                         'and onset is also reported at 0.02 and 0.1.')
    main(ap.parse_args())