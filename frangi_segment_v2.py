"""
STEP 1 v7 — Frangi + Geo-Constrained Noise Suppression
========================================================
Solves the core tension:
    pct=70  → catches all strands BUT includes noise outside geometry
    pct=80  → clean BUT misses bottom/faint strands
    close_r=50 → seals all gaps BUT over-expands geometry

SOLUTION — Two-pass geo-constrained approach:
    Pass 1 (coarse): use close_r_geo (e.g. 50) to define a SPATIAL BOUNDING
                     REGION for the geometry — where we expect strands to be.
                     This bounding region is only used as a spatial mask,
                     NOT as the final geometry output.
    Pass 2 (fine):   apply pct=70 threshold ONLY inside the bounding region.
                     Outside = automatically excluded.
    Result: sensitivity of pct=70, spatial containment of close_r=50,
            NO geometry expansion.

Usage:
    # Preview first:
    python step1_frangi_v7_final.py \
        --input_dir ./images --output_dir ./step1_out \
        --crop_w 95 --crop_h 95 \
        --cx 640 --cy 512 --radius 480 \
        --pct 70 --close_r_geo 50 \
        --preview_only

    # Full batch:
    python step1_frangi_v7_final.py \
        --input_dir ./images --output_dir ./step1_out \
        --crop_w 95 --crop_h 95 \
        --cx 640 --cy 512 --radius 480 \
        --pct 70 --close_r_geo 50 \
        --save_mask_png

Parameters to tune:

  CROP (applied first to every image):
  --crop_w          Keep this %% of width, centered (default: 95)
  --crop_h          Keep this %% of height, centered (default: 95)

  CIRCLE MASK (in cropped image coordinates):
  --cx              Circle center x px (default: cropped_w/2)
  --cy              Circle center y px (default: cropped_h/2)
  --radius          Circle radius px
  --shrink          Shrink radius by N px to trim LED arc (default: 0)

  FRANGI:
  --sigmas          start stop step (default: 2 16 2)
  --clip            CLAHE clip limit (default: 0.03)
  --dark            exclude pixels darker than this (default: 50)
  --bright          exclude pixels brighter than this (default: 230)

  THRESHOLDING — TWO-PASS GEO-CONSTRAINED:
  --pct             Frangi threshold percentile (default: 70)
                    Lower = more sensitive (catches faint strands)
                    Higher = fewer false positives

  --close_r_geo     Closing radius for bounding region ONLY (default: 50)
                    Defines WHERE the geometry is, not how big it appears.
                    Increase if any strand is outside the bounding region.
                    Does NOT expand the final geometry output.

  --geo_dil         Dilate bounding region by N px after closing (default: 15)
                    Safety margin so bounding region doesn't clip strand edges.

  CLEANUP (applied after geo-constrained threshold):
  --min_size1       Min object size pass 1 px² (default: 200)
  --dil_r           Dilation disk radius (default: 2)
  --min_size2       Min object size pass 2 px² (default: 400)

  ADDITIONAL NOISE FILTER (optional, on top of geo-constraint):
  --proximity_filter     If set, also remove components far from the
                         largest component centroid
  --max_dist_px     Max centroid distance px to keep (default: 300)
                    Only used if --proximity_filter is set

  --area_ratio_filter    If set, also remove components smaller than
                         area_ratio * largest_component_area
  --min_area_ratio  Min area ratio to keep (default: 0.005 = 0.5%)
                    Only used if --area_ratio_filter is set

  OUTPUT:
  --preview_only    Process first image only, exit
  --save_mask_png   Save _mask.png (green on black)
  --dpi             Figure DPI (default: 100)
"""

import argparse, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure, morphology, measure
from skimage.filters import frangi
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes
import warnings
warnings.filterwarnings('ignore')

# ── Arguments ─────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--input_dir',       required=True)
p.add_argument('--output_dir',      required=True)
p.add_argument('--pattern',         default='*.tif')

# Crop
p.add_argument('--crop_w',          type=float, default=95.0)
p.add_argument('--crop_h',          type=float, default=95.0)

# Circle
p.add_argument('--cx',              type=int,   default=None)
p.add_argument('--cy',              type=int,   default=None)
p.add_argument('--radius',          type=int,   default=None)
p.add_argument('--shrink',          type=int,   default=0)

# Frangi
p.add_argument('--sigmas',          nargs=3, type=int, default=[2, 16, 2],
               metavar=('START','STOP','STEP'))
p.add_argument('--clip',            type=float, default=0.03)
p.add_argument('--dark',            type=float, default=50)
p.add_argument('--bright',          type=float, default=230)

# Two-pass geo-constrained threshold
p.add_argument('--pct',             type=float, default=70,
               help='Frangi threshold percentile (default: 70)')
p.add_argument('--close_r_geo',     type=int,   default=50,
               help='Closing radius for bounding region only (default: 50)')
p.add_argument('--geo_dil',         type=int,   default=15,
               help='Dilation of bounding region px (default: 15)')

# Cleanup
p.add_argument('--min_size1',       type=int,   default=200)
p.add_argument('--dil_r',           type=int,   default=2)
p.add_argument('--min_size2',       type=int,   default=400)

# Optional extra noise filters
p.add_argument('--proximity_filter',  action='store_true',
               help='Also filter by distance from largest component')
p.add_argument('--max_dist_px',       type=int,   default=300)
p.add_argument('--area_ratio_filter', action='store_true',
               help='Also filter by area ratio to largest component')
p.add_argument('--min_area_ratio',    type=float, default=0.005)

# Output
p.add_argument('--preview_only',    action='store_true')
p.add_argument('--save_mask_png',   action='store_true')
p.add_argument('--dpi',             type=int,   default=100)
args = p.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
sigmas = list(range(args.sigmas[0], args.sigmas[1], args.sigmas[2]))

# ── Load first image → compute fixed crop + circle ────────────────────────────
files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
if not files:
    print(f"No files found."); exit(1)

first_raw  = np.array(Image.open(files[0]))
H_raw, W_raw = first_raw.shape[:2]
keep_w = int(round(W_raw * args.crop_w / 100.0))
keep_h = int(round(H_raw * args.crop_h / 100.0))
x0 = (W_raw - keep_w) // 2;  x1 = x0 + keep_w
y0 = (H_raw - keep_h) // 2;  y1 = y0 + keep_h
W_crop, H_crop = keep_w, keep_h

cx     = args.cx     if args.cx     is not None else W_crop // 2
cy_c   = args.cy     if args.cy     is not None else H_crop // 2
radius = args.radius if args.radius is not None else min(H_crop, W_crop) // 2
radius_eff = max(1, radius - args.shrink)

Yg, Xg      = np.ogrid[:H_crop, :W_crop]
circle_mask = (Xg - cx)**2 + (Yg - cy_c)**2 <= radius_eff**2

print("=" * 60)
print(f"Crop           : {args.crop_w}%W x {args.crop_h}%H → {W_crop}x{H_crop} px")
print(f"Circle         : center=({cx},{cy_c})  r={radius_eff}")
print(f"Sigmas         : {sigmas}")
print(f"pct            : {args.pct}  ← Frangi sensitivity")
print(f"close_r_geo    : {args.close_r_geo}  ← bounding region only")
print(f"geo_dil        : {args.geo_dil} px")
print(f"proximity_filter: {args.proximity_filter}"
      + (f"  max_dist={args.max_dist_px}px" if args.proximity_filter else ""))
print(f"area_ratio_filter: {args.area_ratio_filter}"
      + (f"  min_ratio={args.min_area_ratio}" if args.area_ratio_filter else ""))
print("=" * 60)

# ── Core pipeline ─────────────────────────────────────────────────────────────
def process(img_np, circle_mask, args, sigmas):
    gray = (0.299*img_np[:,:,0] + 0.587*img_np[:,:,1]
            + 0.114*img_np[:,:,2]).astype(float)
    h, w = gray.shape

    valid   = (gray > args.dark) & (gray < args.bright) & circle_mask
    inv_eq  = exposure.equalize_adapthist(1.0 - gray/255.0, clip_limit=args.clip)
    fmap    = frangi(inv_eq, sigmas=sigmas, alpha=0.5, beta=0.5, black_ridges=False)
    fmax    = fmap.max()
    if fmax == 0:
        return (np.zeros((h,w), bool), np.zeros((h,w), bool),
                np.zeros((h,w)), 0.0)
    fnorm_v = (fmap / fmax) * valid

    # ── Pass 1: coarse binary for bounding region ────────────────────────────
    nz      = fnorm_v[fnorm_v > 0.001]
    thresh  = np.percentile(nz, args.pct) if len(nz) > 0 else 0.05
    coarse  = fnorm_v > thresh
    coarse  = morphology.remove_small_objects(coarse, min_size=200)
    coarse  = morphology.binary_dilation(coarse, disk(2))
    coarse  = morphology.remove_small_objects(coarse, min_size=400)

    # ── Bounding region: heavy close + fill + largest component ──────────────
    closed_heavy = morphology.binary_closing(coarse, disk(args.close_r_geo))
    geo_region   = binary_fill_holes(closed_heavy)
    lbl = measure.label(geo_region)
    pr  = sorted(measure.regionprops(lbl), key=lambda p: p.area, reverse=True)
    if pr:
        geo_region = lbl == pr[0].label
    # Dilate bounding region for safety margin
    if args.geo_dil > 0:
        geo_region = morphology.binary_dilation(geo_region, disk(args.geo_dil))
    # Constrain to circle mask
    geo_region = geo_region & circle_mask

    # ── Pass 2: pct threshold ONLY inside bounding region ────────────────────
    fnorm_constrained = fnorm_v * geo_region
    nz2    = fnorm_constrained[fnorm_constrained > 0.001]
    thresh2 = np.percentile(nz2, args.pct) if len(nz2) > 0 else thresh
    binary  = fnorm_constrained > thresh2
    binary  = morphology.remove_small_objects(binary, min_size=args.min_size1)
    binary  = morphology.binary_dilation(binary, disk(args.dil_r))
    binary  = morphology.remove_small_objects(binary, min_size=args.min_size2)

    # ── Optional extra noise filters ─────────────────────────────────────────
    if args.proximity_filter or args.area_ratio_filter:
        lbl2  = measure.label(binary)
        props = sorted(measure.regionprops(lbl2),
                       key=lambda p: p.area, reverse=True)
        if props:
            main_cy, main_cx = props[0].centroid
            max_area = props[0].area
            result   = np.zeros_like(binary)
            for pp in props:
                cy_p, cx_p = pp.centroid
                dist = np.sqrt((cx_p-main_cx)**2 + (cy_p-main_cy)**2)
                keep = True
                if args.proximity_filter and dist > args.max_dist_px:
                    keep = False
                if args.area_ratio_filter and pp.area/max_area < args.min_area_ratio:
                    keep = False
                if keep:
                    result[lbl2 == pp.label] = True
            binary = result

    return binary, geo_region, fnorm_v, thresh2

# ── Process files ─────────────────────────────────────────────────────────────
if args.preview_only:
    files = files[:1]
    print("PREVIEW MODE — first file only\n")

print(f"Processing {len(files)} files...")

for fpath in files:
    fname = os.path.basename(fpath)
    base  = os.path.splitext(fname)[0]
    try:
        raw_full = np.array(Image.open(fpath))
    except Exception as e:
        print(f"  SKIP {fname}: {e}"); continue

    img_c = raw_full[y0:y1, x0:x1]
    binary, geo_region, fnorm_v, thresh = process(img_c, circle_mask, args, sigmas)

    nc  = measure.label(binary).max()
    cov = binary.sum() / binary.size * 100

    # Overlay
    ov = img_c.copy().astype(float)
    ov[binary, 0] = ov[binary,0]*0.25
    ov[binary, 1] = np.clip(ov[binary,1]*0.4+130, 0, 255)
    ov[binary, 2] = ov[binary,2]*0.25
    ov[~circle_mask] = ov[~circle_mask] * 0.2

    # Figure: 6 panels
    fig, axes = plt.subplots(1, 6, figsize=(42, 7))
    fig.patch.set_facecolor('#0a0a14')

    axes[0].imshow(img_c)
    axes[0].set_title(f'Raw (cropped)\n{base}',
                      color='white', fontsize=9, fontweight='bold')

    geo_disp = img_c.copy().astype(float)
    geo_disp[~geo_region] = geo_disp[~geo_region]*0.25
    axes[1].imshow(geo_disp.astype(np.uint8))
    axes[1].set_title(f'Bounding region\n(close_r_geo={args.close_r_geo}  dil={args.geo_dil})',
                      color='#4e9af1', fontsize=9)

    axes[2].imshow(fnorm_v * geo_region, cmap='hot',
                   vmin=0, vmax=(fnorm_v*geo_region).max())
    axes[2].set_title(f'Frangi inside region\n(pct={args.pct}  thresh={thresh:.4f})',
                      color='#cccccc', fontsize=9)

    axes[3].imshow(binary, cmap='gray')
    axes[3].set_title(f'Binary mask\nnc={nc}  cov={cov:.2f}%',
                      color='#aaffaa', fontsize=9)

    axes[4].imshow(ov.astype(np.uint8))
    axes[4].set_title('Overlay on cropped raw',
                      color='#ffffaa', fontsize=9)

    # Compare: raw pct=70 vs constrained
    nz_raw   = fnorm_v[fnorm_v > 0.001]
    t_raw    = np.percentile(nz_raw, args.pct)
    bin_raw  = fnorm_v > t_raw
    bin_raw  = morphology.remove_small_objects(bin_raw, min_size=200)
    bin_raw  = morphology.binary_dilation(bin_raw, disk(2))
    bin_raw  = morphology.remove_small_objects(bin_raw, min_size=400)
    nc_raw   = measure.label(bin_raw).max()
    ov_raw = img_c.copy().astype(float)
    ov_raw[bin_raw,0]=ov_raw[bin_raw,0]*0.25
    ov_raw[bin_raw,1]=np.clip(ov_raw[bin_raw,1]*0.4+130,0,255)
    ov_raw[bin_raw,2]=ov_raw[bin_raw,2]*0.25
    axes[5].imshow(ov_raw.astype(np.uint8))
    axes[5].set_title(f'pct={args.pct} WITHOUT constraint\nnc={nc_raw}  (for comparison)',
                      color='#ff6666', fontsize=9)

    for ax in axes: ax.axis('off')
    plt.suptitle(
        f'{base}  |  pct={args.pct}  close_r_geo={args.close_r_geo}  '
        f'geo_dil={args.geo_dil}  →  nc={nc}  cov={cov:.2f}%\n'
        f'Left panel = noisy without constraint  |  Right panels = clean with geo-constraint',
        color='white', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'{base}_step1.png'),
                dpi=args.dpi, bbox_inches='tight', facecolor='#0a0a14')
    plt.close()

    if args.save_mask_png:
        mask_rgb = np.zeros((*binary.shape, 3), np.uint8)
        mask_rgb[binary] = [120, 255, 120]
        Image.fromarray(mask_rgb).save(
            os.path.join(args.output_dir, f'{base}_mask.png'))

    Image.fromarray(img_c).save(
        os.path.join(args.output_dir, f'{base}_cropped.png'))

    np.savez_compressed(
        os.path.join(args.output_dir, f'{base}_mask.npz'),
        mask=binary, circle_mask=circle_mask,
        geo_region=geo_region,
        crop_box=np.array([x0, y0, x1, y1]))

    print(f"  OK: {fname}  nc={nc}  cov={cov:.2f}%  thresh={thresh:.4f}")

print(f"\nDone. Results in: {args.output_dir}")