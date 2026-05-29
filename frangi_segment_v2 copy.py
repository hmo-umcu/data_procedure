"""
STEP 1 v6 — Frangi Segmentation with Central Crop + Manual Circle Mask
========================================================================
First crops the image to a central percentage of width and height,
then applies the manual circle mask, then runs Frangi segmentation.

Usage:
    # Preview crop + circle on first image only:
    python step1_frangi_v6_crop.py \
        --input_dir ./images --output_dir ./step1_out \
        --crop_w 95 --crop_h 95 \
        --cx 640 --cy 512 --radius 480 \
        --preview_only

    # Full batch run:
    python step1_frangi_v6_crop.py \
        --input_dir ./images --output_dir ./step1_out \
        --crop_w 95 --crop_h 95 \
        --cx 640 --cy 512 --radius 480 \
        --save_mask_png

    NOTE: --cx --cy --radius are in coordinates of the CROPPED image.
          Use --preview_only first to verify alignment.

Parameters:
  CROP (applied first):
  --crop_w      Keep this % of image width,  centered (default: 95)
  --crop_h      Keep this % of image height, centered (default: 95)

  CIRCLE MASK (applied on cropped image):
  --cx          Circle center x px in cropped image (default: cropped_width/2)
  --cy          Circle center y px in cropped image (default: cropped_height/2)
  --radius      Circle radius px (default: half of min cropped dimension)
  --shrink      Shrink radius by N px to trim LED arc at edge (default: 0)

  FRANGI:
  --sigmas      start stop step (default: 2 16 2)
  --clip        CLAHE clip limit (default: 0.03)
  --dark        exclude pixels darker than this gray value (default: 50)
  --bright      exclude pixels brighter than this gray value (default: 230)

  THRESHOLD:
  --thresh_mode global or local (default: local)
  --pct         percentile (default: 80)
  --tile_size   tile size px for local mode (default: 256)

  CLEANUP:
  --min_size1   min object size pass 1 (default: 200)
  --dil_r       dilation disk radius (default: 2)
  --min_size2   min object size pass 2 (default: 400)

  OUTPUT:
  --preview_only  save preview figure and exit (no Frangi processing)
  --save_mask_png also save standalone mask PNG (green on black)
  --dpi           output figure DPI (default: 100)
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
import warnings
warnings.filterwarnings('ignore')

# ── Arguments ─────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--input_dir',     required=True)
p.add_argument('--output_dir',    required=True)
p.add_argument('--pattern',       default='*.tif')

# Crop
p.add_argument('--crop_w',        type=float, default=95.0,
               help='%% of width to keep, centered (default: 95)')
p.add_argument('--crop_h',        type=float, default=95.0,
               help='%% of height to keep, centered (default: 95)')

# Circle mask (in cropped coordinates)
p.add_argument('--cx',            type=int,   default=None,
               help='Circle center x in cropped image (default: cropped_w/2)')
p.add_argument('--cy',            type=int,   default=None,
               help='Circle center y in cropped image (default: cropped_h/2)')
p.add_argument('--radius',        type=int,   default=None,
               help='Circle radius px (default: half of min cropped dimension)')
p.add_argument('--shrink',        type=int,   default=0,
               help='Shrink radius by N px to trim LED arc (default: 0)')

# Frangi
p.add_argument('--sigmas',        nargs=3, type=int, default=[2, 16, 2],
               metavar=('START','STOP','STEP'))
p.add_argument('--clip',          type=float, default=0.03)
p.add_argument('--dark',          type=float, default=50)
p.add_argument('--bright',        type=float, default=230)

# Threshold
p.add_argument('--thresh_mode',   default='local', choices=['global','local'])
p.add_argument('--pct',           type=float, default=80)
p.add_argument('--tile_size',     type=int,   default=256)

# Cleanup
p.add_argument('--min_size1',     type=int,   default=200)
p.add_argument('--dil_r',         type=int,   default=2)
p.add_argument('--min_size2',     type=int,   default=400)

# Misc
p.add_argument('--preview_only',  action='store_true')
p.add_argument('--save_mask_png', action='store_true')
p.add_argument('--dpi',           type=int,   default=100)
args = p.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
sigmas = list(range(args.sigmas[0], args.sigmas[1], args.sigmas[2]))

# ── Load first image to compute crop box ─────────────────────────────────────
files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
if not files:
    print(f"No files found: {args.pattern} in {args.input_dir}"); exit(1)

first_raw = np.array(Image.open(files[0]))
H_raw, W_raw = first_raw.shape[:2]

# Crop box (centered)
keep_w = int(round(W_raw * args.crop_w / 100.0))
keep_h = int(round(H_raw * args.crop_h / 100.0))
x0 = (W_raw - keep_w) // 2
y0 = (H_raw - keep_h) // 2
x1 = x0 + keep_w
y1 = y0 + keep_h

# Cropped dimensions
W_crop, H_crop = keep_w, keep_h

# Circle parameters in cropped coordinates
cx     = args.cx     if args.cx     is not None else W_crop // 2
cy     = args.cy     if args.cy     is not None else H_crop // 2
radius = args.radius if args.radius is not None else min(H_crop, W_crop) // 2
radius_eff = max(1, radius - args.shrink)

# Circle mask on cropped image
Yg, Xg     = np.ogrid[:H_crop, :W_crop]
circle_mask = (Xg - cx)**2 + (Yg - cy)**2 <= radius_eff**2

print("=" * 60)
print(f"Raw image size : {W_raw} x {H_raw} px")
print(f"Crop           : {args.crop_w}% W  x  {args.crop_h}% H")
print(f"Crop box       : x=[{x0}:{x1}]  y=[{y0}:{y1}]")
print(f"Cropped size   : {W_crop} x {H_crop} px")
print(f"Circle (crop coords) : center=({cx},{cy})  "
      f"radius={radius}  effective={radius_eff}  shrink={args.shrink}")
print(f"Sigmas         : {sigmas}")
print(f"Threshold      : {args.thresh_mode}  pct={args.pct}  tile={args.tile_size}")
print("=" * 60)

# ── Crop helper ───────────────────────────────────────────────────────────────
def crop(img_np):
    return img_np[y0:y1, x0:x1]

# ── Preview mode ──────────────────────────────────────────────────────────────
if args.preview_only:
    img_raw    = first_raw
    img_cropped = crop(img_raw)
    theta = np.linspace(0, 2*np.pi, 500)

    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    fig.patch.set_facecolor('#0a0a14')

    # Raw image with crop rectangle
    axes[0].imshow(img_raw)
    rect = plt.Rectangle((x0, y0), keep_w, keep_h,
                          edgecolor='yellow', facecolor='none', lw=2.5,
                          label=f'crop {args.crop_w}%W x {args.crop_h}%H')
    axes[0].add_patch(rect)
    axes[0].legend(fontsize=10)
    axes[0].set_title(f'Raw image ({W_raw}x{H_raw})\nYellow = crop region',
                      color='white', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Cropped image with circle
    axes[1].imshow(img_cropped)
    axes[1].plot(cx + radius_eff*np.cos(theta),
                 cy + radius_eff*np.sin(theta),
                 'lime', lw=2.5, label=f'r={radius_eff}')
    axes[1].plot(cx, cy, 'r+', markersize=18, markeredgewidth=2.5,
                 label=f'center=({cx},{cy})')
    axes[1].legend(fontsize=10)
    axes[1].set_title(f'Cropped image ({W_crop}x{H_crop})\nGreen = circle mask',
                      color='white', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Final valid region
    masked = img_cropped.copy()
    masked[~circle_mask] = [15, 15, 25]
    axes[2].imshow(masked)
    axes[2].set_title('Valid region\n(outside circle excluded)',
                      color='lime', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(
        f'PREVIEW — {os.path.basename(files[0])}\n'
        f'crop_w={args.crop_w}%  crop_h={args.crop_h}%  '
        f'cx={cx}  cy={cy}  radius={radius_eff}\n'
        f'Adjust parameters then rerun without --preview_only',
        color='#ffdd00', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(args.output_dir, 'PREVIEW_crop_circle.png')
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='#0a0a14')
    plt.close()
    print(f"\nPreview saved: {out}")
    exit(0)

# ── Segmentation ──────────────────────────────────────────────────────────────
def process(img_cropped, circle_mask, args, sigmas):
    gray = (0.299*img_cropped[:,:,0] + 0.587*img_cropped[:,:,1]
            + 0.114*img_cropped[:,:,2]).astype(float)
    h, w = gray.shape

    valid   = (gray > args.dark) & (gray < args.bright) & circle_mask
    inv_eq  = exposure.equalize_adapthist(1.0 - gray/255.0, clip_limit=args.clip)
    fmap    = frangi(inv_eq, sigmas=sigmas, alpha=0.5, beta=0.5, black_ridges=False)
    fmax    = fmap.max()
    if fmax == 0:
        return np.zeros((h,w), bool), np.zeros((h,w))
    fnorm_v = (fmap / fmax) * valid

    if args.thresh_mode == 'global':
        nz     = fnorm_v[fnorm_v > 0.001]
        thresh = np.percentile(nz, args.pct) if len(nz) > 0 else 0.05
        binary = fnorm_v > thresh
    else:
        binary = np.zeros((h,w), bool)
        for r in range(0, h, args.tile_size):
            for c in range(0, w, args.tile_size):
                tile = fnorm_v[r:r+args.tile_size, c:c+args.tile_size]
                nz   = tile[tile > 0.001]
                if len(nz) < 30: continue
                t    = np.percentile(nz, args.pct)
                binary[r:r+args.tile_size, c:c+args.tile_size] = tile > t

    binary = binary & circle_mask
    binary = morphology.remove_small_objects(binary, min_size=args.min_size1)
    binary = morphology.binary_dilation(binary, disk(args.dil_r))
    binary = morphology.remove_small_objects(binary, min_size=args.min_size2)
    return binary, fnorm_v

# ── Process all files ─────────────────────────────────────────────────────────
print(f"\nProcessing {len(files)} files...")

for fpath in files:
    fname = os.path.basename(fpath)
    base  = os.path.splitext(fname)[0]
    try:
        img_raw = np.array(Image.open(fpath))
    except Exception as e:
        print(f"  SKIP {fname}: {e}"); continue

    img_c = crop(img_raw)   # cropped image

    binary, fnorm_v = process(img_c, circle_mask, args, sigmas)
    nc  = measure.label(binary).max()
    cov = binary.sum() / binary.size * 100

    # Overlay on cropped image
    ov = img_c.copy().astype(float)
    ov[binary, 0] = ov[binary,0]*0.25
    ov[binary, 1] = np.clip(ov[binary,1]*0.4+130, 0, 255)
    ov[binary, 2] = ov[binary,2]*0.25
    ov[~circle_mask] = ov[~circle_mask] * 0.2

    # Figure: Raw | Cropped | Valid region | Frangi | Mask | Overlay
    fig, axes = plt.subplots(1, 6, figsize=(42, 7))
    fig.patch.set_facecolor('#0a0a14')

    theta = np.linspace(0, 2*np.pi, 400)

    axes[0].imshow(img_raw)
    rect = plt.Rectangle((x0,y0), keep_w, keep_h,
                          edgecolor='yellow', facecolor='none', lw=2)
    axes[0].add_patch(rect)
    axes[0].set_title(f'Raw ({W_raw}x{H_raw})\nYellow=crop region',
                      color='white', fontsize=9, fontweight='bold')

    axes[1].imshow(img_c)
    axes[1].plot(cx + radius_eff*np.cos(theta),
                 cy + radius_eff*np.sin(theta), 'lime', lw=1.5, alpha=0.8)
    axes[1].set_title(f'Cropped ({W_crop}x{H_crop})\n{args.crop_w}%W x {args.crop_h}%H',
                      color='yellow', fontsize=9)

    vdisp = img_c.copy()
    vdisp[~circle_mask] = [15,15,25]
    axes[2].imshow(vdisp)
    axes[2].set_title(f'Valid region\ncx={cx} cy={cy} r={radius_eff}',
                      color='lime', fontsize=9)

    axes[3].imshow(fnorm_v, cmap='hot', vmin=0, vmax=fnorm_v.max())
    axes[3].set_title(f'Frangi heatmap\n{args.thresh_mode} pct={args.pct}',
                      color='#cccccc', fontsize=9)

    axes[4].imshow(binary, cmap='gray')
    axes[4].set_title(f'Binary mask\nnc={nc}  cov={cov:.2f}%',
                      color='#aaffaa', fontsize=9)

    axes[5].imshow(ov.astype(np.uint8))
    axes[5].set_title('Overlay', color='#ffffaa', fontsize=9)

    for ax in axes: ax.axis('off')
    plt.suptitle(f'{base}  |  crop={args.crop_w}%x{args.crop_h}%  '
                 f'circle=({cx},{cy}) r={radius_eff}',
                 color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'{base}_step1.png'),
                dpi=args.dpi, bbox_inches='tight', facecolor='#0a0a14')
    plt.close()

    # Save cropped image
    Image.fromarray(img_c).save(
        os.path.join(args.output_dir, f'{base}_cropped.png'))

    if args.save_mask_png:
        mask_rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
        mask_rgb[binary] = [120, 255, 120]
        Image.fromarray(mask_rgb).save(
            os.path.join(args.output_dir, f'{base}_mask.png'))

    np.savez_compressed(
        os.path.join(args.output_dir, f'{base}_mask.npz'),
        mask=binary, circle_mask=circle_mask,
        crop_box=np.array([x0, y0, x1, y1]))

    print(f"  OK: {fname}  nc={nc}  cov={cov:.2f}%")

print(f"\nDone. Results in: {args.output_dir}")
