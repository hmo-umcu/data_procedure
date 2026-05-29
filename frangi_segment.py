"""
STEP 1 v2 — Frangi Segmentation with Local Thresholding
=========================================================
Adds --thresh_mode local to fix the bottom strand issue:

WHY THE BOTTOM STRAND IS MISSED:
  The global percentile threshold is dominated by the high Frangi
  response in the LED ring / well rim region. This pushes the
  threshold up, cutting off weaker (but real) strand signals.

FIX:
  --thresh_mode local  divides the Frangi map into tiles and
  computes the percentile threshold independently within each tile.
  This makes the threshold adaptive to LOCAL contrast, so a strand
  that is weaker due to uneven illumination still gets detected.

Usage:
    python step1_frangi_segment_v2.py \
        --input_dir  ./images \
        --output_dir ./step1_out \
        --thresh_mode local \
        --tile_size   256 \
        --pct         80

Key parameters:
    --thresh_mode   global or local (default: local)
    --tile_size     tile size for local mode in pixels (default: 256)
                    Smaller tiles = more adaptive but more noise
                    Larger tiles  = more uniform but may miss faint strands
                    Recommended: 128–512
    --pct           percentile within each tile (default: 80)
    --sigmas        Frangi scale range (default: 2 16 2)
    --clip          CLAHE clip limit (default: 0.03)
    --dark          exclude pixels darker than this (default: 50)
    --bright        exclude pixels brighter than this (default: 230)
    --min_size1     remove small objects after threshold (default: 200)
    --dil_r         dilation radius (default: 2)
    --min_size2     remove small objects after dilation (default: 400)
    --center_margin border exclusion fraction (default: 0.15)
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
p.add_argument('--sigmas',        nargs=3, type=int, default=[2, 16, 2],
               metavar=('START','STOP','STEP'))
p.add_argument('--clip',          type=float, default=0.03)
p.add_argument('--dark',          type=float, default=50)
p.add_argument('--bright',        type=float, default=230)
p.add_argument('--thresh_mode',   default='global', choices=['global','local'],
               help='global=single percentile, local=per-tile (fixes bottom strand)')
p.add_argument('--pct',           type=float, default=70)
p.add_argument('--tile_size',     type=int,   default=64,
               help='Tile size in pixels for local mode (default: 256)')
p.add_argument('--min_size1',     type=int,   default=200)
p.add_argument('--dil_r',         type=int,   default=2)
p.add_argument('--min_size2',     type=int,   default=400)
p.add_argument('--center_margin', type=float, default=0.05)
p.add_argument('--save_mask_png', action='store_true')
p.add_argument('--dpi',           type=int,   default=100)
args = p.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
sigmas = list(range(args.sigmas[0], args.sigmas[1], args.sigmas[2]))

print(f"Mode:           {args.thresh_mode}")
print(f"Sigmas:         {sigmas}")
print(f"Percentile:     {args.pct}")
print(f"Tile size:      {args.tile_size}  (local mode only)")
print(f"Valid range:    {args.dark} – {args.bright}")
print(f"Center margin:  {args.center_margin*100:.0f}%")

# ── Thresholding functions ────────────────────────────────────────────────────
def threshold_global(fnorm, pct, center):
    nz     = fnorm[fnorm > 0.001]
    thresh = np.percentile(nz, pct) if len(nz) > 0 else 0.05
    return fnorm > thresh, thresh

def threshold_local(fnorm, pct, tile_size, center, min_px=30):
    h, w   = fnorm.shape
    result = np.zeros((h, w), bool)
    thresh_map = np.zeros((h, w), float)
    for r in range(0, h, tile_size):
        for c in range(0, w, tile_size):
            tile     = fnorm[r:r+tile_size, c:c+tile_size]
            nz_tile  = tile[tile > 0.001]
            if len(nz_tile) < min_px:
                continue
            t = np.percentile(nz_tile, pct)
            result[r:r+tile_size, c:c+tile_size]     = tile > t
            thresh_map[r:r+tile_size, c:c+tile_size] = t
    return result & center, thresh_map

# ── Core segmentation ─────────────────────────────────────────────────────────
def process(img_np, args, sigmas):
    R = img_np[:,:,0].astype(float)
    G = img_np[:,:,1].astype(float)
    B = img_np[:,:,2].astype(float)
    gray = 0.299*R + 0.587*G + 0.114*B
    h, w = gray.shape

    my = int(h * args.center_margin)
    mx = int(w * args.center_margin)
    center = np.zeros((h, w), bool)
    center[my:h-my, mx:w-mx] = True
    valid = (gray > args.dark) & (gray < args.bright) & center

    inv_eq = exposure.equalize_adapthist(1.0 - gray/255.0, clip_limit=args.clip)
    fmap   = frangi(inv_eq, sigmas=sigmas, alpha=0.5, beta=0.5, black_ridges=False)
    fmax   = fmap.max()
    if fmax == 0:
        return np.zeros((h,w), bool), 0.0, np.zeros((h,w)), np.zeros((h,w))

    fnorm_v = (fmap / fmax) * valid

    if args.thresh_mode == 'global':
        binary, thresh_val = threshold_global(fnorm_v, args.pct, center)
        thresh_display = thresh_val
    else:
        binary, thresh_map = threshold_local(fnorm_v, args.pct,
                                              args.tile_size, center)
        thresh_display = thresh_map  # show map as heatmap

    binary = morphology.remove_small_objects(binary, min_size=args.min_size1)
    binary = morphology.binary_dilation(binary, disk(args.dil_r))
    binary = morphology.remove_small_objects(binary, min_size=args.min_size2)

    return binary, thresh_display, fnorm_v, valid

# ── Process files ─────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
print(f"\nFound {len(files)} files.")

for fpath in files:
    fname = os.path.basename(fpath)
    base  = os.path.splitext(fname)[0]
    try:
        img = np.array(Image.open(fpath))
    except Exception as e:
        print(f"  SKIP {fname}: {e}"); continue

    binary, thresh_display, fnorm_v, valid = process(img, args, sigmas)
    nc  = measure.label(binary).max()
    cov = binary.sum() / binary.size * 100

    # Overlay
    ov = img.copy().astype(float)
    ov[binary, 0] = ov[binary, 0] * 0.25
    ov[binary, 1] = np.clip(ov[binary, 1] * 0.4 + 130, 0, 255)
    ov[binary, 2] = ov[binary, 2] * 0.25

    # Figure: Raw | Frangi | Thresh map/val | Mask | Overlay
    fig, axes = plt.subplots(1, 5, figsize=(35, 7))
    fig.patch.set_facecolor('#0a0a14')

    axes[0].imshow(img)
    axes[0].set_title(f'Raw — {base}', color='white', fontsize=10, fontweight='bold')

    axes[1].imshow(fnorm_v, cmap='hot', vmin=0, vmax=fnorm_v.max())
    axes[1].set_title('Frangi heatmap', color='#cccccc', fontsize=10)

    if args.thresh_mode == 'local':
        axes[2].imshow(thresh_display, cmap='viridis')
        axes[2].set_title(f'Local threshold map\ntile={args.tile_size} pct={args.pct}',
                          color='#f1a44e', fontsize=10)
    else:
        axes[2].imshow(fnorm_v > thresh_display, cmap='gray')
        axes[2].set_title(f'Global threshold\nval={thresh_display:.4f} pct={args.pct}',
                          color='#f1a44e', fontsize=10)

    axes[3].imshow(binary, cmap='gray')
    axes[3].set_title(f'Binary mask\nnc={nc}  cov={cov:.2f}%', color='#aaffaa', fontsize=10)

    axes[4].imshow(ov.astype(np.uint8))
    axes[4].set_title('Overlay', color='#ffffaa', fontsize=10)

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'{base}_step1.png'),
                dpi=args.dpi, bbox_inches='tight', facecolor='#0a0a14')
    plt.close()

    if args.save_mask_png:
        mask_rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
        mask_rgb[binary] = [120, 255, 120]
        Image.fromarray(mask_rgb).save(
            os.path.join(args.output_dir, f'{base}_mask.png'))

    np.savez_compressed(os.path.join(args.output_dir, f'{base}_mask.npz'),
                        mask=binary)
    print(f"  OK: {fname}  mode={args.thresh_mode}  nc={nc}  cov={cov:.2f}%")

print(f"\nDone. Outputs in: {args.output_dir}")
