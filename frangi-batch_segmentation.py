"""
Frangi Batch Segmentation
==========================
Processes all .tif images in a folder and saves results.

Usage:
    python frangi_batch.py --input_dir path/to/images --output_dir path/to/output

Optional:
    --pattern      glob pattern to filter files (default: *.tif)
    --sigmas       Frangi sigma values (default: 2 4 6 8 10 12 14)
    --clip         CLAHE clip limit (default: 0.03)
    --dark         exclude pixels darker than this (default: 50)
    --bright       exclude pixels brighter than this (default: 230)
    --pct          Frangi threshold percentile (default: 80)
    --min_size1    min object size first pass (default: 200)
    --dil_r        dilation disk radius (default: 2)
    --min_size2    min object size second pass (default: 400)
    --cols         number of columns per figure row: 2=raw+mask, 4=all (default: 4)
    --dpi          output figure DPI (default: 100)
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure, morphology, measure
from skimage.filters import frangi
from skimage.morphology import disk, skeletonize
import warnings
warnings.filterwarnings('ignore')

# ── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',  required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--pattern',    default='*.tif')
parser.add_argument('--sigmas',     nargs='+', type=int, default=[2,4,6,8,10,12,14])
parser.add_argument('--clip',       type=float, default=0.03)
parser.add_argument('--dark',       type=float, default=50)
parser.add_argument('--bright',     type=float, default=230)
parser.add_argument('--pct',        type=float, default=60)
parser.add_argument('--min_size1',  type=int,   default=200)
parser.add_argument('--dil_r',      type=int,   default=2)
parser.add_argument('--min_size2',  type=int,   default=400)
parser.add_argument('--cols',       type=int,   default=4, choices=[2,4])
parser.add_argument('--dpi',        type=int,   default=100)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Core segmentation ─────────────────────────────────────────────────────────
def process(img_np, args):
    R = img_np[:,:,0].astype(float)
    G = img_np[:,:,1].astype(float)
    B = img_np[:,:,2].astype(float)
    gray_raw = 0.299*R + 0.587*G + 0.114*B

    valid   = (gray_raw > args.dark) & (gray_raw < args.bright)
    inv     = 1.0 - gray_raw / 255.0
    inv_eq  = exposure.equalize_adapthist(inv, clip_limit=args.clip)

    fmap    = frangi(inv_eq, sigmas=args.sigmas,
                     alpha=0.5, beta=0.5, black_ridges=False)
    fmax    = fmap.max()
    if fmax == 0:
        return inv_eq, np.zeros_like(fmap), np.zeros(img_np.shape[:2], bool), 0.0

    fnorm_v = (fmap / fmax) * valid
    nz      = fnorm_v[fnorm_v > 0.001]
    thresh  = np.percentile(nz, args.pct) if len(nz) > 0 else 0.05

    binary  = fnorm_v > thresh
    binary  = morphology.remove_small_objects(binary, min_size=args.min_size1)
    binary  = morphology.binary_dilation(binary, disk(args.dil_r))
    binary  = morphology.remove_small_objects(binary, min_size=args.min_size2)

    return inv_eq, fnorm_v, binary, thresh

# ── Process files ─────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
if not files:
    print(f"No files found matching {args.pattern} in {args.input_dir}")
    exit(1)

print(f"Found {len(files)} files. Processing...")

for fpath in files:
    fname     = os.path.basename(fpath)
    fname_base = os.path.splitext(fname)[0]

    try:
        img = np.array(Image.open(fpath))
    except Exception as e:
        print(f"  SKIP {fname}: {e}")
        continue

    inv_eq, fnorm_v, binary, thresh = process(img, args)
    nc  = measure.label(binary).max()
    cov = binary.sum() / binary.size * 100

    # Save individual PNGs
    # -- Mask
    mask_rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
    mask_rgb[binary] = [120, 255, 120]
    Image.fromarray(mask_rgb).save(
        os.path.join(args.output_dir, f"{fname_base}_mask.png"))

    # -- Overlay + skeleton
    skel = skeletonize(binary) if binary.sum() > 0 else np.zeros_like(binary)
    ov   = img.copy().astype(np.float32)
    ov[binary, 0] = ov[binary, 0] * 0.25
    ov[binary, 1] = np.clip(ov[binary, 1] * 0.4 + 130, 0, 255)
    ov[binary, 2] = ov[binary, 2] * 0.25
    sk2  = morphology.binary_dilation(skel, disk(1))
    ov   = ov.astype(np.uint8)
    ov[sk2, 0] = 255
    ov[sk2, 1] = 240
    ov[sk2, 2] = 20
    Image.fromarray(ov).save(
        os.path.join(args.output_dir, f"{fname_base}_overlay.png"))

    # -- Figure with all panels
    ncols = args.cols
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 7))
    fig.patch.set_facecolor('#0a0a14')

    axes[0].imshow(img)
    axes[0].set_title(f"Raw — {fname_base}", color='white', fontsize=11, fontweight='bold')

    if ncols == 4:
        axes[1].imshow(inv_eq, cmap='gray')
        axes[1].set_title('Inverted+CLAHE', color='#cccccc', fontsize=10)
        axes[2].imshow(fnorm_v, cmap='hot', vmin=0, vmax=fnorm_v.max())
        axes[2].set_title(f'Frangi heatmap\n(thresh={thresh:.3f})', color='#cccccc', fontsize=10)
        axes[3].imshow(binary, cmap='gray')
        axes[3].set_title(f'Binary mask\nnc={nc}  cov={cov:.1f}%', color='#aaffaa', fontsize=10)
    else:
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title(f'Binary mask\n(thresh={thresh:.3f})  nc={nc}  cov={cov:.1f}%',
                          color='#aaffaa', fontsize=10)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"{fname_base}_result.png"),
                dpi=args.dpi, bbox_inches='tight', facecolor='#0a0a14')
    plt.close()
    print(f"  OK: {fname}  thresh={thresh:.3f}  nc={nc}  cov={cov:.1f}%")

print(f"\nDone. Results saved to: {args.output_dir}")
