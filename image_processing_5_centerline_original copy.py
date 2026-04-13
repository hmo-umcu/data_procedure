"""
image_processing_5_centerline.py
─────────────────────────────────────────────────────────────────
STEP 5: Centerline Extraction (Skeletonization)

INPUT : step4_mask_<suffix>.npy      (from Step 4)
        step4_bbox_<suffix>.npy      (for zoom region)
        original .tif image          (for visualization only)
OUTPUT: step5_centerline_<suffix>.npy   full-size skeleton mask
        step5_centerline_<suffix>.png   diagnostic figure

The Step 4 mask (strand pixels only, noise outside bbox zeroed)
is skeletonized to extract the strand centerline. The skeleton is:
  - Strand-width independent  → unaffected by strand width variation
  - 1-pixel wide              → suitable for line fitting
  - The geometric foundation for all shape fidelity metrics

The skeleton is the input for Step 6 (shape fidelity metrics):
  side length error, corner angle deviation, side straightness.

VISUALIZATION (3 panels):
  Left   — original colour image
  Middle — Step 4 mask (black/white), zoomed to construct bbox
  Right  — extracted centerline (white on black), same zoom,
            with the 4-side colour assignment overlaid

USAGE - single file:
  python image_processing_5_centerline.py
      --mask       data/processed/step4_mask_0_4.npy
      --image      data/images/lhs_sample_0_4.tif
      --output_dir data/processed

USAGE - folder batch:
  python image_processing_5_centerline.py
      --folder     data/processed
      --image_dir  data/images
      --output_dir data/processed

  --mask/--image and --folder/--image_dir are mutually exclusive.
─────────────────────────────────────────────────────────────────
"""

import argparse
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage.morphology import skeletonize


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_suffix(path):
    """Extract _row_col suffix, e.g. 'step4_mask_0_4.npy' -> '_0_4'."""
    basename = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'(_\d+_\d+)$', basename)
    return m.group(1) if m else f'_{basename}'


def collect_masks(folder):
    """Return sorted list of step4_mask_*.npy files in folder."""
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if re.match(r'step4_mask_\d+_\d+\.npy$', f)
    ])


def find_image(image_dir, suffix):
    """Find original image whose stem ends with suffix."""
    exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    for f in os.listdir(image_dir):
        stem = os.path.splitext(f)[0]
        if stem.endswith(suffix) and f.lower().endswith(exts):
            return os.path.join(image_dir, f)
    raise FileNotFoundError(
        f"No image with suffix '{suffix}' found in: {image_dir}")


def find_bbox(output_dir, suffix):
    """
    Load bbox from step4_bbox_<suffix>.npy if available.
    Returns (x0, y0, x1, y1) or None.
    """
    bbox_path = os.path.join(output_dir, f'step4_bbox{suffix}.npy')
    if os.path.exists(bbox_path):
        b = np.load(bbox_path)
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    return None


def assign_sides(skel_pts, bbox):
    """
    Assign each skeleton pixel to one of 4 sides (top/bottom/left/right)
    based on its distance to the 4 edges of the strand bounding box.
    Corner pixels (within 15% margin on each axis) are excluded (label=-1).

    Returns:
        assignments : int array, length = len(skel_pts)
                      0=top, 1=bottom, 2=left, 3=right, -1=corner/excluded
    """
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    mx = int(bw * 0.15)   # column margin for top/bottom sides
    my = int(bh * 0.15)   # row margin for left/right sides

    ys = skel_pts[:, 0].astype(float)
    xs = skel_pts[:, 1].astype(float)

    dist_top    = ys - y0
    dist_bottom = y1 - ys
    dist_left   = xs - x0
    dist_right  = x1 - xs

    dists    = np.stack([dist_top, dist_bottom, dist_left, dist_right], axis=1)
    raw_side = np.argmin(dists, axis=1)

    assignments = np.full(len(skel_pts), -1, dtype=int)
    for i, (si, py, px) in enumerate(zip(raw_side, ys, xs)):
        if si in (0, 1):   # top / bottom — exclude columns near corners
            if x0 + mx < px < x1 - mx:
                assignments[i] = si
        else:              # left / right — exclude rows near corners
            if y0 + my < py < y1 - my:
                assignments[i] = si

    return assignments


# ── Core processing ───────────────────────────────────────────────────────────
def run(mask_path, image_path, output_dir):
    """
    Skeletonize the Step 4 mask and save the centerline.
    Returns (skeleton array, skeleton points, assignments).
    """
    suffix = get_suffix(mask_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  STEP 5: Centerline Extraction")
    print(f"{'='*55}")
    print(f"  Step 4 mask  : {mask_path}")
    print(f"  Original img : {image_path}")
    print(f"  Suffix       : {suffix}")
    print(f"  Output dir   : {output_dir}")

    # ── Load ──────────────────────────────────────────────────────────────────
    mask_in = np.load(mask_path)
    img_rgb = np.array(Image.open(image_path))
    h, w    = mask_in.shape
    print(f"  Mask size    : {w} x {h} px")
    print(f"  Mask pixels  : {mask_in.sum()//255}")

    # ── Skeletonize ───────────────────────────────────────────────────────────
    skeleton  = skeletonize(mask_in > 0).astype(np.uint8) * 255
    skel_pts  = np.argwhere(skeleton > 0)   # (row, col) = (y, x)
    n_skel    = len(skel_pts)
    print(f"  Skeleton pts : {n_skel}")

    if n_skel < 20:
        raise ValueError(
            f"Too few skeleton pixels ({n_skel}). "
            f"Check Step 4 mask quality.")

    # ── Load bbox (from step4) for zoom and side assignment ───────────────────
    bbox = find_bbox(output_dir, suffix)
    if bbox is None:
        # Fall back to skeleton bounding box
        ys_s, xs_s = skel_pts[:,0], skel_pts[:,1]
        bbox = (int(xs_s.min()), int(ys_s.min()),
                int(xs_s.max()), int(ys_s.max()))
        print(f"  Bbox (skel)  : x=[{bbox[0]},{bbox[2]}]  "
              f"y=[{bbox[1]},{bbox[3]}]  (step4_bbox not found)")
    else:
        print(f"  Bbox (step4) : x=[{bbox[0]},{bbox[2]}]  "
              f"y=[{bbox[1]},{bbox[3]}]")

    x0, y0, x1, y1 = bbox

    # ── Assign skeleton pixels to 4 sides ────────────────────────────────────
    assignments = assign_sides(skel_pts, bbox)

    side_names  = ['top', 'bottom', 'left', 'right']
    side_colors = {
        0: (255,  80,  80),   # top    — red
        1: ( 80, 200,  80),   # bottom — green
        2: ( 80,  80, 255),   # left   — blue
        3: (255, 165,   0),   # right  — orange
    }
    for idx, name in enumerate(side_names):
        n = int((assignments == idx).sum())
        print(f"  Side {name:8s}: {n:4d} skeleton px")
    print(f"  Corners excl.: {int((assignments == -1).sum())} px")

    # ── Build zoom region (bbox with a small margin for display) ──────────────
    pad_vis = 50
    zx0 = max(0, x0 - pad_vis);  zx1 = min(w, x1 + pad_vis)
    zy0 = max(0, y0 - pad_vis);  zy1 = min(h, y1 + pad_vis)

    # ── Visualisation ─────────────────────────────────────────────────────────
    # Build colour skeleton image (RGB) for right panel
    skel_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, pt in enumerate(skel_pts):
        side = assignments[i]
        col  = side_colors.get(side, (200, 200, 200))   # grey = corner/excl.
        skel_rgb[pt[0], pt[1]] = col

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))

    # Left: full original image with bbox rectangle
    axes[0].imshow(img_rgb)
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    axes[0].add_patch(rect)
    axes[0].set_title('Original image\n(cyan = construct bbox)', fontsize=12)

    # Middle: Step 4 mask zoomed to construct region
    axes[1].imshow(mask_in[zy0:zy1, zx0:zx1], cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(
        f'Step 4 mask (zoomed)\n'
        f'Steps 1-4: HSV → well crop → morphology → bbox mask',
        fontsize=12)

    # Right: colour-coded skeleton zoomed to same region
    axes[2].imshow(skel_rgb[zy0:zy1, zx0:zx1])
    # Legend patches
    legend_items = [
        patches.Patch(color=[c/255 for c in side_colors[0]], label='top'),
        patches.Patch(color=[c/255 for c in side_colors[1]], label='bottom'),
        patches.Patch(color=[c/255 for c in side_colors[2]], label='left'),
        patches.Patch(color=[c/255 for c in side_colors[3]], label='right'),
        patches.Patch(color=[0.8, 0.8, 0.8],                 label='corner (excl.)'),
    ]
    axes[2].legend(handles=legend_items, fontsize=8,
                   loc='lower right', framealpha=0.8)
    axes[2].set_title(
        f'Centerline skeleton (zoomed)\n'
        f'{n_skel} px — colour = side assignment',
        fontsize=12)

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(
        f'Step 5 - Centerline Extraction  |  {os.path.basename(image_path)}',
        fontsize=14, fontweight='bold')
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    fig_path  = os.path.join(output_dir, f'step5_centerline{suffix}.png')
    skel_path = os.path.join(output_dir, f'step5_centerline{suffix}.npy')

    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    np.save(skel_path, skeleton)

    print(f"  Saved: {fig_path}")
    print(f"  Saved: {skel_path}")
    return skeleton, skel_pts, assignments


# ── Batch mode ────────────────────────────────────────────────────────────────
def run_folder(folder, image_dir, output_dir):
    """Process all step4_mask_*.npy files in folder."""
    masks = collect_masks(folder)

    if not masks:
        print(f"\n  ERROR: no step4_mask_*.npy files in: {folder}")
        print(f"  Run Step 4 first to generate them.")
        sys.exit(1)

    print(f"\n{'#'*55}")
    print(f"  BATCH MODE - {len(masks)} mask(s) found")
    print(f"  Mask folder  : {folder}")
    print(f"  Image folder : {image_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"{'#'*55}")

    succeeded = []
    failed    = []

    for i, mask_path in enumerate(masks, 1):
        suffix = get_suffix(mask_path)
        print(f"\n[{i}/{len(masks)}]  {os.path.basename(mask_path)}")
        try:
            image_path = find_image(image_dir, suffix)
            run(mask_path, image_path, output_dir)
            succeeded.append(mask_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((mask_path, str(e)))

    print(f"\n{'#'*55}")
    print(f"  BATCH COMPLETE")
    print(f"  Succeeded : {len(succeeded)} / {len(masks)}")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for path, err in failed:
            print(f"    - {os.path.basename(path)}: {err}")
    print(f"  Output dir: {output_dir}")
    print(f"{'#'*55}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 5: Centerline extraction via skeletonization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  Single file:\n'
            '    python image_processing_5_centerline.py\n'
            '        --mask       data/processed/step4_mask_0_4.npy\n'
            '        --image      data/images/lhs_sample_0_4.tif\n'
            '        --output_dir data/processed\n\n'
            '  Entire folder:\n'
            '    python image_processing_5_centerline.py\n'
            '        --folder     data/processed\n'
            '        --image_dir  data/images\n'
            '        --output_dir data/processed\n'
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--mask',
        type=str,
        metavar='PATH',
        help='Path to a single step4_mask_*.npy file')
    mode.add_argument(
        '--folder',
        type=str,
        metavar='DIR',
        help='Folder containing step4_mask_*.npy files')

    parser.add_argument(
        '--image',
        type=str,
        metavar='PATH',
        help='Path to the original .tif image  (required with --mask)')
    parser.add_argument(
        '--image_dir',
        type=str,
        metavar='DIR',
        help='Folder of original .tif images  (required with --folder)')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save outputs (default: .)')

    args = parser.parse_args()

    if args.mask and not args.image:
        parser.error('--mask requires --image')
    if args.folder and not args.image_dir:
        parser.error('--folder requires --image_dir')

    if args.mask:
        run(args.mask, args.image, args.output_dir)
    else:
        run_folder(args.folder, args.image_dir, args.output_dir)
