"""
image_processing_4_crop.py
─────────────────────────────────────────────────────────────────
STEP 4: Mask Outside Construct Region to Zero

INPUT : step3_strand_mask_<suffix>.npy  (from Step 3)
        original .tif image              (for visualization only)
OUTPUT: step4_mask_<suffix>.npy          full-size mask, zeroed
                                         outside the construct bbox
        step4_crop_<suffix>.png          diagnostic figure

The bounding box of the strand from Step 3 defines the construct
region. Everything outside that box (plus optional padding) is
set to zero in the mask. The interior of the box is kept as-is
from Step 3.

Visualization:
  Left  — original image (colour)
  Right — Step 4 mask (black = zeroed outside, white = strand)
           with the kept region highlighted by a cyan rectangle

USAGE - single file:
  python image_processing_4_crop.py
      --mask       data/processed/step3_strand_mask_0_4.npy
      --image      data/images/lhs_sample_0_4.tif
      --output_dir data/processed

USAGE - folder batch:
  python image_processing_4_crop.py
      --folder     data/processed
      --image_dir  data/images
      --output_dir data/processed

  --mask/--image and --folder/--image_dir are mutually exclusive.

OPTIONAL:
  --pad  N    padding around bbox in pixels (default: 40)
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


# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_PAD = 40


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_suffix(path):
    """
    Extract _row_col suffix from filename.
    e.g. 'step3_strand_mask_0_4.npy'  ->  '_0_4'
    """
    basename = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'(_\d+_\d+)$', basename)
    return m.group(1) if m else f'_{basename}'


def collect_masks(folder):
    """Return sorted list of step3_strand_mask_*.npy files in folder."""
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if re.match(r'step3_strand_mask_\d+_\d+\.npy$', f)
    ])


def find_image(image_dir, suffix):
    """
    Find original image in image_dir whose stem ends with suffix.
    Raises FileNotFoundError if not found.
    """
    exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    for f in os.listdir(image_dir):
        stem = os.path.splitext(f)[0]
        if stem.endswith(suffix) and f.lower().endswith(exts):
            return os.path.join(image_dir, f)
    raise FileNotFoundError(
        f"No image with suffix '{suffix}' found in: {image_dir}")


# ── Core processing ───────────────────────────────────────────────────────────
def run(mask_path, image_path, pad, output_dir):
    """
    Zero out everything in the Step 3 mask that lies outside the
    construct bounding box (+ padding). Save the result as a
    full-size mask and produce a side-by-side diagnostic figure.
    """
    suffix = get_suffix(mask_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  STEP 4: Mask Outside Construct Region to Zero")
    print(f"{'='*55}")
    print(f"  Step 3 mask  : {mask_path}")
    print(f"  Original img : {image_path}")
    print(f"  Suffix       : {suffix}")
    print(f"  Padding      : {pad} px")
    print(f"  Output dir   : {output_dir}")

    # ── Load ──────────────────────────────────────────────────────────────────
    mask_in = np.load(mask_path)          # full-size Step 3 mask
    img_rgb = np.array(Image.open(image_path))
    h, w    = mask_in.shape
    print(f"  Image size   : {w} x {h} px")
    print(f"  Step 3 pixels: {mask_in.sum()//255}")

    # ── Compute bounding box of strand ────────────────────────────────────────
    ys, xs = np.where(mask_in > 0)
    x0_raw, x1_raw = int(xs.min()), int(xs.max())
    y0_raw, y1_raw = int(ys.min()), int(ys.max())

    # Apply padding (clamped to image bounds)
    x0 = max(0, x0_raw - pad)
    y0 = max(0, y0_raw - pad)
    x1 = min(w, x1_raw + pad)
    y1 = min(h, y1_raw + pad)

    print(f"  Strand bbox  : x=[{x0_raw},{x1_raw}]  y=[{y0_raw},{y1_raw}]")
    print(f"  Kept region  : x=[{x0},{x1}]  y=[{y0},{y1}]  (pad={pad}px)")

    # ── Zero everything outside the bbox ──────────────────────────────────────
    mask_out = np.zeros_like(mask_in)          # start fully black
    mask_out[y0:y1, x0:x1] = mask_in[y0:y1, x0:x1]   # restore only bbox region

    n_kept    = int(mask_out.sum() // 255)
    n_removed = int(mask_in.sum() // 255) - n_kept
    print(f"  Pixels kept  : {n_kept}")
    print(f"  Pixels zeroed: {n_removed}  (outside bbox)")

    # ── Visualization ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: original image
    axes[0].imshow(img_rgb)
    # Draw the kept region as a cyan dashed rectangle
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor='cyan', facecolor='none',
        linestyle='--', label=f'kept region ({x1-x0}x{y1-y0} px)')
    axes[0].add_patch(rect)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].set_title('Original image\n(dashed cyan = kept region)',
                       fontsize=12)

    # Right: Step 4 mask (full-size, black outside / white strand)
    axes[1].imshow(mask_out, cmap='gray', vmin=0, vmax=255)
    # Same rectangle overlay on mask
    rect2 = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    axes[1].add_patch(rect2)
    axes[1].set_title(
        f'Step 4 mask\n'
        f'Black = zeroed outside bbox  |  White = strand\n'
        f'Steps: HSV mask → well crop → morphology → bbox mask',
        fontsize=12)

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(
        f'Step 4 - Mask Outside Construct Region to Zero  |  '
        f'{os.path.basename(image_path)}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    fig_path  = os.path.join(output_dir, f'step4_crop{suffix}.png')
    mask_path_out = os.path.join(output_dir, f'step4_mask{suffix}.npy')
    bbox_path = os.path.join(output_dir, f'step4_bbox{suffix}.npy')

    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    np.save(mask_path_out, mask_out)
    np.save(bbox_path, np.array([x0, y0, x1, y1, pad]))

    print(f"  Saved: {fig_path}")
    print(f"  Saved: {mask_path_out}")
    print(f"  Saved: {bbox_path}")
    return mask_out


# ── Batch mode ────────────────────────────────────────────────────────────────
def run_folder(folder, image_dir, pad, output_dir):
    """Process all step3_strand_mask_*.npy files in folder."""
    masks = collect_masks(folder)

    if not masks:
        print(f"\n  ERROR: no step3_strand_mask_*.npy files in: {folder}")
        print(f"  Run Step 3 first to generate them.")
        sys.exit(1)

    print(f"\n{'#'*55}")
    print(f"  BATCH MODE - {len(masks)} mask(s) found")
    print(f"  Mask folder  : {folder}")
    print(f"  Image folder : {image_dir}")
    print(f"  Padding      : {pad} px")
    print(f"  Output dir   : {output_dir}")
    print(f"{'#'*55}")

    succeeded = []
    failed    = []

    for i, mask_path in enumerate(masks, 1):
        suffix = get_suffix(mask_path)
        print(f"\n[{i}/{len(masks)}]  {os.path.basename(mask_path)}")
        try:
            image_path = find_image(image_dir, suffix)
            run(mask_path, image_path, pad, output_dir)
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
        description='Step 4: Zero out mask outside construct bounding box',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  Single file:\n'
            '    python image_processing_4_crop.py\n'
            '        --mask       data/processed/step3_strand_mask_0_4.npy\n'
            '        --image      data/images/lhs_sample_0_4.tif\n'
            '        --output_dir data/processed\n\n'
            '  With custom padding:\n'
            '    python image_processing_4_crop.py\n'
            '        --mask       data/processed/step3_strand_mask_0_4.npy\n'
            '        --image      data/images/lhs_sample_0_4.tif\n'
            '        --pad        60\n'
            '        --output_dir data/processed\n\n'
            '  Entire folder:\n'
            '    python image_processing_4_crop.py\n'
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
        help='Path to a single step3_strand_mask_*.npy file')
    mode.add_argument(
        '--folder',
        type=str,
        metavar='DIR',
        help='Folder containing step3_strand_mask_*.npy files')

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
        '--pad',
        type=int,
        default=DEFAULT_PAD,
        metavar='PX',
        help=f'Padding around bounding box in pixels (default: {DEFAULT_PAD})')
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
        run(args.mask, args.image, args.pad, args.output_dir)
    else:
        run_folder(args.folder, args.image_dir, args.pad, args.output_dir)
