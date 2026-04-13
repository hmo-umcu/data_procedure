"""
image_processing_2_well_crop.py
─────────────────────────────────────────────────────────────────
STEP 2: Well Interior Crop - Remove Border Noise

INPUT : step1_mask_<suffix>.npy produced by Step 1
OUTPUT: step2_mask_<suffix>.npy  (and diagnostics)

The S>180 mask from Step 1 contains noise pixels at the well
boundary (ring light edge, scattered reflections). This step
applies a circular crop mask defined by a user-specified radius
and centre to keep only the relevant inner region of the well.

You specify the crop circle directly in pixels — no automatic
detection. Use the Step 1 diagnostic figure to estimate the
appropriate values for your setup.

USAGE - single mask file:
  python image_processing_2_well_crop.py
      --mask         data/processed/step1_mask_0_4.npy
      --crop_radius  340
      --output_dir   data/processed

  Optionally override the crop centre (default: image centre):
      --crop_center  640 512

USAGE - entire folder (batch):
  python image_processing_2_well_crop.py
      --folder       data/processed
      --crop_radius  340
      --output_dir   data/processed

  Same --crop_radius and --crop_center applied to all masks.
  --mask and --folder are mutually exclusive.

HOW TO CHOOSE --crop_radius:
  Open the step1_hsv_mask_*.png diagnostic figure.
  The ring light appears as a bright ring near the well edge.
  Set --crop_radius to just inside that ring in pixels.
  Typical range: 300-400 px depending on your Dino-Lite zoom.

OUTPUT FILES per mask (suffix extracted from mask filename, e.g. _0_4):
  step2_well_crop_0_4.png       diagnostic figure
  step2_mask_0_4.npy            cropped binary mask
  step2_well_circle_0_4.npy     [cx, cy, crop_radius] in px
─────────────────────────────────────────────────────────────────
"""

import argparse
import os
import re
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_suffix(mask_path):
    """
    Extract the _row_col suffix from the mask filename.
    e.g. 'step1_mask_0_4.npy'  ->  '_0_4'
    """
    basename = os.path.splitext(os.path.basename(mask_path))[0]
    m = re.search(r'(_\d+_\d+)$', basename)
    return m.group(1) if m else f'_{basename}'


def collect_masks(folder):
    """Return sorted list of step1_mask_*.npy files in folder."""
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if re.match(r'step1_mask_\d+_\d+\.npy$', f)
    ])


# ── Core processing ───────────────────────────────────────────────────────────
def run(mask_path, crop_radius, crop_center, output_dir):
    """
    Apply a circular crop to a single Step 1 mask.

    Args:
        mask_path   : path to step1_mask_*.npy
        crop_radius : radius of the crop circle in pixels
        crop_center : (cx, cy) in pixels, or None to use image centre
        output_dir  : directory to write outputs
    Returns:
        mask_out    : cropped binary mask (uint8 array)
    """
    suffix = get_suffix(mask_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  STEP 2: Well Interior Crop")
    print(f"{'='*55}")
    print(f"  Input mask   : {mask_path}")
    print(f"  Suffix       : {suffix}")
    print(f"  Output dir   : {output_dir}")

    # ── Load Step 1 mask ──────────────────────────────────────────────────────
    mask_in = np.load(mask_path)
    h, w    = mask_in.shape
    print(f"  Mask size    : {w} x {h} px  |  {mask_in.sum()//255} masked px")

    # ── Resolve crop centre ───────────────────────────────────────────────────
    if crop_center is None:
        cx, cy = w // 2, h // 2
        print(f"  Crop centre  : ({cx}, {cy})  [image centre — default]")
    else:
        cx, cy = int(crop_center[0]), int(crop_center[1])
        print(f"  Crop centre  : ({cx}, {cy})  [user specified]")

    print(f"  Crop radius  : {crop_radius} px")

    # ── Build circular crop mask ──────────────────────────────────────────────
    yy, xx      = np.mgrid[0:h, 0:w]
    dist        = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.float32)
    circle_crop = (dist <= crop_radius).astype(np.uint8) * 255

    # ── Apply crop ────────────────────────────────────────────────────────────
    mask_out = cv2.bitwise_and(mask_in, circle_crop)
    n_before = int(mask_in.sum()  // 255)
    n_after  = int(mask_out.sum() // 255)
    print(f"  Pixels before: {n_before}")
    print(f"  Pixels after : {n_after}  (removed {n_before - n_after})")

    # ── Figure ────────────────────────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 500)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel 1: Step 1 mask with crop circle overlaid
    axes[0, 0].imshow(mask_in, cmap='gray')
    axes[0, 0].plot(cx + crop_radius * cos_t,
                    cy + crop_radius * sin_t,
                    'c-', lw=2, label=f'crop r={crop_radius} px')
    axes[0, 0].plot(cx, cy, 'r+', markersize=12,
                    markeredgewidth=2, label=f'centre ({cx},{cy})')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    axes[0, 0].set_title('Step 1 mask + crop circle', fontsize=11)

    # Panel 2: Step 1 mask (input)
    axes[0, 1].imshow(mask_in, cmap='gray')
    axes[0, 1].set_title(f'Step 1 mask (input)\n{n_before} px', fontsize=11)

    # Panel 3: circular crop mask
    axes[0, 2].imshow(circle_crop, cmap='gray')
    axes[0, 2].set_title(f'Circular crop mask\n(r={crop_radius} px, '
                          f'centre=({cx},{cy}))', fontsize=11)

    # Panel 4: Step 2 mask (output)
    axes[1, 0].imshow(mask_out, cmap='gray')
    axes[1, 0].set_title(f'Step 2 mask (output)\n{n_after} px', fontsize=11)

    # Panel 5: removed pixels
    removed = cv2.bitwise_and(mask_in, cv2.bitwise_not(mask_out))
    axes[1, 1].imshow(removed, cmap='hot')
    axes[1, 1].set_title(f'Removed pixels\n{n_before - n_after} px', fontsize=11)

    # Panel 6: before / after side-by-side
    comparison = np.hstack([mask_in, mask_out])
    axes[1, 2].imshow(comparison, cmap='gray')
    axes[1, 2].axvline(w, color='yellow', lw=1.5, ls='--')
    axes[1, 2].set_title('Before (left) vs After (right)', fontsize=11)

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(
        f'Step 2 - Well Interior Crop  |  {os.path.basename(mask_path)}\n'
        f'crop_radius={crop_radius} px,  centre=({cx},{cy})',
        fontsize=12, fontweight='bold')
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    fig_path   = os.path.join(output_dir, f'step2_well_crop{suffix}.png')
    mask2_path = os.path.join(output_dir, f'step2_mask{suffix}.npy')
    well_path  = os.path.join(output_dir, f'step2_well_circle{suffix}.npy')

    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    np.save(mask2_path, mask_out)
    np.save(well_path,  np.array([cx, cy, crop_radius]))

    print(f"  Saved: {fig_path}")
    print(f"  Saved: {mask2_path}")
    print(f"  Saved: {well_path}")
    return mask_out


# ── Batch mode ────────────────────────────────────────────────────────────────
def run_folder(folder, crop_radius, crop_center, output_dir):
    """Process all step1_mask_*.npy files found in folder."""
    masks = collect_masks(folder)

    if not masks:
        print(f"\n  ERROR: no step1_mask_*.npy files found in: {folder}")
        print(f"  Run Step 1 first to generate them.")
        sys.exit(1)

    print(f"\n{'#'*55}")
    print(f"  BATCH MODE - {len(masks)} mask(s) found")
    print(f"  Folder       : {folder}")
    print(f"  Crop radius  : {crop_radius} px")
    cx_str = f"({crop_center[0]}, {crop_center[1]})" \
             if crop_center else "image centre"
    print(f"  Crop centre  : {cx_str}")
    print(f"  Output dir   : {output_dir}")
    print(f"{'#'*55}")

    succeeded = []
    failed    = []

    for i, mask_path in enumerate(masks, 1):
        print(f"\n[{i}/{len(masks)}]  {os.path.basename(mask_path)}")
        try:
            run(mask_path, crop_radius, crop_center, output_dir)
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
        description='Step 2: Well interior crop — user-specified circular crop',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  Single mask:\n'
            '    python image_processing_2_well_crop.py\n'
            '        --mask         data/processed/step1_mask_0_4.npy\n'
            '        --crop_radius  340\n'
            '        --output_dir   data/processed\n\n'
            '  Single mask with custom centre:\n'
            '    python image_processing_2_well_crop.py\n'
            '        --mask         data/processed/step1_mask_0_4.npy\n'
            '        --crop_radius  340\n'
            '        --crop_center  650 520\n'
            '        --output_dir   data/processed\n\n'
            '  Entire folder:\n'
            '    python image_processing_2_well_crop.py\n'
            '        --folder       data/processed\n'
            '        --crop_radius  340\n'
            '        --output_dir   data/processed\n'
        )
    )

    # --mask / --folder: mutually exclusive, one required
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--mask',
        type=str,
        metavar='PATH',
        help='Path to a single step1_mask_*.npy file')
    source.add_argument(
        '--folder',
        type=str,
        metavar='DIR',
        help='Folder containing step1_mask_*.npy files — all will be processed')

    parser.add_argument(
        '--crop_radius',
        type=int,
        required=True,
        metavar='PX',
        help='Radius of the circular crop in pixels. '
             'Open the step1_hsv_mask_*.png figure to estimate — '
             'set to just inside the ring light border.')

    parser.add_argument(
        '--crop_center',
        type=int,
        nargs=2,
        metavar=('CX', 'CY'),
        default=None,
        help='Centre of the crop circle as two integers: CX CY  '
             '(default: image centre)')

    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save outputs (default: .)')

    args = parser.parse_args()

    if args.mask:
        run(args.mask, args.crop_radius, args.crop_center, args.output_dir)
    else:
        run_folder(args.folder, args.crop_radius, args.crop_center, args.output_dir)
