"""
image_processing_3_morphology.py
─────────────────────────────────────────────────────────────────
STEP 3: Morphological Cleanup + Largest Component Selection

INPUT : step2_mask_<suffix>.npy + step2_well_circle_<suffix>.npy
        produced by Step 2
OUTPUT: step3_strand_mask_<suffix>.npy  (and diagnostics)

After well-cropping, the mask may still contain small scattered
noise components. This step:
  1. Opens  the mask - removes small isolated noise blobs
  2. Closes the mask - fills small gaps within the strand
  3. Selects the single largest connected component near the
     well centre - this is the printed square construct

USAGE - single mask file:
  python image_processing_3_morphology.py
      --mask       data/processed/step2_mask_0_4.npy
      --output_dir data/processed

USAGE - entire folder (batch):
  python image_processing_3_morphology.py
      --folder     data/processed
      --output_dir data/processed

  Iterates over all step2_mask_*.npy files found in --folder.
  --mask and --folder are mutually exclusive.

OUTPUT FILES per mask (suffix extracted from mask filename, e.g. _0_4):
  step3_morphology_0_4.png      diagnostic figure
  step3_strand_mask_0_4.npy     clean binary strand mask (uint8)
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
from skimage.measure import label, regionprops


# # ── Parameters ────────────────────────────────────────────────────────────────
# OPEN_KERNEL_SIZE   = 3     # removes small noise  (opening)
# OPEN_ITERATIONS    = 1
# CLOSE_KERNEL_SIZE  = 7     # fills gaps in strand (closing)
# CLOSE_ITERATIONS   = 2
# MAX_CENTROID_FRAC  = 0.50  # component centroid must be within this fraction
#                             # of well radius from the well centre


# ── Parameters ────────────────────────────────────────────────────────────────
OPEN_KERNEL_SIZE   = 3     # removes small noise  (opening)
OPEN_ITERATIONS    = 1
CLOSE_KERNEL_SIZE  = 6     # fills gaps in strand (closing)
CLOSE_ITERATIONS   = 6
MAX_CENTROID_FRAC  = 0.50  # component centroid must be within this fraction
                            # of well radius from the well centre


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_suffix(mask_path):
    """
    Extract the _row_col suffix from the mask filename.
    e.g. 'step2_mask_0_4.npy'  ->  '_0_4'
    """
    basename = os.path.splitext(os.path.basename(mask_path))[0]
    m = re.search(r'(_\d+_\d+)$', basename)
    return m.group(1) if m else f'_{basename}'


def collect_masks(folder):
    """
    Return sorted list of step2_mask_*.npy files in folder.
    """
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if re.match(r'step2_mask_\d+_\d+\.npy$', f)
    ])
    return files


# ── Core processing ───────────────────────────────────────────────────────────
def run(mask_path, output_dir):
    """
    Process a single Step 2 mask. Returns the clean strand mask array.
    Reads step2_well_circle_<suffix>.npy from the same directory as mask_path.
    """
    suffix = get_suffix(mask_path)
    os.makedirs(output_dir, exist_ok=True)

    # Well circle lives alongside the Step 2 mask
    mask_dir  = os.path.dirname(mask_path)
    well_path = os.path.join(mask_dir, f'step2_well_circle{suffix}.npy')

    print(f"\n{'='*55}")
    print(f"  STEP 3: Morphological Cleanup + Largest Component")
    print(f"{'='*55}")
    print(f"  Input mask : {mask_path}")
    print(f"  Well circle: {well_path}")
    print(f"  Suffix     : {suffix}")
    print(f"  Output dir : {output_dir}")

    # ── Load Step 2 mask and well circle ──────────────────────────────────────
    mask_in   = np.load(mask_path)
    well_info = np.load(well_path)
    cx_w   = int(well_info[0])
    cy_w   = int(well_info[1])
    well_r = int(well_info[2])
    h, w   = mask_in.shape

    print(f"  Mask size  : {w} x {h} px  |  {mask_in.sum()//255} masked px")
    print(f"  Well centre: ({cx_w}, {cy_w}),  r={well_r} px")

    # ── Step 3a: Opening — remove small noise ─────────────────────────────────
    kern_open  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE))
    mask_open  = cv2.morphologyEx(
        mask_in, cv2.MORPH_OPEN, kern_open, iterations=OPEN_ITERATIONS)
    print(f"  After opening  (k={OPEN_KERNEL_SIZE}, i={OPEN_ITERATIONS}): "
          f"{mask_open.sum()//255} px")

    # ── Step 3b: Closing — fill small gaps in strand ──────────────────────────
    kern_close  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    mask_closed = cv2.morphologyEx(
        mask_open, cv2.MORPH_CLOSE, kern_close, iterations=CLOSE_ITERATIONS)
    print(f"  After closing  (k={CLOSE_KERNEL_SIZE}, i={CLOSE_ITERATIONS}): "
          f"{mask_closed.sum()//255} px")

    # ── Step 3c: Connected components — select the strand ─────────────────────
    labeled  = label(mask_closed > 0)
    props    = regionprops(labeled)
    max_dist = well_r * MAX_CENTROID_FRAC

    print(f"\n  Components found: {len(props)}")
    print(f"  {'Label':>6}  {'Area':>10}  {'BBox W x H':>12}  "
          f"{'Centroid':>18}  {'Dist_ctr':>10}  Status")
    print(f"  {'-'*72}")

    candidates = []
    for p in sorted(props, key=lambda x: -x.area):
        if p.area < 100:
            continue
        bb   = p.bbox
        bh_p = bb[2] - bb[0]
        bw_p = bb[3] - bb[1]
        dist = np.sqrt((p.centroid[1] - cx_w)**2 + (p.centroid[0] - cy_w)**2)
        status = 'CANDIDATE' if dist < max_dist else ''
        print(f"  {p.label:>6}  {p.area:>10.0f}  {bw_p:>5}x{bh_p:<5}  "
              f"  ({p.centroid[1]:>6.0f}, {p.centroid[0]:>6.0f})  "
              f"{dist:>8.0f} px  {status}")
        if dist < max_dist:
            candidates.append(p)

    if not candidates:
        raise ValueError(
            f"No component found within {MAX_CENTROID_FRAC*100:.0f}% of well "
            f"radius from well centre. "
            f"Check HSV threshold (Step 1) or crop fraction (Step 2).")

    best        = max(candidates, key=lambda p: p.area)
    strand_mask = (labeled == best.label).astype(np.uint8) * 255

    ys_m, xs_m = np.where(strand_mask > 0)
    bx0, bx1   = xs_m.min(), xs_m.max()
    by0, by1   = ys_m.min(), ys_m.max()
    bw_f = bx1 - bx0
    bh_f = by1 - by0

    print(f"\n  Selected:  label={best.label},  area={best.area:.0f} px")
    print(f"  Bbox:  x=[{bx0}, {bx1}],  y=[{by0}, {by1}]")
    print(f"  Bbox size:  {bw_f} x {bh_f} px")

    # ── Figure ────────────────────────────────────────────────────────────────
    pad = 60
    zr0 = max(0, by0 - pad);  zr1 = min(h, by1 + pad)
    zc0 = max(0, bx0 - pad);  zc1 = min(w, bx1 + pad)
    zoom_in   = mask_in[zr0:zr1, zc0:zc1]
    zoom_out  = strand_mask[zr0:zr1, zc0:zc1]

    # Colour all components for the component panel
    cmap_cols = [
        (255,  80,  80), ( 80, 255,  80), ( 80,  80, 255),
        (255, 255,  80), (255,  80, 255), ( 80, 255, 255),
        (200, 150,  50),
    ]
    comp_vis = np.zeros((*mask_closed.shape, 3), dtype=np.uint8)
    for i, p in enumerate(sorted(props, key=lambda x: -x.area)[:7]):
        comp_vis[labeled == p.label] = cmap_cols[i % len(cmap_cols)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(mask_in, cmap='gray')
    axes[0, 0].set_title(f'Step 2 mask (input)\n{mask_in.sum()//255} px',
                          fontsize=11)

    axes[0, 1].imshow(mask_open, cmap='gray')
    axes[0, 1].set_title(
        f'After opening  k={OPEN_KERNEL_SIZE}, i={OPEN_ITERATIONS}\n'
        f'{mask_open.sum()//255} px', fontsize=11)

    axes[0, 2].imshow(mask_closed, cmap='gray')
    axes[0, 2].set_title(
        f'After closing  k={CLOSE_KERNEL_SIZE}, i={CLOSE_ITERATIONS}\n'
        f'{mask_closed.sum()//255} px', fontsize=11)

    axes[1, 0].imshow(comp_vis)
    axes[1, 0].set_title(
        f'All connected components\n({len(props)} total, coloured)', fontsize=11)

    axes[1, 1].imshow(strand_mask, cmap='gray')
    axes[1, 1].set_title(
        f'Selected strand component\n{best.area:.0f} px  |  {bw_f}x{bh_f} px bbox',
        fontsize=11)

    # Zoom: input vs output side by side
    comparison = np.hstack([zoom_in, zoom_out])
    axes[1, 2].imshow(comparison, cmap='gray')
    axes[1, 2].axvline(zoom_in.shape[1], color='yellow', lw=1.5, ls='--')
    axes[1, 2].set_title(
        f'Zoom: Step 2 input (left) vs Step 3 output (right)',
        fontsize=11)

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(
        f'Step 3 - Morphological Cleanup + Largest Component  |  '
        f'{os.path.basename(mask_path)}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    fig_path    = os.path.join(output_dir, f'step3_morphology{suffix}.png')
    strand_path = os.path.join(output_dir, f'step3_strand_mask{suffix}.npy')

    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    np.save(strand_path, strand_mask)

    print(f"  Saved: {fig_path}")
    print(f"  Saved: {strand_path}")
    return strand_mask


# ── Batch mode ────────────────────────────────────────────────────────────────
def run_folder(folder, output_dir):
    """Process all step2_mask_*.npy files found in folder."""
    masks = collect_masks(folder)

    if not masks:
        print(f"\n  ERROR: no step2_mask_*.npy files found in: {folder}")
        print(f"  Run Step 2 first to generate them.")
        sys.exit(1)

    print(f"\n{'#'*55}")
    print(f"  BATCH MODE - {len(masks)} mask(s) found")
    print(f"  Folder     : {folder}")
    print(f"  Output dir : {output_dir}")
    print(f"{'#'*55}")

    succeeded = []
    failed    = []

    for i, mask_path in enumerate(masks, 1):
        print(f"\n[{i}/{len(masks)}]  {os.path.basename(mask_path)}")
        try:
            run(mask_path, output_dir)
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
        description='Step 3: Morphological cleanup — operates on Step 2 mask output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  Single mask:\n'
            '    python image_processing_3_morphology.py\n'
            '        --mask       data/processed/step2_mask_0_4.npy\n'
            '        --output_dir data/processed\n\n'
            '  Entire folder (all step2_mask_*.npy files):\n'
            '    python image_processing_3_morphology.py\n'
            '        --folder     data/processed\n'
            '        --output_dir data/processed\n'
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--mask',
        type=str,
        metavar='PATH',
        help='Path to a single step2_mask_*.npy file')
    source.add_argument(
        '--folder',
        type=str,
        metavar='DIR',
        help='Folder containing step2_mask_*.npy files — all will be processed')

    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save outputs (default: .)')

    args = parser.parse_args()

    if args.mask:
        run(args.mask, args.output_dir)
    else:
        run_folder(args.folder, args.output_dir)
