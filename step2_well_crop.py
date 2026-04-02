"""
image_processing_1_hsv_mask.py
─────────────────────────────────────────────────────────────────
STEP 1: HSV Red Mask  (S > 180)

Converts the image to HSV colour space and creates a binary mask
for pixels that are:
  - Red hue  : hue distance from 0° < 20  (covers H<20 and H>160)
  - Highly saturated: S > 180

This threshold was confirmed to segment the red Pluronic F127
strand while excluding low-saturation ring light and weakly
saturated surface reflections.

USAGE:
  python image_processing_1_hsv_mask.py \
      --image C:/Users/hmo/hmo_workspace/data/images/lhs_sample_0_4.tif \
      --output_dir data/processed

OUTPUT FILES  (suffix extracted from filename, e.g. _0_4):
  step1_hsv_mask_0_4.png    diagnostic figure
  step1_mask_0_4.npy        binary mask (uint8, 0/255)
─────────────────────────────────────────────────────────────────
"""

import argparse
import os
import re
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


# ── Parameters ────────────────────────────────────────────────────────────────
HUE_DIST_MAX = 20    # max hue distance from pure red (H=0 or H=180)
SAT_MIN      = 200   # minimum saturation — key threshold from analysis
VAL_MIN      = 30    # exclude near-black pixels (noise)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_suffix(image_path):
    """
    Extract the _row_col suffix from the image filename.
    e.g. 'lhs_sample_0_4.tif'  →  '_0_4'
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    m = re.search(r'(_\d+_\d+)$', basename)
    return m.group(1) if m else ''


# ── Main ──────────────────────────────────────────────────────────────────────
def run(image_path, output_dir):
    suffix = get_suffix(image_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  STEP 1: HSV Red Mask")
    print(f"{'='*55}")
    print(f"  Image      : {image_path}")
    print(f"  Suffix     : {suffix}")
    print(f"  Output dir : {output_dir}")
    print(f"  Params     : hue_dist<{HUE_DIST_MAX}, S>{SAT_MIN}, V>{VAL_MIN}")

    # ── Load ──────────────────────────────────────────────────────────────────
    img_rgb = np.array(Image.open(image_path))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, w    = img_rgb.shape[:2]
    print(f"  Image size : {w} x {h} px")

    H_ch = img_hsv[:, :, 0].astype(np.float32)
    S_ch = img_hsv[:, :, 1].astype(np.float32)
    V_ch = img_hsv[:, :, 2].astype(np.float32)

    # Hue distance from pure red (wraps at 0 / 180)
    hue_dist = np.minimum(H_ch, 180.0 - H_ch)

    # ── Binary mask ───────────────────────────────────────────────────────────
    mask = (
        (hue_dist < HUE_DIST_MAX) &
        (S_ch     > SAT_MIN)      &
        (V_ch     > VAL_MIN)
    ).astype(np.uint8) * 255

    n_px = int(mask.sum() // 255)
    print(f"  Masked pixels: {n_px}  ({100 * n_px / (h * w):.1f}% of image)")

    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        print(f"  Raw bbox: x=[{xs.min()}, {xs.max()}]  "
              f"y=[{ys.min()}, {ys.max()}]")
        print(f"  NOTE: bbox may span full image — border noise removed in Step 2.")

    # ── Overlay ───────────────────────────────────────────────────────────────
    overlay = img_rgb.copy()
    overlay[mask > 0] = [0, 255, 0]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original image', fontsize=11)

    axes[0, 1].imshow(H_ch, cmap='hsv', vmin=0, vmax=180)
    axes[0, 1].set_title('Hue channel', fontsize=11)
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])

    axes[0, 2].imshow(S_ch, cmap='gray')
    axes[0, 2].set_title('Saturation channel', fontsize=11)
    plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2])

    axes[1, 0].imshow(hue_dist, cmap='hot_r', vmin=0, vmax=45)
    axes[1, 0].set_title('Hue distance from red\n(0 = pure red)', fontsize=11)
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0])

    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title(
        f'Step 1 mask\nhue_dist<{HUE_DIST_MAX}, S>{SAT_MIN}, V>{VAL_MIN}\n'
        f'{n_px} px', fontsize=11)

    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(
        'Mask overlay (green)\nBorder noise → removed in Step 2', fontsize=11)

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(f'Step 1 — HSV Red Mask  |  {os.path.basename(image_path)}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    fig_path  = os.path.join(output_dir, f'step1_hsv_mask{suffix}.png')
    mask_path = os.path.join(output_dir, f'step1_mask{suffix}.npy')

    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    np.save(mask_path, mask)

    print(f"\n  Saved: {fig_path}")
    print(f"  Saved: {mask_path}")
    return mask


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 1: HSV red mask for bioprint strand segmentation')
    parser.add_argument(
        '--image',
        type=str, required=True,
        help='Path to input .tif image, e.g. data/images/lhs_sample_0_4.tif')
    parser.add_argument(
        '--output_dir',
        type=str, default='.',
        help='Directory to save outputs (created if missing)')
    args = parser.parse_args()
    run(args.image, args.output_dir)