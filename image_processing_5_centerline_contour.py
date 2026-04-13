import argparse
import os
import re
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.spatial import cKDTree


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
    """Load bbox [x0,y0,x1,y1] from step4_bbox_<suffix>.npy if available."""
    bbox_path = os.path.join(output_dir, f'step4_bbox{suffix}.npy')
    if os.path.exists(bbox_path):
        b = np.load(bbox_path)
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    return None


def assign_sides(centerline_pts, bbox):
    """
    Assign each centerline point to one of 4 sides based on distance
    to the 4 edges of the strand bbox. Corner pixels (within 15% margin
    on each axis) are excluded (label = -1).

    Returns int array: 0=top, 1=bottom, 2=left, 3=right, -1=excluded
    """
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    mx = int(bw * 0.15)
    my = int(bh * 0.15)

    xs = centerline_pts[:, 0].astype(float)
    ys = centerline_pts[:, 1].astype(float)

    dist_top    = ys - y0
    dist_bottom = y1 - ys
    dist_left   = xs - x0
    dist_right  = x1 - xs

    dists    = np.stack([dist_top, dist_bottom, dist_left, dist_right], axis=1)
    raw_side = np.argmin(dists, axis=1)

    assignments = np.full(len(centerline_pts), -1, dtype=int)
    for i, (si, py, px) in enumerate(zip(raw_side, ys, xs)):
        if si in (0, 1):          # top / bottom
            if x0 + mx < px < x1 - mx:
                assignments[i] = si
        else:                     # left / right
            if y0 + my < py < y1 - my:
                assignments[i] = si
    return assignments


# ── Core: contour extraction and midpoint centerline ─────────────────────────
def extract_centerline(mask, min_inner_area=1000):
    """
    Extract the midpoint centerline of the strand from the binary mask.
    Robust to open/gapped squares — does not rely on flood fill.

    Steps:
      1. Outer contour  : RETR_EXTERNAL on the mask
      2. Convex hull    : fill the convex hull of the outer contour
                          as a solid polygon (bridges any gaps)
      3. Interior region: hull_fill minus strand mask = inside of square.
                          Keep only the largest component (ignore noise).
      4. Inner contour  : boundary of the interior region
      5. Centerline     : for each outer contour point, find the nearest
                          inner contour point (KD-tree) and take midpoint

    Args:
        mask           : uint8 binary mask (0/255) from Step 4
        min_inner_area : minimum area in px for a valid inner component
                         (filters out small noise fragments)

    Returns:
        outer_cnt      : outer contour array (N,2) [x,y]
        inner_cnt      : inner contour array (M,2) [x,y]
        centerline_pts : midpoint array     (N,2) [x,y]
        centerline_img : full-size uint8 mask (0/255) of centerline
    """
    h, w   = mask.shape
    binary = (mask > 0).astype(np.uint8)

    # ── Step 1: outer contour ─────────────────────────────────────────────────
    contours_o, _ = cv2.findContours(
        binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours_o:
        raise ValueError("No outer contour found in mask.")
    outer_cnt_raw = max(contours_o, key=cv2.contourArea)
    outer_cnt     = outer_cnt_raw.reshape(-1, 2)   # [x, y]

    # ── Step 2: fill convex hull — bridges any gaps in the perimeter ──────────
    hull     = cv2.convexHull(outer_cnt_raw)
    hull_img = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(hull_img, [hull], 255)

    # ── Step 3: interior = hull fill minus strand mask ────────────────────────
    interior = cv2.bitwise_and(hull_img,
                               cv2.bitwise_not(binary * 255))

    # Keep only components large enough to be the true square interior
    contours_i, _ = cv2.findContours(
        interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    large = [c for c in contours_i if cv2.contourArea(c) >= min_inner_area]
    if not large:
        raise ValueError(
            f"No inner component with area >= {min_inner_area} px found. "
            f"The square may be too narrow, fully open, or the mask too sparse.")

    # ── Step 4: inner contour ─────────────────────────────────────────────────
    inner_cnt_raw = max(large, key=cv2.contourArea)
    inner_cnt     = inner_cnt_raw.reshape(-1, 2)   # [x, y]

    # ── Step 5: midpoint centerline ───────────────────────────────────────────
    tree           = cKDTree(inner_cnt)
    _, idxs        = tree.query(outer_cnt)
    matched_inner  = inner_cnt[idxs]
    centerline_pts = ((outer_cnt.astype(float) +
                       matched_inner.astype(float)) / 2).astype(int)

    # Rasterise to image with slightly thicker centerline
    centerline_img = np.zeros((h, w), dtype=np.uint8)
    for pt in centerline_pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(centerline_img, (x, y), radius=5, color=255, thickness=3)

    return outer_cnt, inner_cnt, centerline_pts, centerline_img


# ── Core processing ───────────────────────────────────────────────────────────
def run(mask_path, image_path, output_dir):
    """
    Extract centerline from Step 4 mask and save outputs.
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
    print(f"  Mask size    : {w} x {h} px  |  {mask_in.sum()//255} px")

    # ── Extract centerline ────────────────────────────────────────────────────
    outer_cnt, inner_cnt, centerline_pts, centerline_img = \
        extract_centerline(mask_in)

    print(f"  Outer contour: {len(outer_cnt)} pts  "
          f"area={cv2.contourArea(outer_cnt.reshape(-1,1,2)):.0f} px²")
    print(f"  Inner contour: {len(inner_cnt)} pts  "
          f"area={cv2.contourArea(inner_cnt.reshape(-1,1,2)):.0f} px²")
    print(f"  Centerline   : {len(centerline_pts)} pts")
    print(f"  Method       : convex hull fill → interior subtraction"
          f"  (robust to open/gapped squares)")

    # ── Load bbox for zoom and side assignment ────────────────────────────────
    bbox = find_bbox(output_dir, suffix)
    if bbox is None:
        xs, ys = centerline_pts[:,0], centerline_pts[:,1]
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()))
        print(f"  Bbox (derived): x=[{bbox[0]},{bbox[2]}] y=[{bbox[1]},{bbox[3]}]")
    else:
        print(f"  Bbox (step4) : x=[{bbox[0]},{bbox[2]}] y=[{bbox[1]},{bbox[3]}]")

    x0, y0, x1, y1 = bbox

    # ── Side assignment ───────────────────────────────────────────────────────
    assignments = assign_sides(centerline_pts, bbox)
    side_names  = ['top', 'bottom', 'left', 'right']
    side_colors_rgb = {
        0: (255,  80,  80),   # top    — red
        1: ( 80, 200,  80),   # bottom — green
        2: ( 80,  80, 255),   # left   — blue
        3: (255, 165,   0),   # right  — orange
    }
    for idx, name in enumerate(side_names):
        n = int((assignments == idx).sum())
        print(f"  Side {name:8s}: {n:4d} pts")
    print(f"  Corners excl.: {int((assignments == -1).sum())} pts")

    # ── Zoom region for display ───────────────────────────────────────────────
    pad_vis = 50
    zx0 = max(0, x0 - pad_vis);  zx1 = min(w, x1 + pad_vis)
    zy0 = max(0, y0 - pad_vis);  zy1 = min(h, y1 + pad_vis)

    # ── Build colour centerline image ─────────────────────────────────────────
    skel_rgb = np.ones((h, w, 3), dtype=np.uint8) * 255
    for i, pt in enumerate(centerline_pts):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            side = assignments[i]
            cv2.circle(
                skel_rgb,
                (x, y),
                radius=1,
                color=side_colors_rgb.get(side, (180, 180, 180)),
                thickness=-1
            )

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))

    # Left: original image
    axes[0].imshow(img_rgb)
    rect = patches.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    axes[0].add_patch(rect)
    axes[0].set_title('Original image\n(cyan = construct bbox)', fontsize=12)

    # Middle: Step 4 mask with outer and inner contours drawn, zoomed
    mask_vis = cv2.cvtColor(mask_in, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(mask_vis,
                     [outer_cnt.reshape(-1,1,2)], -1, (0,255,0), 3)
    cv2.drawContours(mask_vis,
                     [inner_cnt.reshape(-1,1,2)], -1, (255,80,80), 3)
    axes[1].imshow(mask_vis[zy0:zy1, zx0:zx1])
    axes[1].set_title(
        'Step 4 mask (zoomed)\nGreen = outer contour  |  Red = inner contour',
        fontsize=12)

    # Right: colour-coded centerline zoomed
    axes[2].imshow(skel_rgb[zy0:zy1, zx0:zx1])

    legend_items = [
        patches.Patch(color=[c/255 for c in side_colors_rgb[0]], label='top'),
        patches.Patch(color=[c/255 for c in side_colors_rgb[1]], label='bottom'),
        patches.Patch(color=[c/255 for c in side_colors_rgb[2]], label='left'),
        patches.Patch(color=[c/255 for c in side_colors_rgb[3]], label='right'),
        patches.Patch(color=[0.7, 0.7, 0.7],                     label='corner (excl.)'),
    ]
    axes[2].legend(handles=legend_items, fontsize=9,
                   loc='lower right', framealpha=0.85)
    axes[2].set_title(
        f'Centerline (zoomed)  —  {len(centerline_pts)} pts\n'
        f'Midpoint of outer & inner contours  |  colour = side',
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
    np.save(skel_path, centerline_img)

    print(f"  Saved: {fig_path}")
    print(f"  Saved: {skel_path}")
    return centerline_img, centerline_pts, assignments


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
        description='Step 5: Centerline via outer/inner contour midpoint',
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
    mode.add_argument('--mask',   type=str, metavar='PATH',
                      help='Path to a single step4_mask_*.npy file')
    mode.add_argument('--folder', type=str, metavar='DIR',
                      help='Folder containing step4_mask_*.npy files')

    parser.add_argument('--image',     type=str, metavar='PATH',
                        help='Original .tif image  (required with --mask)')
    parser.add_argument('--image_dir', type=str, metavar='DIR',
                        help='Folder of original images  (required with --folder)')
    parser.add_argument('--output_dir', type=str, default='.',
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