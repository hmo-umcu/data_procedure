"""
draw_target_geometry.py
-----------------------
Overlay the target G-code geometry on printed scaffold images and compute IoU.

Print drift correction
----------------------
The 48-well plate is printed column-major: A1→F1, A2→F2, ..., A8→F8.
The printer accumulates a small mechanical drift per well throughout the run.
For any well (col, row), its sequential print number is:
    seq = (col - 1) * 6 + row        col: 1-8 (1-based), row: 0-5 (0-based)

The target geometry centre is adjusted by a per-well offset stored in
WELL_OFFSETS_MM below. Each entry maps filename stem → (dx_mm, dy_mm).

How to add calibration data
----------------------------
For each well-printed image with a reliable predicted mask:
  1. Run HoughCircles on the raw TIF → well centre (wcx, wcy)
  2. Compute centroid of predicted strand mask → (scx, scy)
  3. offset = ((scx - wcx) / PX_PER_MM,  (scy - wcy) / PX_PER_MM)
  4. Add/update the entry in WELL_OFFSETS_MM:
         '3_2': (-0.03, +0.10),
  5. Average across multiple runs of the same well for robustness

For wells without a calibrated entry, the script uses a linear fallback
model (drift_per_well_mm × seq) whose coefficients you set in
DRIFT_FALLBACK below. Set both to 0.0 to disable the fallback entirely.

Usage
-----
    python draw_target_geometry.py <img_dir>
        [--mask_dir <dir>]         folder with *-mask.png (default: img_dir)
        [--output_dir <dir>]       where to save overlays  (default: img_dir)
        [--strand_width_mm <f>]    strand width in mm       (default: 0.41)
        [--strand_gap_mm <f>]      centre-to-centre spacing (default: 2.5)
        [--alpha <f>]              overlay opacity 0–1      (default: 0.5)
        [--iou_threshold <f>]      flag results above this IoU with ★
        [--no_drift]               disable drift correction entirely
"""

import argparse
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import json


# ── geometry constants ────────────────────────────────────────────────────────
PX_PER_MM               = 67.0
DEFAULT_STRAND_WIDTH_MM = 0.41    # 22G nozzle inner diameter
DEFAULT_STRAND_GAP_MM   = 2.5     # centre-to-centre between adjacent strands
N_STRANDS               = 3       # 3 H-strands + 3 V-strands


# ── drift offsets (mm) ───────────────────────────────────────────────────────
# Two independent offsets, both manually tuned:
#
# ROW_STEP: offset added per row step within a column
#   row 0 → no offset, row 1 → 1×ROW_STEP, row 2 → 2×ROW_STEP, ...
#   Filename suffix: col_ROW  e.g. 3_0, 3_1, ..., 3_5
#
# COL_STEP: offset added per column step
#   col 1 → no offset, col 2 → 1×COL_STEP, ..., col 8 → 7×COL_STEP
#   Filename prefix: COL_row  e.g. 1_3, 2_3, ..., 8_3
#
# Total offset for well (col, row):
#   dx = (row) * ROW_STEP[0]  +  (col - 1) * COL_STEP[0]
#   dy = (row) * ROW_STEP[1]  +  (col - 1) * COL_STEP[1]
#
# ── SET THESE VALUES MANUALLY ────────────────────────────────────────────────
ROW_STEP = (-0.15, 0.0)   # (dx_mm, dy_mm) per row increment
COL_STEP = (0.0, 0.10)   # (dx_mm, dy_mm) per column increment


# ── parse filename and compute offset ────────────────────────────────────────
def parse_col_row(stem):
    """Parse 'col_row' stem → (col, row). Returns (None, None) if no match."""
    m = re.match(r'^(\d+)_(\d+)$', stem)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def get_drift_offset(stem, apply_drift=True):
    """
    Compute (dx_mm, dy_mm) for a well from ROW_STEP and COL_STEP.
    Returns (0, 0, 'none') if drift disabled or filename unparseable.
    """
    if not apply_drift:
        return 0.0, 0.0, 'disabled'
    col, row = parse_col_row(stem)
    if col is None:
        return 0.0, 0.0, 'unparsed'
    dx = row * ROW_STEP[0] + (col - 1) * COL_STEP[0]
    dy = row * ROW_STEP[1] + (col - 1) * COL_STEP[1]
    return dx, dy, f'row={row}×{ROW_STEP} col={col-1}×{COL_STEP}'


# ── well centre detection ─────────────────────────────────────────────────────
def read_well_centre_from_json(json_path):
    """Read well centre from a labelme JSON with a 'well' circle shape."""
    with open(json_path) as f:
        data = json.load(f)
    for shape in data.get('shapes', []):
        if shape.get('label') == 'well' and shape.get('shape_type') == 'circle':
            cx, cy = shape['points'][0]
            px, py = shape['points'][1]
            r = float(np.sqrt((px - cx)**2 + (py - cy)**2))
            return float(cx), float(cy), r
    return None


def detect_well_centre_auto(img_rgb):
    """Detect well centre via HoughCircles on the partially visible well arc."""
    import cv2
    h, w = img_rgb.shape[:2]
    gray      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=300, param1=60, param2=40,
        minRadius=300, maxRadius=700
    )
    if circles is None:
        return w / 2, h / 2

    circles = np.round(circles[0]).astype(int)
    best, best_r = None, 0
    for c in circles:
        cx, cy, r = c
        if abs(cx - w/2) < 0.35*w and abs(cy - h/2) < 0.35*h and r > best_r:
            best, best_r = c, r

    return (float(best[0]), float(best[1])) if best is not None else (w/2, h/2)


# ── target geometry mask ──────────────────────────────────────────────────────
def make_target_mask(img_h, img_w, cx, cy,
                     strand_width_mm=DEFAULT_STRAND_WIDTH_MM,
                     strand_gap_mm=DEFAULT_STRAND_GAP_MM,
                     n_strands=N_STRANDS):
    """
    Render the ideal G-code crosshatch as a binary mask (0/1 uint8).
    cx, cy: target geometry centre in pixels (well centre + drift offset).
    """
    half_w_px      = (strand_width_mm / 2) * PX_PER_MM
    half_span      = (n_strands - 1) / 2.0 * strand_gap_mm
    offsets_mm     = [-half_span + i * strand_gap_mm for i in range(n_strands)]
    half_extent_px = (half_span + strand_width_mm / 2) * PX_PER_MM

    canvas = Image.new('L', (img_w, img_h), 0)
    draw   = ImageDraw.Draw(canvas)

    for y_off_mm in offsets_mm:          # horizontal strands
        y_px = cy + y_off_mm * PX_PER_MM
        draw.rectangle([cx - half_extent_px, y_px - half_w_px,
                         cx + half_extent_px, y_px + half_w_px], fill=1)

    for x_off_mm in offsets_mm:          # vertical strands
        x_px = cx + x_off_mm * PX_PER_MM
        draw.rectangle([x_px - half_w_px, cy - half_extent_px,
                         x_px + half_w_px, cy + half_extent_px], fill=1)

    return np.array(canvas, dtype=np.uint8)


# ── IoU ───────────────────────────────────────────────────────────────────────
def compute_iou(pred_mask, target_mask):
    inter = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or (pred_mask, target_mask).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# ── overlay rendering ─────────────────────────────────────────────────────────
def render_overlay(img_rgb, pred_mask, target_mask, alpha=0.5):
    """Red=printed only, Green=target only, Yellow=overlap."""
    overlay = img_rgb.astype(np.float32).copy()
    pred    = pred_mask   > 0
    target  = target_mask > 0
    RED    = np.array([255,  60,  60], dtype=np.float32)
    GREEN  = np.array([ 60, 220,  60], dtype=np.float32)
    YELLOW = np.array([255, 220,   0], dtype=np.float32)
    for mask, colour in [(pred & ~target, RED),
                          (target & ~pred,  GREEN),
                          (pred & target,   YELLOW)]:
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * colour
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ── batch processing ──────────────────────────────────────────────────────────
def process_folder(img_dir, mask_dir, output_dir,
                   strand_width_mm, strand_gap_mm, alpha,
                   iou_threshold, apply_drift):

    img_dir    = Path(img_dir)
    mask_dir   = Path(mask_dir)   if mask_dir   else img_dir
    output_dir = Path(output_dir) if output_dir else img_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(
        p for p in mask_dir.glob('*-mask.png')
        if 'visible' not in p.name and 'target' not in p.name
    )
    if not mask_files:
        print('No *-mask.png files found.')
        return

    print(f'Found {len(mask_files)} mask(s)'
          f'  strand_width={strand_width_mm}mm'
          f'  strand_gap={strand_gap_mm}mm'
          f'  drift={"on" if apply_drift else "off"}\n')

    results = []

    for mask_path in mask_files:
        stem     = mask_path.stem.replace('-mask', '')
        out_path = output_dir / f'{stem}-target-overlay.png'

        # find original TIF
        tif_path = next(
            (img_dir / f'{stem}{ext}'
             for ext in ('.tif', '.tiff', '.TIF', '.TIFF')
             if (img_dir / f'{stem}{ext}').exists()),
            None
        )
        if tif_path is None:
            print(f'[SKIP] no TIF for {stem}')
            continue

        img_rgb   = np.array(Image.open(tif_path).convert('RGB'))
        h, w      = img_rgb.shape[:2]
        pred_mask = np.array(Image.open(mask_path))

        # ── step 1: detect well centre ────────────────────────────────────────
        well_json = img_dir / f'{stem}.json'
        if well_json.exists():
            wres = read_well_centre_from_json(well_json)
            wcx, wcy = (wres[0], wres[1]) if wres else detect_well_centre_auto(img_rgb)
            well_src = 'JSON'
        else:
            wcx, wcy = detect_well_centre_auto(img_rgb)
            well_src = 'auto'

        # ── step 2: apply per-well drift offset ───────────────────────────────
        dx_mm, dy_mm, drift_src = get_drift_offset(stem, apply_drift)
        cx = wcx + dx_mm * PX_PER_MM
        cy = wcy + dy_mm * PX_PER_MM

        # ── step 3: generate target mask and IoU ──────────────────────────────
        target_mask = make_target_mask(h, w, cx, cy, strand_width_mm, strand_gap_mm)
        iou         = compute_iou(pred_mask, target_mask)

        # ── step 4: render and save overlay ───────────────────────────────────
        overlay = render_overlay(img_rgb, pred_mask, target_mask, alpha)
        Image.fromarray(overlay).save(out_path)

        col, row = parse_col_row(stem)
        flag     = '  ★' if iou >= iou_threshold and iou_threshold > 0 else ''
        print(f'[OK] {stem}  IoU={iou:.3f}{flag}'
              f'  well=({wcx:.0f},{wcy:.0f})[{well_src}]'
              f'  drift=({dx_mm:+.2f},{dy_mm:+.2f})mm'
              f'  target=({cx:.0f},{cy:.0f})')
        results.append((stem, iou))

    if results:
        print(f'\n── IoU summary (sorted) ──')
        for stem, iou in sorted(results, key=lambda x: -x[1]):
            flag = '  ★' if iou >= iou_threshold and iou_threshold > 0 else ''
            print(f'  {stem}: {iou:.3f}{flag}')

    print(f'\nDone. Overlays → {output_dir}')


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Overlay G-code target geometry on printed scaffold images.'
    )
    parser.add_argument('img_dir',
        help='Folder containing .tif images')
    parser.add_argument('--mask_dir', '-m', default=None,
        help='Folder with *-mask.png files (default: img_dir)')
    parser.add_argument('--output_dir', '-o', default=None,
        help='Where to save overlay PNGs (default: img_dir)')
    parser.add_argument('--strand_width_mm', type=float,
        default=DEFAULT_STRAND_WIDTH_MM,
        help=f'Strand width mm (default: {DEFAULT_STRAND_WIDTH_MM})')
    parser.add_argument('--strand_gap_mm', type=float,
        default=DEFAULT_STRAND_GAP_MM,
        help=f'Centre-to-centre strand spacing mm (default: {DEFAULT_STRAND_GAP_MM})')
    parser.add_argument('--alpha', type=float, default=0.5,
        help='Overlay opacity 0–1 (default: 0.5)')
    parser.add_argument('--iou_threshold', type=float, default=0.0,
        help='Flag results above this IoU with ★ (default: 0.0 = off)')
    parser.add_argument('--no_drift', action='store_true',
        help='Disable drift correction — use raw well centre only')
    args = parser.parse_args()

    process_folder(
        args.img_dir, args.mask_dir, args.output_dir,
        args.strand_width_mm, args.strand_gap_mm,
        args.alpha, args.iou_threshold,
        apply_drift=not args.no_drift
    )
