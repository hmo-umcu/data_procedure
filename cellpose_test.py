"""
test.py
-------
Load a fine-tuned Cellpose-SAM model and run prediction on a test folder.

Input test folder must contain:
    {sid}_{row}.tif              original image         REQUIRED
    {sid}_{row}-mask.png         annotation mask        REQUIRED (for comparison)
    {sid}_{row}-mask-visible.png annotation visible     copied as-is
    {sid}_{row}.json             labelme JSON           copied as-is
    {sid}_{row}-target-overlay.png target overlay       copied as-is

Usage
-----
    python test.py
        --model_path  /home/hmo/models/cellpose/fold_0/cpsam_scaffold
        --data_dir    /scratch-shared/hmo/scaffold_images/test
        --output_dir  /scratch-shared/hmo/scaffold_images/test_predictions
        [--no_gpu]

Output
------
    <output_dir>/
        {sid}_{row}.tif                  original image (copied)
        {sid}_{row}-mask.png             annotation mask (copied)
        {sid}_{row}-mask-visible.png     annotation visible (copied)
        {sid}_{row}.json                 labelme JSON (copied)
        {sid}_{row}-target-overlay.png   target overlay (copied)
        {sid}_{row}-pred-mask.png        predicted mask (0/1 uint8)
        {sid}_{row}-pred-visible.png     original + predicted mask overlay
        {sid}_{row}-pred-vs-annot.png    original + pred mask + annot mask
        test_info.csv                    per-image metadata table
"""

import argparse
import csv
import shutil
import numpy as np
from pathlib import Path
from PIL import Image


# ── colours for overlays ──────────────────────────────────────────────────────
PRED_COLOUR  = np.array([60, 180, 60],   dtype=np.float32)  # green  = prediction
ANNOT_COLOUR = np.array([220, 60, 60],   dtype=np.float32)  # red    = annotation
ALPHA        = 0.45


def overlay_masks(img_rgb, pred_mask=None, annot_mask=None):
    """
    Blend prediction and/or annotation masks onto image.
    pred_mask  → green
    annot_mask → red
    overlap    → yellow
    """
    out = img_rgb.astype(np.float32).copy()
    if pred_mask is not None:
        pb = pred_mask > 0
        if annot_mask is not None:
            ab = annot_mask > 0
            overlap = pb & ab
            pred_only  = pb & ~ab
            annot_only = ab & ~pb
            out[pred_only]  = (1-ALPHA)*out[pred_only]  + ALPHA*PRED_COLOUR
            out[annot_only] = (1-ALPHA)*out[annot_only] + ALPHA*ANNOT_COLOUR
            out[overlap]    = (1-ALPHA)*out[overlap]    + ALPHA*np.array([255,220,0], dtype=np.float32)
        else:
            out[pb] = (1-ALPHA)*out[pb] + ALPHA*PRED_COLOUR
    return np.clip(out, 0, 255).astype(np.uint8)


def run_test(model_path, data_dir, output_dir, use_gpu):
    from cellpose import models as cp_models

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    print(f'Loading model: {model_path}')
    model = cp_models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=str(model_path),
    )

    # ── collect test TIFs ─────────────────────────────────────────────────────
    tif_files = sorted(
        p for p in data_dir.glob('*.tif')
        if not any(x in p.name for x in ['visible', 'overlay', 'target'])
    )
    tif_files += sorted(
        p for p in data_dir.glob('*.tiff')
        if not any(x in p.name for x in ['visible', 'overlay', 'target'])
    )

    if not tif_files:
        print('[ERROR] No .tif files found in data_dir.')
        return

    print(f'Found {len(tif_files)} test images\n')

    # ── copy all input files first ────────────────────────────────────────────
    COPY_SUFFIXES = [
        '-mask.png', '-mask-visible.png', '-target-overlay.png', '.json'
    ]
    for tif_path in tif_files:
        stem = tif_path.stem
        # copy original TIF
        shutil.copy2(tif_path, output_dir / tif_path.name)
        # copy companion files if they exist
        for suf in COPY_SUFFIXES:
            src = data_dir / f'{stem}{suf}'
            if src.exists():
                shutil.copy2(src, output_dir / src.name)

    # ── run prediction ────────────────────────────────────────────────────────
    csv_rows = []

    for tif_path in tif_files:
        stem    = tif_path.stem
        img_rgb = np.array(Image.open(tif_path).convert('RGB'))

        # print(f'  Predicting: {tif_path.name}')
        # pred_masks, flows, styles = model.eval(
        #     img_rgb,
        #     diameter=None,
        #     normalize=True,
        # )

        print(f'  Predicting: {tif_path.name}')
        pred_masks, flows, styles = model.eval(
            img_rgb,
            diameter=100,
            normalize=True,
        )

        # binary predicted mask (0/1 uint8)
        pred_binary = (pred_masks > 0).astype(np.uint8)

        # save predicted mask
        pred_mask_path = output_dir / f'{stem}-pred-mask.png'
        Image.fromarray(pred_binary).save(pred_mask_path)

        # save pred-visible (original + prediction overlay)
        pred_vis = overlay_masks(img_rgb, pred_mask=pred_binary)
        Image.fromarray(pred_vis).save(output_dir / f'{stem}-pred-visible.png')

        # save pred-vs-annot (original + pred + annotation)
        annot_path = data_dir / f'{stem}-mask.png'
        if annot_path.exists():
            annot_arr = np.array(Image.open(annot_path))
            if annot_arr.ndim == 3:
                annot_arr = annot_arr[:, :, 0]
            annot_binary = (annot_arr > 0).astype(np.uint8)
        else:
            annot_binary = None

        both_vis = overlay_masks(img_rgb, pred_mask=pred_binary, annot_mask=annot_binary)
        Image.fromarray(both_vis).save(output_dir / f'{stem}-pred-vs-annot.png')

        # parse sid and row from stem
        parts = stem.split('_')
        sid   = parts[0] if len(parts) >= 2 else stem
        row   = parts[1] if len(parts) >= 2 else ''

        csv_rows.append({
            'stem':            stem,
            'Sample_ID':       sid,
            'row':             row,
            'tif':             tif_path.name,
            'pred_mask':       f'{stem}-pred-mask.png',
            'pred_visible':    f'{stem}-pred-visible.png',
            'pred_vs_annot':   f'{stem}-pred-vs-annot.png',
            'annot_mask':      f'{stem}-mask.png' if annot_binary is not None else '',
            'pred_px':         int(pred_binary.sum()),
            'annot_px':        int(annot_binary.sum()) if annot_binary is not None else '',
            'iou_pred_annot':  '',   # filled by evaluate.py
            'iou_pred_target': '',   # filled by evaluate.py
        })

    # ── write test_info.csv ───────────────────────────────────────────────────
    csv_path = output_dir / 'test_info.csv'
    fieldnames = [
        'stem', 'Sample_ID', 'row', 'tif',
        'pred_mask', 'pred_visible', 'pred_vs_annot', 'annot_mask',
        'pred_px', 'annot_px', 'iou_pred_annot', 'iou_pred_target',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f'\n✓ Predictions saved → {output_dir}')
    print(f'✓ Test info table  → {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Cellpose-SAM inference on a test folder.'
    )
    parser.add_argument('--model_path', required=True,
        help='Path to fine-tuned model (e.g. .../cpsam_scaffold)')
    parser.add_argument('--data_dir', required=True,
        help='Folder with test *.tif + *-mask.png files')
    parser.add_argument('--output_dir', required=True,
        help='Where to save predictions and copies of inputs')
    parser.add_argument('--no_gpu', action='store_true',
        help='Disable GPU')
    args = parser.parse_args()

    run_test(
        args.model_path, args.data_dir, args.output_dir,
        use_gpu=not args.no_gpu,
    )