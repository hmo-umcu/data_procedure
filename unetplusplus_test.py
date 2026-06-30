"""
unetplusplus_test.py
--------------------
Load a trained U-Net++ model and run inference on a test folder.

Input test folder must contain:
    {sid}_{row}.tif              original image         REQUIRED
    {sid}_{row}-mask.png         annotation mask        REQUIRED (for comparison)
    {sid}_{row}-mask-visible.png annotation visible     copied as-is
    {sid}_{row}.json             labelme JSON           copied as-is
    {sid}_{row}-target-overlay.png target overlay       copied as-is

Usage
-----
    python unetplusplus_test.py
        --model_path  /home/hmo/.../models/unetplusplus/run_01/best_model.pth
        --data_dir    /home/hmo/.../data/dev_images/dev_annot_test
        --output_dir  /home/hmo/.../data/dev_images/dev_annot_test_pred
        [--img_size   512]
        [--threshold  0.5]
        [--no_gpu]

Output
------
    <output_dir>/
        {sid}_{row}.tif                  original image (copied)
        {sid}_{row}-mask.png             annotation mask (copied)
        {sid}_{row}-mask-visible.png     annotation visible (copied)
        {sid}_{row}.json                 labelme JSON (copied)
        {sid}_{row}-target-overlay.png   target overlay (copied)
        {sid}_{row}-pred-mask.png        predicted binary mask (0/1)
        {sid}_{row}-pred-visible.png     original + prediction overlay (green)
        {sid}_{row}-pred-vs-annot.png    original + pred (green/red) + annot (red/blue)
        test_info.csv                    per-image metadata table (;-separated)
"""

import argparse
import csv
import shutil
import numpy as np
from pathlib import Path
from PIL import Image


# ── colour constants ──────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PRED_COLOUR  = np.array([60,  220, 60 ], dtype=np.float32)   # green
ANNOT_COLOUR = np.array([220, 60,  60 ], dtype=np.float32)   # red
ALPHA = 0.45


def preprocess_image(img_rgb, img_size):
    """Resize + ImageNet normalise → [3,H,W] float32 numpy."""
    import cv2
    img  = cv2.resize(img_rgb, (img_size, img_size),
                      interpolation=cv2.INTER_LINEAR)
    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    return img_f.transpose(2, 0, 1)   # CHW


def overlay_masks(img_rgb, pred_mask=None, annot_mask=None):
    """
    Colour overlay:
        green  = prediction only
        red    = annotation only
        yellow = overlap (pred AND annot)
    """
    out = img_rgb.astype(np.float32).copy()
    if pred_mask is not None:
        pb = pred_mask > 0
        if annot_mask is not None:
            ab       = annot_mask > 0
            overlap  = pb & ab
            pred_only  = pb & ~ab
            annot_only = ab & ~pb
            out[pred_only]  = (1-ALPHA)*out[pred_only]  + ALPHA*PRED_COLOUR
            out[annot_only] = (1-ALPHA)*out[annot_only] + ALPHA*ANNOT_COLOUR
            out[overlap]    = (1-ALPHA)*out[overlap]    + ALPHA*np.array([255,220,0],
                                                                          dtype=np.float32)
        else:
            out[pb] = (1-ALPHA)*out[pb] + ALPHA*PRED_COLOUR
    return np.clip(out, 0, 255).astype(np.uint8)


def run_test(model_path, data_dir, output_dir, img_size, threshold, use_gpu):
    import torch
    import segmentation_models_pytorch as smp

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available()
                          else 'cpu')

    # ── load model ────────────────────────────────────────────────────────────
    print(f'Loading model: {model_path}')
    checkpoint = torch.load(str(model_path), map_location=device)
    arch       = checkpoint.get('arch',    'unetplusplus')
    encoder    = checkpoint.get('encoder', 'resnet34')

    arch_map = {
        'unetplusplus': smp.UnetPlusPlus,
        'unet':         smp.Unet,
        'fpn':          smp.FPN,
    }
    model = arch_map[arch](
        encoder_name=encoder,
        encoder_weights=None,   # weights loaded from checkpoint
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f'  Architecture : {arch} ({encoder})')
    print(f'  Trained epoch: {checkpoint.get("epoch", "?")}')
    if 'val_iou' in checkpoint:
        print(f'  Best val IoU : {checkpoint["val_iou"]:.4f}')

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

    print(f'\nFound {len(tif_files)} test images\n')

    # ── copy all companion files ───────────────────────────────────────────────
    COPY_SUFFIXES = ['-mask.png', '-mask-visible.png',
                     '-target-overlay.png', '.json']
    for tif_path in tif_files:
        stem = tif_path.stem
        shutil.copy2(tif_path, output_dir / tif_path.name)
        for suf in COPY_SUFFIXES:
            src = data_dir / f'{stem}{suf}'
            if src.exists():
                shutil.copy2(src, output_dir / src.name)

    # ── run inference ─────────────────────────────────────────────────────────
    csv_rows = []

    with torch.no_grad():
        for tif_path in tif_files:
            stem    = tif_path.stem
            img_rgb = np.array(Image.open(tif_path).convert('RGB'))
            h_orig, w_orig = img_rgb.shape[:2]

            print(f'  Predicting: {tif_path.name}')

            # preprocess → tensor
            img_t = preprocess_image(img_rgb, img_size)
            img_t = torch.from_numpy(img_t).unsqueeze(0).to(device)

            # inference
            logits = model(img_t)                      # [1,1,H,W]
            prob   = torch.sigmoid(logits).squeeze().cpu().numpy()

            # resize back to original resolution
            import cv2
            prob_full = cv2.resize(prob, (w_orig, h_orig),
                                   interpolation=cv2.INTER_LINEAR)
            pred_binary = (prob_full > threshold).astype(np.uint8)

            # save predicted mask
            Image.fromarray(pred_binary).save(
                output_dir / f'{stem}-pred-mask.png')

            # save pred-visible
            pred_vis = overlay_masks(img_rgb, pred_mask=pred_binary)
            Image.fromarray(pred_vis).save(
                output_dir / f'{stem}-pred-visible.png')

            # save pred-vs-annot
            annot_path = data_dir / f'{stem}-mask.png'
            if annot_path.exists():
                annot_arr = np.array(Image.open(annot_path))
                if annot_arr.ndim == 3:
                    annot_arr = annot_arr[:, :, 0]
                annot_binary = (annot_arr > 0).astype(np.uint8)
            else:
                annot_binary = None

            both_vis = overlay_masks(img_rgb,
                                     pred_mask=pred_binary,
                                     annot_mask=annot_binary)
            Image.fromarray(both_vis).save(
                output_dir / f'{stem}-pred-vs-annot.png')

            # CSV row
            parts = stem.split('_')
            sid   = parts[0] if len(parts) >= 2 else stem
            row_  = parts[1] if len(parts) >= 2 else ''

            csv_rows.append({
                'stem':            stem,
                'Sample_ID':       sid,
                'row':             row_,
                'tif':             tif_path.name,
                'pred_mask':       f'{stem}-pred-mask.png',
                'pred_visible':    f'{stem}-pred-visible.png',
                'pred_vs_annot':   f'{stem}-pred-vs-annot.png',
                'annot_mask':      f'{stem}-mask.png'
                                   if annot_binary is not None else '',
                'pred_px':         int(pred_binary.sum()),
                'annot_px':        int(annot_binary.sum())
                                   if annot_binary is not None else '',
                'iou_pred_annot':  '',   # filled by evaluate.py
                'iou_pred_target': '',
                'dice_pred_annot': '',
                'pixel_acc':       '',
            })

    # ── write test_info.csv ───────────────────────────────────────────────────
    csv_path   = output_dir / 'test_info.csv'
    fieldnames = [
        'stem', 'Sample_ID', 'row', 'tif',
        'pred_mask', 'pred_visible', 'pred_vs_annot', 'annot_mask',
        'pred_px', 'annot_px',
        'iou_pred_annot', 'iou_pred_target', 'dice_pred_annot', 'pixel_acc',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f'\n✓ Predictions saved → {output_dir}')
    print(f'✓ Test info table  → {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run U-Net++ inference on a test folder.'
    )
    parser.add_argument('--model_path', required=True,
        help='Path to best_model.pth or final_model.pth')
    parser.add_argument('--data_dir',   required=True,
        help='Folder with test *.tif + *-mask.png files')
    parser.add_argument('--output_dir', required=True,
        help='Where to save predictions and copies of inputs')
    parser.add_argument('--img_size',   type=int, default=512,
        help='Inference resize (must match training, default: 512)')
    parser.add_argument('--threshold',  type=float, default=0.5,
        help='Sigmoid threshold for binary prediction (default: 0.5)')
    parser.add_argument('--no_gpu',     action='store_true')
    args = parser.parse_args()

    run_test(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        threshold=args.threshold,
        use_gpu=not args.no_gpu,
    )
