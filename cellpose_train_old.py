"""
train.py
--------
Fine-tune Cellpose-SAM (cpsam) on a folder of annotated scaffold images.

Input folder must contain pairs of:
    {sid}_{row}.tif          original scaffold image
    {sid}_{row}-mask.png     strand_clean mask from json_to_mask.py
                             pixel values: 0=background, >0=strand_clean
                             pores already subtracted inside json_to_mask.py

Usage
-----
    python train.py
        --data_dir   /scratch-shared/hmo/scaffold_images/train
        --model_dir  /home/hmo/models/cellpose/fold_0
        [--n_epochs      100]
        [--learning_rate 1e-5]
        [--weight_decay  0.1]
        [--min_size      500]
        [--no_gpu]

Output
------
    <model_dir>/
        cpsam_scaffold           fine-tuned model weights
        training_log.txt         per-epoch loss values (tab-separated)
        training_curves.png      plot of train loss (+ val loss if val data provided)
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for HPC
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy import ndimage


# ── mask loading ──────────────────────────────────────────────────────────────
def load_mask_as_instance(mask_path, min_size=500):
    """
    Load *-mask.png (strand_clean binary, pixel>0=strand) and convert to
    Cellpose instance mask via connected-component labelling.

    Returns (instance_mask uint16, n_instances int).
    """
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]          # all channels equal

    binary = (arr > 0).astype(np.uint8)
    labeled, n_raw = ndimage.label(binary)

    instance = np.zeros_like(labeled, dtype=np.uint16)
    new_id = 1
    for i in range(1, n_raw + 1):
        comp = labeled == i
        if comp.sum() >= min_size:
            instance[comp] = new_id
            new_id += 1

    return instance, new_id - 1


# ── pair collection ───────────────────────────────────────────────────────────
def collect_pairs(data_dir, min_size=500):
    """
    Scan data_dir for (tif, mask) pairs.
    Returns (images, masks, stems) lists.
    """
    data_dir = Path(data_dir)
    images, masks, stems = [], [], []

    mask_files = sorted(
        p for p in data_dir.glob('*-mask.png')
        if 'visible' not in p.name and 'target' not in p.name
    )

    for mask_path in mask_files:
        stem = mask_path.stem.replace('-mask', '')
        tif_path = next(
            (data_dir / f'{stem}{ext}'
             for ext in ('.tif', '.tiff', '.TIF', '.TIFF')
             if (data_dir / f'{stem}{ext}').exists()),
            None
        )
        if tif_path is None:
            print(f'  [SKIP] no TIF for {stem}')
            continue

        img = np.array(Image.open(tif_path).convert('RGB'))
        instance_mask, n_inst = load_mask_as_instance(mask_path, min_size)

        images.append(img)
        masks.append(instance_mask)
        stems.append(stem)
        print(f'  [OK] {stem}  instances={n_inst}')

    return images, masks, stems


# ── training curve + log ──────────────────────────────────────────────────────
def save_training_outputs(model_dir, train_losses, test_losses,
                          n_epochs, learning_rate, weight_decay,
                          min_size, use_gpu, data_dir, stems, val_stems=None):
    """
    Save training log (text) and training curves (PNG).

    train_losses : list of float, one value per epoch
    test_losses  : list of float or None/empty
    """
    model_dir = Path(model_dir)

    # ── 1. text log ───────────────────────────────────────────────────────────
    log_path = model_dir / 'training_log.txt'
    has_val  = test_losses is not None and len(test_losses) > 0

    lines = [
        f'model:          cpsam_scaffold',
        f'data_dir:       {data_dir}',
        f'n_train_images: {len(stems)}',
        f'n_val_images:   {len(val_stems) if val_stems else 0}',
        f'n_epochs:       {n_epochs}',
        f'learning_rate:  {learning_rate}',
        f'weight_decay:   {weight_decay}',
        f'min_size:       {min_size}',
        f'use_gpu:        {use_gpu}',
        f'train_stems:    {stems}',
        f'val_stems:      {val_stems if val_stems else []}',
        f'',
        f'{"epoch":<8}{"train_loss":<16}{"val_loss":<16}',
        f'{"─"*40}',
    ]

    for epoch in range(len(train_losses)):
        tl = f'{train_losses[epoch]:.6f}'
        vl = f'{test_losses[epoch]:.6f}' if has_val and epoch < len(test_losses) else 'n/a'
        lines.append(f'{epoch+1:<8}{tl:<16}{vl:<16}')

    if train_losses:
        lines += [
            f'{"─"*40}',
            f'final train loss : {train_losses[-1]:.6f}',
            f'min   train loss : {min(train_losses):.6f}  (epoch {train_losses.index(min(train_losses))+1})',
        ]
        if has_val:
            lines += [
                f'final val   loss : {test_losses[-1]:.6f}',
                f'min   val   loss : {min(test_losses):.6f}  (epoch {test_losses.index(min(test_losses))+1})',
            ]

    log_path.write_text('\n'.join(lines))
    print(f'✓ Training log    → {log_path}')

    # ── 2. training curve PNG ─────────────────────────────────────────────────
    if not train_losses:
        print('  [WARNING] No loss values to plot.')
        return

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, color='steelblue', linewidth=1.5,
            label='Train loss')
    if has_val and len(test_losses) == len(train_losses):
        ax.plot(epochs, test_losses, color='tomato', linewidth=1.5,
                linestyle='--', label='Val loss')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Cellpose-SAM fine-tuning  |  {len(stems)} images  |  lr={learning_rate}',
                 fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # annotate minimum train loss
    min_tl_idx = int(np.argmin(train_losses))
    ax.annotate(
        f'min {train_losses[min_tl_idx]:.4f}\n(epoch {min_tl_idx+1})',
        xy=(min_tl_idx + 1, train_losses[min_tl_idx]),
        xytext=(min_tl_idx + 1 + max(1, len(epochs)*0.05),
                train_losses[min_tl_idx] + (max(train_losses)-min(train_losses))*0.08),
        fontsize=8, color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0),
    )

    fig.tight_layout()
    curve_path = model_dir / 'training_curves.png'
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f'✓ Training curves → {curve_path}')


# ── training ──────────────────────────────────────────────────────────────────
def train(data_dir, model_dir, n_epochs, learning_rate, weight_decay,
          min_size, use_gpu, val_frac=0.15):

    from cellpose import models, train as cp_train, io as cp_io

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # activate cellpose logger so epoch lines appear in SLURM .out file
    cp_io.logger_setup()

    print(f'\n[1/2] Loading training data from: {data_dir}')
    images, masks, stems = collect_pairs(data_dir, min_size)

    if not images:
        data_dir_p = Path(data_dir)
        tifs   = list(data_dir_p.glob('*.tif')) + list(data_dir_p.glob('*.tiff'))
        masks  = [p for p in data_dir_p.glob('*-mask.png')
                  if 'visible' not in p.name and 'target' not in p.name]
        print('[ERROR] No (tif, mask) pairs found.')
        print(f'  data_dir : {data_dir_p.resolve()}')
        print(f'  .tif files found    : {len(tifs)}  {[p.name for p in tifs[:5]]}')
        print(f'  *-mask.png found    : {len(masks)}  {[p.name for p in masks[:5]]}')
        print('  → data_dir must contain BOTH *.tif and *-mask.png in the same folder.')
        sys.exit(1)

    print(f'\n      {len(images)} images loaded from {len(set(s.split("_")[0] for s in stems))} samples.')

    # ── sample-level validation split ────────────────────────────────────────
    # Group stems by Sample_ID (the part before the first underscore).
    # All replicates of a sample go to the same split — prevents leakage.
    if val_frac > 0.0 and len(images) >= 6:
        import random
        sample_ids = sorted(set(s.split('_')[0] for s in stems))
        n_val_samples = max(1, round(len(sample_ids) * val_frac))

        rng = random.Random(42)
        val_sample_ids = set(rng.sample(sample_ids, n_val_samples))
        train_sample_ids = set(sample_ids) - val_sample_ids

        train_idx = [i for i, s in enumerate(stems) if s.split('_')[0] in train_sample_ids]
        val_idx   = [i for i, s in enumerate(stems) if s.split('_')[0] in val_sample_ids]

        train_images = [images[i] for i in train_idx]
        train_masks  = [masks[i]  for i in train_idx]
        train_stems  = [stems[i]  for i in train_idx]
        val_images   = [images[i] for i in val_idx]
        val_masks    = [masks[i]  for i in val_idx]

        print(f'      Val split: {n_val_samples} samples ({len(val_idx)} images) held out')
        print(f'      Val sample IDs: {sorted(val_sample_ids)}')
        print(f'      Train: {len(train_images)} images  |  Val: {len(val_images)} images')
    else:
        if val_frac > 0.0:
            print(f'      [NOTE] Too few images for val split — training on all data.')
        else:
            print(f'      val_frac=0.0 — no validation split.')
        train_images, train_masks, train_stems = images, masks, stems
        val_images,   val_masks                = None, None

    if len(train_images) < 5:
        print(f'  [WARNING] Only {len(train_images)} training images — '
              f'results may overfit. Aim for 15+ for generalisation.')

    print(f'\n[2/2] Fine-tuning cpsam...')
    print(f'      n_epochs={n_epochs}  lr={learning_rate}  wd={weight_decay}')

    model_save = str(model_dir / 'cpsam_scaffold')
    model      = models.CellposeModel(gpu=use_gpu, model_type='cpsam')

    result = cp_train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=val_images,
        test_labels=val_masks,
        normalize=True,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        min_train_masks=1,
        model_name=model_save,
    )

    if isinstance(result, tuple) and len(result) == 3:
        _, train_losses, test_losses = result
        train_losses = list(train_losses) if train_losses is not None else []
        test_losses  = list(test_losses)  if test_losses  is not None else []
        # filter out placeholder 0.0 val losses (Cellpose returns 0.0 when no val data)
        if all(v == 0.0 for v in test_losses):
            test_losses = []
    else:
        print('  [NOTE] This cellpose version does not return loss arrays. '
              'Log will contain config only; no curve plotted.')
        train_losses, test_losses = [], []

    print(f'\n✓ Model saved → {model_save}')

    save_training_outputs(
        model_dir=model_dir,
        train_losses=train_losses,
        test_losses=test_losses,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        min_size=min_size,
        use_gpu=use_gpu,
        data_dir=data_dir,
        stems=train_stems,
        val_stems=[stems[i] for i in val_idx] if val_images else [],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune Cellpose-SAM on scaffold images.'
    )
    parser.add_argument('--data_dir', required=True,
        help='Folder with *.tif + *-mask.png training pairs')
    parser.add_argument('--model_dir', required=True,
        help='Where to save the fine-tuned model')
    parser.add_argument('--n_epochs', type=int, default=100,
        help='Training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
        help='Learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
        help='Weight decay (default: 0.1)')
    parser.add_argument('--min_size', type=int, default=500,
        help='Min instance px to keep (default: 500)')
    parser.add_argument('--val_frac', type=float, default=0.15,
        help='Fraction of samples held out for validation (default: 0.15). '
             'Split is at sample level — all replicates of a sample stay together. '
             'Set to 0.0 to disable validation split.')
    parser.add_argument('--no_gpu', action='store_true',
        help='Disable GPU')
    args = parser.parse_args()

    train(
        args.data_dir, args.model_dir,
        args.n_epochs, args.learning_rate, args.weight_decay,
        args.min_size, use_gpu=not args.no_gpu,
        val_frac=args.val_frac,
    )