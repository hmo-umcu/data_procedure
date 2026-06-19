"""
cross_validate.py
-----------------
Run k-fold cross-validation for Cellpose-SAM fine-tuning on scaffold images.

Splitting strategy
------------------
Splits at SAMPLE level (by Sample_ID), not at image level.
All 6 replicate images of a sample stay together in train or test.
This prevents data leakage between replicates of the same parameter set.

With 32 samples and k=4 folds:
    fold 0 : test=8 samples (48 images),  train=24 samples (144 images)
    fold 1 : test=8 samples (48 images),  train=24 samples (144 images)
    fold 2 : test=8 samples (48 images),  train=24 samples (144 images)
    fold 3 : test=8 samples (48 images),  train=24 samples (144 images)

Input
-----
Single flat folder containing all annotated images:
    {sid}_{row}.tif
    {sid}_{row}-mask.png
    {sid}_{row}-mask-visible.png   (optional)
    {sid}_{row}.json               (optional)
    {sid}_{row}-target-overlay.png (optional)

Usage
-----
    python cross_validate.py
        --data_dir   /scratch-shared/hmo/scaffold_images
        --output_dir /scratch-shared/hmo/cv_results
        [--k          4]
        [--n_epochs   100]
        [--learning_rate 1e-5]
        [--weight_decay  0.1]
        [--min_size   500]
        [--strand_width_mm 0.41]
        [--strand_gap_mm   2.5]
        [--seed       42]
        [--no_drift]
        [--no_gpu]

Output
------
    <output_dir>/
        fold_0/
            train/           copies of train images+masks
            test/            copies of test images+masks
            model/
                cpsam_scaffold           trained model weights
                training_log.txt         per-epoch loss table
                training_curves.png      loss curve for this fold
            predictions/
                {stem}-pred-mask.png
                {stem}-pred-visible.png
                {stem}-pred-vs-annot.png
                test_info.csv            with all IoU metrics filled
        fold_1/  ...
        fold_2/  ...
        fold_3/  ...
        cv_summary.csv              per-fold aggregated IoU metrics
        cv_training_curves.png      all fold loss curves on one plot
"""

import argparse
import csv
import json
import re
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# ── discover sample IDs in data folder ───────────────────────────────────────
def discover_samples(data_dir):
    """
    Parse all *.tif filenames to find unique Sample_IDs and their replicate rows.
    Filename pattern: {Sample_ID}_{row}.tif

    Returns dict: {sample_id (int): [row (int), ...]}
    """
    data_dir = Path(data_dir)
    samples  = defaultdict(list)

    for tif in data_dir.glob('*.tif'):
        if any(x in tif.name for x in ['visible', 'overlay', 'target']):
            continue
        m = re.match(r'^(\d+)_(\d+)$', tif.stem)
        if m:
            sid = int(m.group(1))
            row = int(m.group(2))
            samples[sid].append(row)

    return dict(samples)


# ── k-fold split at sample level ─────────────────────────────────────────────
def kfold_sample_split(sample_ids, k=4, seed=42):
    """
    Split list of sample_ids into k folds.
    Returns list of (train_ids, test_ids) tuples.
    """
    rng    = np.random.default_rng(seed)
    sids   = sorted(sample_ids)
    sids   = list(rng.permutation(sids))
    folds  = [sids[i::k] for i in range(k)]

    splits = []
    for i in range(k):
        test_ids  = folds[i]
        train_ids = [s for j, fold in enumerate(folds) if j != i for s in fold]
        splits.append((sorted(train_ids), sorted(test_ids)))

    return splits


# ── copy files for a split ───────────────────────────────────────────────────
SUFFIXES = ['.tif', '.tiff', '-mask.png', '-mask-visible.png',
            '.json', '-target-overlay.png']

def copy_split(data_dir, dest_dir, sample_ids, rows_map):
    """
    Copy all files for given sample_ids to dest_dir.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    for sid in sample_ids:
        rows = rows_map.get(sid, [])
        for row in rows:
            stem = f'{sid}_{row}'
            for suf in SUFFIXES:
                src = data_dir / f'{stem}{suf}'
                if src.exists():
                    shutil.copy2(src, dest_dir / src.name)


# ── invoke train / test / evaluate as functions ───────────────────────────────
def run_train(train_dir, model_dir, n_epochs, learning_rate,
              weight_decay, min_size, use_gpu):
    """Call train.py logic directly."""
    import sys
    sys.argv = ['train.py']   # prevent argparse conflicts
    from cellpose_train import train as _train
    _train(
        data_dir=str(train_dir),
        model_dir=str(model_dir),
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        min_size=min_size,
        use_gpu=use_gpu,
    )


def run_test(model_path, test_dir, pred_dir, use_gpu):
    """Call test.py logic directly."""
    from cellpose_test import run_test as _test
    _test(
        model_path=str(model_path),
        data_dir=str(test_dir),
        output_dir=str(pred_dir),
        use_gpu=use_gpu,
    )


def run_evaluate(pred_dir, strand_width_mm, strand_gap_mm, apply_drift):
    """Call evaluate.py logic directly."""
    from cellpose_evaluate import evaluate as _evaluate
    _evaluate(
        pred_dir=str(pred_dir),
        strand_width_mm=strand_width_mm,
        strand_gap_mm=strand_gap_mm,
        apply_drift=apply_drift,
    )


# ── read fold metrics from test_info.csv ─────────────────────────────────────
def read_fold_metrics(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)

    iou_pa, iou_pt = [], []
    for r in rows:
        if r.get('iou_pred_annot'):
            try:
                iou_pa.append(float(r['iou_pred_annot']))
            except ValueError:
                pass
        if r.get('iou_pred_target'):
            try:
                iou_pt.append(float(r['iou_pred_target']))
            except ValueError:
                pass

    return {
        'n_test_images':       len(rows),
        'mean_iou_pred_annot': np.mean(iou_pa)  if iou_pa  else '',
        'std_iou_pred_annot':  np.std(iou_pa)   if iou_pa  else '',
        'mean_iou_pred_target':np.mean(iou_pt)  if iou_pt  else '',
        'std_iou_pred_target': np.std(iou_pt)   if iou_pt  else '',
    }


# ── cross-fold training curve plot ───────────────────────────────────────────
def plot_cv_curves(output_dir, k):
    """
    Read training_log.txt from each fold's model/ directory and produce
    a single PNG with all fold loss curves overlaid.

    One subplot per metric (train loss, val loss if present).
    Each fold gets its own colour; mean ± std band drawn in grey.
    """
    COLOURS = plt.cm.tab10.colors

    fold_train, fold_val = {}, {}

    for fold_idx in range(k):
        log_path = output_dir / f'fold_{fold_idx}' / 'model' / 'training_log.txt'
        if not log_path.exists():
            print(f'  [WARNING] no training_log.txt for fold {fold_idx} — skipping')
            continue

        train_losses, val_losses = [], []
        in_table = False

        for line in log_path.read_text().splitlines():
            # detect start of epoch table
            if line.strip().startswith('epoch'):
                in_table = True
                continue
            if line.startswith('─'):
                continue
            if not in_table:
                continue
            # epoch data lines: "1       0.423100        n/a"
            parts = line.split()
            if not parts or not parts[0].isdigit():
                in_table = False
                continue
            try:
                tl = float(parts[1])
                train_losses.append(tl)
            except (IndexError, ValueError):
                continue
            try:
                vl = float(parts[2]) if parts[2] != 'n/a' else None
                if vl is not None:
                    val_losses.append(vl)
            except (IndexError, ValueError):
                pass

        if train_losses:
            fold_train[fold_idx] = train_losses
        if val_losses and len(val_losses) == len(train_losses):
            fold_val[fold_idx] = val_losses

    if not fold_train:
        print('  [WARNING] No training logs found — skipping cv_training_curves.png')
        return

    has_val = len(fold_val) > 0
    n_panels = 2 if has_val else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    def _plot_panel(ax, fold_data, title):
        # align to shortest fold (in case of early stop)
        min_len  = min(len(v) for v in fold_data.values())
        matrix   = np.array([v[:min_len] for v in fold_data.values()])
        epochs   = np.arange(1, min_len + 1)
        mean_    = matrix.mean(axis=0)
        std_     = matrix.std(axis=0)

        for i, (fold_idx, losses) in enumerate(fold_data.items()):
            ax.plot(epochs, losses[:min_len],
                    color=COLOURS[fold_idx % len(COLOURS)],
                    linewidth=1.2, alpha=0.75,
                    label=f'fold {fold_idx}')

        ax.fill_between(epochs, mean_ - std_, mean_ + std_,
                        color='grey', alpha=0.18, label='mean ± std')
        ax.plot(epochs, mean_,
                color='black', linewidth=2.0, linestyle='--', label='mean')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss',  fontsize=12)
        ax.set_title(title,    fontsize=11)
        ax.legend(fontsize=9,  loc='upper right')
        ax.grid(True, alpha=0.3)

    _plot_panel(axes[0][0], fold_train, 'Train loss — all folds')
    if has_val:
        _plot_panel(axes[0][1], fold_val, 'Val loss — all folds')

    fig.suptitle(f'Cross-validation training curves  ({k} folds)', fontsize=13)
    fig.tight_layout()

    out_path = output_dir / 'cv_training_curves.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'✓ CV training curves → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── discover samples ──────────────────────────────────────────────────────
    print(f'[0] Scanning {data_dir} ...')
    rows_map = discover_samples(data_dir)
    all_sids = sorted(rows_map.keys())
    print(f'    Found {len(all_sids)} unique Sample_IDs: {all_sids}')
    total_images = sum(len(v) for v in rows_map.values())
    print(f'    Total images: {total_images}')

    # save sample discovery log
    discovery = {str(sid): rows_map[sid] for sid in all_sids}
    (output_dir / 'samples_discovered.json').write_text(
        json.dumps(discovery, indent=2)
    )

    # ── k-fold splits ─────────────────────────────────────────────────────────
    splits = kfold_sample_split(all_sids, k=args.k, seed=args.seed)

    splits_log = {}
    for i, (train_ids, test_ids) in enumerate(splits):
        splits_log[f'fold_{i}'] = {
            'train_sample_ids': train_ids,
            'test_sample_ids':  test_ids,
            'n_train_samples':  len(train_ids),
            'n_test_samples':   len(test_ids),
        }
    (output_dir / 'cv_splits.json').write_text(json.dumps(splits_log, indent=2))
    print(f'\n    {args.k}-fold splits saved → {output_dir}/cv_splits.json\n')

    # ── run each fold ─────────────────────────────────────────────────────────
    fold_summaries = []

    for fold_idx, (train_ids, test_ids) in enumerate(splits):
        fold_dir  = output_dir / f'fold_{fold_idx}'
        train_dir = fold_dir / 'train'
        test_dir  = fold_dir / 'test'
        model_dir = fold_dir / 'model'
        pred_dir  = fold_dir / 'predictions'

        print(f'\n{"="*60}')
        print(f'  FOLD {fold_idx}   train={len(train_ids)} samples '
              f'({sum(len(rows_map[s]) for s in train_ids)} images)   '
              f'test={len(test_ids)} samples '
              f'({sum(len(rows_map[s]) for s in test_ids)} images)')
        print(f'  train samples: {train_ids}')
        print(f'  test  samples: {test_ids}')
        print(f'{"="*60}')

        # copy files
        print(f'\n  [1/3] Copying train/test files...')
        copy_split(data_dir, train_dir, train_ids, rows_map)
        copy_split(data_dir, test_dir,  test_ids,  rows_map)

        # train
        print(f'\n  [2/3] Training...')
        run_train(
            train_dir=train_dir,
            model_dir=model_dir,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            min_size=args.min_size,
            use_gpu=not args.no_gpu,
        )

        # test
        model_path = model_dir / 'cpsam_scaffold'
        print(f'\n  [3a/3] Predicting test set...')
        run_test(
            model_path=model_path,
            test_dir=test_dir,
            pred_dir=pred_dir,
            use_gpu=not args.no_gpu,
        )

        # evaluate
        print(f'\n  [3b/3] Evaluating...')
        run_evaluate(
            pred_dir=pred_dir,
            strand_width_mm=args.strand_width_mm,
            strand_gap_mm=args.strand_gap_mm,
            apply_drift=not args.no_drift,
        )

        # collect fold metrics
        csv_path = pred_dir / 'test_info.csv'
        metrics  = read_fold_metrics(csv_path)
        metrics['fold']             = fold_idx
        metrics['train_sample_ids'] = str(train_ids)
        metrics['test_sample_ids']  = str(test_ids)
        fold_summaries.append(metrics)

        print(f'\n  Fold {fold_idx} done.')
        print(f'    mean IoU (pred|annot) : {metrics["mean_iou_pred_annot"]}')
        print(f'    mean IoU (pred|target): {metrics["mean_iou_pred_target"]}')

    # ── cross-validation summary ──────────────────────────────────────────────
    summary_path = output_dir / 'cv_summary.csv'
    fieldnames = [
        'fold', 'n_test_images',
        'mean_iou_pred_annot', 'std_iou_pred_annot',
        'mean_iou_pred_target', 'std_iou_pred_target',
        'train_sample_ids', 'test_sample_ids',
    ]
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';',
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(fold_summaries)

    # overall mean across folds
    all_iou_pa = [m['mean_iou_pred_annot']  for m in fold_summaries
                  if isinstance(m['mean_iou_pred_annot'], float)]
    all_iou_pt = [m['mean_iou_pred_target'] for m in fold_summaries
                  if isinstance(m['mean_iou_pred_target'], float)]

    # ── aggregated training curves across folds ───────────────────────────────
    print(f'\nPlotting cross-fold training curves...')
    plot_cv_curves(output_dir, args.k)

    print(f'\n{"="*60}')
    print(f'  CROSS-VALIDATION COMPLETE  ({args.k} folds)')
    if all_iou_pa:
        print(f'  mean IoU (pred|annot)  across folds: '
              f'{np.mean(all_iou_pa):.3f} ± {np.std(all_iou_pa):.3f}')
    if all_iou_pt:
        print(f'  mean IoU (pred|target) across folds: '
              f'{np.mean(all_iou_pt):.3f} ± {np.std(all_iou_pt):.3f}')
    print(f'  Summary          → {summary_path}')
    print(f'  CV training plot → {output_dir / "cv_training_curves.png"}')
    print(f'{"="*60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='k-fold cross-validation for Cellpose-SAM scaffold segmentation.'
    )
    parser.add_argument('--data_dir', required=True,
        help='Flat folder with all annotated *.tif + *-mask.png files')
    parser.add_argument('--output_dir', required=True,
        help='Where to save all fold outputs')
    parser.add_argument('--k', type=int, default=4,
        help='Number of folds (default: 4)')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--min_size', type=int, default=500)
    parser.add_argument('--strand_width_mm', type=float, default=0.41)
    parser.add_argument('--strand_gap_mm', type=float, default=2.5)
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for fold assignment (default: 42)')
    parser.add_argument('--no_drift', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()

    main(args)