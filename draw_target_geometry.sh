#!/bin/bash
#SBATCH --job-name=draw_target_geom
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/draw_target_geom_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/draw_target_geom_%j.err
# CPU-only job — no CUDA needed, use rome partition to save GPU budget

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

# ── adjust these paths ────────────────────────────────────────────────────────
# IMG_DIR="data/dev_images/dev_annot_trans_260529_renamed"
# MASK_DIR="data/dev_images/dev_annot_trans_260529_renamed"
# OUTPUT_DIR="data/dev_images/dev_annot_trans_260529_renamed"

IMG_DIR="data/dev_images/cv_unet/fold_0/predictions"
MASK_DIR="data/dev_images/cv_unet/fold_0/predictions"
OUTPUT_DIR="data/dev_images/cv_unet/fold_0/predictions"



echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "Img dir     : $IMG_DIR"
echo "Mask dir    : $MASK_DIR"
echo "Output dir  : $OUTPUT_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/draw_target_geometry.py" \
    "$IMG_DIR" \
    --mask_dir         "$MASK_DIR" \
    --output_dir       "$OUTPUT_DIR" \
    --strand_width_mm  0.41 \
    --strand_gap_mm    2.5

# add --no_drift to disable drift correction
# add --iou_threshold <f> to flag high-IoU results with a star in the printed summary

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"