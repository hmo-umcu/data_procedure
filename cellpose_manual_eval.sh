#!/bin/bash
#SBATCH --job-name=cpsam_eval
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/cpsam_eval_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/cpsam_eval_%j.err

# CPU-only job — no CUDA needed, use rome partition to save GPU budget
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

# ── adjust these paths ────────────────────────────────────────────────────────
PRED_DIR="data/dev_images/prediction_results/test_predictions"

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "Pred dir    : $PRED_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/cellpose_manual_eval.py" \
    --pred_dir         "$PRED_DIR" \
    --strand_width_mm  0.41 \
    --strand_gap_mm    2.5

# add --no_drift to disable drift correction

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"