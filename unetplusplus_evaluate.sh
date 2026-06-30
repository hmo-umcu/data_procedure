#!/bin/bash
#SBATCH --job-name=unetpp_eval
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_eval_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_eval_%j.err

# CPU-only — no CUDA needed, use rome to save GPU budget
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

# ── adjust this path ──────────────────────────────────────────────────────────
# pred_dir must contain:
#   *-pred-mask.png          from unetplusplus_test.py
#   *-mask.png               annotation masks (copied by test script)
#   *-target-overlay.png     from draw_target_geometry.py (copied by test script)
PRED_DIR="$SCRIPT_DIR/data/dev_images/dev_annot_test_unetpp_pred"

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "Pred dir    : $PRED_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/unetplusplus_evaluate.py" \
    --pred_dir "$PRED_DIR"

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"