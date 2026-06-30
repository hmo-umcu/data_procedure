#!/bin/bash
#SBATCH --job-name=unet_eval
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unet_eval_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unet_eval_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"
PRED_DIR="$SCRIPT_DIR/data/dev_images/dev_annot_test_unet_pred"

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "Architecture: unet"
echo "Pred dir    : $PRED_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/unetplusplus_evaluate.py" \
    --pred_dir "$PRED_DIR"

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
