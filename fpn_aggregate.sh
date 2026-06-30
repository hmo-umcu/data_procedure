#!/bin/bash
#SBATCH --job-name=fpn_agg
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/fpn_agg_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/fpn_agg_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"
PRED_DIR="$SCRIPT_DIR/data/dev_images/dev_annot_test_fpn_pred"

echo "Job started : $(date)"
echo "Architecture: fpn"
echo "Pred dir    : $PRED_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/unetplusplus_aggregate.py" \
    --pred_dir "$PRED_DIR"

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
