#!/bin/bash
#SBATCH --job-name=bo_gen_nc
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/bo_gen_nc_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/bo_gen_nc_%j.err
# CPU only, very fast — just text parsing + file write

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/ml_optimization"

# ── EDIT THESE ────────────────────────────────────────────────────────────────
ITERATION=1                             # which row from bo_recommendation_log.csv to use
BO_LOG="$SCRIPT_DIR/results/BO/bo_recommendation_log.csv"
TEMPLATE_NC="$SCRIPT_DIR/data_collection_48well_8cols_s0-s49.nc"
OUTPUT_NC="$SCRIPT_DIR/bo_iter${ITERATION}_col1.nc"
# ─────────────────────────────────────────────────────────────────────────────

echo "Job started  : $(date)"
echo "BO log       : $BO_LOG"
echo "Iteration    : $ITERATION"
echo "Template NC  : $TEMPLATE_NC"
echo "Output NC    : $OUTPUT_NC"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/generate_bo_print_nc.py" \
    --bo_log       "$BO_LOG" \
    --iteration    "$ITERATION" \
    --template_nc  "$TEMPLATE_NC" \
    --output_nc    "$OUTPUT_NC"

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
