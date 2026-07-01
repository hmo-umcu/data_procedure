#!/bin/bash
#SBATCH --job-name=sf_bo_recommend
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/sf_bo_recommend_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/sf_bo_recommend_%j.err
# Run AFTER you've validated the surrogate via model_gpr.py (check R² there
# first — this script trusts the GPR fully and uses it to recommend a real
# print to run next).

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/ml_optimization"
DATA="$SCRIPT_DIR/sample_sf_summary_w30.csv"
OUTDIR="$SCRIPT_DIR/results/BO"

echo "Job started : $(date)"
echo "Data        : $DATA"
echo "Output dir  : $OUTDIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/bo_recommend.py" \
    --data    "$DATA" \
    --outdir  "$OUTDIR" \
    --seed    42 \
    --nu      1.5 \
    --xi      0.01 \
    --n_restarts 50 \
    --top_k 3    \
    --zoffset_bounds 0.05 0.8
# Search bounds default to the observed data range. To search outside it:
#   --pressure_bounds 70 130 --speed_bounds 4 16 --zoffset_bounds 0.05 0.8

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"