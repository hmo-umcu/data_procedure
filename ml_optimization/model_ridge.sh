#!/bin/bash
#SBATCH --job-name=sf_ridge
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/sf_ridge_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/sf_ridge_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/ml_optimization"
DATA="$SCRIPT_DIR/sample_sf_summary_w30.csv"
OUTDIR="$SCRIPT_DIR/results/Ridge"

echo "Job started : $(date)"
echo "Data        : $DATA"
echo "Output dir  : $OUTDIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/model_ridge.py" \
    --data    "$DATA" \
    --outdir  "$OUTDIR" \
    --n_folds 4 \
    --seed    42 \
    --alphas  0.01 0.1 1.0 10.0 100.0 1000.0

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
