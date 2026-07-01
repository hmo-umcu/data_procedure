#!/bin/bash
#SBATCH --job-name=sf_correlation
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/sf_correlation_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/sf_correlation_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/ml_optimization"
DATA="$SCRIPT_DIR/sample_sf_summary_w30.csv"
OUTDIR="$SCRIPT_DIR/figures/correlation"

echo "Job started : $(date)"
echo "Data        : $DATA"
echo "Output dir  : $OUTDIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/multivariate_correlation.py" \
    --data   "$DATA" \
    --outdir "$OUTDIR"

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
