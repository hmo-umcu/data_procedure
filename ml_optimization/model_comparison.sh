#!/bin/bash
#SBATCH --job-name=sf_comparison
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/sf_comparison_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/sf_comparison_%j.err
# Run AFTER model_ridge.sh, model_gpr.sh, model_ngboost.sh have completed

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"
RESULTS_DIR="$SCRIPT_DIR/results"
OUTDIR="$RESULTS_DIR/comparison"

echo "Job started  : $(date)"
echo "Results dir  : $RESULTS_DIR"
echo "Output dir   : $OUTDIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/model_comparison.py" \
    --results_dir "$RESULTS_DIR" \
    --outdir      "$OUTDIR" \
    --models      Ridge GPR NGBoost

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
