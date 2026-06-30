#!/bin/bash
#SBATCH --job-name=compare_cv
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/compare_cv_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/compare_cv_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

echo "Job started : $(date)"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/compare_cv_results.py" \
    --base_dir  "$SCRIPT_DIR/data/dev_images" \
    --output_dir "$SCRIPT_DIR/data/dev_images"

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"