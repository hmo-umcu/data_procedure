#!/bin/bash
#SBATCH --job-name=pore_analysis
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/pore_analysis_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/pore_analysis_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

# Point this at any folder containing {stem}.tif, {stem}-target-overlay.png,
# and {stem}-pred-mask.png — e.g. one fold's predictions/ dir from a CV run,
# or a flat test_unetpp_pred dir from unetplusplus_test.py.
DATA_DIR="$SCRIPT_DIR/data/dev_images/cv_unetpp/fold_0/predictions"
OUTPUT_DIR="$SCRIPT_DIR/data/dev_images/cv_unetpp/fold_0/pore_analysis"

mkdir -p "$OUTPUT_DIR"

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "Data dir    : $DATA_DIR"
echo "Output dir  : $OUTPUT_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/pore_analysis.py" \
    --data_dir            "$DATA_DIR" \
    --output_dir          "$OUTPUT_DIR" \
    --w                   0.25 \
    --min_pore_px         1000 \
    --max_pore_px         150000 \
    --max_aspect_ratio    3.0 \
    --match_overlap_frac  0.3 \
    --close_kernel        21

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"