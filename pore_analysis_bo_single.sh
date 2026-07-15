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

# Point this at a cross-validation output dir containing fold_0, fold_1, ...,
# each with a predictions/ subfolder (the layout unetplusplus_cross_validate.py
# writes) — e.g. $SCRIPT_DIR/data/dev_images/cv_unetplusplus
CV_PARENT_DIR="$SCRIPT_DIR/data/dev_images/pluronic_bo_recommendation"


echo "Job started    : $(date)"
echo "Node           : $SLURMD_NODENAME"
echo "CV parent dir  : $CV_PARENT_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/pore_analysis.py" \
    --cv_parent_dir       "$CV_PARENT_DIR" \
    --w                   0.30 \
    --min_pore_px         1000 \
    --max_pore_px         150000 \
    --max_aspect_ratio    3.0 \
    --match_overlap_frac  0.3 \
    --close_kernel        21 \
    --n_pore_cols         4

# ── alternative: single-folder mode (one flat predictions dir, no folds) ──
# python "$SCRIPT_DIR/pore_analysis.py" \
#     --data_dir            "$SCRIPT_DIR/data/dev_images/dev_annot_test_unetplusplus_pred" \
#     --output_dir          "$SCRIPT_DIR/data/dev_images/pore_analysis_unetplusplus" \
#     --w                   0.25 \
#     --min_pore_px         10000 \
#     --max_pore_px         150000 \
#     --max_aspect_ratio    3.0 \
#     --match_overlap_frac  0.3 \
#     --close_kernel        21

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"