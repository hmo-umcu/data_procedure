#!/bin/bash
#SBATCH --job-name=cpsam_cv
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/cpsam_cv_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/cpsam_cv_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

# ── adjust these paths ────────────────────────────────────────────────────────
DATA_DIR="data/dev_images/dev_annot_trans_260529_renamed"
OUTPUT_DIR="$SCRIPT_DIR/data/dev_images/cv_results"

mkdir -p "$OUTPUT_DIR"

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Data dir    : $SCRIPT_DIR/$DATA_DIR"
echo "Output dir  : $OUTPUT_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/cellpose_cross-val.py" \
    --data_dir         "$SCRIPT_DIR/$DATA_DIR" \
    --output_dir       "$OUTPUT_DIR" \
    --k                4 \
    --n_epochs         5 \
    --learning_rate    1e-5 \
    --weight_decay     0.1 \
    --min_size         500 \
    --strand_width_mm  0.41 \
    --strand_gap_mm    2.5 \
    --seed             42

# add --no_drift  to disable drift correction
# add --no_gpu    to force CPU (not recommended)

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"