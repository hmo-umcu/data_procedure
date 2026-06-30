#!/bin/bash
#SBATCH --job-name=unetpp_test
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_test_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_test_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

# ── adjust these paths ────────────────────────────────────────────────────────
MODEL_PATH="$SCRIPT_DIR/models/unetplusplus/run_01/best_model.pth"
DATA_DIR="$SCRIPT_DIR/data/dev_images/dev_annot_test"
OUTPUT_DIR="$SCRIPT_DIR/data/dev_images/dev_annot_test_unetpp_pred"

mkdir -p "$OUTPUT_DIR"

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Model       : $MODEL_PATH"
echo "Data dir    : $DATA_DIR"
echo "Output dir  : $OUTPUT_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/unetplusplus_test.py" \
    --model_path  "$MODEL_PATH" \
    --data_dir    "$DATA_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --img_size    512 \
    --threshold   0.5

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
