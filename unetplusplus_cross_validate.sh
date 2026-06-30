#!/bin/bash
#SBATCH --job-name=unetpp_cv
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_cv_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_cv_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

# ── adjust these paths ────────────────────────────────────────────────────────
# data_dir: flat folder with ALL annotated images (all 32 samples)
DATA_DIR="$SCRIPT_DIR/data/dev_images/dev_annot_trans_260529_renamed"
OUTPUT_DIR="$SCRIPT_DIR/data/dev_images/cv_unetpp"

mkdir -p "$OUTPUT_DIR"

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Data dir    : $DATA_DIR"
echo "Output dir  : $OUTPUT_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/unetplusplus_cross_validate.py" \
    --data_dir       "$DATA_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --k              4 \
    --architecture   unetplusplus \
    --encoder        resnet34 \
    --n_epochs       200 \
    --batch_size     4 \
    --learning_rate  1e-5 \
    --weight_decay   1e-5 \
    --val_frac       0.0 \
    --patience       0 \
    --img_size       512 \
    --threshold      0.5 \
    --seed           42

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
