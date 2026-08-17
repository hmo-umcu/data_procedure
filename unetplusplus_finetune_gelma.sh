#!/bin/bash
#SBATCH --job-name=unetpp_gelma
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_gelma_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_gelma_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

## cell-free annoated images path:data/dev_images/gelma_annot_cell-free
## cell-laden annoated images path:data/dev_images/gelma_annot_cell-laden
## trained unetpp path: data/dev_images/cv_unetpp/fold_0/model/pluronic_unetpp_fold_0.pth

# ── adjust these paths ────────────────────────────────────────────────────────
# GelMA annotated images (R-GEN 200)
DATA_DIR="$SCRIPT_DIR/data/dev_images/gelma_annot_cell-laden"

# pretrained Pluronic model to fine-tune from
PRETRAINED="$SCRIPT_DIR/data/dev_images/cv_unetpp/fold_0/model/pluronic_unetpp_fold_0.pth"

# where to save fine-tuned model
MODEL_DIR="$SCRIPT_DIR/models/unetplusplus/gelma_finetune_cell-laden-trained"

mkdir -p "$MODEL_DIR"

echo "Job started    : $(date)"
echo "Node           : $SLURMD_NODENAME"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Data dir       : $DATA_DIR"
echo "Pretrained from: $PRETRAINED"
echo "Model dir      : $MODEL_DIR"
echo "────────────────────────────────────────────────────"

# Option A: fine-tune from Pluronic checkpoint
# Use lower LR (1e-5 instead of 1e-4) to preserve learned features
python "$SCRIPT_DIR/unetplusplus_train.py" \
    --data_dir          "$DATA_DIR" \
    --model_dir         "$MODEL_DIR" \
    --architecture      unetplusplus \
    --encoder           resnet34 \
    --pretrained_model  "$PRETRAINED" \
    --n_epochs          100 \
    --batch_size        4 \
    --learning_rate     1e-5 \
    --weight_decay      1e-4 \
    --val_frac          0.15 \
    --patience          20 \
    --img_size          512

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
