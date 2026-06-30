#!/bin/bash
#SBATCH --job-name=fpn_train
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/fpn_train_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/fpn_train_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source /home/hmo/venvs/bioprint/bin/activate

SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"
DATA_DIR="$SCRIPT_DIR/data/dev_images/dev_annot_train"
MODEL_DIR="$SCRIPT_DIR/models/fpn/run_01"

mkdir -p "$MODEL_DIR"

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Architecture: fpn"
echo "Data dir    : $DATA_DIR"
echo "Model dir   : $MODEL_DIR"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/unetplusplus_train.py" \
    --data_dir       "$DATA_DIR" \
    --model_dir      "$MODEL_DIR" \
    --architecture   fpn \
    --encoder        resnet34 \
    --n_epochs       200 \
    --batch_size     4 \
    --learning_rate  1e-5 \
    --weight_decay   1e-5 \
    --val_frac       0.15 \
    --patience       20 \
    --img_size       512

echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
