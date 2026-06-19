#!/bin/bash
#SBATCH --job-name=cpsam_train
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/cpsam_train_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/cpsam_train_%j.err
 
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source /home/hmo/venvs/bioprint/bin/activate
 
SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"

# ── adjust these paths ────────────────────────────────────────────────────────
DATA_DIR="data/dev_images/dev_annot_train"
MODEL_DIR="models/cellpose/run_01"
 



echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Data dir    : $DATA_DIR"
echo "Model dir   : $MODEL_DIR"
echo "────────────────────────────────────────────────────"
 
python "$SCRIPT_DIR/cellpose_train.py" \
    --data_dir       "$DATA_DIR" \
    --model_dir      "$MODEL_DIR" \
    --n_epochs       200 \
    --learning_rate  1e-5 \
    --weight_decay   0.1 \
    --min_size       500
 
echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
 