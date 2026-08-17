#!/bin/bash
#SBATCH --job-name=unetpp_deploy
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_deploy_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/unetpp_deploy_%j.err
 
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source /home/hmo/venvs/bioprint/bin/activate
 
SCRIPT_DIR="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure"
 
# ── adjust these two ──────────────────────────────────────────────────────────
# One model for the whole tree. The cell-free checkpoint is used here because
# it performs well on both cell-free and cell-laden. Swap the path to switch.
MODEL_PATH="$SCRIPT_DIR/models/unetplusplus/gelma_finetune_cell-free-trained/best_model.pth"
DATA_ROOT="$SCRIPT_DIR/data/dev_images/gelma_deployment"
 
echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Model       : $MODEL_PATH"
echo "Data root   : $DATA_ROOT"
echo "────────────────────────────────────────────────────"
 
# Dry run: lists every folder found and its image count, writes nothing.
python "$SCRIPT_DIR/unetplusplus_test_gelma.py" \
    --model_path "$MODEL_PATH" \
    --data_dir   "$DATA_ROOT" \
    --recursive \
    --dry_run
 
echo "────────────────────────────────────────────────────"
 
# Real run. Predictions land next to the input images, folder by folder.
# --skip_existing makes the job resumable if it hits the wall clock.
python "$SCRIPT_DIR/unetplusplus_test_gelma.py" \
    --model_path "$MODEL_PATH" \
    --data_dir   "$DATA_ROOT" \
    --recursive \
    --img_size   512 \
    --threshold  0.5 \
    --skip_existing
 
echo "────────────────────────────────────────────────────"
echo "Job finished: $(date)"
 
# ── OPTIONAL second stage: target geometry + IoU ──────────────────────────────
# unetplusplus_test.py writes {stem}-pred-mask.png, so draw_target_geometry.py
# needs --pred_masks to read those instead of the annotation *-mask.png.
# Uncomment once the segmentation looks right.
#
# for d in "$DATA_ROOT"/*/*/ ; do
#     echo "Target geometry: $d"
#     python "$SCRIPT_DIR/draw_target_geometry.py" "$d" \
#         --output_dir "$d" \
#         --pred_masks \
#         --strand_width_mm 0.41 \
#         --strand_gap_mm   2.5
# done