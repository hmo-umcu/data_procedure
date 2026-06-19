#!/bin/bash
#SBATCH --job-name=env_setup
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/env_setup_%j.out
#SBATCH --error=/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/logs/env_setup_%j.err

# ── modules ───────────────────────────────────────────────────────────────────
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

# ── create venv if it doesn't exist ───────────────────────────────────────────
VENV_DIR="/home/hmo/venvs/bioprint"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "venv created at $VENV_DIR"
else
    echo "venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── pip upgrade ───────────────────────────────────────────────────────────────
pip install --upgrade pip

# ── PyTorch (pinned, cu124) ───────────────────────────────────────────────────
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── SAM2 from source ──────────────────────────────────────────────────────────
SAM2_DIR="/home/hmo/sam2"
if [ ! -d "$SAM2_DIR" ]; then
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
fi
cd "$SAM2_DIR"
pip install -e .

# download checkpoints
cd "$SAM2_DIR/checkpoints"
bash download_ckpts.sh

# ── requirements.txt ──────────────────────────────────────────────────────────
REQUIREMENTS="/home/hmo/BioRT/Rheology-informed-optimization/data_procedure/requirements.txt"
pip install -r "$REQUIREMENTS"

echo "───────────────────────────────"
echo "Setup complete. Installed packages:"
pip list
echo "───────────────────────────────"