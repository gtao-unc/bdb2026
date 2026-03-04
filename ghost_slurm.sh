#!/bin/bash
#SBATCH --job-name=ghost_helio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ghost_%j.out
#SBATCH --error=logs/ghost_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --partition a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# ---------------------------------------------------------------------------
# Heliocentricity Transformer – SLURM job script
# Companion to ghost.py
# ---------------------------------------------------------------------------

set -euo pipefail

# Resolve the directory this script lives in so the job is portable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $(hostname)"
echo "Working dir   : $PWD"
echo "Start time    : $(date)"
echo "=============================================="

# Environment
module load cuda/12.1
PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/ghost/bin/python"

# Confirm GPU visibility
echo ""
echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "(no nvidia-smi)"
echo ""

# Create output directories if they don't already exist
mkdir -p logs
mkdir -p dataset/processed
mkdir -p dataset/pretrained

# --- Run the pipeline ---
echo "--- Starting ghost.py ---"
$PYTHON_ENV ghost.py

echo ""
echo "=============================================="
echo "Job finished : $(date)"
echo "=============================================="
