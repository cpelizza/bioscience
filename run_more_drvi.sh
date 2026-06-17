#!/bin/bash
#SBATCH --job-name=drvi_marimo
#SBATCH --output=/mnt/netapp1/Store_RES/home/res/cnag71/resh000982/projects/bioscience/logs/drvi_%A_%a.out
#SBATCH --error=/mnt/netapp1/Store_RES/home/res/cnag71/resh000982/projects/bioscience/logs/drvi_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=245G
#SBATCH --time=3-00:00:00
#SBATCH --partition=medium
#SBATCH --gres=gpu:a100:2
#SBATCH --array=0-4

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export WANDB_API_KEY=$(grep "password" ~/.netrc | awk '{print $2}')

PROJECT_DIR="/mnt/netapp1/Store_RES/home/res/cnag71/resh000982/projects/bioscience"
cd "$PROJECT_DIR"
mkdir -p html results

NOTEBOOK="$PROJECT_DIR/notebooks/drvi_train.py"

LATENTS=(10 10 20 20 40)
SEEDS=(456 789 456 789 456)

LATENT=${LATENTS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Running: LATENT=$LATENT  SEED=$SEED  (array task $SLURM_ARRAY_TASK_ID)"

pixi run marimo export html "$NOTEBOOK" \
    -o "${PROJECT_DIR}/html/drvi_L${LATENT}_S${SEED}.html" \
    -- --n_latent "$LATENT" --seed "$SEED"
