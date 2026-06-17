#!/bin/bash
# =============================================================================
# SLURM array job: export marimo notebook to HTML for each results folder
#
# Usage:
#   sbatch run_marimo_export.sh
#
# To limit concurrent jobs (e.g. max 4 at once):
#   sbatch --array=0-<N>%4 run_marimo_export.sh
# =============================================================================

# --------------- SLURM directives ---------------
#SBATCH -J drvi_marimo_export                       # Job name
#SBATCH --ntasks=1                                  # One task per array element
#SBATCH --cpus-per-task=64                          # CPUs per task (adjust as needed)
#SBATCH --mem=254G                                  # RAM per CPU (adjust as needed)
#SBATCH --time=24:00:00                             # Max walltime per job (HH:MM:SS)
#SBATCH --array=0-3                                 # <-- SET THIS: 0 to (N_FOLDERS - 1)
#SBATCH -o /dev/null
#SBATCH --error=/mnt/netapp1/Store_RES/home/res/cnag71/resh000982/projects/bioscience/drvi_%A_%a.err
# ------------------------------------------------

# ============================================================
# USER CONFIGURATION — edit these before submitting
# ============================================================

# Path to your marimo notebook
NOTEBOOK="/mnt/netapp1/Store_RES/home/res/cnag71/resh000982/projects/bioscience/notebooks/generate_analysis.py"

# Root of your project (where pixi.toml lives)
PROJECT_DIR="/mnt/netapp1/Store_RES/home/res/cnag71/resh000982/projects/bioscience"

# Results base folder (same as RESULTS_FOLDER in the notebook)
RESULTS_FOLDER="/mnt/netapp1/Store_RES/home/res/cnag71/resh000982/projects/bioscience/results"

# List of folder names to process — one per array task
# Add or remove entries; update --array above to match (0 to N-1)
FOLDERS=(
    "L10_S456_260612162002"
    "L10_S789_260612163231"
    "L20_S456_260612164555"
    "L20_S789_260612170526"

)

# ============================================================
# RUNTIME — no edits needed below this line
# ============================================================


# Pick this task's folder using the SLURM array index
FOLDER="${FOLDERS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "$FOLDER" ]]; then
    echo "ERROR: No folder defined for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "============================================"
echo "Job array ID : ${SLURM_ARRAY_JOB_ID}"
echo "Task ID      : ${SLURM_ARRAY_TASK_ID}"
echo "Folder       : ${FOLDER}"
echo "Start time   : $(date)"
echo "Node         : $(hostname)"
echo "============================================"

# Output HTML goes into the same folder as the model results
OUTPUT_DIR="${RESULTS_FOLDER}/${FOLDER}"
OUTPUT_HTML="${OUTPUT_DIR}/report_${FOLDER}.html"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Run the marimo export via pixi
pixi run -C "${PROJECT_DIR}" \
    marimo export html "${NOTEBOOK}" \
    -o "${OUTPUT_HTML}" \
    -- --folder "${FOLDER}"

EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : ${EXIT_CODE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "HTML saved to: ${OUTPUT_HTML}"
else
    echo "ERROR: marimo export failed for folder: ${FOLDER}"
fi
echo "============================================"

exit ${EXIT_CODE}
