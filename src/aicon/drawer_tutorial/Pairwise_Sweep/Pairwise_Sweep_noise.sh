#!/bin/bash
#SBATCH --job-name=AICON_Single_Param_Sweep
#SBATCH --cpus-per-task=2
#SBATCH --partition=c0,c1a,c1b
#SBATCH --mem-per-cpu=6G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A_%a.out

source $(conda info --base)/etc/profile.d/conda.sh
conda activate domip2

export PYTHONPATH=/scratch/aldo/aicon/src/aicon/drawer_tutorial:/scratch/aldo/aicon/robosuite/robosuite-task-zoo:$PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    JOB_COUNT=$(python -u "$SCRIPT_DIR/Pairwise_Sweep.py" --count-jobs 2>/dev/null || true)
    if [[ -n "$JOB_COUNT" && "$JOB_COUNT" =~ ^[0-9]+$ && "$JOB_COUNT" -gt 0 ]]; then
        MAX_INDEX=$((JOB_COUNT - 1))
        echo "Submitting pairwise sweep array with ${JOB_COUNT} jobs..."
        sbatch --array="0-${MAX_INDEX}%200" "$SCRIPT_DIR/$(basename "$0")"
        exit 0
    fi
fi

# Accept an optional job id as the first argument; prefer SLURM array task id, then SLURM_JOB_ID
JOB_ID=${1:-${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ID}}}
JOB_ID=$((JOB_ID))
export JOB_ID

echo "Running job with ID: $JOB_ID"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Ensure logs directory exists and run Python, saving a per-job .out file
mkdir -p logs
python -u Pairwise_Sweep.py "$JOB_ID" 0 20

