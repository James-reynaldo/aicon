#!/bin/bash
#SBATCH --job-name=AICON_Single_Param_Sweep
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs_d/%A_%a.out
#SBATCH --array=0-475%200

source $(conda info --base)/etc/profile.d/conda.sh
conda activate domip2

export PYTHONPATH=/scratch/aldo/aicon/src/aicon/drawer_tutorial:/scratch/aldo/aicon/robosuite/robosuite-task-zoo:$PYTHONPATH

# Accept an optional job id as the first argument; prefer SLURM array task id, then SLURM_JOB_ID
JOB_ID=${1:-${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ID}}}
JOB_ID=$((JOB_ID))
export JOB_ID

echo "Running job with ID: $JOB_ID"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Ensure logs directory exists and run Python, saving a per-job .out file
mkdir -p logs_d
python -u Sweep_hpc.py "$JOB_ID" 1

