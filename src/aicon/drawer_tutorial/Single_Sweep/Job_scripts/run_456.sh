#!/bin/bash
#SBATCH --job-name=AICON_Job456
#SBATCH --cpus-per-task=2
#SBATCH --partition=c1b,c2
#SBATCH --mem-per-cpu=6G
#SBATCH --time=01:00:00
#SBATCH --output=logs_t/job456_%A_%a.out
#SBATCH --array=0-3

source $(conda info --base)/etc/profile.d/conda.sh
conda activate domip2

export PYTHONPATH=/scratch/aldo/aicon/src:/scratch/aldo/aicon/robosuite/robosuite-task-zoo:$PYTHONPATH

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs

JOB_ID=456

echo "Running job index: $JOB_ID"
echo "Scenario: $SLURM_ARRAY_TASK_ID"

case $SLURM_ARRAY_TASK_ID in

    0)
        echo "===== NORMAL ====="
        python -u Sweep_hpc.py "$JOB_ID"
        ;;

    1)
        echo "===== DISTURBANCE ====="
        python -u Sweep_hpc.py "$JOB_ID" --disturbance 0.5
        ;;

    2)
        echo "===== NOISE ====="
        python -u Sweep_hpc.py "$JOB_ID" --noise-scale 20
        ;;

    3)
        echo "===== BAD PRIOR ====="
        python -u Sweep_hpc.py "$JOB_ID" --prior-noise-std 0.2
        ;;

    *)
        echo "Invalid scenario: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "===== SCENARIO COMPLETED ====="