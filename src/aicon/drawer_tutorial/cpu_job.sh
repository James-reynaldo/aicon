#!/bin/bash
#SBATCH --job-name=AICON_Single_Param_Sweep
#SBATCH --partition=c2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=Sweep.out

source $(conda info --base)/etc/profile.d/conda.sh
conda activate domip2

export PYTHONPATH=/scratch/aldo/drawer_tutorial:/scratch/aldo/drawer_tutorial/aicon/robosuite/robosuite-task-zoo:$PYTHONPATH

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0

python -u Sweep.py

