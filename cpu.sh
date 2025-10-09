#!/bin/bash
#SBATCH --job-name=cpu_job 
#SBATCH --partition=regular
#SBATCH --cpus-per-task=1 
#SBATCH --time=8:00:00 
#SBATCH --mem=16G 
#SBATCH --output=logs/output.%j.log  # Output log
#SBATCH --error=logs/error.%j.log   # Error log
module purge
module load CUDA/12.4.0
source "/home3/s3799042/lc0_venv2/bin/activate"
srun python3 main.py