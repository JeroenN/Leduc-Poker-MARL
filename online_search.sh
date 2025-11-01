#!/bin/bash
#SBATCH --job-name=cfr_online 
#SBATCH --partition=regular
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=8G 
#SBATCH --output=logs/output.%j.log  # Output log
#SBATCH --error=logs/error.%j.log   # Error log
module purge
module load meson-python/0.18.0-GCCcore-14.2.0
source ".venv/bin/activate"
srun python online_cfr.py