#!/bin/bash
#SBATCH --job-name=mamba-k8
#SBATCH --gres=gpu:8
#SBATCH --mem=1024G
#SBATCH --output=logs/mamba-k8.log

# Load any necessary modules and activate your environment
# module load python/3.7.3
source activate zecheng

# Run your program
bash scripts/370M_mamba_k8.sh
