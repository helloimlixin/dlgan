#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --requeue
#SBATCH --job-name=xl598-vqgan
#SBATCH --nodes=1
#SBATCH --mem=16000
#SBATCH --time=05:00:00
#SBATCH --output=test.out
#SBATCH --error=test.err

srun nvidia-smi
srun python dlgan_training.py

