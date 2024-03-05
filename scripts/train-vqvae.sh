#! /bin/bash
#SBATCH --partition=main
#SBATCH --requeue
#SBATCH --job-name=vqvae-train-test
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=124000
#SBATCH --time=02:00:00
#SBATCH --output=slurm.%N.$j.out
#SBATCH --error=slurm.%N.%j.err
cd /scratch/$USER
srun python print('hello')
