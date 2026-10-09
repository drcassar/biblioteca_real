#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=15:00:00
#SBATCH --job-name=bias_var
#SBATCH --output=outputs/res_bias_var-%j.out
#SBATCH --error=errors/python_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

export N_CPUS=$SLURM_CPUS_PER_TASK
echo "Inicio: $(date)"
echo "No: $(hostname)"
echo "CPUs disponiveis: $N_CPUS"

python -u bias_var.py > bias_var.out

echo "Fim: $(date)"
