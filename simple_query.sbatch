#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -A hackathon
#SBATCH --reservation=hackathon
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -t 01:00
#SBATCH -o slurm%j.out
#SBATCH -e slurm%j.err
hostname
env
#nvprof -o trace_small.nvprof
#mpirun -np 2 ./simple_query
srun -n 2 ./simple_query
