#!/bin/bash
#SBATCH --job-name=mb_verify
#SBATCH --partition=cpu_1
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
cd $HOME/code/multibench && source mb/env.sh
python -m mb.verify --benches boxinggym,babyai,alfworld,sciworld --workers 6 --out $HOME/results/mb/verify_heavy.json
