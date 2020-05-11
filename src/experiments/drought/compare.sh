#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=drought
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/drought/logs/drought_compare_%j.log

module load Anaconda3
source activate rbig_eo

srun --nodes 1 --ntasks 1 python -u src/experiments/drought/compare.py --compare 2 --temporal 12 --spatial 1 --save v1