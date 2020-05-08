#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=rbig_spain
#SBATCH --output=src/experiments/spatemp/logs/rbig_euro_%j.log

module load Anaconda3
source activate rbig_eo


srun --ntasks 1 python -u src/experiments/spatemp/spain.py --dataset gpp --verbose 1 --save v4 --subsample 10_000 &&
srun --ntasks 1 python -u src/experiments/spatemp/spain.py --dataset rm --verbose 1 --save v4 --subsample 10_000 &&
srun --ntasks 1 python -u src/experiments/spatemp/spain.py --dataset lst --verbose 1 --save v4 --subsample 10_000 &&
srun --ntasks 1 python -u src/experiments/spatemp/spain.py --dataset precip --verbose 1 --save v4 --subsample 10_000 &&
srun --ntasks 1 python -u src/experiments/spatemp/spain.py --dataset sm --verbose 1 --save v4 --subsample 10_000 &&
srun --ntasks 1 python -u src/experiments/spatemp/spain.py --dataset wv --verbose 1 --save v4 --subsample 10_000 