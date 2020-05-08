#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=rbig_euro
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/rbig_euro_%j.log

module load Anaconda3
source activate rbig_eo


srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/europe.py --dataset gpp --verbose 1 --save v2 &
srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/europe.py --dataset rm --verbose 1 --save v2 &
srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/europe.py --dataset lst --verbose 1 --save v2 