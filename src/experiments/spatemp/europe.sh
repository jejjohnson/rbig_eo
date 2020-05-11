#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=rbig_euro
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/rbig_euro_%j.log

module load Anaconda3
source activate rbig_eo

for VARIABLE in gpp rm lst
do 
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth.py --dataset gpp --verbose 1 --save v2
    wait
done