#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=h_world_s5
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/h_world_s5_%j.log

module load Anaconda3
source activate rbig_eo

for VARIABLE in gpp rm lst precip lai
do 
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/entropy_earth.py --region world --period 2002_2010 --variable $VARIABLE --save v2 --subsample 200000
    wait
done



