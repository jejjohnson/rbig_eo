#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=world_info556_rs
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/world_info556_rs_%j.log

module load Anaconda3
source activate rbig_eo

for VARIABLE in gpp rm lst precip lai
do
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth_rs.py --region north --period 2002_2010 --spatial 5 --temporal 6 --variable $VARIABLE --subsample 200000 --save v1 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth_rs.py --region south --period 2002_2010 --spatial 5 --temporal 6 --variable $VARIABLE --subsample 200000 --save v1
    wait
done
