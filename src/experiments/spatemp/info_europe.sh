#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=euro_info
#SBATCH --mail-user=jemanjohnson34@gmail.com
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/euro_info_%j.log

module load Anaconda3
source activate rbig_eo

for VARIABLE in gpp rm lst precip lai
do 
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_sub.py --region europe --period 2002_2010 --spatial 1 --temporal 1 --variable $VARIABLE --subsample 500000 --save v1 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_sub.py --region europe --period 2002_2010 --spatial 7 --temporal 1 --variable $VARIABLE --subsample 500000 --save v1 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_sub.py --region europe --period 2002_2010 --spatial 3 --temporal 6 --variable $VARIABLE --subsample 500000 --save v1 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_sub.py --region europe --period 2002_2010 --spatial 1 --temporal 46 --variable $VARIABLE --subsample 500000 --save v1
    wait
done