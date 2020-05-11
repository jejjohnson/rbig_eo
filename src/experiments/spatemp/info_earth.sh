#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=spain_info
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/spain_info_%j.log

module load Anaconda3
source activate rbig_eo

for VARIABLE in gpp rm sm lst precip wv
do 
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth.py --region spain --period 2010 --spatial 1 --temporal 1 --variable $VARIABLE --subsample 200000 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth.py --region spain --period 2010 --spatial 7 --temporal 1 --variable $VARIABLE --subsample 200000 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth.py --region spain --period 2010 --spatial 3 --temporal 6 --variable $VARIABLE --subsample 200000 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth.py --region spain --period 2010 --spatial 1 --temporal 46 --variable $VARIABLE --subsample 200000
    wait
done

for VARIABLE in gpp rm sm lst precip wv
do 
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth.py --region spain --period 2002_2010 --spatial 1 --temporal 1 --variable $VARIABLE --subsample 200000 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth.py --region spain --period 2002_2010 --spatial 7 --temporal 1 --variable $VARIABLE --subsample 200000 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth.py --region spain --period 2002_2010 --spatial 3 --temporal 6 --variable $VARIABLE --subsample 200000 &&
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/info_earth.py --region spain --period 2002_2010 --spatial 1 --temporal 46 --variable $VARIABLE --subsample 200000
    wait
done