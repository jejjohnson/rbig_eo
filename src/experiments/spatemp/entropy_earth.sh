#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=spain_h_eo
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/spain_h_eo_%j.log

module load Anaconda3
source activate rbig_eo

for VARIABLE in gpp rm sm lst precip wv
do 
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/entropy_earth.py --region spain --period 2010 --variable $VARIABLE &
    wait
done

for VARIABLE in gpp rm sm lst precip wv
do 
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/entropy_earth.py --region spain --period 2002_2010 --variable $VARIABLE
    wait
done
