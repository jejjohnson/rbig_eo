#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=euro_h
#SBATCH --mail-user=jemanjohnson34@gmail.com
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/euro_h_%j.log

module load Anaconda3
source activate rbig_eo

for VARIABLE in gpp rm sm lst precip wv
do 
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/entropy_earth.py --region europe --period 2010 --variable $VARIABLE --subsample 200000
    wait
done

for VARIABLE in gpp rm sm lst precip wv
do 
    srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/entropy_earth.py --region europe --period 2002_2010 --variable $VARIABLE --subsample 200000
    wait
done
