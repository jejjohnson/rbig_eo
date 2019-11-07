#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=clima-amip-global
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src//experiments/logs/climate/rbig-it-job-%j.log

module load Anaconda3
source activate 2019_rbig_ad

# Global Experiment

# Individual
srun --nodes 1 --ntasks 1 python -u src/experiments/climate/amip_global.py --save trial_v1 --exp individual 