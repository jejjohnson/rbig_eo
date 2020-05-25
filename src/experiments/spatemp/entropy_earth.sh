#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=h_world
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/h_world_%j.log
#SBATCH --array=0-5

module load Anaconda3
source activate rbig_eo

srun --nodes 1 --ntasks 1 python -u src/experiments/spatemp/entropy_earth_slurm.py --region world --period 2002_2010 --job $SLURM_ARRAY_TASK_ID --save v1 --subsample 200000 -rs 1MS --method old -rc
