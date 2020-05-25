#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2020_rbig_rs/
#SBATCH --job-name=world_info
#SBATCH --output=/home/emmanuel/projects/2020_rbig_rs/src/experiments/spatemp/logs/world_info_%A_%j_%a.log
#SBATCH --array=0-15

module load Anaconda3
source activate rbig_eo



srun -n 1 python -u src/experiments/spatemp/info_earth_slurm.py --period 2002_2010 --job $SLURM_ARRAY_TASK_ID --subsample 500000 --save v2 --region world -rs 1MS --method new
