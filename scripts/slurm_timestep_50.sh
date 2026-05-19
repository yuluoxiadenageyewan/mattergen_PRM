#!/bin/bash
#SBATCH --job-name=timestep_50
#SBATCH --partition=v100g32fat
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --time=84:00:00
#SBATCH --output=ablation/timestep_50/slurm-%j.out

module load gcc/9.3.0
cd /data/home/heheda/matinvent
source .venv/bin/activate

EXPNAME="timestep_50"
mkdir -p ablation/${EXPNAME}

python -u main.py \
    expname=${EXPNAME} \
    pipeline=timestep_50 \
    model=mattergen \
    reward=formation_energy \
    logger=csv \
    > ablation/${EXPNAME}/${EXPNAME}.log 2>&1
