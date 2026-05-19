#!/bin/bash
#SBATCH --job-name=official
#SBATCH --partition=v100g32fat
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --time=84:00:00
#SBATCH --output=ablation/official/slurm-%j.out

module load gcc/9.3.0
cd /data/home/heheda/matinvent
source .venv/bin/activate

EXPNAME="official"
mkdir -p ablation/${EXPNAME}

python -u main.py \
    expname=${EXPNAME} \
    pipeline=ablation_official \
    model=mattergen \
    reward=formation_energy \
    logger=csv \
    > ablation/${EXPNAME}/${EXPNAME}.log 2>&1
