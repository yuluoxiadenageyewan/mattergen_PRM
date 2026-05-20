#!/bin/bash
set -e

# Check ionic_surro is accessible
IONIC_SURRO_PATH=${IONIC_SURRO_PATH:-"../ionic_surro"}
if [ ! -f "${IONIC_SURRO_PATH}/scorer.py" ]; then
    echo "ERROR: ionic_surro not found at ${IONIC_SURRO_PATH}"
    echo "  Clone it there, or set: export IONIC_SURRO_PATH=/path/to/ionic_surro"
    exit 1
fi

[ ! -d "exp_res" ] && mkdir -p exp_res

EXPNAME="dpo_solid_electrolyte"

PROJ_DIR=$(pwd)

nohup python -u main.py \
    expname=${EXPNAME} \
    pipeline=dpo_solid_electrolyte \
    model=mattergen \
    reward=ionic_conductivity \
    logger=csv \
    device=cuda:0 \
    reference_path=${PROJ_DIR}/mattergen/data-release/alex-mp/reference_MP2020correction.gz \
    potential_load_path=${PROJ_DIR}/MatterSim-v1.0.0-5M.pth \
    model.model_path=${PROJ_DIR}/mattergen/checkpoints/mattergen_base \
    > exp_res/${EXPNAME}.log 2>&1 &

echo "Started DPO training, PID=$!, log: exp_res/${EXPNAME}.log"
echo "Monitor: tail -f exp_res/${EXPNAME}.log"
