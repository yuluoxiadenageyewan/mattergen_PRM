#!/bin/bash

[ ! -d "exp_res" ] && mkdir -p exp_res

EXPNAME="dpo_solid_electrolyte"

nohup python -u main.py \
    expname=${EXPNAME} \
    pipeline=dpo_solid_electrolyte \
    model=mattergen \
    reward=ionic_conductivity \
    logger=csv \
    device=cuda:0 \
    > exp_res/${EXPNAME}.log 2>&1 &

echo "Started DPO training, PID=$!, log: exp_res/${EXPNAME}.log"
