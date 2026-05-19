#!/bin/bash

[ ! -d "exp_res" ] && mkdir -p exp_res

for T in 50 200 500; do
    EXPNAME="timestep_${T}"
    nohup python -u main.py \
        expname=${EXPNAME} \
        pipeline=timestep_${T} \
        model=mattergen \
        reward=formation_energy \
        logger=csv \
        device=cuda:0 \
        > exp_res/${EXPNAME}.log 2>&1 &
    echo "Launched ${EXPNAME} (PID $!)"
done
