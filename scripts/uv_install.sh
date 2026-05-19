#!/bin/bash

pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match

git clone https://github.com/microsoft/mattergen.git
cd mattergen
git checkout 5bb2b397a36de85a8dc9583b7d1d6353989de72c
uv pip install -e .
cd ..

# ionic_surro dependencies
IONIC_SURRO_PATH=${IONIC_SURRO_PATH:-"../ionic_surro"}
if [ -f "${IONIC_SURRO_PATH}/requirements.txt" ]; then
    uv pip install -r "${IONIC_SURRO_PATH}/requirements.txt" --index-strategy unsafe-best-match
    echo "ionic_surro dependencies installed from ${IONIC_SURRO_PATH}"
else
    echo "WARNING: ionic_surro not found at ${IONIC_SURRO_PATH}, skipping. Set IONIC_SURRO_PATH to fix."
fi
