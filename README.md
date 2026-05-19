# matprm: MatterGen-based Solid Electrolyte RL/DPO

Reinforcement learning and DPO fine-tuning of MatterGen diffusion models for solid electrolyte material discovery.

## Environment Setup

**Requirements:** Linux, NVIDIA GPU (CUDA >= 11.8), Python 3.10

### Method 1: uv (recommended, ~2 min)

```bash
bash scripts/uv_install.sh
source .venv/bin/activate
```

Creates a `.venv` (Python 3.10) and installs from `requirements.txt` + MatterGen from source.

> `ionic_surro` path defaults to `../ionic_surro`. Set `IONIC_SURRO_PATH` to override.

### Method 2: conda/mamba (>10 min)

```bash
conda env create -f env.yml
conda activate matinvent
```

### FairChem (optional)

For specific heat capacity rewards using [FairChem](https://github.com/facebookresearch/fairchem) + eSEN-30M-OAM, set up a separate conda environment:

```bash
conda env create -f fairchem.env.yml
```

See [rewards/calculators/fairchem/README.md](rewards/calculators/fairchem/README.md) for details.

## Running RL

```bash
python -u main.py expname=test pipeline=mat_invent model=mattergen reward=hhi logger=wandb
# or
bash scripts/run_rl.sh
```

Key parameters (managed via [hydra](https://github.com/facebookresearch/hydra)):
- `expname`: experiment name
- `model`: diffusion model config ([configs/model](configs/model))
- `reward`: target property ([configs/reward](configs/reward))
- `logger`: `wandb` or `csv`

Model checkpoints are saved every `save_freq` steps (default 50) and on manual early stop (Ctrl+C).

## Generation & Evaluation

```bash
bash scripts/gen_eval.sh
```

Edit environment variables in the script before running.

## Citation

```bibtex
@article{matinvent,
  title={Accelerating inverse materials design using generative diffusion models with reinforcement learning},
  author={Chen, Junwu and Guo, Jeff and Fako, Edvin and Schwaller, Philippe},
  journal={arXiv preprint arXiv:2511.03112},
  year={2025}
}
```
