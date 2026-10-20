#!/bin/bash
#SBATCH --job-name=hs3-prepare
#SBATCH --partition=tigris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/tigris/env.sh
HF_HUB_OFFLINE=0 python prepare.py --config "${CONFIG_PATH:-config.toml}" --output "${DATA_DIR:-data/llama33-hs3}"
