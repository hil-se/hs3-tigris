#!/bin/bash
#SBATCH --job-name=hs3-prototype
#SBATCH --partition=tigris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/tigris/env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/scratch/${USER}/hs3-tigris}"
FEATURES="${ARTIFACT_ROOT}/features-${SLURM_JOB_ID}"
python embed.py --data "${DATA_DIR:-data}" --output "${FEATURES}" --device cuda
python train.py --features "${FEATURES}" --output "results/job-${SLURM_JOB_ID}"
