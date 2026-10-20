#!/bin/bash
#SBATCH --job-name=hs3-llama70b-bt
#SBATCH --partition=tigris
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=384G
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source cluster/tigris/env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${HOSTS[0]}"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
# Socket transport avoids the cluster's observed RDMA initialization failures.
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=rctrust.571
srun --kill-on-bad-exit=1 bash -c '
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
  exec python -u train.py
'
