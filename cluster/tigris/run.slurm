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
# Require RDMA for inter-node NCCL traffic and log network initialization.
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_DEBUG=INFO
# NCCL still uses sockets for bootstrap, before RDMA data transfers.
export NCCL_SOCKET_IFNAME=rctrust.571
srun --kill-on-bad-exit=1 bash -c '
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
  exec python -u train.py
'
