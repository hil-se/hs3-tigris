#!/bin/bash
# Source from the repository root. Each student uses their own writable cache.
source /.autofs/tools/spack/share/spack/setup-env.sh
spack env activate default-ml-aarch64-25050701
source .venv/bin/activate
unset PYTHONPATH
export HF_HOME="${HF_HOME:-/scratch/${USER}/hs3-tigris/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
