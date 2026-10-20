# Train Llama 3.3 70B on HelpSteer3

Train the full Llama backbone and a scalar reward head on English HelpSteer3
(General, STEM, Code), using scaled Bradley–Terry loss from the
[paper](https://arxiv.org/html/2505.11475v1#A8).

Settings are in `config.toml`: one epoch, 4096-token limit, batch of 64 pairs,
learning rate 2e-6. The FSDP launcher requests **16 GH200 GPUs** and requires RDMA.

## Run on TIGRIS

Get Hugging Face access to [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).
Replace `YOUR_RIT_USERNAME` and `YOUR_ACCOUNT` with your RIT username and Slurm project account.

```bash
ssh YOUR_RIT_USERNAME@tigris.rc.rit.edu
git clone https://github.com/hil-se/hs3-tigris.git
cd hs3-tigris
source /.autofs/tools/spack/share/spack/setup-env.sh
spack env activate default-ml-aarch64-25050701
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
source cluster/tigris/env.sh
huggingface-cli login

prep=$(sbatch -M tigris --parsable --account=YOUR_ACCOUNT cluster/tigris/prepare.slurm)
sbatch -M tigris --account=YOUR_ACCOUNT --dependency=afterok:${prep%%;*} cluster/tigris/run.slurm
```

`prepare.py` downloads the model/data and tokenizes pairs on CPU. After it
succeeds, `train.py` trains offline and selects the checkpoint with the lowest validation loss.

## Outputs

- Prepared data: `data/llama33-hs3/`
- Model-only checkpoints: `/scratch/$USER/hs3-tigris/checkpoints/job-JOB_ID/`
- Metrics and selected checkpoint: `results/job-JOB_ID/`
- Logs: `slurm-hs3-llama70b-bt-JOB_ID.out` and `.err`
