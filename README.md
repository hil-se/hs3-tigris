# Train Llama 3.3 70B on HelpSteer3

Full-model scalar reward training following the **English reward model** in the
[HelpSteer3-Preference paper](https://arxiv.org/html/2505.11475v1#A8): General,
STEM and Code. Read `prepare.py`, then `train.py`. Settings are in `config.toml`.

The paper uses **scaled Bradley–Terry**:

```text
loss = mean(abs(preference) * softplus(-(reward_chosen - reward_rejected)))
```

The backbone and scalar reward head both train. The head reads the last
non-padding token. Inputs use the released NVIDIA reward model's chat template.

| Setting | Default |
| --- | --- |
| Base model | meta-llama/Llama-3.3-70B-Instruct |
| Sequence limit | 4096 tokens |
| Training duration | One epoch, max_steps=-1 |
| Global batch | 128 response records = 64 preference pairs |
| AdamW | LR 2e-6, betas (0.9, 0.98), weight decay 0.1 |
| Schedule | 10 warm-up steps, then constant LR |
| Validate and save | Every 50 steps, plus the final step |
| Selection | Lowest validation scaled-BT loss; earlier checkpoint on ties |

The learning rate is the paper's selected English-RM setting (Table 8).
The implementation uses PyTorch FSDP; the paper used NeMo-Aligner. The official
validation split stays validation, and overlapping training prompts are removed
to prevent leakage. Raw repeated rows otherwise remain as released. Preference
ties and overlength pairs are excluded; full chat context is preserved.
These changes mean this is not a claim of identical paper scores. The paper's
reported checkpoint was step 200; this run selects using its own validation loss.

## Run on TIGRIS

Your Hugging Face account needs access to the
[Meta model](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).
Use your own RIT login and Slurm project account.

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

The CPU preparation job downloads pinned model/data revisions and tokenizes pairs
before GPUs are allocated. Training runs offline. Both scripts take no command-line
arguments: edit `config.toml` for model/data and training settings. Preparation
writes to `data/llama33-hs3/`; running it again rewrites those prepared files.
Each training job uses its Slurm job ID for its output directories.

The launcher requests **16 GH200 GPUs**, one per node. FP32 master parameters
and optimizer states are sharded; GPU computation uses BF16 and activation
checkpointing. The 384 GiB host-memory request accommodates rank zero loading
the full FP32 model before sharding. Inter-node NCCL communication requires the
IB/RoCE RDMA transport (`NCCL_NET=IB`), with INFO logs for checking network
initialization. `rctrust.571` remains the socket interface for NCCL bootstrap.

Checkpoints go to `/scratch/$USER/hs3-tigris/checkpoints/job-JOB_ID/`. Each saved
FP32 model is roughly 280 GB, so reserve several TB for a full epoch's snapshots.
These are distributed model-only checkpoints, without optimizer/RNG resume state.
No checkpoints are pruned. Small records are in `results/job-JOB_ID/`:

- `history.json`: validation loss/accuracy at each checkpoint.
- `selected.json`: selected checkpoint path and step.
- `run.json`: completion, actual final step, seed, pinned revisions and source snapshot.

```bash
squeue -M tigris -u "$USER"
sacct -M tigris -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -f slurm-hs3-llama70b-bt-JOB_ID.out
```
